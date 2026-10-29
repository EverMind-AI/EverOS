"""The clustering strategies are safe to re-execute, as OME's runner requires.

OME delivers at least once: a strategy run may be replayed after a crash
between its upsert and its emit, or retried after the emit failed. The
runner's docstring makes that the strategy's problem: "strategy bodies must
be safe to re-execute with the same payload". For the two clustering
strategies that means a replayed episode (or case) must not be folded into
its cluster a second time -- ``count`` is what ``extract_user_profile``
gates on, and the centroid would be re-weighted.

For every generated sequence of events, with replays drawn in, against the
real repository on a temp SQLite file and the real geometry merge (the
LLM merge of the skill strategy is stood in by a same-shape fake):

1. every cluster's ``count`` equals its number of member rows;
2. every distinct entry is a member of exactly one cluster;
3. running an event again changes nothing in the cluster tables.

Before the fix a replay raised ``count`` by one per replay while the member
rows stayed, so a user's profile extraction fired early.
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from everalgo.clustering import Cluster as AlgoCluster
from hypothesis import given, settings
from hypothesis import strategies as st
from sqlalchemy import select
from sqlmodel import SQLModel

from everos.config import SqliteSettings
from everos.infra.persistence import MemoryRoot
from everos.infra.persistence.sqlite import (
    SqliteClusterStore,
    create_session_factory,
    create_system_engine,
)
from everos.infra.persistence.sqlite.repos.cluster import ClusterRepo
from everos.infra.persistence.sqlite.tables import Cluster, ClusterMember
from everos.memory._partition_locks import _reset_for_tests
from everos.memory.events import AgentCaseExtracted, EpisodeExtracted
from everos.memory.strategies import (
    trigger_profile_clustering,
    trigger_skill_clustering,
)
from tests.fakes import make_deps, make_stores

# Two orthogonal directions: entries of one group merge, the groups do not.
_VECTORS = {0: [1.0, 0.0, 0.0, 0.0], 1: [0.0, 1.0, 0.0, 0.0]}
_plans = st.lists(
    st.tuples(st.integers(0, 5), st.integers(0, 1)), min_size=1, max_size=10
)


async def _fake_acluster_by_llm(
    new_cluster: AlgoCluster, existing: list[AlgoCluster], *, llm: Any
) -> AlgoCluster | None:
    """Same shape as the LLM merge: fold into the first cluster of the same group."""
    from everalgo.clustering import cluster_by_geometry

    return cluster_by_geometry(new_cluster, existing, threshold=0.9)


class _GroupEmbedder:
    """Embeds each entry's text into its group's direction."""

    dim = 4

    def __init__(self) -> None:
        self.vectors_of: dict[str, list[float]] = {}

    async def embed(self, text: str) -> list[float]:
        return self.vectors_of[text]

    async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.vectors_of[t] for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self.vectors_of[text]


async def _run(plan: list[tuple[int, int]], which: str) -> None:
    tmp = Path(tempfile.mkdtemp(prefix="cluster_replay_"))
    try:
        root = MemoryRoot(tmp)
        root.ensure()
        engine = create_system_engine(root.system_db, SqliteSettings())
        factory = create_session_factory(engine)
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)
        repo = ClusterRepo(session_factory=factory)
        _reset_for_tests()
        embedder = _GroupEmbedder()
        deps = make_deps(
            embedding=embedder,
            stores=make_stores(cluster=SqliteClusterStore(repo)),
        )

        async def snapshot() -> dict[
            str, tuple[int, tuple[str, ...], tuple[float, ...]]
        ]:
            async with factory() as s:
                cs = list((await s.execute(select(Cluster))).scalars().all())
                ms = list((await s.execute(select(ClusterMember))).scalars().all())
            return {
                c.cluster_id: (
                    c.count,
                    tuple(
                        sorted(m.member_id for m in ms if m.cluster_id == c.cluster_id)
                    ),
                    tuple(np.frombuffer(c.centroid_blob, dtype=np.float32).tolist()),
                )
                for c in cs
            }

        llm_merge = patch(
            "everos.memory.strategies.trigger_skill_clustering.acluster_by_llm",
            _fake_acluster_by_llm,
        )
        with llm_merge:
            seen: set[str] = set()
            for n, group in plan:
                entry = f"e{n}"
                text = f"entry {n}"
                embedder.vectors_of[text] = _VECTORS[group]
                if which == "profile":
                    ev: Any = EpisodeExtracted(
                        memcell_id=f"mc_{n}",
                        episode_entry_id=entry,
                        episode_text=text,
                        episode_timestamp_ms=1_700_000_000_000 + n,
                        owner_id="u",
                        session_id="s",
                    )
                    run = trigger_profile_clustering
                else:
                    ev = AgentCaseExtracted(
                        memcell_id=f"mc_{n}",
                        case_entry_id=entry,
                        agent_id="a",
                        task_intent=text,
                        approach="x",
                        key_insight="y",
                        quality_score=0.5,
                        case_timestamp_ms=1_700_000_000_000 + n,
                    )
                    run = trigger_skill_clustering
                before = await snapshot() if entry in seen else None
                await run(ev, deps)
                after = await snapshot()
                if before is not None:  # a replay
                    assert after == before, (entry, before, after)
                seen.add(entry)
                for cid, (count, members, _) in after.items():
                    assert count == len(members), (cid, count, members, plan)
                placed = [m for _, members, _ in after.values() for m in members]
                assert sorted(placed) == sorted(seen), (placed, seen)
        await engine.dispose()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@pytest.mark.parametrize("which", ["profile", "skill"])
@given(plan=_plans)
@settings(max_examples=40, deadline=None)
def test_clustering_is_replay_safe(which: str, plan: list[tuple[int, int]]) -> None:
    asyncio.run(_run(plan, which))

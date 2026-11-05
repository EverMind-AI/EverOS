"""Reflection restamps a cluster from the members it has now (#563).

Reflection reads a cluster's members, calls the LLM, writes the merged
episode, and only then folds the sources into it. Profile clustering may add
an episode to the same cluster in between. Before #563 the finisher wrote
``count=1`` and the merged episode's vector as the centroid regardless, so
the new member stayed in ``members`` but neither ``count`` nor ``centroid``
nor ``last_ts`` described it, and ``extract_user_profile``'s gate (a sum of
``count``) undercounted.

Real SQLite cluster store, real ``trigger_profile_clustering`` and real
``ReflectionFinisher``; the interleaving is forced instead of raced.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import pytest

from everos.config import SqliteSettings
from everos.infra.persistence import MemoryRoot
from everos.infra.persistence.sqlite import (
    SqliteClusterStore,
    SqliteDatabase,
    SqliteRepos,
)
from everos.memory.events import EpisodeExtracted, ReflectionContext
from everos.memory.models import Cluster, Episode
from everos.memory.ports import Scope
from everos.memory.reflection.orchestrator import ReflectionFinisher
from everos.memory.strategies import trigger_profile_clustering
from tests.fakes import make_deps, make_indexes, make_stores, seed_sources

_DIM = 8
_T0 = 1_700_000_000_000
_OWNER = "u_stale"
_CLUSTER = "cl_stale"
_SCOPE = Scope(
    app_id="default", project_id="default", owner_id=_OWNER, owner_type="user"
)
_MERGED_TEXT = "merged episode"
_JOINER_TEXT = "an episode that joined while reflection ran"
_VALUES = {_MERGED_TEXT: 0.2, _JOINER_TEXT: 0.1}


class _Embedder:
    """Every text embeds to a constant vector, so centroids are readable."""

    dim = _DIM

    async def embed(self, text: str) -> list[float]:
        return [_VALUES[text]] * _DIM

    async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]


@pytest.fixture
async def cluster_store(tmp_path: Path) -> AsyncIterator[SqliteClusterStore]:
    database = SqliteDatabase(MemoryRoot(tmp_path / "root"), SqliteSettings())
    await database.create_schema()
    try:
        yield SqliteClusterStore(SqliteRepos.over(database).cluster)
    finally:
        await database.dispose()


async def _scenario(cluster_store: SqliteClusterStore, *, joiner: bool) -> Cluster:
    stores = make_stores(cluster=cluster_store)
    deps = make_deps(embedding=_Embedder(), stores=stores)
    await cluster_store.upsert(
        _SCOPE,
        "user_memory",
        Cluster(
            id=_CLUSTER,
            centroid=(0.1,) * _DIM,
            count=2,
            last_ts=_T0,
            preview=["p1", "p2"],
            members=["ep1", "ep2"],
        ),
        member_type="episode",
    )
    # Reflection starts: this is what it read before its LLM call.
    context = ReflectionContext(
        cluster_id=_CLUSTER,
        mode="init",
        source_members=[("ep1", "episode"), ("ep2", "episode")],
        episodes=[],
        merged_episode=_MERGED_TEXT,
        last_ts=_T0 + 1000,
    )
    if joiner:  # meanwhile profile clustering adds a new episode
        seed_sources(deps, memcell_id="mc_3", owner_id=_OWNER)
        stores.episode.seed(  # type: ignore[attr-defined]
            _SCOPE,
            "ep3",
            Episode(
                owner_id=_OWNER,
                episode=_JOINER_TEXT,
                timestamp=_T0 + 2000,
                subject="s",
                summary="s",
                parent_id="mc_3",
            ),
        )
        await trigger_profile_clustering(
            EpisodeExtracted(
                memcell_id="mc_3",
                episode_entry_id="ep3",
                episode_text=_JOINER_TEXT,
                episode_timestamp_ms=_T0 + 2000,
                owner_id=_OWNER,
                session_id="s1",
            ),
            deps,
        )
        joined = await cluster_store.get(_CLUSTER)
        assert joined is not None and "ep3" in joined.members, "precondition"
    stores.episode.seed(  # type: ignore[attr-defined]
        _SCOPE,
        "ep_merged",
        Episode(
            owner_id=_OWNER,
            episode=_MERGED_TEXT,
            timestamp=_T0 + 1000,
            subject="m",
            summary="m",
            parent_id=_CLUSTER,
        ),
    )
    indexes = make_indexes()
    finisher = ReflectionFinisher(
        cluster_store=cluster_store,
        episode_store=stores.episode,
        atomic_fact_store=stores.atomic_fact,
        atomic_fact_index=indexes.atomic_fact,
        report_store=stores.reflection_report,
        embedder=_Embedder(),
    )
    await finisher.finish(scope=_SCOPE, merged_entry_id="ep_merged", context=context)
    final = await cluster_store.get(_CLUSTER)
    assert final is not None
    return final


async def test_no_joiner_leaves_the_merged_episode_alone(
    cluster_store: SqliteClusterStore,
) -> None:
    final = await _scenario(cluster_store, joiner=False)

    assert final.members == ["ep_merged"]
    assert final.count == 1
    assert list(final.centroid) == pytest.approx([0.2] * _DIM)
    assert final.last_ts == _T0 + 1000


async def test_a_member_that_joined_mid_reflection_is_counted(
    cluster_store: SqliteClusterStore,
) -> None:
    final = await _scenario(cluster_store, joiner=True)

    assert sorted(final.members) == ["ep3", "ep_merged"]
    assert final.count == 2, "count describes only the merged episode"
    assert list(final.centroid) == pytest.approx([0.15] * _DIM), (
        "centroid ignores the member that joined"
    )
    assert final.last_ts == _T0 + 2000, "last_ts went back to before the joiner"

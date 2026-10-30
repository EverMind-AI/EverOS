"""Reflection end to end over markdown truth and the cascade-built index.

Drives both Reflection steps the way the two strategies chain them --
``ReflectionOrchestrator.run`` (merge, write the merged episode) then
``ReflectionFinisher.finish`` (deprecate what the merge replaced) -- over
real markdown stores, the real SQLite cluster / report tables and a real
LanceDB index that only the cascade handlers write, with the reflector
answered by a ``FakeLLMClient``.

Reflection marks deprecation in markdown only; the cascade carries the
marks into the index. So every assertion on the index follows a cascade
pass, and the last test drops the index and rebuilds it from markdown to
show the marks survive (the index is rebuildable, the md is the truth).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import AsyncIterator, Callable, Sequence
from pathlib import Path

import numpy as np
import pytest
from everalgo.clustering import Cluster as AlgoCluster
from everalgo.testing import FakeLLMClient
from everalgo.user_memory import EpisodeReflector

from everos.infra.cascade._projections import AtomicFactHandler, EpisodeHandler
from everos.infra.persistence import MemoryRoot
from everos.infra.persistence.index.adapters import (
    AtomicFactIndexAdapter,
    EpisodeIndexAdapter,
)
from everos.infra.persistence.markdown import (
    MarkdownAtomicFactStore,
    MarkdownEpisodeStore,
)
from everos.infra.persistence.sqlite import (
    SqliteClusterStore,
    SqliteReflectionReportStore,
)
from everos.memory._partition_locks import _reset_for_tests
from everos.memory.events import EpisodeExtracted
from everos.memory.models import AtomicFact, Episode, ReflectionReport
from everos.memory.ports import MemoryRef, Scope, all_of, eq, is_null
from everos.memory.reflection import ReflectionFinisher, ReflectionOrchestrator
from tests.fakes import user_track_script
from tests.fakes.storage import Storage, handler_deps

OWNER = "u_test"
CLUSTER = "cl_test_reflect"
SCOPE = Scope(app_id="default", project_id="default", owner_id=OWNER, owner_type="user")
_HOUR = 3_600_000
_T0 = 1_781_085_600_000  # 2026-06-10 10:00 UTC


class _StubEmbedder:
    """Deterministic unit vectors seeded by the text; no network."""

    dim: int = 1024

    async def embed(self, text: str) -> list[float]:
        seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")
        vec = np.random.default_rng(seed).standard_normal(self.dim).astype(np.float32)
        return (vec / (float(np.linalg.norm(vec)) or 1.0)).tolist()

    async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]


class _StubTokenizer:
    def tokenize(self, text: str) -> list[str]:
        return text.lower().split()

    def tokenize_batch(self, texts: Sequence[str]) -> list[list[str]]:
        return [self.tokenize(t) for t in texts]


def _reply(content: str, title: str) -> str:
    return json.dumps({"content": content, "summary": title, "title": title})


class _World:
    """Markdown stores, the index, the cascade handlers and both Reflection steps."""

    def __init__(
        self, storage: Storage, root: MemoryRoot, reflector_llm: FakeLLMClient
    ) -> None:
        self.storage = storage
        self.root = root
        self.episodes = MarkdownEpisodeStore(root)
        self.facts = MarkdownAtomicFactStore(root)
        self.index = storage.repos
        deps = handler_deps(
            memory_root=root,
            tokenizer=_StubTokenizer(),
            index=storage.repos,
            embedding=_StubEmbedder(),
        )
        self._handlers = {
            "episodes": EpisodeHandler(deps),
            ".atomic_facts": AtomicFactHandler(deps),
        }
        clusters = SqliteClusterStore(storage.sqlite.cluster)
        reports = SqliteReflectionReportStore(storage.sqlite.reflection_report)
        self.orchestrator = ReflectionOrchestrator(
            cluster_store=clusters,
            episode_store=self.episodes,
            episode_index=EpisodeIndexAdapter(storage.repos.episode),
            report_store=reports,
            reflector=EpisodeReflector(llm=reflector_llm),
        )
        self.finisher = ReflectionFinisher(
            cluster_store=clusters,
            episode_store=self.episodes,
            atomic_fact_store=self.facts,
            atomic_fact_index=AtomicFactIndexAdapter(storage.repos.atomic_fact),
            report_store=reports,
            embedder=_StubEmbedder(),
        )

    async def cascade(self) -> None:
        """Project every episode and atomic-fact md into the index."""
        for folder, handler in self._handlers.items():
            for md in sorted(self.root.root.rglob(f"{folder}/*.md")):
                await handler.handle_added_or_modified(
                    str(md.relative_to(self.root.root))
                )

    async def rebuild_index(self) -> None:
        """Drop every business table and rebuild it from markdown alone."""
        await self.storage.index.drop_business_tables()
        await self.storage.index.ensure_business_indexes()
        await self.cascade()

    async def seed(self, n: int, *, start: int = 0) -> list[MemoryRef]:
        """``n`` memcell episodes (one atomic fact each), indexed, in the cluster."""
        episodes = [
            Episode(
                owner_id=OWNER,
                episode=f"Andrew pet event {start + i}",
                summary=f"pet event {start + i}",
                timestamp=_T0 + (start + i) * _HOUR,
                session_id="s_test",
                parent_id=f"mc_{start + i:03d}",
            )
            for i in range(n)
        ]
        refs = await self.episodes.append_many(SCOPE, episodes)
        await self.facts.append_many(
            SCOPE,
            [
                AtomicFact(
                    owner_id=OWNER,
                    content=f"fact of {ep.episode}",
                    timestamp=ep.timestamp,
                    session_id="s_test",
                    parent_id=ref.entry_id,
                )
                for ep, ref in zip(episodes, refs, strict=True)
            ],
        )
        await self.cascade()
        return refs

    async def reflect(self) -> tuple[list[EpisodeExtracted], list[ReflectionReport]]:
        """Both steps, then a cascade pass so the index sees the marks."""
        merged = await self.orchestrator.run(scope=SCOPE)
        reports = []
        for event in merged:
            assert event.reflection is not None
            report = await self.finisher.finish(
                scope=SCOPE,
                merged_entry_id=event.episode_entry_id,
                context=event.reflection,
            )
            if report is not None:
                reports.append(report)
        await self.cascade()
        return merged, reports

    async def deprecated_by(self, repo_name: str, entry_id: str) -> str | None:
        repo = getattr(self.index, repo_name)
        (row,) = await repo.find_where(
            all_of(eq("owner_id", OWNER), eq("entry_id", entry_id))
        )
        return row.deprecated_by

    async def live_episodes(self) -> list[str]:
        rows = await self.index.episode.find_where(
            all_of(eq("owner_id", OWNER), is_null("deprecated_by"))
        )
        return sorted(row.entry_id for row in rows)


async def _cluster_of(storage: Storage, members: list[str]) -> None:
    await storage.sqlite.cluster.upsert_with_members(
        AlgoCluster(
            id=CLUSTER,
            centroid=np.zeros(1024, dtype=np.float32),
            count=len(members),
            last_ts=_T0,
            preview=[],
            members=members,
        ),
        owner_id=OWNER,
        owner_type="user",
        kind="user_memory",
        member_type="episode",
    )


async def _join_cluster(storage: Storage, member: str) -> None:
    """Add one member to the seeded cluster, snapshot unchanged."""
    cluster = await storage.sqlite.cluster.get_with_members(CLUSTER)
    assert cluster is not None
    await storage.sqlite.cluster.upsert_with_members(
        cluster.model_copy(update={"members": [*cluster.members, member]}),
        owner_id=OWNER,
        owner_type="user",
        kind="user_memory",
        member_type="episode",
    )


@pytest.fixture(autouse=True)
def _reset_partition_locks() -> None:
    _reset_for_tests()


@pytest.fixture
async def world_factory(
    tmp_path: Path, storage: Storage
) -> AsyncIterator[Callable[..., _World]]:
    root = MemoryRoot(tmp_path)
    root.ensure()
    await storage.create_schema()
    await storage.index.ensure_business_indexes()

    def build(*replies: str) -> _World:
        return _World(storage, root, FakeLLMClient(responses=list(replies)))

    yield build


async def test_init_merge_deprecates_members_and_their_facts(
    world_factory: Callable[..., _World],
) -> None:
    world = world_factory(user_track_script("reflect"))
    refs = await world.seed(3)
    await _cluster_of(world.storage, [r.entry_id for r in refs])

    emitted, (report,) = await world.reflect()

    merged_id = report.merged_entry_id
    assert (report.mode, report.source_count, report.cluster_id) == ("init", 3, CLUSTER)
    (event,) = emitted
    assert (event.episode_entry_id, event.source) == (merged_id, "reflection")
    for ref in refs:
        assert await world.deprecated_by("episode", ref.entry_id) == merged_id
    facts = await world.index.atomic_fact.find_where(eq("owner_id", OWNER))
    assert len(facts) == 3
    assert {f.deprecated_by for f in facts} == {merged_id}
    assert await world.live_episodes() == [merged_id]
    merged = await world.index.episode.find_where(eq("entry_id", merged_id))
    assert merged[0].parent_type == "cluster"
    assert merged[0].summary == json.loads(user_track_script("reflect"))["summary"]
    members = await world.storage.sqlite.cluster.get_members_with_type(CLUSTER)
    assert members == [(merged_id, "episode")]


async def test_update_merge_replaces_the_previous_merge(
    world_factory: Callable[..., _World],
) -> None:
    world = world_factory(
        _reply("Andrew adopted Toby and Buddy.", "Andrew pets v1"),
        _reply("Andrew adopted Toby, Buddy, and Scout.", "Andrew pets v2"),
    )
    refs = await world.seed(3)
    await _cluster_of(world.storage, [r.entry_id for r in refs])
    _, (first,) = await world.reflect()

    (fourth,) = await world.seed(1, start=3)
    await _join_cluster(world.storage, fourth.entry_id)
    _, (second,) = await world.reflect()

    assert (second.mode, second.source_count) == ("update", 2)
    v1, v2 = first.merged_entry_id, second.merged_entry_id
    assert await world.deprecated_by("episode", v1) == v2
    assert await world.deprecated_by("episode", fourth.entry_id) == v2
    for ref in refs:  # the first merge's members stay retired by it
        assert await world.deprecated_by("episode", ref.entry_id) == v1
    assert await world.live_episodes() == [v2]


async def test_deprecation_survives_an_index_rebuild(
    world_factory: Callable[..., _World],
) -> None:
    """Drop the index, rebuild it from markdown: the marks are still there."""
    world = world_factory(user_track_script("reflect"))
    refs = await world.seed(3)
    await _cluster_of(world.storage, [r.entry_id for r in refs])
    _, (report,) = await world.reflect()

    await world.rebuild_index()

    merged_id = report.merged_entry_id
    assert await world.live_episodes() == [merged_id]
    for ref in refs:
        assert await world.deprecated_by("episode", ref.entry_id) == merged_id
    facts = await world.index.atomic_fact.find_where(eq("owner_id", OWNER))
    assert {f.deprecated_by for f in facts} == {merged_id}
    session_rows = await world.index.episode.find_where(
        all_of(
            eq("owner_id", OWNER), is_null("deprecated_by"), eq("session_id", "s_test")
        )
    )
    assert session_rows == [], "the merged episode belongs to no session"

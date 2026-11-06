"""Reflection restamps a cluster from the members it has now (#563).

Reflection reads a cluster's members, calls the LLM, writes the merged
episode, and only then folds the sources into it. Profile clustering may add
an episode to the same cluster in between. Before #563 the finisher wrote
``count=1`` and the merged episode's vector as the centroid regardless, so
the new member stayed in ``members`` but neither ``count`` nor ``centroid``
nor ``last_ts`` described it, and ``extract_user_profile``'s gate (a sum of
``count``) undercounted.

Real SQLite cluster store, real ``trigger_profile_clustering`` and real
``ReflectionFinisher``; the interleaving is forced instead of raced. The
finisher must also wait for clustering's read-to-write owner lock (#546).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

import pytest

from everos.config import SqliteSettings
from everos.core.locks import LOCKS
from everos.infra.persistence import MemoryRoot
from everos.infra.persistence.sqlite import (
    SqliteClusterStore,
    SqliteDatabase,
    SqliteRepos,
)
from everos.memory.events import EpisodeExtracted, ReflectedEpisode, ReflectionContext
from everos.memory.models import Cluster, ClusterKind, Episode
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


async def test_reflection_waits_for_clustering_read_to_write(
    cluster_store: SqliteClusterStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    stores = make_stores(cluster=cluster_store)
    deps = make_deps(embedding=_Embedder(), stores=stores)
    for i in (1, 2, 3):
        seed_sources(deps, memcell_id=f"mc_{i}", owner_id=_OWNER)
        stores.episode.seed(  # type: ignore[attr-defined]
            _SCOPE,
            f"ep{i}",
            Episode(
                owner_id=_OWNER,
                episode=_JOINER_TEXT if i == 3 else f"source {i}",
                summary="source",
                timestamp=_T0,
                parent_id=f"mc_{i}",
            ),
        )
    stores.episode.seed(  # type: ignore[attr-defined]
        _SCOPE,
        "ep_merged",
        Episode(
            owner_id=_OWNER,
            episode=_MERGED_TEXT,
            summary="merged",
            timestamp=_T0,
            parent_id=_CLUSTER,
            parent_type="cluster",
        ),
    )
    await cluster_store.upsert(
        _SCOPE,
        "user_memory",
        Cluster(
            id=_CLUSTER,
            centroid=(0.1,) * _DIM,
            count=2,
            last_ts=_T0,
            members=["ep1", "ep2"],
        ),
        member_type="episode",
    )
    sources = await stores.episode.read_many(_SCOPE, ["ep1", "ep2"])
    context = ReflectionContext(
        cluster_id=_CLUSTER,
        mode="init",
        source_members=await cluster_store.members(_CLUSTER),
        episodes=[
            ReflectedEpisode(
                entry_id=s.ref.entry_id,
                md_path=s.ref.md_path,
                parent_id=s.item.parent_id,
            )
            for s in sources
        ],
        merged_episode=_MERGED_TEXT,
        last_ts=_T0,
    )
    finisher = ReflectionFinisher(
        cluster_store=cluster_store,
        episode_store=stores.episode,
        atomic_fact_store=stores.atomic_fact,
        atomic_fact_index=make_indexes().atomic_fact,
        report_store=stores.reflection_report,
        embedder=_Embedder(),
    )
    read_finished, release_clustering = asyncio.Event(), asyncio.Event()
    reflection_progress, reflection_written = asyncio.Event(), asyncio.Event()
    owner_lock = (
        "partition:trigger_profile_clustering",
        f"{_SCOPE.app_id}:{_SCOPE.project_id}:{_OWNER}",
    )
    real_list = cluster_store.list_for_owner
    real_replace = cluster_store.replace_members
    finisher_task: asyncio.Task[Any] | None = None

    async def pause_after_read(scope: Scope, kind: ClusterKind) -> list[Cluster]:
        snapshot = await real_list(scope, kind)
        assert owner_lock in LOCKS.held()
        read_finished.set()
        await release_clustering.wait()
        return snapshot

    async def record_replace(*args: Any, **kwargs: Any) -> None:
        await real_replace(*args, **kwargs)
        reflection_written.set()
        reflection_progress.set()

    def acquiring(
        task: asyncio.Task[Any],
        wanted: tuple[str, str],
        held: frozenset[tuple[str, str]],
    ) -> None:
        if task is finisher_task and wanted == owner_lock:
            reflection_progress.set()

    monkeypatch.setattr(cluster_store, "list_for_owner", pause_after_read)
    monkeypatch.setattr(cluster_store, "replace_members", record_replace)
    LOCKS.set_acquire_hook(acquiring)
    tasks: list[asyncio.Task[Any]] = []
    try:
        async with asyncio.timeout(5):
            clustering_task = asyncio.create_task(
                trigger_profile_clustering(
                    EpisodeExtracted(
                        memcell_id="mc_3",
                        episode_entry_id="ep3",
                        episode_text=_JOINER_TEXT,
                        episode_timestamp_ms=_T0,
                        owner_id=_OWNER,
                        session_id="s1",
                    ),
                    deps,
                )
            )
            tasks.append(clustering_task)
            await read_finished.wait()
            finisher_task = asyncio.create_task(
                finisher.finish(
                    scope=_SCOPE, merged_entry_id="ep_merged", context=context
                )
            )
            tasks.append(finisher_task)
            # The hook runs just before the real acquire. With the lock
            # removed, the real SQLite write wakes us instead and must fail.
            await reflection_progress.wait()
            assert not reflection_written.is_set(), (
                "reflection wrote while clustering still held its read snapshot"
            )
            assert not finisher_task.done()
            assert owner_lock in LOCKS.held(clustering_task)
            release_clustering.set()
            events, report = await asyncio.gather(clustering_task, finisher_task)
            assert len(events) == 1
            assert report is not None
    finally:
        release_clustering.set()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        LOCKS.set_acquire_hook(None)

    assert reflection_written.is_set()
    final = await cluster_store.get(_CLUSTER)
    assert final is not None
    assert sorted(final.members) == ["ep3", "ep_merged"]
    assert final.count == 2

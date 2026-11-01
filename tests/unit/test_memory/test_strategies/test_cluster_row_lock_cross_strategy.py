"""Regression for #546: clustering and reflection must lock the cluster row.

Real SQLite cluster store and real locks; only the embedders are fixed.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from everos.config import SqliteSettings
from everos.infra.persistence import MemoryRoot
from everos.infra.persistence.sqlite import (
    SqliteClusterStore,
    SqliteDatabase,
    SqliteRepos,
)
from everos.memory._partition_locks import _pools
from everos.memory.events import EpisodeExtracted, ReflectionContext
from everos.memory.models import Cluster
from everos.memory.ports import Scope
from everos.memory.reflection.orchestrator import ReflectionFinisher
from everos.memory.strategies import trigger_profile_clustering
from tests.fakes import make_deps, make_indexes, make_stores

_DIM = 8
_T0 = 1_700_000_000_000


class _Embedder:
    def __init__(self, value: float) -> None:
        self.dim = _DIM
        self.value = value

    async def embed(self, _text: str) -> list[float]:
        return [self.value] * _DIM


class _GatingStore:
    """Real cluster store whose list_for_owner pauses after it reads."""

    def __init__(self, inner: SqliteClusterStore) -> None:
        self._inner = inner
        self.read_done = asyncio.Event()
        self.go = asyncio.Event()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def list_for_owner(self, *args: Any, **kwargs: Any) -> Any:
        rows = await self._inner.list_for_owner(*args, **kwargs)
        self.read_done.set()
        await self.go.wait()
        return rows


def _event(owner_id: str) -> EpisodeExtracted:
    return EpisodeExtracted(
        memcell_id="mc_1",
        episode_entry_id="ep3",
        episode_text="text ep3",
        episode_timestamp_ms=_T0 + 1_000,
        owner_id=owner_id,
        session_id="s1",
    )


async def _seed(store: SqliteClusterStore, scope: Scope, cluster_id: str) -> None:
    cluster = Cluster(
        id=cluster_id,
        centroid=(0.1,) * _DIM,
        count=2,
        last_ts=_T0,
        preview=["p1", "p2"],
        members=["ep1", "ep2"],
    )
    await store.upsert(scope, "user_memory", cluster, member_type="episode")


async def _merge(store: SqliteClusterStore, scope: Scope, cluster_id: str) -> None:
    stores, indexes = make_stores(), make_indexes()
    finisher = ReflectionFinisher(
        cluster_store=store,
        episode_store=stores.episode,
        atomic_fact_store=stores.atomic_fact,
        atomic_fact_index=indexes.atomic_fact,
        report_store=stores.reflection_report,
        embedder=_Embedder(0.2),
    )
    context = ReflectionContext(
        cluster_id=cluster_id,
        mode="init",
        source_members=[("ep1", "episode"), ("ep2", "episode")],
        episodes=[],
        merged_episode="merged episode",
        last_ts=_T0 + 1_000,
    )
    assert await finisher.finish(
        scope=scope, merged_entry_id="ep_merged", context=context
    )


def _assert_merged(cluster: Cluster | None) -> None:
    assert cluster is not None
    assert cluster.count == len(cluster.members) == 2
    assert sorted(cluster.members) == ["ep3", "ep_merged"]
    assert cluster.centroid[0] == pytest.approx(0.15)


@pytest.fixture
async def store(tmp_path: Path) -> AsyncIterator[SqliteClusterStore]:
    database = SqliteDatabase(MemoryRoot(tmp_path / "root"), SqliteSettings())
    await database.create_schema()
    try:
        yield SqliteClusterStore(SqliteRepos.over(database).cluster)
    finally:
        await database.dispose()


async def test_clustering_does_not_overwrite_reflections_merge(
    store: SqliteClusterStore,
) -> None:
    serial = Scope(
        app_id="default", project_id="default", owner_id="u_serial", owner_type="user"
    )
    await _seed(store, serial, "cl_serial")
    await _merge(store, serial, "cl_serial")
    deps = make_deps(embedding=_Embedder(0.1), stores=make_stores(cluster=store))
    await trigger_profile_clustering(_event("u_serial"), deps)
    _assert_merged(await store.get("cl_serial"))

    race = Scope(
        app_id="default", project_id="default", owner_id="u_race", owner_type="user"
    )
    await _seed(store, race, "cl_race")
    gated = _GatingStore(store)
    deps = make_deps(embedding=_Embedder(0.1), stores=make_stores(cluster=gated))
    clustering = asyncio.create_task(trigger_profile_clustering(_event("u_race"), deps))
    await asyncio.wait_for(gated.read_done.wait(), timeout=10)
    # Reflection is not blocked by the clustering strategy lock; it writes the
    # row, and clustering must re-read it instead of replaying the stale snapshot.
    await asyncio.wait_for(_merge(store, race, "cl_race"), timeout=10)
    mid = await store.get("cl_race")
    assert mid is not None and mid.count == 1 and sorted(mid.members) == ["ep_merged"]

    gated.go.set()
    await asyncio.wait_for(clustering, timeout=10)
    _assert_merged(await store.get("cl_race"))
    # Both paths write one row, so they share a data-keyed (not strategy) lock.
    assert "default:default:cl_race" in _pools["cluster"]

"""Regression for #546: clustering must re-read a cluster row reflection merged.

Real SQLite cluster store and real locks; only the embedders, the source
episodes, and the snapshot returned by list_for_owner are fixed.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
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
from everos.memory.events import EpisodeExtracted, ReflectionContext
from everos.memory.models import Cluster, Episode
from everos.memory.ports import Scope
from everos.memory.reflection.orchestrator import ReflectionFinisher
from everos.memory.strategies import trigger_profile_clustering
from tests.fakes import (
    make_deps,
    make_indexes,
    make_stores,
    seed_sources,
)

_DIM = 8
_T0 = 1_700_000_000_000
_OWNER = "u_race"
_CLUSTER = "cl_race"


class _Embedder:
    def __init__(self, value: float) -> None:
        self.dim = _DIM
        self.value = value

    async def embed(self, _text: str) -> list[float]:
        return [self.value] * _DIM


class _StaleListStore:
    """Real cluster store that serves a snapshot captured before a merge."""

    def __init__(self, inner: SqliteClusterStore, stale: list[Cluster]) -> None:
        self._inner = inner
        self._stale = stale

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def list_for_owner(self, *args: Any, **kwargs: Any) -> list[Cluster]:
        return self._stale


def _event(owner_id: str) -> EpisodeExtracted:
    return EpisodeExtracted(
        memcell_id="mc_1",
        episode_entry_id="ep3",
        episode_text="text ep3",
        episode_timestamp_ms=_T0 + 1_000,
        owner_id=owner_id,
        session_id="s1",
    )


def _episode(owner_id: str, parent_id: str, text: str) -> Episode:
    return Episode(
        owner_id=owner_id,
        episode=text,
        summary=text,
        timestamp=_T0,
        subject=text,
        parent_id=parent_id,
    )


async def _seed(store: SqliteClusterStore, scope: Scope) -> None:
    cluster = Cluster(
        id=_CLUSTER,
        centroid=(0.1,) * _DIM,
        count=2,
        last_ts=_T0,
        preview=["p1", "p2"],
        members=["ep1", "ep2"],
    )
    await store.upsert(scope, "user_memory", cluster, member_type="episode")


async def _merge(store: SqliteClusterStore, scope: Scope) -> None:
    stores, indexes = make_stores(), make_indexes()
    stores.episode.seed(  # type: ignore[attr-defined]
        scope, "ep_merged", _episode(scope.owner_id, _CLUSTER, "merged")
    )
    finisher = ReflectionFinisher(
        cluster_store=store,
        episode_store=stores.episode,
        atomic_fact_store=stores.atomic_fact,
        atomic_fact_index=indexes.atomic_fact,
        report_store=stores.reflection_report,
        embedder=_Embedder(0.2),
    )
    context = ReflectionContext(
        cluster_id=_CLUSTER,
        mode="init",
        source_members=[("ep1", "episode"), ("ep2", "episode")],
        episodes=[],
        merged_episode="merged episode",
        last_ts=_T0 + 1_000,
    )
    report = await finisher.finish(
        scope=scope, merged_entry_id="ep_merged", context=context
    )
    assert report is not None


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


async def test_clustering_re_reads_reflections_merge(store: SqliteClusterStore) -> None:
    scope = Scope(
        app_id="default",
        project_id="default",
        owner_id=_OWNER,
        owner_type="user",
    )
    await _seed(store, scope)
    # The snapshot clustering would have read before reflection merged the row.
    stale = await store.list_for_owner(scope, "user_memory")
    await _merge(store, scope)

    deps = make_deps(
        embedding=_Embedder(0.1),
        stores=make_stores(cluster=_StaleListStore(store, stale)),
    )
    seed_sources(
        deps,
        memcell_id="mc_1",
        episode_entry_id="ep3",
        owner_id=_OWNER,
    )
    await trigger_profile_clustering(_event(_OWNER), deps)

    _assert_merged(await store.get(_CLUSTER))
    # The clustering upsert and the reflection replace_members both take the
    # data-keyed row lock, so the lock exists for this cluster after both ran.
    assert ("partition:cluster", "default:default:cl_race") in LOCKS._locks

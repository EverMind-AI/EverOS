"""Tests for :func:`trigger_profile_clustering`.

Mirrors the skill-side test layout: mock embedder + cluster_repo +
cluster_by_geometry, drive the strategy via :class:`FakeStrategyContext`,
verify a single :class:`ProfileClusterUpdated` event is emitted.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import structlog.testing
from everalgo.clustering import Cluster as AlgoCluster

from everos.infra.ome.testing import FakeStrategyContext
from everos.memory._partition_locks import _reset_for_tests
from everos.memory.events import EpisodeExtracted, ProfileClusterUpdated
from everos.memory.strategies.trigger_profile_clustering import (
    trigger_profile_clustering,
)


@pytest.fixture(autouse=True)
def _isolate_partition_locks() -> None:
    _reset_for_tests()


def _event(
    *,
    owner_id: str = "u_alice",
    memcell_id: str = "mc_aaaaaaaaaaa1",
    episode_entry_id: str = "ep_20260517_0001",
    episode_text: str = "alice likes hiking",
    episode_timestamp_ms: int = 1_700_000_001_000,
) -> EpisodeExtracted:
    return EpisodeExtracted(
        memcell_id=memcell_id,
        episode_entry_id=episode_entry_id,
        episode_text=episode_text,
        episode_timestamp_ms=episode_timestamp_ms,
        owner_id=owner_id,
        session_id="s_test",
    )


async def test_strategy_meta_is_attached() -> None:
    meta = trigger_profile_clustering.meta
    assert meta.name == "trigger_profile_clustering"
    assert EpisodeExtracted in meta.trigger.on
    assert meta.emits == frozenset({ProfileClusterUpdated})
    assert meta.max_retries == 2
    assert meta.applies_to is not None


@pytest.mark.asyncio
async def test_creates_new_cluster_when_no_existing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty existing → cluster_by_geometry returns None → new cluster persisted."""
    embedder = MagicMock()
    embedder.embed = AsyncMock(return_value=[0.1] * 1024)
    ctx = FakeStrategyContext()

    with (
        patch(
            "everos.memory.strategies.trigger_profile_clustering.get_embedder",
            return_value=embedder,
        ),
        patch(
            "everos.memory.strategies.trigger_profile_clustering.cluster_repo"
        ) as mock_repo,
        patch(
            "everos.memory.strategies.trigger_profile_clustering.cluster_by_geometry",
            new=MagicMock(return_value=None),
        ) as mock_cluster,
        patch(
            "everos.memory.strategies.trigger_profile_clustering.mint_cluster_id",
            return_value="cl_newuser00001",
        ),
        structlog.testing.capture_logs() as captured,
    ):
        mock_repo.list_for_owner = AsyncMock(return_value=[])
        mock_repo.upsert_with_members = AsyncMock(return_value=None)

        await trigger_profile_clustering(_event(), ctx)

    args, _ = mock_cluster.call_args
    new_cluster, existing = args
    assert isinstance(new_cluster, AlgoCluster)
    assert new_cluster.id == "cl_newuser00001"
    assert new_cluster.count == 1
    assert new_cluster.last_ts == 1_700_000_001_000
    assert new_cluster.members == ["ep_20260517_0001"]
    assert new_cluster.preview == ["alice likes hiking"]
    assert existing == []

    upsert_args = mock_repo.upsert_with_members.call_args
    persisted = upsert_args.args[0]
    assert persisted.id == "cl_newuser00001"
    assert upsert_args.kwargs == {
        "owner_id": "u_alice",
        "owner_type": "user",
        "kind": "user_memory",
        "member_type": "episode",
        "app_id": "default",
        "project_id": "default",
    }

    emitted = [e for e in ctx.emitted if isinstance(e, ProfileClusterUpdated)]
    assert len(emitted) == 1
    assert emitted[0].memcell_id == "mc_aaaaaaaaaaa1"
    assert emitted[0].cluster_id == "cl_newuser00001"
    assert emitted[0].owner_id == "u_alice"

    matching = [r for r in captured if r.get("event") == "profile_cluster_updated"]
    assert matching, "expected profile_cluster_updated log line"


@pytest.mark.asyncio
async def test_merges_into_existing_cluster_when_algo_matches() -> None:
    """algo returns merged Cluster → persisted under the existing id."""
    embedder = MagicMock()
    embedder.embed = AsyncMock(return_value=[0.2] * 1024)
    ctx = FakeStrategyContext()

    existing_cluster = AlgoCluster(
        id="cl_existing0001",
        centroid=np.array([0.15] * 1024, dtype=np.float32),
        count=1,
        last_ts=1_700_000_000_000,
        preview=["earlier episode"],
        members=["ep_20260517_0000"],
    )
    merged_cluster = AlgoCluster(
        id="cl_existing0001",
        centroid=np.array([0.17] * 1024, dtype=np.float32),
        count=2,
        last_ts=1_700_000_001_000,
        preview=["earlier episode", "alice likes hiking"],
        members=["ep_20260517_0000", "ep_20260517_0001"],
    )

    with (
        patch(
            "everos.memory.strategies.trigger_profile_clustering.get_embedder",
            return_value=embedder,
        ),
        patch(
            "everos.memory.strategies.trigger_profile_clustering.cluster_repo"
        ) as mock_repo,
        patch(
            "everos.memory.strategies.trigger_profile_clustering.cluster_by_geometry",
            new=MagicMock(return_value=merged_cluster),
        ),
    ):
        mock_repo.list_for_owner = AsyncMock(return_value=[existing_cluster])
        mock_repo.upsert_with_members = AsyncMock(return_value=None)

        await trigger_profile_clustering(_event(), ctx)

    persisted = mock_repo.upsert_with_members.call_args.args[0]
    assert persisted.id == "cl_existing0001"
    assert persisted.count == 2

    emitted = [e for e in ctx.emitted if isinstance(e, ProfileClusterUpdated)]
    assert len(emitted) == 1
    assert emitted[0].cluster_id == "cl_existing0001"


# ── partition lock (owner_id-level serialisation) ────────────────────────


async def _run_serialisation_probe(owner_a: str, owner_b: str) -> list[str]:
    """Drive two trigger_profile_clustering runs and record entry/exit order."""
    log: list[str] = []

    def mock_cluster_by_geometry(_new_cluster, _existing, **_kw):
        # Sync, matching the real algo signature (must not be awaited).
        return None

    async def mock_upsert(cluster, **_kwargs):
        # Delay inside the partition-lock critical section so two concurrent
        # runs on the same owner are observably serialised. cluster_by_geometry
        # is synchronous now, so the await point moves here.
        mid = cluster.members[0]
        log.append(f"enter:{mid}")
        await asyncio.sleep(0.01)
        log.append(f"leave:{mid}")

    mock_embedder = MagicMock()
    mock_embedder.embed = AsyncMock(return_value=np.zeros(1024, dtype=np.float32))

    with (
        patch(
            "everos.memory.strategies.trigger_profile_clustering.get_embedder",
            return_value=mock_embedder,
        ),
        patch(
            "everos.memory.strategies.trigger_profile_clustering.cluster_repo"
        ) as mock_repo,
        patch(
            "everos.memory.strategies.trigger_profile_clustering.cluster_by_geometry",
            new=mock_cluster_by_geometry,
        ),
    ):
        mock_repo.list_for_owner = AsyncMock(return_value=[])
        mock_repo.upsert_with_members = mock_upsert

        await asyncio.gather(
            trigger_profile_clustering(
                _event(owner_id=owner_a, episode_entry_id="ep_run_a"),
                FakeStrategyContext(),
            ),
            trigger_profile_clustering(
                _event(owner_id=owner_b, episode_entry_id="ep_run_b"),
                FakeStrategyContext(),
            ),
        )
    return log


async def test_partition_lock_serialises_runs_on_same_owner() -> None:
    """Two runs sharing ``owner_id`` must not overlap critical sections."""
    log = await _run_serialisation_probe("u_alice", "u_alice")
    assert log in (
        ["enter:ep_run_a", "leave:ep_run_a", "enter:ep_run_b", "leave:ep_run_b"],
        ["enter:ep_run_b", "leave:ep_run_b", "enter:ep_run_a", "leave:ep_run_a"],
    )


async def test_partition_lock_lets_different_owners_run_in_parallel() -> None:
    """Runs on distinct ``owner_id`` must overlap (no false serialisation)."""
    log = await _run_serialisation_probe("u_alice", "u_bob")
    assert log.index("enter:ep_run_a") < log.index("leave:ep_run_b")
    assert log.index("enter:ep_run_b") < log.index("leave:ep_run_a")


async def test_applies_to_rejects_non_pipeline_source() -> None:
    """Events with source != 'pipeline' must not pass the applies_to gate."""
    meta = trigger_profile_clustering.meta
    pipeline_event = _event()
    assert meta.applies_to(pipeline_event) is True

    reflection_event = EpisodeExtracted(
        memcell_id="mc_merged",
        episode_entry_id="ep_20260517_0002",
        episode_text="merged narrative",
        episode_timestamp_ms=1_700_000_001_000,
        owner_id="u_alice",
        session_id="reflection",
        source="reflection",
    )
    assert meta.applies_to(reflection_event) is False

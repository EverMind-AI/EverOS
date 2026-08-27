"""Tests for :func:`trigger_decision_clustering`.

Mirrors :mod:`test_trigger_profile_clustering`: mock embedder +
cluster_repo + cluster_by_geometry, drive the strategy via
:class:`FakeStrategyContext`, verify a single
:class:`DecisionClusterUpdated` event is emitted with the sqlite
``kind=decision`` / ``member_type=decision`` write path.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import structlog.testing
from everalgo.clustering import Cluster as AlgoCluster

from everos.component.embedding import EmbeddingCapability, EmbeddingProvider
from everos.infra.ome.testing import FakeStrategyContext
from everos.memory._partition_locks import _reset_for_tests
from everos.memory.events import DecisionClusterUpdated, DecisionExtracted
from everos.memory.strategies.trigger_decision_clustering import (
    trigger_decision_clustering,
)


def _install_embedder(
    monkeypatch: pytest.MonkeyPatch, embedder: EmbeddingProvider
) -> None:
    """Install ``embedder`` as the process-wide embedding capability."""
    import everos.component.embedding.accessor as acc

    monkeypatch.setattr(acc, "_capability", EmbeddingCapability(provider=embedder))


@pytest.fixture(autouse=True)
def _isolate_partition_locks() -> None:
    _reset_for_tests()


def _event(
    *,
    owner_id: str = "u_alice",
    memcell_id: str = "mc_aaaaaaaaaaa1",
    decision_entry_id: str = "dc_20260517_0001",
    title: str = "Use Rust on device",
    decision_text: str = "Device Runtime uses Rust.",
    reason: str = "Need deterministic latency.",
    impact: str | None = "Keep Python in the agent runtime.",
    tags: list[str] | None = None,
    decision_timestamp_ms: int = 1_700_000_001_000,
    source: str = "pipeline",
) -> DecisionExtracted:
    return DecisionExtracted(
        memcell_id=memcell_id,
        decision_entry_id=decision_entry_id,
        title=title,
        decision_text=decision_text,
        reason=reason,
        impact=impact,
        tags=tags if tags is not None else ["runtime"],
        decision_timestamp_ms=decision_timestamp_ms,
        owner_id=owner_id,
        session_id="s_test",
        source=source,
    )


async def test_strategy_meta_is_attached() -> None:
    meta = trigger_decision_clustering.meta
    assert meta.name == "trigger_decision_clustering"
    assert DecisionExtracted in meta.trigger.on
    assert meta.emits == frozenset({DecisionClusterUpdated})
    assert meta.max_retries == 2
    assert meta.applies_to is not None


@pytest.mark.asyncio
async def test_creates_new_cluster_when_no_existing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty existing → cluster_by_geometry returns None → new cluster persisted."""
    embedder = MagicMock()
    embedder.embed = AsyncMock(return_value=[0.1] * 1024)
    _install_embedder(monkeypatch, embedder)
    ctx = FakeStrategyContext()

    with (
        patch(
            "everos.memory.strategies.trigger_decision_clustering.cluster_repo"
        ) as mock_repo,
        patch(
            "everos.memory.strategies.trigger_decision_clustering.cluster_by_geometry",
            new=MagicMock(return_value=None),
        ) as mock_cluster,
        patch(
            "everos.memory.strategies.trigger_decision_clustering.mint_cluster_id",
            return_value="cl_newdec000001",
        ),
        structlog.testing.capture_logs() as captured,
    ):
        mock_repo.list_for_owner = AsyncMock(return_value=[])
        mock_repo.upsert_with_members = AsyncMock(return_value=None)

        await trigger_decision_clustering(_event(), ctx)

    args, _ = mock_cluster.call_args
    new_cluster, existing = args
    assert isinstance(new_cluster, AlgoCluster)
    assert new_cluster.id == "cl_newdec000001"
    assert new_cluster.count == 1
    assert new_cluster.last_ts == 1_700_000_001_000
    assert new_cluster.members == ["dc_20260517_0001"]
    assert new_cluster.preview == ["Device Runtime uses Rust."]
    assert existing == []

    mock_repo.list_for_owner.assert_awaited_once_with(
        "u_alice",
        "decision",
        app_id="default",
        project_id="default",
    )

    upsert_args = mock_repo.upsert_with_members.call_args
    persisted = upsert_args.args[0]
    assert persisted.id == "cl_newdec000001"
    assert upsert_args.kwargs == {
        "owner_id": "u_alice",
        "owner_type": "user",
        "kind": "decision",
        "member_type": "decision",
        "app_id": "default",
        "project_id": "default",
    }

    emitted = [e for e in ctx.emitted if isinstance(e, DecisionClusterUpdated)]
    assert len(emitted) == 1
    assert emitted[0].memcell_id == "mc_aaaaaaaaaaa1"
    assert emitted[0].decision_entry_id == "dc_20260517_0001"
    assert emitted[0].cluster_id == "cl_newdec000001"
    assert emitted[0].owner_id == "u_alice"
    assert emitted[0].title == "Use Rust on device"
    assert emitted[0].decision_text == "Device Runtime uses Rust."
    assert emitted[0].reason == "Need deterministic latency."
    assert emitted[0].impact == "Keep Python in the agent runtime."
    assert emitted[0].tags == ["runtime"]
    assert emitted[0].decision_timestamp_ms == 1_700_000_001_000

    matching = [r for r in captured if r.get("event") == "decision_cluster_updated"]
    assert matching, "expected decision_cluster_updated log line"


@pytest.mark.asyncio
async def test_merges_into_existing_cluster_when_algo_matches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """algo returns merged Cluster → persisted under the existing id."""
    embedder = MagicMock()
    embedder.embed = AsyncMock(return_value=[0.2] * 1024)
    _install_embedder(monkeypatch, embedder)
    ctx = FakeStrategyContext()

    existing_cluster = AlgoCluster(
        id="cl_existing0001",
        centroid=np.array([0.15] * 1024, dtype=np.float32),
        count=1,
        last_ts=1_700_000_000_000,
        preview=["earlier decision"],
        members=["dc_20260517_0000"],
    )
    merged_cluster = AlgoCluster(
        id="cl_existing0001",
        centroid=np.array([0.17] * 1024, dtype=np.float32),
        count=2,
        last_ts=1_700_000_001_000,
        preview=["earlier decision", "Device Runtime uses Rust."],
        members=["dc_20260517_0000", "dc_20260517_0001"],
    )

    with (
        patch(
            "everos.memory.strategies.trigger_decision_clustering.cluster_repo"
        ) as mock_repo,
        patch(
            "everos.memory.strategies.trigger_decision_clustering.cluster_by_geometry",
            new=MagicMock(return_value=merged_cluster),
        ),
    ):
        mock_repo.list_for_owner = AsyncMock(return_value=[existing_cluster])
        mock_repo.upsert_with_members = AsyncMock(return_value=None)

        await trigger_decision_clustering(_event(), ctx)

    persisted = mock_repo.upsert_with_members.call_args.args[0]
    assert persisted.id == "cl_existing0001"
    assert persisted.count == 2

    emitted = [e for e in ctx.emitted if isinstance(e, DecisionClusterUpdated)]
    assert len(emitted) == 1
    assert emitted[0].cluster_id == "cl_existing0001"


async def _run_serialisation_probe(
    owner_a: str, owner_b: str, monkeypatch: pytest.MonkeyPatch
) -> list[str]:
    """Drive two trigger_decision_clustering runs and record entry/exit order."""
    log: list[str] = []

    def mock_cluster_by_geometry(_new_cluster, _existing, **_kw):
        return None

    async def mock_upsert(cluster, **_kwargs):
        mid = cluster.members[0]
        log.append(f"enter:{mid}")
        await asyncio.sleep(0.01)
        log.append(f"leave:{mid}")

    mock_embedder = MagicMock()
    mock_embedder.embed = AsyncMock(return_value=np.zeros(1024, dtype=np.float32))
    _install_embedder(monkeypatch, mock_embedder)

    with (
        patch(
            "everos.memory.strategies.trigger_decision_clustering.cluster_repo"
        ) as mock_repo,
        patch(
            "everos.memory.strategies.trigger_decision_clustering.cluster_by_geometry",
            new=mock_cluster_by_geometry,
        ),
    ):
        mock_repo.list_for_owner = AsyncMock(return_value=[])
        mock_repo.upsert_with_members = mock_upsert

        await asyncio.gather(
            trigger_decision_clustering(
                _event(owner_id=owner_a, decision_entry_id="dc_run_a"),
                FakeStrategyContext(),
            ),
            trigger_decision_clustering(
                _event(owner_id=owner_b, decision_entry_id="dc_run_b"),
                FakeStrategyContext(),
            ),
        )
    return log


async def test_partition_lock_serialises_runs_on_same_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two runs sharing ``owner_id`` must not overlap critical sections."""
    log = await _run_serialisation_probe("u_alice", "u_alice", monkeypatch)
    assert log in (
        ["enter:dc_run_a", "leave:dc_run_a", "enter:dc_run_b", "leave:dc_run_b"],
        ["enter:dc_run_b", "leave:dc_run_b", "enter:dc_run_a", "leave:dc_run_a"],
    )


async def test_partition_lock_lets_different_owners_run_in_parallel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runs on distinct ``owner_id`` must overlap (no false serialisation)."""
    log = await _run_serialisation_probe("u_alice", "u_bob", monkeypatch)
    assert log.index("enter:dc_run_a") < log.index("leave:dc_run_b")
    assert log.index("enter:dc_run_b") < log.index("leave:dc_run_a")


@pytest.mark.asyncio
async def test_returns_without_side_effects_when_embedding_unavailable() -> None:
    """Capability unavailable → early return; no embed, no repo, no emit."""
    ctx = FakeStrategyContext()
    with (
        patch(
            "everos.memory.strategies.trigger_decision_clustering"
            ".get_embedding_capability",
            return_value=EmbeddingCapability(provider=None),
        ),
        patch(
            "everos.memory.strategies.trigger_decision_clustering.cluster_repo"
        ) as mock_repo,
        structlog.testing.capture_logs() as captured,
    ):
        mock_repo.list_for_owner = AsyncMock(
            side_effect=AssertionError("cluster_repo must not be touched"),
        )
        mock_repo.upsert_with_members = AsyncMock(
            side_effect=AssertionError("cluster_repo must not be touched"),
        )

        await trigger_decision_clustering(_event(), ctx)

    assert ctx.emitted == []
    gated = [
        e
        for e in captured
        if e.get("event") == "strategy_gated_off_embedding_unavailable"
    ]
    assert len(gated) == 1
    assert gated[0]["strategy_name"] == "trigger_decision_clustering"
    assert gated[0]["owner_id"] == "u_alice"


async def test_applies_to_rejects_non_pipeline_source() -> None:
    """Events with source != 'pipeline' must not pass the applies_to gate."""
    meta = trigger_decision_clustering.meta
    pipeline_event = _event()
    assert meta.applies_to(pipeline_event) is True

    reflection_event = _event(
        memcell_id="mc_merged",
        decision_entry_id="dc_20260517_0002",
        decision_text="merged decision",
        source="reflection",
    )
    assert meta.applies_to(reflection_event) is False

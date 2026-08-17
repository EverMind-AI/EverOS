"""Tests for :func:`trigger_skill_clustering`.

Mock surface: ``cluster_by_llm``, ``get_llm_client``, ``cluster_repo`` —
plus the process-wide :class:`EmbeddingCapability` singleton on the
accessor module, swapped in via :func:`_install_embedder` for tests that
must exercise the strategy past its body-guard.
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
from everos.memory.events import AgentCaseExtracted, SkillClusterUpdated
from everos.memory.strategies.trigger_skill_clustering import (
    trigger_skill_clustering,
)


def _install_embedder(
    monkeypatch: pytest.MonkeyPatch, embedder: EmbeddingProvider
) -> None:
    """Install ``embedder`` as the process-wide embedding capability.

    The strategy resolves the embedder via
    ``get_embedding_capability().require()``, so the only injection knob
    is the accessor's cached capability. The autouse fixture in
    ``tests/conftest.py`` seeds ``Capability(provider=None)`` for
    hermeticity; this helper swaps in a live provider for both the
    body-guard check and the subsequent ``.require().embed()`` call.
    """
    import everos.component.embedding.accessor as acc

    monkeypatch.setattr(acc, "_capability", EmbeddingCapability(provider=embedder))


@pytest.fixture(autouse=True)
def _isolate_partition_locks() -> None:
    _reset_for_tests()


def _event(
    *,
    quality_score: float = 0.8,
    case_entry_id: str = "ac_20260517_0001",
    agent_id: str = "agent_42",
    task_intent: str = "summarise the doc",
    case_timestamp_ms: int = 1_700_000_001_000,
    approach: str = "",
    key_insight: str | None = None,
) -> AgentCaseExtracted:
    return AgentCaseExtracted(
        memcell_id="mc_a",
        case_entry_id=case_entry_id,
        task_intent=task_intent,
        approach=approach,
        key_insight=key_insight,
        quality_score=quality_score,
        case_timestamp_ms=case_timestamp_ms,
        agent_id=agent_id,
    )


async def test_strategy_meta_is_attached() -> None:
    meta = trigger_skill_clustering.meta
    assert meta.name == "trigger_skill_clustering"
    assert AgentCaseExtracted in meta.trigger.on
    assert meta.emits == frozenset({SkillClusterUpdated})
    assert meta.max_retries == 2


async def test_skips_when_quality_score_below_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """quality_score < 0.2 → log + early return; no embedding, no LLM, no repo call."""
    exploding_embedder = MagicMock()
    exploding_embedder.embed = AsyncMock(
        side_effect=AssertionError("embedder must not be called on low-quality skip"),
    )
    _install_embedder(monkeypatch, exploding_embedder)

    ctx = FakeStrategyContext()
    with (
        patch(
            "everos.memory.strategies.trigger_skill_clustering.cluster_repo"
        ) as mock_repo,
        patch(
            "everos.memory.strategies.trigger_skill_clustering.cluster_by_llm"
        ) as mock_cluster,
        structlog.testing.capture_logs() as captured,
    ):
        await trigger_skill_clustering(_event(quality_score=0.1), ctx)

    exploding_embedder.embed.assert_not_awaited()
    mock_repo.list_for_owner.assert_not_called()
    mock_cluster.assert_not_called()
    assert ctx.emitted == []
    matching = [
        e for e in captured if e.get("event") == "skill_clustering_skipped_low_quality"
    ]
    assert matching, "expected low-quality skip log line"


async def test_creates_new_cluster_when_no_existing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty existing list → cluster_by_llm returns None → new cluster persisted."""
    embedder = MagicMock()
    embedder.embed = AsyncMock(return_value=[0.1] * 1024)
    _install_embedder(monkeypatch, embedder)
    ctx = FakeStrategyContext()

    with (
        patch(
            "everos.memory.strategies.trigger_skill_clustering.get_llm_client",
            return_value=object(),
        ),
        patch(
            "everos.memory.strategies.trigger_skill_clustering.cluster_repo"
        ) as mock_repo,
        patch(
            "everos.memory.strategies.trigger_skill_clustering.cluster_by_llm",
            new=AsyncMock(return_value=None),
        ) as mock_cluster,
        patch(
            "everos.memory.strategies.trigger_skill_clustering.mint_cluster_id",
            return_value="cl_newxxxx0001",
        ),
    ):
        mock_repo.list_for_owner = AsyncMock(return_value=[])
        mock_repo.upsert_with_members = AsyncMock(return_value=None)

        await trigger_skill_clustering(_event(), ctx)

    # cluster_by_llm called with the size-1 new cluster + empty existing.
    args, _kwargs = mock_cluster.call_args
    new_cluster, existing = args
    assert isinstance(new_cluster, AlgoCluster)
    assert new_cluster.id == "cl_newxxxx0001"
    assert new_cluster.count == 1
    assert new_cluster.last_ts == 1_700_000_001_000
    assert new_cluster.members == ["ac_20260517_0001"]
    assert new_cluster.preview == ["summarise the doc"]
    np.testing.assert_allclose(
        np.asarray(new_cluster.centroid), np.array([0.1] * 1024, dtype=np.float32)
    )
    assert existing == []

    # upsert called with the new cluster (since merge returned None).
    upsert_args = mock_repo.upsert_with_members.call_args
    persisted = upsert_args.args[0]
    assert persisted.id == "cl_newxxxx0001"
    assert upsert_args.kwargs == {
        "owner_id": "agent_42",
        "owner_type": "agent",
        "kind": "agent_case",
        "member_type": "case",
        "app_id": "default",
        "project_id": "default",
    }

    emitted = [e for e in ctx.emitted if isinstance(e, SkillClusterUpdated)]
    assert len(emitted) == 1
    assert emitted[0].cluster_id == "cl_newxxxx0001"
    assert emitted[0].case_entry_id == "ac_20260517_0001"
    assert emitted[0].agent_id == "agent_42"


async def test_emit_passes_through_case_body_and_vector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """1.2.3+ trigger_skill_clustering passes case body verbatim and includes
    the embedding it just computed so extract_agent_skill doesn't need to
    re-embed for the top-k branch.
    """
    fake_vector = [0.1] * 1024
    embedder = MagicMock()
    embedder.embed = AsyncMock(return_value=fake_vector)
    _install_embedder(monkeypatch, embedder)
    ctx = FakeStrategyContext()

    incoming = _event(
        task_intent="restore replica",
        approach="stop, resync, verify",
        key_insight="watch oplog",
        quality_score=0.8,
        case_timestamp_ms=1_700_000_000_000,
        agent_id="a1",
        case_entry_id="c1",
    )

    with (
        patch(
            "everos.memory.strategies.trigger_skill_clustering.get_llm_client",
            return_value=object(),
        ),
        patch(
            "everos.memory.strategies.trigger_skill_clustering.cluster_repo"
        ) as mock_repo,
        patch(
            "everos.memory.strategies.trigger_skill_clustering.cluster_by_llm",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "everos.memory.strategies.trigger_skill_clustering.mint_cluster_id",
            return_value="cl_newxxxx0001",
        ),
    ):
        mock_repo.list_for_owner = AsyncMock(return_value=[])
        mock_repo.upsert_with_members = AsyncMock(return_value=None)

        await trigger_skill_clustering(incoming, ctx)

    emitted = [e for e in ctx.emitted if isinstance(e, SkillClusterUpdated)]
    assert len(emitted) == 1
    ev = emitted[0]
    assert ev.task_intent == "restore replica"
    assert ev.approach == "stop, resync, verify"
    assert ev.key_insight == "watch oplog"
    assert ev.quality_score == 0.8
    assert ev.case_timestamp_ms == 1_700_000_000_000
    assert ev.case_vector == fake_vector


async def test_merges_into_existing_cluster_when_algo_matches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """algo returns a merged Cluster → persisted with the existing id."""
    embedder = MagicMock()
    embedder.embed = AsyncMock(return_value=[0.2] * 1024)
    _install_embedder(monkeypatch, embedder)
    ctx = FakeStrategyContext()

    existing_cluster = AlgoCluster(
        id="cl_existing0001",
        centroid=np.array([0.15] * 1024, dtype=np.float32),
        count=2,
        last_ts=1_700_000_000_000,
        preview=["earlier intent"],
        members=["ac_20260517_0000"],
    )
    # Simulate everalgo _merge: id passes through from existing, members appended.
    merged_cluster = AlgoCluster(
        id="cl_existing0001",
        centroid=np.array([0.17] * 1024, dtype=np.float32),
        count=3,
        last_ts=1_700_000_001_000,
        preview=["earlier intent", "summarise the doc"],
        members=["ac_20260517_0000", "ac_20260517_0001"],
    )

    with (
        patch(
            "everos.memory.strategies.trigger_skill_clustering.get_llm_client",
            return_value=object(),
        ),
        patch(
            "everos.memory.strategies.trigger_skill_clustering.cluster_repo"
        ) as mock_repo,
        patch(
            "everos.memory.strategies.trigger_skill_clustering.cluster_by_llm",
            new=AsyncMock(return_value=merged_cluster),
        ),
    ):
        mock_repo.list_for_owner = AsyncMock(return_value=[existing_cluster])
        mock_repo.upsert_with_members = AsyncMock(return_value=None)

        await trigger_skill_clustering(_event(), ctx)

    upsert_args = mock_repo.upsert_with_members.call_args
    persisted = upsert_args.args[0]
    assert persisted.id == "cl_existing0001"
    assert persisted.members == ["ac_20260517_0000", "ac_20260517_0001"]
    assert persisted.count == 3

    emitted = [e for e in ctx.emitted if isinstance(e, SkillClusterUpdated)]
    assert len(emitted) == 1
    assert emitted[0].cluster_id == "cl_existing0001"


# ── body-guard (embedding capability) ────────────────────────────────────


async def test_returns_without_side_effects_when_embedding_unavailable() -> None:
    """Capability unavailable → early return; no embed, no repo, no emit.

    Registration is unconditional (OME registry is frozen after start()),
    so the strategy self-gates at body entry. When capability flips back
    on later the very next dispatch runs the full body without a restart.
    """
    ctx = FakeStrategyContext()
    with (
        patch(
            "everos.memory.strategies.trigger_skill_clustering.get_embedding_capability",
            return_value=EmbeddingCapability(provider=None),
        ),
        patch(
            "everos.memory.strategies.trigger_skill_clustering.cluster_repo"
        ) as mock_repo,
        patch(
            "everos.memory.strategies.trigger_skill_clustering.cluster_by_llm",
            new=AsyncMock(
                side_effect=AssertionError("cluster_by_llm must not be called")
            ),
        ),
        structlog.testing.capture_logs() as captured,
    ):
        mock_repo.list_for_owner = AsyncMock(
            side_effect=AssertionError("cluster_repo must not be touched"),
        )
        mock_repo.upsert_with_members = AsyncMock(
            side_effect=AssertionError("cluster_repo must not be touched"),
        )

        await trigger_skill_clustering(_event(), ctx)

    assert ctx.emitted == []
    gated = [
        e
        for e in captured
        if e.get("event") == "strategy_gated_off_embedding_unavailable"
    ]
    assert len(gated) == 1
    assert gated[0]["strategy_name"] == "trigger_skill_clustering"
    assert gated[0]["agent_id"] == "agent_42"


# ── partition lock (agent_id-level serialisation) ────────────────────────


async def _run_serialisation_probe(
    agent_a: str, agent_b: str, monkeypatch: pytest.MonkeyPatch
) -> list[str]:
    """Drive two trigger_skill_clustering runs and record entry/exit order.

    The clustering LLM call is the only awaited work inside the locked
    region — replacing it with a tiny ``asyncio.sleep`` keeps the test
    fast while still proving the lock either does or does not interleave
    the two critical sections.
    """
    log: list[str] = []

    async def mock_cluster_by_llm(new_cluster, _existing, **_kwargs):
        log.append(f"enter:{new_cluster.members[0]}")
        await asyncio.sleep(0.01)
        log.append(f"leave:{new_cluster.members[0]}")
        return None  # no merge → caller persists the size-1 cluster

    mock_embedder = MagicMock()
    mock_embedder.embed = AsyncMock(return_value=np.zeros(1024, dtype=np.float32))
    _install_embedder(monkeypatch, mock_embedder)

    with (
        patch(
            "everos.memory.strategies.trigger_skill_clustering.get_llm_client",
            return_value=object(),
        ),
        patch(
            "everos.memory.strategies.trigger_skill_clustering.cluster_repo"
        ) as mock_repo,
        patch(
            "everos.memory.strategies.trigger_skill_clustering.cluster_by_llm",
            new=mock_cluster_by_llm,
        ),
    ):
        mock_repo.list_for_owner = AsyncMock(return_value=[])
        mock_repo.upsert_with_members = AsyncMock(return_value=None)

        await asyncio.gather(
            trigger_skill_clustering(
                _event(agent_id=agent_a, case_entry_id="ac_run_a"),
                FakeStrategyContext(),
            ),
            trigger_skill_clustering(
                _event(agent_id=agent_b, case_entry_id="ac_run_b"),
                FakeStrategyContext(),
            ),
        )
    return log


async def test_partition_lock_serialises_runs_on_same_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two runs sharing ``agent_id`` must not overlap critical sections."""
    log = await _run_serialisation_probe("agent_42", "agent_42", monkeypatch)
    assert log in (
        ["enter:ac_run_a", "leave:ac_run_a", "enter:ac_run_b", "leave:ac_run_b"],
        ["enter:ac_run_b", "leave:ac_run_b", "enter:ac_run_a", "leave:ac_run_a"],
    )


async def test_partition_lock_lets_different_agents_run_in_parallel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runs on distinct ``agent_id`` must overlap (no false serialisation)."""
    log = await _run_serialisation_probe("agent_42", "agent_43", monkeypatch)
    assert log.index("enter:ac_run_a") < log.index("leave:ac_run_b")
    assert log.index("enter:ac_run_b") < log.index("leave:ac_run_a")

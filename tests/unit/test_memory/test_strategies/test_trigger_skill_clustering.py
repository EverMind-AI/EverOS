"""Tests for :func:`trigger_skill_clustering`.

In-memory cluster store and fake embedder from :mod:`tests.fakes`;
``cluster_by_llm`` and ``mint_cluster_id`` patched on the strategy module.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
import structlog.testing
from everalgo.clustering import Cluster as AlgoCluster

from everos.memory._partition_locks import _reset_for_tests
from everos.memory.events import AgentCaseExtracted, SkillClusterUpdated
from everos.memory.models import Cluster
from everos.memory.ports import Scope
from everos.memory.strategies import (
    TRIGGER_SKILL_CLUSTERING_SPEC,
    trigger_skill_clustering,
)
from tests.fakes import FakeEmbeddingProvider, make_deps

_MODULE = "everos.memory.strategies.trigger_skill_clustering"
_AGENT = Scope(
    app_id="default", project_id="default", owner_id="agent_42", owner_type="agent"
)


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


def test_spec() -> None:
    spec = TRIGGER_SKILL_CLUSTERING_SPEC
    assert spec.name == "trigger_skill_clustering"
    assert AgentCaseExtracted in spec.trigger.on
    assert spec.emits == frozenset({SkillClusterUpdated})
    assert spec.max_retries == 2


async def test_skips_when_quality_score_below_threshold() -> None:
    """quality_score < 0.2 → log + early return; no embedding, no LLM, no store."""
    deps = make_deps()
    with (
        patch(f"{_MODULE}.cluster_by_llm") as mock_cluster,
        structlog.testing.capture_logs() as captured,
    ):
        events = await trigger_skill_clustering(_event(quality_score=0.1), deps)

    assert events == []
    assert deps.embedding.calls == []
    assert deps.stores.cluster.upserts == []
    mock_cluster.assert_not_called()
    assert any(
        e.get("event") == "skill_clustering_skipped_low_quality" for e in captured
    )


async def test_creates_new_cluster_when_no_existing() -> None:
    """Empty existing list → cluster_by_llm returns None → new cluster persisted."""
    deps = make_deps()
    with (
        patch(
            f"{_MODULE}.cluster_by_llm", new=AsyncMock(return_value=None)
        ) as mock_cluster,
        patch(f"{_MODULE}.mint_cluster_id", return_value="cl_newxxxx0001"),
    ):
        events = await trigger_skill_clustering(_event(), deps)

    # cluster_by_llm called with the size-1 new cluster + empty existing,
    # and the strategy's LLM.
    (new_cluster, existing), kwargs = mock_cluster.call_args
    assert kwargs == {"llm": deps.llm}
    assert isinstance(new_cluster, AlgoCluster)
    assert new_cluster.id == "cl_newxxxx0001"
    assert new_cluster.count == 1
    assert new_cluster.last_ts == 1_700_000_001_000
    assert new_cluster.members == ["ac_20260517_0001"]
    assert new_cluster.preview == ["summarise the doc"]
    expected_vector = await FakeEmbeddingProvider().embed("summarise the doc")
    np.testing.assert_allclose(
        np.asarray(new_cluster.centroid), np.asarray(expected_vector, np.float32)
    )
    assert existing == []

    clusters = deps.stores.cluster
    assert list(clusters.clusters) == ["cl_newxxxx0001"]
    assert clusters.owners["cl_newxxxx0001"] == (_AGENT, "agent_case")
    assert clusters.member_types["cl_newxxxx0001"] == {"ac_20260517_0001": "case"}

    (announced,) = events
    assert isinstance(announced, SkillClusterUpdated)
    assert announced.cluster_id == "cl_newxxxx0001"
    assert announced.case_entry_id == "ac_20260517_0001"
    assert announced.agent_id == "agent_42"


async def test_event_passes_through_case_body_and_vector() -> None:
    """The returned event carries the case body verbatim and the embedding just
    computed, so extract_agent_skill doesn't need to re-embed for top-k.
    """
    incoming = _event(
        task_intent="restore replica",
        approach="stop, resync, verify",
        key_insight="watch oplog",
        quality_score=0.8,
        case_timestamp_ms=1_700_000_000_000,
        agent_id="a1",
        case_entry_id="c1",
    )
    with patch(f"{_MODULE}.cluster_by_llm", new=AsyncMock(return_value=None)):
        (ev,) = await trigger_skill_clustering(incoming, make_deps())

    assert ev.task_intent == "restore replica"
    assert ev.approach == "stop, resync, verify"
    assert ev.key_insight == "watch oplog"
    assert ev.quality_score == 0.8
    assert ev.case_timestamp_ms == 1_700_000_000_000
    assert ev.case_vector == await FakeEmbeddingProvider().embed("restore replica")


async def test_merges_into_existing_cluster_when_algo_matches() -> None:
    """algo returns a merged Cluster → persisted with the existing id."""
    deps = make_deps()
    deps.stores.cluster.seed(
        _AGENT,
        "agent_case",
        Cluster(
            id="cl_existing0001",
            centroid=(0.15,) * 8,
            count=2,
            last_ts=1_700_000_000_000,
            preview=["earlier intent"],
            members=["ac_20260517_0000"],
        ),
    )
    # Simulate everalgo _merge: id passes through from existing, members appended.
    merged_cluster = AlgoCluster(
        id="cl_existing0001",
        centroid=np.array([0.17] * 8, dtype=np.float32),
        count=3,
        last_ts=1_700_000_001_000,
        preview=["earlier intent", "summarise the doc"],
        members=["ac_20260517_0000", "ac_20260517_0001"],
    )
    with patch(f"{_MODULE}.cluster_by_llm", new=AsyncMock(return_value=merged_cluster)):
        (announced,) = await trigger_skill_clustering(_event(), deps)

    persisted = deps.stores.cluster.clusters["cl_existing0001"]
    assert persisted.members == ["ac_20260517_0000", "ac_20260517_0001"]
    assert persisted.count == 3
    assert announced.cluster_id == "cl_existing0001"


# ── body-guard (embedding provider) ──────────────────────────────────────


async def test_returns_without_side_effects_when_embedding_unavailable() -> None:
    """No embedding provider → early return; no store access, no LLM, no event."""
    deps = make_deps(embedding=None)
    with (
        patch(
            f"{_MODULE}.cluster_by_llm",
            new=AsyncMock(
                side_effect=AssertionError("cluster_by_llm must not be called")
            ),
        ),
        structlog.testing.capture_logs() as captured,
    ):
        events = await trigger_skill_clustering(_event(), deps)

    assert events == []
    assert deps.stores.cluster.upserts == []
    gated = [
        e
        for e in captured
        if e.get("event") == "strategy_gated_off_embedding_unavailable"
    ]
    assert len(gated) == 1
    assert gated[0]["strategy_name"] == "trigger_skill_clustering"
    assert gated[0]["agent_id"] == "agent_42"


# ── partition lock (agent_id-level serialisation) ────────────────────────


async def _run_serialisation_probe(agent_a: str, agent_b: str) -> list[str]:
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

    deps = make_deps()
    with patch(f"{_MODULE}.cluster_by_llm", new=mock_cluster_by_llm):
        await asyncio.gather(
            trigger_skill_clustering(
                _event(agent_id=agent_a, case_entry_id="ac_run_a"), deps
            ),
            trigger_skill_clustering(
                _event(agent_id=agent_b, case_entry_id="ac_run_b"), deps
            ),
        )
    return log


async def test_partition_lock_serialises_runs_on_same_agent() -> None:
    """Two runs sharing ``agent_id`` must not overlap critical sections."""
    log = await _run_serialisation_probe("agent_42", "agent_42")
    assert log in (
        ["enter:ac_run_a", "leave:ac_run_a", "enter:ac_run_b", "leave:ac_run_b"],
        ["enter:ac_run_b", "leave:ac_run_b", "enter:ac_run_a", "leave:ac_run_a"],
    )


async def test_partition_lock_lets_different_agents_run_in_parallel() -> None:
    """Runs on distinct ``agent_id`` must overlap (no false serialisation)."""
    log = await _run_serialisation_probe("agent_42", "agent_43")
    assert log.index("enter:ac_run_a") < log.index("leave:ac_run_b")
    assert log.index("enter:ac_run_b") < log.index("leave:ac_run_a")

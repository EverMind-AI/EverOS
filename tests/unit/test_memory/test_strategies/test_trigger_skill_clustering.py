"""Tests for :func:`trigger_skill_clustering`.

In-memory cluster store and fake embedder from :mod:`tests.fakes`; the real
``acluster_by_llm`` answers from the canned agent-track scripts. Only the
lock probes patch it, to hold the critical section open.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import numpy as np
import pytest
import structlog.testing

from everos.memory._partition_locks import _reset_for_tests
from everos.memory.deps import MemoryDeps
from everos.memory.events import AgentCaseExtracted, SkillClusterUpdated
from everos.memory.models import Cluster
from everos.memory.ports import Scope
from everos.memory.strategies import (
    TRIGGER_SKILL_CLUSTERING_SPEC,
    trigger_skill_clustering,
)
from tests.fakes import FakeEmbeddingProvider, agent_track_llm, make_deps

_MODULE = "everos.memory.strategies.trigger_skill_clustering"
_AGENT = Scope(
    app_id="default", project_id="default", owner_id="agent_42", owner_type="agent"
)
_ADMITTED = {
    "policy": "decision_case_v1",
    "method": "decision_api",
    "input_tokens": 1_200,
    "bypass_token_threshold": 50_000,
    "success_threshold": 0.7,
    "success_probability": 0.9,
}


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
    case_gate: dict | None = None,
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
        case_gate=case_gate,
    )


def test_spec() -> None:
    spec = TRIGGER_SKILL_CLUSTERING_SPEC
    assert spec.name == "trigger_skill_clustering"
    assert AgentCaseExtracted in spec.trigger.on
    assert spec.emits == frozenset({SkillClusterUpdated})
    assert spec.max_retries == 2


async def test_skips_when_quality_score_below_threshold() -> None:
    """quality_score < 0.2 → log + early return; no embedding, no LLM, no store."""
    deps = make_deps()  # an LLM that fails the test when called
    with structlog.testing.capture_logs() as captured:
        events = await trigger_skill_clustering(_event(quality_score=0.1), deps)

    assert events == []
    assert deps.embedding.calls == []
    assert deps.stores.cluster.upserts == []
    assert any(
        e.get("event") == "skill_clustering_skipped_low_quality" for e in captured
    )


async def test_clusters_an_admitted_case_whatever_its_quality() -> None:
    """A gate-admitted case drives a skill at any quality; the decision rides on."""
    deps = make_deps()
    (announced,) = await trigger_skill_clustering(
        _event(quality_score=0.1, case_gate=_ADMITTED), deps
    )
    assert announced.case_gate == _ADMITTED
    assert list(deps.stores.cluster.clusters) == [announced.cluster_id]


async def test_creates_new_cluster_when_no_existing() -> None:
    """No existing cluster → the size-1 cluster is persisted under the minted id."""
    deps = make_deps()  # no existing cluster, so the model is never asked
    with patch(f"{_MODULE}.mint_cluster_id", return_value="cl_newxxxx0001"):
        events = await trigger_skill_clustering(_event(), deps)

    persisted = deps.stores.cluster.clusters["cl_newxxxx0001"]
    assert persisted.count == 1
    assert persisted.last_ts == 1_700_000_001_000
    assert persisted.members == ["ac_20260517_0001"]
    assert persisted.preview == ["summarise the doc"]
    expected_vector = await FakeEmbeddingProvider().embed("summarise the doc")
    np.testing.assert_allclose(
        np.asarray(persisted.centroid), np.asarray(expected_vector, np.float32)
    )

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
    (ev,) = await trigger_skill_clustering(incoming, make_deps())

    assert ev.task_intent == "restore replica"
    assert ev.approach == "stop, resync, verify"
    assert ev.key_insight == "watch oplog"
    assert ev.quality_score == 0.8
    assert ev.case_timestamp_ms == 1_700_000_000_000
    assert ev.case_vector == await FakeEmbeddingProvider().embed("restore replica")
    assert ev.case_gate is None


def _seed_dissimilar_cluster(deps: MemoryDeps) -> None:
    """One existing cluster far enough from any case that the model is asked."""
    deps.stores.cluster.seed(
        _AGENT,
        "agent_case",
        Cluster(
            id="cl_existing0001",
            centroid=(-1.0,) * 8,
            count=2,
            last_ts=1_700_000_000_000,
            preview=["earlier intent"],
            members=["ac_20260517_0000"],
        ),
    )


async def test_merges_into_existing_cluster_when_the_model_picks_it() -> None:
    """The model picks the existing cluster → persisted under its id."""
    llm = agent_track_llm()
    deps = make_deps(llm=llm)
    _seed_dissimilar_cluster(deps)

    (announced,) = await trigger_skill_clustering(_event(), deps)

    assert [c.response_format.__name__ for c in llm.calls] == [
        "_ClusterAssignmentReply"
    ]
    persisted = deps.stores.cluster.clusters["cl_existing0001"]
    assert persisted.members == ["ac_20260517_0000", "ac_20260517_0001"]
    assert persisted.count == 3
    assert announced.cluster_id == "cl_existing0001"


async def test_keeps_a_new_cluster_when_the_model_picks_none() -> None:
    deps = make_deps(llm=agent_track_llm(skill_cluster="skill_cluster_empty"))
    _seed_dissimilar_cluster(deps)
    with patch(f"{_MODULE}.mint_cluster_id", return_value="cl_newxxxx0001"):
        (announced,) = await trigger_skill_clustering(_event(), deps)

    assert announced.cluster_id == "cl_newxxxx0001"
    assert sorted(deps.stores.cluster.clusters) == ["cl_existing0001", "cl_newxxxx0001"]


# ── body-guard (embedding provider) ──────────────────────────────────────


async def test_returns_without_side_effects_when_embedding_unavailable() -> None:
    """No embedding provider → early return; no store access, no LLM, no event."""
    deps = make_deps(embedding=None)  # an LLM that fails the test when called
    with structlog.testing.capture_logs() as captured:
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

    async def mock_acluster_by_llm(new_cluster, _existing, **_kwargs):
        log.append(f"enter:{new_cluster.members[0]}")
        await asyncio.sleep(0.01)
        log.append(f"leave:{new_cluster.members[0]}")
        return None  # no merge → caller persists the size-1 cluster

    deps = make_deps()
    with patch(f"{_MODULE}.acluster_by_llm", new=mock_acluster_by_llm):
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

"""Tests for :func:`extract_agent_case`.

The real :class:`AgentCaseExtractor` runs against the canned agent-track
scripts (``tests.fakes.agent_track_llm``); stores are in-memory doubles. The
strategy files one case per assistant sender and logs the gate that rejected
a memcell when there is no case. With a case judge in the deps, everalgo's
correctness gate admits the memcell and its decision rides on the case.
"""

from __future__ import annotations

import json

import structlog.testing
from everalgo.testing import FakeLLMClient
from everalgo.types import ChatMessage, MemCell

from everos.component.llm import LLMCaseJudge
from everos.memory.events import AgentCaseExtracted, AgentPipelineStarted
from everos.memory.strategies import EXTRACT_AGENT_CASE_SPEC, extract_agent_case
from tests.fakes import (
    agent_track_llm,
    agent_track_script,
    agent_trajectory,
    make_deps,
)

_COMPRESSED = json.loads(agent_track_script("case_compress"))


def _event(memcell: MemCell | None = None) -> AgentPipelineStarted:
    return AgentPipelineStarted(
        memcell_id="mc_a", session_id="s1", memcell=memcell or agent_trajectory()
    )


def _replies(llm: FakeLLMClient) -> list[str]:
    return [call.response_format.__name__ for call in llm.calls]


def test_spec() -> None:
    spec = EXTRACT_AGENT_CASE_SPEC
    assert spec.name == "extract_agent_case"
    assert AgentPipelineStarted in spec.trigger.on
    assert spec.emits == frozenset({AgentCaseExtracted})
    assert spec.max_retries == 2


async def test_writes_the_case_and_announces_it() -> None:
    deps = make_deps(llm=agent_track_llm())
    with structlog.testing.capture_logs() as captured:
        events = await extract_agent_case(_event(), deps)

    (stored,) = deps.stores.agent_case.records
    assert (stored.scope.owner_id, stored.scope.owner_type) == ("agent_42", "agent")
    assert (stored.item.session_id, stored.item.parent_id) == ("s1", "mc_a")
    assert stored.item.task_intent == _COMPRESSED["task_intent"]
    assert stored.item.approach == _COMPRESSED["approach"]
    assert stored.item.key_insight == _COMPRESSED["key_insight"]
    assert stored.item.quality_score == _COMPRESSED["quality_score"]
    assert stored.item.case_gate is None

    (announced,) = events
    assert isinstance(announced, AgentCaseExtracted)
    assert announced.memcell_id == "mc_a"
    assert announced.case_entry_id == stored.ref.entry_id
    assert announced.task_intent == _COMPRESSED["task_intent"]
    assert announced.approach == _COMPRESSED["approach"]
    assert announced.key_insight == _COMPRESSED["key_insight"]
    assert announced.quality_score == _COMPRESSED["quality_score"]
    assert announced.case_timestamp_ms == agent_trajectory().timestamp
    assert announced.agent_id == "agent_42"
    assert announced.case_gate is None

    record = next(e for e in captured if e.get("event") == "agent_case_extracted")
    assert record["owner_ids"] == ["agent_42"]
    assert record["fanout"] == 1


async def test_fans_out_per_assistant_sender() -> None:
    """One extraction, then a write + an event per distinct assistant sender.

    Case text is third-person (``the agent did X``) so the same body is a
    valid reference experience for every assistant sender that took part.
    """
    trajectory = agent_trajectory(agent_id="agent_lead")
    *steps, answer = trajectory.items
    handed_over = trajectory.model_copy(
        update={"items": [*steps, answer.model_copy(update={"sender_id": "agent_b"})]}
    )
    llm = agent_track_llm()
    deps = make_deps(llm=llm)

    events = await extract_agent_case(_event(handed_over), deps)

    assert _replies(llm) == ["_CaseFilterReply", "_CaseCompressReply"]
    stored = deps.stores.agent_case.records
    assert [s.scope.owner_id for s in stored] == ["agent_lead", "agent_b"]
    assert [e.agent_id for e in events] == ["agent_lead", "agent_b"]
    assert [e.case_entry_id for e in events] == [s.ref.entry_id for s in stored]
    assert {e.task_intent for e in events} == {_COMPRESSED["task_intent"]}


async def test_logs_the_reason_when_the_filter_rejects_the_trajectory() -> None:
    deps = make_deps(llm=agent_track_llm(case_filter="case_filter_empty"))
    with structlog.testing.capture_logs() as captured:
        events = await extract_agent_case(_event(), deps)

    assert events == []
    assert deps.stores.agent_case.records == []
    (skipped,) = [e for e in captured if e.get("event") == "agent_case_skipped"]
    assert skipped["reason"] == "filter_rejected"
    want = json.loads(agent_track_script("case_filter_empty"))["reason"]
    assert skipped["detail"] == {"llm_reason": want}


async def test_logs_the_reason_when_algo_rejects_before_any_model_call() -> None:
    """A tool-free single-turn exchange is rejected structurally."""
    chat = MemCell(
        items=[
            ChatMessage(
                id="m1", role="user", content="hi", timestamp=1, sender_id="u_alice"
            ),
            ChatMessage(
                id="m2", role="assistant", content="hello", timestamp=2, sender_id="a"
            ),
        ],
        timestamp=2,
    )
    deps = make_deps()  # an LLM that fails the test when called
    with structlog.testing.capture_logs() as captured:
        events = await extract_agent_case(_event(chat), deps)

    assert events == []
    (skipped,) = [e for e in captured if e.get("event") == "agent_case_skipped"]
    assert skipped["reason"] == "no_tool_single_user"


async def test_skips_when_no_assistant_sender() -> None:
    """No assistant in the cell → no agent_id can be inferred; algo not called."""
    user_only = MemCell(
        items=[
            ChatMessage(
                id="m1",
                role="user",
                content="hi",
                timestamp=1_700_000_000_000,
                sender_id="u_alice",
            ),
        ],
        timestamp=1_700_000_000_000,
    )
    deps = make_deps()
    with structlog.testing.capture_logs() as captured:
        events = await extract_agent_case(_event(user_only), deps)

    assert events == []
    assert deps.stores.agent_case.records == []
    assert any(e.get("event") == "agent_case_skipped_no_assistant" for e in captured)


# ── correctness gate ([agent_memory] case_gate_enabled) ──────────────────


async def test_a_gate_admitted_case_carries_the_decision() -> None:
    """With a judge the gate replaces the filter; the decision rides along."""
    llm = agent_track_llm()
    deps = make_deps(llm=llm, case_judge=LLMCaseJudge(llm))

    (announced,) = await extract_agent_case(_event(), deps)

    assert _replies(llm) == ["_SuccessProbability", "_CaseCompressReply"]
    (stored,) = deps.stores.agent_case.records
    assert stored.item.case_gate is not None
    assert stored.item.case_gate["method"] == "decision_api"
    assert stored.item.case_gate["success_probability"] == 0.9
    assert announced.case_gate == stored.item.case_gate


async def test_logs_the_reason_when_the_gate_rejects_the_trajectory() -> None:
    llm = agent_track_llm(case_gate="case_gate_empty")
    deps = make_deps(llm=llm, case_judge=LLMCaseJudge(llm))
    with structlog.testing.capture_logs() as captured:
        events = await extract_agent_case(_event(), deps)

    assert events == []
    assert _replies(llm) == ["_SuccessProbability"]
    (skipped,) = [e for e in captured if e.get("event") == "agent_case_skipped"]
    assert skipped["reason"] == "success_gate_rejected"
    assert skipped["detail"]["case_gate"]["success_probability"] == 0.1

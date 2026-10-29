"""Conversions between everalgo ``AgentCase`` and the domain case."""

from __future__ import annotations

from everalgo.types import AgentCase as AlgoAgentCase

from everos.memory.mappers import agent_case

_GATE = {
    "policy": "decision_case_v1",
    "method": "length_bypass",
    "input_tokens": 60_000,
    "bypass_token_threshold": 50_000,
    "success_threshold": 0.7,
    "success_probability": None,
}


def _algo(**overrides: object) -> AlgoAgentCase:
    fields: dict[str, object] = {
        "id": "algo_uuid",
        "timestamp": 1_700_000_000_000,
        "task_intent": "summarise doc",
        "approach": "read + condense",
        "quality_score": 0.75,
        "key_insight": "batch-then-summarise",
    }
    return AlgoAgentCase(**{**fields, **overrides})


def test_from_algo_takes_the_context_from_the_caller() -> None:
    case = agent_case.from_algo(
        _algo(), owner_id="agent_42", session_id="s1", parent_id="mc_a"
    )
    assert (case.owner_id, case.session_id, case.parent_id) == (
        "agent_42",
        "s1",
        "mc_a",
    )
    assert (case.task_intent, case.approach) == ("summarise doc", "read + condense")
    assert (case.quality_score, case.key_insight) == (0.75, "batch-then-summarise")
    assert case.case_gate is None
    assert "id" not in case.model_dump()


def test_from_algo_stores_an_empty_key_insight_as_absent() -> None:
    case = agent_case.from_algo(
        _algo(key_insight=""), owner_id="a", session_id="s", parent_id="p"
    )
    assert case.key_insight is None


def test_from_algo_keeps_the_gate_decision_and_drops_other_extras() -> None:
    case = agent_case.from_algo(
        _algo(case_gate=_GATE, vector=[0.1]),
        owner_id="a",
        session_id="s",
        parent_id="p",
    )
    assert case.case_gate == _GATE
    assert "vector" not in case.model_dump()


def test_to_algo_addresses_the_case_by_its_entry_id() -> None:
    case = agent_case.from_algo(
        _algo(key_insight=""), owner_id="a", session_id="s", parent_id="p"
    )
    algo = agent_case.to_algo(case, entry_id="ac_1")
    assert algo == _algo(id="ac_1", key_insight="")
    assert not hasattr(algo, "case_gate")


def test_to_algo_carries_the_gate_decision() -> None:
    case = agent_case.from_algo(
        _algo(case_gate=_GATE), owner_id="a", session_id="s", parent_id="p"
    )
    assert agent_case.to_algo(case, entry_id="ac_1").case_gate == _GATE

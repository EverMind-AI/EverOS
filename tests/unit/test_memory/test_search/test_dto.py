"""Unit tests for ``memory.search.dto`` validation rules."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from everos.memory.search import (
    SearchData,
    SearchMethod,
    SearchRequest,
    SearchResponse,
)


def _minimal_request_kwargs() -> dict:
    return {
        "user_id": "alice",
        "query": "hello",
    }


def test_enable_llm_rerank_defaults_to_false() -> None:
    """HYBRID should NOT auto-trigger LLM Phase-5 rerank by default.

    The caller opts in explicitly when they want the extra LLM pass;
    leaving it off keeps a default HYBRID call cheap (no LLM ``chat``).
    """
    req = SearchRequest(**_minimal_request_kwargs())
    assert req.enable_llm_rerank is False


def test_enable_llm_rerank_accepts_true() -> None:
    req = SearchRequest(**_minimal_request_kwargs(), enable_llm_rerank=True)
    assert req.enable_llm_rerank is True


def test_minimal_request_uses_hybrid_default() -> None:
    req = SearchRequest(**_minimal_request_kwargs())
    assert req.method == SearchMethod.HYBRID
    assert req.top_k == -1
    assert req.include_profile is False
    assert req.filters is None
    assert req.radius is None
    assert req.min_score is None


def test_top_k_zero_rejected() -> None:
    with pytest.raises(ValidationError) as exc:
        SearchRequest(**_minimal_request_kwargs(), top_k=0)
    assert "top_k" in str(exc.value)


def test_top_k_above_100_rejected() -> None:
    with pytest.raises(ValidationError):
        SearchRequest(**_minimal_request_kwargs(), top_k=101)


def test_top_k_below_minus_one_rejected() -> None:
    with pytest.raises(ValidationError):
        SearchRequest(**_minimal_request_kwargs(), top_k=-2)


def test_top_k_minus_one_accepted() -> None:
    req = SearchRequest(**_minimal_request_kwargs(), top_k=-1)
    assert req.top_k == -1


def test_top_k_in_range_accepted() -> None:
    req = SearchRequest(**_minimal_request_kwargs(), top_k=50)
    assert req.top_k == 50


def test_radius_out_of_range_rejected() -> None:
    with pytest.raises(ValidationError):
        SearchRequest(**_minimal_request_kwargs(), radius=1.5)
    with pytest.raises(ValidationError):
        SearchRequest(**_minimal_request_kwargs(), radius=-0.1)


def test_min_score_out_of_range_rejected() -> None:
    with pytest.raises(ValidationError):
        SearchRequest(**_minimal_request_kwargs(), min_score=1.5)
    with pytest.raises(ValidationError):
        SearchRequest(**_minimal_request_kwargs(), min_score=-0.1)


def test_min_score_in_range_accepted() -> None:
    req = SearchRequest(**_minimal_request_kwargs(), min_score=0.4)
    assert req.min_score == 0.4


def test_neither_user_id_nor_agent_id_rejected() -> None:
    """The xor validator requires exactly one of user_id / agent_id."""
    with pytest.raises(ValidationError, match="exactly one of"):
        SearchRequest(query="hello")  # neither set


def test_both_user_id_and_agent_id_rejected() -> None:
    """The xor validator rejects ambiguous owner identity."""
    with pytest.raises(ValidationError, match="exactly one of"):
        SearchRequest(user_id="alice", agent_id="agent_x", query="hello")


def test_empty_query_rejected() -> None:
    with pytest.raises(ValidationError):
        SearchRequest(user_id="alice", query="")


def test_empty_user_id_rejected() -> None:
    with pytest.raises(ValidationError):
        SearchRequest(user_id="", query="hello")


def test_extra_top_level_field_rejected() -> None:
    """``extra='forbid'`` keeps the contract tight."""
    with pytest.raises(ValidationError):
        SearchRequest(
            **_minimal_request_kwargs(),
            unexpected_field="x",  # type: ignore[call-arg]
        )


def test_filters_extra_keys_allowed() -> None:
    """FilterNode is open-shape; safety is enforced in the compiler."""
    req = SearchRequest(
        **_minimal_request_kwargs(),
        filters={"session_id": "sess_a", "AND": [{"timestamp": {"gte": 1}}]},
    )
    assert req.filters is not None
    dumped = req.filters.model_dump(exclude_none=True)
    assert dumped["session_id"] == "sess_a"
    assert dumped["AND"][0]["timestamp"]["gte"] == 1


def test_response_default_arrays_present() -> None:
    """Every ``data.*`` array must exist so callers can iterate unconditionally."""
    resp = SearchResponse(request_id="0" * 32, data=SearchData())
    assert resp.data.episodes == []
    assert resp.data.profiles == []
    assert resp.data.agent_cases == []
    assert resp.data.agent_skills == []


def test_method_enum_serialises_to_lowercase() -> None:
    req = SearchRequest(**_minimal_request_kwargs(), method="agentic")  # type: ignore[arg-type]
    assert req.method == SearchMethod.AGENTIC
    assert req.method.value == "agentic"

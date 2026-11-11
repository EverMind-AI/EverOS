"""Unit tests for ``memory.search.dto`` validation rules.

Multi-round retrieval is the only search method: another method, an agent
scope and the removed methods' options are refused at validation.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from everos.memory.search import (
    SearchData,
    SearchMethod,
    SearchRequest,
    SearchResponse,
)
from everos.memory.search.dto import MULTIROUND_ONLY


def _minimal_request_kwargs() -> dict:
    return {
        "user_id": "alice",
        "query": "hello",
    }


def test_minimal_request_uses_the_multiround_default() -> None:
    req = SearchRequest(**_minimal_request_kwargs())
    assert req.method == SearchMethod.LLM_MULTIROUND
    assert req.top_k == -1
    assert req.include_profile is False
    assert req.filters is None


@pytest.mark.parametrize("method", ["keyword", "vector", "hybrid", "agentic", "x"])
def test_a_removed_method_is_refused_with_the_only_supported_one(method: str) -> None:
    """Multi-round retrieval is the only search method: naming another one is
    a client error that says which method is supported."""
    with pytest.raises(ValidationError) as exc:
        SearchRequest(**_minimal_request_kwargs(), method=method)  # type: ignore[arg-type]
    text = str(exc.value)
    assert MULTIROUND_ONLY in text and repr(method) in text


def test_the_multiround_method_is_accepted_by_name() -> None:
    req = SearchRequest(**_minimal_request_kwargs(), method="llm_multiround")  # type: ignore[arg-type]
    assert req.method is SearchMethod.LLM_MULTIROUND
    assert req.method.value == "llm_multiround"


@pytest.mark.parametrize(
    "fields",
    [{"agent_id": "agent_x"}, {"agent_id": "agent_x", "context_group_id": "g"}],
)
def test_an_agent_scope_is_refused(fields: dict) -> None:
    """Multi-round retrieval covers user and group memory only."""
    with pytest.raises(ValidationError, match="agent memory is not searchable"):
        SearchRequest(query="hello", **fields)


@pytest.mark.parametrize(
    "field", [{"radius": 0.5}, {"min_score": 0.4}, {"enable_llm_rerank": True}]
)
def test_the_removed_methods_options_are_unknown_fields(field: dict) -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        SearchRequest(**_minimal_request_kwargs(), **field)


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


def test_no_owner_scope_rejected() -> None:
    """Exactly one of user_id / group_id / group_ids is required."""
    with pytest.raises(ValidationError, match="exactly one of"):
        SearchRequest(query="hello")  # neither set


def test_both_user_id_and_group_id_rejected() -> None:
    """The one-scope validator rejects ambiguous owner identity."""
    with pytest.raises(ValidationError, match="exactly one of"):
        SearchRequest(user_id="alice", group_id="g", query="hello")


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
    assert resp.data.images == []
    assert resp.data.profiles == []
    assert resp.data.unprocessed_messages == []
    assert "agent_cases" not in resp.data.model_dump()

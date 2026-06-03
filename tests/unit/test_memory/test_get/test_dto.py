"""Tests for ``memory.get.dto``.

Pydantic-side guarantees the manager / route can rely on:

* ``GetRequest`` defaults match the wiki spec (``page=1`` /
  ``page_size=20`` / ``sort_by="timestamp"`` / ``sort_order="desc"``)
* ``page_size`` upper bound (1–100)
* ``owner_type`` × ``memory_type`` strict pairing
* Unknown fields on the request are rejected (``extra="forbid"``)

Filter DSL coverage lives in ``test_memory/test_search/test_filters.py``
since ``/get`` shares :class:`everos.memory.search.FilterNode`.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from everos.memory.get.dto import (
    GetMemoryType,
    GetRequest,
)

# ── GetRequest defaults / shape ──────────────────────────────────────────


def test_get_request_defaults_match_wiki() -> None:
    """``page`` / ``page_size`` / ``sort_by`` / ``sort_order`` come from the wiki."""
    req = GetRequest(
        user_id="u1",
        memory_type=GetMemoryType.EPISODE,
    )
    assert req.page == 1
    assert req.page_size == 20
    assert req.sort_by == "timestamp"
    assert req.sort_order == "desc"
    assert req.filters is None


def test_get_request_page_size_upper_bound() -> None:
    """101 → ValidationError (wiki cap is 100)."""
    with pytest.raises(ValidationError):
        GetRequest(
            user_id="u1",
            memory_type=GetMemoryType.EPISODE,
            page_size=101,
        )


def test_get_request_page_size_lower_bound() -> None:
    """0 → ValidationError (page_size ≥ 1)."""
    with pytest.raises(ValidationError):
        GetRequest(
            user_id="u1",
            memory_type=GetMemoryType.EPISODE,
            page_size=0,
        )


def test_get_request_page_lower_bound() -> None:
    """0 → ValidationError (page ≥ 1; 1-indexed)."""
    with pytest.raises(ValidationError):
        GetRequest(
            user_id="u1",
            memory_type=GetMemoryType.EPISODE,
            page=0,
        )


def test_get_request_rejects_unknown_field() -> None:
    """``extra='forbid'`` — typos surface as a 422, not silent drops."""
    with pytest.raises(ValidationError):
        GetRequest(
            user_id="u1",
            memory_type=GetMemoryType.EPISODE,
            unknown_extra=True,  # type: ignore[call-arg]
        )


def test_get_request_rejects_empty_user_id() -> None:
    """``user_id`` carries ``min_length=1`` — empty string is 422."""
    with pytest.raises(ValidationError):
        GetRequest(
            user_id="",
            memory_type=GetMemoryType.EPISODE,
        )


def test_get_request_rejects_missing_memory_type() -> None:
    """``memory_type`` is required — omission is 422."""
    with pytest.raises(ValidationError):
        GetRequest(  # type: ignore[call-arg]
            user_id="u1",
        )


def test_get_request_rejects_missing_owner_identity() -> None:
    """Neither ``user_id`` nor ``agent_id`` → xor validator rejects."""
    with pytest.raises(ValidationError, match="exactly one of"):
        GetRequest(  # type: ignore[call-arg]
            memory_type=GetMemoryType.EPISODE,
        )


def test_get_request_rejects_both_user_and_agent_id() -> None:
    """Both ``user_id`` and ``agent_id`` set → xor validator rejects."""
    with pytest.raises(ValidationError, match="exactly one of"):
        GetRequest(
            user_id="u1",
            agent_id="agent_x",
            memory_type=GetMemoryType.EPISODE,
        )


def test_get_request_rejects_invalid_memory_type_value() -> None:
    """A value outside the four-kind enum is 422."""
    with pytest.raises(ValidationError):
        GetRequest.model_validate(
            {
                "user_id": "u1",
                "memory_type": "atomic_fact",  # not a top-level kind
            }
        )


def test_get_request_rejects_invalid_sort_order() -> None:
    """``sort_order`` is a tight Literal — typos / casing variants are 422."""
    with pytest.raises(ValidationError):
        GetRequest.model_validate(
            {
                "user_id": "u1",
                "memory_type": "episode",
                "sort_order": "DESC",  # must be lowercase
            }
        )


# ── owner_type × memory_type pairing ─────────────────────────────────────


@pytest.mark.parametrize(
    "id_field, memory_type",
    [
        ("user_id", GetMemoryType.EPISODE),
        ("user_id", GetMemoryType.PROFILE),
        ("agent_id", GetMemoryType.AGENT_CASE),
        ("agent_id", GetMemoryType.AGENT_SKILL),
    ],
)
def test_get_request_allows_valid_owner_memory_pair(
    id_field: str,
    memory_type: GetMemoryType,
) -> None:
    """The four valid (owner-kind, memory_type) combinations."""
    req = GetRequest(**{id_field: "u1"}, memory_type=memory_type)
    assert req.memory_type is memory_type
    expected_owner_type = "user" if id_field == "user_id" else "agent"
    assert req.owner_type == expected_owner_type


@pytest.mark.parametrize(
    "id_field, memory_type",
    [
        ("user_id", GetMemoryType.AGENT_CASE),
        ("user_id", GetMemoryType.AGENT_SKILL),
        ("agent_id", GetMemoryType.EPISODE),
        ("agent_id", GetMemoryType.PROFILE),
    ],
)
def test_get_request_rejects_cross_owner_memory_pair(
    id_field: str,
    memory_type: GetMemoryType,
) -> None:
    """Cross-pairs (user_id+agent_case etc.) are 422 at the DTO layer."""
    with pytest.raises(ValidationError):
        GetRequest(**{id_field: "u1"}, memory_type=memory_type)

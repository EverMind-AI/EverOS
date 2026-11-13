"""Tests for ``memory.get.dto``.

Pydantic-side guarantees the manager / route can rely on:

* ``GetRequest`` defaults match the wiki spec (``page=1`` /
  ``page_size=20`` / ``sort_by="timestamp"`` / ``sort_order="desc"``)
* ``page_size`` upper bound (1–100)
* ``owner_type`` × ``memory_type`` strict pairing
* An agent scope is refused (EverOS 2.0 has no agent memory)
* Unknown fields on the request are rejected (``extra="forbid"``)

Filter DSL coverage lives in ``test_memory/test_search/test_filters.py``
since ``/get`` shares :class:`everos.memory.search.FilterNode`.
"""

from __future__ import annotations

import re
from typing import Any

import pytest
from pydantic import ValidationError

from everos.memory.get.dto import (
    GetMemoryType,
    GetRequest,
)
from everos.memory.search import NO_AGENT_MEMORY

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
    """No owner scope (``user_id`` / ``group_id`` / ``group_ids``) → rejected."""
    with pytest.raises(ValidationError, match="exactly one of"):
        GetRequest(  # type: ignore[call-arg]
            memory_type=GetMemoryType.EPISODE,
        )


def test_get_request_rejects_two_owner_scopes() -> None:
    """Both ``user_id`` and ``group_id`` set → the one-scope validator rejects."""
    with pytest.raises(ValidationError, match="exactly one of"):
        GetRequest(
            user_id="u1",
            group_id="g1",
            asker_id="u1",
            memory_type=GetMemoryType.EPISODE,
        )


@pytest.mark.parametrize(
    "fields",
    [
        {"agent_id": "agent_x", "memory_type": "episode"},
        {"user_id": "u1", "agent_id": "agent_x", "memory_type": "episode"},
        {"agent_id": "agent_x", "memory_type": "agent_case"},
        {"agent_id": "agent_x", "memory_type": "agent_skill"},
        {"user_id": "u1", "memory_type": "agent_case"},
        {"user_id": "u1", "memory_type": "agent_skill"},
        {
            "agent_id": "agent_x",
            "context_group_id": "g1",
            "asker_id": "u1",
            "memory_type": "agent_case",
        },
        {"user_id": "u1", "context_group_id": "g1", "memory_type": "episode"},
    ],
)
def test_get_request_refuses_an_agent_scope(fields: dict[str, Any]) -> None:
    """``agent_id``, ``context_group_id`` and the agent kinds are refused by name."""
    with pytest.raises(ValidationError, match=re.escape(NO_AGENT_MEMORY)):
        GetRequest.model_validate(fields)


def test_get_request_rejects_invalid_memory_type_value() -> None:
    """A value outside the enum is 422."""
    with pytest.raises(ValidationError):
        GetRequest.model_validate(
            {
                "user_id": "u1",
                "memory_type": "atomic_fact",  # a removed kind, never accepted
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
    "memory_type",
    [
        GetMemoryType.EPISODE,
        GetMemoryType.PROFILE,
        GetMemoryType.SCENE,
        GetMemoryType.IMAGE_MEMORY,
    ],
)
def test_get_request_allows_valid_owner_memory_pair(
    memory_type: GetMemoryType,
) -> None:
    """The user-owned kinds pair with ``user_id``."""
    req = GetRequest(user_id="u1", memory_type=memory_type)
    assert req.memory_type is memory_type
    assert req.owner_type == "user"


@pytest.mark.parametrize(
    "memory_type",
    [
        GetMemoryType.MEMBER_PROFILE,
        GetMemoryType.GROUP_PROFILE,
    ],
)
def test_get_request_rejects_cross_owner_memory_pair(
    memory_type: GetMemoryType,
) -> None:
    """A group kind with ``user_id`` is 422 at the DTO layer."""
    with pytest.raises(ValidationError, match="not valid when user_id is set"):
        GetRequest(user_id="u1", memory_type=memory_type)

"""Public DTOs for ``POST /api/v1/memory/get``.

Contract per the final design (mirrors :mod:`memory.search.dto` shape,
minus ``score`` because /get is a paginated listing rather than a
ranked retrieval):

* ``owner_type`` × ``memory_type`` are strictly paired:

  - ``user`` → ``episode`` | ``profile``
  - ``agent`` → ``agent_case`` | ``agent_skill``

* ``GetData`` always contains four kind arrays for symmetry with
  ``/search``; only the requested kind is populated. ``total_count``
  is the predicate's true match count; ``count`` is the page size
  actually returned.

* ``filters`` reuses :class:`everos.memory.search.FilterNode` —
  same DSL, same compile path, ``AND`` / ``OR`` combinators allowed.
  The earlier ``/get``-only ban on combinators (from wiki appendix C)
  was dropped: the legacy opensource memsys ``/get`` always supported
  combinators and there is no engine-side reason to forbid them.
"""

from __future__ import annotations

import datetime as _dt
from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from everos.memory.search import FilterNode


class GetMemoryType(StrEnum):
    """The four kinds enumerated by ``/get``.

    ``episode`` and ``profile`` are user-owned; ``agent_case`` and
    ``agent_skill`` are agent-owned. Cross-pairs are rejected by
    :meth:`GetRequest._validate_owner_memory_type_pair`.

    Naming note: all four values use the bare kind name (no
    ``_memory`` suffix) and match the LanceDB table name + everalgo
    type name for that kind.
    """

    EPISODE = "episode"
    PROFILE = "profile"
    AGENT_CASE = "agent_case"
    AGENT_SKILL = "agent_skill"


# ── Request ──────────────────────────────────────────────────────────────


class GetRequest(BaseModel):
    """Request body for ``POST /api/v1/memory/get``.

    Callers identify the memory owner via ``user_id`` XOR ``agent_id`` —
    exactly one must be set. Internally the manager keeps using
    ``owner_id`` / ``owner_type`` (the storage tables' columns); those
    are exposed as derived properties so the rename only affects the
    wire contract.
    """

    model_config = ConfigDict(extra="forbid")

    user_id: str | None = Field(default=None, min_length=1)
    agent_id: str | None = Field(default=None, min_length=1)
    """Memory owner — provide ``user_id`` for ``episode`` / ``profile`` or
    ``agent_id`` for ``agent_case`` / ``agent_skill``; exactly one must be set."""
    app_id: str = "default"
    project_id: str = "default"
    """App / project scope (default ``"default"``). Pinned into the query
    ``where`` so a listing never crosses into another space's rows."""
    memory_type: GetMemoryType
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=20, ge=1, le=100)
    sort_by: Literal["timestamp", "updated_at"] = "timestamp"
    """Sort column. ``profile`` and ``agent_skill`` silently override
    to ``updated_at`` (profile has no timestamp; agent_skill is a
    named entity with no temporal column)."""

    sort_order: Literal["asc", "desc"] = "desc"
    filters: FilterNode | None = None
    """Filters DSL — same shape as ``/search``, ``AND`` / ``OR``
    combinators allowed."""

    @model_validator(mode="after")
    def _validate_user_xor_agent(self) -> Self:
        if (self.user_id is None) == (self.agent_id is None):
            raise ValueError("exactly one of user_id / agent_id must be provided")
        return self

    @model_validator(mode="after")
    def _validate_owner_memory_type_pair(self) -> Self:
        # Runs after the xor validator (declaration order), so ``owner_type``
        # is well-defined here.
        user_kinds = {GetMemoryType.EPISODE, GetMemoryType.PROFILE}
        agent_kinds = {GetMemoryType.AGENT_CASE, GetMemoryType.AGENT_SKILL}
        if self.owner_type == "user" and self.memory_type not in user_kinds:
            raise ValueError(
                f"memory_type {self.memory_type.value!r} is not valid "
                "when user_id is set"
            )
        if self.owner_type == "agent" and self.memory_type not in agent_kinds:
            raise ValueError(
                f"memory_type {self.memory_type.value!r} is not valid "
                "when agent_id is set"
            )
        return self

    @property
    def owner_id(self) -> str:
        """Derived from whichever of ``user_id`` / ``agent_id`` is set."""
        return self.user_id or self.agent_id or ""

    @property
    def owner_type(self) -> Literal["user", "agent"]:
        """``"user"`` if ``user_id`` is set, else ``"agent"``."""
        return "user" if self.user_id is not None else "agent"


# ── Item DTOs (mirror Search*Item shapes minus score) ────────────────────


class GetEpisodeItem(BaseModel):
    """Episode listing item — always user-scoped."""

    model_config = ConfigDict(extra="forbid")

    id: str
    user_id: str | None
    app_id: str = "default"
    project_id: str = "default"
    session_id: str
    timestamp: _dt.datetime
    sender_ids: list[str] = Field(default_factory=list)
    summary: str
    subject: str
    episode: str
    type: Literal["Conversation"]


class GetProfileItem(BaseModel):
    """Owner profile — at most one per response, only for user owners."""

    model_config = ConfigDict(extra="forbid")

    id: str
    user_id: str | None
    app_id: str = "default"
    project_id: str = "default"
    profile_data: dict[str, object]


class GetAgentCaseItem(BaseModel):
    """Agent case listing item — always agent-scoped."""

    model_config = ConfigDict(extra="forbid")

    id: str
    agent_id: str
    app_id: str = "default"
    project_id: str = "default"
    session_id: str
    task_intent: str
    approach: str
    quality_score: float
    key_insight: str | None = None
    timestamp: _dt.datetime


class GetAgentSkillItem(BaseModel):
    """Agent skill listing item — always agent-scoped."""

    model_config = ConfigDict(extra="forbid")

    id: str
    agent_id: str
    app_id: str = "default"
    project_id: str = "default"
    name: str
    description: str
    content: str
    confidence: float
    maturity_score: float
    source_case_ids: list[str] = Field(default_factory=list)


# ── Response envelope ────────────────────────────────────────────────────


class GetData(BaseModel):
    """Body of ``response.data``.

    All four arrays are always present so client code can iterate
    without branching on ``memory_type``; the route populates exactly
    one.
    """

    model_config = ConfigDict(extra="forbid")

    episodes: list[GetEpisodeItem] = Field(default_factory=list)
    profiles: list[GetProfileItem] = Field(default_factory=list)
    agent_cases: list[GetAgentCaseItem] = Field(default_factory=list)
    agent_skills: list[GetAgentSkillItem] = Field(default_factory=list)
    total_count: int = 0
    """Total rows matching the request's owner + filter predicate."""

    count: int = 0
    """Number of items in this page (``len(items)`` after slicing)."""


class GetResponse(BaseModel):
    """Top-level response envelope."""

    model_config = ConfigDict(extra="forbid")

    request_id: str
    data: GetData

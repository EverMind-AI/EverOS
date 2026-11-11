"""Public DTOs for ``POST /api/v2/memory/get``.

Contract per the final design (mirrors :mod:`memory.search.dto` shape,
minus ``score`` because /get is a paginated listing rather than a
ranked retrieval):

* ``owner_type`` × ``memory_type`` are strictly paired:

  - ``user`` → ``episode`` | ``profile``
  - ``agent`` → ``agent_case`` | ``agent_skill``
  - ``group`` → ``episode`` | ``scene`` | ``image_memory`` | ``member_profile``
    (the last with ``member_id``)

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

from everos.memory.search import FilterNode, GroupProfileEntry, SearchSource


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
    SCENE = "scene"
    """A group's scene versions (group scopes only)."""
    IMAGE_MEMORY = "image_memory"
    """A group's picture memories (group scopes only)."""
    MEMBER_PROFILE = "member_profile"
    """One member's profile in the group(s), with ``member_id`` (group scopes)."""
    GROUP_PROFILE = "group_profile"
    """The group-profile items the asker can see (group scopes); with
    ``include_superseded`` every version, not only the current ones."""


# ── Request ──────────────────────────────────────────────────────────────


class GetRequest(BaseModel):
    """Request body for ``POST /api/v2/memory/get``.

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
    ``agent_id`` for ``agent_case`` / ``agent_skill``; exactly one owner scope
    must be set (``user_id``, ``agent_id``, ``group_id`` or ``group_ids``)."""
    group_id: str | None = Field(default=None, min_length=1)
    group_ids: list[str] | None = Field(default=None, min_length=1, max_length=16)
    """A group scope (docs/group-memory.md, 5.4): ``episode`` only for now. The
    listing holds what the asker's membership windows contain."""
    asker_id: str | None = Field(default=None, min_length=1)
    """Who asks: required for a group scope (or the identity header); with
    ``user_id`` optional and equal to it."""
    audience: Literal["asker", "group"] = "asker"
    context_group_id: str | None = Field(default=None, min_length=1)
    """With ``agent_id`` only: list the agent's cases in this group's context
    (the cases learned in it join those learned outside groups)."""
    member_id: str | None = Field(default=None, min_length=1)
    """With ``memory_type = member_profile`` only: whose profile to list."""
    include_superseded: bool = False
    """With ``memory_type = group_profile`` only: list the versions later items
    replaced too, each marked ``superseded``. Forgotten items never return."""
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
        scopes = [
            self.user_id is not None,
            self.agent_id is not None,
            self.group_id is not None,
            self.group_ids is not None,
        ]
        if sum(scopes) != 1:
            raise ValueError(
                "exactly one of user_id / agent_id / group_id / group_ids "
                "must be provided"
            )
        return self

    @model_validator(mode="after")
    def _validate_context_group(self) -> Self:
        if self.context_group_id is not None and self.agent_id is None:
            raise ValueError("context_group_id is valid with agent_id only")
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
        group_kinds = {
            GetMemoryType.EPISODE,
            GetMemoryType.SCENE,
            GetMemoryType.IMAGE_MEMORY,
            GetMemoryType.MEMBER_PROFILE,
            GetMemoryType.GROUP_PROFILE,
        }
        if self.owner_type == "group" and self.memory_type not in group_kinds:
            raise ValueError(
                f"memory_type {self.memory_type.value!r} is not valid for a group scope"
            )
        if self.owner_type != "group" and self.memory_type in group_kinds - {
            GetMemoryType.EPISODE
        }:
            raise ValueError(
                f"memory_type {self.memory_type.value!r} needs a group scope"
            )
        wants_member = self.memory_type == GetMemoryType.MEMBER_PROFILE
        if wants_member != (self.member_id is not None):
            raise ValueError("member_id goes with memory_type member_profile, only")
        profile = self.memory_type == GetMemoryType.GROUP_PROFILE
        if self.include_superseded and not profile:
            raise ValueError("include_superseded goes with memory_type group_profile")
        if profile and self.filters is not None:
            raise ValueError("filters are not supported for memory_type group_profile")
        return self

    @property
    def scope_group_ids(self) -> tuple[str, ...]:
        """The groups of a group scope, in request order; ``()`` otherwise."""
        if self.group_id is not None:
            return (self.group_id,)
        return tuple(dict.fromkeys(self.group_ids or ()))

    @property
    def owner_id(self) -> str:
        """Derived from whichever owner scope is set (the first group of a pool)."""
        groups = self.scope_group_ids
        return self.user_id or self.agent_id or (groups[0] if groups else "")

    @property
    def owner_type(self) -> Literal["user", "agent", "group"]:
        """``"user"``, ``"agent"`` or ``"group"``, by the owner scope set."""
        if self.user_id is not None:
            return "user"
        if self.agent_id is not None:
            return "agent"
        return "group"


# ── Item DTOs (mirror Search*Item shapes minus score) ────────────────────


class GetEpisodeItem(BaseModel):
    """Episode (or group scene) listing item."""

    model_config = ConfigDict(extra="forbid")

    id: str
    user_id: str | None
    owner_type: Literal["group"] | None = None
    """``group`` on a group record; unset on one-to-one memory."""
    group_id: str | None = None
    """Owning group, for a group episode (``user_id`` is then ``None``)."""
    source: SearchSource | None = None
    """A group record's segment and span (epoch ms); unset on one-to-one."""
    app_id: str = "default"
    project_id: str = "default"
    session_id: str | None = None
    """``None`` for an episode Reflection merged from several sessions."""
    timestamp: _dt.datetime
    sender_ids: list[str] = Field(default_factory=list)
    subject_ids: list[str] = Field(default_factory=list)
    origin: Literal["member", "assistant", "mixed"] | None = None
    summary: str
    subject: str
    episode: str
    type: Literal["Conversation", "Scene"]


class GetProfileItem(BaseModel):
    """A user's profile, or a member's profile inside a group.

    A member-in-group profile (``owner_type = group``) holds only the items
    the asker's windows allow.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    user_id: str | None
    owner_type: Literal["group"] | None = None
    group_id: str | None = None
    member_id: str | None = None
    app_id: str = "default"
    project_id: str = "default"
    profile_data: dict[str, object]


class GetImageItem(BaseModel):
    """A group picture memory listing item -- one appearance of one picture."""

    model_config = ConfigDict(extra="forbid")

    id: str
    entry_id: str
    image_id: str
    group_id: str
    app_id: str = "default"
    project_id: str = "default"
    session_id: str | None = None
    timestamp: _dt.datetime
    sender_ids: list[str] = Field(default_factory=list)
    """Who posted the picture."""
    subject_ids: list[str] = Field(default_factory=list)
    origin: Literal["member", "assistant", "mixed"] | None = None
    description: str
    ocr: str = ""
    name: str = ""
    original_path: str = ""
    media_type: str = ""
    width: int = 0
    height: int = 0


class GetAgentCaseItem(BaseModel):
    """Agent case listing item — always agent-scoped."""

    model_config = ConfigDict(extra="forbid")

    id: str
    entry_id: str
    """The case's id inside its owner's markdown (``ac_<YYYYMMDD>_<NNNN>``);
    what a skill's ``source_case_ids`` refers to. ``id`` is the index row key,
    ``<agent_id>_<entry_id>``."""
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
    scenes: list[GetEpisodeItem] = Field(default_factory=list)
    """A group's current scene versions (``memory_type = scene``)."""
    images: list[GetImageItem] = Field(default_factory=list)
    """A group's picture memories (``memory_type = image_memory``)."""
    profiles: list[GetProfileItem] = Field(default_factory=list)
    group_profile: list[GroupProfileEntry] = Field(default_factory=list)
    """A group's profile items (``memory_type = group_profile``), newest first."""
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

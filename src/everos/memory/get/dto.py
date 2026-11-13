"""Public DTOs for ``POST /api/v2/memory/get``.

Contract per the final design (mirrors :mod:`memory.search.dto` shape,
minus ``score`` because /get is a paginated listing rather than a
ranked retrieval):

* ``owner_type`` × ``memory_type`` are strictly paired:

  - ``user`` → ``episode`` | ``profile`` | ``scene`` | ``image_memory``
  - ``group`` → ``episode`` | ``scene`` | ``image_memory`` | ``member_profile``
    (the last with ``member_id``)

  ``include_groups`` (off by default) widens a ``user`` listing of
  ``episode`` or ``scene`` by the user's current groups, paged as one list.
  EverOS 2.0 has no agent memory: ``agent_id`` is refused
  (:data:`~everos.memory.search.NO_AGENT_MEMORY`).

* ``GetData`` always contains every kind array for symmetry with
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
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from everos.memory.search import (
    NO_AGENT_MEMORY,
    FilterNode,
    GroupProfileEntry,
    SearchSource,
)


class GetMemoryType(StrEnum):
    """The kinds enumerated by ``/get``.

    ``episode``, ``profile``, ``scene`` and ``image_memory`` are listed for a
    user; ``image_memory`` and the group kinds for a group. Cross-pairs are
    rejected by
    :meth:`GetRequest._validate_owner_memory_type_pair`.
    """

    EPISODE = "episode"
    PROFILE = "profile"
    SCENE = "scene"
    """An owner's current scene versions (user and group scopes)."""
    IMAGE_MEMORY = "image_memory"
    """An owner's current picture memories (user and group scopes)."""
    MEMBER_PROFILE = "member_profile"
    """One member's profile in the group(s), with ``member_id`` (group scopes)."""
    GROUP_PROFILE = "group_profile"
    """The group-profile items the asker can see (group scopes); with
    ``include_superseded`` every version, not only the current ones."""


# ── Request ──────────────────────────────────────────────────────────────


class GetRequest(BaseModel):
    """Request body for ``POST /api/v2/memory/get``.

    Callers identify the memory owner via ``user_id`` or a group scope --
    exactly one must be set. Internally the manager keeps using
    ``owner_id`` / ``owner_type`` (the storage tables' columns); those
    are exposed as derived properties so the rename only affects the
    wire contract. An ``agent_id`` is refused with :data:`NO_AGENT_MEMORY`.
    """

    model_config = ConfigDict(extra="forbid")

    user_id: str | None = Field(default=None, min_length=1)
    """Memory owner for ``episode`` / ``profile`` / ``scene``; exactly one owner
    scope must be set (``user_id``, ``group_id`` or ``group_ids``)."""
    group_id: str | None = Field(default=None, min_length=1)
    group_ids: list[str] | None = Field(default=None, min_length=1, max_length=16)
    """A group scope (docs/group-memory.md, 5.4): ``episode`` only for now. The
    listing holds what the asker's membership windows contain."""
    asker_id: str | None = Field(default=None, min_length=1)
    """Who asks: required for a group scope (or the identity header); with
    ``user_id`` optional and equal to it."""
    audience: Literal["asker", "group"] = "asker"
    member_id: str | None = Field(default=None, min_length=1)
    """With ``memory_type = member_profile`` only: whose profile to list."""
    include_superseded: bool = False
    """With ``memory_type = group_profile`` only: list the versions later items
    replaced too, each marked ``superseded``. Forgotten items never return."""
    include_groups: bool = False
    """With ``user_id`` and ``memory_type`` ``episode`` or ``scene`` (off by
    default): list the user's groups' records too -- every group of
    ``app_id`` / ``project_id`` the user currently belongs to, as a group
    listing by the user lists it (inside the user's windows) -- in one listing
    with the personal ones, sorted and paged together. Every item then names
    its owner (``owner_type`` ``user``, or ``group`` with ``group_id``).
    Without it a request is answered exactly as before."""
    include_group_ids: list[str] | None = Field(
        default=None, min_length=1, max_length=16
    )
    """With ``include_groups``: only these of the user's groups. An id that is
    not a group the user currently belongs to adds nothing."""
    app_id: str = "default"
    project_id: str = "default"
    """App / project scope (default ``"default"``). Pinned into the query
    ``where`` so a listing never crosses into another space's rows."""
    memory_type: GetMemoryType
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=20, ge=1, le=100)
    sort_by: Literal["timestamp", "updated_at"] = "timestamp"
    """Sort column. ``profile`` silently overrides to ``updated_at`` (a
    profile has no timestamp)."""

    sort_order: Literal["asc", "desc"] = "desc"
    filters: FilterNode | None = None
    """Filters DSL — same shape as ``/search``, ``AND`` / ``OR``
    combinators allowed."""

    @model_validator(mode="before")
    @classmethod
    def _refuse_agent_scope(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        memory_type = data.get("memory_type")
        if (
            data.get("agent_id") is not None
            or data.get("context_group_id") is not None
            or (isinstance(memory_type, str) and memory_type in _AGENT_KINDS)
        ):
            raise ValueError(NO_AGENT_MEMORY)
        return data

    @model_validator(mode="after")
    def _validate_one_scope(self) -> Self:
        scopes = [
            self.user_id is not None,
            self.group_id is not None,
            self.group_ids is not None,
        ]
        if sum(scopes) != 1:
            raise ValueError(
                "exactly one of user_id / group_id / group_ids must be provided"
            )
        return self

    @model_validator(mode="after")
    def _validate_owner_memory_type_pair(self) -> Self:
        # Runs after the scope validator (declaration order), so ``owner_type``
        # is well-defined here.
        user_kinds = {
            GetMemoryType.EPISODE,
            GetMemoryType.PROFILE,
            GetMemoryType.SCENE,
            GetMemoryType.IMAGE_MEMORY,
        }
        if self.owner_type == "user" and self.memory_type not in user_kinds:
            raise ValueError(
                f"memory_type {self.memory_type.value!r} is not valid "
                "when user_id is set"
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
        if self.owner_type != "group" and self.memory_type in group_kinds - user_kinds:
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

    @model_validator(mode="after")
    def _validate_include_groups(self) -> Self:
        if self.include_group_ids is not None and not self.include_groups:
            raise ValueError("include_group_ids goes with include_groups")
        if not self.include_groups:
            return self
        if self.owner_type != "user":
            raise ValueError("include_groups extends a user_id listing")
        if self.memory_type not in {GetMemoryType.EPISODE, GetMemoryType.SCENE}:
            raise ValueError(
                "include_groups lists episodes and scenes; a member's profile in "
                "a group is listed with group_id and memory_type member_profile"
            )
        if self.audience == "group":
            raise ValueError(
                "audience 'group' is for a reply posted into one group: list "
                "that group with group_id"
            )
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
        return self.user_id or (groups[0] if groups else "")

    @property
    def owner_type(self) -> Literal["user", "group"]:
        """``"user"`` or ``"group"``, by the owner scope set."""
        if self.user_id is not None:
            return "user"
        return "group"


_AGENT_KINDS = frozenset({"agent_case", "agent_skill"})
"""The memory types of the agent memory 2.0 removed (refused by name)."""


# ── Item DTOs (mirror Search*Item shapes minus score) ────────────────────


class GetEpisodeItem(BaseModel):
    """Episode (or group scene) listing item."""

    model_config = ConfigDict(extra="forbid")

    id: str
    user_id: str | None
    owner_type: Literal["user", "group"] | None = None
    """``group`` on a group record; ``user`` on a personal one when the listing
    had ``include_groups``; unset on one-to-one memory otherwise."""
    group_id: str | None = None
    """Owning group, for a group episode (``user_id`` is then ``None``)."""
    source: SearchSource | None = None
    """A group record's segment and span (epoch ms); unset on one-to-one."""
    app_id: str = "default"
    project_id: str = "default"
    session_id: str | None = None
    """``None`` for an episode merged from several sessions (by the Reflection
    of an earlier 2.0 build)."""
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
    """A picture memory listing item -- one appearance of one picture.

    A user's names the user (``user_id``), a group's the group (``group_id``),
    as an episode listing item does.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    entry_id: str
    image_id: str
    user_id: str | None = None
    group_id: str | None = None
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


# ── Response envelope ────────────────────────────────────────────────────


class GetData(BaseModel):
    """Body of ``response.data``.

    Every array is always present so client code can iterate without
    branching on ``memory_type``; the route populates exactly one.
    """

    model_config = ConfigDict(extra="forbid")

    episodes: list[GetEpisodeItem] = Field(default_factory=list)
    scenes: list[GetEpisodeItem] = Field(default_factory=list)
    """An owner's current scene versions (``memory_type = scene``)."""
    images: list[GetImageItem] = Field(default_factory=list)
    """An owner's picture memories (``memory_type = image_memory``)."""
    profiles: list[GetProfileItem] = Field(default_factory=list)
    group_profile: list[GroupProfileEntry] = Field(default_factory=list)
    """A group's profile items (``memory_type = group_profile``), newest first."""
    total_count: int = 0
    """Total rows matching the request's owner + filter predicate."""

    count: int = 0
    """Number of items in this page (``len(items)`` after slicing)."""


class GetResponse(BaseModel):
    """Top-level response envelope."""

    model_config = ConfigDict(extra="forbid")

    request_id: str
    data: GetData

"""Request and response models for the group routes (docs/group-memory.md, 5.1).

``PUT /groups/{group_id}`` creates or updates a group, ``POST
/groups/{group_id}/roster`` appends roster events, ``GET /groups/{group_id}``
reads a group. The same :class:`RosterEventDTO` shape is the ``roster`` field
of ``/memory/add``.

Strings that reach markdown as one inline line (names, titles, aliases) are
refused when they hold a line break or a control character, as a session id
is: stored, such a value would read back as another one.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

from .ids import (
    PATH_SAFE_CHARSET,
    AppScopeId,
    PathSafeId,
    ProjectScopeId,
    one_trimmed_line,
)

OneLine = Annotated[str, AfterValidator(one_trimmed_line)]
MemberId = Annotated[
    PathSafeId, Field(min_length=1, max_length=128, pattern=PATH_SAFE_CHARSET)
]
AskerId = Annotated[str, Field(min_length=1, max_length=128)]
DelivererId = Annotated[
    str, AfterValidator(one_trimmed_line), Field(min_length=1, max_length=128)
]


class RosterEventDTO(BaseModel):
    """One roster event: ``join`` opens an interval, ``leave`` closes it,
    ``update`` changes ``display_name`` / ``aliases`` / ``role`` / ``title``."""

    model_config = ConfigDict(extra="forbid")

    op: Literal["join", "leave", "update"]
    member_id: MemberId
    kind: Literal["human", "assistant"] | None = None
    display_name: OneLine | None = Field(default=None, max_length=256)
    aliases: list[Annotated[OneLine, Field(min_length=1, max_length=128)]] | None = (
        Field(default=None, max_length=32)
    )
    role: Literal["owner", "admin", "member", "assistant"] | None = None
    title: OneLine | None = Field(default=None, max_length=256)
    at: int | None = Field(
        default=None,
        gt=0,
        description="Epoch ms the event takes effect; defaults to now.",
    )


class _GroupScope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    app_id: AppScopeId = Field(
        default="default", min_length=1, max_length=128, pattern=PATH_SAFE_CHARSET
    )
    project_id: ProjectScopeId = Field(
        default="default", min_length=1, max_length=128, pattern=PATH_SAFE_CHARSET
    )
    asker_id: AskerId | None = Field(
        default=None,
        description=(
            "Who asks. Required unless the deployment's identity header "
            "([access] identity_header) carries it; a different value there is 403."
        ),
    )


class GroupUpsertRequest(_GroupScope):
    """``PUT /groups/{group_id}``: create the group, or update its meta."""

    name: OneLine | None = Field(default=None, max_length=256)
    description: str | None = Field(default=None, max_length=4096)
    created_at: int | None = Field(default=None, gt=0)
    history_visible_from: Literal["join", "creation"] | None = None
    extra: dict[str, Any] | None = None
    roster: list[RosterEventDTO] = Field(default_factory=list, max_length=500)
    deliverer_id: DelivererId | None = Field(
        default=None,
        description=(
            "The identity the gateway asserts for the group's key: the one "
            "program that forwards the group's chat. Once set, only it may "
            "add messages to the group's sessions or flush them, and it holds "
            "every right of an owner. Left out: unchanged; null: cleared. Set "
            "and cleared by owners, admins and the deliverer itself."
        ),
    )


class RosterChangeRequest(_GroupScope):
    """``POST /groups/{group_id}/roster``: append roster events."""

    events: list[RosterEventDTO] = Field(min_length=1, max_length=500)


class MemberIntervalDTO(BaseModel):
    """One membership interval, ``[joined_at, left_at)`` in epoch ms."""

    member_id: str
    seq: int
    kind: Literal["human", "assistant"]
    display_name: str
    aliases: list[str] = Field(default_factory=list)
    role: Literal["owner", "admin", "member", "assistant"]
    title: str = ""
    joined_at: int
    left_at: int | None = None
    source: Literal["api", "inline", "implicit"]


class GroupDTO(BaseModel):
    """A group's meta and roster."""

    group_id: str
    app_id: str
    project_id: str
    name: str
    description: str
    created_at: int
    created_by: str | None = None
    history_visible_from: Literal["join", "creation"]
    extra: dict[str, Any] = Field(default_factory=dict)
    roster: list[MemberIntervalDTO] = Field(
        description="Current members; every interval with include_history=true."
    )
    deliverer_id: str | None = Field(
        default=None,
        description="The group's deliverer, or null when anyone may deliver.",
    )


class GroupUpsertResponseData(BaseModel):
    created: bool
    events_applied: int
    group: GroupDTO


class RosterChangeResponseData(BaseModel):
    events_applied: int
    group: GroupDTO

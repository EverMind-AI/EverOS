"""Request / response DTOs of a forget inside a group (docs/group-memory.md, 5.5).

``POST /memory/forget`` takes either body: a personal picture forget
(``user_id`` + ``image_id``, :class:`ForgetImageRequest`) or a group forget
(``group_id`` + ``asker_id`` + ``target``, :class:`GroupForgetRequest`); the
presence of ``group_id`` picks the shape. ``POST /memory/forget/undo`` takes a
picture's undo or a group forget's ``forget_id``, the same way.

A group target is one of five kinds:

* ``message`` -- ``session_id`` + ``message_ids`` (the client's ids when the
  batch sent them);
* ``member`` -- ``member_id``: every message of the member in the group, and
  the member's profile in it;
* ``session`` -- ``session_id``: every message of one of the group's
  sessions;
* ``record`` -- ``record_ids``: episodes, picture memories, scenes,
  group-profile items (``<group>_<entry>`` or the entry id) and
  member-profile items (``pi_...``);
* ``image`` -- ``image_id``: the group's memories of one picture.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag

from .forget import ForgetImageRequest, UndoForgetRequest
from .ids import (
    PATH_SAFE_CHARSET,
    AppScopeId,
    PathSafeId,
    ProjectScopeId,
    SessionId,
)

_IMAGE_ID = r"^[0-9a-f]{64}$"
_FORGET_ID = r"^fg_[0-9a-f]{16}$"
_RECORD_ID = r"^[a-zA-Z0-9_.@+-]{1,300}$"


class MessageTarget(BaseModel):
    """Messages of one session of the group, by id."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["message"]
    session_id: SessionId = Field(min_length=1, max_length=128)
    message_ids: list[Annotated[str, Field(min_length=1, max_length=256)]] = Field(
        min_length=1, max_length=100
    )


class MemberTarget(BaseModel):
    """Every message one member sent in the group, and their profile in it."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["member"]
    member_id: PathSafeId = Field(min_length=1, max_length=128)


class SessionTarget(BaseModel):
    """Every message of one of the group's sessions (owners and admins)."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["session"]
    session_id: SessionId = Field(min_length=1, max_length=128)


class RecordTarget(BaseModel):
    """Records by id: episodes, picture memories, scenes, profile items."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["record"]
    record_ids: list[Annotated[str, Field(pattern=_RECORD_ID)]] = Field(
        min_length=1, max_length=50
    )


class ImageTarget(BaseModel):
    """The group's memories of one picture, by content hash."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["image"]
    image_id: str = Field(min_length=64, max_length=64, pattern=_IMAGE_ID)


GroupTarget = Annotated[
    MessageTarget | MemberTarget | SessionTarget | RecordTarget | ImageTarget,
    Field(discriminator="kind"),
]


class GroupForgetRequest(BaseModel):
    """A forget inside a group (docs/group-memory.md, section 5.5).

    Who may forget what: a message its sender, a member that member, a record
    a member whose own messages alone it rests on, a picture its poster; the
    group's owners and admins anything; a session only they. A former member
    or a stranger is refused (403). ``dry_run`` changes nothing and answers
    with exactly what the real run would touch.
    """

    model_config = ConfigDict(extra="forbid")

    app_id: AppScopeId = Field(
        default="default", min_length=1, max_length=128, pattern=PATH_SAFE_CHARSET
    )
    project_id: ProjectScopeId = Field(
        default="default", min_length=1, max_length=128, pattern=PATH_SAFE_CHARSET
    )
    group_id: PathSafeId = Field(min_length=1, max_length=128)
    asker_id: PathSafeId | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        description="Who asks; required unless the identity header names them.",
    )
    target: GroupTarget
    reason: str | None = Field(default=None, max_length=500)
    dry_run: bool = False


class ForgottenMessageData(BaseModel):
    """One message a group forget marked forgotten."""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    message_id: str


class GroupForgetResponseData(BaseModel):
    """What a forget inside a group touched (or would, on a dry run).

    Records are named as the read side names them (``<group>_<entry>``);
    member-profile items by their ``item_id``. ``episodes`` are the versions
    taken out -- each either rewritten (``rewritten`` new versions, ids in
    ``new_episodes``) or dropped when nothing of its segment was left.
    ``stale_*`` items were taken out and are written again from their
    remaining sources by the profile's next update.
    """

    model_config = ConfigDict(extra="forbid")

    forget_id: str = ""
    dry_run: bool = False
    group_id: str
    target: dict[str, Any] = Field(default_factory=dict)
    messages: list[ForgottenMessageData] = Field(default_factory=list)
    episodes: list[str] = Field(default_factory=list)
    rewritten: int = 0
    new_episodes: list[str] = Field(default_factory=list)
    image_memories: list[str] = Field(default_factory=list)
    images: list[str] = Field(default_factory=list)
    scenes: list[str] = Field(default_factory=list)
    rewritten_scenes: int = 0
    new_scenes: list[str] = Field(default_factory=list)
    member_profiles: list[str] = Field(default_factory=list)
    member_profile_items: list[str] = Field(default_factory=list)
    stale_member_profile_items: list[str] = Field(default_factory=list)
    group_profile_items: list[str] = Field(default_factory=list)
    stale_group_profile_items: list[str] = Field(default_factory=list)


class GroupUndoForgetRequest(BaseModel):
    """Undo one forget of a group by its id, with the forget's own authorisation."""

    model_config = ConfigDict(extra="forbid")

    app_id: AppScopeId = Field(
        default="default", min_length=1, max_length=128, pattern=PATH_SAFE_CHARSET
    )
    project_id: ProjectScopeId = Field(
        default="default", min_length=1, max_length=128, pattern=PATH_SAFE_CHARSET
    )
    group_id: PathSafeId = Field(min_length=1, max_length=128)
    asker_id: PathSafeId | None = Field(default=None, min_length=1, max_length=128)
    forget_id: str = Field(pattern=_FORGET_ID)


class GroupUndoForgetResponseData(BaseModel):
    """What an undo of a group forget put back.

    ``applied`` is ``False`` when the forget was undone already. ``withdrawn``
    are the records the forget had written, now taken out again.
    """

    model_config = ConfigDict(extra="forbid")

    forget_id: str
    applied: bool
    group_id: str
    messages: list[ForgottenMessageData] = Field(default_factory=list)
    episodes: list[str] = Field(default_factory=list)
    image_memories: list[str] = Field(default_factory=list)
    images: list[str] = Field(default_factory=list)
    scenes: list[str] = Field(default_factory=list)
    member_profiles: list[str] = Field(default_factory=list)
    member_profile_items: list[str] = Field(default_factory=list)
    group_profile_items: list[str] = Field(default_factory=list)
    withdrawn: list[str] = Field(default_factory=list)


def _by_group_id(body: Any) -> str:
    """``group`` for a body naming a group, ``user`` for any other."""
    if isinstance(body, dict):
        named = body.get("group_id") is not None
    else:
        named = isinstance(body, GroupForgetRequest | GroupUndoForgetRequest)
    return "group" if named else "user"


ForgetRequest = Annotated[
    Annotated[ForgetImageRequest, Tag("user")]
    | Annotated[GroupForgetRequest, Tag("group")],
    Discriminator(_by_group_id),
]
"""``POST /memory/forget``: a personal picture, or a target inside a group."""

UndoRequest = Annotated[
    Annotated[UndoForgetRequest, Tag("user")]
    | Annotated[GroupUndoForgetRequest, Tag("group")],
    Discriminator(_by_group_id),
]
"""``POST /memory/forget/undo``: a personal picture's, or a group forget's."""

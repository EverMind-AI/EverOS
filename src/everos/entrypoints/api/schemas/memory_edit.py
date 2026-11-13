"""Request / response DTOs of ``/memory/update``, ``/memory/history`` and a
user's ``/memory/forget`` by target (docs/api.md).

A record is named by the id a read returned: ``<owner>_<entry_id>`` (or the
bare entry id) for an episode, a picture memory or a scene version, and the
``item_id`` (``pi_<hex>``) for a profile item. A user's memory is named by
``user_id``; a group's by ``group_id`` with ``asker_id``.
"""

from __future__ import annotations

import datetime as _dt
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .ids import (
    PATH_SAFE_CHARSET,
    AppScopeId,
    PathSafeId,
    ProjectScopeId,
    SessionId,
)

_RECORD_ID = r"^[a-zA-Z0-9_.@+-]{1,300}$"
_IMAGE_ID = r"^[0-9a-f]{64}$"
_FORGET_ID = r"^fg_[0-9a-f]{16}$"


class UpdateMemoryRequest(BaseModel):
    """A person's change of one memory: a new version, linked to the old one.

    Fields a kind takes: an episode or a scene ``subject``, ``summary`` and
    ``content``; a picture memory ``description`` and ``ocr``; a profile item
    ``description``. A field left out keeps the stored text. ``hold`` (default
    true) keeps the new version from the model's later updates; ``hold:
    false`` alone releases a held version. ``expected_version`` (default:
    ``id``) is the version the caller read; when a newer one exists the
    answer is 409 and nothing is written.
    """

    model_config = ConfigDict(extra="forbid")

    app_id: AppScopeId = Field(
        default="default", min_length=1, max_length=128, pattern=PATH_SAFE_CHARSET
    )
    project_id: ProjectScopeId = Field(
        default="default", min_length=1, max_length=128, pattern=PATH_SAFE_CHARSET
    )
    user_id: PathSafeId | None = Field(default=None, min_length=1, max_length=128)
    group_id: PathSafeId | None = Field(default=None, min_length=1, max_length=128)
    asker_id: PathSafeId | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        description="Who asks; required for a group unless the identity header "
        "names them. Recorded on the new version as edited_by.",
    )
    id: str = Field(pattern=_RECORD_ID, description="The memory to change.")
    expected_version: str | None = Field(
        default=None,
        pattern=_RECORD_ID,
        description="The version the caller read; default: id.",
    )
    subject: str | None = Field(default=None, max_length=1_000)
    summary: str | None = Field(default=None, max_length=8_000)
    content: str | None = Field(default=None, min_length=1, max_length=40_000)
    description: str | None = Field(default=None, min_length=1, max_length=4_000)
    ocr: str | None = Field(default=None, max_length=4_000)
    hold: bool = True

    @model_validator(mode="after")
    def _one_owner(self) -> UpdateMemoryRequest:
        if (self.user_id is None) == (self.group_id is None):
            raise ValueError("name exactly one of user_id and group_id")
        return self


class UpdateMemoryResponseData(BaseModel):
    """The version a change wrote (or the current one, when nothing changed)."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["episode", "image_memory", "scene", "profile_item"]
    id: str
    supersedes: str | None = None
    changed: bool
    held: bool
    edited_by: str | None = None
    edited_at: _dt.datetime | None = None
    member_id: str | None = None


class MemoryVersionData(BaseModel):
    """One version of a memory.

    ``state`` is ``current``, ``superseded`` (a newer version replaced it),
    ``merged`` (a merge of several did) or ``forgotten``; ``next`` the version
    that replaced it. ``sources`` say what it rests on (times there in epoch
    ms), ``content`` what it says.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    state: Literal["current", "superseded", "forgotten", "merged"]
    timestamp: _dt.datetime | None = None
    next: str | None = None
    supersedes: str | None = None
    edited_by: str | None = None
    edited_at: _dt.datetime | None = None
    held: bool = False
    replaced_at: _dt.datetime | None = None
    reason: str | None = None
    sources: dict[str, Any] = Field(default_factory=dict)
    content: dict[str, Any] = Field(default_factory=dict)


class MemoryHistoryResponseData(BaseModel):
    """Every version of one memory the asker may see, current first."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal[
        "episode", "image_memory", "scene", "profile_item", "group_profile_item"
    ]
    id: str
    current: str | None = None
    member_id: str | None = None
    versions: list[MemoryVersionData] = Field(default_factory=list)


# ── a user's forget by target ────────────────────────────────────────────


class UserRecordTarget(BaseModel):
    """Records by id, of any kind: episodes, picture memories, scenes,
    profile items (and replaced values, by the id a read lists)."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["record"]
    record_ids: list[Annotated[str, Field(pattern=_RECORD_ID)]] = Field(
        min_length=1, max_length=100
    )


class UserSessionTarget(BaseModel):
    """Everything one conversation established for the user."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["session"]
    session_id: SessionId = Field(min_length=1, max_length=128)


class UserMessageTarget(BaseModel):
    """Messages of one of the user's conversations, by id: each segment that
    held one is remembered again without it."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["message"]
    session_id: SessionId = Field(min_length=1, max_length=128)
    message_ids: list[Annotated[str, Field(min_length=1, max_length=256)]] = Field(
        min_length=1, max_length=100
    )


class UserOwnerTarget(BaseModel):
    """Everything of the user: episodes, pictures, scenes, the whole profile."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["owner"]


class UserConditionTarget(BaseModel):
    """The memories of some kinds whose time falls in ``[start, end)``.

    Epoch ms; either bound may be left open. A memory's time is when what it
    says happened; a profile item's, when it was last observed.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["condition"]
    memory_types: list[Literal["episode", "image_memory", "scene", "profile"]] = Field(
        min_length=1, max_length=4
    )
    start: int | None = Field(default=None, ge=0)
    end: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _a_range(self) -> UserConditionTarget:
        if self.start is None and self.end is None:
            raise ValueError("a condition needs start, end or both")
        if self.start is not None and self.end is not None and self.end <= self.start:
            raise ValueError("end must be after start")
        return self


class UserImageTarget(BaseModel):
    """One picture by content hash: its memories, and the references to it."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["image"]
    image_id: str = Field(min_length=64, max_length=64, pattern=_IMAGE_ID)


UserTarget = Annotated[
    UserRecordTarget
    | UserSessionTarget
    | UserMessageTarget
    | UserOwnerTarget
    | UserConditionTarget
    | UserImageTarget,
    Field(discriminator="kind"),
]


class UserForgetRequest(BaseModel):
    """A forget in one user's memory, by target; a dry run unless told not to.

    ``dry_run`` (default **true**) changes nothing and answers with what the
    real run would touch; send ``dry_run: false`` to forget. A real forget
    answers with its ``forget_id``, the token ``/memory/forget/undo`` takes.
    """

    model_config = ConfigDict(extra="forbid")

    app_id: AppScopeId = Field(
        default="default", min_length=1, max_length=128, pattern=PATH_SAFE_CHARSET
    )
    project_id: ProjectScopeId = Field(
        default="default", min_length=1, max_length=128, pattern=PATH_SAFE_CHARSET
    )
    user_id: PathSafeId = Field(min_length=1, max_length=128)
    asker_id: PathSafeId | None = Field(default=None, min_length=1, max_length=128)
    target: UserTarget
    reason: str | None = Field(default=None, max_length=500)
    dry_run: bool = True


class UserForgetResponseData(BaseModel):
    """What a forget in a user's memory touched (or would, on a dry run).

    Records are named as a read names them (``<user>_<entry>``), profile items
    by ``item_id``. ``episodes`` and ``scenes`` are the versions taken out of
    default reads -- marked forgotten, or (a picture's forget) replaced by the
    ``new_episodes`` / ``new_scenes`` written without the picture. ``images``
    are picture files taken out of circulation.
    """

    model_config = ConfigDict(extra="forbid")

    forget_id: str = ""
    dry_run: bool
    user_id: str
    target: dict[str, Any] = Field(default_factory=dict)
    episodes: list[str] = Field(default_factory=list)
    new_episodes: list[str] = Field(default_factory=list)
    image_memories: list[str] = Field(default_factory=list)
    images: list[str] = Field(default_factory=list)
    scenes: list[str] = Field(default_factory=list)
    new_scenes: list[str] = Field(default_factory=list)
    profile_items: list[str] = Field(default_factory=list)


class UserUndoForgetRequest(BaseModel):
    """Undo one forget of a user by its id (the ``forget_id`` it answered)."""

    model_config = ConfigDict(extra="forbid")

    app_id: AppScopeId = Field(
        default="default", min_length=1, max_length=128, pattern=PATH_SAFE_CHARSET
    )
    project_id: ProjectScopeId = Field(
        default="default", min_length=1, max_length=128, pattern=PATH_SAFE_CHARSET
    )
    user_id: PathSafeId = Field(min_length=1, max_length=128)
    asker_id: PathSafeId | None = Field(default=None, min_length=1, max_length=128)
    forget_id: str = Field(pattern=_FORGET_ID)


class UserUndoForgetResponseData(BaseModel):
    """What an undo of a user's forget put back; ``applied: false`` when it was
    undone already. ``profile_items`` counts items."""

    model_config = ConfigDict(extra="forbid")

    forget_id: str
    applied: bool
    user_id: str
    target: dict[str, Any] | None = None
    episodes: list[str] = Field(default_factory=list)
    image_memories: list[str] = Field(default_factory=list)
    images: list[str] = Field(default_factory=list)
    scenes: list[str] = Field(default_factory=list)
    profile_items: int = 0

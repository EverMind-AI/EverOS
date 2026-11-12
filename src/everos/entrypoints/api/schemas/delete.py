"""Request / response DTOs of ``POST /memory/delete``."""

from __future__ import annotations

import datetime as _dt
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from everos.service import NO_AGENT_MEMORY

from .ids import (
    PATH_SAFE_CHARSET,
    AppScopeId,
    PathSafeId,
    ProjectScopeId,
    SessionId,
)

_EPISODE_ID = re.compile(r"(?P<owner>[a-zA-Z0-9_.@+-]{1,128})_ep_(?P<day>\d{8})_\d+")


class DeleteRequest(BaseModel):
    """Which memories to delete: an owner, a session, or one owner's session.

    At least one of ``user_id`` and ``session_id``. EverOS 2.0 has no hard
    delete: a delete is a forget of the same scope (``POST /memory/forget``,
    run at once), reversible by the ``forget_ids`` it answers with. EverOS 2.0
    has no agent memory: an ``agent_id`` is refused (422).

    With ``group_id`` the delete is the group's (docs/group-memory.md,
    section 15): an owner's or admin's forget -- soft, propagated, undone by
    the returned ``forget_id`` -- of one of the group's sessions
    (``session_id``) or of one member's messages (``user_id``, the member).
    ``asker_id`` names the owner or admin.
    """

    model_config = ConfigDict(extra="forbid")

    app_id: AppScopeId = Field(
        default="default", min_length=1, max_length=128, pattern=PATH_SAFE_CHARSET
    )
    project_id: ProjectScopeId = Field(
        default="default", min_length=1, max_length=128, pattern=PATH_SAFE_CHARSET
    )
    user_id: PathSafeId | None = Field(default=None, min_length=1, max_length=128)
    session_id: SessionId | None = Field(default=None, min_length=1, max_length=128)
    group_id: PathSafeId | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        description="A group-scoped delete: an owner's or admin's forget of one "
        "session (session_id) or one member's messages (user_id).",
    )
    asker_id: PathSafeId | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        description="Who asks, for a group-scoped delete (an owner or admin).",
    )

    @model_validator(mode="before")
    @classmethod
    def _refuse_agent_scope(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("agent_id") is not None:
            raise ValueError(NO_AGENT_MEMORY)
        return data

    @model_validator(mode="after")
    def _one_scope(self) -> DeleteRequest:
        if self.user_id is None and self.session_id is None:
            raise ValueError("set at least one of user_id, session_id")
        if self.group_id is not None:
            if self.user_id is not None and self.session_id is not None:
                raise ValueError(
                    "a group-scoped delete names one session_id or one user_id"
                )
        elif self.asker_id is not None:
            raise ValueError("asker_id is taken only with group_id")
        return self


class DeleteResponseData(BaseModel):
    """What a delete forgot: EverOS 2.0 has no hard delete.

    Attributes:
        filters: The scope fields the request set.
        count: Memories taken out of default reads (episodes, picture
            memories, scenes and profile items marked forgotten).
        forget_ids: The forgets it ran, one per user it reached;
            ``POST /memory/forget/undo`` takes each.
    """

    filters: list[str]
    count: int
    forget_ids: list[str] = Field(default_factory=list)


class GroupDeleteResponseData(BaseModel):
    """What a group-scoped delete forgot, and the forget's id for its undo.

    Attributes:
        filters: The scope fields the request set.
        count: Records taken out of the group's memory.
        forget_id: The forget it ran (``POST /memory/forget/undo`` takes it).
    """

    filters: list[str]
    count: int
    forget_id: str


class DeleteReason(BaseModel):
    """Why the caller deletes: logged as ``code``; ``note`` is never logged."""

    model_config = ConfigDict(extra="forbid")

    code: Literal[
        "hallucination",
        "outdated",
        "wrong_subject",
        "wrong_time",
        "redundant",
        "missing",
        "privacy",
        "style",
        "other",
    ]
    note: str | None = Field(default=None, max_length=256)


class DeleteByIdsRequest(BaseModel):
    """Episodes to delete, by memory id ``<owner_id>_<entry_id>``.

    A forget of those episodes (target ``record``, per owner): nothing leaves
    markdown. The whole request is rejected (422, nothing forgotten) if any id
    is not an episode id of a path-safe owner with a real date.
    """

    model_config = ConfigDict(extra="forbid")

    app_id: AppScopeId = Field(
        default="default", min_length=1, max_length=128, pattern=PATH_SAFE_CHARSET
    )
    project_id: ProjectScopeId = Field(
        default="default", min_length=1, max_length=128, pattern=PATH_SAFE_CHARSET
    )
    memory_type: Literal["episode"]
    memory_ids: list[str] = Field(min_length=1, max_length=50)
    reason: DeleteReason | None = None

    @field_validator("memory_ids")
    @classmethod
    def _episode_ids(cls, ids: list[str]) -> list[str]:
        for memory_id in ids:
            match = _EPISODE_ID.fullmatch(memory_id)
            if match is None or match["owner"] in (".", ".."):
                raise ValueError(f"not an episode memory id: {memory_id!r}")
            try:
                _dt.date(
                    int(match["day"][:4]), int(match["day"][4:6]), int(match["day"][6:])
                )
            except ValueError:
                raise ValueError(f"no such date in {memory_id!r}") from None
        return list(dict.fromkeys(ids))


class DeleteByIdsResponseData(BaseModel):
    """``{"memory_type": "episode", "success": true}`` (Cloud's shape), with
    the forgets it ran (``forget_ids``, one per owner)."""

    memory_type: Literal["episode"] = "episode"
    success: bool = True
    forget_ids: list[str] = Field(default_factory=list)

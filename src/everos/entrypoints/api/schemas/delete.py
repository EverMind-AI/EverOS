"""Request / response DTOs of ``POST /memory/delete``."""

from __future__ import annotations

import datetime as _dt
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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

    At least one of ``user_id``, ``agent_id`` and ``session_id``; never both
    ``user_id`` and ``agent_id``.
    """

    model_config = ConfigDict(extra="forbid")

    app_id: AppScopeId = Field(
        default="default", min_length=1, max_length=128, pattern=PATH_SAFE_CHARSET
    )
    project_id: ProjectScopeId = Field(
        default="default", min_length=1, max_length=128, pattern=PATH_SAFE_CHARSET
    )
    user_id: PathSafeId | None = Field(default=None, min_length=1, max_length=128)
    agent_id: PathSafeId | None = Field(default=None, min_length=1, max_length=128)
    session_id: SessionId | None = Field(default=None, min_length=1, max_length=128)

    @model_validator(mode="after")
    def _one_scope(self) -> DeleteRequest:
        if self.user_id is None and self.agent_id is None and self.session_id is None:
            raise ValueError("set at least one of user_id, agent_id, session_id")
        if self.user_id is not None and self.agent_id is not None:
            raise ValueError("user_id and agent_id are mutually exclusive")
        return self


class DeleteResponseData(BaseModel):
    """What a deletion removed.

    Attributes:
        filters: The scope fields the request set.
        count: Memories removed (entries, plus one per skill and one for the
            profile when a whole owner is deleted).
    """

    filters: list[str]
    count: int


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

    The whole request is rejected (422, nothing deleted) if any id is not an
    episode id of a path-safe owner with a real date.
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
    """``{"memory_type": "episode", "success": true}`` (Cloud's shape)."""

    memory_type: Literal["episode"] = "episode"
    success: bool = True

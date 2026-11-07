"""Request / response DTOs of ``POST /memory/forget``."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .ids import (
    PATH_SAFE_CHARSET,
    AppScopeId,
    PathSafeId,
    ProjectScopeId,
)

_IMAGE_ID = r"^[0-9a-f]{64}$"
"""A picture id is the lowercase hex SHA-256 of its bytes.

Pinned as a pattern rather than a free string because the id becomes a file
name: an id that is not a hash cannot name a file this store wrote, and
letting one through would turn a forget into an arbitrary-path operation.
"""


class ForgetImageRequest(BaseModel):
    """Which picture to forget, for which user.

    A picture is owned per user even though its file is shared: the memories
    written about it are the user's, and one user forgetting a picture must not
    take it out of another user's memory.
    """

    model_config = ConfigDict(extra="forbid")

    app_id: AppScopeId = Field(
        default="default", min_length=1, max_length=128, pattern=PATH_SAFE_CHARSET
    )
    project_id: ProjectScopeId = Field(
        default="default", min_length=1, max_length=128, pattern=PATH_SAFE_CHARSET
    )
    user_id: PathSafeId = Field(min_length=1, max_length=128)
    image_id: str = Field(min_length=64, max_length=64, pattern=_IMAGE_ID)
    reason: str | None = Field(default=None, max_length=500)


class ForgetImageResponseData(BaseModel):
    """What a forget touched.

    ``episodes`` are the entry ids of the *new* versions written without the
    picture. The versions they supersede are still in markdown, on purpose:
    asking what was said before a forget has to keep working.
    """

    model_config = ConfigDict(extra="forbid")

    image_id: str
    image_memories: int
    episodes: list[str] = Field(default_factory=list)
    profiles: int = 0

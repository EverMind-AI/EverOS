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


class ForgottenPictureData(BaseModel):
    """One picture memory the request touched.

    ``entry_id`` names the occurrence, ``image_id`` the file, and ``name`` is
    what the client called the picture on that arrival — provenance a caller
    needs to map the memory back to its own input.
    """

    model_config = ConfigDict(extra="forbid")

    entry_id: str
    image_id: str
    name: str = ""


class UndoForgetRequest(ForgetImageRequest):
    """Which picture's newest forget to undo, for which user.

    Same fields as :class:`ForgetImageRequest` on purpose: an undo addresses
    exactly what the forget addressed, and a caller that kept its forget
    request can reuse it. The forget it reverses is the newest one for that
    picture by that user that no earlier undo already reversed.
    """


class UndoForgetResponseData(BaseModel):
    """What an undo put back.

    ``applied`` is ``False`` when there was nothing to undo — the picture was
    never forgotten, or an earlier undo already reversed it. The rest of the
    payload is still filled in, so a caller sees the state it ends in rather
    than having to ask again.
    """

    model_config = ConfigDict(extra="forbid")

    image_id: str
    forget_id: str = ""
    applied: bool
    image_memories: int
    episodes: list[str] = Field(default_factory=list)
    profiles: int = 0
    pictures: list[ForgottenPictureData] = Field(default_factory=list)


class ForgetImageResponseData(BaseModel):
    """What a forget touched.

    ``episodes`` are the entry ids of the *new* versions written without the
    picture. The versions they supersede are still in markdown, on purpose:
    asking what was said before a forget has to keep working.

    ``pictures`` lists the picture memories themselves, in entry order;
    ``image_memories`` repeats its length so a caller that only needs the
    count does not have to walk it.
    """

    model_config = ConfigDict(extra="forbid")

    image_id: str
    image_memories: int
    episodes: list[str] = Field(default_factory=list)
    profiles: int = 0
    pictures: list[ForgottenPictureData] = Field(default_factory=list)

"""Request / response DTOs of ``POST /memory/delete``."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .ids import PATH_SAFE_CHARSET, AppScopeId, PathSafeId, ProjectScopeId


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
    session_id: str | None = Field(default=None, min_length=1, max_length=128)

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

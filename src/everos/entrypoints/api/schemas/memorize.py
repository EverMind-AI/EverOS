"""Request and response models for ``/memory/add`` and ``/memory/flush``.

DTOs follow the v1 API brief (01_v1_api_brief.md §2 / §3).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .ids import PATH_SAFE_CHARSET, AppScopeId, PathSafeId, ProjectScopeId


class ToolFunctionDTO(BaseModel):
    name: str
    arguments: str  # JSON string per OpenAI Chat Completions spec


class ToolCallDTO(BaseModel):
    id: str
    type: str = "function"
    function: ToolFunctionDTO


class ContentItemDTO(BaseModel):
    """Content piece (v1 API brief appendix A)."""

    type: Literal["text", "image", "audio", "doc", "pdf", "html", "email"]
    text: str | None = None
    uri: str | None = None
    base64: str | None = None
    ext: str | None = None
    name: str | None = None
    extras: dict[str, Any] | None = None

    model_config = ConfigDict(extra="forbid")


class MessageItemDTO(BaseModel):
    # ``sender_id`` becomes ``owner_id`` and then a directory segment on the
    # episode write path, so it carries the same path-safety guard as
    # ``app_id`` / ``project_id`` (charset whitelist + ``.``/``..`` rejection).
    sender_id: PathSafeId = Field(
        ...,
        min_length=1,
        max_length=128,
        pattern=PATH_SAFE_CHARSET,
    )
    sender_name: str | None = None
    role: Literal["user", "assistant", "tool"]
    timestamp: int = Field(
        ...,
        gt=0,
        description=(
            "Message event time as Unix epoch in **milliseconds** "
            "(v1 API contract; the algo layer auto-detects sec vs ms "
            "for backward compat but the contract is ms)."
        ),
    )
    content: str | list[ContentItemDTO]
    tool_calls: list[ToolCallDTO] | None = None
    tool_call_id: str | None = None

    @model_validator(mode="after")
    def _tool_row_needs_call_id(self) -> MessageItemDTO:
        # An orphan tool row cannot be mapped to a ConversationItem and would
        # surface as a 500 from deep inside extraction; refuse it here (422).
        if self.role == "tool" and not self.tool_call_id:
            raise ValueError("tool_call_id is required when role is 'tool'")
        return self


class MemorizeAddRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=128)
    app_id: AppScopeId = Field(
        default="default",
        min_length=1,
        max_length=128,
        pattern=PATH_SAFE_CHARSET,
    )
    project_id: ProjectScopeId = Field(
        default="default",
        min_length=1,
        max_length=128,
        pattern=PATH_SAFE_CHARSET,
    )
    messages: list[MessageItemDTO] = Field(..., min_length=1, max_length=500)
    defer_extraction: bool = Field(
        default=False,
        description=(
            "Persist messages in the durable unprocessed buffer without running "
            "boundary detection or extraction. A later /flush commits the batch."
        ),
    )
    async_mode: bool = Field(
        default=False,
        description=(
            "Persist the batch to the memorize queue and return 202 at once "
            "instead of running boundary detection and extraction inline. Poll "
            "GET /tasks/{request_id}. A session that already "
            "has queued batches is queued even when this is false, so batches "
            "never run out of order."
        ),
    )


class AddResponseData(BaseModel):
    message_count: int
    status: Literal["accumulated", "extracted"]


class AcceptedResponseData(BaseModel):
    """202 body: the batch is persisted and queued for the memorize worker."""

    status: Literal["queued"] = "queued"
    message_count: int
    pending: int = Field(
        description="Unfinished queued batches of this session, this one included"
    )


class MemorizeFlushRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=128)
    app_id: AppScopeId = Field(
        default="default",
        min_length=1,
        max_length=128,
        pattern=PATH_SAFE_CHARSET,
    )
    project_id: ProjectScopeId = Field(
        default="default",
        min_length=1,
        max_length=128,
        pattern=PATH_SAFE_CHARSET,
    )


class FlushResponseData(BaseModel):
    status: Literal["extracted", "no_extraction"]

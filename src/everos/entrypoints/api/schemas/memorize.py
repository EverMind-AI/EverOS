"""Request and response models for ``/memory/add`` and ``/memory/flush``.

DTOs follow the v1 API brief (01_v1_api_brief.md §2 / §3).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from everos.core.utils.datetime import MAX_EPOCH_MS

from .groups import RosterEventDTO
from .ids import (
    PATH_SAFE_CHARSET,
    AppScopeId,
    PathSafeId,
    ProjectScopeId,
    SessionId,
)


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
        le=MAX_EPOCH_MS,
        description=(
            "Message event time as Unix epoch in **milliseconds** (the "
            "contract). For backward compatibility a value below 10^10 is "
            "read as seconds (up to 2286-11-20); every larger value is "
            "milliseconds, so an instant from 1970-04-26 on is sent in "
            "milliseconds, before 2001-09-09 too."
        ),
    )
    content: str | list[ContentItemDTO]
    tool_calls: list[ToolCallDTO] | None = None
    tool_call_id: str | None = None
    message_id: SessionId | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        description=(
            "Optional client id of the message: the chat platform's id, unique "
            "within the group. Kept for a group session (message-level forget, "
            "source quotes) and what the group dedupes by -- a message whose "
            "id the group already holds, in any of its sessions, is skipped; "
            "ignored on a one-to-one session."
        ),
    )

    @model_validator(mode="after")
    def _tool_row_needs_call_id(self) -> MessageItemDTO:
        # An orphan tool row cannot be mapped to a ConversationItem and would
        # surface as a 500 from deep inside extraction; refuse it here (422).
        if self.role == "tool" and not self.tool_call_id:
            raise ValueError("tool_call_id is required when role is 'tool'")
        return self


class MemorizeAddRequest(BaseModel):
    session_id: SessionId = Field(..., min_length=1, max_length=128)
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
    group_id: PathSafeId | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        pattern=PATH_SAFE_CHARSET,
        description=(
            "The group the session belongs to. The first batch naming it binds "
            "the session; a later batch without it inherits the binding; a "
            "batch naming another group is 409. An unknown group is created "
            "([groups] implicit_create)."
        ),
    )
    roster: list[RosterEventDTO] = Field(
        default_factory=list,
        max_length=500,
        description=(
            "Roster events applied before the messages (an event without `at` "
            "takes the batch's first message time). For an existing group the "
            "asker must be an owner or admin."
        ),
    )
    asker_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        description=(
            "Who sends the batch; needed for roster events of an existing group, "
            "and for any batch to a group with a deliverer, which takes batches "
            "from its deliverer only (403 otherwise). The gateway's identity "
            "header, when configured, is the asker instead."
        ),
    )
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

    @model_validator(mode="after")
    def _group_fields(self) -> MemorizeAddRequest:
        if self.roster and self.group_id is None:
            raise ValueError("roster events need a group_id")
        ids = [m.message_id for m in self.messages if m.message_id is not None]
        if len(ids) != len(set(ids)):
            raise ValueError("message_id must be unique within the batch")
        return self


class AddResponseData(BaseModel):
    message_count: int
    status: Literal["accumulated", "extracted"]
    rejected: int = Field(
        default=0,
        description=(
            "Pictures refused at ingest: over a [images] limit (bytes, per "
            "message, per request), not a picture, or unreadable. The rest of "
            "their message is kept."
        ),
    )
    failed: int = Field(
        default=0,
        description=(
            "Non-text items that could not be read into text (a parser call "
            "that failed). The rest of their message is kept."
        ),
    )
    duplicates: int | None = Field(
        default=None,
        description=(
            "Messages of the batch that were sent before, skipped: ones its "
            "group already held (by message_id when one was sent, else by "
            "sender, timestamp and content), and ones its session already "
            "held, buffered or segmented (the same sender, timestamp and "
            "content). Present only when the batch had some."
        ),
    )


class AcceptedResponseData(BaseModel):
    """202 body: the batch is persisted and queued for the memorize worker."""

    status: Literal["queued"] = "queued"
    message_count: int
    pending: int = Field(
        description="Unfinished queued batches of this session, this one included"
    )


class MemorizeFlushRequest(BaseModel):
    session_id: SessionId = Field(..., min_length=1, max_length=128)
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
    asker_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        description=(
            "Who flushes; read only for a session of a group with a deliverer, "
            "which only its deliverer may flush (403 otherwise). The gateway's "
            "identity header, when configured, is the asker instead."
        ),
    )


class FlushResponseData(BaseModel):
    status: Literal["extracted", "no_extraction"]

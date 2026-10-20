"""POST /api/v2/memory/add and /api/v2/memory/flush.

DTOs follow the v1 API brief (01_v1_api_brief.md §2 / §3). Routes are
thin adapters: validate the DTO, dump to dict, hand to service. No
business logic lives here.

``/flush`` force-closes a session's open buffer tail. The cloud edition
exposes the same endpoint.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Request, Response
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator
from starlette.status import HTTP_202_ACCEPTED

from everos.entrypoints.api.utils import extract_request_id, memorize_queue
from everos.service import (
    MemorizeQueueWorker,
    memorize,
)

router = APIRouter(prefix="/memory", tags=["memory"])


# ── Path-safe identifier ────────────────────────────────────────────────────
# ``app_id`` / ``project_id`` / ``sender_id`` all become directory segments
# under the memory root (``sender_id`` flows through to ``owner_id`` and is
# joined into the daily-log write path), so they must reject ``.`` and ``..``
# (path traversal). The basic character whitelist is enforced via ``pattern``
# (pydantic_core uses the Rust regex engine, which does NOT support
# lookaround), and the two reserved tokens are filtered out with a follow-up
# ``AfterValidator``.
#
# ``@`` and ``+`` are admitted so real-world ids survive (email-style
# ``user@example.com``, plus-addressing ``user+tag``); both are legal,
# non-separator filename chars on every target filesystem (incl. NTFS, whose
# reserved set is ``< > : " / \ | ? *``). The genuinely path-dangerous chars
# (``/`` ``\`` NUL) stay out of the whitelist, and ``.``/``..`` stay blocked
# by the token filter; the markdown writer's ``_ensure_within_root`` is the
# final backstop regardless.
_PATH_SAFE_CHARSET = r"^[a-zA-Z0-9_.@+-]+$"
_PATH_TRAVERSAL_TOKENS = frozenset({".", ".."})


_PATH_SAFE_RE = re.compile(_PATH_SAFE_CHARSET)


def _reject_path_traversal(value: str) -> str:
    if value in _PATH_TRAVERSAL_TOKENS:
        raise ValueError("'.' and '..' are reserved (path traversal)")
    if not _PATH_SAFE_RE.match(value):
        raise ValueError(
            "Only alphanumerics, underscore, dot, hyphen, @, and + are allowed"
        )
    return value


def _reject_reserved_scope_dir(reserved: str) -> Callable[[str], str]:
    """Refuse the on-disk directory name of the default scope as that scope's id.

    ``app_dir_name("default")`` is ``default_app`` and ``app_id_from_dir``
    maps it back to ``default``; an app whose id is literally ``default_app``
    would share the default app's directory and have its index rows
    attributed to ``default``. The same for ``project_id`` and
    ``default_project``. Only that field/name pair aliases, so only it is
    refused: ``sender_id`` never goes through the scope mapping, and
    ``app_id="default_project"`` round-trips unchanged.
    """

    def check(value: str) -> str:
        _reject_path_traversal(value)
        if value == reserved:
            raise ValueError(
                f"{reserved!r} is the directory name of the default scope; "
                "use 'default'"
            )
        return value

    return check


PathSafeId = Annotated[str, AfterValidator(_reject_path_traversal)]
AppScopeId = Annotated[str, AfterValidator(_reject_reserved_scope_dir("default_app"))]
ProjectScopeId = Annotated[
    str, AfterValidator(_reject_reserved_scope_dir("default_project"))
]


# DTOs ────────────────────────────────────────────────────────────────────────


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
        pattern=_PATH_SAFE_CHARSET,
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
        pattern=_PATH_SAFE_CHARSET,
    )
    project_id: ProjectScopeId = Field(
        default="default",
        min_length=1,
        max_length=128,
        pattern=_PATH_SAFE_CHARSET,
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
        pattern=_PATH_SAFE_CHARSET,
    )
    project_id: ProjectScopeId = Field(
        default="default",
        min_length=1,
        max_length=128,
        pattern=_PATH_SAFE_CHARSET,
    )


class FlushResponseData(BaseModel):
    status: Literal["extracted", "no_extraction"]


class SuccessEnvelope[T](BaseModel):
    """200 wrapper: ``request_id`` sits at the top level, not inside ``data``."""

    request_id: str
    data: T


# Route ──────────────────────────────────────────────────────────────────────


async def _queue_for(
    request: Request,
    req: MemorizeAddRequest | MemorizeFlushRequest,
    *,
    async_mode: bool,
) -> MemorizeQueueWorker | None:
    """The worker to enqueue on, or ``None`` when the call runs inline.

    Queued when the caller asked for ``async_mode`` (and the queue accepts
    new async work) **or** the session already has unfinished queued rows --
    running inline then would overtake batches the client was already told
    are accepted. ``/flush`` has no ``async_mode`` (the Cloud API has none
    either): it only ever queues behind its session's backlog.
    """
    queue = memorize_queue(request)
    if queue is None:
        return None
    if (async_mode and queue.accepting) or await queue.has_pending(
        session_id=req.session_id, app_id=req.app_id, project_id=req.project_id
    ):
        return queue
    return None


async def _accept(
    queue: MemorizeQueueWorker,
    response: Response,
    payload: dict[str, Any],
    *,
    request_id: str,
    is_final: bool,
    defer_extraction: bool,
) -> SuccessEnvelope[AcceptedResponseData]:
    pending = await queue.enqueue(
        payload,
        request_id=request_id,
        is_final=is_final,
        defer_extraction=defer_extraction,
    )
    response.status_code = HTTP_202_ACCEPTED
    return SuccessEnvelope(
        request_id=request_id,
        data=AcceptedResponseData(
            message_count=len(payload["messages"]), pending=pending
        ),
    )


_ADD_RESPONSES: dict[int | str, dict[str, Any]] = {
    202: {
        "model": SuccessEnvelope[AcceptedResponseData],
        "description": "Queued: async_mode, or the session already had a backlog",
    },
    413: {"description": "Body over memorize.async_queue.max_payload_bytes"},
    429: {"description": "Queue at its per-session or global cap"},
}
_FLUSH_RESPONSES: dict[int | str, dict[str, Any]] = {
    202: {
        "model": SuccessEnvelope[AcceptedResponseData],
        "description": "Queued behind the session's backlog, how a flush is queued",
    },
    429: {"description": "Queue at its per-session or global cap"},
}


@router.post("/add", responses=_ADD_RESPONSES)
async def add_memory(
    req: Annotated[MemorizeAddRequest, ...],
    request: Request,
    response: Response,
) -> SuccessEnvelope[AddResponseData] | SuccessEnvelope[AcceptedResponseData]:
    """Add messages into the user-memory + agent-memory pipelines."""
    request_id = extract_request_id(request)
    payload = req.model_dump(exclude={"async_mode", "defer_extraction"})
    queue = await _queue_for(request, req, async_mode=req.async_mode)
    if queue is not None:
        return await _accept(
            queue,
            response,
            payload,
            request_id=request_id,
            is_final=False,
            defer_extraction=req.defer_extraction,
        )
    result = await memorize(payload, defer_extraction=req.defer_extraction)
    return SuccessEnvelope(
        request_id=request_id,
        data=AddResponseData(
            message_count=result.message_count,
            status=result.status,
        ),
    )


@router.post("/flush", responses=_FLUSH_RESPONSES)
async def flush_memory(
    req: Annotated[MemorizeFlushRequest, ...],
    request: Request,
    response: Response,
) -> SuccessEnvelope[FlushResponseData] | SuccessEnvelope[AcceptedResponseData]:
    """Force boundary detection over the current ``session_id`` buffer.

    Runs inline unless the session still has queued batches, in which case
    the flush joins the queue behind them and answers 202.
    """
    request_id = extract_request_id(request)
    payload: dict[str, Any] = {
        "session_id": req.session_id,
        "app_id": req.app_id,
        "project_id": req.project_id,
        "messages": [],
    }
    queue = await _queue_for(request, req, async_mode=False)
    if queue is not None:
        return await _accept(
            queue,
            response,
            payload,
            request_id=request_id,
            is_final=True,
            defer_extraction=False,
        )
    result = await memorize(payload, is_final=True)
    # service's ``accumulated`` = nothing to flush (buffer was empty);
    # ``extracted`` = at least one cell carved out.
    status: Literal["extracted", "no_extraction"] = (
        "extracted" if result.status == "extracted" else "no_extraction"
    )
    return SuccessEnvelope(
        request_id=request_id,
        data=FlushResponseData(status=status),
    )

"""POST /api/v1/memory/add and /api/v1/memory/flush.

DTOs follow the v1 API brief (01_v1_api_brief.md §2 / §3). Routes are
thin adapters: validate the DTO, dump to dict, hand to service. No
business logic lives here.

``/flush`` is OSS-only (the cloud edition decides boundary timing
server-side and does not expose this endpoint).
"""

from __future__ import annotations

import re
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Request
from pydantic import AfterValidator, BaseModel, ConfigDict, Field

from everos.entrypoints.api.utils import extract_request_id
from everos.service import memorize

router = APIRouter(prefix="/api/v1/memory", tags=["memory"])


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


PathSafeId = Annotated[str, AfterValidator(_reject_path_traversal)]


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


class MemorizeAddRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=128)
    app_id: PathSafeId = Field(
        default="default",
        min_length=1,
        max_length=128,
        pattern=_PATH_SAFE_CHARSET,
    )
    project_id: PathSafeId = Field(
        default="default",
        min_length=1,
        max_length=128,
        pattern=_PATH_SAFE_CHARSET,
    )
    messages: list[MessageItemDTO] = Field(..., min_length=1, max_length=500)


class AddResponseData(BaseModel):
    message_count: int
    status: Literal["accumulated", "extracted"]


class MemorizeFlushRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=128)
    app_id: PathSafeId = Field(
        default="default",
        min_length=1,
        max_length=128,
        pattern=_PATH_SAFE_CHARSET,
    )
    project_id: PathSafeId = Field(
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


@router.post("/add")
async def add_memory(
    req: Annotated[MemorizeAddRequest, ...],
    request: Request,
) -> SuccessEnvelope[AddResponseData]:
    """Add messages into the user-memory + agent-memory pipelines."""
    request_id = extract_request_id(request)
    result = await memorize(req.model_dump())
    return SuccessEnvelope(
        request_id=request_id,
        data=AddResponseData(
            message_count=result.message_count,
            status=result.status,
        ),
    )


@router.post("/flush")
async def flush_memory(
    req: Annotated[MemorizeFlushRequest, ...],
    request: Request,
) -> SuccessEnvelope[FlushResponseData]:
    """Force boundary detection over the current ``session_id`` buffer.

    [OSS-only] — cloud edition decides boundary timing server-side and
    does not expose this endpoint.
    """
    request_id = extract_request_id(request)
    result = await memorize(
        {
            "session_id": req.session_id,
            "app_id": req.app_id,
            "project_id": req.project_id,
            "messages": [],
        },
        is_final=True,
    )
    # service's ``accumulated`` = nothing to flush (buffer was empty);
    # ``extracted`` = at least one cell carved out.
    status: Literal["extracted", "no_extraction"] = (
        "extracted" if result.status == "extracted" else "no_extraction"
    )
    return SuccessEnvelope(
        request_id=request_id,
        data=FlushResponseData(status=status),
    )

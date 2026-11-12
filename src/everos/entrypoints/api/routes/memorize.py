"""POST /api/v2/memory/add and /api/v2/memory/flush.

Routes are thin adapters: validate the DTO (``schemas.memorize``), dump
to dict, hand to service. No business logic lives here.

``/flush`` force-closes a session's open buffer tail. The cloud edition
exposes the same endpoint.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Response
from starlette.status import HTTP_202_ACCEPTED

from everos.entrypoints.api.dependencies import (
    ApplicationDep,
    AskerHeaderDep,
    RequestIdDep,
)
from everos.entrypoints.api.schemas import (
    AcceptedResponseData,
    AddResponseData,
    FlushResponseData,
    MemorizeAddRequest,
    MemorizeFlushRequest,
    SuccessEnvelope,
)
from everos.service import Queued

router = APIRouter(prefix="/memory", tags=["memory"])


def _accepted(
    response: Response, request_id: str, queued: Queued
) -> SuccessEnvelope[AcceptedResponseData]:
    response.status_code = HTTP_202_ACCEPTED
    return SuccessEnvelope(
        request_id=request_id,
        data=AcceptedResponseData(
            message_count=queued.message_count, pending=queued.pending
        ),
    )


_ADD_RESPONSES: dict[int | str, dict[str, Any]] = {
    202: {
        "model": SuccessEnvelope[AcceptedResponseData],
        "description": "Queued: async_mode, or the session already had a backlog",
    },
    403: {"description": "Roster events from an asker who may not send them"},
    409: {"description": "Session bound to another group, or a roster conflict"},
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


def _payload(req: MemorizeAddRequest, asker_header: str | None) -> dict[str, Any]:
    """The request as the service's payload dict.

    The group fields appear only when the request uses them, so a one-to-one
    payload (also what the ``async_mode`` queue persists) is exactly what it
    was before groups existed.
    """
    payload = req.model_dump(exclude={"async_mode", "defer_extraction"})
    for key in ("group_id", "roster", "asker_id"):
        if not payload.get(key):
            payload.pop(key, None)
    for message in payload["messages"]:
        if message.get("message_id") is None:
            message.pop("message_id", None)
    if asker_header is not None:
        payload["asserted_asker"] = asker_header
    return payload


@router.post("/add", responses=_ADD_RESPONSES)
async def add_memory(
    req: Annotated[MemorizeAddRequest, ...],
    response: Response,
    app: ApplicationDep,
    request_id: RequestIdDep,
    asker_header: AskerHeaderDep,
) -> SuccessEnvelope[AddResponseData] | SuccessEnvelope[AcceptedResponseData]:
    """Add messages into the conversation pipeline (chat memory).

    With ``group_id`` the session becomes (or already is) a group session:
    its messages become group memory, never per-sender personal memory.
    Without it, a session of two or more people (role ``user`` senders) is
    their implicit group; one person's conversation is their own memory.
    ``rejected`` / ``failed`` count the non-text items dropped at ingest; a
    queued batch (202) is ingested later and logs them
    (``memorize.ingest_dropped``).
    """
    result = await app.memory.add(
        _payload(req, asker_header),
        request_id=request_id,
        async_mode=req.async_mode,
        defer_extraction=req.defer_extraction,
    )
    if isinstance(result, Queued):
        return _accepted(response, request_id, result)
    return SuccessEnvelope(
        request_id=request_id,
        data=AddResponseData(
            message_count=result.message_count,
            status=result.status,
            rejected=result.rejected,
            failed=result.failed,
        ),
    )


@router.post("/flush", responses=_FLUSH_RESPONSES)
async def flush_memory(
    req: Annotated[MemorizeFlushRequest, ...],
    response: Response,
    app: ApplicationDep,
    request_id: RequestIdDep,
) -> SuccessEnvelope[FlushResponseData] | SuccessEnvelope[AcceptedResponseData]:
    """Force boundary detection over the current ``session_id`` buffer.

    Runs inline unless the session still has queued batches, in which case
    the flush joins the queue behind them and answers 202.
    """
    result = await app.memory.flush(
        session_id=req.session_id,
        request_id=request_id,
        app_id=req.app_id,
        project_id=req.project_id,
    )
    if isinstance(result, Queued):
        return _accepted(response, request_id, result)
    return SuccessEnvelope(
        request_id=request_id,
        data=FlushResponseData(status=result.status),
    )

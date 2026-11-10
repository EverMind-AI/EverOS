"""POST /api/v2/memory/delete and /delete_by_ids — delete memories."""

from __future__ import annotations

from fastapi import APIRouter

from everos.entrypoints.api.dependencies import ApplicationDep, RequestIdDep
from everos.entrypoints.api.schemas import (
    DeleteByIdsRequest,
    DeleteByIdsResponseData,
    DeleteRequest,
    DeleteResponseData,
    ErrorResponse,
    SuccessEnvelope,
)

router = APIRouter(prefix="/memory", tags=["memory"])

_RESPONSES: dict[int | str, dict[str, object]] = {
    422: {"description": "No scope, or both user_id and agent_id"},
    500: {
        "model": ErrorResponse,
        "description": "Deletion incomplete; send the same request again",
    },
    503: {
        "model": ErrorResponse,
        "description": "SESSION_BUSY: the session is still being memorized; "
        "nothing was deleted, retry later",
    },
}


@router.post("/delete", responses=_RESPONSES)
async def delete_memory(
    req: DeleteRequest, app: ApplicationDep, request_id: RequestIdDep
) -> SuccessEnvelope[DeleteResponseData]:
    """Delete one owner's memories, one session's, or one owner's of a session."""
    result = await app.memory.delete(
        app_id=req.app_id,
        project_id=req.project_id,
        user_id=req.user_id,
        agent_id=req.agent_id,
        session_id=req.session_id,
    )
    return SuccessEnvelope(
        request_id=request_id,
        data=DeleteResponseData(filters=result.filters, count=result.count),
    )


@router.post("/delete_by_ids", responses={500: _RESPONSES[500]})
async def delete_memories_by_ids(
    req: DeleteByIdsRequest, app: ApplicationDep, request_id: RequestIdDep
) -> SuccessEnvelope[DeleteByIdsResponseData]:
    """Delete episodes by id, with what was merged into them and their facts.

    Ids that match nothing are skipped (not 404); sending the same request
    again is harmless.
    """
    deleted = await app.memory.delete_by_ids(
        app_id=req.app_id,
        project_id=req.project_id,
        memory_type=req.memory_type,
        ids=req.memory_ids,
        user_id=req.user_id,
        reason_code=req.reason.code if req.reason else None,
        note=req.reason.note if req.reason else None,
    )
    return SuccessEnvelope(
        request_id=request_id, data=DeleteByIdsResponseData(deleted=deleted)
    )

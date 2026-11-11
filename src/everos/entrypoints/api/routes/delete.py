"""POST /api/v2/memory/delete and /delete_by_ids — delete memories.

A delete naming ``group_id`` is the group's: an owner's or admin's forget,
soft and propagated (docs/group-memory.md, section 15).
"""

from __future__ import annotations

from fastapi import APIRouter

from everos.entrypoints.api.dependencies import (
    ApplicationDep,
    AskerHeaderDep,
    RequestIdDep,
)
from everos.entrypoints.api.schemas import (
    DeleteByIdsRequest,
    DeleteByIdsResponseData,
    DeleteRequest,
    DeleteResponseData,
    ErrorResponse,
    GroupDeleteResponseData,
    SuccessEnvelope,
)

router = APIRouter(prefix="/memory", tags=["memory"])

_RESPONSES: dict[int | str, dict[str, object]] = {
    403: {
        "model": ErrorResponse,
        "description": "Group-scoped: the asker is not an owner or admin",
    },
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
    req: DeleteRequest,
    app: ApplicationDep,
    request_id: RequestIdDep,
    asker_header: AskerHeaderDep,
) -> SuccessEnvelope[DeleteResponseData | GroupDeleteResponseData]:
    """Delete one owner's memories, one session's, or one owner's of a session.

    With ``group_id`` it is the group's delete (docs/group-memory.md, section
    15): an owner's or admin's forget of one session or one member's
    messages -- soft and propagated -- whose ``forget_id`` undoes it.
    """
    if req.group_id is not None:
        deleted, forgotten = await app.memory.delete_in_group(
            app_id=req.app_id,
            project_id=req.project_id,
            group_id=req.group_id,
            asker_id=req.asker_id,
            session_id=req.session_id,
            member_id=req.user_id,
            asserted_asker=asker_header,
        )
        return SuccessEnvelope(
            request_id=request_id,
            data=GroupDeleteResponseData(
                filters=deleted.filters,
                count=deleted.count,
                forget_id=forgotten.forget_id,
            ),
        )
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
    await app.memory.delete_by_ids(
        app_id=req.app_id,
        project_id=req.project_id,
        memory_type=req.memory_type,
        ids=req.memory_ids,
        reason_code=req.reason.code if req.reason else None,
        note=req.reason.note if req.reason else None,
    )
    return SuccessEnvelope(request_id=request_id, data=DeleteByIdsResponseData())

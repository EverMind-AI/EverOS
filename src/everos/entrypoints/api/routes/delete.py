"""POST /api/v2/memory/delete and /delete_by_ids -- forgets, in EverOS 2.0.

2.0 has no hard delete: both routes are kept for the callers that send them
and answer with a forget of the same scope, run at once and reversible by the
``forget_ids`` they return (``POST /memory/forget/undo``). A delete naming
``group_id`` is the group's: an owner's or admin's forget, soft and
propagated (docs/group-memory.md, section 15). An agent's memory has no forget
yet, so a delete naming ``agent_id`` is refused (422).
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
    422: {
        "description": "No scope, both user_id and agent_id, an agent's memory "
        "(it has no forget), or a session bound to a group without group_id"
    },
    500: {
        "model": ErrorResponse,
        "description": "Forget incomplete; send the same request again",
    },
    503: {
        "model": ErrorResponse,
        "description": "SESSION_BUSY: the session is still being memorized; "
        "nothing was forgotten, retry later",
    },
}


@router.post("/delete", responses=_RESPONSES)
async def delete_memory(
    req: DeleteRequest,
    app: ApplicationDep,
    request_id: RequestIdDep,
    asker_header: AskerHeaderDep,
) -> SuccessEnvelope[DeleteResponseData | GroupDeleteResponseData]:
    """Forget one user's memories, one session's, or one user's of a session.

    EverOS 2.0 has no hard delete: this is ``POST /memory/forget`` with an
    ``owner`` or ``session`` target, run at once. Nothing leaves markdown; the
    answer's ``forget_ids`` (one per user reached) undo it. With ``group_id``
    it is the group's delete (docs/group-memory.md, section 15): an owner's or
    admin's forget of one session or one member's messages -- soft and
    propagated -- whose ``forget_id`` undoes it.
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
        data=DeleteResponseData(
            filters=result.filters,
            count=result.count,
            forget_ids=list(result.forget_ids),
        ),
    )


@router.post("/delete_by_ids", responses={500: _RESPONSES[500]})
async def delete_memories_by_ids(
    req: DeleteByIdsRequest, app: ApplicationDep, request_id: RequestIdDep
) -> SuccessEnvelope[DeleteByIdsResponseData]:
    """Forget episodes by id (EverOS 2.0 has no hard delete).

    ``POST /memory/forget`` with a ``record`` target, per owner, run at once;
    the answer's ``forget_ids`` undo it. Ids that match nothing are skipped
    (not 404); sending the same request again forgets nothing more.
    """
    forget_ids = await app.memory.delete_by_ids(
        app_id=req.app_id,
        project_id=req.project_id,
        memory_type=req.memory_type,
        ids=req.memory_ids,
        reason_code=req.reason.code if req.reason else None,
    )
    return SuccessEnvelope(
        request_id=request_id,
        data=DeleteByIdsResponseData(forget_ids=list(forget_ids)),
    )

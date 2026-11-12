"""``POST /memory/update`` and ``GET /memory/history``.

A change of a memory is a new version linked to the old one; the history is
every version of it (docs/api.md). Protocol translation only.
"""

from __future__ import annotations

import datetime as _dt
from typing import Annotated

from fastapi import APIRouter, Query

from everos.core.errors import InvalidInputError
from everos.core.utils.datetime import from_timestamp_ms
from everos.entrypoints.api.dependencies import (
    ApplicationDep,
    AskerHeaderDep,
    RequestIdDep,
)
from everos.entrypoints.api.schemas import (
    ErrorResponse,
    MemoryHistoryResponseData,
    MemoryVersionData,
    SuccessEnvelope,
    UpdateMemoryRequest,
    UpdateMemoryResponseData,
)
from everos.service import MemoryChanges, MemoryHistory

router = APIRouter(prefix="/memory", tags=["memory"])

_ID_PATTERN = r"^[a-zA-Z0-9_.@+-]{1,300}$"
_SCOPE_PATTERN = r"^[a-zA-Z0-9_.@+-]+$"


@router.post(
    "/update",
    responses={
        403: {
            "model": ErrorResponse,
            "description": "Someone else's personal memory; a group asker who may "
            "not change the record",
        },
        404: {"model": ErrorResponse, "description": "No such memory"},
        409: {
            "model": ErrorResponse,
            "description": "VERSION_CONFLICT: expected_version is not the current "
            "version (the message names it); MEMORY_FORGOTTEN: the memory is "
            "forgotten, undo the forget first",
        },
    },
)
async def update_memory(
    req: UpdateMemoryRequest,
    app: ApplicationDep,
    request_id: RequestIdDep,
    asker_header: AskerHeaderDep,
) -> SuccessEnvelope[UpdateMemoryResponseData]:
    """Change one memory: write a new version that supersedes the one read.

    Nothing is edited in place: the new version names the old one
    (``supersedes``) and who wrote it, the old one stays readable through
    ``GET /memory/history``, and only the new one is searchable once the
    index has it. The new version is held (the model's later updates leave
    it as it is) unless ``hold`` is false.
    """
    result = await app.memory.update_memory(
        app_id=req.app_id,
        project_id=req.project_id,
        user_id=req.user_id,
        group_id=req.group_id,
        asker_id=req.asker_id,
        asserted_asker=asker_header,
        record_id=req.id,
        expected_version=req.expected_version,
        changes=MemoryChanges(
            subject=req.subject,
            summary=req.summary,
            content=req.content,
            description=req.description,
            ocr=req.ocr,
        ),
        hold=req.hold,
    )
    return SuccessEnvelope(
        request_id=request_id,
        data=UpdateMemoryResponseData(
            kind=result.kind,
            id=result.id,
            supersedes=result.supersedes,
            changed=result.changed,
            held=result.held,
            edited_by=result.edited_by,
            edited_at=_time(result.edited_at),
            member_id=result.member_id,
        ),
    )


@router.get(
    "/history",
    responses={
        403: {
            "model": ErrorResponse,
            "description": "Someone else's personal memory; not a member of the group",
        },
        404: {
            "model": ErrorResponse,
            "description": "No such memory, or none of its versions the asker may see",
        },
    },
)
async def memory_history(
    app: ApplicationDep,
    request_id: RequestIdDep,
    asker_header: AskerHeaderDep,
    record_id: Annotated[
        str,
        Query(
            alias="id",
            pattern=_ID_PATTERN,
            description="Any version's id, as a read returned it.",
        ),
    ],
    app_id: Annotated[str, Query(pattern=_SCOPE_PATTERN, max_length=128)] = "default",
    project_id: Annotated[
        str, Query(pattern=_SCOPE_PATTERN, max_length=128)
    ] = "default",
    user_id: Annotated[
        str | None, Query(pattern=_SCOPE_PATTERN, max_length=128)
    ] = None,
    group_id: Annotated[
        str | None, Query(pattern=_SCOPE_PATTERN, max_length=128)
    ] = None,
    asker_id: Annotated[
        str | None,
        Query(
            pattern=_SCOPE_PATTERN,
            max_length=128,
            description="Who asks; required for a group unless the identity "
            "header names them.",
        ),
    ] = None,
) -> SuccessEnvelope[MemoryHistoryResponseData]:
    """Every version of one memory, current first.

    Each version says its state (``current``, ``superseded``, ``merged``,
    ``forgotten``), its time, who wrote it when a person did, what it rests
    on, and the version that replaced it (``next``). Inside a group only the
    versions the asker may see.
    """
    if (user_id is None) == (group_id is None):
        raise InvalidInputError("name exactly one of user_id and group_id")
    history = await app.memory.memory_history(
        app_id=app_id,
        project_id=project_id,
        user_id=user_id,
        group_id=group_id,
        asker_id=asker_id,
        asserted_asker=asker_header,
        record_id=record_id,
    )
    return SuccessEnvelope(request_id=request_id, data=_history_data(history))


def _history_data(history: MemoryHistory) -> MemoryHistoryResponseData:
    return MemoryHistoryResponseData(
        kind=history.kind,
        id=history.id,
        current=history.current,
        member_id=history.member_id,
        versions=[
            MemoryVersionData(
                id=v.id,
                state=v.state,
                timestamp=_time(v.timestamp),
                next=v.next,
                supersedes=v.supersedes,
                edited_by=v.edited_by,
                edited_at=_time(v.edited_at),
                held=v.held,
                replaced_at=_time(v.replaced_at),
                reason=v.reason,
                sources=dict(v.sources),
                content=dict(v.content),
            )
            for v in history.versions
        ],
    )


def _time(epoch_ms: int | None) -> _dt.datetime | None:
    return from_timestamp_ms(epoch_ms) if epoch_ms is not None else None

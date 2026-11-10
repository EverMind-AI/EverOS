"""The group routes: create or update a group, change its roster, read it.

``PUT /groups/{group_id}``, ``POST /groups/{group_id}/roster`` and ``GET
/groups/{group_id}`` (docs/group-memory.md, section 5.1). Protocol translation
only: the asker comes from the body or the gateway's identity header, and the
service decides everything else -- who may, what folds, what a member sees.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query

from everos.entrypoints.api.dependencies import (
    ApplicationDep,
    AskerHeaderDep,
    RequestIdDep,
)
from everos.entrypoints.api.schemas import (
    ErrorResponse,
    GroupDTO,
    GroupUpsertRequest,
    GroupUpsertResponseData,
    MemberIntervalDTO,
    RosterChangeRequest,
    RosterChangeResponseData,
    RosterEventDTO,
    SuccessEnvelope,
)
from everos.entrypoints.api.schemas.ids import PathSafeId
from everos.service import GroupView, RosterEventInput

router = APIRouter(prefix="/groups", tags=["groups"])

_RESPONSES: dict[int | str, dict[str, Any]] = {
    403: {
        "model": ErrorResponse,
        "description": "Not a current member, or not allowed to change this",
    },
    409: {"model": ErrorResponse, "description": "A roster event conflicts"},
}


def _events(events: list[RosterEventDTO]) -> list[RosterEventInput]:
    return [RosterEventInput(**event.model_dump()) for event in events]


def _group(view: GroupView) -> GroupDTO:
    return GroupDTO(
        group_id=view.group_id,
        app_id=view.app_id,
        project_id=view.project_id,
        name=view.name,
        description=view.description,
        created_at=view.created_at,
        created_by=view.created_by,
        history_visible_from=view.history_visible_from,
        extra=dict(view.extra),
        roster=[MemberIntervalDTO(**interval) for interval in view.roster],
    )


@router.put("/{group_id}", responses=_RESPONSES)
async def upsert_group(
    group_id: PathSafeId,
    req: GroupUpsertRequest,
    app: ApplicationDep,
    request_id: RequestIdDep,
    asker_header: AskerHeaderDep,
) -> SuccessEnvelope[GroupUpsertResponseData]:
    """Create the group, or update its meta (owners and admins).

    A new group whose roster names no owner gets the asker as its owner.
    """
    result = await app.memory.upsert_group(
        app_id=req.app_id,
        project_id=req.project_id,
        group_id=group_id,
        asker_id=req.asker_id,
        asserted_asker=asker_header,
        name=req.name,
        description=req.description,
        created_at=req.created_at,
        history_visible_from=req.history_visible_from,
        extra=req.extra,
        roster=_events(req.roster),
    )
    return SuccessEnvelope(
        request_id=request_id,
        data=GroupUpsertResponseData(
            created=result.created,
            events_applied=result.events_applied,
            group=_group(result.group),
        ),
    )


@router.post("/{group_id}/roster", responses=_RESPONSES)
async def change_roster(
    group_id: PathSafeId,
    req: RosterChangeRequest,
    app: ApplicationDep,
    request_id: RequestIdDep,
    asker_header: AskerHeaderDep,
) -> SuccessEnvelope[RosterChangeResponseData]:
    """Append roster events; owners and admins send any, a member their own leave.

    Messages the group's sessions still hold in their buffers are cut into
    segments first, so no segment spans the change.
    """
    result = await app.memory.change_roster(
        app_id=req.app_id,
        project_id=req.project_id,
        group_id=group_id,
        asker_id=req.asker_id,
        asserted_asker=asker_header,
        events=_events(req.events),
    )
    return SuccessEnvelope(
        request_id=request_id,
        data=RosterChangeResponseData(
            events_applied=result.events_applied, group=_group(result.group)
        ),
    )


@router.get("/{group_id}", responses={403: _RESPONSES[403]})
async def read_group(
    group_id: PathSafeId,
    app: ApplicationDep,
    request_id: RequestIdDep,
    asker_header: AskerHeaderDep,
    app_id: str = Query(default="default"),
    project_id: str = Query(default="default"),
    asker_id: str | None = Query(default=None),
    include_history: bool = Query(default=False),
) -> SuccessEnvelope[GroupDTO]:
    """The group's meta and current roster; every interval with history."""
    view = await app.memory.read_group(
        app_id=app_id,
        project_id=project_id,
        group_id=group_id,
        asker_id=asker_id,
        asserted_asker=asker_header,
        include_history=include_history,
    )
    return SuccessEnvelope(request_id=request_id, data=_group(view))

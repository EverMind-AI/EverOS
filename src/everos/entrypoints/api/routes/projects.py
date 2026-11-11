"""The project routes: the asker's view of a project (docs/group-memory.md, 8.7).

``GET /projects/{project_id}`` lists the groups of the project the asker is a
current member of, with their members; ``GET /projects/{project_id}/profile``
the visible current group-profile items of those groups, each tagged with its
group. Protocol translation only: the asker comes from the query or the
gateway's identity header, and the service decides what they may see.
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
    ProjectDTO,
    ProjectGroupDTO,
    ProjectMemberDTO,
    ProjectProfileDTO,
    SuccessEnvelope,
)
from everos.entrypoints.api.schemas.ids import PathSafeId

router = APIRouter(prefix="/projects", tags=["projects"])

_RESPONSES: dict[int | str, dict[str, Any]] = {
    403: {"model": ErrorResponse, "description": "In none of the project's groups"},
}


@router.get("/{project_id}", responses=_RESPONSES)
async def read_project(
    project_id: PathSafeId,
    app: ApplicationDep,
    request_id: RequestIdDep,
    asker_header: AskerHeaderDep,
    app_id: str = Query(default="default"),
    asker_id: str | None = Query(default=None),
) -> SuccessEnvelope[ProjectDTO]:
    """The asker's groups of the project and their current members."""
    view = await app.memory.read_project(
        app_id=app_id,
        project_id=project_id,
        asker_id=asker_id,
        asserted_asker=asker_header,
    )
    return SuccessEnvelope(
        request_id=request_id,
        data=ProjectDTO(
            app_id=view.app_id,
            project_id=view.project_id,
            groups=[ProjectGroupDTO(**group) for group in view.groups],
            members=[
                ProjectMemberDTO(
                    member_id=m.member_id,
                    display_name=m.display_name,
                    kind=m.kind,
                    group_ids=list(m.group_ids),
                )
                for m in view.members
            ],
        ),
    )


@router.get("/{project_id}/profile", responses=_RESPONSES)
async def read_project_profile(
    project_id: PathSafeId,
    app: ApplicationDep,
    request_id: RequestIdDep,
    asker_header: AskerHeaderDep,
    app_id: str = Query(default="default"),
    asker_id: str | None = Query(default=None),
) -> SuccessEnvelope[ProjectProfileDTO]:
    """The visible current group-profile items of the asker's groups."""
    profile = await app.memory.project_profile(
        app_id=app_id,
        project_id=project_id,
        asker_id=asker_id,
        asserted_asker=asker_header,
    )
    return SuccessEnvelope(
        request_id=request_id,
        data=ProjectProfileDTO(
            app_id=profile.app_id,
            project_id=profile.project_id,
            group_ids=list(profile.group_ids),
            items=list(profile.items),
        ),
    )

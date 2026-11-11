"""POST /api/v2/memory/purge, /purge/confirm and GET /purge/{job_id}."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Path

from everos.entrypoints.api.dependencies import ApplicationDep, RequestIdDep
from everos.entrypoints.api.schemas import (
    ErrorResponse,
    PurgeConfirmRequest,
    PurgeJobData,
    PurgeRequest,
    PurgeRequestData,
    SuccessEnvelope,
)

router = APIRouter(prefix="/memory", tags=["memory"])

_PathJobId = Annotated[
    str, Path(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
]


@router.post("/purge")
async def request_purge(
    req: PurgeRequest, app: ApplicationDep, request_id: RequestIdDep
) -> SuccessEnvelope[PurgeRequestData]:
    """Step 1 of 2: count what the scope holds and get a nonce.

    Nothing is deleted. The purge is irreversible once confirmed: ``space``
    clears the whole memory root, ``app`` one app, ``project`` one project.
    """
    result = await app.purge.request(req.scope, req.app_id, req.project_id)
    return SuccessEnvelope(
        request_id=request_id, data=PurgeRequestData.from_result(result)
    )


@router.post(
    "/purge/confirm",
    status_code=202,
    responses={
        404: {
            "model": ErrorResponse,
            "description": "PURGE_NONCE_INVALID: unknown, expired or used nonce",
        },
        503: {
            "model": ErrorResponse,
            "description": "The job could not be recorded; the nonce is still valid",
        },
    },
)
async def confirm_purge(
    req: PurgeConfirmRequest, app: ApplicationDep, request_id: RequestIdDep
) -> SuccessEnvelope[PurgeJobData]:
    """Step 2 of 2: start the purge the nonce was issued for.

    With a purge already running (any scope) this returns that job: compare
    its ``scope`` with the one you asked for.
    """
    job = await app.purge.confirm(req.nonce)
    return SuccessEnvelope(request_id=request_id, data=PurgeJobData.from_view(job))


@router.get(
    "/purge/{job_id}",
    responses={404: {"model": ErrorResponse, "description": "No such job"}},
)
async def get_purge(
    job_id: _PathJobId, app: ApplicationDep, request_id: RequestIdDep
) -> SuccessEnvelope[PurgeJobData]:
    """A purge job's status and per-store progress."""
    job = await app.purge.get(job_id)
    return SuccessEnvelope(request_id=request_id, data=PurgeJobData.from_view(job))

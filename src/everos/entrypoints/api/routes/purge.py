"""POST /api/v2/memory/purge, /purge/confirm and GET /purge/{job_id}."""

from __future__ import annotations

from typing import Annotated, Any
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, Path, Request

from everos.core.errors import CrossSiteRefusedError
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

REFUSED_DURING_PURGE: dict[int | str, dict[str, Any]] = {
    409: {
        "model": ErrorResponse,
        "description": "PURGE_IN_PROGRESS: a running purge is clearing this "
        "app / project; resend once the job named in the message has ended",
    },
}
"""The answer of a write route while a purge clears the range it names."""


def _refuse_cross_site(request: Request) -> None:
    """Refuse a purge sent by a browser page of another site.

    There is no authentication and CORS allows any origin by default, so any
    page the user opens could read a nonce and confirm it. Browsers send
    ``Origin`` with these requests; curl, SDKs, MCP and the plugins send
    none and pass, and so does a page this server serves itself.
    """
    origin = request.headers.get("origin")
    host = request.headers.get("host", "")
    if origin is not None and urlsplit(origin).netloc.lower() != host.lower():
        raise CrossSiteRefusedError(
            "a purge is refused from a browser page of another site; "
            "send it from curl, an SDK or a plugin"
        )


_CROSS_SITE_RESPONSE: dict[int | str, dict[str, Any]] = {
    403: {
        "model": ErrorResponse,
        "description": "CROSS_SITE_REFUSED: sent by a browser page of another site",
    },
}

_PathJobId = Annotated[
    str, Path(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
]


@router.post(
    "/purge",
    dependencies=[Depends(_refuse_cross_site)],
    responses=_CROSS_SITE_RESPONSE,
)
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
    dependencies=[Depends(_refuse_cross_site)],
    responses={
        **_CROSS_SITE_RESPONSE,
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

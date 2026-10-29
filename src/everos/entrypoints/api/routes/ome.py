"""OME trigger route — manually invoke a registered strategy."""

from __future__ import annotations

from fastapi import APIRouter

from everos.entrypoints.api.dependencies import ApplicationDep
from everos.entrypoints.api.schemas import RunSummary, TriggerRequest, TriggerResponse

router = APIRouter(prefix="/ome", tags=["ome"])


@router.post("/trigger", response_model=TriggerResponse)
async def trigger(req: TriggerRequest, app: ApplicationDep) -> TriggerResponse:
    """Manually trigger a registered OME strategy and wait for its runs to
    settle. Returns without waiting for the LanceDB index — the response
    reflects markdown state only; poll ``GET /health``'s ``cascade.pending``
    for index convergence (two consecutive zero samples to guard against the
    watcher-input window). See docs/api.md#eventual-consistency.
    """
    result = await app.ops.ome_trigger(
        req.name, force=req.force, wait_seconds=req.timeout
    )
    return TriggerResponse(
        status=result.status,
        name=req.name,
        dispatched=result.dispatched,
        runs=[
            RunSummary(run_id=r.run_id, status=r.status, error=r.error)
            for r in result.runs
        ],
    )

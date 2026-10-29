"""Health check route."""

from __future__ import annotations

import dataclasses

from fastapi import APIRouter

from everos import __version__
from everos.entrypoints.api.dependencies import ApplicationDep
from everos.entrypoints.api.schemas import (
    CascadeHealthBlock,
    HealthCapabilities,
    HealthResponse,
    MemorizeQueueHealthBlock,
)

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(app: ApplicationDep) -> HealthResponse:
    """Liveness + capabilities + cascade readiness probe.

    ``status`` stays ``"ok"`` whenever the process is up — the HTTP code
    is a *liveness* signal and a degraded cascade must not trigger a
    restart (a crash-loop fixes neither a bad md file nor disk bloat).
    The ``cascade`` block is the *readiness* signal: ``healthy=false``
    with human-readable ``reasons`` **only** when the projection pipeline
    itself is stuck (drain failing, optimize stuck, version cleanup
    stalled). ``failed_permanent`` — files awaiting ``cascade fix`` — is
    a data-quality backlog reported as an informational count that does
    not flip ``healthy``. Alert on ``cascade.healthy``.
    """
    report = await app.ops.health()
    return HealthResponse(
        status="ok",
        version=__version__,
        capabilities=HealthCapabilities(**dataclasses.asdict(report.providers)),
        disabled_features=report.disabled_features,
        cascade=(
            None
            if report.cascade is None
            else CascadeHealthBlock(**dataclasses.asdict(report.cascade))
        ),
        memorize_queue=(
            None
            if report.memorize_queue is None
            else MemorizeQueueHealthBlock(**report.memorize_queue.model_dump())
        ),
    )

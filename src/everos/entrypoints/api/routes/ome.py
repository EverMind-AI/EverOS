"""OME trigger route — manually invoke a registered strategy."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from everos.core.errors import NotFoundError
from everos.core.observability.logging import get_logger

router = APIRouter(prefix="/api/v1/ome", tags=["ome"])

logger = get_logger(__name__)


class TriggerRequest(BaseModel):
    """Request body for ``POST /api/v1/ome/trigger``."""

    name: str
    timeout: float = 120.0
    force: bool = False


class TriggerResponse(BaseModel):
    """Response body for ``POST /api/v1/ome/trigger``."""

    status: str
    name: str


@router.post("/trigger", response_model=TriggerResponse)
async def trigger(req: TriggerRequest) -> TriggerResponse:
    """Manually trigger a registered OME strategy and wait for completion."""
    # Deferred: avoid importing heavy OME engine at module level.
    from everos.service.memorize import _get_engine

    engine = _get_engine()
    try:
        await engine.trigger_manual(req.name, force=req.force)
    except KeyError:
        raise NotFoundError(f"strategy '{req.name}' not found") from None
    logger.info("ome_trigger_manual", strategy=req.name)
    idle = await engine.wait_idle(timeout=req.timeout)
    if not idle:
        logger.warning("ome_trigger_timeout", strategy=req.name, timeout=req.timeout)
        return TriggerResponse(status="timeout", name=req.name)
    return TriggerResponse(status="ok", name=req.name)

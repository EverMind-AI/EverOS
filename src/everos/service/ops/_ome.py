"""The OME use case: a manual strategy trigger, waited on (bounded)."""

from __future__ import annotations

__all__ = ["trigger"]

from typing import TYPE_CHECKING

from everos.core.errors import NotFoundError
from everos.core.observability.logging import get_logger

from ._models import OmeRun, OmeTriggerResult

if TYPE_CHECKING:
    from everos.infra.ome import OfflineEngine

logger = get_logger("everos.service.ops")


async def trigger(
    engine: OfflineEngine, name: str, *, force: bool, wait_seconds: float
) -> OmeTriggerResult:
    """Trigger a registered strategy and wait (bounded) for its runs.

    Raises:
        NotFoundError: No strategy is registered under ``name``.
    """
    try:
        event, routes = await engine.trigger_manual(name, force=force)
    except KeyError:
        raise NotFoundError(f"strategy '{name}' not found") from None

    if not routes:
        logger.info("ome_trigger_manual_not_dispatched", strategy=name)
        return OmeTriggerResult(status="not_dispatched", dispatched=0, runs=[])

    logger.info("ome_trigger_manual", strategy=name, dispatched=len(routes))
    idle = await engine.wait_idle(timeout=wait_seconds)
    records = await engine.list_runs_by_event_id(event.event_id)
    runs = [
        OmeRun(run_id=r.run_id, status=r.status.value, error=r.error) for r in records
    ]
    if not idle:
        logger.warning("ome_trigger_timeout", strategy=name, timeout=wait_seconds)
        return OmeTriggerResult(status="timeout", dispatched=len(routes), runs=runs)
    return OmeTriggerResult(status="ok", dispatched=len(routes), runs=runs)

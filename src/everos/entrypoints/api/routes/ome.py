"""OME trigger route — manually invoke a registered strategy."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter
from pydantic import BaseModel

from everos.core.errors import NotFoundError
from everos.core.observability.logging import get_logger

if TYPE_CHECKING:
    # Type-only — used solely to annotate `_summarize_runs`. Importing the
    # engine eagerly costs ~750ms (apscheduler + aiosqlite, 26 modules), so
    # keep it out of the runtime path even though `service.memorize` already
    # imports it eagerly and today's app startup pays that cost regardless:
    # this router's own `_get_engine` import is deliberately deferred (see
    # `trigger`), and an eager import here would contradict it.
    from everos.infra.ome.engine import OfflineEngine

router = APIRouter(prefix="/ome", tags=["ome"])

logger = get_logger(__name__)


class TriggerRequest(BaseModel):
    """Request body for ``POST /api/v2/ome/trigger``."""

    name: str
    timeout: float = 120.0
    force: bool = False


class RunSummary(BaseModel):
    """One strategy run within a trigger response."""

    run_id: str
    status: str
    """One of: running / success / failed / dead_letter / crashed."""
    error: str | None = None


class TriggerResponse(BaseModel):
    """Response body for ``POST /api/v2/ome/trigger``.

    ``status`` distinguishes three outcomes that were previously masked as
    a single ``ok``:

    - ``ok``: at least one strategy was dispatched and all runs settled
      within ``timeout``. Individual run outcomes are in ``runs`` (a
      ``dead_letter`` there is still ``ok`` at this level — the strategy
      *ran*, it just failed permanently).
    - ``timeout``: at least one strategy was dispatched but the engine did
      not go idle within ``timeout``. Runs may be partially complete;
      poll ``GET /health`` for cascade convergence separately.
    - ``not_dispatched``: no strategy was dispatched — the subscriber was
      rejected by one of the dispatch gates (``_routes_to`` / ``enabled`` /
      ``applies_to`` / ``Counter``). Common cause: forgetting
      ``force=true`` on a strategy that is ``enabled=false`` in ome.toml.
    """

    status: str
    """One of: ok / timeout / not_dispatched."""
    name: str
    dispatched: int
    """Number of strategy routes that were enqueued. ``0`` iff status is
    ``not_dispatched``."""
    runs: list[RunSummary] = []
    """One entry per strategy run *attempt*, not per dispatched route: a
    strategy that retried before settling contributes multiple entries
    sharing one ``event_id``. Includes dead-lettered runs whose errors
    would otherwise be invisible to the caller (they live in the SQLite
    ``run_record`` table with no HTTP surface until this field was
    added)."""


@router.post("/trigger", response_model=TriggerResponse)
async def trigger(req: TriggerRequest) -> TriggerResponse:
    """Manually trigger a registered OME strategy and wait for its runs to
    settle. Returns without waiting for the LanceDB index — the response
    reflects markdown state only; poll ``GET /health``'s ``cascade.pending``
    for index convergence (two consecutive zero samples to guard against the
    watcher-input window). See docs/api.md#eventual-consistency.
    """
    # Deferred: avoid importing heavy OME engine at module level.
    from everos.service.memorize import _get_engine

    engine = _get_engine()
    try:
        event, routes = await engine.trigger_manual(req.name, force=req.force)
    except KeyError:
        raise NotFoundError(f"strategy '{req.name}' not found") from None

    if not routes:
        logger.info("ome_trigger_manual_not_dispatched", strategy=req.name)
        return TriggerResponse(
            status="not_dispatched", name=req.name, dispatched=0, runs=[]
        )

    logger.info("ome_trigger_manual", strategy=req.name, dispatched=len(routes))
    idle = await engine.wait_idle(timeout=req.timeout)
    runs = await _summarize_runs(engine, event.event_id)
    if not idle:
        logger.warning("ome_trigger_timeout", strategy=req.name, timeout=req.timeout)
        return TriggerResponse(
            status="timeout", name=req.name, dispatched=len(routes), runs=runs
        )
    return TriggerResponse(
        status="ok", name=req.name, dispatched=len(routes), runs=runs
    )


async def _summarize_runs(engine: OfflineEngine, event_id: str) -> list[RunSummary]:
    """Fetch and shape run records for one event into the response DTO."""
    records = await engine.list_runs_by_event_id(event_id)
    return [
        RunSummary(run_id=r.run_id, status=r.status.value, error=r.error)
        for r in records
    ]

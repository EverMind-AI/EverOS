"""Memorize queue lifespan provider — starts/stops the ``async_mode`` worker.

Ordered after OME (``order=50``): the memorize pipelines emit through the
OME engine, and lifespans stop in reverse order, so a worker ordered before
OME would deliver into a stopped engine on shutdown and into a not-yet
started one on boot. ``[memorize.async_queue] enabled = false`` only stops
the route from accepting new ``async_mode`` work: the worker still runs, so
rows accepted before the switch are delivered and their sessions keep their
ordering guarantee.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI

from everos.config import load_settings
from everos.core.lifespan import LifespanProvider
from everos.core.observability.logging import get_logger
from everos.service import MemorizeQueueWorker

logger = get_logger(__name__)


class MemorizeQueueLifespanProvider(LifespanProvider):
    """Manage the ``async_mode`` memorize worker for the app lifecycle."""

    def __init__(self, order: int = 60) -> None:
        super().__init__(name="memorize_queue", order=order)
        self._worker: MemorizeQueueWorker | None = None

    async def startup(self, app: FastAPI) -> Any:
        cfg = load_settings().memorize.async_queue
        if not cfg.enabled:
            logger.info("memorize_queue_not_accepting_async")
        self._worker = MemorizeQueueWorker(cfg, accept_async=cfg.enabled)
        await self._worker.start()
        logger.info("memorize_queue_lifespan_ready")
        return self._worker

    async def shutdown(self, app: FastAPI) -> None:
        if self._worker is not None:
            await self._worker.stop()
            self._worker = None

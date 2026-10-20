"""Memorize lifespan provider — drains in-flight extractions at shutdown.

``memorize()`` lets the extraction of committed memcells outlive a cancelled
request (see :class:`everos.service.memorize.ExtractionSupervisor`). On
shutdown this provider waits for what is still running so a memcell does
not lose its derived memory to the process exit; the wait is bounded and
what is left is logged.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI

from everos.core.lifespan import LifespanProvider
from everos.core.observability.logging import get_logger
from everos.service.memorize import extractions

logger = get_logger(__name__)

_DRAIN_TIMEOUT_SECONDS = 30.0


class MemorizeLifespanProvider(LifespanProvider):
    """Wait for extractions that outlived their requests."""

    def __init__(self, order: int = 55) -> None:
        super().__init__(name="memorize", order=order)

    async def startup(self, app: FastAPI) -> Any:
        return None

    async def shutdown(self, app: FastAPI) -> None:
        running = extractions.in_flight
        if not running:
            return
        logger.info("memorize_extractions_draining", in_flight=running)
        left = await extractions.drain(_DRAIN_TIMEOUT_SECONDS)
        if left:
            logger.error(
                "memorize_extractions_abandoned",
                in_flight=left,
                waited_seconds=_DRAIN_TIMEOUT_SECONDS,
            )

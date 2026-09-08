"""OME engine lifespan provider (HTTP API entrypoint).

Startup: build the singleton engine via service.memorize._get_engine
(which also registers strategies) and start it.

Shutdown: stop the engine.
"""

from __future__ import annotations

import importlib
from typing import Any

from fastapi import FastAPI

from everos.core.lifespan import LifespanProvider
from everos.core.observability.logging import get_logger

logger = get_logger(__name__)


class OmeLifespanProvider(LifespanProvider):
    """Manage the OfflineEngine lifecycle for the FastAPI app."""

    def __init__(self, order: int = 50) -> None:
        super().__init__(name="ome", order=order)

    async def startup(self, app: FastAPI) -> Any:
        svc = importlib.import_module("everos.service.memorize")
        engine = svc._get_engine()
        await engine.start()
        logger.info("ome_engine_started")
        return engine

    async def shutdown(self, app: FastAPI) -> None:
        svc = importlib.import_module("everos.service.memorize")
        engine = svc._get_engine()
        await engine.stop()
        logger.info("ome_engine_stopped")

"""OME engine lifespan provider (HTTP API entrypoint).

Startup: build the singleton engine via service.memorize._get_engine
(which also registers strategies) and start it.

Shutdown: stop the engine.
"""

from __future__ import annotations

import importlib
import os
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
        # A read-only retrieval server does not extract, so the OfflineEngine -- and its
        # exclusive per-store lock -- is pure overhead. The lock also stops a second
        # server from sharing one pre-built store root, which is how a parallel-lane
        # evaluation is run. Off by default (unset), so an ingesting daemon that needs
        # extraction is unaffected. Mirrors ``EVEROS_DISABLE_CASCADE``.
        if os.getenv("EVEROS_DISABLE_OME", "").strip().lower() in (
            "1",
            "true",
            "yes",
        ):
            logger.info("ome_lifespan_disabled_by_env")
            return None
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

"""Tracing lifespan provider.

Builds the OpenTelemetry ``TracerProvider`` at startup (from the
``[observability]`` settings) and flushes + tears it down at shutdown.
Chassis-level (backend-agnostic), so it lives here alongside
``MetricsLifespanProvider`` rather than under the API entrypoint.

Registered with a low ``order`` so the tracer is live before other
providers start and can themselves be traced.
"""

from __future__ import annotations

from fastapi import FastAPI

from everos.config import load_settings
from everos.core.observability.logging import get_logger
from everos.core.observability.tracing import (
    init_score_sink,
    init_tracing,
    shutdown_score_sink,
    shutdown_tracing,
)

from .base import LifespanProvider

logger = get_logger(__name__)


class TracingLifespanProvider(LifespanProvider):
    """Manages the OTel tracer provider + recall-score sink over the app life."""

    def __init__(self, order: int = 1) -> None:
        super().__init__(name="tracing", order=order)

    async def startup(self, app: FastAPI) -> bool:
        """Install the tracer provider + recall-score sink when configured.

        Returns True if tracing was enabled and a provider installed.
        """
        settings = load_settings().observability
        enabled = init_tracing(settings)
        scores = await init_score_sink(settings)
        logger.info("tracing_lifespan_startup", enabled=enabled, scores=scores)
        return enabled

    async def shutdown(self, app: FastAPI) -> None:
        await shutdown_score_sink()
        shutdown_tracing()
        logger.info("tracing_lifespan_shutdown")

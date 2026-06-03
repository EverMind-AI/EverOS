"""Metrics lifespan provider.

Confirms the metrics registry is ready and logs that the ``/metrics`` HTTP
endpoint is mounted on the main API. Kept as a placeholder to demonstrate
the lifespan pattern; replace or extend with a standalone metrics server
(e.g. ``prometheus_client.start_http_server`` on a separate port) if you
need to expose metrics on a dedicated socket.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI

from everos.core.observability.logging import get_logger
from everos.core.observability.metrics import get_metrics_registry

from .base import LifespanProvider

logger = get_logger(__name__)


class MetricsLifespanProvider(LifespanProvider):
    """No-op startup that warms the metrics registry and logs readiness."""

    def __init__(self, order: int = 5) -> None:
        super().__init__(name="metrics", order=order)

    async def startup(self, app: FastAPI) -> Any:
        registry = get_metrics_registry()
        logger.info("metrics_registry_ready", endpoint="/metrics")
        return registry

    async def shutdown(self, app: FastAPI) -> None:
        logger.info("metrics_lifespan_shutdown")

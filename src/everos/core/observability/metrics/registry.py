"""The Prometheus registry every metric registers into and ``/metrics`` renders.

Process-wide by nature (a scrape reads one registry), so it is module state
-- but settled, not lazily built: it starts as ``prometheus_client.REGISTRY``
and :func:`configure_metrics` is the one place that may replace it.
"""

from __future__ import annotations

__all__ = ["configure_metrics", "generate_metrics_response", "get_metrics_registry"]

from prometheus_client import REGISTRY, CollectorRegistry, generate_latest

from everos.core.observability.logging import get_logger

logger = get_logger(__name__)

_registry: CollectorRegistry = REGISTRY


def get_metrics_registry() -> CollectorRegistry:
    """Return the registry metrics register into."""
    return _registry


def configure_metrics(registry: CollectorRegistry = REGISTRY) -> CollectorRegistry:
    """Settle the process-wide registry once, where the application is assembled.

    Metrics register at construction, so a registry configured after a metric
    was built does not see it; a test that wants an isolated registry
    configures one before building its metrics.

    Args:
        registry: Where metrics register; the ``prometheus_client`` default
            unless a caller (a test) wants its own.

    Returns:
        The configured registry.
    """
    global _registry
    _registry = registry
    logger.info("metrics_registry_ready", endpoint="/metrics")
    return registry


def generate_metrics_response() -> bytes:
    """Render the current registry into Prometheus exposition format."""
    return generate_latest(_registry)

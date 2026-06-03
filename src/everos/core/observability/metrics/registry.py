"""Prometheus metrics registry singleton."""

from __future__ import annotations

from prometheus_client import REGISTRY, CollectorRegistry, generate_latest

_registry: CollectorRegistry | None = None


def get_metrics_registry() -> CollectorRegistry:
    """Return the global metrics registry.

    Defaults to ``prometheus_client.REGISTRY``.
    """
    global _registry
    if _registry is None:
        _registry = REGISTRY
    return _registry


def set_metrics_registry(registry: CollectorRegistry) -> None:
    """Override the global registry (mainly for tests)."""
    global _registry
    _registry = registry


def generate_metrics_response() -> bytes:
    """Render the current registry into Prometheus exposition format."""
    return generate_latest(get_metrics_registry())


def reset_metrics_registry() -> None:
    """Reset the global registry override (mainly for tests)."""
    global _registry
    _registry = None

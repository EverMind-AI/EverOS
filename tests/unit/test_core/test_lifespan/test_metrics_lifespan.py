"""``MetricsLifespanProvider`` — startup returns registry, shutdown logs."""

from __future__ import annotations

from fastapi import FastAPI
from prometheus_client import CollectorRegistry

from everos.core.lifespan.metrics_lifespan import MetricsLifespanProvider
from everos.core.observability.metrics import (
    reset_metrics_registry,
    set_metrics_registry,
)


async def test_startup_returns_registry() -> None:
    fresh = CollectorRegistry()
    set_metrics_registry(fresh)
    try:
        p = MetricsLifespanProvider()
        result = await p.startup(FastAPI())
        assert result is fresh
    finally:
        reset_metrics_registry()


async def test_shutdown_is_noop() -> None:
    # Smoke test — must not raise.
    p = MetricsLifespanProvider()
    await p.shutdown(FastAPI())


def test_provider_metadata() -> None:
    p = MetricsLifespanProvider(order=42)
    assert p.name == "metrics"
    assert p.order == 42

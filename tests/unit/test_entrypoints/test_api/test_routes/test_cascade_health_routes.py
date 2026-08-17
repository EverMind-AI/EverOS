"""Cascade-health surfacing on ``GET /health``.

The route reads the running :class:`CascadeOrchestrator` off
``app.state.lifespan_data["cascade"]`` (stashed by the cascade
lifespan). These tests inject an autospec orchestrator so the route
logic is exercised without booting sqlite / lancedb / the worker.
Cascade degradation is surfaced *only* here (a readiness signal), never
on the write DTOs — a degraded projection does not change what ``/add``
returns.
"""

from __future__ import annotations

import unittest.mock as mock

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from everos.entrypoints.api.routes.health import router as health_router
from everos.memory.cascade import CascadeHealth, CascadeOrchestrator


def _healthy() -> CascadeHealth:
    return CascadeHealth(
        healthy=True,
        reasons=[],
        pending=0,
        failed_permanent=0,
        failed_retryable=0,
        drain_consecutive_failures=0,
        unrecoverable_total=0,
        optimize_failure_streak=0,
        prune_stale_seconds=0.0,
    )


def _degraded() -> CascadeHealth:
    return CascadeHealth(
        healthy=False,
        reasons=["version cleanup stalled (1200s since last prune — disk may grow)"],
        pending=42,
        failed_permanent=3,
        failed_retryable=0,
        drain_consecutive_failures=0,
        unrecoverable_total=3,
        optimize_failure_streak=0,
        prune_stale_seconds=1200.0,
    )


def _orch(health: CascadeHealth) -> mock.MagicMock:
    """An autospec orchestrator — ``isinstance(_, CascadeOrchestrator)`` holds."""
    orch = mock.create_autospec(CascadeOrchestrator, instance=True)
    orch.health.return_value = health  # create_autospec makes this an AsyncMock
    return orch


def _client(orch: object | None) -> AsyncClient:
    app = FastAPI()
    app.include_router(health_router)
    app.state.lifespan_data = {"cascade": orch} if orch is not None else {}
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def test_health_without_cascade_is_plain_liveness() -> None:
    """No cascade lifespan → plain 200 liveness, no cascade block."""
    async with _client(None) as c:
        resp = await c.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["cascade"] is None  # no cascade lifespan → block omitted


async def test_health_healthy_cascade() -> None:
    async with _client(_orch(_healthy())) as c:
        resp = await c.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["cascade"]["healthy"] is True
    assert body["cascade"]["reasons"] == []


async def test_health_degraded_stays_200_but_flags_cascade() -> None:
    """Degraded cascade must NOT fail liveness (no crash-loop) — the
    readiness signal lives in the cascade block."""
    async with _client(_orch(_degraded())) as c:
        resp = await c.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["cascade"]["healthy"] is False
    assert body["cascade"]["failed_permanent"] == 3
    assert body["cascade"]["prune_stale_seconds"] == 1200.0
    assert any("cleanup stalled" in r for r in body["cascade"]["reasons"])


async def test_health_probe_exception_stays_200_and_flags_unhealthy() -> None:
    """The probe reads SQLite; a locked / full / mid-migration DB makes
    ``orch.health()`` raise. That must NOT turn /health into a 500 (which
    would flip liveness and restart the container) — the endpoint reports
    unhealthy readiness with a reason and keeps HTTP 200 (review P1-3)."""
    orch = mock.create_autospec(CascadeOrchestrator, instance=True)
    orch.health.side_effect = RuntimeError("database is locked")
    async with _client(orch) as c:
        resp = await c.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["cascade"]["healthy"] is False
    assert any("probe failed" in r for r in body["cascade"]["reasons"])
    assert any("database is locked" in r for r in body["cascade"]["reasons"])

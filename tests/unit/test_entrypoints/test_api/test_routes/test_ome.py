"""Route tests for ``POST /api/v1/ome/trigger``.

Pins the widened ``TriggerResponse`` contract from Task 7: a ``dispatched``
count and a per-run ``runs`` list are always present, and ``status`` gains
``not_dispatched`` for strategies rejected by a dispatch gate (as opposed to
``ok``/``timeout``, which mean the strategy was actually enqueued and either
settled or didn't within the wait window).

Each test wires a bare FastAPI app carrying only the ``ome`` router to a
real, started ``OfflineEngine`` (no lifespan, no LanceDB, no LLM) — the
route's deferred ``_get_engine()`` import resolves to whatever
``everos.service.memorize._ome_engine`` holds, which is patched per test.
"""

from __future__ import annotations

import importlib
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from everos.entrypoints.api.routes.ome import router as ome_router
from everos.infra.ome.config import OMEConfig
from everos.infra.ome.context import StrategyContext
from everos.infra.ome.decorator import offline_strategy
from everos.infra.ome.engine import OfflineEngine
from everos.infra.ome.events import ManualTick
from everos.infra.ome.triggers import Immediate


async def _client_for(
    engine: OfflineEngine, monkeypatch: pytest.MonkeyPatch
) -> AsyncClient:
    """FastAPI app exposing only the ome router, wired to ``engine``."""
    svc = importlib.import_module("everos.service.memorize")
    monkeypatch.setattr(svc, "_ome_engine", engine, raising=False)
    app = FastAPI()
    app.include_router(ome_router, prefix="/api/v1")
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


@pytest.fixture
async def gated_off_engine(tmp_path: Path) -> AsyncIterator[OfflineEngine]:
    """Engine with one strategy registered but ``enabled=False``."""

    @offline_strategy(
        name="gated_off_strategy",
        trigger=Immediate(on=[ManualTick]),
        emits=[],
        enabled=False,
    )
    async def _s(event: ManualTick, ctx: StrategyContext) -> None:
        return None

    engine = OfflineEngine(
        config=OMEConfig(jobstore_path=tmp_path / "ome.db", config_watch=False)
    )
    engine.register(_s)
    await engine.start()
    try:
        yield engine
    finally:
        await engine.stop()


@pytest.fixture
async def always_fails_engine(tmp_path: Path) -> AsyncIterator[OfflineEngine]:
    """Engine with a strategy that raises unconditionally.

    ``max_retries=0`` reaches ``dead_letter`` on the very first attempt —
    no retry backoff sleep, so the test stays fast under the default runner.
    """

    @offline_strategy(
        name="always_fails",
        trigger=Immediate(on=[ManualTick]),
        emits=[],
        max_retries=0,
    )
    async def _s(event: ManualTick, ctx: StrategyContext) -> None:
        raise RuntimeError("boom")

    engine = OfflineEngine(
        config=OMEConfig(jobstore_path=tmp_path / "ome.db", config_watch=False)
    )
    engine.register(_s)
    await engine.start()
    try:
        yield engine
    finally:
        await engine.stop()


async def test_trigger_returns_not_dispatched_when_strategy_gated_off(
    gated_off_engine: OfflineEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A strategy disabled by config, triggered without ``force``, yields
    ``dispatched=0`` and ``status='not_dispatched'`` with no runs."""
    async with await _client_for(gated_off_engine, monkeypatch) as client:
        resp = await client.post(
            "/api/v1/ome/trigger", json={"name": "gated_off_strategy"}
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "not_dispatched"
    assert body["dispatched"] == 0
    assert body["runs"] == []


async def test_trigger_returns_runs_including_dead_letter(
    always_fails_engine: OfflineEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A strategy that raises through all retries dead-letters; the run
    still appears in ``runs`` with its error, and the top-level ``status``
    stays ``ok`` — dispatch happened and the run settled (dead-letter is a
    settled state, not an in-flight one)."""
    async with await _client_for(always_fails_engine, monkeypatch) as client:
        resp = await client.post(
            "/api/v1/ome/trigger",
            json={"name": "always_fails", "force": True, "timeout": 15},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["dispatched"] == 1
    assert len(body["runs"]) == 1
    assert body["runs"][0]["status"] == "dead_letter"
    assert body["runs"][0]["error"]

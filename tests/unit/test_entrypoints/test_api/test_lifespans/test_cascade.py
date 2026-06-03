"""``CascadeLifespanProvider`` — startup builds orchestrator, shutdown stops it."""

from __future__ import annotations

import pytest
from fastapi import FastAPI

from everos.entrypoints.api.lifespans import cascade as cascade_lifespan_mod
from everos.entrypoints.api.lifespans.cascade import CascadeLifespanProvider


class _StubOrchestrator:
    def __init__(self, *args: object, **kwargs: object) -> None:
        self.start_calls = 0
        self.stop_calls = 0

    async def start(self) -> None:
        self.start_calls += 1

    async def stop(self) -> None:
        self.stop_calls += 1


def test_provider_metadata() -> None:
    p = CascadeLifespanProvider(order=42)
    assert p.name == "cascade"
    assert p.order == 42


async def test_startup_constructs_and_starts_orchestrator(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EVEROS_MEMORY__ROOT", str(tmp_path))
    monkeypatch.setenv("EVEROS_EMBEDDING__MODEL", "stub-model")
    monkeypatch.setenv("EVEROS_EMBEDDING__BASE_URL", "http://stub.invalid/v1")
    monkeypatch.setenv("EVEROS_EMBEDDING__API_KEY", "stub-key")

    captured: list[_StubOrchestrator] = []

    def fake_orch(**kwargs: object) -> _StubOrchestrator:
        o = _StubOrchestrator()
        captured.append(o)
        return o

    monkeypatch.setattr(cascade_lifespan_mod, "CascadeOrchestrator", fake_orch)

    p = CascadeLifespanProvider()
    result = await p.startup(FastAPI())
    assert len(captured) == 1
    assert result is captured[0]
    assert captured[0].start_calls == 1


async def test_shutdown_without_startup_is_noop() -> None:
    p = CascadeLifespanProvider()
    await p.shutdown(FastAPI())  # must not raise


async def test_shutdown_stops_orchestrator_and_clears_reference(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EVEROS_MEMORY__ROOT", str(tmp_path))
    monkeypatch.setenv("EVEROS_EMBEDDING__MODEL", "stub-model")
    monkeypatch.setenv("EVEROS_EMBEDDING__BASE_URL", "http://stub.invalid/v1")
    monkeypatch.setenv("EVEROS_EMBEDDING__API_KEY", "stub-key")

    captured: list[_StubOrchestrator] = []

    def fake_orch(**kwargs: object) -> _StubOrchestrator:
        o = _StubOrchestrator()
        captured.append(o)
        return o

    monkeypatch.setattr(cascade_lifespan_mod, "CascadeOrchestrator", fake_orch)

    p = CascadeLifespanProvider()
    app = FastAPI()
    await p.startup(app)
    await p.shutdown(app)
    assert captured[0].stop_calls == 1
    # Second shutdown is a no-op (reference cleared).
    await p.shutdown(app)
    assert captured[0].stop_calls == 1

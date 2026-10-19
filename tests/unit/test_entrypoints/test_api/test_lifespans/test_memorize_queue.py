"""``MemorizeQueueLifespanProvider`` — builds + starts the worker, or stays off."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI

from everos.config import AsyncQueueSettings, load_settings
from everos.entrypoints.api.lifespans import OmeLifespanProvider
from everos.entrypoints.api.lifespans import memorize_queue as lifespan_mod
from everos.entrypoints.api.lifespans.memorize_queue import (
    MemorizeQueueLifespanProvider,
)


class _StubWorker:
    def __init__(
        self, settings: AsyncQueueSettings, *, accept_async: bool = True
    ) -> None:
        self.settings = settings
        self.accepting = accept_async
        self.start_calls = 0
        self.stop_calls = 0

    async def start(self) -> None:
        self.start_calls += 1

    async def stop(self) -> None:
        self.stop_calls += 1


@pytest.fixture
def captured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[_StubWorker]:
    monkeypatch.setenv("EVEROS_ROOT", str(tmp_path))
    load_settings.cache_clear()
    built: list[_StubWorker] = []

    def fake_worker(
        settings: AsyncQueueSettings, *, accept_async: bool = True
    ) -> _StubWorker:
        w = _StubWorker(settings, accept_async=accept_async)
        built.append(w)
        return w

    monkeypatch.setattr(lifespan_mod, "MemorizeQueueWorker", fake_worker)
    yield built
    load_settings.cache_clear()


def test_provider_metadata() -> None:
    p = MemorizeQueueLifespanProvider(order=42)
    assert p.name == "memorize_queue"
    assert p.order == 42


def test_worker_starts_after_and_stops_before_ome() -> None:
    """memorize() emits through the OME engine; shutdown runs in reverse order."""
    assert MemorizeQueueLifespanProvider().order > OmeLifespanProvider().order


async def test_startup_builds_worker_from_settings_and_starts_it(
    captured: list[_StubWorker],
) -> None:
    p = MemorizeQueueLifespanProvider()

    result = await p.startup(FastAPI())

    assert len(captured) == 1
    assert result is captured[0]
    assert captured[0].start_calls == 1
    assert captured[0].accepting is True
    assert captured[0].settings.max_pending_per_session == 50


async def test_startup_disabled_still_runs_worker_but_not_accepting(
    captured: list[_StubWorker], monkeypatch: pytest.MonkeyPatch
) -> None:
    """``enabled = false`` closes the door; leftovers from before are still drained."""
    monkeypatch.setenv("EVEROS_MEMORIZE__ASYNC_QUEUE__ENABLED", "false")
    load_settings.cache_clear()
    p = MemorizeQueueLifespanProvider()

    result = await p.startup(FastAPI())

    assert len(captured) == 1 and result is captured[0]
    assert captured[0].accepting is False
    assert captured[0].start_calls == 1
    await p.shutdown(FastAPI())
    assert captured[0].stop_calls == 1


async def test_shutdown_stops_worker_and_clears_reference(
    captured: list[_StubWorker],
) -> None:
    p = MemorizeQueueLifespanProvider()
    await p.startup(FastAPI())

    await p.shutdown(FastAPI())
    await p.shutdown(FastAPI())  # idempotent

    assert captured[0].stop_calls == 1

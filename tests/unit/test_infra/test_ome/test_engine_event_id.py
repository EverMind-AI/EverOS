"""Tests for OfflineEngine event_id tracking (P3)."""

from __future__ import annotations

import pytest

from everos.infra.ome import BaseEvent, Immediate, RunStatus, offline_strategy
from everos.infra.ome.testing import StrategyTestHarness


class _Ping(BaseEvent):
    """Test event."""


@offline_strategy(
    name="echo",
    trigger=Immediate(on=[_Ping]),
    emits=[],
)
async def _echo_strategy(event: BaseEvent, ctx: object) -> None:
    pass


@pytest.mark.asyncio
async def test_list_runs_by_event_id() -> None:
    async with StrategyTestHarness() as h:
        h.register(_echo_strategy)
        await h.start()
        ping = _Ping()
        await h.emit(ping)
        await h.drain(timeout=5)
        runs = await h._engine.list_runs_by_event_id(ping.event_id)
        assert len(runs) == 1
        assert runs[0].event_id == ping.event_id
        assert runs[0].status == RunStatus.SUCCESS


@pytest.mark.asyncio
async def test_wait_for_event_returns_on_success() -> None:
    async with StrategyTestHarness() as h:
        h.register(_echo_strategy)
        await h.start()
        ping = _Ping()
        await h.emit(ping)
        runs = await h._engine.wait_for_event(ping.event_id, timeout=5)
        assert len(runs) == 1
        assert runs[0].status == RunStatus.SUCCESS


@pytest.mark.asyncio
async def test_wait_for_event_times_out_on_no_runs() -> None:
    async with StrategyTestHarness() as h:
        h.register(_echo_strategy)
        await h.start()
        with pytest.raises(TimeoutError):
            await h._engine.wait_for_event("nonexistent_event", timeout=0.3)


class _Boom(BaseEvent):
    """Event that triggers a failing strategy."""


@offline_strategy(
    name="fail_strategy",
    trigger=Immediate(on=[_Boom]),
    emits=[],
    max_retries=0,
)
async def _fail_strategy(event: BaseEvent, ctx: object) -> None:
    raise RuntimeError("intentional failure")


@pytest.mark.asyncio
async def test_wait_for_event_returns_on_terminal_failure() -> None:
    async with StrategyTestHarness() as h:
        h.register(_fail_strategy)
        await h.start()
        boom = _Boom()
        await h.emit(boom)
        runs = await h._engine.wait_for_event(boom.event_id, timeout=5)
        assert len(runs) == 1
        assert runs[0].status == RunStatus.DEAD_LETTER

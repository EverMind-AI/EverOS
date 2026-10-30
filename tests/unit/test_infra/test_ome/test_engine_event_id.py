"""Tests for OfflineEngine event_id tracking (P3)."""

from __future__ import annotations

import pytest

from everos.infra.ome import RunStatus
from everos.memory.deps import MemoryDeps
from everos.memory.ports import Event, Immediate
from tests.fakes import make_deps
from tests.helpers.ome import StrategyTestHarness, strategy_spec


class _Ping(Event):
    """Test event."""


@strategy_spec(
    name="echo",
    trigger=Immediate(on=[_Ping]),
    emits=[],
)
async def _echo_strategy(event: Event, deps: MemoryDeps) -> list[Event]:
    return []


@pytest.mark.asyncio
async def test_list_runs_by_event_id() -> None:
    async with StrategyTestHarness() as h:
        h.register(_echo_strategy, make_deps())
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
        h.register(_echo_strategy, make_deps())
        await h.start()
        ping = _Ping()
        await h.emit(ping)
        runs = await h._engine.wait_for_event(ping.event_id, timeout=5)
        assert len(runs) == 1
        assert runs[0].status == RunStatus.SUCCESS


@pytest.mark.asyncio
async def test_wait_for_event_times_out_on_no_runs() -> None:
    async with StrategyTestHarness() as h:
        h.register(_echo_strategy, make_deps())
        await h.start()
        with pytest.raises(TimeoutError):
            await h._engine.wait_for_event("nonexistent_event", timeout=0.3)


class _Boom(Event):
    """Event that triggers a failing strategy."""


@strategy_spec(
    name="fail_strategy",
    trigger=Immediate(on=[_Boom]),
    emits=[],
    max_retries=0,
)
async def _fail_strategy(event: Event, deps: MemoryDeps) -> list[Event]:
    raise RuntimeError("intentional failure")


@pytest.mark.asyncio
async def test_wait_for_event_returns_on_terminal_failure() -> None:
    async with StrategyTestHarness() as h:
        h.register(_fail_strategy, make_deps())
        await h.start()
        boom = _Boom()
        await h.emit(boom)
        runs = await h._engine.wait_for_event(boom.event_id, timeout=5)
        assert len(runs) == 1
        assert runs[0].status == RunStatus.DEAD_LETTER


class _Flaky(Event):
    """Event whose strategy fails once, then succeeds."""


_flaky_calls: dict[str, int] = {}


@strategy_spec(
    name="flaky_strategy",
    trigger=Immediate(on=[_Flaky]),
    emits=[],
    max_retries=1,
)
async def _flaky_strategy(event: Event, deps: MemoryDeps) -> list[Event]:
    n = _flaky_calls.get(event.event_id, 0) + 1
    _flaky_calls[event.event_id] = n
    if n == 1:
        raise RuntimeError("first attempt fails")
    return []


@pytest.mark.asyncio
async def test_wait_for_event_waits_out_a_pending_retry() -> None:
    """A FAILED row is written *before* the backoff sleep, and the retry runs
    under a fresh run_id; until it starts, the event's only row is FAILED.
    ``wait_for_event`` must not report that as the event's outcome."""
    async with StrategyTestHarness() as h:
        h._engine._config.retry_backoff_base_seconds = 0.3  # a real backoff window
        h.register(_flaky_strategy, make_deps())
        await h.start()
        flaky = _Flaky()
        await h.emit(flaky)
        returned = await h._engine.wait_for_event(
            flaky.event_id, timeout=5, poll_interval=0.02
        )
        await h.drain(timeout=5)
        final = await h._engine.list_runs_by_event_id(flaky.event_id)
        assert sorted(r.status for r in final) == [RunStatus.FAILED, RunStatus.SUCCESS]
        assert sorted(r.status for r in returned) == sorted(r.status for r in final), (
            "wait_for_event returned before the retry ran"
        )

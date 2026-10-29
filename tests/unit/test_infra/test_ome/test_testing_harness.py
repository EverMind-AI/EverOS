"""Tests for the OME StrategyTestHarness."""

from __future__ import annotations

from everos.infra.ome.testing import StrategyTestHarness
from everos.memory.deps import MemoryDeps
from everos.memory.ports import Event, Immediate, StrategySpec
from tests.fakes import make_deps


class _E(Event):
    """Simple test event."""


async def test_harness_runs_strategy_end_to_end() -> None:
    """The harness registers a spec with its deps and runs it end to end."""
    seen: list[Event] = []

    async def s(event: _E, deps: MemoryDeps) -> list[Event]:
        seen.append(event)
        return []

    spec = StrategySpec(name="s_t23", fn=s, trigger=Immediate(on=[_E]))
    async with StrategyTestHarness() as h:
        h.register(spec, make_deps())
        await h.start()
        await h.emit(_E())
        await h.drain(timeout=5)
        runs = await h.list_runs("s_t23")
        assert len(runs) == 1
        assert len(seen) == 1

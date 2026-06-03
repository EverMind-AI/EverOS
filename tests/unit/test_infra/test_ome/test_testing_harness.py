"""Tests for OME testing helpers (FakeStrategyContext + StrategyTestHarness)."""

from __future__ import annotations

import pytest

from everos.infra.ome.context import StrategyContext
from everos.infra.ome.decorator import offline_strategy
from everos.infra.ome.events import BaseEvent
from everos.infra.ome.testing import FakeStrategyContext, StrategyTestHarness
from everos.infra.ome.triggers import Immediate


class _E(BaseEvent):
    """Simple test event."""

    pass


@pytest.mark.asyncio
async def test_fake_strategy_context_collects_emits() -> None:
    """FakeStrategyContext should collect emit() calls into a list."""
    ctx = FakeStrategyContext()
    await ctx.emit(_E())
    assert len(ctx.emitted) == 1


@pytest.mark.asyncio
async def test_harness_runs_strategy_end_to_end() -> None:
    """StrategyTestHarness should execute a strategy end-to-end."""
    seen: list[BaseEvent] = []

    @offline_strategy(name="s_t23", trigger=Immediate(on=[_E]), emits=[])
    async def s(event: _E, ctx: StrategyContext) -> None:
        seen.append(event)

    async with StrategyTestHarness() as h:
        h.register(s)
        await h.start()
        await h.emit(_E())
        await h.drain(timeout=5)
        runs = await h.list_runs("s_t23")
        assert len(runs) == 1
        assert len(seen) == 1

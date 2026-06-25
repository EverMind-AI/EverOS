from __future__ import annotations

import pytest

from everos.infra.ome.context import StrategyContext
from everos.infra.ome.decorator import Strategy, StrategyMeta, offline_strategy
from everos.infra.ome.events import BaseEvent
from everos.infra.ome.gates import Counter
from everos.infra.ome.triggers import Immediate


class _E(BaseEvent):
    user_id: str


def test_decorator_attaches_metadata() -> None:
    @offline_strategy(name="x", trigger=Immediate(on=[_E]), emits=[_E])
    async def s(event: _E, ctx: StrategyContext) -> None:
        return None

    meta: StrategyMeta = s.meta
    assert meta.name == "x"
    assert meta.emits == frozenset({_E})
    assert meta.gate is None
    assert meta.applies_to is None
    assert meta.max_retries is None
    assert meta.enabled is True
    assert isinstance(s, Strategy)
    assert callable(meta.func)


def test_decorator_with_full_params() -> None:
    @offline_strategy(
        name="cluster",
        trigger=Immediate(on=[_E]),
        emits=[_E],
        applies_to="user_id",
        gate=Counter(threshold=5),
        max_retries=3,
        enabled=False,
    )
    async def s(event: _E, ctx: StrategyContext) -> None:
        return None

    meta = s.meta
    assert meta.applies_to == "user_id"
    assert meta.gate.threshold == 5
    assert meta.max_retries == 3
    assert meta.enabled is False


def test_decorator_callable_applies_to() -> None:
    def is_paid(e: _E) -> bool:
        return e.user_id.startswith("paid_")

    @offline_strategy(
        name="paid_only",
        trigger=Immediate(on=[_E]),
        emits=[_E],
        applies_to=is_paid,
    )
    async def s(event: _E, ctx: StrategyContext) -> None:
        return None

    meta = s.meta
    assert meta.applies_to is is_paid


def test_decorator_rejects_blank_name() -> None:
    with pytest.raises(ValueError):

        @offline_strategy(name="", trigger=Immediate(on=[_E]), emits=[_E])
        async def _s(event: _E, ctx: StrategyContext) -> None:
            return None


def test_decorator_rejects_non_async_function() -> None:
    with pytest.raises(TypeError):

        @offline_strategy(name="x", trigger=Immediate(on=[_E]), emits=[_E])
        def _s(event: _E, ctx: StrategyContext) -> None:  # not async
            return None

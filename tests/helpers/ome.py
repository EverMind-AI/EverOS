"""Declare throw-away OME strategies in tests the way production does.

Production code declares a :class:`~everos.memory.ports.StrategySpec` and
registers ``to_ome_strategy(spec, deps)``; :func:`ome_strategy` is that pair
as a decorator, so an engine test states a strategy in one place::

    @ome_strategy(name="s", trigger=Immediate(on=[Ping]), emits=[Pong])
    async def s(event: Ping, deps: MemoryDeps) -> list[Event]:
        return [Pong()]

    engine.register(s)
"""

from __future__ import annotations

__all__ = ["ome_strategy", "strategy_spec"]

from collections.abc import Callable, Iterable
from typing import Any

from everos.infra.ome import to_ome_strategy
from everos.memory.deps import MemoryDeps
from everos.memory.ports import Event, StrategyFn, StrategySpec, Trigger
from tests.fakes import make_deps


def strategy_spec(
    *,
    name: str,
    trigger: Trigger,
    emits: Iterable[type[Event]] = (),
    **options: Any,
) -> Callable[[StrategyFn], StrategySpec]:
    """Declare the decorated function as a :class:`StrategySpec`.

    ``options`` are the remaining spec fields (``applies_to``, ``gate``,
    ``max_retries``, ``enabled``).
    """

    def wrap(fn: StrategyFn) -> StrategySpec:
        return StrategySpec(
            name=name, fn=fn, trigger=trigger, emits=frozenset(emits), **options
        )

    return wrap


def ome_strategy(
    *,
    name: str,
    trigger: Trigger,
    emits: Iterable[type[Event]] = (),
    deps: MemoryDeps | None = None,
    **options: Any,
) -> Callable[[StrategyFn], Any]:
    """Declare the decorated function as a spec and adapt it for OME.

    ``deps`` defaults to in-memory doubles; the remaining keywords are as
    for :func:`strategy_spec`.
    """

    def wrap(fn: StrategyFn) -> Any:
        spec = strategy_spec(name=name, trigger=trigger, emits=emits, **options)(fn)
        return to_ome_strategy(spec, deps if deps is not None else make_deps())

    return wrap

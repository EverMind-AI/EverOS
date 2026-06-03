"""@offline_strategy decorator — attaches StrategyMeta to the function.

Decorator is side-effect-free; engine collects via explicit
`engine.register(func)`.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from everos.infra.ome.context import StrategyContext
from everos.infra.ome.events import BaseEvent
from everos.infra.ome.gates import Counter
from everos.infra.ome.triggers import Trigger

type AppliesTo = str | Callable[[BaseEvent], bool] | None
type StrategyFn = Callable[[BaseEvent, StrategyContext], Awaitable[None]]


@dataclass(frozen=True)
class StrategyMeta:
    """Captured at decoration time; consumed by engine.register()."""

    name: str
    trigger: Trigger
    emits: frozenset[type[BaseEvent]]
    applies_to: AppliesTo
    gate: Counter | None
    max_retries: int | None
    enabled: bool
    func: StrategyFn


def offline_strategy(
    *,
    name: str,
    trigger: Trigger,
    emits: list[type[BaseEvent]],
    applies_to: AppliesTo = None,
    gate: Counter | None = None,
    max_retries: int | None = None,
    enabled: bool = True,
) -> Callable[[StrategyFn], StrategyFn]:
    """Mark an async function as an OME strategy."""

    if not name or not name.strip():
        raise ValueError("offline_strategy: name must be a non-empty string")

    def wrap(func: StrategyFn) -> StrategyFn:
        if not inspect.iscoroutinefunction(func):
            raise TypeError(
                f"offline_strategy: {func.__name__} must be async (coroutine function)"
            )
        meta = StrategyMeta(
            name=name,
            trigger=trigger,
            emits=frozenset(emits),
            applies_to=applies_to,
            gate=gate,
            max_retries=max_retries,
            enabled=enabled,
            func=func,
        )
        func._ome_strategy_meta = meta  # type: ignore[attr-defined]
        return func

    return wrap

"""@offline_strategy decorator — attaches StrategyMeta to a Strategy wrapper.

Decorator is side-effect-free; engine collects via explicit
``engine.register(strategy)``.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypeVar, overload

from everos.infra.ome.context import StrategyContext
from everos.infra.ome.events import BaseEvent, CronTick, IdleTick
from everos.infra.ome.gates import Counter
from everos.infra.ome.triggers import Cron, Idle, Immediate, Trigger

type AppliesTo = str | Callable[[BaseEvent], bool] | None
type StrategyFn = Callable[[BaseEvent, StrategyContext], Awaitable[None]]

_E = TypeVar("_E", bound=BaseEvent)

_CronStrategyFn = Callable[[CronTick, StrategyContext], Awaitable[None]]
_IdleStrategyFn = Callable[[IdleTick, StrategyContext], Awaitable[None]]
_EventStrategyFn = Callable[[_E, StrategyContext], Awaitable[None]]


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


class Strategy:
    """Wrapper returned by :func:`offline_strategy`.

    Carries typed :attr:`meta` and delegates ``__call__`` to the
    original async function — so ``await my_strategy(event, ctx)``
    works transparently in both production and tests.

    Args:
        meta: Frozen strategy metadata captured at decoration time.
    """

    __slots__ = ("meta",)

    def __init__(self, meta: StrategyMeta) -> None:
        self.meta = meta

    async def __call__(self, event: BaseEvent, ctx: StrategyContext) -> None:
        await self.meta.func(event, ctx)

    def __repr__(self) -> str:
        return f"Strategy({self.meta.name!r})"


@overload
def offline_strategy(
    *,
    name: str,
    trigger: Cron,
    emits: list[type[BaseEvent]],
    applies_to: AppliesTo = ...,
    gate: Counter | None = ...,
    max_retries: int | None = ...,
    enabled: bool = ...,
) -> Callable[[_CronStrategyFn], Strategy]: ...


@overload
def offline_strategy(
    *,
    name: str,
    trigger: Idle,
    emits: list[type[BaseEvent]],
    applies_to: AppliesTo = ...,
    gate: Counter | None = ...,
    max_retries: int | None = ...,
    enabled: bool = ...,
) -> Callable[[_IdleStrategyFn], Strategy]: ...


@overload
def offline_strategy(
    *,
    name: str,
    trigger: Immediate,
    emits: list[type[BaseEvent]],
    applies_to: AppliesTo = ...,
    gate: Counter | None = ...,
    max_retries: int | None = ...,
    enabled: bool = ...,
) -> Callable[[_EventStrategyFn[_E]], Strategy]: ...


def offline_strategy(
    *,
    name: str,
    trigger: Trigger,
    emits: list[type[BaseEvent]],
    applies_to: AppliesTo = None,
    gate: Counter | None = None,
    max_retries: int | None = None,
    enabled: bool = True,
) -> Any:  # overloads above provide call-site precision
    """Mark an async function as an OME strategy.

    Args:
        name: Unique strategy name (used for logging, run records, config).
        trigger: When to fire — ``Cron``, ``Idle``, or ``Immediate``.
        emits: Event types this strategy may emit via ``ctx.emit()``.
        applies_to: Optional gate predicate (None = all events).
        gate: Optional counter-based rate limiter.
        max_retries: Override engine default; ``None`` uses engine config.
        enabled: ``False`` disables without unregistering.

    Returns:
        Decorator that wraps the function in a :class:`Strategy` instance.

    Raises:
        ValueError: If ``name`` is empty or whitespace-only.
        TypeError: If the decorated function is not async.
    """

    if not name or not name.strip():
        raise ValueError("offline_strategy: name must be a non-empty string")

    def wrap(func: StrategyFn) -> Strategy:
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
        return Strategy(meta)

    return wrap

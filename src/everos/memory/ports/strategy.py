"""What a strategy is: when it fires, what it may emit, and its function.

A strategy is an async function ``fn(event, deps) -> list[Event]``: it
receives the triggering event and the domain's dependencies, does its work
through them, and returns the follow-up events instead of publishing them.
:class:`StrategySpec` declares one such function together with its trigger
and contract; the scheduler adapter in ``infra.ome`` turns a spec into a
scheduled job and publishes what the function returns.

Load-bearing rules:

- ``name`` is the strategy's identity across processes: run records and the
  per-strategy overrides in ``ome.toml`` are keyed by it, so it never changes.
- A returned event whose class is not in ``emits`` is a contract violation;
  the adapter rejects the whole return value before publishing any of it.
- ``applies_to`` is evaluated at dispatch time, before a run exists: a
  strategy it rejects leaves no run record.
"""

from __future__ import annotations

__all__ = [
    "AppliesTo",
    "Counter",
    "Cron",
    "Idle",
    "Immediate",
    "StrategyFn",
    "StrategySpec",
    "Trigger",
]

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, Self

from apscheduler.triggers.cron import CronTrigger
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from everos.memory.ports.events import Event

if TYPE_CHECKING:
    from everos.memory.deps import MemoryDeps

# ── Triggers ─────────────────────────────────────────────────────────────


class _TriggerBase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Immediate(_TriggerBase):
    """Fire as soon as an event of any class in ``on`` is published."""

    on: Annotated[list[type[Event]], Field(min_length=1)]


class Cron(_TriggerBase):
    """Fire on a cron schedule; the scheduler hands the strategy a ``CronTick``."""

    expr: Annotated[str, Field(min_length=1)]

    @field_validator("expr")
    @classmethod
    def _validate_crontab(cls, v: str) -> str:
        # Delegates to APS's own parser so the trigger object cannot
        # represent any crontab that APS would later refuse.
        CronTrigger.from_crontab(v)
        return v


class Idle(_TriggerBase):
    """Fire after every class in ``on`` has been silent (bucketed by
    ``event_field``) for ``idle_seconds`` — AND across classes. The
    scheduler hands the strategy an ``IdleTick``.
    """

    on: Annotated[list[type[Event]], Field(min_length=1)]
    event_field: str
    idle_seconds: Annotated[int, Field(gt=0)]
    scan_interval_seconds: Annotated[
        int,
        Field(gt=0, description="Per-strategy scan cadence; <= idle_seconds / 2."),
    ] = 60

    @model_validator(mode="after")
    def _validate_event_field(self) -> Self:
        for event_cls in self.on:
            if self.event_field not in event_cls.model_fields:  # type: ignore[operator]  # Pydantic model_fields → dict via @deprecated_instance_property (pydantic/main.py:277)
                available = list(event_cls.model_fields)  # type: ignore[arg-type]  # same as above
                raise ValueError(
                    f"event_field {self.event_field!r} not found in "
                    f"{event_cls.__name__} fields (available: {available})"
                )
        return self

    @model_validator(mode="after")
    def _validate_scan_interval_bound(self) -> Self:
        if self.scan_interval_seconds > self.idle_seconds // 2:
            raise ValueError(
                f"Idle: scan_interval_seconds ({self.scan_interval_seconds}) "
                f"must be <= idle_seconds // 2 ({self.idle_seconds // 2})"
            )
        return self


Trigger = Immediate | Cron | Idle


# ── Gates ────────────────────────────────────────────────────────────────


class Counter(BaseModel):
    """Counter gate: batch trigger by accumulated event count per bucket.

    Each event increments the bucket counter; the ``threshold``-th event
    passes and resets.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    threshold: Annotated[
        int,
        Field(
            gt=0,
            description=(
                "Pass once every `threshold` events; threshold=1 lets every event pass."
            ),
        ),
    ]
    cooldown_seconds: Annotated[
        int,
        Field(
            ge=0,
            description=(
                "Minimum seconds between consecutive passes per bucket; 0 disables."
            ),
        ),
    ] = 0
    event_field: Annotated[
        str | None,
        Field(
            description=(
                'Bucket dimension on the event (e.g. "user_id"); '
                "None means a single global bucket."
            ),
        ),
    ] = None


# ── Strategy ─────────────────────────────────────────────────────────────

# The event parameter is ``Any`` because each strategy narrows it to the
# event classes its own trigger delivers; ``Callable`` parameters are
# contravariant, so no single event type would admit every strategy.
type StrategyFn = Callable[[Any, MemoryDeps], Awaitable[list[Event]]]
type AppliesTo = str | Callable[[Any, MemoryDeps], bool] | None


@dataclass(frozen=True)
class StrategySpec:
    """Declaration of one strategy.

    Args:
        name: Unique, stable strategy name (run records and ``ome.toml``
            overrides are keyed by it).
        fn: The strategy function.
        trigger: When to fire — ``Immediate``, ``Cron`` or ``Idle``.
        emits: Event classes ``fn`` may return.
        applies_to: Dispatch-time filter. A string names an event field
            whose truthiness decides; a callable receives the event and the
            dependencies; ``None`` accepts every event the trigger delivers.
        gate: Optional counter gate.
        max_retries: Retry budget; ``None`` uses the engine default.
        enabled: ``False`` keeps the strategy registered but never dispatched.

    Raises:
        ValueError: ``name`` is empty or whitespace-only.
        TypeError: ``fn`` is not a coroutine function.
    """

    name: str
    fn: StrategyFn
    trigger: Trigger
    emits: frozenset[type[Event]] = frozenset()
    applies_to: AppliesTo = None
    gate: Counter | None = None
    max_retries: int | None = None
    enabled: bool = True

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("StrategySpec: name must be a non-empty string")
        if not inspect.iscoroutinefunction(self.fn):
            raise TypeError(
                f"StrategySpec {self.name!r}: fn must be a coroutine function"
            )

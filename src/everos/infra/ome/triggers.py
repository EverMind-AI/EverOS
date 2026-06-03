"""OME trigger types — declarative descriptors of when a strategy fires.

Three concrete triggers: Immediate / Cron / Idle. Engine dispatches via
`isinstance(meta.trigger, ...)` to pick the registration path.
"""

from __future__ import annotations

from typing import Annotated, Self

from apscheduler.triggers.cron import CronTrigger
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from everos.infra.ome.events import BaseEvent


class _TriggerBase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Immediate(_TriggerBase):
    """Fire as soon as an event of any class in `on` is dispatched."""

    on: Annotated[list[type[BaseEvent]], Field(min_length=1)]


class Cron(_TriggerBase):
    """Fire on a cron schedule. Engine emits CronTick to the strategy."""

    expr: Annotated[str, Field(min_length=1)]

    @field_validator("expr")
    @classmethod
    def _validate_crontab(cls, v: str) -> str:
        # Delegates to APS's own parser so the trigger object cannot
        # represent any crontab that APS would later refuse.
        CronTrigger.from_crontab(v)
        return v


class Idle(_TriggerBase):
    """Fire after every class in `on` has been silent (bucketed by
    `event_field`) for `idle_seconds` — AND across classes. Engine
    emits IdleTick.
    """

    on: Annotated[list[type[BaseEvent]], Field(min_length=1)]
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

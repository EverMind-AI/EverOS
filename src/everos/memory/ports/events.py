"""Event vocabulary of the memory domain and of the scheduler that runs it.

Every domain event subclasses :class:`Event`; the scheduler emits the three
tick events for its time-driven triggers. Events are frozen Pydantic models
because the scheduler persists each triggering event as JSON and re-imports
its class by :meth:`Event.topic` during crash recovery — an event must
round-trip through ``model_dump_json`` / ``model_validate_json`` unchanged.
"""

from __future__ import annotations

__all__ = ["CronTick", "Event", "IdleTick", "ManualTick"]

from datetime import datetime
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from everos.core.utils.datetime import get_utc_now


class Event(BaseModel):
    """Base of every event that flows between strategies."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: str = Field(default_factory=lambda: uuid4().hex)
    ts: datetime = Field(default_factory=get_utc_now)

    @classmethod
    def topic(cls) -> str:
        """Stable cross-process identifier of this event class.

        Returns ``"<module>:<class>"`` (colon-separated, mirroring the
        Python event-sourcing community convention). The scheduler persists
        it with each run and resolves it back to the class on recovery.
        """
        return f"{cls.__module__}:{cls.__name__}"


class CronTick(Event):
    """Scheduler-emitted event for a strategy with ``trigger=Cron(...)``."""

    strategy_name: str


class IdleTick(Event):
    """Scheduler-emitted event for a strategy with ``trigger=Idle(...)``."""

    strategy_name: str
    bucket_key: str
    idle_seconds: int


class ManualTick(Event):
    """Scheduler-emitted event for a manual trigger without an event."""

    strategy_name: str

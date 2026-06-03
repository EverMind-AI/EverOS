"""OME event base class + built-in tick events.

All business events should subclass BaseEvent. OME emits three built-in
ticks for engine-driven triggers (Cron / Idle / Manual).
"""

from __future__ import annotations

import importlib
from datetime import datetime
from functools import cache
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from everos.component.utils.datetime import get_utc_now


class BaseEvent(BaseModel):
    """Base for all events flowing through OME.

    Subclasses must be Pydantic v2 models (immutable) so `model_dump_json` /
    `model_validate_json` work for crash-recovery payload persistence.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: str = Field(default_factory=lambda: uuid4().hex)
    ts: datetime = Field(default_factory=get_utc_now)

    @classmethod
    def topic(cls) -> str:
        """Stable cross-process identifier of this event class.

        Returns ``"<module>:<class>"`` (colon-separated, mirroring the
        Python event-sourcing community convention). Used by OME to
        persist event identity into RunRecord.event_topic and to re-import
        the class during crash recovery via ``resolve_topic``.
        """
        return f"{cls.__module__}:{cls.__name__}"


@cache
def resolve_topic(topic: str) -> type[BaseEvent]:
    """Inverse of ``BaseEvent.topic()``; imports and returns the class.

    Cached because crash recovery may resolve the same topic many times in
    a tight loop, and ``importlib.import_module`` is non-trivial.
    """
    module_name, sep, cls_name = topic.partition(":")
    if not sep or not cls_name:
        raise ValueError(f"invalid event topic: {topic!r}")
    mod: Any = importlib.import_module(module_name)
    cls = getattr(mod, cls_name, None)
    if not (isinstance(cls, type) and issubclass(cls, BaseEvent)):
        raise TypeError(f"topic {topic!r} did not resolve to a BaseEvent subclass")
    return cls


class CronTick(BaseEvent):
    """Engine-emitted event for a strategy with `trigger=Cron(...)`."""

    strategy_name: str


class IdleTick(BaseEvent):
    """Engine-emitted event for a strategy with `trigger=Idle(...)`."""

    strategy_name: str
    bucket_key: str
    idle_seconds: int


class ManualTick(BaseEvent):
    """Engine-emitted event for `engine.trigger_manual(name, event=None)`."""

    strategy_name: str

"""Every event names its scope, or is a known unscoped tick (C11).

The purge gate drops a background event older than the latest purge of its
scope, by ``app_id`` / ``project_id``. An event type without them would
never be recognised as stale and could write a purged scope back.
"""

from __future__ import annotations

import everos.memory.events  # noqa: F401 - registers the business events
from everos.memory.ports import CronTick, Event, IdleTick, ManualTick

UNSCOPED = {CronTick, IdleTick, ManualTick}


def _all_subclasses(cls: type) -> set[type]:
    found: set[type] = set()
    for sub in cls.__subclasses__():
        found |= {sub, *_all_subclasses(sub)}
    return found


def test_event_scope_pinned() -> None:
    events = {e for e in _all_subclasses(Event) if e.__module__.startswith("everos.")}
    assert events  # the walk found the business events
    unscoped = {
        e for e in events if not {"app_id", "project_id"} <= set(e.model_fields)
    }
    assert unscoped == UNSCOPED

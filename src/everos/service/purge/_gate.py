"""The OME gate of a purge: hold background runs while it runs, drop stale ones."""

from __future__ import annotations

__all__ = ["PurgeRunGate"]

from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime
from typing import Literal

from everos.core.purge_marks import CURRENT_EVENT, EventStamp, PurgeMarks


def _stamp(event: object) -> EventStamp | None:
    """The event's scope and time; ``None`` for an unscoped event (a tick)."""
    app_id = getattr(event, "app_id", None)
    project_id = getattr(event, "project_id", None)
    ts = getattr(event, "ts", None)
    if app_id is None or project_id is None or not isinstance(ts, datetime):
        return None
    return EventStamp(app_id=app_id, project_id=project_id, ts=ts)


class PurgeRunGate:
    """:class:`~everos.infra.ome.RunGate` over one root's purge marks.

    A scoped event older than the latest purge of its scope is dropped. Any
    other event waits while a purge of the root runs (the whole root's
    writes are held), then runs. A tombstone never drops an unscoped event.
    """

    def __init__(self, marks: PurgeMarks) -> None:
        self._marks = marks

    def verdict(self, event: object) -> Literal["run", "discard", "wait"]:
        stamp = _stamp(event)
        if stamp is not None:
            purged_at = self._marks.tombstone_at(stamp.app_id, stamp.project_id)
            if purged_at is not None and purged_at > stamp.ts:
                return "discard"
        return "wait" if self._marks.active_job_id is not None else "run"

    @asynccontextmanager
    async def lifted(self, event: object) -> AsyncIterator[None]:
        event_id = getattr(event, "event_id", None)
        if event_id is not None:
            self._marks.waiting_events.add(event_id)
        try:
            await self._marks.wait_lifted()
        finally:
            self._marks.waiting_events.discard(event_id)
        yield

    @contextmanager
    def bind(self, event: object) -> Iterator[None]:
        stamp = _stamp(event)
        if stamp is None:
            yield
            return
        token = CURRENT_EVENT.set(stamp)
        try:
            yield
        finally:
            CURRENT_EVENT.reset(token)

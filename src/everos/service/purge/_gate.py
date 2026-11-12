"""The OME gate of a purge: drop background work whose input predates it."""

from __future__ import annotations

__all__ = ["PurgeRunGate"]

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from typing import Literal

from everos.core.purge_marks import ORIGIN, PurgeMarks


class PurgeRunGate:
    """:class:`~everos.infra.ome.RunGate` over one root's tombstones.

    A scoped event older than the latest purge of its scope is dropped; any
    other event runs. While it runs, its ``ts`` is the task's ``ORIGIN``, so
    the events it emits carry the time of the input they stem from. A tick
    (no scope) has no input of its own: what it emits stems from what it
    reads when it starts, so the start is its origin.
    """

    def __init__(self, marks: PurgeMarks) -> None:
        self._marks = marks

    def verdict(self, event: object) -> Literal["run", "discard"]:
        app_id = getattr(event, "app_id", None)
        project_id = getattr(event, "project_id", None)
        ts = getattr(event, "ts", None)
        if app_id is None or project_id is None or not isinstance(ts, datetime):
            return "run"  # a tick: it reads what is there now
        purged_at = self._marks.tombstone_at(app_id, project_id)
        return "discard" if purged_at is not None and purged_at > ts else "run"

    @contextmanager
    def bind(self, event: object) -> Iterator[None]:
        ts = getattr(event, "ts", None)
        scoped = getattr(event, "app_id", None) is not None
        token = ORIGIN.set(ts if scoped and isinstance(ts, datetime) else None)
        try:
            yield
        finally:
            ORIGIN.reset(token)

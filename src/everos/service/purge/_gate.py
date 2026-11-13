"""The OME gate of a purge: drop background work whose input predates it."""

from __future__ import annotations

__all__ = ["PurgeRunGate"]

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from typing import Literal

from everos.core.purge_marks import ORIGIN, PurgeMarks
from everos.core.utils.datetime import get_utc_now


def _scoped_ts(event: object) -> tuple[str, str, datetime] | None:
    """``(app_id, project_id, ts)`` of a scoped event; ``None`` for a tick."""
    app_id = getattr(event, "app_id", None)
    project_id = getattr(event, "project_id", None)
    ts = getattr(event, "ts", None)
    if app_id is None or project_id is None or not isinstance(ts, datetime):
        return None
    return app_id, project_id, ts


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
        scoped = _scoped_ts(event)
        if scoped is None:
            return "run"  # a tick: it reads what is there now
        app_id, project_id, ts = scoped
        purged_at = self._marks.tombstone_at(app_id, project_id)
        return "discard" if purged_at is not None and purged_at > ts else "run"

    @contextmanager
    def bind(self, event: object) -> Iterator[None]:
        scoped = _scoped_ts(event)
        token = ORIGIN.set(scoped[2] if scoped is not None else get_utc_now())
        try:
            yield
        finally:
            ORIGIN.reset(token)

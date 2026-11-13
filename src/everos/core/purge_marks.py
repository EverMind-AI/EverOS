"""Which scopes of which memory root were purged, and what is still in flight.

A purge leaves a *tombstone* -- the scope and the time ``T`` it was
confirmed. Background work whose input the system took in before ``T`` (an
OME backlog, a replayed queue row) is stale for that scope and is dropped
before it runs; nothing else is held or refused.

"Took in" is the *origin* of a piece of work: the arrival of the write
request, the ``created_at`` of a queued row, the ``ts`` of the event an OME
run handles. :data:`ORIGIN` carries it through the task doing the work, and
``Event.ts`` defaults to it, so an event emitted while handling an older
input carries that input's time.

The purge waits for the work in flight that predates ``T`` before it clears
the scope a second time; :meth:`PurgeMarks.track` registers such work.

The tables are per root and in memory; the purge service rebuilds the
tombstones from ``system.db`` at startup. Use the module-level :data:`MARKS`.
"""

from __future__ import annotations

__all__ = [
    "MARKS",
    "ORIGIN",
    "PURGE_DISCARD_PREFIX",
    "PurgeMarkRegistry",
    "PurgeMarks",
    "PurgeScope",
    "Tombstone",
    "origin_or_now",
]

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from everos.core.utils.datetime import get_utc_now

ORIGIN: ContextVar[datetime | None] = ContextVar("everos_origin", default=None)
"""When the system took in the input the current task is working on."""

PURGE_DISCARD_PREFIX = "purge_in_progress: "
"""``last_error`` prefix of an async request a purge dropped (``/tasks`` code)."""


def origin_or_now() -> datetime:
    """The current task's input origin; now, outside such a task."""
    return ORIGIN.get() or get_utc_now()


@dataclass(frozen=True)
class PurgeScope:
    """A purged range: the whole root, one app, or one project of an app.

    ``app_dir`` / ``project_dir`` are the on-disk names of ``app_id`` /
    ``project_id`` (``default`` lives in ``default_app/default_project``);
    the caller maps them, this layer does not know the layout.
    """

    kind: Literal["space", "app", "project"]
    app_id: str | None = None
    project_id: str | None = None
    app_dir: str | None = None
    project_dir: str | None = None

    def covers(self, app_id: str, project_id: str) -> bool:
        """Whether data stored under ``(app_id, project_id)`` is in range."""
        if self.kind == "space":
            return True
        if self.kind == "app":
            return app_id == self.app_id
        return app_id == self.app_id and project_id == self.project_id


@dataclass(frozen=True)
class Tombstone:
    """A purge of ``scope`` confirmed at ``at`` (aware UTC)."""

    scope: PurgeScope
    at: datetime


class PurgeMarks:
    """The tombstones, the active job and the tracked work of one root.

    The active job serves the one-purge-at-a-time rule, the trash sweep at
    startup and the refusal of writes into its range.
    """

    def __init__(self) -> None:
        self._job_id: str | None = None
        self._scope: PurgeScope | None = None
        self._tombstones: list[Tombstone] = []
        self._tracked: dict[object, tuple[datetime, tuple[str, str] | None]] = {}

    @property
    def active_job_id(self) -> str | None:
        return self._job_id

    @property
    def active_scope(self) -> PurgeScope | None:
        return self._scope

    def raise_mark(self, job_id: str, scope: PurgeScope) -> None:
        self._job_id, self._scope = job_id, scope

    def lift_mark(self) -> None:
        self._job_id = self._scope = None

    def set_tombstones(self, items: Iterable[Tombstone]) -> None:
        self._tombstones = list(items)

    def add_tombstone(self, item: Tombstone) -> None:
        self._tombstones.append(item)

    def remove_tombstone(self, item: Tombstone) -> None:
        if item in self._tombstones:
            self._tombstones.remove(item)

    def tombstone_at(self, app_id: str, project_id: str) -> datetime | None:
        """The latest purge covering ``(app_id, project_id)``, if any."""
        return max(
            (t.at for t in self._tombstones if t.scope.covers(app_id, project_id)),
            default=None,
        )

    @contextmanager
    def track(
        self, origin: datetime, where: tuple[str, str] | None = None
    ) -> Iterator[None]:
        """Register work on an input taken in at ``origin`` while it runs.

        ``where`` is the ``(app_id, project_id)`` it writes, when known.
        """
        token = object()
        self._tracked[token] = (origin, where)
        try:
            yield
        finally:
            self._tracked.pop(token, None)

    def tracked_before(self, at: datetime, scope: PurgeScope) -> int:
        """Tracked work on input older than ``at`` that may write ``scope``."""
        return sum(
            1
            for origin, where in self._tracked.values()
            if origin < at and (where is None or scope.covers(*where))
        )


class PurgeMarkRegistry:
    """:class:`PurgeMarks` per memory root; use the module-level :data:`MARKS`."""

    def __init__(self) -> None:
        self._by_root: dict[Path, PurgeMarks] = {}

    def for_root(self, root: Path) -> PurgeMarks:
        key = root.resolve()
        marks = self._by_root.get(key)
        if marks is None:
            marks = self._by_root[key] = PurgeMarks()
        return marks

    def reset(self) -> None:
        """Forget every root (tests, and a simulated restart)."""
        self._by_root.clear()


MARKS = PurgeMarkRegistry()

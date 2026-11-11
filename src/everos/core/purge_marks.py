"""Which scope of which memory root is being purged, and since when.

A purge raises a *mark* on its root for as long as it runs: every write to
the purged scope is refused (:func:`ensure_writable`) and background work
for it waits. It also leaves a *tombstone* -- the scope and the time ``T``
the purge was confirmed -- that outlives it: an event stamped before ``T``
that reaches the scope afterwards (an OME backlog, a replayed queue row) is
stale and must not write it back.

The table is per root and in memory; the purge service rebuilds it from
``system.db`` at startup. Use the module-level :data:`MARKS`.
"""

from __future__ import annotations

__all__ = [
    "CURRENT_EVENT",
    "MARKS",
    "PURGING",
    "EventStamp",
    "PurgeMarkRegistry",
    "PurgeMarks",
    "PurgeScope",
    "Tombstone",
    "ensure_writable",
]

import asyncio
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from everos.core.errors import PurgeInProgressError

_SYSTEM_DIRS = frozenset({".index", ".tmp"})
"""Root-level directories that are not an app (an app may still start with
a dot: ``AppScopeId`` allows ``.foo``)."""


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

    def covers_dirs(self, app_dir: str, project_dir: str) -> bool:
        """:meth:`covers`, by directory names."""
        if self.kind == "space":
            return True
        if self.kind == "app":
            return app_dir == self.app_dir
        return app_dir == self.app_dir and project_dir == self.project_dir


@dataclass(frozen=True)
class Tombstone:
    """A purge of ``scope`` confirmed at ``at`` (aware UTC)."""

    scope: PurgeScope
    at: datetime


@dataclass(frozen=True)
class EventStamp:
    """The event the current task is running for: its scope and time."""

    app_id: str
    project_id: str
    ts: datetime


PURGING: ContextVar[bool] = ContextVar("everos_purging", default=False)
"""Set by the purge itself: its own writes pass :func:`ensure_writable`."""

CURRENT_EVENT: ContextVar[EventStamp | None] = ContextVar(
    "everos_current_event", default=None
)
"""Set while a background run handles a scoped event (see the OME gate)."""


class PurgeMarks:
    """The mark and tombstones of one memory root."""

    def __init__(self) -> None:
        self._job_id: str | None = None
        self._scope: PurgeScope | None = None
        self._tombstones: list[Tombstone] = []
        self._waiters: list[asyncio.Future[None]] = []
        self.waiting_events: set[str] = set()
        """Ids of events held at the OME gate until the mark lifts."""

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
        waiters, self._waiters = self._waiters, []
        for waiter in waiters:
            if not waiter.done():
                waiter.set_result(None)

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

    def tombstone_at_dirs(self, app_dir: str, project_dir: str) -> datetime | None:
        """:meth:`tombstone_at`, by directory names."""
        return max(
            (
                t.at
                for t in self._tombstones
                if t.scope.covers_dirs(app_dir, project_dir)
            ),
            default=None,
        )

    async def wait_lifted(self) -> None:
        """Return once no mark is up (at once if none is)."""
        while self._job_id is not None:
            waiter = asyncio.get_running_loop().create_future()
            self._waiters.append(waiter)
            await waiter

    @asynccontextmanager
    async def lifted(self) -> AsyncIterator[None]:
        """Enter once no mark is up."""
        await self.wait_lifted()
        yield


class PurgeMarkRegistry:
    """Marks per memory root; use the module-level :data:`MARKS`."""

    def __init__(self) -> None:
        self._by_root: dict[Path, PurgeMarks] = {}

    def for_root(self, root: Path) -> PurgeMarks:
        key = root.resolve()
        marks = self._by_root.get(key)
        if marks is None:
            marks = self._by_root[key] = PurgeMarks()
        return marks

    def roots(self) -> list[tuple[Path, PurgeMarks]]:
        return list(self._by_root.items())

    def reset(self) -> None:
        """Forget every root (tests, and a simulated restart)."""
        self._by_root.clear()


MARKS = PurgeMarkRegistry()


def ensure_writable(target: Path) -> None:
    """Refuse a write to ``target`` that a purge has taken off limits.

    Passes the purge's own writes (:data:`PURGING`), paths outside every
    known root, and root-level files and system directories. Otherwise
    raises when the target's scope is being purged, or when the current
    background event is older than the latest purge of that scope.

    Raises:
        PurgeInProgressError: The write must not happen.
    """
    if PURGING.get():
        return
    for root, marks in MARKS.roots():
        try:
            parts = target.relative_to(root).parts
        except ValueError:
            continue
        if len(parts) < 2 or parts[0] in _SYSTEM_DIRS:
            return
        app_dir, project_dir = parts[0], parts[1]
        scope, job_id = marks.active_scope, marks.active_job_id
        if scope is not None and scope.covers_dirs(app_dir, project_dir):
            raise PurgeInProgressError(
                f"purge {job_id} in progress; poll GET /api/v2/memory/purge/{job_id}"
            )
        stamp = CURRENT_EVENT.get()
        if stamp is not None:
            purged_at = marks.tombstone_at_dirs(app_dir, project_dir)
            if purged_at is not None and purged_at > stamp.ts:
                raise PurgeInProgressError(
                    f"this event ({stamp.ts.isoformat()}) predates a purge of its"
                    f" scope at {purged_at.isoformat()}; it is not applied"
                )
        return

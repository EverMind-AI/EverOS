"""Every in-process lock EverOS takes, by name, and which task holds which.

Two kinds of entry share one table:

* **Real locks** -- :meth:`NamedLocks.hold`. One ``asyncio.Lock`` per
  ``(kind, key)``: the memorize lock of a session (``"session"``), the write
  lock of a memory owner (``"owner"``), a strategy's partition lock
  (``"partition:<strategy>"``). Their wait and hold times are histograms.
* **Observed locks** -- :meth:`NamedLocks.observe`. Locks that live elsewhere
  (a markdown file's path lock, a LanceDB table lock, a Milvus collection
  lock, a SQLite session checkout) are only *registered* here while held, so
  the acquire hook and :meth:`NamedLocks.held` see them; this module does not
  lock them. An observed lock counts as held from the moment its owner starts
  waiting for it.

Holders are recorded per ``asyncio.Task``, not per ``ContextVar``: a task
spawned while a lock is held must not appear to hold it, which a context
copied into the child would claim.

Lock objects are never evicted: a lock with pending waiters must outlive
its table entry, or a waiter and a new acquirer end up on two different
locks (CPython bpo-28427). The table grows with the distinct keys the
process has seen -- sessions, owners, partitions --
``everos_lock_table_entries`` makes that growth visible.

One process owns one memory root, so the table is process state.
"""

from __future__ import annotations

__all__ = ["LOCKS", "NamedLocks", "owner_key"]

import asyncio
import time
from collections import Counter
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager, contextmanager
from typing import Any

from everos.core.observability.metrics import Gauge, Histogram

type LockId = tuple[str, str]
type AcquireHook = Callable[[asyncio.Task[Any], LockId, frozenset[LockId]], None]

_BUCKETS = (
    0.001,
    0.002,
    0.005,
    0.01,
    0.02,
    0.05,
    0.1,
    0.25,
    0.5,
    1,
    2.5,
    5,
    10,
    15,
    30,
)
_DELETE_COUNT = "delete_count"

_wait_seconds = Histogram(
    name="lock_wait_seconds",
    description="Time spent waiting for a named lock.",
    labelnames=("kind",),
    namespace="everos",
    buckets=_BUCKETS,
)
_hold_seconds = Histogram(
    name="lock_hold_seconds",
    description="Time a named lock was held.",
    labelnames=("kind",),
    namespace="everos",
    buckets=_BUCKETS,
)
_table_entries = Gauge(
    name="lock_table_entries",
    description="Distinct keys in the named-lock and delete-count tables.",
    labelnames=("kind",),
    namespace="everos",
)


def owner_key(app_id: str, project_id: str, owner_type: str, owner_id: str) -> str:
    """The key of one memory owner's write lock and delete counter."""
    return f"{app_id}\x1f{project_id}\x1f{owner_type}\x1f{owner_id}"


class NamedLocks:
    """The process's lock table; use the module-level :data:`LOCKS`."""

    def __init__(self) -> None:
        self._locks: dict[LockId, asyncio.Lock] = {}
        self._held: dict[asyncio.Task[Any], list[LockId]] = {}
        self._delete_counts: dict[str, int] = {}
        self._sizes: Counter[str] = Counter()
        self._hook: AcquireHook | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    @asynccontextmanager
    async def hold(
        self, kind: str, key: str, *, max_wait: float | None = None
    ) -> AsyncIterator[None]:
        """Hold the lock ``(kind, key)``.

        Args:
            kind: The lock family; also the histogram label.
            key: Which lock of the family.
            max_wait: Seconds to wait for the lock before raising
                :class:`TimeoutError`; ``None`` waits forever. Only the wait
                is bounded, never the body (so not ``asyncio.timeout`` around
                the ``async with``).
        """
        lock_id = (kind, key)
        locks = self._locks_of_running_loop()
        lock = locks.get(lock_id)
        if lock is None:
            lock = locks[lock_id] = asyncio.Lock()
            self._grow(kind)
        task = self._announce(lock_id)
        started = time.monotonic()
        async with asyncio.timeout(max_wait):
            await lock.acquire()
        acquired = time.monotonic()
        _wait_seconds.labels(kind=kind).observe(acquired - started)
        self._push(task, lock_id)
        try:
            yield
        finally:
            self._pop(task, lock_id)
            lock.release()
            _hold_seconds.labels(kind=kind).observe(time.monotonic() - acquired)

    @contextmanager
    def observe(self, kind: str, key: str) -> Iterator[None]:
        """Register the current task as holding a lock kept elsewhere."""
        lock_id = (kind, key)
        task = self._announce(lock_id)
        self._push(task, lock_id)
        try:
            yield
        finally:
            self._pop(task, lock_id)

    def held(self, task: asyncio.Task[Any] | None = None) -> frozenset[LockId]:
        """The locks ``task`` (default: the current task) holds right now."""
        task = task or asyncio.current_task()
        return frozenset(self._held.get(task, ())) if task else frozenset()

    def set_acquire_hook(self, hook: AcquireHook | None) -> None:
        """Test hook, called before every acquire: ``(task, wanted, held)``."""
        self._hook = hook

    def delete_count(self, owner_key: str) -> int:
        """How many deletions have rewritten this owner's files."""
        return self._delete_counts.get(owner_key, 0)

    def bump_delete_count(self, owner_key: str) -> None:
        """Record one more deletion of this owner's files."""
        if owner_key not in self._delete_counts:
            self._grow(_DELETE_COUNT)
        self._delete_counts[owner_key] = self._delete_counts.get(owner_key, 0) + 1

    def reset(self) -> None:
        """Drop every lock, holder, counter and the hook (tests only).

        Each test runs on its own event loop; a lock carried over from an
        earlier test may be bound to a closed one.
        """
        for kind in self._sizes:
            _table_entries.labels(kind=kind).set(0)
        self._locks.clear()
        self._held.clear()
        self._delete_counts.clear()
        self._sizes.clear()
        self._hook = None
        self._loop = None

    def _locks_of_running_loop(self) -> dict[LockId, asyncio.Lock]:
        """The lock table, emptied when the running loop is not its loop.

        A lock that ever had a waiter is bound to that loop, so a table
        carried into a second loop in the same process (an application
        built twice, a CLI that runs ``asyncio.run`` twice) would fail on
        its first contention. EverOS runs its async code on one loop at a
        time, so no task of the earlier loop still holds one of these locks.
        """
        loop = asyncio.get_running_loop()
        if loop is not self._loop:
            for kind in {kind for kind, _ in self._locks}:
                del self._sizes[kind]
                _table_entries.labels(kind=kind).set(0)
            self._locks.clear()
            self._loop = loop
        return self._locks

    def _grow(self, kind: str) -> None:
        self._sizes[kind] += 1
        _table_entries.labels(kind=kind).set(self._sizes[kind])

    def _announce(self, lock_id: LockId) -> asyncio.Task[Any] | None:
        task = asyncio.current_task()
        if task is not None and self._hook is not None:
            self._hook(task, lock_id, self.held(task))
        return task

    def _push(self, task: asyncio.Task[Any] | None, lock_id: LockId) -> None:
        if task is not None:
            self._held.setdefault(task, []).append(lock_id)

    def _pop(self, task: asyncio.Task[Any] | None, lock_id: LockId) -> None:
        held = self._held.get(task) if task is not None else None
        if not held or lock_id not in held:  # reset() ran while it was held
            return
        held.remove(lock_id)
        if not held:
            del self._held[task]


LOCKS = NamedLocks()

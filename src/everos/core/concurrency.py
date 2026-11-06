"""Task-owned concurrency slots that can yield capacity while waiting for a lock.

An executor enters :class:`ConcurrencySlot` for one attempt. Cooperative locks
use :func:`acquire_without_slot` to return that task's capacity while acquiring
the lock, then reclaim it before running the critical section. Ownership is by
Task, not ContextVar: a spawned child must not release its parent's slot.
"""

from __future__ import annotations

__all__ = ["ConcurrencySlot", "acquire_without_slot"]

import asyncio
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any

_slots: dict[asyncio.Task[Any], ConcurrencySlot] = {}


class ConcurrencySlot:
    """Own at most one semaphore permit for the entering task.

    Cancellation while capacity is suspended exits without releasing a permit
    owned by somebody else. Executors can call :meth:`acquire` after a timeout
    to keep their failure bookkeeping within the same concurrency limit.
    """

    def __init__(self, semaphore: asyncio.Semaphore) -> None:
        self._semaphore = semaphore
        self._held = False
        self._owner: asyncio.Task[Any] | None = None
        self._previous: ConcurrencySlot | None = None

    async def acquire(self) -> None:
        """Reclaim this slot if a cooperative wait returned it."""
        if not self._held:
            await self._semaphore.acquire()
            self._held = True

    def release(self) -> None:
        """Return only the permit this slot currently owns."""
        if self._held:
            self._held = False
            self._semaphore.release()

    async def __aenter__(self) -> ConcurrencySlot:
        await self.acquire()
        self._owner = asyncio.current_task()
        assert self._owner is not None
        self._previous = _slots.get(self._owner)
        _slots[self._owner] = self
        return self

    async def __aexit__(self, *exc: object) -> None:
        assert self._owner is not None
        if self._previous is None:
            del _slots[self._owner]
        else:
            _slots[self._owner] = self._previous
        self.release()


@asynccontextmanager
async def acquire_without_slot[T](
    lock: AbstractAsyncContextManager[T],
) -> AsyncIterator[T]:
    """Acquire ``lock`` without occupying the current task's execution slot.

    The lock stays held while capacity is reacquired, and the body only runs
    after both have been acquired. The lock's own context manager releases it
    even if cancellation lands during that second wait. Ordinary exceptions
    restore capacity before reaching code that may catch them and continue;
    cancellation propagates without waiting for capacity to become available.
    Calls outside a slot, including child tasks, retain ordinary lock semantics.
    """
    task = asyncio.current_task()
    slot = _slots.get(task) if task is not None else None
    if slot is None:
        async with lock as value:
            yield value
        return

    slot.release()
    try:
        async with lock as value:
            await slot.acquire()
            yield value
    except Exception:
        await slot.acquire()
        raise

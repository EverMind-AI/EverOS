"""In-flight strategy-run accounting: how the engine knows it is idle.

A run is in flight from the moment it is enqueued onto APScheduler until
``OfflineEngine.dispatch_run`` finishes with it. Counting it at enqueue (not
at start) means a caller that emits and then waits sees a non-zero count at
once. APScheduler cannot answer this itself: APS 3.x
``AsyncIOExecutor.shutdown(wait=True)`` does not honour ``wait`` for async
coroutines (``apscheduler/executors/asyncio.py:24``), so ``stop()`` and
``wait_idle()`` drain through this counter instead.

Invariant: the count never goes below zero, and while armed the idle event is
set exactly when the count is zero.
"""

from __future__ import annotations

import asyncio

from everos.core.observability.logging import get_logger

logger = get_logger(__name__)


class InFlightRuns:
    """Counts enqueued-but-unfinished runs and signals when none are left.

    Args:
        engine_id: The owning engine's id, for the underflow log.
    """

    def __init__(self, engine_id: str) -> None:
        self._engine_id = engine_id
        self._active = 0
        self._idle_event: asyncio.Event | None = None

    @property
    def active(self) -> int:
        """Runs enqueued and not yet finished."""
        return self._active

    def arm(self) -> None:
        """Start signalling idleness (engine start); begins idle."""
        self._idle_event = asyncio.Event()
        self._idle_event.set()

    def disarm(self) -> None:
        """Stop signalling and forget the count (engine stop or failed start)."""
        self._idle_event = None
        self._active = 0

    def enqueued(self) -> None:
        """Bump in-flight count and mark the engine non-idle."""
        self._active += 1
        if self._idle_event is not None:
            self._idle_event.clear()

    def completed(self) -> None:
        """Drop in-flight count; mark the engine idle if the count hit zero.

        Never lets the counter dip below zero — that would mask a bookkeeping
        bug rather than fix it, and a stuck-clear idle_event would deadlock
        ``wait_idle``.
        """
        if self._active <= 0:
            logger.error(
                "active_runs_underflow",
                engine_id=self._engine_id,
            )
            self._active = 0
            if self._idle_event is not None:
                self._idle_event.set()
            return
        self._active -= 1
        if self._active == 0 and self._idle_event is not None:
            self._idle_event.set()

    async def wait_idle(self, *, timeout: float) -> bool:  # noqa: ASYNC109
        """Block until the count reaches zero; ``False`` if ``timeout`` elapses."""
        if self._idle_event is None:
            return self._active == 0
        try:
            await asyncio.wait_for(self._idle_event.wait(), timeout=timeout)
            return True
        except TimeoutError:
            return False

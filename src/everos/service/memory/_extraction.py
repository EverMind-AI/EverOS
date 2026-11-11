"""The extraction tasks a memorize call hands off, kept alive past the request."""

from __future__ import annotations

__all__ = ["ExtractionSupervisor"]

import asyncio
from typing import TYPE_CHECKING, Any

from everos.core.observability.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Collection, Coroutine

logger = get_logger("everos.service.memory")


class ExtractionSupervisor:
    """Owns the extraction tasks that may outlive their request.

    A request whose client went away is cancelled while its extraction
    runs; the memcell is already committed, so the extraction must finish.
    ``supervise`` keeps the task referenced until it ends and retrieves its
    outcome: a failure is logged with the memcell ids (it is not retried
    here -- the pipelines write markdown and emit events and are not
    idempotent, the same reason the async queue does not retry a failed
    delivery), and ``drain`` lets shutdown wait for what is still running
    instead of dropping it.

    A memory deletion is supervised the same way (``label`` names it in the
    logs): one cut short by a disconnect would leave a scope half deleted.
    """

    def __init__(self) -> None:
        self._tasks: set[asyncio.Task[Any]] = set()
        self._sessions: dict[asyncio.Task[Any], str] = {}

    def supervise[T](
        self,
        coro: Coroutine[Any, Any, T],
        *,
        session_id: str | None,
        memcell_ids: list[str],
        label: str = "memorize_extraction",
        expected: tuple[type[BaseException], ...] = (),
    ) -> asyncio.Task[T]:
        """Run ``coro`` as a task that outlives its caller.

        A failure is logged as ``<label>_failed`` unless it is one of
        ``expected`` (an answer the caller reports itself).
        """
        task: asyncio.Task[T] = asyncio.ensure_future(coro)
        self._tasks.add(task)
        if session_id is not None:
            self._sessions[task] = session_id

        def _done(t: asyncio.Task[T]) -> None:
            self._tasks.discard(t)
            self._sessions.pop(t, None)
            if t.cancelled():
                logger.error(
                    f"{label}_cancelled",
                    session_id=session_id,
                    memcell_ids=memcell_ids,
                )
                return
            exc = t.exception()
            if exc is not None and not isinstance(exc, expected):
                logger.error(
                    f"{label}_failed",
                    session_id=session_id,
                    memcell_ids=memcell_ids,
                    error=f"{type(exc).__name__}: {exc}",
                )

        task.add_done_callback(_done)
        return task

    @property
    def in_flight(self) -> int:
        return len(self._tasks)

    async def drain(self, wait_seconds: float) -> int:
        """Wait up to ``wait_seconds`` for running extractions; return what is left."""
        pending = {t for t in self._tasks if not t.done()}
        if pending:
            await asyncio.wait(pending, timeout=wait_seconds)
        return sum(1 for t in self._tasks if not t.done())

    async def settle(self, session_ids: Collection[str], wait_seconds: float) -> int:
        """Wait up to ``wait_seconds`` for what runs for ``session_ids``.

        A forget inside a group waits for the extractions of the sessions it
        touches, so every segment cut before it was recorded is written by the
        time it looks (docs/group-memory.md, 9.4). Returns how many are still
        running.
        """
        wanted = set(session_ids)
        pending = {t for t, s in self._sessions.items() if s in wanted and not t.done()}
        if pending:
            await asyncio.wait(pending, timeout=wait_seconds)
        return sum(1 for t in pending if not t.done())

    async def shutdown(self, wait_seconds: float) -> None:
        """:meth:`drain` for shutdown: log what runs, and what is abandoned."""
        running = self.in_flight
        if not running:
            return
        logger.info("memorize_extractions_draining", in_flight=running)
        left = await self.drain(wait_seconds)
        if left:
            logger.error(
                "memorize_extractions_abandoned",
                in_flight=left,
                waited_seconds=wait_seconds,
            )

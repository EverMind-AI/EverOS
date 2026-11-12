"""The extraction tasks a memorize call hands off, kept alive past the request."""

from __future__ import annotations

__all__ = ["ExtractionSupervisor"]

import asyncio
from datetime import datetime
from typing import TYPE_CHECKING, Any

from everos.core.observability.logging import get_logger
from everos.core.purge_marks import ORIGIN
from everos.core.utils.datetime import get_utc_now

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

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
        self._tasks: dict[
            asyncio.Task[Any], tuple[datetime, tuple[str, str] | None]
        ] = {}
        """Each running task: the origin of its input, the scope it writes."""

    def supervise[T](
        self,
        coro: Coroutine[Any, Any, T],
        *,
        session_id: str | None,
        memcell_ids: list[str],
        label: str = "memorize_extraction",
        expected: tuple[type[BaseException], ...] = (),
        where: tuple[str, str] | None = None,
    ) -> asyncio.Task[T]:
        """Run ``coro`` as a task that outlives its caller.

        A failure is logged as ``<label>_failed`` unless it is one of
        ``expected`` (an answer the caller reports itself). ``where`` is the
        ``(app_id, project_id)`` the task writes, when known (a purge of
        another scope does not wait for it).
        """
        task: asyncio.Task[T] = asyncio.ensure_future(coro)
        self._tasks[task] = (ORIGIN.get() or get_utc_now(), where)

        def _done(t: asyncio.Task[T]) -> None:
            self._tasks.pop(t, None)
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

    def started_before(self, at: datetime, covers: Callable[[str, str], bool]) -> int:
        """Running extractions of input taken in before ``at`` that may write
        where ``covers`` says (unknown scope: may write anywhere)."""
        return sum(
            1
            for origin, where in self._tasks.values()
            if origin < at and (where is None or covers(*where))
        )

    async def drain(self, wait_seconds: float) -> int:
        """Wait up to ``wait_seconds`` for running extractions; return what is left."""
        pending = {t for t in self._tasks if not t.done()}
        if pending:
            await asyncio.wait(pending, timeout=wait_seconds)
        return sum(1 for t in self._tasks if not t.done())

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

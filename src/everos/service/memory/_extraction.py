"""The extraction tasks a memorize call hands off, kept alive past the request."""

from __future__ import annotations

__all__ = ["ExtractionSupervisor"]

import asyncio
from typing import TYPE_CHECKING, Any

from everos.core.observability.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Coroutine

    from everos.service.memory._models import MemorizeResult

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
    """

    def __init__(self) -> None:
        self._tasks: set[asyncio.Task[MemorizeResult]] = set()

    def supervise(
        self,
        coro: Coroutine[Any, Any, MemorizeResult],
        *,
        session_id: str,
        memcell_ids: list[str],
    ) -> asyncio.Task[MemorizeResult]:
        task: asyncio.Task[MemorizeResult] = asyncio.ensure_future(coro)
        self._tasks.add(task)

        def _done(t: asyncio.Task[MemorizeResult]) -> None:
            self._tasks.discard(t)
            if t.cancelled():
                logger.error(
                    "memorize_extraction_cancelled",
                    session_id=session_id,
                    memcell_ids=memcell_ids,
                )
                return
            exc = t.exception()
            if exc is not None:
                logger.error(
                    "memorize_extraction_failed",
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

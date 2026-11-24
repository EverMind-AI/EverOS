"""The extraction tasks a memorize call hands off, kept alive past the request."""

from __future__ import annotations

__all__ = ["ExtractionSupervisor", "FinishedMemcells"]

import asyncio
from collections import Counter
from typing import TYPE_CHECKING, Any

from everos.core.observability.logging import get_logger
from everos.memory.ports import all_of, eq

if TYPE_CHECKING:
    from collections.abc import Collection, Coroutine

    from everos.memory.ports import EpisodeIndex

logger = get_logger("everos.service.memory")

_SUCCEEDED_KEPT = 10_000
"""How many memcells that extracted cleanly :meth:`ExtractionSupervisor.accounted_for`
remembers. It only has to outlast the episode index's lag (the cascade's
delay, seconds); older memcells are found in the index.

ponytail: a fixed count, not a time; at 15 memcells a second it still spans
ten minutes."""


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
        self._running: Counter[str] = Counter()
        self._succeeded: dict[str, None] = {}  # insertion-ordered set

    def accounted_for(self, memcell_ids: Collection[str]) -> set[str]:
        """Of ``memcell_ids``, those extracting now or recently extracted here.

        Recently: among the last ``_SUCCEEDED_KEPT`` that finished without
        an error in this process.
        """
        return {m for m in memcell_ids if m in self._running or m in self._succeeded}

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
        self._running.update(memcell_ids)

        def _done(t: asyncio.Task[T]) -> None:
            self._tasks.discard(t)
            self._running.subtract(memcell_ids)
            self._running = +self._running  # drop the ids no task holds
            if not t.cancelled() and t.exception() is None:
                self._succeeded.update(dict.fromkeys(memcell_ids))
                while len(self._succeeded) > _SUCCEEDED_KEPT:
                    del self._succeeded[next(iter(self._succeeded))]
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


class FinishedMemcells:
    """Which memcells a resend of their messages may be dropped for (#609).

    Finished means the memcell gets its episode without the resend: its
    extraction still runs in this process or extracted cleanly here lately
    (:meth:`ExtractionSupervisor.accounted_for`), or an episode with the
    memcell as parent is indexed (one a previous process extracted). Anything
    else -- a memcell a crash or an abandoned shutdown left without one, or
    whose extraction failed -- is not, and its messages are cut again: an
    extra memcell beats losing them.

    The one place that decides it; a delivery state kept on the memcell
    (#516) replaces it.

    ponytail: after a restart the index is the only record, and it trails the
    markdown by the cascade's delay: a memcell extracted just before a crash
    and replayed before the restarted cascade indexed its episode is cut again
    (a duplicate, as before #609). The delivery state of #516 closes it.
    """

    def __init__(
        self, extractions: ExtractionSupervisor, episodes: EpisodeIndex
    ) -> None:
        self._extractions = extractions
        self._episodes = episodes

    async def __call__(self, memcell_ids: Collection[str]) -> set[str]:
        finished = self._extractions.accounted_for(memcell_ids)
        for memcell_id in set(memcell_ids) - finished:
            parent = all_of(eq("parent_type", "memcell"), eq("parent_id", memcell_id))
            if await self._episodes.find(parent, limit=1):
                finished.add(memcell_id)
        return finished

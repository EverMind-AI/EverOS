"""``async_mode`` memorize: a durable inbox + one in-process delivery worker.

The HTTP route inserts an accepted ``/add`` or ``/flush`` body into
``memorize_inbox`` (one row, one commit) and answers 202. This worker
claims rows in ``id`` order, never two of one session at a time and at
most ``max_concurrency`` overall, and hands each to the inline add step
it was built with (the one behind :meth:`MemoryService.add`): a session's
batches and its flush run in the order the client sent them, and sessions
do not wait for each other. The per-session lock inside ``memorize`` still provides
mutual exclusion against synchronous calls; the inbox ``id`` provides the
order. A row whose delivery fails is marked ``failed`` and skipped -- its
session, and every other session, keeps moving.

Cloud runs the same shape as gateway -> msgbus -> Kafka -> consumer ->
engine, partitioned by group so groups run in parallel and each group in
order. Here the whole chain lives in one process: the route is the
gateway, the SQLite table is the topic, this worker is the consumer with
the session as the partition, and ``memorize()`` is the engine.

The worker is a private collaborator of :class:`~everos.service.MemoryService`,
which builds it over the injected inbox store and owns its start / stop.
Its stateless steps live in ``_queue_steps``, its data in ``_queue_models``.
"""

from __future__ import annotations

__all__ = ["MemorizeQueueWorker"]

import asyncio
import contextlib
import datetime as dt
from typing import Any

from everos.config import AsyncQueueSettings
from everos.core.observability.logging import get_logger
from everos.core.utils.datetime import get_utc_now
from everos.memory.models import MemorizeTask
from everos.memory.ports import MemorizeInboxStore, SessionKey

from ._queue_models import QUEUE_LOGGER, Deliver, QueueHealth
from ._queue_steps import (
    admit,
    cancel_all,
    deliver_row,
    encode_payload,
    fail_abandoned,
    purge_expired,
    queue_health,
    recover_processing,
    wait_any,
    wait_for_wake,
    write_status,
)

logger = get_logger(QUEUE_LOGGER)

# ponytail: fixed grace on stop; a setting if shutdowns ever need tuning.
_STOP_GRACE_SECONDS = 30.0
_MARK_RETRY_SECONDS = 1.0


class MemorizeQueueWorker:
    """Single consumer over ``memorize_inbox``; see the module docstring.

    ``accept_async`` is the ``[memorize.async_queue] enabled`` switch: off, the
    route ignores ``async_mode`` and runs calls inline, but the worker still
    runs so rows accepted earlier are delivered, and a session that still
    has such rows still queues its synchronous calls behind them.

    Args:
        settings: The ``[memorize.async_queue]`` section.
        inbox: The durable queue the rows live in.
        deliver: The memorize call made for each claimed row.
        accept_async: Whether new ``async_mode`` work is accepted.
        stop_grace_seconds: How long :meth:`stop` waits for the delivery in
            flight before cancelling it.
        mark_retry_seconds: The pause before retrying a failed status write.
    """

    def __init__(
        self,
        settings: AsyncQueueSettings,
        *,
        inbox: MemorizeInboxStore,
        deliver: Deliver,
        accept_async: bool = True,
        stop_grace_seconds: float = _STOP_GRACE_SECONDS,
        mark_retry_seconds: float = _MARK_RETRY_SECONDS,
    ) -> None:
        self._cfg = settings
        self._stop_grace = stop_grace_seconds
        self._mark_retry = mark_retry_seconds
        self._inbox = inbox
        self._deliver = deliver
        self.accepting = accept_async
        self._wake = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._stopping = False
        self._status_write_lost = False
        self._loop_restarts = 0
        self._last_loop_error_at: dt.datetime | None = None
        # Rows being delivered right now, by id; the scheduler's own view,
        # which is what lets the sweep tell a lost status write from a row
        # that is simply still running.
        self._in_flight: dict[int, tuple[MemorizeTask, asyncio.Task[None]]] = {}

    # ── producer side (called by the route) ────────────────────────────────

    async def has_pending(
        self,
        *,
        session_id: str,
        app_id: str = "default",
        project_id: str = "default",
    ) -> bool:
        """True when the session has rows still ``queued`` or ``processing``.

        A synchronous call on such a session must join the queue too, or it
        would run ahead of batches the client was already told are accepted.
        """
        count = await self._inbox.count_pending(
            SessionKey(app_id=app_id, project_id=project_id, session_id=session_id)
        )
        return count > 0

    async def enqueue(
        self,
        payload: dict[str, Any],
        *,
        request_id: str,
        is_final: bool,
        defer_extraction: bool,
    ) -> int:
        """Persist one request as a ``queued`` row and wake the worker.

        Returns the session's unfinished row count, this row included.
        Raises :class:`PayloadTooLargeError` (413) or
        :class:`QueueSaturatedError` (429) **before** inserting; a rejected
        batch was never accepted and the caller must keep it.
        """
        body = encode_payload(payload, max_bytes=self._cfg.max_payload_bytes)
        session = SessionKey(
            app_id=str(payload.get("app_id", "default")),
            project_id=str(payload.get("project_id", "default")),
            session_id=str(payload["session_id"]),
        )
        pending = await admit(self._inbox, session, self._cfg)
        row = await self._inbox.enqueue(
            session,
            request_id=request_id,
            is_final=is_final,
            defer_extraction=defer_extraction,
            payload_json=body,
            message_count=len(payload.get("messages") or []),
        )
        self._wake.set()
        if self._task is not None and self._task.done():
            await self.ensure_running()  # a dead loop heals on the next use
        logger.info(
            "memorize_queue_enqueued",
            row_id=row.id,
            session_id=session.session_id,
            is_final=is_final,
            pending=pending + 1,
        )
        return pending + 1

    # ── lifecycle ──────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Recover ``processing`` rows, purge expired ones, launch the loop."""
        if self._task is not None:
            return
        await recover_processing(self._inbox, max_replays=self._cfg.max_replays)
        await self._sweep()
        self._stopping = False
        self._wake.set()  # deliver whatever survived the restart right away
        self._task = asyncio.create_task(self._run(), name="memorize-queue-worker")
        self._task.add_done_callback(self._on_loop_done)
        logger.info(
            "memorize_queue_started",
            accepting=self.accepting,
            max_concurrency=self._cfg.max_concurrency,
        )

    async def stop(self) -> None:
        """Finish the rows in flight (bounded), then exit the loop.

        Cancelling mid-``memorize()`` would leave rows ``processing`` after
        their memcells may already be written; the restart would then replay
        them. Waiting keeps a normal restart exactly-once for rows that finish
        within the grace period; a longer row is replayed like a crash.
        """
        if self._task is None:
            return
        self._stopping = True
        self._wake.set()
        try:
            async with asyncio.timeout(self._stop_grace):
                await self._task
        except TimeoutError:
            logger.warning("memorize_queue_stop_grace_exceeded")
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        self._task = None
        logger.info("memorize_queue_stopped")

    # ── self-healing ───────────────────────────────────────────────────────

    def _on_loop_done(self, task: asyncio.Task[None]) -> None:
        """The loop ended: expected on ``stop()``, an error otherwise.

        A cancellation that did not come from ``stop()`` (some embedding
        code cancelling tasks wholesale) counts as the loop dying too.
        """
        if self._stopping or task is not self._task:
            return
        self._last_loop_error_at = get_utc_now()
        logger.error(
            "memorize_queue_loop_died",
            error="cancelled" if task.cancelled() else repr(task.exception()),
            restarts=self._loop_restarts,
        )
        asyncio.get_running_loop().create_task(
            self._restart_later(), name="memorize-queue-restart"
        )

    async def _restart_later(self) -> None:
        # Back off, so a loop that dies at once cannot restart in a hot loop.
        await asyncio.sleep(min(2.0**self._loop_restarts, 60.0))
        await self.ensure_running()

    async def ensure_running(self) -> bool:
        """Relaunch the loop if it died; ``True`` when it is running after.

        ``False`` while stopping, and for a worker that was never started
        (the lifespan owns the first start). Called from ``enqueue``, so the
        queue heals on its next use even before the restart scheduled by the
        loop's own exit has run.
        """
        if self._stopping or self._task is None:
            return False
        if not self._task.done():
            return True
        self._loop_restarts += 1
        await self._abandon_in_flight()
        self._task = None
        logger.warning("memorize_queue_loop_restarting", restarts=self._loop_restarts)
        await self.start()
        return True

    async def _abandon_in_flight(self) -> None:
        """Cancel the rows in flight; the next ``start()`` requeues them."""
        await cancel_all([task for _, task in self._in_flight.values()])
        self._in_flight.clear()

    @property
    def started(self) -> bool:
        """Whether the worker was started and not stopped since."""
        return self._task is not None

    async def health(self) -> QueueHealth:
        """Probe for ``GET /health``: observe, never act."""
        return await queue_health(
            self._inbox,
            clock=get_utc_now,
            alive=self._task is not None and not self._task.done(),
            stopping=self._stopping,
            accepting=self.accepting,
            loop_restarts=self._loop_restarts,
            last_loop_error_at=self._last_loop_error_at,
        )

    async def drain_once(self) -> int:
        """Deliver every ``queued`` row; return how many were started.

        Up to ``max_concurrency`` rows run at once, never two of one session:
        a session's rows go in ``id`` order, sessions run side by side.
        Returns once nothing is queued and nothing is in flight (after
        ``stop()``: once the rows in flight finished). One scheduler at a
        time: not for use while the loop is running.
        """
        started = 0
        try:
            while True:
                if self._status_write_lost:
                    await self._reap_lost()  # before the next claim, not after
                while (
                    not self._stopping
                    and len(self._in_flight) < self._cfg.max_concurrency
                ):
                    row = await self._inbox.claim_next(
                        exclude_sessions=self._busy_sessions()
                    )
                    if row is None:
                        break
                    task = asyncio.create_task(
                        self._process(row), name=f"memorize-queue-row-{row.id}"
                    )
                    self._in_flight[row.id] = (row, task)
                    started += 1
                    logger.info(
                        "memorize_queue_claimed",
                        row_id=row.id,
                        request_id=row.request_id,
                        session_id=row.session_id,
                        in_flight=len(self._in_flight),
                    )
                if not self._in_flight:
                    break
                await self._wait_for_progress()
        except asyncio.CancelledError:
            # stop() ran out of grace: abandon the rows in flight; the next
            # start() requeues them.
            await self._abandon_in_flight()
            raise
        if self._status_write_lost:
            await self._reap_lost()
        return started

    def _busy_sessions(self) -> set[SessionKey]:
        return {
            SessionKey(
                app_id=row.app_id, project_id=row.project_id, session_id=row.session_id
            )
            for row, _ in self._in_flight.values()
        }

    async def _wait_for_progress(self) -> None:
        """Block until a row in flight finishes or an enqueue wakes us.

        The wake matters when slots are free but every queued row belongs
        to a session already in flight: a new session's row must not wait
        for the longest running batch.
        """
        await wait_any([task for _, task in self._in_flight.values()], self._wake)
        for row_id, (_, task) in list(self._in_flight.items()):
            if task.done():
                del self._in_flight[row_id]
                # _process handles Exception itself; anything else means the
                # process is going down and the row stays ``processing``.
                task.result()

    # ── consumer side ──────────────────────────────────────────────────────

    async def _run(self) -> None:
        while not self._stopping:
            # The enqueue wake is the main path; the timeout is the fallback
            # sweep that also catches rows left behind by a missed wake.
            await wait_for_wake(self._wake, timeout=self._cfg.scan_interval_seconds)
            try:
                await self._sweep()
                await self.drain_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                self._last_loop_error_at = get_utc_now()
                logger.exception("memorize_queue_loop_error")

    async def _process(self, row: MemorizeTask) -> None:
        await deliver_row(row, deliver=self._deliver, mark=self._mark, cfg=self._cfg)

    async def _mark(self, row_id: int, **values: Any) -> None:
        """Write a row's status; retry a failing write a few times.

        A write that still fails leaves the row ``processing`` and raises the
        ``_status_write_lost`` flag; :meth:`_reap_lost` then marks it
        ``failed`` at the first write that succeeds, and :meth:`_sweep` is
        the fallback.
        """
        written = await write_status(
            self._inbox, row_id, retry_seconds=self._mark_retry, **values
        )
        if not written:
            self._status_write_lost = True

    async def _reap_lost(self) -> None:
        """Fail every ``processing`` row this scheduler is not running."""
        # Clear the flag before the write, not after: a row still running
        # may lose its own status write meanwhile and raise it again, and
        # that signal must survive this reap.
        self._status_write_lost = False
        if not await fail_abandoned(self._inbox, exclude_ids=list(self._in_flight)):
            self._status_write_lost = True

    async def _sweep(self) -> None:
        """Expire old rows; fail ``processing`` rows nobody is working on.

        The scheduler knows the rows it has in flight, so any other
        ``processing`` row lost its status write. The outcome is unknown (the
        lost write is usually the ``success`` one), so the row is marked
        ``failed`` rather than replayed. Housekeeping must never stop
        delivery: each step logs and moves on.
        """
        before = get_utc_now() - dt.timedelta(hours=self._cfg.row_retention_hours)
        await purge_expired(self._inbox, before=before)
        # ponytail: in-process bookkeeping; a lease column if workers ever run
        # in more than one process.
        await self._reap_lost()

"""``async_mode`` memorize: a durable inbox + one in-process delivery worker.

The HTTP route inserts an accepted ``/add`` or ``/flush`` body into
``memorize_inbox`` (one row, one commit) and answers 202. This worker
claims rows in ``id`` order, never two of one session at a time and at
most ``max_concurrency`` overall, and calls the unchanged :func:`memorize`
for each: a session's batches and its flush run in the order the client
sent them, and sessions do not wait for each other. The per-session lock
inside ``memorize()`` still provides mutual exclusion against synchronous
calls; the inbox ``id`` provides the order. A row whose delivery fails is
marked ``failed`` and skipped -- its session, and every other session,
keeps moving.

Cloud runs the same shape as gateway -> msgbus -> Kafka -> consumer ->
engine, partitioned by group so groups run in parallel and each group in
order. Here the whole chain lives in one process: the route is the
gateway, the SQLite table is the topic, this worker is the consumer with
the session as the partition, and ``memorize()`` is the engine.

External usage::

    from everos.service import MemorizeQueueWorker, get_task, list_tasks
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import json
from typing import Any, Literal

import structlog
from pydantic import BaseModel

from everos.component.utils.datetime import get_utc_now
from everos.config import AsyncQueueSettings
from everos.core.context import reset_request_id, set_request_id
from everos.core.errors import (
    InvalidInputError,
    PayloadTooLargeError,
    QueueSaturatedError,
)
from everos.core.observability.logging import get_logger
from everos.infra.persistence.sqlite import (
    LOST_STATUS_WRITE,
    REPLAYS_EXHAUSTED_PREFIX,
    MemorizeInbox,
    memorize_inbox_repo,
)
from everos.service.memorize import memorize

logger = get_logger(__name__)

_ERROR_SUMMARY_CHARS = 500
# Exceptions from these packages carry SQL text and bound parameters, i.e.
# the request body, in their message; GET /tasks must not echo that back.
_OPAQUE_ERROR_MODULES = ("sqlalchemy", "sqlite3", "aiosqlite")


def _error_summary(exc: BaseException) -> str:
    """``last_error`` for a failed batch: the type, plus the message unless
    the exception comes from the storage layer (whose messages embed SQL and
    parameters)."""
    name = type(exc).__name__
    module = type(exc).__module__ or ""
    if module.split(".", 1)[0] in _OPAQUE_ERROR_MODULES:
        return f"{name}: storage error (details in the server log)"
    return f"{name}: {exc}"[:_ERROR_SUMMARY_CHARS]


# ponytail: fixed grace on stop; a setting if shutdowns ever need tuning.
_STOP_GRACE_SECONDS = 30.0
# A status write that fails (sqlite busy past its timeout, disk full) is
# retried this many times before the row is left for the stuck-row sweep.
_MARK_ATTEMPTS = 3
_MARK_RETRY_SECONDS = 1.0


TaskStatus = Literal["queued", "processing", "success", "failed"]
TASK_STATUSES: tuple[str, ...] = ("queued", "processing", "success", "failed")


class TaskItem(BaseModel):
    """``GET /tasks/{task_id}`` -- one accepted request, in the Cloud's shape.

    ``id`` is the request id of the accepting call. Cloud also emits
    ``pending`` (accepted, boundary not run yet); this build never does, so a
    client written for Cloud sees a subset of the statuses it already handles.
    """

    id: str
    status: TaskStatus
    task_type: Literal["memory_add", "memory_flush"]
    error: str | None = None
    error_code: str | None = None
    created_at: dt.datetime
    finished_at: dt.datetime | None = None


class TaskPage(BaseModel):
    """``GET /tasks`` -- one page of :class:`TaskItem`, newest first."""

    total: int
    page: int
    page_size: int
    items: list[TaskItem]


class QueueHealth(BaseModel):
    """The ``memorize_queue`` block of ``GET /health``.

    ``healthy`` is the delivery loop being alive. The probe only observes:
    a dead loop is reported as such, and it is the loop's own exit callback
    (and the next ``enqueue``) that relaunches it, so ``loop_restarts``
    counts every time that happened. The counts are for alerting: a growing
    ``oldest_pending_age_s`` with a live loop means delivery is slower than
    arrival (or the LLM is down).
    """

    healthy: bool
    reasons: list[str]
    loop_alive: bool
    accepting: bool
    pending: int
    oldest_pending_age_s: float | None
    failed: int
    loop_restarts: int
    last_loop_error_at: dt.datetime | None


class MemorizeQueueWorker:
    """Single consumer over ``memorize_inbox``; see the module docstring.

    ``accept_async`` is the ``[memorize.async_queue] enabled`` switch: off, the
    route ignores ``async_mode`` and runs calls inline, but the worker still
    runs so rows accepted earlier are delivered, and a session that still
    has such rows still queues its synchronous calls behind them.
    """

    def __init__(self, settings: AsyncQueueSettings, *, accept_async: bool = True):
        self._cfg = settings
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
        self._in_flight: dict[int, tuple[MemorizeInbox, asyncio.Task[None]]] = {}

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
        count = await memorize_inbox_repo.count_pending(
            session_id=session_id, app_id=app_id, project_id=project_id
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
        body = json.dumps(payload, ensure_ascii=False)
        try:
            size = len(body.encode("utf-8"))
        except UnicodeEncodeError as exc:
            raise InvalidInputError(
                "payload contains a lone surrogate character (not valid UTF-8)"
            ) from exc
        if size > self._cfg.max_payload_bytes:
            raise PayloadTooLargeError(
                f"payload is {size} bytes, limit is {self._cfg.max_payload_bytes}"
            )
        session_id = str(payload["session_id"])
        app_id = str(payload.get("app_id", "default"))
        project_id = str(payload.get("project_id", "default"))
        pending = await memorize_inbox_repo.count_pending(
            session_id=session_id, app_id=app_id, project_id=project_id
        )
        if pending >= self._cfg.max_pending_per_session:
            raise QueueSaturatedError(
                f"memorize queue saturated for session {session_id}: {pending} pending"
            )
        total = await memorize_inbox_repo.count_pending()
        if total >= self._cfg.max_pending_total:
            raise QueueSaturatedError(
                f"memorize queue saturated: {total} pending across all sessions"
            )
        row = await memorize_inbox_repo.add(
            MemorizeInbox(
                request_id=request_id,
                app_id=app_id,
                project_id=project_id,
                session_id=session_id,
                is_final=is_final,
                defer_extraction=defer_extraction,
                payload_json=body,
                message_count=len(payload.get("messages") or []),
            )
        )
        self._wake.set()
        if self._task is not None and self._task.done():
            await self.ensure_running()  # a dead loop heals on the next use
        logger.info(
            "memorize_queue_enqueued",
            row_id=row.id,
            session_id=session_id,
            is_final=is_final,
            pending=pending + 1,
        )
        return pending + 1

    # ── lifecycle ──────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Recover ``processing`` rows, purge expired ones, launch the loop."""
        if self._task is not None:
            return
        requeued, gave_up = await memorize_inbox_repo.requeue_processing(
            max_replays=self._cfg.max_replays
        )
        if requeued:
            logger.info("memorize_queue_requeued_processing", count=requeued)
        if gave_up:
            logger.error(
                "memorize_queue_replays_exhausted",
                count=gave_up,
                max_replays=self._cfg.max_replays,
            )
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
            async with asyncio.timeout(_STOP_GRACE_SECONDS):
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
        for _, task in self._in_flight.values():
            task.cancel()
        await asyncio.gather(
            *(task for _, task in self._in_flight.values()), return_exceptions=True
        )
        self._in_flight.clear()

    async def health(self) -> QueueHealth:
        """Probe for ``GET /health``: observe, never act."""
        reasons: list[str] = []
        alive = self._task is not None and not self._task.done()
        if not alive:
            reasons.append(
                "delivery loop is stopping"
                if self._stopping
                else "delivery loop is not running"
            )
        pending = await memorize_inbox_repo.count_pending()
        oldest = await memorize_inbox_repo.oldest_pending_created_at()
        age = (
            None
            if oldest is None
            else max(0.0, (get_utc_now() - oldest).total_seconds())
        )
        return QueueHealth(
            healthy=alive,
            reasons=reasons,
            loop_alive=alive,
            accepting=self.accepting,
            pending=pending,
            oldest_pending_age_s=age,
            failed=await memorize_inbox_repo.count_status("failed"),
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
                    row = await memorize_inbox_repo.claim_next(
                        exclude_sessions=self._busy_sessions()
                    )
                    if row is None:
                        break
                    assert row.id is not None
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

    def _busy_sessions(self) -> set[tuple[str, str, str]]:
        return {
            (row.app_id, row.project_id, row.session_id)
            for row, _ in self._in_flight.values()
        }

    async def _wait_for_progress(self) -> None:
        """Block until a row in flight finishes or an enqueue wakes us.

        The wake matters when slots are free but every queued row belongs
        to a session already in flight: a new session's row must not wait
        for the longest running batch.
        """
        tasks = [task for _, task in self._in_flight.values()]
        waker = asyncio.ensure_future(self._wake.wait())
        done, _ = await asyncio.wait(
            [*tasks, waker], return_when=asyncio.FIRST_COMPLETED
        )
        if waker in done:
            self._wake.clear()
        else:
            waker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await waker
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
            try:
                async with asyncio.timeout(self._cfg.scan_interval_seconds):
                    await self._wake.wait()
            except TimeoutError:
                pass
            self._wake.clear()
            try:
                await self._sweep()
                await self.drain_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                self._last_loop_error_at = get_utc_now()
                logger.exception("memorize_queue_loop_error")

    async def _process(self, row: MemorizeInbox) -> None:
        assert row.id is not None
        # Carry the accepting request's id so spans and logs emitted inside
        # memorize() correlate with the 202 the client received.
        token = set_request_id(row.request_id)
        structlog.contextvars.bind_contextvars(request_id=row.request_id)
        try:
            for attempt in range(1, self._cfg.max_attempts + 1):
                try:
                    # Inside the try: a row whose body was hand-edited into
                    # garbage fails with that reason, not the whole pass.
                    payload = json.loads(row.payload_json)
                    await memorize(
                        payload,
                        is_final=row.is_final,
                        defer_extraction=row.defer_extraction,
                    )
                except Exception as exc:
                    error = _error_summary(exc)
                    if attempt < self._cfg.max_attempts:
                        await self._mark(
                            row.id,
                            status="processing",
                            attempt=attempt,
                            last_error=error,
                        )
                        logger.warning(
                            "memorize_queue_retry",
                            row_id=row.id,
                            session_id=row.session_id,
                            attempt=attempt,
                            error=error,
                        )
                        await asyncio.sleep(self._cfg.retry_backoff_seconds)
                        continue
                    # One write: a lost second write would let the sweep overwrite
                    # a known failure with "outcome unknown".
                    await self._mark(
                        row.id, status="failed", attempt=attempt, last_error=error
                    )
                    logger.error(
                        "memorize_queue_failed",
                        row_id=row.id,
                        session_id=row.session_id,
                        attempts=attempt,
                        message_count=row.message_count,
                        error=error,
                    )
                    return
                # Drop the body: nothing reads a delivered row's payload, and
                # keeping it would hold seven days of request bodies on disk.
                await self._mark(
                    row.id, status="success", attempt=attempt, payload_json=""
                )
                logger.info(
                    "memorize_queue_delivered",
                    row_id=row.id,
                    session_id=row.session_id,
                    attempt=attempt,
                )
                return
        finally:
            structlog.contextvars.unbind_contextvars("request_id")
            reset_request_id(token)

    async def _mark(self, row_id: int, **values: Any) -> None:
        """Write a row's status; retry a failing write a few times.

        A write that still fails leaves the row ``processing`` and raises the
        ``_status_write_lost`` flag; :meth:`_reap_lost` then marks it
        ``failed`` at the first write that succeeds, and :meth:`_sweep` is
        the fallback.
        """
        for attempt in range(1, _MARK_ATTEMPTS + 1):
            try:
                await memorize_inbox_repo.mark(row_id, **values)
                return
            except Exception:
                if attempt == _MARK_ATTEMPTS:
                    self._status_write_lost = True
                    logger.exception(
                        "memorize_queue_status_write_lost",
                        row_id=row_id,
                        status=values.get("status"),
                    )
                    return
                await asyncio.sleep(_MARK_RETRY_SECONDS)

    async def _reap_lost(self) -> None:
        """Fail every ``processing`` row this scheduler is not running."""
        # Clear the flag before the write, not after: a row still running
        # may lose its own status write meanwhile and raise it again, and
        # that signal must survive this reap.
        self._status_write_lost = False
        try:
            stuck = await memorize_inbox_repo.fail_abandoned(
                exclude_ids=list(self._in_flight)
            )
        except Exception:
            self._status_write_lost = True
            logger.exception("memorize_queue_reap_failed")
            return
        if stuck:
            logger.warning("memorize_queue_failed_abandoned", count=stuck)

    async def _sweep(self) -> None:
        """Expire old rows; fail ``processing`` rows nobody is working on.

        The scheduler knows the rows it has in flight, so any other
        ``processing`` row lost its status write. The outcome is unknown (the
        lost write is usually the ``success`` one), so the row is marked
        ``failed`` rather than replayed. Housekeeping must never stop
        delivery: each step logs and moves on.
        """
        before = get_utc_now() - dt.timedelta(hours=self._cfg.row_retention_hours)
        try:
            deleted, undelivered = await memorize_inbox_repo.purge_expired(
                before=before
            )
        except Exception:
            logger.exception("memorize_queue_purge_failed")
        else:
            if undelivered:
                # Rows aged out without ever being delivered: the worker was
                # down (or disabled) for longer than the retention window.
                logger.warning(
                    "memorize_queue_purged_undelivered",
                    count=undelivered,
                    deleted=deleted,
                )
            elif deleted:
                logger.info("memorize_queue_purged", count=deleted)
        # ponytail: in-process bookkeeping; a lease column if workers ever run
        # in more than one process.
        await self._reap_lost()


# ── task readers (no worker needed; plain table reads) ─────────────────────


def _task_item(row: MemorizeInbox) -> TaskItem:
    terminal = row.status in ("success", "failed")
    error_code: str | None = None
    if row.last_error == LOST_STATUS_WRITE:
        error_code = "status_write_lost"
    elif row.last_error and row.last_error.startswith(REPLAYS_EXHAUSTED_PREFIX):
        error_code = "replays_exhausted"
    elif row.last_error and ": " in row.last_error:
        error_code = row.last_error.split(": ", 1)[0]  # the exception type
    return TaskItem(
        id=row.request_id,
        status=row.status,  # type: ignore[arg-type]
        task_type="memory_flush" if row.is_final else "memory_add",
        error=row.last_error,
        error_code=error_code,
        created_at=row.created_at,
        finished_at=row.updated_at if terminal else None,
    )


async def get_task(request_id: str) -> TaskItem | None:
    row = await memorize_inbox_repo.find_one(request_id=request_id)
    return None if row is None else _task_item(row)


async def list_tasks(
    *,
    session_id: str | None = None,
    status: str | None = None,
    page: int = 1,
    page_size: int = 20,
    app_id: str = "default",
    project_id: str = "default",
) -> TaskPage:
    rows, total = await memorize_inbox_repo.list_tasks(
        session_id=session_id,
        status=status,
        app_id=app_id,
        project_id=project_id,
        offset=(page - 1) * page_size,
        limit=page_size,
    )
    return TaskPage(
        total=total, page=page, page_size=page_size, items=[_task_item(r) for r in rows]
    )

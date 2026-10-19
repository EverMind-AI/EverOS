"""``async_mode`` memorize: a durable inbox + one in-process delivery worker.

The HTTP route inserts an accepted ``/add`` or ``/flush`` body into
``memorize_inbox`` (one row, one commit) and answers 202. This worker
claims rows in ``id`` order and calls the unchanged :func:`memorize` for
each, so a session's batches and its flush run in the order the client
sent them. The per-session lock inside ``memorize()`` still provides
mutual exclusion against synchronous calls; the inbox ``id`` provides the
order. A row whose delivery fails is marked ``failed`` and skipped -- its
session, and every other session, keeps moving.

Cloud runs the same shape as gateway -> msgbus -> Kafka -> consumer ->
engine. Here the whole chain lives in one process: the route is the
gateway, the SQLite table is the topic, this worker is the consumer, and
``memorize()`` is the engine.

External usage::

    from everos.service import (
        MemorizeQueueWorker, get_request_status, get_session_status,
    )
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import json
from typing import Any, Literal

import structlog
from pydantic import BaseModel

from everos.component.utils.datetime import get_utc_now, to_timestamp_ms
from everos.config import AsyncQueueSettings
from everos.core.context import reset_request_id, set_request_id
from everos.core.errors import (
    InvalidInputError,
    PayloadTooLargeError,
    QueueSaturatedError,
)
from everos.core.observability.logging import get_logger
from everos.infra.persistence.sqlite import (
    MemorizeInbox,
    memorize_inbox_repo,
    unprocessed_buffer_repo,
)
from everos.service._boundary import _TRACK
from everos.service.memorize import memorize

logger = get_logger(__name__)

_ERROR_SUMMARY_CHARS = 500
# ponytail: fixed grace on stop; a setting if shutdowns ever need tuning.
_STOP_GRACE_SECONDS = 30.0
# A status write that fails (sqlite busy past its timeout, disk full) is
# retried this many times before the row is left for the stuck-row sweep.
_MARK_ATTEMPTS = 3
_MARK_RETRY_SECONDS = 1.0


class QueueRequestStatus(BaseModel):
    """``GET /status?request_id=`` -- one accepted request's outcome."""

    status: Literal["queued", "processing", "success", "failed"]
    session_id: str
    message_count: int
    attempt: int
    last_error: str | None
    created_at: dt.datetime
    updated_at: dt.datetime


class QueueSessionStatus(BaseModel):
    """``GET /status?session_id=`` -- how much of a session is still in flight.

    ``pending == 0`` means every batch reached ``memorize()``; it does not
    mean every message became an episode -- ``buffered`` counts messages
    still waiting in ``unprocessed_buffer`` for a boundary or a flush.
    """

    pending: int
    oldest_pending_age_ms: int | None
    failed: int
    buffered: int


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
        requeued = await memorize_inbox_repo.requeue_processing()
        if requeued:
            logger.info("memorize_queue_requeued_processing", count=requeued)
        await self._sweep()
        self._stopping = False
        self._wake.set()  # deliver whatever survived the restart right away
        self._task = asyncio.create_task(self._run(), name="memorize-queue-worker")
        logger.info("memorize_queue_started", accepting=self.accepting)

    async def stop(self) -> None:
        """Finish the row in flight (bounded), then exit the loop.

        Cancelling mid-``memorize()`` would leave a row ``processing`` after
        its memcell may already be written; the restart would then replay
        it. Waiting keeps a normal restart exactly-once for rows that finish
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

    async def drain_once(self) -> int:
        """Deliver every ``queued`` row in ``id`` order; return how many.

        One consumer at a time: not for use while the loop is running.
        """
        delivered = 0
        while not self._stopping:
            if self._status_write_lost:
                await self._reap_lost()  # before the next batch, not after it
            row = await memorize_inbox_repo.claim_next()
            if row is None:
                break
            await self._process(row)
            delivered += 1
        if self._status_write_lost:
            await self._reap_lost()
        return delivered

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
                logger.exception("memorize_queue_loop_error")

    async def _process(self, row: MemorizeInbox) -> None:
        assert row.id is not None
        payload = json.loads(row.payload_json)
        # Carry the accepting request's id so spans and logs emitted inside
        # memorize() correlate with the 202 the client received.
        token = set_request_id(row.request_id)
        structlog.contextvars.bind_contextvars(request_id=row.request_id)
        try:
            for attempt in range(1, self._cfg.max_attempts + 1):
                try:
                    await memorize(
                        payload,
                        is_final=row.is_final,
                        defer_extraction=row.defer_extraction,
                    )
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"[:_ERROR_SUMMARY_CHARS]
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
        """Fail every ``processing`` row; only called when no row is in flight."""
        try:
            stuck = await memorize_inbox_repo.fail_abandoned()
        except Exception:
            logger.exception("memorize_queue_reap_failed")
            return
        self._status_write_lost = False
        if stuck:
            logger.warning("memorize_queue_failed_abandoned", count=stuck)

    async def _sweep(self) -> None:
        """Expire old rows; fail ``processing`` rows nobody is working on.

        Runs only between rows (never during ``_process``), so with one
        worker every ``processing`` row it sees lost its status write. The
        outcome is unknown (the lost write is usually the ``success`` one), so
        the row is marked ``failed`` rather than replayed. Housekeeping must
        never stop delivery: each step logs and moves on.
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
        # ponytail: single-worker sweep; a lease column if workers ever run in
        # more than one process.
        await self._reap_lost()


# ── status readers (no worker needed; plain table reads) ───────────────────


async def get_request_status(request_id: str) -> QueueRequestStatus | None:
    row = await memorize_inbox_repo.find_one(request_id=request_id)
    if row is None:
        return None
    return QueueRequestStatus(
        status=row.status,
        session_id=row.session_id,
        message_count=row.message_count,
        attempt=row.attempt,
        last_error=row.last_error,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


async def get_session_status(
    *,
    session_id: str,
    app_id: str = "default",
    project_id: str = "default",
) -> QueueSessionStatus:
    summary = await memorize_inbox_repo.session_summary(
        session_id=session_id, app_id=app_id, project_id=project_id
    )
    buffered = await unprocessed_buffer_repo.list_for_track(
        session_id, _TRACK, app_id=app_id, project_id=project_id
    )
    oldest = summary.oldest_pending_created_at
    age_ms = (
        None
        if oldest is None
        else max(0, to_timestamp_ms(get_utc_now()) - to_timestamp_ms(oldest))
    )
    return QueueSessionStatus(
        pending=summary.pending,
        oldest_pending_age_ms=age_ms,
        failed=summary.failed,
        buffered=len(buffered),
    )

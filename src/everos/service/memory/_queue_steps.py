"""The queue worker's stateless steps: admission, delivery, status, upkeep.

Each step takes the inbox (and the settings it needs) as arguments and
holds no state, so :class:`~everos.service.memory._queue.MemorizeQueueWorker`
keeps only the scheduling: what is in flight, when the loop runs, how it
heals.
"""

from __future__ import annotations

__all__ = [
    "admit",
    "cancel_all",
    "deliver_row",
    "encode_payload",
    "fail_abandoned",
    "purge_expired",
    "queue_health",
    "recover_processing",
    "wait_any",
    "wait_for_wake",
    "write_status",
]

import asyncio
import contextlib
import datetime as dt
import json
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from everos.config import AsyncQueueSettings
from everos.core.context import reset_request_id, set_request_id
from everos.core.errors import (
    InvalidInputError,
    PayloadTooLargeError,
    QueueSaturatedError,
)
from everos.core.observability.logging import bound_log_context, get_logger
from everos.memory.models import MemorizeTask
from everos.memory.ports import MemorizeInboxStore, SessionKey

from ._queue_models import QUEUE_LOGGER, Deliver, QueueHealth, error_summary

logger = get_logger(QUEUE_LOGGER)

# A status write that fails (sqlite busy past its timeout, disk full) is
# retried this many times before the row is left for the stuck-row sweep.
_MARK_ATTEMPTS = 3


def encode_payload(payload: dict[str, Any], *, max_bytes: int) -> str:
    """The row body for ``payload``.

    Raises:
        InvalidInputError: The payload is not valid UTF-8.
        PayloadTooLargeError: The body is over ``max_bytes``.
    """
    body = json.dumps(payload, ensure_ascii=False)
    try:
        size = len(body.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise InvalidInputError(
            "payload contains a lone surrogate character (not valid UTF-8)"
        ) from exc
    if size > max_bytes:
        raise PayloadTooLargeError(f"payload is {size} bytes, limit is {max_bytes}")
    return body


async def admit(
    inbox: MemorizeInboxStore, session: SessionKey, cfg: AsyncQueueSettings
) -> int:
    """The session's unfinished row count, once the queue has room for one more.

    Raises:
        QueueSaturatedError: The session or the whole queue is at its cap.
    """
    pending = await inbox.count_pending(session)
    if pending >= cfg.max_pending_per_session:
        raise QueueSaturatedError(
            f"memorize queue saturated for session {session.session_id}: "
            f"{pending} pending"
        )
    total = await inbox.count_pending()
    if total >= cfg.max_pending_total:
        raise QueueSaturatedError(
            f"memorize queue saturated: {total} pending across all sessions"
        )
    return pending


async def recover_processing(inbox: MemorizeInboxStore, *, max_replays: int) -> None:
    """Requeue the rows a previous process left ``processing``."""
    requeued, gave_up = await inbox.requeue_processing(max_replays=max_replays)
    if requeued:
        logger.info("memorize_queue_requeued_processing", count=requeued)
    if gave_up:
        logger.error(
            "memorize_queue_replays_exhausted", count=gave_up, max_replays=max_replays
        )


async def write_status(
    inbox: MemorizeInboxStore, row_id: int, *, retry_seconds: float, **values: Any
) -> bool:
    """Write a row's status, retrying a failing write; ``False`` when it was lost."""
    for attempt in range(1, _MARK_ATTEMPTS + 1):
        try:
            await inbox.mark(row_id, **values)
            return True
        except Exception:
            if attempt == _MARK_ATTEMPTS:
                logger.exception(
                    "memorize_queue_status_write_lost",
                    row_id=row_id,
                    status=values.get("status"),
                )
                return False
            await asyncio.sleep(retry_seconds)
    return False


async def purge_expired(inbox: MemorizeInboxStore, *, before: dt.datetime) -> None:
    """Delete rows older than ``before``; log, never raise."""
    try:
        deleted, undelivered = await inbox.purge_expired(before=before)
    except Exception:
        logger.exception("memorize_queue_purge_failed")
        return
    if undelivered:
        # Rows aged out without ever being delivered: the worker was down
        # (or disabled) for longer than the retention window.
        logger.warning(
            "memorize_queue_purged_undelivered", count=undelivered, deleted=deleted
        )
    elif deleted:
        logger.info("memorize_queue_purged", count=deleted)


async def queue_health(
    inbox: MemorizeInboxStore,
    *,
    clock: Callable[[], dt.datetime],
    alive: bool,
    stopping: bool,
    accepting: bool,
    loop_restarts: int,
    last_loop_error_at: dt.datetime | None,
) -> QueueHealth:
    """The ``GET /health`` block for the worker's state and the inbox counts."""
    reasons: list[str] = []
    if not alive:
        reasons.append(
            "delivery loop is stopping" if stopping else "delivery loop is not running"
        )
    pending = await inbox.count_pending()
    oldest = await inbox.oldest_pending_at()
    return QueueHealth(
        healthy=alive,
        reasons=reasons,
        loop_alive=alive,
        accepting=accepting,
        pending=pending,
        oldest_pending_age_s=(
            None if oldest is None else max(0.0, (clock() - oldest).total_seconds())
        ),
        failed=await inbox.count_status("failed"),
        loop_restarts=loop_restarts,
        last_loop_error_at=last_loop_error_at,
    )


async def deliver_row(
    row: MemorizeTask,
    *,
    deliver: Deliver,
    mark: Callable[..., Awaitable[None]],
    cfg: AsyncQueueSettings,
) -> None:
    """Deliver one claimed row, retrying up to ``max_attempts``; record the end.

    ``mark`` writes the row's status (the worker's, which tracks lost writes).
    """
    # Carry the accepting request's id so spans and logs emitted inside
    # memorize() correlate with the 202 the client received.
    token = set_request_id(row.request_id)
    try:
        with bound_log_context(request_id=row.request_id):
            await _deliver_with_retries(row, deliver=deliver, mark=mark, cfg=cfg)
    finally:
        reset_request_id(token)


async def _deliver_with_retries(
    row: MemorizeTask,
    *,
    deliver: Deliver,
    mark: Callable[..., Awaitable[None]],
    cfg: AsyncQueueSettings,
) -> None:
    assert row.id is not None
    for attempt in range(1, cfg.max_attempts + 1):
        try:
            # Inside the try: a row whose body was hand-edited into
            # garbage fails with that reason, not the whole pass.
            payload = json.loads(row.payload_json)
            await deliver(
                payload,
                is_final=row.is_final,
                defer_extraction=row.defer_extraction,
            )
        except Exception as exc:
            error = error_summary(exc)
            if attempt < cfg.max_attempts:
                await mark(
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
                await asyncio.sleep(cfg.retry_backoff_seconds)
                continue
            # One write: a lost second write would let the sweep overwrite
            # a known failure with "outcome unknown".
            await mark(row.id, status="failed", attempt=attempt, last_error=error)
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
        await mark(row.id, status="success", attempt=attempt, payload_json="")
        logger.info(
            "memorize_queue_delivered",
            row_id=row.id,
            session_id=row.session_id,
            attempt=attempt,
        )
        return


async def fail_abandoned(inbox: MemorizeInboxStore, *, exclude_ids: list[int]) -> bool:
    """Fail every ``processing`` row not in ``exclude_ids``; ``False`` on error."""
    try:
        stuck = await inbox.fail_abandoned(exclude_ids=exclude_ids)
    except Exception:
        logger.exception("memorize_queue_reap_failed")
        return False
    if stuck:
        logger.warning("memorize_queue_failed_abandoned", count=stuck)
    return True


async def cancel_all(tasks: Sequence[asyncio.Task[None]]) -> None:
    """Cancel ``tasks`` and wait until every one of them has ended."""
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


async def wait_any(tasks: Sequence[asyncio.Task[None]], wake: asyncio.Event) -> None:
    """Return once one of ``tasks`` finishes or ``wake`` is set (and clear it)."""
    waker = asyncio.ensure_future(wake.wait())
    done, _ = await asyncio.wait([*tasks, waker], return_when=asyncio.FIRST_COMPLETED)
    if waker in done:
        wake.clear()
    else:
        waker.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await waker


async def wait_for_wake(wake: asyncio.Event, *, timeout: float) -> None:  # noqa: ASYNC109
    """Return once ``wake`` is set or ``timeout`` passed, then clear it."""
    try:
        async with asyncio.timeout(timeout):
            await wake.wait()
    except TimeoutError:
        pass
    wake.clear()

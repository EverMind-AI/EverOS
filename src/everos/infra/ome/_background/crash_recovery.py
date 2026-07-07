"""Startup crash recovery — stale RUNNING rows → CRASHED + re-enqueue.

Runs once at engine.start() before normal dispatching begins. Rows
whose started_at is older than ``timeout_seconds`` are marked CRASHED
and re-enqueued with a fresh run_id reusing the original event payload.
Fresher RUNNING rows are skipped — APScheduler's own jobstore may have
already reattached them.

At-most-once: ``mark_crashed`` and ``add_job`` are not atomic. If
``add_job`` fails after ``mark_crashed``, the row stays CRASHED and
the event is lost. Strategies needing at-least-once must add their own
retry / monitor layer.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import timedelta
from uuid import uuid4

from everos.component.utils.datetime import get_utc_now
from everos.core.observability.logging import get_logger
from everos.infra.ome._stores.run_record import RunRecordStore

logger = get_logger(__name__)


async def scan_and_resume(
    *,
    run_record_store: RunRecordStore,
    timeout_seconds: int,
    add_job: Callable[[str, str, str, str, int], Awaitable[None]],
) -> None:
    """Scan ``run_record`` for stale RUNNING rows, mark them CRASHED, and
    re-enqueue each via ``add_job``. See module docstring for the
    at-most-once caveat.

    ``add_job`` is called with positional args
    ``(strategy_name, run_id, event_topic, event_payload, max_retries)``.

    Raises:
        ValueError: If ``timeout_seconds`` is not positive.
    """
    if timeout_seconds <= 0:
        raise ValueError(f"timeout_seconds must be > 0, got {timeout_seconds}")
    now = get_utc_now()
    cutoff = now - timedelta(seconds=timeout_seconds)
    running = await run_record_store.find_running()
    for rec in running:
        if rec.started_at >= cutoff:
            continue
        await run_record_store.mark_crashed(
            run_id=rec.run_id,
            finished_at=now,
            error="crash recovery: marked CRASHED after start scan",
        )
        new_run_id = uuid4().hex
        try:
            await add_job(
                rec.strategy_name,
                new_run_id,
                rec.event_topic,
                rec.event_payload,
                rec.max_retries_snapshot,
            )
            logger.info(
                "crash_recovery_resumed",
                strategy_name=rec.strategy_name,
                event_topic=rec.event_topic,
                old_run_id=rec.run_id,
                new_run_id=new_run_id,
            )
        except Exception:
            logger.exception(
                "crash_recovery_resume_failed",
                strategy_name=rec.strategy_name,
                event_topic=rec.event_topic,
                old_run_id=rec.run_id,
            )

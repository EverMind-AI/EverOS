"""Startup crash recovery — every RUNNING row → CRASHED + re-enqueue.

Runs once at ``engine.start()`` before normal dispatching begins, and
takes every RUNNING row as a crash of a previous process: ``start``
acquires the jobstore's exclusive lock first, so no run of an earlier
engine on this jobstore can still be alive, whatever its age. APScheduler
is no help here -- it removes a one-shot job from its jobstore when it
*submits* it, not when the coroutine finishes, so while a strategy runs
its RUNNING row is the only persistent trace (#540: an age filter used to
skip rows younger than 30 minutes, and a run cut short by a restart inside
that window was never looked at again).

At-most-once: ``mark_crashed`` and ``add_job`` are not atomic. If
``add_job`` fails after ``mark_crashed``, the row stays CRASHED and
the event is lost. Strategies needing at-least-once must add their own
retry / monitor layer.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from uuid import uuid4

from everos.core.observability.logging import get_logger
from everos.core.utils.datetime import get_utc_now
from everos.infra.ome._stores.run_record import RunRecordStore

logger = get_logger(__name__)


async def scan_and_resume(
    *,
    run_record_store: RunRecordStore,
    add_job: Callable[[str, str, str, str, int], Awaitable[None]],
) -> None:
    """Mark every RUNNING row CRASHED and re-enqueue each via ``add_job``.

    See the module docstring for why age does not matter and for the
    at-most-once caveat.

    ``add_job`` is called with positional args
    ``(strategy_name, run_id, event_topic, event_payload, max_retries)``.
    """
    now = get_utc_now()
    running = await run_record_store.find_running()
    for rec in running:
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

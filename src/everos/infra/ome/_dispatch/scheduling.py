"""The APScheduler boundary: how OME jobs are added and how they call back.

APScheduler's SQLAlchemy jobstore persists every job, callable included, so a
job outlives the process that added it. The callables therefore have to be
importable module-level functions with plain arguments — no bound methods, no
closures. The three entry functions here are those callables: each looks the
live engine up in :data:`ENGINES` by id and hands over to it. The ``add_*`` and
``reschedule_*`` helpers are the only places that build jobs, so the job ids
(``cron::<name>``, ``idle::<name>``, the run id) and argument tuples have one
definition.

APScheduler stores a callable as ``<module>:<qualname>``. Jobs written before
these functions lived here name ``everos.infra.ome.engine``, which therefore
keeps importing them.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

from apscheduler.executors.asyncio import AsyncIOExecutor
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from everos.core.observability.logging import get_logger
from everos.core.utils.datetime import get_utc_now
from everos.infra.ome._dispatch._state import _CURRENT_STRATEGY
from everos.infra.ome._dispatch.strategy import StrategyMeta
from everos.memory.ports import Cron, CronTick, Idle

if TYPE_CHECKING:
    from everos.infra.ome.engine import OfflineEngine

logger = get_logger(__name__)

ENGINES: dict[str, OfflineEngine] = {}
"""Running engines by id: how a persisted job finds the engine it belongs to.
An engine registers itself on start and leaves on stop."""


async def runner_entry(
    engine_id: str,
    strategy_name: str,
    run_id: str,
    event_topic: str,
    event_payload: str,
    max_retries_snapshot: int,
    traceparent: str = "",
) -> None:
    """Module-level APS jobstore callback for a single run.

    Looks the engine up by id and hands off to
    :meth:`OfflineEngine.dispatch_run`. Pickle-safe (no closures, no
    bound methods captured into APS jobstore args). ``traceparent`` defaults
    to "" so crash-recovered jobs enqueued before this field existed still
    unpack (they simply root their own trace).
    """
    engine = ENGINES.get(engine_id)
    if engine is None:
        logger.error(
            "no_engine_for_runner",
            engine_id=engine_id,
            run_id=run_id,
        )
        return
    await engine.dispatch_run(
        strategy_name=strategy_name,
        run_id=run_id,
        event_topic=event_topic,
        event_payload=event_payload,
        max_retries_snapshot=max_retries_snapshot,
        traceparent=traceparent,
    )


async def cron_entry(engine_id: str, strategy_name: str) -> None:
    """Module-level APS jobstore callback for Cron triggers.

    Looks the engine up by id and emits ``CronTick`` so the event flows
    back through the standard dispatch pipeline.

    Clears ``_CURRENT_STRATEGY`` because APScheduler may inherit the
    context from a running strategy task, which would trip the
    ``refuse_inside_strategy`` guard on ``engine.emit()``.
    """
    _CURRENT_STRATEGY.set(None)
    engine = ENGINES.get(engine_id)
    if engine is None:
        logger.error(
            "no_engine_for_cron",
            engine_id=engine_id,
            strategy_name=strategy_name,
        )
        return
    await engine.emit(CronTick(strategy_name=strategy_name))


async def idle_entry(engine_id: str, strategy_name: str) -> None:
    """Module-level APS jobstore callback for Idle IntervalTriggers.

    Looks the engine up by id and hands off to
    :meth:`OfflineEngine.run_idle_scan`. Clears ``_CURRENT_STRATEGY``
    for the same reason as :func:`cron_entry`.
    """
    _CURRENT_STRATEGY.set(None)
    engine = ENGINES.get(engine_id)
    if engine is None:
        logger.error(
            "no_engine_for_idle",
            engine_id=engine_id,
            strategy_name=strategy_name,
        )
        return
    await engine.run_idle_scan(strategy_name)


def launch_scheduler(aps_jobstore_path: Path) -> AsyncIOScheduler:
    """Wire up AsyncIOScheduler + SQLAlchemyJobStore and start it.

    The APS jobstore lives in its own SQLite file (``aps_jobstore_path``) so
    APS's sync SQLAlchemy writes never contend with OME's async aiosqlite
    writes for the same file lock — both writers had previously raced on a
    single ``ome.db`` and manifested as flaky ``SQLITE_BUSY: database is
    locked`` during concurrent strategy dispatch.
    """
    scheduler = AsyncIOScheduler(
        jobstores={
            "default": SQLAlchemyJobStore(
                url=f"sqlite:///{aps_jobstore_path}",
            ),
        },
        executors={"default": AsyncIOExecutor()},
    )
    scheduler.start()
    return scheduler


def add_run_job(
    scheduler: AsyncIOScheduler,
    *,
    engine_id: str,
    strategy_name: str,
    run_id: str,
    event_topic: str,
    event_payload: str,
    max_retries: int,
    traceparent: str | None = None,
) -> None:
    """Add the one-shot job that runs one strategy run now.

    ``traceparent`` is ``None`` for a crash-recovered run, whose job keeps the
    six-argument shape :func:`runner_entry` accepts with its default.
    """
    args: list[object] = [
        engine_id,
        strategy_name,
        run_id,
        event_topic,
        event_payload,
        max_retries,
    ]
    if traceparent is not None:
        args.append(traceparent)
    scheduler.add_job(
        runner_entry,
        trigger="date",
        run_date=get_utc_now(),
        args=args,
        id=run_id,
        replace_existing=False,
        misfire_grace_time=None,  # type: ignore[arg-type]  # APS accepts None ("no expiry"); stub omits it (apscheduler/job.py:213)
    )


def add_trigger_jobs(
    scheduler: AsyncIOScheduler,
    engine_id: str,
    strategies: Iterable[StrategyMeta],
) -> None:
    """Add Cron / Idle APS jobs for every strategy with such a trigger.

    Immediate-trigger strategies have nothing scheduled here — they
    fire only when their declared event class is dispatched.
    """
    for meta in strategies:
        if isinstance(meta.trigger, Cron):
            scheduler.add_job(
                cron_entry,
                trigger=CronTrigger.from_crontab(meta.trigger.expr),
                args=[engine_id, meta.name],
                id=f"cron::{meta.name}",
                replace_existing=True,
            )
        elif isinstance(meta.trigger, Idle):
            scheduler.add_job(
                idle_entry,
                trigger=IntervalTrigger(seconds=meta.trigger.scan_interval_seconds),
                args=[engine_id, meta.name],
                id=f"idle::{meta.name}",
                replace_existing=True,
            )


def reschedule_cron_job(scheduler: AsyncIOScheduler, name: str, expr: str) -> None:
    """Move a Cron strategy's job to a new crontab (atomic in APS)."""
    scheduler.reschedule_job(
        job_id=f"cron::{name}",
        trigger=CronTrigger.from_crontab(expr),
    )


def reschedule_idle_job(
    scheduler: AsyncIOScheduler, name: str, scan_interval_seconds: int
) -> None:
    """Move an Idle strategy's scan job to a new interval."""
    scheduler.reschedule_job(
        job_id=f"idle::{name}",
        trigger=IntervalTrigger(seconds=scan_interval_seconds),
    )

"""OfflineEngine — OME runtime and scheduler.

Manages strategy registration, start-stop lifecycle, event dispatch, and
scheduling of Cron and Idle triggers via APScheduler. Enforces single-engine
guard via portalocker for concurrent access safety.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

import portalocker
from apscheduler.executors.asyncio import AsyncIOExecutor
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from everos.component.utils.datetime import get_utc_now
from everos.core.observability.logging import get_logger
from everos.infra.ome._background.config_reloader import ConfigReloader
from everos.infra.ome._background.crash_recovery import scan_and_resume
from everos.infra.ome._background.idle_scanner import IdleScanner
from everos.infra.ome._dispatch._state import _CURRENT_STRATEGY
from everos.infra.ome._dispatch.dispatcher import EventDispatcher
from everos.infra.ome._dispatch.registry import StrategyRegistry
from everos.infra.ome._dispatch.runner import Runner
from everos.infra.ome._stores.counter import CounterStore
from everos.infra.ome._stores.idle import IdleStore
from everos.infra.ome._stores.run_record import RunRecordStore
from everos.infra.ome._stores.storage import OMEStorage
from everos.infra.ome.config import OMEConfig
from everos.infra.ome.decorator import Strategy, StrategyMeta
from everos.infra.ome.events import BaseEvent, CronTick, ManualTick, resolve_topic
from everos.infra.ome.exceptions import (
    EngineCallFromStrategyError,
    EngineLockHeldError,
    OMEError,
)
from everos.infra.ome.records import RunRecord, RunStatus, StrategyRouteInfo
from everos.infra.ome.triggers import Cron, Idle

logger = get_logger(__name__)

_ENGINES: dict[str, OfflineEngine] = {}


def _refuse_inside_strategy(method: Any) -> Any:
    """Raise :class:`EngineCallFromStrategyError` when called from a strategy.

    Strategies must interact with the engine only via the ``(event, ctx)``
    parameters Runner provides; direct calls bypass the declared
    ``emits=[...]`` contract enforced by ``ctx.emit``. Wraps sync and async
    methods alike.
    """
    if inspect.iscoroutinefunction(method):

        @functools.wraps(method)
        async def async_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            current = _CURRENT_STRATEGY.get()
            if current is not None:
                raise EngineCallFromStrategyError(
                    strategy=current.name, method=method.__name__
                )
            return await method(self, *args, **kwargs)

        return async_wrapper

    @functools.wraps(method)
    def sync_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        current = _CURRENT_STRATEGY.get()
        if current is not None:
            raise EngineCallFromStrategyError(
                strategy=current.name, method=method.__name__
            )
        return method(self, *args, **kwargs)

    return sync_wrapper


async def _runner_entry(
    engine_id: str,
    strategy_name: str,
    run_id: str,
    event_topic: str,
    event_payload: str,
    max_retries_snapshot: int,
) -> None:
    """Module-level APS jobstore callback for a single run.

    Looks the engine up by id and hands off to
    :meth:`OfflineEngine.dispatch_run`. Pickle-safe (no closures, no
    bound methods captured into APS jobstore args).
    """
    engine = _ENGINES.get(engine_id)
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
    )


async def _cron_entry(engine_id: str, strategy_name: str) -> None:
    """Module-level APS jobstore callback for Cron triggers.

    Looks the engine up by id and emits ``CronTick`` so the event flows
    back through the standard dispatch pipeline.

    Clears ``_CURRENT_STRATEGY`` because APScheduler may inherit the
    context from a running strategy task, which would trip the
    ``_guard_no_strategy_call`` check on ``engine.emit()``.
    """
    _CURRENT_STRATEGY.set(None)
    engine = _ENGINES.get(engine_id)
    if engine is None:
        logger.error(
            "no_engine_for_cron",
            engine_id=engine_id,
            strategy_name=strategy_name,
        )
        return
    await engine.emit(CronTick(strategy_name=strategy_name))


async def _idle_entry(engine_id: str, strategy_name: str) -> None:
    """Module-level APS jobstore callback for Idle IntervalTriggers.

    Looks the engine up by id and hands off to
    :meth:`OfflineEngine.run_idle_scan`. Clears ``_CURRENT_STRATEGY``
    for the same reason as :func:`_cron_entry`.
    """
    _CURRENT_STRATEGY.set(None)
    engine = _ENGINES.get(engine_id)
    if engine is None:
        logger.error(
            "no_engine_for_idle",
            engine_id=engine_id,
            strategy_name=strategy_name,
        )
        return
    await engine.run_idle_scan(strategy_name)


class OfflineEngine:
    """Offline Memory Engine — orchestrates strategy registration, scheduling,
    and event dispatch.

    Lifecycle::

        engine = OfflineEngine(config=cfg)
        engine.register(my_strategy)        # before start()
        engine.on_dead_letter(cb)           # before start()
        await engine.start()                # acquires file lock, boots scheduler
        await engine.emit(SomeEvent(...))   # fan out through dispatcher
        await engine.stop()                 # graceful shutdown

    Single-process invariant: a file lock on
    ``<jobstore_path>.lock`` guarantees at most one engine per jobstore
    at any time (cross-process safe via ``portalocker``).
    """

    def __init__(
        self,
        *,
        config: OMEConfig,
    ) -> None:
        self._config = config
        self._registry = StrategyRegistry()
        self._storage = OMEStorage(db_path=config.jobstore_path)
        self._lock_handle: Any = None
        self._started = False
        self._on_dead_letter: Callable[[RunRecord], None] | None = None

        # late-bound (set in start())
        self._counter_store: CounterStore | None = None
        self._run_record_store: RunRecordStore | None = None
        self._dispatcher: EventDispatcher | None = None
        self._runner: Runner | None = None
        self._engine_sem: asyncio.Semaphore | None = None
        self._idle_store: IdleStore | None = None
        self._engine_id = uuid4().hex
        self._scheduler: AsyncIOScheduler | None = None
        self._config_reloader: ConfigReloader | None = None

        # In-flight strategy-run accounting. Incremented at the moment a
        # run is enqueued onto APS (so callers that emit-then-wait observe
        # a non-zero count immediately), decremented in dispatch_run's
        # finally. APS 3.x AsyncIOExecutor.shutdown(wait=True) does NOT
        # honor wait for async coroutines (see apscheduler/executors/
        # asyncio.py:24); this counter is how stop() / drain() learn the
        # engine is genuinely idle.
        self._active_runs = 0
        self._idle_event: asyncio.Event | None = None

    def register(self, strategy: Strategy) -> None:
        """Register a :class:`Strategy` returned by ``@offline_strategy``.

        Must be called before :meth:`start`; registering after start raises
        :class:`OMEError` because the scheduler has already snapshotted
        the strategy set for Cron / Idle job creation.

        Args:
            strategy: A Strategy instance produced by the decorator.

        Raises:
            OMEError: If the engine has already started.
        """
        if self._started:
            raise OMEError("register: cannot register after start()")
        self._registry.register(strategy)

    @_refuse_inside_strategy
    def reschedule_cron_job(self, name: str, expr: str) -> None:
        """Reschedule a Cron strategy's APScheduler job to a new crontab.

        APS reschedule_job is atomic: on success, pending invocations are
        recomputed against the new trigger; on failure it raises and APS
        state is unchanged, so callers can roll back paired registry
        mutations.
        """
        if self._scheduler is None:
            raise OMEError("reschedule_cron_job: engine not started")
        self._scheduler.reschedule_job(
            job_id=f"cron::{name}",
            trigger=CronTrigger.from_crontab(expr),
        )

    @_refuse_inside_strategy
    def reschedule_idle_job(self, name: str, scan_interval_seconds: int) -> None:
        """Reschedule an Idle strategy's APScheduler scan job to a new interval."""
        if self._scheduler is None:
            raise OMEError("reschedule_idle_job: engine not started")
        self._scheduler.reschedule_job(
            job_id=f"idle::{name}",
            trigger=IntervalTrigger(seconds=scan_interval_seconds),
        )

    def on_dead_letter(self, callback: Callable[[RunRecord], None]) -> None:
        """Register a callback invoked after a run is marked DEAD_LETTER.

        Must be set before start(); calls after start() are silently ignored
        (logged at WARNING) to avoid racing with the already-instantiated
        Runner that captured a snapshot of the callback. If called multiple
        times before start(), only the last callback wins (no chaining).
        """
        if self._started:
            logger.warning("on_dead_letter_after_start_ignored")
            return
        self._on_dead_letter = callback

    async def start(self) -> None:
        """Boot the engine: acquire the jobstore lock, validate the strategy
        DAG, wire up late-bound stores, launch APScheduler, run crash
        recovery, register Cron / Idle jobs, and optionally start the
        config-reloader.

        Idempotent: a second call while running is a no-op. On failure,
        every partially-initialised resource (lock, scheduler thread,
        :data:`_ENGINES` slot, config reloader) is rolled back so a retry
        starts from a clean state.
        """
        if self._started:
            return
        await self._storage.init()
        self._acquire_lock()
        try:
            self._registry.validate()
            self._init_components()
            self._idle_event = asyncio.Event()
            self._idle_event.set()
            self._launch_scheduler()
            _ENGINES[self._engine_id] = self
            await self._run_crash_recovery()
            self._register_scheduled_jobs()
            self._start_config_reloader()
            self._started = True
        except Exception:
            await self._rollback_partial_start()
            raise

    def _init_components(self) -> None:
        """Instantiate stores / dispatcher / runner / semaphore.

        Called from :meth:`start` after the file lock is held and DAG
        validation passed; never from anywhere else.
        """
        self._counter_store = CounterStore(storage=self._storage)
        self._run_record_store = RunRecordStore(
            storage=self._storage,
            max_records_per_strategy=self._config.max_records_per_strategy,
        )
        self._dispatcher = EventDispatcher(
            registry=self._registry,
            counter_store=self._counter_store,
        )
        self._engine_sem = asyncio.Semaphore(self._config.max_concurrent_runs)
        self._runner = Runner(
            run_record_store=self._run_record_store,
            engine_sem=self._engine_sem,
            emit_hook=self._dispatch_event,
            on_dead_letter=self._on_dead_letter,
            engine=self,
        )
        self._idle_store = IdleStore(storage=self._storage)

    def _launch_scheduler(self) -> None:
        """Wire up AsyncIOScheduler + SQLAlchemyJobStore and start it.

        The APS jobstore lives in its own SQLite file
        (``aps_jobstore_path``) so APS's sync SQLAlchemy writes never
        contend with OME's async aiosqlite writes for the same file lock
        — both writers had previously raced on a single ``ome.db`` and
        manifested as flaky ``SQLITE_BUSY: database is locked`` during
        concurrent strategy dispatch.
        """
        self._scheduler = AsyncIOScheduler(
            jobstores={
                "default": SQLAlchemyJobStore(
                    url=f"sqlite:///{self._config.aps_jobstore_path}",
                ),
            },
            executors={"default": AsyncIOExecutor()},
        )
        self._scheduler.start()

    async def _run_crash_recovery(self) -> None:
        """Scan ``run_record`` for stale RUNNING rows and re-enqueue them.

        Treats rows whose ``started_at`` is older than
        ``crash_recovery_timeout_seconds`` as crashes from a previous
        engine session: they are marked CRASHED and re-added to APS with
        a fresh ``run_id`` reusing the original event payload.
        """
        await scan_and_resume(
            run_record_store=self._run_record_store,
            timeout_seconds=self._config.crash_recovery_timeout_seconds,
            add_job=self._enqueue_recovery_job,
        )

    async def _enqueue_recovery_job(
        self,
        name: str,
        run_id: str,
        event_topic: str,
        event_payload: str,
        max_retries: int,
    ) -> None:
        """Add one APS job for a re-enqueued crashed run (callback for
        :func:`scan_and_resume`).

        Same enqueue-time bookkeeping as :meth:`_enqueue_run`: the run
        will reach :meth:`dispatch_run` like any other, so the +1/-1
        pair must wrap the ``add_job`` call here too.
        """
        self._on_run_enqueued()
        try:
            self._scheduler.add_job(
                _runner_entry,
                trigger="date",
                run_date=get_utc_now(),
                args=[
                    self._engine_id,
                    name,
                    run_id,
                    event_topic,
                    event_payload,
                    max_retries,
                ],
                id=run_id,
                replace_existing=False,
                misfire_grace_time=None,  # type: ignore[arg-type]  # APS accepts None ("no expiry"); stub omits it (apscheduler/job.py:213)
            )
        except Exception:
            self._on_run_completed()
            raise

    def _register_scheduled_jobs(self) -> None:
        """Add Cron / Idle APS jobs for every strategy with such a trigger.

        Immediate-trigger strategies have nothing scheduled here — they
        fire only when their declared event class is dispatched.
        """
        for meta in self._registry.all():
            if isinstance(meta.trigger, Cron):
                self._scheduler.add_job(
                    _cron_entry,
                    trigger=CronTrigger.from_crontab(meta.trigger.expr),
                    args=[self._engine_id, meta.name],
                    id=f"cron::{meta.name}",
                    replace_existing=True,
                )
            elif isinstance(meta.trigger, Idle):
                self._scheduler.add_job(
                    _idle_entry,
                    trigger=IntervalTrigger(seconds=meta.trigger.scan_interval_seconds),
                    args=[self._engine_id, meta.name],
                    id=f"idle::{meta.name}",
                    replace_existing=True,
                )

    def _start_config_reloader(self) -> None:
        """Start :class:`ConfigReloader` iff ``config_watch`` is on and a
        ``config_path`` is provided.
        """
        if self._config.config_watch and self._config.config_path is not None:
            self._config_reloader = ConfigReloader(
                config_path=self._config.config_path,
                registry=self._registry,
                engine=self,
                debounce_ms=self._config.config_watch_debounce_ms,
            )
            self._config_reloader.start()

    async def _rollback_partial_start(self) -> None:
        """Reverse-order cleanup of whatever :meth:`start` had already
        wired up before the failure: stop reloader, drain in-flight runs
        (best-effort, short timeout — startup failure shouldn't block on
        recovery jobs), shut the scheduler, drop ``_ENGINES`` slot, and
        release the file lock.

        Same ``wait_idle → shutdown(wait=False)`` order as :meth:`stop`
        for the same reasons (pause would freeze recovery jobs that
        already own a +1).
        """
        if self._config_reloader is not None:
            try:
                await self._config_reloader.stop()
            finally:
                self._config_reloader = None
        if self._scheduler is not None:
            try:
                await self.wait_idle(timeout=5.0)
                self._scheduler.shutdown(wait=False)
            finally:
                self._scheduler = None
        _ENGINES.pop(self._engine_id, None)
        self._release_lock()
        self._idle_event = None
        self._active_runs = 0

    async def wait_idle(self, *, timeout: float = 30.0) -> bool:  # noqa: ASYNC109
        """Block until every in-flight strategy run has settled.

        Returns ``True`` on idle, ``False`` if ``timeout`` elapses with
        runs still active. "In flight" means anywhere between
        :meth:`_enqueue_run` (which bumps the counter just before the
        ``add_job`` call) and the end of :meth:`dispatch_run` (which
        releases it in ``finally``).

        Why this exists: APS 3.x ``AsyncIOExecutor.shutdown(wait=True)``
        documents — in the executor source — that it cannot honor wait
        for async coroutines and simply cancels their futures
        (``apscheduler/executors/asyncio.py:24``). Anything depending on
        "all jobs really completed" has to drain through this counter,
        not the scheduler.
        """
        if self._idle_event is None:
            return self._active_runs == 0
        try:
            await asyncio.wait_for(self._idle_event.wait(), timeout=timeout)
            return True
        except TimeoutError:
            return False

    async def stop(self) -> None:
        """Shut the engine down gracefully: stop the config reloader, drain
        in-flight strategy runs, shut the scheduler, drop the global
        ``_ENGINES`` slot, and release the jobstore lock.

        Idempotent: calling stop on an already-stopped engine is a no-op.

        Drain ordering matters and is *deliberately* not
        ``pause → wait_idle → shutdown``.

        - We cannot ``pause()`` first: APS ``pause()`` freezes jobstore
          dispatch including jobs already enqueued (see
          ``apscheduler/schedulers/base.py:pause``: "prevent the scheduler
          from waking up to do job processing"). Each such job already
          owns a +1 in ``_active_runs`` from :meth:`_enqueue_run`, so
          freezing dispatch deadlocks :meth:`wait_idle`.

        - We cannot use ``shutdown(wait=True)``: APS 3.x
          ``AsyncIOExecutor.shutdown`` documents in its own source that
          it cannot honor wait for async coroutines and cancels their
          futures (``apscheduler/executors/asyncio.py:24``). Cascade
          ``CancelledError`` / "Event loop is closed" warnings follow.

        Order used here: ``wait_idle`` first (lets APS finish dispatching
        everything in the jobstore and lets every dispatch_run release its
        counter), then ``shutdown(wait=False)`` (drops the executor cleanly
        because there is nothing left in flight).

        ``_ENGINES`` is popped only after the drain so ``_runner_entry``
        can still find this engine via its id while finishing the last
        few jobs.
        """
        if not self._started:
            return
        if self._config_reloader is not None:
            await self._config_reloader.stop()
            self._config_reloader = None
        if self._scheduler is not None:
            drained = await self.wait_idle(timeout=30.0)
            if not drained:
                logger.warning(
                    "ome_stop_drain_timeout",
                    engine_id=self._engine_id,
                    active_runs=self._active_runs,
                )
            self._scheduler.shutdown(wait=False)
            self._scheduler = None
        _ENGINES.pop(self._engine_id, None)
        self._release_lock()
        self._started = False
        self._idle_event = None
        self._active_runs = 0

    def _acquire_lock(self) -> None:
        lock_path = Path(str(self._config.jobstore_path) + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(lock_path, "a+")  # noqa: SIM115
        try:
            portalocker.lock(handle, portalocker.LOCK_EX | portalocker.LOCK_NB)
        except portalocker.LockException as e:
            handle.close()
            raise EngineLockHeldError(
                f"another OfflineEngine instance already holds {lock_path}"
            ) from e
        self._lock_handle = handle

    def _release_lock(self) -> None:
        if self._lock_handle is not None:
            try:
                portalocker.unlock(self._lock_handle)
            finally:
                self._lock_handle.close()
                self._lock_handle = None

    @_refuse_inside_strategy
    async def emit(self, event: BaseEvent) -> None:
        """Public engine event entry point.

        Strategies must NOT call this directly; use ``ctx.emit`` instead.
        The :func:`_refuse_inside_strategy` guard raises
        :class:`EngineCallFromStrategyError` on in-strategy calls — only
        ``ctx.emit`` enforces the strategy's declared ``emits=[...]``
        contract.
        """
        await self._dispatch_event(event)

    async def _dispatch_event(self, event: BaseEvent) -> None:
        """Internal: actually run an event through dispatch.

        Used by Runner's ``emit_hook`` so ``ctx.emit`` flows through
        dispatch without tripping the public-method guard.
        """
        if not self._started:
            raise OMEError("emit: engine not started")
        # Touch idle_store for any Idle strategy listening on this event type
        # (best-effort; errors do not block dispatch)
        for meta in self._registry.all():
            if isinstance(meta.trigger, Idle) and type(event) in meta.trigger.on:
                bucket = getattr(event, meta.trigger.event_field, None)
                if bucket is not None:
                    try:
                        await self._idle_store.touch(  # type: ignore[union-attr]
                            meta.name,
                            str(bucket),
                            at=get_utc_now(),
                        )
                    except Exception as e:
                        logger.warning(
                            "idle_touch_failed",
                            strategy_name=meta.name,
                            event_field=meta.trigger.event_field,
                            error=str(e),
                        )
        routes = await self._dispatcher.dispatch(event)
        for meta, run_id in routes:
            self._enqueue_run(meta, event, run_id)

    @_refuse_inside_strategy
    async def trigger_manual(
        self,
        name: str,
        *,
        event: BaseEvent | None = None,
        force: bool = False,
    ) -> None:
        """Manually trigger one strategy.

        - ``event=None`` → engine self-emits ``ManualTick(strategy_name=name)``
        - ``force=True`` → bypass the ``enabled`` gate (``applies_to`` and
          ``Counter`` still apply)

        Routes through :meth:`EventDispatcher.dispatch` with
        ``strategy_filter=name`` so the same three-gate logic is applied
        as for engine-driven dispatch.
        """
        if not self._started:
            raise OMEError("trigger_manual: engine not started")
        if event is None:
            event = ManualTick(strategy_name=name)
        routes = await self._dispatcher.dispatch(
            event,
            force_enabled=force,
            strategy_filter=name,
        )
        for meta, run_id in routes:
            self._enqueue_run(meta, event, run_id)

    def _enqueue_run(self, meta: StrategyMeta, event: BaseEvent, run_id: str) -> None:
        """Add a one-shot APScheduler job that hands the event to Runner.

        Computes ``max_retries_snapshot`` from meta or engine default and
        packages a pickle-safe args tuple — the dispatch tail shared by
        ``_dispatch_event``, ``trigger_manual``, and crash recovery.

        Counter ``self._active_runs`` is bumped *before* ``add_job`` so a
        caller that ``emit`` s then immediately ``wait_idle`` s observes a
        non-zero count; the matching decrement lives in
        :meth:`dispatch_run` (which is guaranteed to run for every job
        APS dispatches). If ``add_job`` itself raises, the counter is
        rolled back here.
        """
        max_retries_snapshot = (
            meta.max_retries
            if meta.max_retries is not None
            else self._config.max_retries
        )
        event_topic = type(event).topic()
        self._on_run_enqueued()
        try:
            self._scheduler.add_job(
                _runner_entry,
                trigger="date",
                run_date=get_utc_now(),
                args=[
                    self._engine_id,
                    meta.name,
                    run_id,
                    event_topic,
                    event.model_dump_json(),
                    max_retries_snapshot,
                ],
                id=run_id,
                replace_existing=False,
                misfire_grace_time=None,  # type: ignore[arg-type]  # APS accepts None ("no expiry"); stub omits it (apscheduler/job.py:213)
            )
        except Exception:
            self._on_run_completed()
            raise

    def _on_run_enqueued(self) -> None:
        """Bump in-flight count and mark the engine non-idle."""
        self._active_runs += 1
        if self._idle_event is not None:
            self._idle_event.clear()

    def _on_run_completed(self) -> None:
        """Drop in-flight count; mark the engine idle if the count hit zero.

        Never lets the counter dip below zero — that would mask a bookkeeping
        bug rather than fix it, and a stuck-clear idle_event would deadlock
        ``wait_idle``.
        """
        if self._active_runs <= 0:
            logger.error(
                "active_runs_underflow",
                engine_id=self._engine_id,
            )
            self._active_runs = 0
            if self._idle_event is not None:
                self._idle_event.set()
            return
        self._active_runs -= 1
        if self._active_runs == 0 and self._idle_event is not None:
            self._idle_event.set()

    async def dispatch_run(
        self,
        *,
        strategy_name: str,
        run_id: str,
        event_topic: str,
        event_payload: str,
        max_retries_snapshot: int,
    ) -> None:
        """APS jobstore callback target for one strategy run.

        Public because the module-level :func:`_runner_entry` callback
        must cross the pickle boundary — a bound method on ``self`` is
        not picklable into the APS jobstore. Not part of the
        strategy-author API; intended to be called only by
        ``_runner_entry`` (and crash recovery). Not guarded with
        ``_refuse_inside_strategy`` because APS executors may inherit
        the calling task's ContextVar — a strategy that ``ctx.emit``s
        and triggers a cascade would falsely trip the guard here.

        Closes the +1 the matching enqueue path opened, in ``finally``
        so cancellation, retries, and crashes all release the count.
        """
        try:
            cls = resolve_topic(event_topic)
            event = cls.model_validate_json(event_payload)
            meta = self._registry.get(strategy_name)
            await self._runner.run(
                meta,
                event,
                run_id=run_id,
                max_retries_snapshot=max_retries_snapshot,
            )
        finally:
            self._on_run_completed()

    async def run_idle_scan(self, strategy_name: str) -> None:
        """APS IntervalTrigger callback target for one Idle strategy.

        Constructs an :class:`IdleScanner` against the engine's idle_store
        and runs one scan, emitting :class:`IdleTick` for each overdue
        bucket. Public for the same APS-pickle reason as
        :meth:`dispatch_run`; unguarded for the same ContextVar-
        inheritance reason.
        """
        meta = self._registry.get(strategy_name)
        if not isinstance(meta.trigger, Idle):
            logger.error(
                "idle_entry_bad_trigger_type",
                strategy_name=strategy_name,
                trigger_type=type(meta.trigger).__name__,
            )
            return
        scanner = IdleScanner(
            strategy_name=strategy_name,
            trigger=meta.trigger,
            idle_store=self._idle_store,  # type: ignore[arg-type]
            emit=self.emit,
        )
        await scanner.scan_once()

    @_refuse_inside_strategy
    async def inspect_dispatch(self, event: BaseEvent) -> list[StrategyRouteInfo]:
        """Return per-strategy routing info for event (read-only).

        Calls the dispatcher in inspect mode (no counter mutation).
        """
        if not self._started:
            raise OMEError("inspect_dispatch: engine not started")
        return await self._dispatcher.inspect(event)

    @_refuse_inside_strategy
    async def list_runs(
        self,
        strategy_name: str,
        *,
        status: RunStatus | None = None,
        limit: int = 100,
    ) -> list[RunRecord]:
        """Return run records for ``strategy_name``, optionally filtered by status.

        Args:
            strategy_name: Strategy whose runs to fetch.
            status: Terminal status filter (e.g., ``RunStatus.SUCCESS``); ``None``
                returns runs in any state.
            limit: Maximum number of records to return; results are ordered
                ``started_at DESC``.

        Returns:
            Up to ``limit`` ``RunRecord`` instances, newest first.

        Raises:
            OMEError: Engine has not been started.
        """
        if not self._started:
            raise OMEError("list_runs: engine not started")
        return await self._run_record_store.list_runs(
            strategy_name=strategy_name,
            status=status,
            limit=limit,
        )

    @_refuse_inside_strategy
    async def get_run_status(self, run_id: str) -> RunRecord | None:
        """Fetch a single run record by ``run_id``.

        Args:
            run_id: The 32-character ``uuid4().hex`` assigned at dispatch.

        Returns:
            The matching ``RunRecord``, or ``None`` if no row exists for that id.

        Raises:
            OMEError: Engine has not been started.
        """
        if not self._started:
            raise OMEError("get_run_status: engine not started")
        return await self._run_record_store.get(run_id)

    async def list_runs_by_event_id(self, event_id: str) -> list[RunRecord]:
        """Return all run records triggered by ``event_id``.

        Not guarded by ``_refuse_inside_strategy`` — designed to be
        called from inside a strategy (e.g. Reflection waiting for
        downstream extraction runs).

        Args:
            event_id: The ``BaseEvent.event_id`` of the triggering event.

        Returns:
            All ``RunRecord`` instances whose ``event_id`` matches,
            newest first.

        Raises:
            OMEError: Engine has not been started.
        """
        if not self._started:
            raise OMEError("list_runs_by_event_id: engine not started")
        return await self._run_record_store.list_by_event_id(event_id)

    _TERMINAL_STATUSES = frozenset(
        {
            RunStatus.SUCCESS,
            RunStatus.FAILED,
            RunStatus.DEAD_LETTER,
            RunStatus.CRASHED,
        }
    )

    async def wait_for_event(
        self,
        event_id: str,
        *,
        timeout: float = 120.0,  # noqa: ASYNC109
        poll_interval: float = 0.5,
    ) -> list[RunRecord]:
        """Poll until all runs triggered by ``event_id`` reach a terminal status.

        Not guarded by ``_refuse_inside_strategy`` — designed to be
        called from inside a strategy (e.g. Reflection waiting for
        downstream extraction runs to complete before deprecating).

        Args:
            event_id: The ``BaseEvent.event_id`` of the triggering event.
            timeout: Maximum seconds to wait before raising ``TimeoutError``.
            poll_interval: Seconds between polls.

        Returns:
            All ``RunRecord`` instances for ``event_id`` once every run
            has reached a terminal status (SUCCESS / FAILED / DEAD_LETTER
            / CRASHED).

        Raises:
            TimeoutError: If any run is still non-terminal after ``timeout``.
            OMEError: Engine has not been started.
        """
        if not self._started:
            raise OMEError("wait_for_event: engine not started")
        deadline = asyncio.get_event_loop().time() + timeout
        while True:
            runs = await self._run_record_store.list_by_event_id(event_id)
            if runs and all(r.status in self._TERMINAL_STATUSES for r in runs):
                return runs
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                raise TimeoutError(
                    f"wait_for_event: timed out after {timeout}s "
                    f"for event_id={event_id!r}"
                )
            await asyncio.sleep(min(poll_interval, remaining))

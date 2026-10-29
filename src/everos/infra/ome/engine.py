"""OfflineEngine — OME runtime and scheduler.

Manages strategy registration, start-stop lifecycle, event dispatch, and
scheduling of Cron and Idle triggers via APScheduler. Enforces single-engine
guard via portalocker for concurrent access safety.

Collaborators: the APScheduler boundary (job building and the persisted entry
callbacks) is :mod:`everos.infra.ome._dispatch.scheduling`; in-flight run
accounting is :class:`~everos.infra.ome._dispatch.in_flight.InFlightRuns`; the
in-strategy call guard is
:func:`~everos.infra.ome._dispatch._state.refuse_inside_strategy`.
"""

from __future__ import annotations

__all__ = ["OfflineEngine"]

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

import portalocker
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from everos.core.observability.logging import get_logger
from everos.core.observability.tracing import current_traceparent
from everos.core.utils.datetime import get_utc_now
from everos.infra.ome._background.config_reloader import ConfigReloader
from everos.infra.ome._background.crash_recovery import scan_and_resume
from everos.infra.ome._background.idle_scanner import IdleScanner
from everos.infra.ome._dispatch import scheduling
from everos.infra.ome._dispatch._state import refuse_inside_strategy
from everos.infra.ome._dispatch.dispatcher import EventDispatcher
from everos.infra.ome._dispatch.in_flight import InFlightRuns
from everos.infra.ome._dispatch.registry import StrategyRegistry
from everos.infra.ome._dispatch.runner import Runner
from everos.infra.ome._dispatch.strategy import StrategyMeta
from everos.infra.ome._stores.counter import CounterStore
from everos.infra.ome._stores.idle import IdleStore
from everos.infra.ome._stores.run_record import RunRecordStore
from everos.infra.ome._stores.storage import OMEStorage
from everos.infra.ome.config import OMEConfig
from everos.infra.ome.events import resolve_topic
from everos.infra.ome.exceptions import (
    EngineLockHeldError,
    OMEError,
)
from everos.infra.ome.records import (
    RunRecord,
    RunStatus,
    StrategyRouteInfo,
    runs_settled,
)
from everos.memory.ports import Event, Idle, ManualTick

logger = get_logger(__name__)

# APScheduler persists a job's callable as ``<module>:<qualname>``; jobs written
# before the entry functions moved name this module, so they stay resolvable
# here, or the scheduler drops such a job (a lost run) when it restores it.
_runner_entry = scheduling.runner_entry
_cron_entry = scheduling.cron_entry
_idle_entry = scheduling.idle_entry


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
        self._in_flight = InFlightRuns(self._engine_id)

    def register(self, strategy: StrategyMeta) -> None:
        """Register a strategy built by ``to_ome_strategy``.

        Must be called before :meth:`start`; registering after start raises
        :class:`OMEError` because the scheduler has already snapshotted
        the strategy set for Cron / Idle job creation.

        Args:
            strategy: The adapted strategy.

        Raises:
            OMEError: If the engine has already started.
        """
        if self._started:
            raise OMEError("register: cannot register after start()")
        self._registry.register(strategy)

    @refuse_inside_strategy
    def reschedule_cron_job(self, name: str, expr: str) -> None:
        """Reschedule a Cron strategy's APScheduler job to a new crontab.

        APS reschedule_job is atomic: on success, pending invocations are
        recomputed against the new trigger; on failure it raises and APS
        state is unchanged, so callers can roll back paired registry
        mutations.
        """
        if self._scheduler is None:
            raise OMEError("reschedule_cron_job: engine not started")
        scheduling.reschedule_cron_job(self._scheduler, name, expr)

    @refuse_inside_strategy
    def reschedule_idle_job(self, name: str, scan_interval_seconds: int) -> None:
        """Reschedule an Idle strategy's APScheduler scan job to a new interval."""
        if self._scheduler is None:
            raise OMEError("reschedule_idle_job: engine not started")
        scheduling.reschedule_idle_job(self._scheduler, name, scan_interval_seconds)

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
        :data:`scheduling.ENGINES` slot, config reloader) is rolled back so a retry
        starts from a clean state.
        """
        if self._started:
            return
        await self._storage.init()
        self._acquire_lock()
        try:
            await self._storage.hold_open()
            self._registry.validate()
            self._init_components()
            self._in_flight.arm()
            self._scheduler = scheduling.launch_scheduler(
                self._config.aps_jobstore_path
            )
            scheduling.ENGINES[self._engine_id] = self
            if self._config.crash_recovery_enabled:
                await self._run_crash_recovery()
            scheduling.add_trigger_jobs(
                self._scheduler, self._engine_id, self._registry.all()
            )
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
            config=self._config,
            on_dead_letter=self._on_dead_letter,
        )
        self._idle_store = IdleStore(storage=self._storage)

    async def _run_crash_recovery(self) -> None:
        """Scan ``run_record`` for stale RUNNING rows and re-enqueue them.

        Runs on :meth:`start` when :attr:`OMEConfig.crash_recovery_enabled`
        is ``True`` (default). One-shot engines that register a subset of
        strategies (backfill Phase 2/3) opt out via
        ``crash_recovery_enabled=False`` to avoid re-enqueueing the
        server's stale rows into their own scheduler, whose registry
        doesn't contain those strategy names.

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
        self._in_flight.enqueued()
        try:
            scheduling.add_run_job(
                self._scheduler,
                engine_id=self._engine_id,
                strategy_name=name,
                run_id=run_id,
                event_topic=event_topic,
                event_payload=event_payload,
                max_retries=max_retries,
            )
        except Exception:
            self._in_flight.completed()
            raise

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
        recovery jobs), shut the scheduler, drop the ``scheduling.ENGINES`` slot, and
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
        scheduling.ENGINES.pop(self._engine_id, None)
        await self._storage.release()
        self._release_lock()
        self._in_flight.disarm()

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
        return await self._in_flight.wait_idle(timeout=timeout)

    async def stop(self) -> None:
        """Shut the engine down gracefully: stop the config reloader, drain
        in-flight strategy runs, shut the scheduler, drop the global
        ``scheduling.ENGINES`` slot, and release the jobstore lock.

        Idempotent: calling stop on an already-stopped engine is a no-op.

        Drain ordering matters and is *deliberately* not
        ``pause → wait_idle → shutdown``.

        - We cannot ``pause()`` first: APS ``pause()`` freezes jobstore
          dispatch including jobs already enqueued (see
          ``apscheduler/schedulers/base.py:pause``: "prevent the scheduler
          from waking up to do job processing"). Each such job already
          owns a +1 in the in-flight count from :meth:`_enqueue_run`, so
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

        ``scheduling.ENGINES`` is popped only after the drain so ``runner_entry``
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
                    active_runs=self._in_flight.active,
                )
            self._scheduler.shutdown(wait=False)
            self._scheduler = None
        scheduling.ENGINES.pop(self._engine_id, None)
        await self._storage.release()
        self._release_lock()
        self._started = False
        self._in_flight.disarm()

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

    @refuse_inside_strategy
    async def emit(self, event: Event) -> None:
        """Public engine event entry point.

        Strategies must NOT call this directly: they return their follow-up
        events. The :func:`refuse_inside_strategy` guard raises
        :class:`EngineCallFromStrategyError` on in-strategy calls — only the
        run context enforces the strategy's declared ``emits``.
        """
        await self._dispatch_event(event)

    async def _dispatch_event(self, event: Event) -> None:
        """Internal: actually run an event through dispatch.

        Used by Runner's ``emit_hook`` so a strategy's returned events flow
        through dispatch without tripping the public-method guard.
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

    @refuse_inside_strategy
    async def trigger_manual(
        self,
        name: str,
        *,
        event: Event | None = None,
        force: bool = False,
    ) -> tuple[Event, list[tuple[StrategyMeta, str]]]:
        """Manually trigger one strategy.

        - ``event=None`` → engine self-emits ``ManualTick(strategy_name=name)``;
          a strategy that does not list ``ManualTick`` in its trigger is not
          dispatched (``routes`` comes back empty)
        - ``force=True`` → bypass the ``enabled`` gate (``applies_to`` and
          ``Counter`` still apply)

        Routes through :meth:`EventDispatcher.dispatch` with
        ``strategy_filter=name`` so the same three-gate logic is applied
        as for engine-driven dispatch.

        Returns:
            Tuple of ``(event, routes)``:
                - ``event``: the event that was dispatched (either supplied
                  or the engine-generated ``ManualTick``).
                - ``routes``: the ``(meta, run_id)`` pairs that were
                  enqueued. Empty list when every dispatch gate rejected
                  the strategy.
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
        return event, routes

    def _enqueue_run(self, meta: StrategyMeta, event: Event, run_id: str) -> None:
        """Add a one-shot APScheduler job that hands the event to Runner.

        Computes ``max_retries_snapshot`` from meta or engine default and
        packages a pickle-safe args tuple — the dispatch tail shared by
        ``_dispatch_event``, ``trigger_manual``, and crash recovery.

        The in-flight count is bumped *before* ``add_job`` so a
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
        # Capture the triggering request's trace context (if any) here — this
        # runs synchronously in the caller's task, so its span is still active.
        # Carried as a pickle-safe string across the APScheduler boundary.
        traceparent = current_traceparent() or ""
        self._in_flight.enqueued()
        try:
            scheduling.add_run_job(
                self._scheduler,
                engine_id=self._engine_id,
                strategy_name=meta.name,
                run_id=run_id,
                event_topic=event_topic,
                event_payload=event.model_dump_json(),
                max_retries=max_retries_snapshot,
                traceparent=traceparent,
            )
        except Exception:
            self._in_flight.completed()
            raise

    async def dispatch_run(
        self,
        *,
        strategy_name: str,
        run_id: str,
        event_topic: str,
        event_payload: str,
        max_retries_snapshot: int,
        traceparent: str | None = None,
    ) -> None:
        """APS jobstore callback target for one strategy run.

        Public because the module-level ``runner_entry`` callback
        must cross the pickle boundary — a bound method on ``self`` is
        not picklable into the APS jobstore. Not part of the
        strategy-author API; intended to be called only by
        ``runner_entry`` (and crash recovery). Not guarded with
        ``refuse_inside_strategy`` because APS executors may inherit
        the calling task's ContextVar — a strategy whose returned events
        trigger a cascade would falsely trip the guard here.

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
                traceparent=traceparent,
            )
        finally:
            self._in_flight.completed()

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

    @refuse_inside_strategy
    async def inspect_dispatch(self, event: Event) -> list[StrategyRouteInfo]:
        """Return per-strategy routing info for event (read-only).

        Calls the dispatcher in inspect mode (no counter mutation).
        """
        if not self._started:
            raise OMEError("inspect_dispatch: engine not started")
        return await self._dispatcher.inspect(event)

    @refuse_inside_strategy
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

    @refuse_inside_strategy
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

        Not guarded by ``refuse_inside_strategy``: a read-only query
        that cannot bypass the emit contract.

        Args:
            event_id: The ``Event.event_id`` of the triggering event.

        Returns:
            All ``RunRecord`` instances whose ``event_id`` matches,
            newest first.

        Raises:
            OMEError: Engine has not been started.
        """
        if not self._started:
            raise OMEError("list_runs_by_event_id: engine not started")
        return await self._run_record_store.list_by_event_id(event_id)

    async def wait_for_event(
        self,
        event_id: str,
        *,
        timeout: float = 120.0,  # noqa: ASYNC109
        poll_interval: float = 0.5,
    ) -> list[RunRecord]:
        """Poll until all runs triggered by ``event_id`` reach a terminal status.

        Not guarded by ``refuse_inside_strategy``: a read-only query
        that cannot bypass the emit contract.

        Args:
            event_id: The ``Event.event_id`` of the triggering event.
            timeout: Maximum seconds to wait before raising ``TimeoutError``.
            poll_interval: Seconds between polls.

        Returns:
            All ``RunRecord`` instances for ``event_id`` once every run
            has reached a terminal status (SUCCESS / DEAD_LETTER /
            CRASHED; a FAILED row is a retry pending, see
            ``records.TERMINAL_STATUSES``). A fan-out sibling whose job has not
            started yet has no row, so the set can still grow in that
            window; callers that need an exact count must know it from
            the dispatch.

        Raises:
            TimeoutError: If any run is still non-terminal after ``timeout``.
            OMEError: Engine has not been started.
        """
        if not self._started:
            raise OMEError("wait_for_event: engine not started")
        deadline = asyncio.get_event_loop().time() + timeout
        while True:
            runs = await self._run_record_store.list_by_event_id(event_id)
            if runs and runs_settled(runs):
                return runs
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                raise TimeoutError(
                    f"wait_for_event: timed out after {timeout}s "
                    f"for event_id={event_id!r}"
                )
            await asyncio.sleep(min(poll_interval, remaining))

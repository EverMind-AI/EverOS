"""LanceDB index maintenance for the tables the cascade projects into.

:class:`IndexMaintainer` owns three per-kind jobs and keeps them from
committing on the same table manifest at once:

- **optimize** — throttled, trailing-edge compaction after writes
  (:meth:`IndexMaintainer.schedule_optimize`); every
  ``optimize_prune_interval_seconds`` the beat is a heavy one that also
  prunes old versions under the table's write lock.
- **heartbeat** — a periodic sweep through the same throttle so an unindexed
  tail left by a restart is merged without new writes.
- **rebuild** — a slow ``drop_index + create_index`` sweep that bounds active
  index growth (an upstream lancedb gap), with lost commit races retried on a
  recorded deadline rather than a sleep.

Optimize is a performance and storage-hygiene step, never a visibility one:
rows are searchable as soon as their upsert commits (flat scan over the
unindexed tail). Invariant: at most one optimize and one rebuild task per kind,
each waiting — bounded — for the other before touching the manifest.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import time
from collections.abc import Mapping, Sequence

from everos.core.observability.logging import get_logger
from everos.infra.persistence.index import IndexMaintenance

from ..projections import Handler
from ._kind_state import KindOptimizerState
from ._stop import wait_or_stop
from ._tuning import (
    OPTIMIZE_FAILURE_ALERT_THRESHOLD,
    REBUILD_LOOP_TICK_SECONDS,
)

logger = get_logger(__name__)


def is_benign_commit_conflict(exc: BaseException) -> bool:
    """True for a LanceDB optimistic-concurrency retry error.

    LanceDB surfaces the Rust ``Retryable commit conflict`` as a plain
    exception whose message carries the phrase — there is no dedicated
    class to catch. Either beat can lose the race: the light beat is
    lock-free, and the heavy beat's write lock is in-process only, so a
    second process (CLI ``cascade sync`` / ``backfill``) can preempt it.
    Both are expected under churn and benign (the next beat retries), so
    they log at ``debug`` and do not count toward the failure streak.

    Match only the specific ``commit conflict`` phrase (a substring of the
    Rust message), not a bare ``retryable`` — the latter appears in the
    repr of unrelated recoverable errors (e.g. ``ExternalServiceError``
    carrying ``retryable=True``) and would silently swallow real failures
    (review P2).
    """
    return "commit conflict" in str(exc).lower()


class IndexMaintainer:
    """Schedules optimize / prune / rebuild per kind and reports their health.

    Args:
        handlers: The worker's ``{kind: Handler}`` map; a handler's
            ``index_repo`` (when it has one) is the table maintained for
            that kind.
        stop: The worker's shutdown event; every cooldown and sweep
            observes it.
        optimize_min_interval_seconds: Throttle between optimize runs on one
            kind.
        optimize_heartbeat_seconds: Period of the heartbeat sweep.
        optimize_prune_interval_seconds: Cadence of the heavy (prune) beat.
        optimize_prune_retention_seconds: ``cleanup_older_than`` passed to
            prune.
        optimize_rebuild_interval_seconds: Period of the full rebuild sweep.
        maintenance_task_timeout_seconds: Deadline on one maintenance call
            and on each bounded wait between optimize and rebuild.
        rebuild_conflict_backoffs_seconds: Delays before re-attempting a
            rebuild that lost a commit race.
    """

    def __init__(
        self,
        handlers: Mapping[str, Handler],
        stop: asyncio.Event,
        *,
        optimize_min_interval_seconds: float,
        optimize_heartbeat_seconds: float,
        optimize_prune_interval_seconds: float,
        optimize_prune_retention_seconds: float,
        optimize_rebuild_interval_seconds: float,
        maintenance_task_timeout_seconds: float,
        rebuild_conflict_backoffs_seconds: Sequence[float],
    ) -> None:
        self._handlers = handlers
        self._stop = stop
        self._optimize_min_interval = optimize_min_interval_seconds
        self._optimize_heartbeat = optimize_heartbeat_seconds
        self._optimize_prune_interval = optimize_prune_interval_seconds
        self._optimize_prune_retention = optimize_prune_retention_seconds
        self._optimize_rebuild_interval = optimize_rebuild_interval_seconds
        self._maintenance_timeout = maintenance_task_timeout_seconds
        self._rebuild_conflict_backoffs = tuple(rebuild_conflict_backoffs_seconds)
        self._optimizer_states: dict[str, KindOptimizerState] = {}

    def optimize_failure_streak(self) -> int:
        """Worst consecutive non-benign optimize failure count across kinds."""
        states = self._optimizer_states.values()
        return max((s.optimize_failures for s in states), default=0)

    def prune_staleness(self, started_at: float) -> tuple[float, str | None]:
        """Staleness of the **worst** kind: ``(seconds, kind)``.

        Per kind, staleness is the time since its own last successful
        prune — or since ``started_at`` (worker start) if it has never
        pruned — and the worst (largest) one is reported. Taking the worst
        rather than the newest prune across kinds is what makes the signal
        work on a multi-kind deployment: one kind whose cleanup dies grows
        that table's index dir unbounded, and the ~5 healthy kinds pruning
        on schedule must not hide it (that masking was the pre-fix bug).

        Returns ``(0.0, None)`` before the worker has started
        (``started_at == 0``) or before any kind has registered an
        optimizer state — no prune beat has run, so nothing is stale yet.
        """
        states = list(self._optimizer_states.items())
        if not states or started_at == 0.0:
            return 0.0, None
        now = time.monotonic()
        worst_seconds = -1.0
        worst_kind: str | None = None
        for kind, state in states:
            baseline = max(state.last_prune_at, started_at)
            stale = max(0.0, now - baseline)
            if stale > worst_seconds:
                worst_seconds, worst_kind = stale, kind
        return max(0.0, worst_seconds), worst_kind

    def _index_for(self, kind: str) -> IndexMaintenance | None:
        """The index table ``kind``'s projection writes, if it writes one."""
        handler = self._handlers.get(kind)
        return getattr(handler, "index_repo", None) if handler else None

    def schedule_optimize(self, kind: str) -> None:
        """Throttle + trailing-edge schedule for a kind's optimize.

        Per-kind semantics — for any one ``kind``:

        - The first call after the throttle window starts an optimize
          immediately (initial_delay=0).
        - Subsequent calls within the window only set ``dirty=True``;
          the in-flight runner picks the flag up and re-runs after the
          throttle interval has elapsed.
        - A call while a task is in flight returns without starting a
          new task — only the dirty flag matters. This guarantees at
          most one concurrent ``optimize()`` per kind, which is what
          LanceDB's per-table manifest version expects.

        No-op when the handler for ``kind`` doesn't expose a
        ``index_repo`` (test stubs, handlers that intentionally skip
        LanceDB).

        Idempotent and cheap (a single dict lookup + flag write in
        the hot path) — safe to call on every batch and from the
        heartbeat sweep.
        """
        repo = self._index_for(kind)
        if repo is None:
            return
        state = self._optimizer_states.setdefault(kind, KindOptimizerState())
        state.dirty = True
        if state.task is not None and not state.task.done():
            return
        elapsed = time.monotonic() - state.last_run_at
        delay = max(0.0, self._optimize_min_interval - elapsed)
        state.task = asyncio.create_task(
            self._optimize_runner(kind, initial_delay=delay),
            name=f"cascade-optimize-{kind}",
        )

    async def _optimize_runner(self, kind: str, *, initial_delay: float) -> None:
        """Run optimize for ``kind`` until ``dirty`` clears.

        Honours the throttle interval on entry (when scheduled
        mid-cooldown) and between consecutive runs (when a write
        re-raised ``dirty`` during the previous ``optimize()``). The
        cooldown waits respect the worker's stop signal so shutdown
        doesn't have to outlast the throttle window.

        If a rebuild is in flight for this kind we wait for it before
        touching the manifest — concurrent ``optimize`` + ``rebuild``
        on the same LanceDB table would race on the version commit.
        """
        state = self._optimizer_states[kind]
        try:
            if initial_delay > 0 and await wait_or_stop(self._stop, initial_delay):
                return
            # Serialise behind any in-flight rebuild (rare; only during the
            # 12h sweep). Failures are absorbed in _run_rebuild_once.
            #
            # Bounded, and symmetric with the wait on the other side: whichever
            # of the two maintenance jobs arrives second parks on the first, so
            # an unbounded wait here is the same defect as an unbounded wait
            # there — this kind's task slot never frees, schedule_optimize
            # keeps short-circuiting on it, and that table silently stops being
            # pruned. It was left unbounded on the argument that
            # rebuild_indexes carries its own 300s deadline; that only covers
            # the critical section, not the task's dispatch and teardown around
            # it, so the transitive bound was never real.
            #
            # ``asyncio.wait``, not ``asyncio.timeout`` around ``await task``:
            # the latter cancels the task it was waiting for when the deadline
            # fires. Here that killed the rebuild mid-flight, and because the
            # rebuild loop's ``_run_rebuild_once`` only catches ``Exception``,
            # the ``CancelledError`` ended the loop itself — silently, since
            # the done-callback treats a cancelled task as a ``stop()``. Found
            # by the maintenance state-machine test. A failed rebuild is
            # already logged and counted by ``_run_rebuild_once``.
            if state.rebuild_task is not None and not state.rebuild_task.done():
                done, _ = await asyncio.wait(
                    {state.rebuild_task}, timeout=self._maintenance_timeout
                )
                if not done:
                    # Give up this beat rather than compact under a live
                    # rebuild — the two commit on the same manifest, which is
                    # what the wait exists to prevent. Writes keep the dirty
                    # flag set, so the next beat retries.
                    logger.warning(
                        "cascade_lancedb_optimize_skipped_rebuild_unfinished",
                        kind=kind,
                        waited_seconds=self._maintenance_timeout,
                    )
                    return
            while state.dirty and not self._stop.is_set():
                state.dirty = False
                state.last_run_at = time.monotonic()
                await self._run_optimize_once(kind)
                if (
                    state.dirty
                    and not self._stop.is_set()
                    and await wait_or_stop(self._stop, self._optimize_min_interval)
                ):
                    return
        finally:
            if state.task is asyncio.current_task():
                state.task = None

    async def _run_optimize_once(self, kind: str) -> None:
        """Run one ``optimize()`` for ``kind``, opportunistically pruning.

        Most calls take the **light** path — lock-free ``optimize()``,
        pure compaction + index merge, fast. Every
        ``_optimize_prune_interval`` seconds the next call takes the
        **heavy** path — ``prune()`` under the per-table write lock,
        which compacts *and* physically deletes files belonging to
        versions older than ``_optimize_prune_retention`` (a short
        window decoupled from the beat cadence; see
        :data:`DEFAULT_OPTIMIZE_PRUNE_RETENTION_SECONDS`).

        Pruning is opt-in per call rather than a separate task so the
        existing per-kind serialisation (one in-flight runner per kind)
        keeps holding — LanceDB serialises writes on the table's
        manifest, and prune is a write.
        """
        repo = self._index_for(kind)
        if repo is None:
            return
        state = self._optimizer_states.get(kind)
        now = time.monotonic()
        should_prune = (
            state is None
            # 0.0 means "never attempted" — always prune, don't compare clocks.
            # ``monotonic()`` is boot-relative, so ``now - 0 >= interval`` is
            # false for the first ~cadence of machine/container uptime and the
            # catch-up prune would be skipped exactly when a fresh process most
            # needs it.
            or state.last_prune_attempt_at == 0.0
            or (now - state.last_prune_attempt_at) >= self._optimize_prune_interval
        )
        try:
            if should_prune:
                # Heavy beat: physically reclaim old versions under the
                # per-table write lock (inside ``repo.prune``) so churn
                # can't preempt the cleanup commit. ``prune`` uses
                # ``delete_unverified=False`` so it stays safe even against a
                # second process (CLI ``cascade sync``). The retention window
                # is short + decoupled from the cadence (see
                # DEFAULT_OPTIMIZE_PRUNE_RETENTION_SECONDS) so superseded
                # full-table copies don't pile up between beats.
                #
                # Advance the *attempt* clock before the call: if prune hangs
                # and the write-lock timeout kills it, the next beat still
                # waits a full cadence instead of retrying immediately and
                # pinning the write lock (review N1). The *success* clock
                # (last_prune_at, for the staleness health signal) advances
                # only after the call returns.
                if state is not None:
                    state.last_prune_attempt_at = now
                async with asyncio.timeout(self._maintenance_timeout):
                    await repo.prune(
                        dt.timedelta(seconds=self._optimize_prune_retention)
                    )
                if state is not None:
                    state.last_prune_at = now
                # Same heavy beat: a table that crossed the ANN threshold since
                # startup gets its vector index, and one whose light beats piled
                # up delta indices is retrained; a no-op otherwise.
                async with asyncio.timeout(self._maintenance_timeout):
                    await repo.ensure_vector_indexes()
            else:
                # Light beat: lock-free compaction. A commit conflict here
                # is benign — handled below.
                async with asyncio.timeout(self._maintenance_timeout):
                    await repo.optimize()
            if state is not None:
                state.optimize_failures = 0
                state.failures_since_fallback = 0
            logger.debug(
                "cascade_lancedb_optimized",
                kind=kind,
                pruned=should_prune,
            )
        except Exception as exc:
            # Benign light-beat commit conflict: the lock-free compaction
            # lost the optimistic-concurrency race against a live writer.
            # Expected under churn, self-heals next beat — log at debug and
            # do NOT count it toward the streak (which would otherwise pin
            # high on a busy table) or trigger a fallback rebuild.
            #
            # This applies to the HEAVY beat too: the per-table write lock is
            # in-process only (see LanceRepoBase.prune), so a second process
            # — a long `cascade backfill`, a `cascade sync` — can still
            # preempt prune's Rewrite commit. Counting those as real failures
            # let ~25min of cross-process churn reach the threshold and fire a
            # spurious fallback rebuild, which drops every index before
            # recreating it; if the rebuild lost the race too, its failure was
            # swallowed as a warning and the table sat with no FTS index (all
            # `/search` on that kind 500s) until the next 12h sweep. A prune
            # that genuinely stops succeeding is caught by the prune-staleness
            # health signal instead (per-kind, see prune_staleness) — that is
            # the right detector for it, and unlike a rebuild it does not
            # destroy indexes to "fix" a lost race.
            if is_benign_commit_conflict(exc):
                # ``pruned`` is the whole diagnostic value of this line. Lance
                # labels both beats' commit the same way ("This Rewrite
                # transaction was preempted by ..."), so the message alone
                # cannot tell them apart — but the consequences differ:
                # a lost LIGHT beat is free (compaction retries ~10s later),
                # while a lost HEAVY beat means that table skipped a whole
                # prune cadence and its index dir keeps the superseded files.
                # Without the flag, reading a disk-growth incident off the
                # logs means back-inferring which beats were heavy from the
                # 300s cadence (done once during the storage soak — slow and
                # fragile). Mirrors ``pruned`` on the sibling failure log.
                logger.debug(
                    "cascade_lancedb_optimize_conflict",
                    kind=kind,
                    pruned=should_prune,
                    error=f"{type(exc).__name__}: {exc}",
                )
                return
            failures = 0
            since_fallback = 0
            if state is not None:
                state.optimize_failures += 1
                state.failures_since_fallback += 1
                failures = state.optimize_failures
                since_fallback = state.failures_since_fallback
            log = (
                logger.error
                if failures >= OPTIMIZE_FAILURE_ALERT_THRESHOLD
                else logger.warning
            )
            log(
                "cascade_lancedb_optimize_failed",
                kind=kind,
                pruned=should_prune,
                consecutive_failures=failures,
                error=f"{type(exc).__name__}: {exc}",
            )
            if since_fallback >= OPTIMIZE_FAILURE_ALERT_THRESHOLD:
                logger.info(
                    "cascade_lancedb_optimize_fallback_rebuild",
                    kind=kind,
                    consecutive_failures=failures,
                )
                await self._run_rebuild_once(kind)
                # Reset the *rate limiter* even when the rebuild fails, so the
                # fallback fires at most once per threshold failures. A failed
                # rebuild defers cleanup to the 12h periodic sweep — harmless
                # for correctness (see _run_rebuild_once docstring).
                #
                # ``optimize_failures`` is deliberately NOT reset here: it is
                # the health signal, and zeroing it in the branch that fires at
                # the threshold is what made the alert unreachable.
                if state is not None:
                    state.failures_since_fallback = 0

    async def heartbeat_loop(self) -> None:
        """Periodic safety net for the optimizer.

        Sweeps every kind through :meth:`schedule_optimize` once per
        ``optimize_heartbeat_seconds``. Without this, a worker that
        restarts with an unindexed tail (e.g. after a crash between
        write and optimize) would only merge it in once new writes
        arrive — those rows stay searchable meanwhile (flat-scan), but
        the tail keeps the scan slow and the dead files on disk; the
        sweep bounds both. It goes through the same throttle gate so it
        can never storm — kinds with an in-flight optimize or a fresh
        ``last_run_at`` are coalesced.
        """
        while not self._stop.is_set():
            if await wait_or_stop(self._stop, self._optimize_heartbeat):
                return
            for kind in self._handlers:
                self.schedule_optimize(kind)

    async def rebuild_loop(self) -> None:
        """Slow per-kind ``drop_index + create_index`` loop.

        Workaround for the upstream lancedb / lance gap documented on
        :meth:`LanceRepoBase.rebuild_indexes`. Every
        ``_optimize_rebuild_interval`` seconds we sweep each kind and
        do a full rebuild — this is the **only** lever we have on the
        current stack (lancedb 0.30.2 / lance 4.0) to bound active
        index UUID / FTS ``part_N`` accumulation.

        First sweep fires immediately on worker start to bound any
        accumulation from a previous session. Subsequent sweeps honour
        the interval. Both sweep and each per-kind step respect
        ``self._stop`` so shutdown is prompt.

        Rebuild is serialised through the per-kind
        :class:`KindOptimizerState.task` slot so it does not race with
        an in-flight ``optimize()``. Failures are caught and logged —
        a missed rebuild just defers cleanup to the next sweep, which
        is harmless for correctness (queries / writes keep working
        against the existing indices).
        """
        # First sweep: catch up from any prior session before honouring the
        # interval. Rebuild is cheap (~0.3s per 50k rows × indexed columns
        # in local benchmarks); deferring it 12h after startup risks long
        # accumulation if the daemon restarts often.
        for kind in self._handlers:
            if self._stop.is_set():
                return
            await self._run_rebuild_once(kind)
        last_sweep = time.monotonic()
        # Never coarser than the configured cadence: the tick exists to give
        # conflict retries a 60s granularity against a 12h sweep, and must not
        # quantise a deployment (or a test) that sets a shorter interval.
        tick = min(REBUILD_LOOP_TICK_SECONDS, self._optimize_rebuild_interval)
        while not self._stop.is_set():
            if await wait_or_stop(self._stop, tick):
                return
            now = time.monotonic()
            if now - last_sweep >= self._optimize_rebuild_interval:
                last_sweep = now
                for kind in self._handlers:
                    if self._stop.is_set():
                        return
                    await self._run_rebuild_once(kind)
                continue
            # Between sweeps, serve only kinds whose conflict backoff is due.
            for kind in self._handlers:
                state = self._optimizer_states.get(kind)
                if state is None or not state.rebuild_retry_at:
                    continue
                if now < state.rebuild_retry_at:
                    continue
                state.rebuild_retry_at = 0.0
                if self._stop.is_set():
                    return
                await self._run_rebuild_once(kind)

    async def _run_rebuild_once(self, kind: str) -> None:
        """Drop + re-create all indexes on ``kind``'s LanceDB table.

        Waits for any in-flight ``optimize()`` task to settle, then
        claims the per-kind task slot so ``schedule_optimize`` calls
        during the rebuild coalesce instead of racing on the manifest.
        """
        repo = self._index_for(kind)
        if repo is None:
            return
        state = self._optimizer_states.setdefault(kind, KindOptimizerState())
        # Drain any in-flight optimize before taking the rebuild slot —
        # both would commit on the same manifest version. The optimize
        # runner reciprocates (it awaits ``state.rebuild_task`` on entry).
        # Skip when ``state.task`` is the current task: the fallback-rebuild
        # path in ``_run_optimize_once`` reaches here from *inside* the
        # optimize runner itself, so awaiting ``state.task`` would be
        # self-await (asyncio raises RuntimeError). Suppress catches it,
        # but relying on that is fragile — the explicit check is the
        # correctness contract; suppress remains only for unexpected
        # optimize failures.
        if (
            state.task is not None
            and not state.task.done()
            and state.task is not asyncio.current_task()
        ):
            # Bounded: an optimize task that hangs must not park the rebuild
            # loop behind it. Same defect class as the table-handle await —
            # a wait with no deadline in a path a scheduler depends on.
            # ``asyncio.wait`` so the deadline only stops the waiting: the
            # ``asyncio.timeout`` form cancelled the runner it was waiting for,
            # killing the beat it was meant to stay clear of (see the mirror
            # wait in ``_optimize_runner``).
            done, _ = await asyncio.wait(
                {state.task}, timeout=self._maintenance_timeout
            )
            if not done:
                # Skip this sweep rather than rebuild concurrently with an
                # optimize that is still running: dropping indices under it is
                # exactly the interleaving this wait exists to prevent. The 12h
                # loop retries, and the prune-staleness signal covers the stall.
                logger.warning(
                    "cascade_lancedb_rebuild_skipped_optimize_unfinished",
                    kind=kind,
                    waited_seconds=self._maintenance_timeout,
                )
                return
        # Retry a lost commit race in place. Lance labels it "Retryable" and
        # means it: the rebuild transaction was preempted by a concurrent
        # writer (another process, since the write lock is in-process only) and
        # nothing about the table is wrong. Without a retry, a conflict costs a
        # whole rebuild cadence — 12h in production — for what a second-scale
        # backoff resolves. A soak run at 600s cadence hit 3 conflicts in 119
        # attempts (2.5%), all while a concurrent CLI storm was running.
        rebuild_task = asyncio.create_task(
            repo.rebuild_indexes(), name=f"cascade-rebuild-{kind}-inner"
        )
        state.rebuild_task = rebuild_task
        try:
            await rebuild_task
            logger.info("cascade_lancedb_rebuilt", kind=kind)
            state.rebuild_attempt = 0
            state.rebuild_retry_at = 0.0
        except Exception as exc:
            attempt = state.rebuild_attempt
            if is_benign_commit_conflict(exc) and attempt < len(
                self._rebuild_conflict_backoffs
            ):
                # Lost the manifest race to a concurrent writer; the table is
                # fine. Record a deadline instead of sleeping so the other
                # kinds in this sweep are not parked behind the backoff.
                delay = self._rebuild_conflict_backoffs[attempt]
                state.rebuild_attempt = attempt + 1
                state.rebuild_retry_at = time.monotonic() + delay
                logger.info(
                    "cascade_lancedb_rebuild_conflict_retry_scheduled",
                    kind=kind,
                    attempt=attempt,
                    retry_in_seconds=delay,
                    error=f"{type(exc).__name__}: {exc}",
                )
            else:
                state.rebuild_attempt = 0
                state.rebuild_retry_at = 0.0
                logger.warning(
                    "cascade_lancedb_rebuild_failed",
                    kind=kind,
                    attempt=attempt,
                    error=f"{type(exc).__name__}: {exc}",
                )
        finally:
            if state.rebuild_task is rebuild_task:
                state.rebuild_task = None

    async def flush(self) -> None:
        """Wait for every in-flight optimize task to settle.

        Drain-loop path is fire-and-forget for throughput; this is the
        explicit barrier used by CLI ``cascade sync`` and worker
        shutdown to let in-flight optimizes finish merging the unindexed
        tail before the call returns. Not a visibility barrier — drained
        rows are searchable via flat-scan regardless; this just yields a
        fully merged index (and, on shutdown, no orphaned mid-write).

        Exceptions from optimize tasks are already logged in
        :meth:`_run_optimize_once`; ``return_exceptions=True`` here
        keeps the flush itself from raising.
        """
        pending: list[asyncio.Task[None]] = []
        for state in self._optimizer_states.values():
            if state.task is not None and not state.task.done():
                pending.append(state.task)
            if state.rebuild_task is not None and not state.rebuild_task.done():
                pending.append(state.rebuild_task)
        if not pending:
            return
        await asyncio.gather(*pending, return_exceptions=True)

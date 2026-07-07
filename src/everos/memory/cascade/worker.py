"""Cascade worker — consumes pending rows and runs the matching handler.

The worker is the only piece that crosses the md → LanceDB boundary.
Each cycle:

1. ``claim_pending_batch(BATCH_SIZE)`` atomically flips pending rows to
   ``processing`` and returns them in LSN order.
2. For each row, look up the kind's :class:`Handler` and call either
   :meth:`handle_added_or_modified` or :meth:`handle_deleted` based on
   the row's ``change_type``.
3. On success: ``mark_done``.
4. On :class:`RecoverableError`: retry inline up to ``MAX_RETRY``; if
   all attempts fail, ``mark_failed(retryable=True)``.
5. On any other exception: ``mark_failed(retryable=False)`` (treated
   as unrecoverable, surfaces in ``cascade fix`` for the user to
   triage by editing the md).

Batch processing is concurrent inside a batch (``asyncio.gather``);
ordering across rows is best-effort — the LSN gives a deterministic
prefix but the handlers themselves are independent.

After a batch completes, each kind that mutated its LanceDB table is
passed to :meth:`_schedule_optimize` — a per-kind throttle + trailing
edge scheduler that fires LanceDB ``optimize()`` as a separate task,
so the drain loop is never blocked by index maintenance. ``optimize()``
is a performance/storage-hygiene step, **not** a visibility one: new
rows are searchable immediately via LanceDB's flat-scan over the
unindexed tail (see :meth:`LanceRepoBase.optimize`), so optimizing only
keeps that tail small and prunes dead files. A 60-second heartbeat
sweeps every kind through the same gate so an unindexed tail doesn't
accumulate after a worker restart even without new writes. See
:meth:`_schedule_optimize` for the exact semantics.

A separate 12-hour loop (:meth:`_rebuild_loop`) does a full
``drop_index + create_index`` per kind to bound the **active** index
UUID / FTS segment count growth — a workaround for an upstream gap
in the lancedb Python async API; see
:meth:`everos.core.persistence.lancedb.LanceRepoBase.rebuild_indexes`
for the full provenance and the conditions under which this scheduler
can be removed.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import time
from dataclasses import dataclass

from everos.core.observability.logging import get_logger
from everos.infra.persistence.sqlite import MdChangeState, md_change_state_repo

from .errors import RecoverableError
from .handlers import Handler

logger = get_logger(__name__)

# Conservative defaults — surface in settings if tuning is needed.
DEFAULT_BATCH_SIZE = 50
DEFAULT_MAX_RETRY = 3
DEFAULT_POLL_INTERVAL_SECONDS = 1.0
DEFAULT_RETRY_BACKOFF_SECONDS = 2.0
DEFAULT_OPTIMIZE_MIN_INTERVAL_SECONDS = 10.0
DEFAULT_OPTIMIZE_HEARTBEAT_SECONDS = 60.0
DEFAULT_OPTIMIZE_REBUILD_INTERVAL_SECONDS = 12 * 60 * 60.0
"""How often (per kind) to do a full ``drop_index + create_index`` rebuild.

This is the **only** application-level mechanism we have to bound the
active index UUID / segment count growth — see
:meth:`LanceRepoBase.rebuild_indexes` for the full provenance: Rust
``OptimizeOptions.num_indices_to_merge`` is the right knob but
``lancedb.AsyncTable.optimize()`` does not expose it (verified on
lancedb main 2026-05-28), and on the embedded ``lance crate 4.0`` the
merge behaviour itself is broken so even calling Lance directly
wouldn't help.

12 hours is a conservative pick: rebuild cost is ~0.3s per 50k rows
× indexed columns (measured locally), so even a small EverOS
deployment can absorb it without scheduling around peaks. Smaller
intervals work fine functionally; we just don't need them — under
realistic single-user / small-team write rates 12h keeps active UUIDs
bounded well below any FD ceiling. Tune via the constructor argument.

**Remove this scheduler** once lancedb exposes ``num_indices_to_merge``
on the async Python API and the embedded lance crate ships the
working merge implementation; ``optimize(num_indices_to_merge=1)``
in the regular hot path will do the same job for ~free.
"""

DEFAULT_OPTIMIZE_PRUNE_INTERVAL_SECONDS = 300.0
"""How often (per kind) to add ``cleanup_older_than`` to ``optimize()``.

``optimize()`` without ``cleanup_older_than`` compacts fragments and
merges new data into indexes, but **leaves stale physical files on disk
forever** (replaced data fragments, historical manifests, stale index
UUID files). On a lightweight (single-user / small-team) deployment
with steady-state cascade ingest, that file count grows without bound
and eventually
exhausts file descriptors at index-scan time (observed: macOS / Linux
default ``ulimit -n`` of 1024 — the ``os error 24`` reported in CI).

The prune itself is cheap when scoped to recent versions; we just don't
want to pay it on every optimize throttle tick. 5 minutes is the
shortest interval that comfortably outlives any in-flight query / index
build, while keeping the on-disk footprint bounded. It is also passed
as ``cleanup_older_than`` itself (semantically: "the retention window
equals the prune cadence") — every file replaced more than one cadence
ago becomes eligible.

Does **not** shrink active index internals (FTS ``part_N`` count or
vector index UUID count): those only collapse via ``drop_index +
create_index``, which is intentionally out of scope here.
"""


@dataclass
class _KindOptimizerState:
    """Per-kind throttle state for LanceDB ``optimize()``.

    ``dirty`` is the trailing-edge signal: every write sets it, the
    runner consumes it before each ``optimize()`` call. If a write
    arrives mid-optimize, ``dirty`` is re-raised and the runner loops
    once more after honouring the throttle interval.

    ``task`` holds the in-flight runner; at most one runner exists
    per kind so concurrent LanceDB writes never collide on the same
    table's manifest.

    ``last_prune_at`` is the monotonic timestamp of the last
    ``optimize()`` call that passed ``cleanup_older_than``; the runner
    consults it to decide whether the next call should also prune. ``0``
    means "never pruned" — the first run after worker startup always
    prunes, which is what we want for catching up from a prior session.
    """

    last_run_at: float = 0.0
    last_prune_at: float = 0.0
    dirty: bool = False
    task: asyncio.Task[None] | None = None
    rebuild_task: asyncio.Task[None] | None = None
    """In-flight rebuild task slot, separate from ``task`` so ordinary
    ``_schedule_optimize`` calls during a rebuild can still register
    ``dirty`` + spawn an optimize runner. The runner itself waits for
    ``rebuild_task`` before touching the LanceDB manifest, so the two
    operations never race on commits — only the dispatch slot is split.
    """


class CascadeWorker:
    """Owns the claim → dispatch → mark cycle.

    Created with the ``{kind: Handler}`` map produced by
    :func:`memory.cascade.registry.build_handlers`. Holds no other
    state — every per-row decision goes through the repo.
    """

    def __init__(
        self,
        handlers: dict[str, Handler],
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_retry: int = DEFAULT_MAX_RETRY,
        poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
        retry_backoff_seconds: float = DEFAULT_RETRY_BACKOFF_SECONDS,
        optimize_min_interval_seconds: float = DEFAULT_OPTIMIZE_MIN_INTERVAL_SECONDS,
        optimize_heartbeat_seconds: float = DEFAULT_OPTIMIZE_HEARTBEAT_SECONDS,
        optimize_prune_interval_seconds: float = (
            DEFAULT_OPTIMIZE_PRUNE_INTERVAL_SECONDS
        ),
        optimize_rebuild_interval_seconds: float = (
            DEFAULT_OPTIMIZE_REBUILD_INTERVAL_SECONDS
        ),
    ) -> None:
        self._handlers = handlers
        self._batch_size = batch_size
        self._max_retry = max_retry
        self._poll_interval = poll_interval_seconds
        self._retry_backoff = retry_backoff_seconds
        self._optimize_min_interval = optimize_min_interval_seconds
        self._optimize_heartbeat = optimize_heartbeat_seconds
        self._optimize_prune_interval = optimize_prune_interval_seconds
        self._optimize_rebuild_interval = optimize_rebuild_interval_seconds
        self._task: asyncio.Task[None] | None = None
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._rebuild_task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()
        self._optimizer_states: dict[str, _KindOptimizerState] = {}

    async def start(self) -> None:
        if self._task is not None:
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._run_loop(), name="cascade-worker")
        self._heartbeat_task = asyncio.create_task(
            self._heartbeat_loop(), name="cascade-worker-heartbeat"
        )
        self._rebuild_task = asyncio.create_task(
            self._rebuild_loop(), name="cascade-worker-rebuild"
        )
        logger.info("cascade_worker_started", batch_size=self._batch_size)

    async def stop(self) -> None:
        if self._task is None:
            return
        self._stop.set()
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await self._task
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._heartbeat_task
        if self._rebuild_task is not None:
            self._rebuild_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._rebuild_task
        # Optimize tasks coalesce on the stop signal (their inter-run
        # cooldowns observe ``self._stop``), so flushing them just
        # waits out the currently in-flight commit rather than
        # blocking on a fresh throttle window.
        await self._flush_optimizers()
        self._task = None
        self._heartbeat_task = None
        self._rebuild_task = None
        logger.info("cascade_worker_stopped")

    async def drain_once(self) -> int:
        """Process one batch, return the number of rows handled.

        Used by CLI ``cascade sync`` and ``fix --apply`` to flush the
        queue without spinning the background task. Returns ``0`` when
        the queue is empty.

        For each kind that mutated its LanceDB table this batch,
        :meth:`_schedule_optimize` records a throttled optimize
        intent. The actual ``optimize()`` runs as a separate task so
        drain throughput is decoupled from index maintenance. Drained
        rows are already searchable at this point (flat-scan over the
        unindexed tail); callers that additionally want the index fully
        merged before returning (CLI ``cascade sync``) call
        :meth:`_flush_optimizers` — :meth:`drain_until_empty` does this
        on their behalf.
        """
        batch = await md_change_state_repo.claim_pending_batch(self._batch_size)
        if not batch:
            return 0
        results = await asyncio.gather(*(self._process_one(row) for row in batch))
        touched_kinds = {kind for kind in results if kind is not None}
        for kind in touched_kinds:
            self._schedule_optimize(kind)
        return len(batch)

    async def drain_until_empty(self, *, max_passes: int = 100) -> int:
        """Drain repeatedly until the queue is empty (or ``max_passes``).

        Returns the total number of rows processed. Bounded passes
        prevent a livelock if a stuck row keeps re-failing back to
        pending (which can't happen in the current design but is a
        cheap safety net).

        Awaits :meth:`_flush_optimizers` before returning so callers
        (CLI ``cascade sync``) get a fully merged index — not for
        visibility (the data is already searchable) but so ``sync``
        returns a deterministically optimized state.
        """
        total = 0
        for _ in range(max_passes):
            processed = await self.drain_once()
            if processed == 0:
                break
            total += processed
        await self._flush_optimizers()
        return total

    # ── internals ──────────────────────────────────────────────────────────

    async def _run_loop(self) -> None:
        while not self._stop.is_set():
            try:
                processed = await self.drain_once()
            except Exception as exc:
                logger.exception("cascade_worker_drain_failed", error=str(exc))
                processed = 0
            if processed == 0:
                try:
                    await asyncio.wait_for(
                        self._stop.wait(), timeout=self._poll_interval
                    )
                except TimeoutError:
                    continue

    async def _process_one(self, row: MdChangeState) -> str | None:
        """Process one ``md_change_state`` row.

        Returns the ``row.kind`` when the handler actually mutated the
        kind's LanceDB table (``upserted`` or ``deleted`` > 0) so the
        caller can collect a set of "touched kinds" and optimize them
        after the batch. Returns ``None`` for skipped-only rows, failed
        rows, and rows where no handler is registered — the optimize
        step is gated on actual writes happening this batch.
        """
        handler = self._handlers.get(row.kind)
        if handler is None:
            await md_change_state_repo.mark_failed(
                row.md_path,
                retryable=False,
                error=f"no handler registered for kind {row.kind!r}",
                new_retry_count=row.retry_count,
            )
            return None

        retry_count = row.retry_count
        last_error: str = ""
        for attempt in range(self._max_retry + 1):
            try:
                if row.change_type == "deleted":
                    outcome = await handler.handle_deleted(row.md_path)
                else:
                    outcome = await handler.handle_added_or_modified(row.md_path)
            except RecoverableError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "cascade_worker_recoverable",
                    md_path=row.md_path,
                    attempt=attempt,
                    error=last_error,
                )
                if attempt < self._max_retry:
                    retry_count += 1
                    await asyncio.sleep(self._retry_backoff * (attempt + 1))
                    continue
                await md_change_state_repo.mark_failed(
                    row.md_path,
                    retryable=True,
                    error=last_error,
                    new_retry_count=retry_count,
                )
                return None
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                logger.exception(
                    "cascade_worker_unrecoverable",
                    md_path=row.md_path,
                    kind=row.kind,
                )
                await md_change_state_repo.mark_failed(
                    row.md_path,
                    retryable=False,
                    error=last_error,
                    new_retry_count=retry_count,
                )
                return None

            logger.info(
                "cascade_worker_processed",
                md_path=row.md_path,
                kind=row.kind,
                change_type=row.change_type,
                upserted=outcome.upserted,
                deleted=outcome.deleted,
                skipped=outcome.skipped,
            )
            await md_change_state_repo.mark_done(row.md_path)
            # Only flag the kind as "touched" when we actually wrote
            # something — skipped rows leave the table untouched, so
            # optimizing would be pure overhead.
            return row.kind if (outcome.upserted or outcome.deleted) else None
        return None

    # ── optimizer scheduling ───────────────────────────────────────────────

    def _schedule_optimize(self, kind: str) -> None:
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
        ``lance_repo`` (test stubs, handlers that intentionally skip
        LanceDB).

        Idempotent and cheap (a single dict lookup + flag write in
        the hot path) — safe to call on every batch and from the
        heartbeat sweep.
        """
        handler = self._handlers.get(kind)
        repo = getattr(handler, "lance_repo", None) if handler else None
        if repo is None:
            return
        state = self._optimizer_states.setdefault(kind, _KindOptimizerState())
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
            if initial_delay > 0 and await self._wait_or_stop(initial_delay):
                return
            # Serialise behind any in-flight rebuild (rare; only during
            # the 12h sweep). Failures are absorbed in _run_rebuild_once.
            if state.rebuild_task is not None and not state.rebuild_task.done():
                with contextlib.suppress(Exception):
                    await state.rebuild_task
            while state.dirty and not self._stop.is_set():
                state.dirty = False
                state.last_run_at = time.monotonic()
                await self._run_optimize_once(kind)
                if (
                    state.dirty
                    and not self._stop.is_set()
                    and await self._wait_or_stop(self._optimize_min_interval)
                ):
                    return
        finally:
            if state.task is asyncio.current_task():
                state.task = None

    async def _wait_or_stop(self, seconds: float) -> bool:
        """Sleep up to ``seconds``; return True if stop was set."""
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=seconds)
        except TimeoutError:
            return False
        return True

    async def _run_optimize_once(self, kind: str) -> None:
        """Run one ``optimize()`` for ``kind``, opportunistically pruning.

        Most calls take the **light** path — pure compaction + index
        merge, fast. Every ``_optimize_prune_interval`` seconds the
        next call takes the **heavy** path: same work plus
        ``cleanup_older_than`` so the storage layer physically deletes
        files belonging to versions older than one cadence.

        Pruning is opt-in per call rather than a separate task so the
        existing per-kind serialisation (one in-flight runner per kind)
        keeps holding — LanceDB serialises writes on the table's
        manifest, and prune is a write.
        """
        handler = self._handlers.get(kind)
        repo = getattr(handler, "lance_repo", None) if handler else None
        if repo is None:
            return
        state = self._optimizer_states.get(kind)
        now = time.monotonic()
        should_prune = (
            state is None
            or (now - state.last_prune_at) >= self._optimize_prune_interval
        )
        cleanup = (
            dt.timedelta(seconds=self._optimize_prune_interval)
            if should_prune
            else None
        )
        try:
            await repo.optimize(cleanup_older_than=cleanup)
            if should_prune and state is not None:
                state.last_prune_at = now
            logger.debug(
                "cascade_lancedb_optimized",
                kind=kind,
                pruned=should_prune,
            )
        except Exception as exc:
            logger.warning(
                "cascade_lancedb_optimize_failed",
                kind=kind,
                pruned=should_prune,
                error=f"{type(exc).__name__}: {exc}",
            )

    async def _heartbeat_loop(self) -> None:
        """Periodic safety net for the optimizer.

        Sweeps every kind through :meth:`_schedule_optimize` once per
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
            if await self._wait_or_stop(self._optimize_heartbeat):
                return
            for kind in self._handlers:
                self._schedule_optimize(kind)

    async def _rebuild_loop(self) -> None:
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
        :class:`_KindOptimizerState.task` slot so it does not race with
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
        while not self._stop.is_set():
            if await self._wait_or_stop(self._optimize_rebuild_interval):
                return
            for kind in self._handlers:
                if self._stop.is_set():
                    return
                await self._run_rebuild_once(kind)

    async def _run_rebuild_once(self, kind: str) -> None:
        """Drop + re-create all indexes on ``kind``'s LanceDB table.

        Waits for any in-flight ``optimize()`` task to settle, then
        claims the per-kind task slot so ``schedule_optimize`` calls
        during the rebuild coalesce instead of racing on the manifest.
        """
        handler = self._handlers.get(kind)
        repo = getattr(handler, "lance_repo", None) if handler else None
        if repo is None:
            return
        state = self._optimizer_states.setdefault(kind, _KindOptimizerState())
        # Drain any in-flight optimize before taking the rebuild slot —
        # both would commit on the same manifest version. The optimize
        # runner reciprocates (it awaits ``state.rebuild_task`` on entry).
        if state.task is not None and not state.task.done():
            with contextlib.suppress(Exception):
                await state.task
        rebuild_task = asyncio.create_task(
            repo.rebuild_indexes(), name=f"cascade-rebuild-{kind}-inner"
        )
        state.rebuild_task = rebuild_task
        try:
            await rebuild_task
            logger.info("cascade_lancedb_rebuilt", kind=kind)
        except Exception as exc:
            logger.warning(
                "cascade_lancedb_rebuild_failed",
                kind=kind,
                error=f"{type(exc).__name__}: {exc}",
            )
        finally:
            if state.rebuild_task is rebuild_task:
                state.rebuild_task = None

    async def _flush_optimizers(self) -> None:
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

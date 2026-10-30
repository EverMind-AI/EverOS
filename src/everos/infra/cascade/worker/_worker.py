"""Cascade worker — consumes pending rows and runs the matching handler.

The worker is the only piece that crosses the md → LanceDB boundary.
Each cycle:

1. ``claim_pending_batch(BATCH_SIZE)`` atomically flips pending rows to
   ``processing`` and returns them in LSN order.
2. For each row, look up the kind's :class:`Handler` and call either
   :meth:`handle_added_or_modified` or :meth:`handle_deleted` based on
   the row's ``change_type``.
3. On success: ``mark_done`` with the change the handler actually
   applied (a vanished file folds into a deletion and is recorded as
   ``deleted``, so the reconciler re-adds it if it comes back).
4. On :class:`~everos.core.errors.ExternalServiceError`: retry inline
   up to ``MAX_RETRY``; if all attempts fail,
   ``mark_failed(retryable=True)`` — unless the cross-cycle retry
   budget (:data:`MAX_TOTAL_RETRIES`) is also exhausted, in which
   case ``retryable=False`` is set here directly so the row skips
   one extra scanner cycle before the entry-check would demote it.
5. On any other exception: ``mark_failed(retryable=False)`` (treated
   as unrecoverable, surfaces in ``cascade fix`` for the user to
   triage by editing the md).

Before step 2, a row whose ``retry_count`` has already reached
:data:`MAX_TOTAL_RETRIES` (across scanner re-enqueue cycles, not just
this batch's inline retries) is short-circuited straight to
``mark_failed(retryable=False)`` without invoking the handler — this
bounds the total retry budget so a persistently-failing row cannot
retry forever.

Batch processing is bounded-concurrent inside a batch; rows are claimed in
batches of 50 but only a small fixed number of handlers execute at once.
Ordering across rows is best-effort — the LSN gives a deterministic prefix
but the handlers themselves are independent.

After a batch completes, each kind that mutated its LanceDB table is handed
to the :class:`IndexMaintainer` (``schedule_optimize``), which owns the
throttled optimize / prune beats, the heartbeat sweep and the 12-hour index
rebuild — see :mod:`._maintenance`. The three long-lived loops (drain,
heartbeat, rebuild) run under a :class:`LoopSupervisor`
(:mod:`._supervision`). Tuning values and their rationale live in
:mod:`._tuning`.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Callable, Sequence

from everos.core.errors import ExternalServiceError
from everos.core.observability.logging import get_logger
from everos.infra.persistence.sqlite import MdChangeState, MdChangeStateRepo

from ..models import ChangeType
from ..projections import Handler
from ._health import CascadeWorkerHealth
from ._maintenance import IndexMaintainer
from ._supervision import LoopSupervisor, request_process_exit
from ._tuning import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_HANDLER_CONCURRENCY,
    DEFAULT_MAX_RETRY,
    DEFAULT_OPTIMIZE_HEARTBEAT_SECONDS,
    DEFAULT_OPTIMIZE_MIN_INTERVAL_SECONDS,
    DEFAULT_OPTIMIZE_PRUNE_INTERVAL_SECONDS,
    DEFAULT_OPTIMIZE_PRUNE_RETENTION_SECONDS,
    DEFAULT_OPTIMIZE_REBUILD_INTERVAL_SECONDS,
    DEFAULT_POLL_INTERVAL_SECONDS,
    DEFAULT_RETRY_BACKOFF_SECONDS,
    LOOP_RESTART_BACKOFF_SECONDS,
    MAINTENANCE_TASK_TIMEOUT_SECONDS,
    MAX_TOTAL_RETRIES,
    REBUILD_CONFLICT_BACKOFFS_SECONDS,
)

logger = get_logger(__name__)


class CascadeWorker:
    """Owns the claim → dispatch → mark cycle.

    Created with the ``{kind: Handler}`` map produced by
    :func:`infra.cascade.registry.build_handlers` and the ``md_change_state``
    ledger it claims rows from. Every per-row decision goes through the
    ledger; index maintenance and loop supervision are delegated to the
    :class:`IndexMaintainer` and :class:`LoopSupervisor` it owns.

    The timing keyword arguments default to the constants in
    :mod:`._tuning`, whose docstrings say how each is sized.
    ``on_unrecoverable`` is called with a loop's name once its restart budget
    is spent; by default it asks this process to exit (see
    :func:`request_process_exit`).
    """

    def __init__(
        self,
        handlers: dict[str, Handler],
        ledger: MdChangeStateRepo,
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
        handler_concurrency: int = DEFAULT_HANDLER_CONCURRENCY,
        max_retry: int = DEFAULT_MAX_RETRY,
        poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
        retry_backoff_seconds: float = DEFAULT_RETRY_BACKOFF_SECONDS,
        optimize_min_interval_seconds: float = DEFAULT_OPTIMIZE_MIN_INTERVAL_SECONDS,
        optimize_heartbeat_seconds: float = DEFAULT_OPTIMIZE_HEARTBEAT_SECONDS,
        optimize_prune_interval_seconds: float = (
            DEFAULT_OPTIMIZE_PRUNE_INTERVAL_SECONDS
        ),
        optimize_prune_retention_seconds: float = (
            DEFAULT_OPTIMIZE_PRUNE_RETENTION_SECONDS
        ),
        optimize_rebuild_interval_seconds: float = (
            DEFAULT_OPTIMIZE_REBUILD_INTERVAL_SECONDS
        ),
        maintenance_task_timeout_seconds: float = MAINTENANCE_TASK_TIMEOUT_SECONDS,
        rebuild_conflict_backoffs_seconds: Sequence[float] = (
            REBUILD_CONFLICT_BACKOFFS_SECONDS
        ),
        loop_restart_backoffs_seconds: Sequence[float] = LOOP_RESTART_BACKOFF_SECONDS,
        on_unrecoverable: Callable[[str], None] | None = None,
    ) -> None:
        if handler_concurrency <= 0:
            raise ValueError("handler_concurrency must be positive")
        self._handlers = handlers
        self._ledger = ledger
        self._batch_size = batch_size
        self._handler_concurrency = handler_concurrency
        self._handler_semaphore = asyncio.Semaphore(handler_concurrency)
        self._max_retry = max_retry
        self._poll_interval = poll_interval_seconds
        self._retry_backoff = retry_backoff_seconds
        self._stop = asyncio.Event()
        self._maintenance = IndexMaintainer(
            handlers,
            self._stop,
            optimize_min_interval_seconds=optimize_min_interval_seconds,
            optimize_heartbeat_seconds=optimize_heartbeat_seconds,
            optimize_prune_interval_seconds=optimize_prune_interval_seconds,
            optimize_prune_retention_seconds=optimize_prune_retention_seconds,
            optimize_rebuild_interval_seconds=optimize_rebuild_interval_seconds,
            maintenance_task_timeout_seconds=maintenance_task_timeout_seconds,
            rebuild_conflict_backoffs_seconds=rebuild_conflict_backoffs_seconds,
        )
        self._supervisor = LoopSupervisor(
            self._stop,
            restart_backoffs=loop_restart_backoffs_seconds,
            on_unrecoverable=on_unrecoverable or request_process_exit,
        )
        self._task: asyncio.Task[None] | None = None
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._rebuild_task: asyncio.Task[None] | None = None
        # ── in-memory health signals (see :meth:`health`) ──────────────
        self._started_at: float = 0.0
        self._drain_consecutive_failures: int = 0
        self._unrecoverable_total: int = 0

    async def start(self) -> None:
        if self._task is not None:
            return
        self._stop.clear()
        self._started_at = time.monotonic()
        self._task = self._supervisor.spawn("drain", self._run_loop, "cascade-worker")
        self._heartbeat_task = self._supervisor.spawn(
            "heartbeat", self._maintenance.heartbeat_loop, "cascade-worker-heartbeat"
        )
        self._rebuild_task = self._supervisor.spawn(
            "rebuild", self._maintenance.rebuild_loop, "cascade-worker-rebuild"
        )
        logger.info(
            "cascade_worker_started",
            batch_size=self._batch_size,
            handler_concurrency=self._handler_concurrency,
        )

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
        await self._maintenance.flush()
        self._task = None
        self._heartbeat_task = None
        self._rebuild_task = None
        logger.info("cascade_worker_stopped")

    async def drain_once(self, *, kinds: set[str] | None = None) -> int:
        """Process one batch, return the number of rows handled.

        Used by CLI ``cascade sync`` and ``fix --apply`` to flush the
        queue without spinning the background task. Returns ``0`` when
        the queue is empty.

        For each kind that mutated its LanceDB table this batch,
        :meth:`IndexMaintainer.schedule_optimize` records a throttled optimize
        intent. The actual ``optimize()`` runs as a separate task so
        drain throughput is decoupled from index maintenance. Drained
        rows are already searchable at this point (flat-scan over the
        unindexed tail); callers that additionally want the index fully
        merged before returning (CLI ``cascade sync``) call
        :meth:`IndexMaintainer.flush` — :meth:`drain_until_empty` does this
        on their behalf.

        Args:
            kinds: Optional restriction on which ``kind`` values are
                claimed. ``None`` (the default) claims across every
                kind. Phase-3 backfill passes ``{"agent_skill"}`` so
                the drain cannot flip an unrelated queued kind (e.g. a
                queued ``knowledge_topic`` md whose handler this
                process doesn't have registered) to
                ``failed(retryable=False)``.
        """
        batch = await self._ledger.claim_pending_batch(self._batch_size, kinds=kinds)
        if not batch:
            return 0

        async def process_bounded(row: MdChangeState) -> str | None:
            async with self._handler_semaphore:
                return await self._process_one(row)

        results = await asyncio.gather(*(process_bounded(row) for row in batch))
        touched_kinds = {kind for kind in results if kind is not None}
        for kind in touched_kinds:
            self._maintenance.schedule_optimize(kind)
        return len(batch)

    async def drain_until_empty(
        self,
        *,
        max_passes: int = 100,
        kinds: set[str] | None = None,
    ) -> int:
        """Drain repeatedly until the queue is empty (or ``max_passes``).

        Returns the total number of rows processed. Bounded passes
        prevent a livelock if a stuck row keeps re-failing back to
        pending (which can't happen in the current design but is a
        cheap safety net).

        Awaits :meth:`IndexMaintainer.flush` before returning so callers
        (CLI ``cascade sync``) get a fully merged index — not for
        visibility (the data is already searchable) but so ``sync``
        returns a deterministically optimized state.

        The ``kinds`` filter is forwarded to :meth:`drain_once` on every
        pass so the scoping intent holds end-to-end for scoped syncs
        (Phase-3 backfill's ``{"agent_skill"}`` sweep); ``None`` keeps
        the CLI ``cascade sync`` path drainining every kind.
        """
        total = 0
        for _ in range(max_passes):
            processed = await self.drain_once(kinds=kinds)
            if processed == 0:
                break
            total += processed
        await self._maintenance.flush()
        return total

    def health(self) -> CascadeWorkerHealth:
        """Snapshot the worker's in-memory health signals (no IO).

        Combines the drain-loop and unrecoverable counters with the worst
        per-kind optimize streak and the prune-staleness clock.
        """
        stale_seconds, stale_kind = self._maintenance.prune_staleness(self._started_at)
        return CascadeWorkerHealth(
            drain_consecutive_failures=self._drain_consecutive_failures,
            unrecoverable_total=self._unrecoverable_total,
            optimize_failure_streak=self._maintenance.optimize_failure_streak(),
            prune_stale_seconds=stale_seconds,
            prune_stale_kind=stale_kind,
        )

    # ── internals ──────────────────────────────────────────────────────────

    async def _run_loop(self) -> None:
        while not self._stop.is_set():
            try:
                processed = await self.drain_once()
                self._drain_consecutive_failures = 0
            except Exception as exc:
                self._drain_consecutive_failures += 1
                logger.exception(
                    "cascade_worker_drain_failed",
                    error=str(exc),
                    consecutive_failures=self._drain_consecutive_failures,
                )
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
            await self._ledger.mark_failed(
                row.md_path,
                retryable=False,
                error=f"no handler registered for kind {row.kind!r}",
                new_retry_count=row.retry_count,
            )
            return None

        if row.retry_count >= MAX_TOTAL_RETRIES:
            logger.warning(
                "cascade_worker_retry_budget_exhausted",
                md_path=row.md_path,
                kind=row.kind,
                retry_count=row.retry_count,
            )
            await self._ledger.mark_failed(
                row.md_path,
                retryable=False,
                error=f"retry budget exhausted after {row.retry_count} attempts",
                new_retry_count=row.retry_count,
            )
            return None

        retry_count = row.retry_count
        last_error: str = ""
        applied: ChangeType = row.change_type  # type: ignore[assignment]  # str column
        for attempt in range(self._max_retry + 1):
            try:
                if row.change_type == "deleted":
                    outcome = await handler.handle_deleted(row.md_path)
                else:
                    try:
                        outcome = await handler.handle_added_or_modified(row.md_path)
                    except FileNotFoundError:
                        # The md disappeared between scanner enqueue and here
                        # (delete/modify race — cascade delete event may not
                        # arrive if it fires while the row is already in
                        # ``processing``). Fold into a deletion so the row
                        # completes with ``mark_done`` instead of failing, and
                        # record the fold: the row must describe the index.
                        outcome = await handler.handle_deleted(row.md_path)
                        applied = "deleted"
            except ExternalServiceError as exc:
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
                # Inline attempts exhausted. Only stay retryable when
                # the cross-cycle budget still has room; otherwise
                # demote directly instead of taking an extra scanner
                # cycle to hit the entry-check short-circuit.
                await self._ledger.mark_failed(
                    row.md_path,
                    retryable=retry_count < MAX_TOTAL_RETRIES,
                    error=last_error,
                    new_retry_count=retry_count,
                )
                return None
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                self._unrecoverable_total += 1
                logger.exception(
                    "cascade_worker_unrecoverable",
                    md_path=row.md_path,
                    kind=row.kind,
                    unrecoverable_total=self._unrecoverable_total,
                )
                await self._ledger.mark_failed(
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
                change_type=applied,
                upserted=outcome.upserted,
                deleted=outcome.deleted,
                skipped=outcome.skipped,
            )
            await self._ledger.mark_done(row.md_path, change_type=applied)
            # Only flag the kind as "touched" when we actually wrote
            # something — skipped rows leave the table untouched, so
            # optimizing would be pure overhead.
            return row.kind if (outcome.upserted or outcome.deleted) else None
        return None

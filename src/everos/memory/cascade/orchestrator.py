"""Cascade orchestrator — wires watcher + scanner + worker for the lifespan.

One :class:`CascadeOrchestrator` per process. The lifespan provider
constructs it at startup, calls :meth:`start` once, and calls
:meth:`stop` at shutdown. CLI ``cascade sync`` constructs its own
instance but only invokes :meth:`drain_once` (no background tasks).

Construction is dependency-injected: the tokenizer provider and the
memory-root come in as constructor args so tests can swap them
without monkey-patching module-level singletons. Embedding is a soft
dependency handled directly by handlers via the capability accessor,
not threaded through here.
"""

from __future__ import annotations

import asyncio
import dataclasses
import os

from everos.component.tokenizer import Tokenizer
from everos.config import load_settings
from everos.core.observability.logging import get_logger
from everos.core.persistence import MemoryRoot
from everos.infra.persistence.index import count_business_rows
from everos.infra.persistence.sqlite import (
    MdChangeState,
    QueueSummary,
    md_change_state_repo,
)

from .handlers import HandlerDeps
from .registry import build_handlers
from .scanner import CascadeScanner
from .watcher import CascadeWatcher
from .worker import (
    DEFAULT_HANDLER_CONCURRENCY,
    DEFAULT_OPTIMIZE_HEARTBEAT_SECONDS,
    DEFAULT_OPTIMIZE_PRUNE_INTERVAL_SECONDS,
    DEFAULT_OPTIMIZE_PRUNE_RETENTION_SECONDS,
    DEFAULT_OPTIMIZE_REBUILD_INTERVAL_SECONDS,
    CascadeWorker,
)

logger = get_logger(__name__)

INDEX_RECONCILE_VERSION = 1
"""Bump when the index's row identity changes and every indexed md must be
reconciled against the index once. v1 (1.4.2): rows are keyed on
``(app_id, project_id, id)`` instead of ``id`` alone — a multi-project root
indexed by an earlier release lost rows to the overwrite, and the handlers'
per-entry diff brings them back.

The pass recovers rows that are missing or whose content digest changed;
it does not rewrite rows whose other columns changed. A change to a
column, a scope field or a derived value needs ``cascade rebuild``, not a
bump here."""

_RECONCILE_MARKER = ".index_reconcile_version"
_DRAIN_IDLE_LIMIT_SECONDS = 120.0
"""How long :meth:`CascadeOrchestrator.drain_fully` keeps polling a ledger
that holds rows nobody is landing (a handler hung mid-batch) before it
returns and lets the caller report what is left."""


async def reconcile_index_once(
    memory_root: MemoryRoot, *, force: bool = False
) -> list[MdChangeState]:
    """Force-enqueue every indexed md the first time a root is opened at
    :data:`INDEX_RECONCILE_VERSION`, or whenever ``force`` is set; return
    the rows enqueued (empty when the marker is already current).

    ``<index_dir>/.index_reconcile_version`` records the version the root
    was last reconciled for, so every later start is a no-op. A forced pass
    (``cascade sync --force-all``) settles the marker too, so the unscoped
    :meth:`CascadeOrchestrator.sync_once` that follows it does not queue
    the ledger a second time. The handlers skip entries whose rows are
    present and unchanged by digest, so a healthy root pays one read pass
    and no embedding; only rows that were actually lost are re-embedded.
    This takes no lock of its own. The exclusive CLI commands hold the OME
    lock for their whole run; the server acquires it only at lifespan
    order 50, after the cascade lifespan (order 12) has started the worker,
    so a server starting while ``cascade sync --force-all`` runs writes the
    index alongside it until its own lock attempt fails — a window that
    predates this pass, which now fills it with the whole ledger instead of
    the pending backlog.
    """
    marker = memory_root.index_dir / _RECONCILE_MARKER
    try:
        current = int(marker.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, ValueError):
        current = 0
    if not force and current >= INDEX_RECONCILE_VERSION:
        return []
    rows = await md_change_state_repo.force_enqueue_all()
    marker.parent.mkdir(parents=True, exist_ok=True)
    # max(): an older binary's forced pass must not roll a newer root back.
    marker.write_text(str(max(current, INDEX_RECONCILE_VERSION)), encoding="utf-8")
    logger.info(
        "cascade_index_reconcile_enqueued",
        version=INDEX_RECONCILE_VERSION,
        count=len(rows),
        forced=force,
    )
    return rows


@dataclasses.dataclass(frozen=True)
class CascadeHealth:
    """Cascade health verdict for ``/health``.

    ``healthy`` reflects **operational** health only — is the md →
    LanceDB projection pipeline itself working: drain loop alive,
    optimize not stuck, version cleanup (prune) not stalled. It is the
    boolean ops/alerting acts on.

    ``failed_permanent`` is deliberately **not** part of that verdict:
    a handful of md files failing to index is a normal data-quality
    backlog (almost always non-zero in a real deployment), so folding
    it into ``healthy`` would pin the signal red forever. It is reported
    as an informational count; per-file triage lives in the
    ``cascade status`` / ``cascade fix`` CLI. ``reasons`` is the
    operational "why not" list (empty when healthy).
    """

    healthy: bool
    reasons: list[str]
    pending: int
    failed_permanent: int
    """Informational: md files awaiting ``cascade fix``. Does NOT affect
    :attr:`healthy` (see class docstring)."""
    failed_retryable: int
    drain_consecutive_failures: int
    unrecoverable_total: int
    optimize_failure_streak: int
    prune_stale_seconds: float
    index_rows: dict[str, int]
    """Rows per business table, **as counted once at startup**.

    A snapshot for the same reason ``capabilities.embed`` is one: counting
    is a round trip per collection on a remote backend and ``/health`` is
    polled. It does not move while the process runs, so a rebuild shows up
    only after a restart -- or immediately in ``cascade status``, which
    counts live.
    """
    index_emptied: bool
    """The ledger says every file is indexed, and every table is empty.

    Deleting ``.index/lancedb`` leaves exactly this state: the ledger lives
    in ``.index/sqlite`` and survives, so the scanner enqueues nothing and
    every surface reports a healthy, fully drained pipeline over a store
    that holds nothing (issue #501). Reported without flipping
    :attr:`healthy`, like ``failed_permanent``: this is a startup snapshot,
    and a verdict that cannot clear until the next restart is a worse
    signal than a number an operator can read. Recovery is ``everos
    cascade rebuild``; markdown is untouched throughout.
    """


@dataclasses.dataclass(frozen=True)
class CascadeConfig:
    """Construction-time knobs for the orchestrator.

    Defaults are sized for a lightweight (single-user / small-team) dev box.
    The maintenance cadences come from :class:`everos.config.CascadeSettings`
    via :meth:`from_settings`, which every production construction path uses —
    they were constructor-only for long enough that the 12h rebuild sweep could
    not be exercised by any soak run short of half a day.

    Deliberately *not* configurable: the deadlines that bound a hung call
    (read / write / prune / rebuild). Those are hang-catchers sized from
    measured durations; too low manufactures failures, too high makes a wedged
    table invisible for longer. They stay as constants beside the code they
    guard, each with its measurement in the docstring.
    """

    scan_interval_seconds: float = 30.0
    worker_batch_size: int = 50
    worker_handler_concurrency: int = DEFAULT_HANDLER_CONCURRENCY
    worker_max_retry: int = 3
    worker_poll_interval_seconds: float = 1.0
    worker_retry_backoff_seconds: float = 2.0
    optimize_heartbeat_seconds: float = DEFAULT_OPTIMIZE_HEARTBEAT_SECONDS
    optimize_prune_interval_seconds: float = DEFAULT_OPTIMIZE_PRUNE_INTERVAL_SECONDS
    optimize_prune_retention_seconds: float = DEFAULT_OPTIMIZE_PRUNE_RETENTION_SECONDS
    optimize_rebuild_interval_seconds: float = DEFAULT_OPTIMIZE_REBUILD_INTERVAL_SECONDS

    @classmethod
    def from_settings(cls) -> CascadeConfig:
        """Build with maintenance cadences taken from ``[cascade]`` settings."""
        cascade = load_settings().cascade
        return cls(
            optimize_heartbeat_seconds=cascade.optimize_heartbeat_seconds,
            optimize_prune_interval_seconds=cascade.optimize_prune_interval_seconds,
            optimize_prune_retention_seconds=(cascade.optimize_prune_retention_seconds),
            optimize_rebuild_interval_seconds=(
                cascade.optimize_rebuild_interval_seconds
            ),
            scan_interval_seconds=cascade.scan_interval_seconds,
        )


class CascadeOrchestrator:
    """Composite owner of the cascade subsystem."""

    def __init__(
        self,
        *,
        memory_root: MemoryRoot,
        tokenizer: Tokenizer,
        config: CascadeConfig | None = None,
    ) -> None:
        self._memory_root = memory_root
        self._config = config or CascadeConfig.from_settings()
        deps = HandlerDeps(
            memory_root=memory_root,
            tokenizer=tokenizer,
        )
        self._handlers = build_handlers(deps)
        self._scanner = CascadeScanner(
            memory_root,
            scan_interval_seconds=self._config.scan_interval_seconds,
        )
        self._worker = CascadeWorker(
            self._handlers,
            batch_size=self._config.worker_batch_size,
            handler_concurrency=self._config.worker_handler_concurrency,
            max_retry=self._config.worker_max_retry,
            poll_interval_seconds=self._config.worker_poll_interval_seconds,
            retry_backoff_seconds=self._config.worker_retry_backoff_seconds,
            optimize_heartbeat_seconds=self._config.optimize_heartbeat_seconds,
            optimize_prune_interval_seconds=(
                self._config.optimize_prune_interval_seconds
            ),
            optimize_prune_retention_seconds=(
                self._config.optimize_prune_retention_seconds
            ),
            optimize_rebuild_interval_seconds=(
                self._config.optimize_rebuild_interval_seconds
            ),
        )
        self._watcher: CascadeWatcher | None = None
        self._started = False
        self._index_rows: dict[str, int] = {}
        self._index_emptied = False

    async def start(self) -> None:
        """Launch the watcher (sync thread) + scanner + worker tasks.

        Before launching, reset any stale ``processing`` rows back to
        ``pending``: cascade runs single-process today, so anything in
        ``processing`` at boot is leftover from a prior crash that
        ``claim_pending_batch`` can't re-claim on its own (the WHERE
        filter is ``status='pending'``).

        ``EVEROS_DISABLE_CASCADE_WATCHER`` skips the inotify half. The scanner
        already re-derives the same truth from the filesystem every
        ``scan_interval`` seconds, so the only thing lost is latency: an md edit
        is picked up within one scan instead of immediately. That trade is
        forced on a shared host, where inotify watches are a per-user kernel
        resource -- an IDE indexing the workspace can hold 350k of the 524k
        ceiling by itself, and every EverOS server then dies at startup with
        ``OSError: [Errno 28] inotify watch limit reached``. Unlike
        ``EVEROS_DISABLE_CASCADE`` this keeps the worker, so md still reaches
        LanceDB -- which an ingesting run cannot do without.
        """
        if self._started:
            return
        orphans = await md_change_state_repo.recover_orphan_processing()
        if orphans:
            logger.info("cascade_recovered_orphan_processing", count=orphans)
        await reconcile_index_once(self._memory_root)
        await self._snapshot_index_rows()
        if os.getenv("EVEROS_DISABLE_CASCADE_WATCHER", "").strip().lower() in (
            "1",
            "true",
            "yes",
        ):
            logger.info("cascade_watcher_disabled_by_env")
        else:
            loop = asyncio.get_running_loop()
            self._watcher = CascadeWatcher(self._memory_root, loop)
            self._watcher.start()
        await self._scanner.start()
        await self._worker.start()
        self._started = True
        logger.info("cascade_orchestrator_started")

    async def _snapshot_index_rows(self) -> None:
        """Count what the index holds, and say so when it holds nothing.

        Runs after :func:`reconcile_index_once` so a pass that re-armed
        itself gets to enqueue first -- otherwise this would report an
        emptiness the next drain is already fixing.

        The verdict needs all three facts: the ledger claims **live**
        sources were indexed, it claims nothing is waiting, and every table
        is empty. A fresh root has nothing done and a draining one has work
        pending, so neither trips it.

        "Live" is load-bearing. ``QueueSummary.done`` includes completed
        deletions, which the reconciler keeps on purpose, so a user who
        deleted their last markdown file would otherwise look identical to
        a wiped index -- reported forever, and pointed at a rebuild that
        could not add anything back.
        """
        try:
            self._index_rows = await count_business_rows()
        except Exception as exc:  # pragma: no cover - backend-specific
            # Counting is diagnostics; a backend that cannot answer must
            # not stop the cascade from starting.
            logger.warning("cascade_index_row_count_failed", error=repr(exc))
            return
        summary = await md_change_state_repo.queue_summary()
        live = await md_change_state_repo.count_done_live_sources()
        self._index_emptied = (
            live > 0 and summary.pending == 0 and not any(self._index_rows.values())
        )
        if self._index_emptied:
            logger.warning(
                "cascade_index_empty_but_ledger_done",
                done=live,
                rows=self._index_rows,
                hint=(
                    "the ledger says every file is indexed but every index "
                    "table is empty -- searches will return nothing. If the "
                    "index directory was removed, run `everos cascade "
                    "rebuild` to re-index from markdown."
                ),
            )

    async def stop(self) -> None:
        """Shut everything down in reverse order."""
        if not self._started:
            return
        await self._worker.stop()
        await self._scanner.stop()
        if self._watcher is not None:
            self._watcher.stop()
            self._watcher = None
        self._started = False
        logger.info("cascade_orchestrator_stopped")

    async def sync_once(self, *, kinds: set[str] | None = None) -> int:
        """One scan + drain cycle (used by CLI ``cascade sync`` and
        Phase-3 backfill's post-write skill-file sync).

        Args:
            kinds: Optional restriction on which md kinds the scan
                walks (e.g. ``{"agent_skill"}``). ``None`` — the default
                for CLI ``cascade sync`` — scans every registered kind.
                Phase 3 uses ``{"agent_skill"}`` so an unscoped sweep
                doesn't tag unrelated kinds (notably ``knowledge_*``,
                which the current process may not have handlers for)
                as permanently failed.

        Returns the number of rows processed in this drain; the worker
        bounds one drain, so the CLI keeps calling :meth:`drain_once` until
        a pass processes nothing.

        An unscoped pass is a full pass over the root, so it first runs
        :func:`reconcile_index_once`; a ``kinds``-scoped pass drains only
        those kinds and must not leave the rest queued behind it.

        The ``kinds`` filter is threaded through **both** the scanner
        AND the worker drain: the scanner scopes what gets enqueued,
        the worker scopes what gets claimed. Round-1 wired only the
        scanner side, so a scoped Phase-3 sync could still drain a
        knowledge md queued in a prior tick and mark it permanently
        failed — round-2 closes that end.
        """
        if kinds is None:
            await reconcile_index_once(self._memory_root)
        await self._scanner.scan_once(kinds=kinds)
        return await self._worker.drain_until_empty(kinds=kinds)

    async def force_enqueue_all(self) -> list[MdChangeState]:
        """``cascade sync --force-all``: the reconcile pass on demand.

        Settles the marker as well, so the :meth:`sync_once` that follows
        does not queue the whole ledger a second time.
        """
        return await reconcile_index_once(self._memory_root, force=True)

    async def drain_once(self) -> int:
        """Drain the queue exactly once without scanning first."""
        return await self._worker.drain_until_empty()

    async def drain_fully(self) -> int:
        """One scan, then drain until the ledger holds nothing pending.

        Two things end a single drain early. The worker bounds one
        ``drain_until_empty`` (``max_passes`` x batch size), and the ledger
        the one-time reconcile queues after an upgrade can exceed it. Inside
        a server the background worker claims batches too, so a pass of ours
        can find nothing pending while the worker still holds the last
        batch; ``POST /cascade/quiesce`` cancels the worker right after,
        which would orphan that batch in ``processing`` and freeze a partial
        index (seen on a real host: 50 rows left of 12000). So the ledger,
        not our own pass, says when the drain is done -- ``pending`` counts
        ``processing`` rows -- and a pass that finds nothing waits one poll
        interval for the other drainer. A ledger that stops moving for
        :data:`_DRAIN_IDLE_LIMIT_SECONDS` ends the wait; the caller reports
        the rows left.
        """
        processed = await self.sync_once()
        idle_seconds = 0.0
        while (await self.queue_summary()).pending:
            more = await self.drain_once()
            processed += more
            if more:
                idle_seconds = 0.0
                continue
            if idle_seconds >= _DRAIN_IDLE_LIMIT_SECONDS:
                logger.warning(
                    "cascade_drain_fully_gave_up",
                    idle_seconds=idle_seconds,
                    reason="rows stayed pending or processing with nothing landing",
                )
                break
            await asyncio.sleep(self._config.worker_poll_interval_seconds)
            idle_seconds += self._config.worker_poll_interval_seconds
        return processed

    async def queue_summary(self) -> QueueSummary:
        """Forward to the repo so callers don't reach past this class."""
        return await md_change_state_repo.queue_summary()

    async def health(self) -> CascadeHealth:
        """Verdict for ``/health``: operational health + informational counts.

        One query (:meth:`queue_summary`) plus the worker's in-memory
        counters. ``healthy`` is driven **only** by operational signals
        (:meth:`CascadeWorkerHealth.reasons` — drain / optimize / prune),
        never by ``failed_permanent``: a per-file triage backlog is normal
        steady state and must not pin the health signal red (see
        :class:`CascadeHealth`). ``failed_permanent`` is still reported as
        an informational count.
        """
        wh = self._worker.health()
        summary = await self.queue_summary()
        reasons = wh.reasons()  # operational only
        return CascadeHealth(
            healthy=not reasons,
            reasons=reasons,
            pending=summary.pending,
            failed_permanent=summary.failed_permanent,
            failed_retryable=summary.failed_retryable,
            drain_consecutive_failures=wh.drain_consecutive_failures,
            unrecoverable_total=wh.unrecoverable_total,
            optimize_failure_streak=wh.optimize_failure_streak,
            prune_stale_seconds=wh.prune_stale_seconds,
            index_rows=dict(self._index_rows),
            index_emptied=self._index_emptied,
        )

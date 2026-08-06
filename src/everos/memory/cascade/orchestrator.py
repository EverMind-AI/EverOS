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

from everos.component.tokenizer import Tokenizer
from everos.config import load_settings
from everos.core.observability.logging import get_logger
from everos.core.persistence import MemoryRoot
from everos.infra.persistence.sqlite import QueueSummary, md_change_state_repo

from .handlers import HandlerDeps
from .registry import build_handlers
from .scanner import CascadeScanner
from .watcher import CascadeWatcher
from .worker import (
    DEFAULT_OPTIMIZE_HEARTBEAT_SECONDS,
    DEFAULT_OPTIMIZE_PRUNE_INTERVAL_SECONDS,
    DEFAULT_OPTIMIZE_PRUNE_RETENTION_SECONDS,
    DEFAULT_OPTIMIZE_REBUILD_INTERVAL_SECONDS,
    CascadeWorker,
)

logger = get_logger(__name__)


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

    async def start(self) -> None:
        """Launch the watcher (sync thread) + scanner + worker tasks.

        Before launching, reset any stale ``processing`` rows back to
        ``pending``: cascade runs single-process today, so anything in
        ``processing`` at boot is leftover from a prior crash that
        ``claim_pending_batch`` can't re-claim on its own (the WHERE
        filter is ``status='pending'``).
        """
        if self._started:
            return
        orphans = await md_change_state_repo.recover_orphan_processing()
        if orphans:
            logger.info("cascade_recovered_orphan_processing", count=orphans)
        loop = asyncio.get_running_loop()
        self._watcher = CascadeWatcher(self._memory_root, loop)
        self._watcher.start()
        await self._scanner.start()
        await self._worker.start()
        self._started = True
        logger.info("cascade_orchestrator_started")

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

        Returns the number of rows processed in this drain. The CLI
        loops on the returned count to know when to stop.

        The ``kinds`` filter is threaded through **both** the scanner
        AND the worker drain: the scanner scopes what gets enqueued,
        the worker scopes what gets claimed. Round-1 wired only the
        scanner side, so a scoped Phase-3 sync could still drain a
        knowledge md queued in a prior tick and mark it permanently
        failed — round-2 closes that end.
        """
        await self._scanner.scan_once(kinds=kinds)
        return await self._worker.drain_until_empty(kinds=kinds)

    async def drain_once(self) -> int:
        """Drain the queue exactly once without scanning first."""
        return await self._worker.drain_until_empty()

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
        )

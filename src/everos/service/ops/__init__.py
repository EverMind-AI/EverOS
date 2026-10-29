"""OpsService -- the operator's use cases behind one facade.

Cascade (status / sync / fix / rebuild / quiesce / backfill), OME (manual
trigger), health signals and the ``async_mode`` task readers. Each method
reaches the object that does the work through what the application
bootstrap injected -- the cascade orchestrator and its storage admin, the
OME engine and the root's lock, the memorize inbox -- so neither the HTTP
routes nor the CLI construct or look up any of them.

Presentation stays with the caller: the CLI prints and picks exit codes,
the routes shape response DTOs. What remains here is "which object, which
call, in which order", plus the rules those surfaces used to carry (which
task statuses exist, what a failed health probe reads as, when a backfill
run stops).

Two lifecycles meet here. The server starts the cascade orchestrator through
:meth:`OpsService.start` (unless ``EVEROS_DISABLE_CASCADE`` is set) and the
health block reports it only while that is so. A CLI command never starts
it; it opens the storage itself with :meth:`OpsService.index_session` and
drives the orchestrator one pass at a time.

Backfill's three phases are private step modules of this package
(``_backfill_vectors`` / ``_backfill_clusters`` / ``_backfill_skills``);
:meth:`OpsService.backfill` runs them in order.
"""

from __future__ import annotations

__all__ = [
    "BACKFILL_PHASES",
    "BackfillDeps",
    "BackfillOutcome",
    "BackfillPresenter",
    "BackfillSummary",
    "Capabilities",
    "CascadeHealthReport",
    "CascadeStatus",
    "ClustersPhaseResult",
    "HealthReport",
    "NullBackfillPresenter",
    "OmeRun",
    "OmeTriggerResult",
    "OpsService",
    "ProviderStatus",
    "QuiesceResult",
    "RebuildResult",
    "SkillsPhaseResult",
    "VectorsPhaseResult",
    "compute_disabled_features",
    "format_tokens",
]

import contextlib
import dataclasses
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING

from everos.core.errors import InvalidInputError, NotFoundError
from everos.core.observability.logging import get_logger
from everos.memory.ports import SessionKey
from everos.service.memorize_queue import (
    TASK_STATUSES,
    QueueHealth,
    TaskItem,
    TaskPage,
    to_task_item,
)

from ._backfill import BACKFILL_LOGGER, PhaseResult
from ._backfill import BACKFILL_PHASES as BACKFILL_PHASES
from ._backfill import BackfillDeps as BackfillDeps
from ._backfill import BackfillOutcome as BackfillOutcome
from ._backfill import BackfillPresenter as BackfillPresenter
from ._backfill import BackfillSummary as BackfillSummary
from ._backfill import ClustersPhaseResult as ClustersPhaseResult
from ._backfill import NullBackfillPresenter as NullBackfillPresenter
from ._backfill import SkillsPhaseResult as SkillsPhaseResult
from ._backfill import VectorsPhaseResult as VectorsPhaseResult
from ._backfill import format_tokens as format_tokens
from ._backfill_clusters import run_clusters_phase
from ._backfill_skills import run_skills_phase
from ._backfill_vectors import check_vector_tables, run_vectors_phase
from ._health import (
    cascade_probe_failed,
    cascade_report,
    queue_probe_failed,
)
from ._health import compute_disabled_features as compute_disabled_features
from ._models import Capabilities as Capabilities
from ._models import CascadeHealthReport as CascadeHealthReport
from ._models import CascadeStatus as CascadeStatus
from ._models import HealthReport as HealthReport
from ._models import OmeRun as OmeRun
from ._models import OmeTriggerResult as OmeTriggerResult
from ._models import ProviderStatus as ProviderStatus
from ._models import QuiesceResult as QuiesceResult
from ._models import RebuildResult as RebuildResult

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from everos.config import Settings
    from everos.infra.cascade import CascadeAdmin, CascadeOrchestrator
    from everos.infra.ome import OfflineEngine, OmeLock
    from everos.infra.persistence import MemoryRoot
    from everos.infra.persistence.sqlite import MdChangeState
    from everos.memory.ports import MemorizeInboxStore

logger = get_logger(__name__)
backfill_logger = get_logger(BACKFILL_LOGGER)

_MAX_TASK_PAGE_SIZE = 100

_PHASE_RUNNERS: dict[str, Callable[..., Awaitable[PhaseResult]]] = {
    "vectors": run_vectors_phase,
    "clusters": run_clusters_phase,
    "skills": run_skills_phase,
}


async def _no_queue() -> QueueHealth | None:
    return None


class OpsService:
    """Cascade, OME, health and task operations over injected collaborators.

    Args:
        settings: Application settings (``[embedding]`` for the rebuild's
            attestation).
        root: The memory root.
        admin: System DB, index and cascade-ledger operations.
        cascade: The cascade orchestrator (built whether or not it runs).
        run_cascade: Whether :meth:`start` starts the orchestrator.
        engine: The OME engine manual triggers go to.
        ome_lock: The root's exclusive lock, which a running server holds.
        inbox: The memorize inbox the task readers read.
        backfill: What the backfill phases run against.
        capabilities: The providers the application was assembled with.
        queue_health: The memorize queue's health probe, reported by
            :meth:`health`; ``None`` while the queue is not running.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        root: MemoryRoot,
        admin: CascadeAdmin,
        cascade: CascadeOrchestrator,
        run_cascade: bool,
        engine: OfflineEngine,
        ome_lock: OmeLock,
        inbox: MemorizeInboxStore,
        backfill: BackfillDeps,
        capabilities: Capabilities,
        queue_health: Callable[[], Awaitable[QueueHealth | None]] = _no_queue,
    ) -> None:
        check_vector_tables(backfill.vector_tables)
        self._settings = settings
        self._root = root
        self._admin = admin
        self._cascade = cascade
        self._run_cascade = run_cascade
        self._engine = engine
        self._ome_lock = ome_lock
        self._inbox = inbox
        self._backfill = backfill
        self._capabilities = capabilities
        self._queue_health = queue_health
        self._cascade_attached = False

    @property
    def capabilities(self) -> Capabilities:
        return self._capabilities

    # ── server lifecycle ───────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the cascade subsystem unless it is switched off.

        A read-only retrieval server does not ingest markdown, so the
        watcher / scanner / worker are pure overhead -- and the periodic scan
        re-enqueues a large store's whole markdown set, which starves search
        on a dense store. ``EVEROS_DISABLE_CASCADE`` keeps it off.
        """
        if not self._run_cascade:
            logger.info("cascade_lifespan_disabled_by_env")
            return
        if self._capabilities.embedding:
            logger.info("cascade_startup_embed_available")
        else:
            logger.info(
                "cascade_startup_embed_unavailable",
                reason="embedding not configured; keyword-only mode",
            )
        await self._cascade.start()
        self._cascade_attached = True
        logger.info("cascade_lifespan_ready")

    async def aclose(self) -> None:
        if self._cascade_attached:
            await self._cascade.stop()
            self._cascade_attached = False

    # ── health ─────────────────────────────────────────────────────────────

    async def health(self) -> HealthReport:
        """Provider flags, disabled features, and the cascade / queue blocks.

        A probe that raises (a locked or full SQLite, say) is reported as an
        unhealthy block with the error as its reason instead of propagating:
        the health endpoint is a liveness signal, and turning it into a 500
        would get the process restarted, which fixes neither.
        """
        providers = ProviderStatus(
            llm=True,
            embed=self._admin.embedding_probed(),
            rerank=self._capabilities.rerank,
            multimodal_llm=self._capabilities.multimodal_llm,
            parser=self._capabilities.parser,
        )
        return HealthReport(
            providers=providers,
            # ``embed`` reports the startup probe, but ``disabled_features`` is
            # a contract about what the server will refuse, and every gate
            # (search, the embedding strategies, backfill) asks whether a
            # provider was built.
            disabled_features=compute_disabled_features(
                {
                    **dataclasses.asdict(providers),
                    "embed": self._capabilities.embedding,
                }
            ),
            cascade=await self._probe_cascade(),
            memorize_queue=await self._probe_queue(),
        )

    async def _probe_cascade(self) -> CascadeHealthReport | None:
        if not self._cascade_attached:
            return None
        try:
            return cascade_report(await self._cascade.health())
        except Exception as exc:
            logger.warning("cascade_health_probe_failed", error=repr(exc))
            return cascade_probe_failed(exc)

    async def _probe_queue(self) -> QueueHealth | None:
        try:
            return await self._queue_health()
        except Exception as exc:
            logger.warning("memorize_queue_health_probe_failed", error=repr(exc))
            return queue_probe_failed(exc)

    # ── tasks ──────────────────────────────────────────────────────────────

    async def get_task(self, request_id: str) -> TaskItem | None:
        """One accepted request's outcome; ``None`` once it aged out (or never was)."""
        row = await self._inbox.get_task(request_id)
        return None if row is None else to_task_item(row)

    async def list_tasks(
        self,
        *,
        session_id: str | None = None,
        status: str | None = None,
        page: int = 1,
        page_size: int = 20,
        app_id: str = "default",
        project_id: str = "default",
    ) -> TaskPage:
        """Accepted requests, newest first; ``page_size`` is capped at 100.

        Raises:
            InvalidInputError: ``status`` is not one of :data:`TASK_STATUSES`.
        """
        if status is not None and status not in TASK_STATUSES:
            raise InvalidInputError(
                f"status must be one of {', '.join(TASK_STATUSES)}; got {status!r}"
            )
        page_size = min(page_size, _MAX_TASK_PAGE_SIZE)
        session: SessionKey | None = None
        if session_id is not None:
            session = SessionKey(
                app_id=app_id, project_id=project_id, session_id=session_id
            )
        tasks = await self._inbox.list_tasks(
            session=session,
            status=status,
            offset=(page - 1) * page_size,
            limit=page_size,
        )
        return TaskPage(
            total=tasks.total,
            page=page,
            page_size=page_size,
            items=[to_task_item(t) for t in tasks.items],
        )

    # ── OME ────────────────────────────────────────────────────────────────

    async def ome_trigger(
        self, name: str, *, force: bool, wait_seconds: float
    ) -> OmeTriggerResult:
        """Trigger a registered strategy and wait (bounded) for its runs.

        Raises:
            NotFoundError: No strategy is registered under ``name``.
        """
        try:
            event, routes = await self._engine.trigger_manual(name, force=force)
        except KeyError:
            raise NotFoundError(f"strategy '{name}' not found") from None

        if not routes:
            logger.info("ome_trigger_manual_not_dispatched", strategy=name)
            return OmeTriggerResult(status="not_dispatched", dispatched=0, runs=[])

        logger.info("ome_trigger_manual", strategy=name, dispatched=len(routes))
        idle = await self._engine.wait_idle(timeout=wait_seconds)
        records = await self._engine.list_runs_by_event_id(event.event_id)
        runs = [
            OmeRun(run_id=r.run_id, status=r.status.value, error=r.error)
            for r in records
        ]
        if not idle:
            logger.warning("ome_trigger_timeout", strategy=name, timeout=wait_seconds)
            return OmeTriggerResult(status="timeout", dispatched=len(routes), runs=runs)
        return OmeTriggerResult(status="ok", dispatched=len(routes), runs=runs)

    # ── cascade, server side ───────────────────────────────────────────────

    async def cascade_quiesce(self) -> QuiesceResult | None:
        """Drain the projection queue, then stop watcher + scanner + worker.

        Drain BEFORE stopping: ``drain_fully`` is a full scan + drain to
        empty, so it also picks up markdown the watcher never saw, and it
        does not stop at the worker's per-drain bound. Stopping first would
        freeze a partial index.

        Returns:
            ``None`` when the server is not running the cascade.
        """
        if not self._cascade_attached:
            return None
        before = await self._cascade.queue_summary()
        drained = await self._cascade.drain_fully()
        await self._cascade.stop()
        after = await self._cascade.queue_summary()
        logger.info(
            "cascade_quiesced",
            drained=drained,
            pending_before=before.pending,
            pending_after=after.pending,
            failed_permanent=after.failed_permanent,
        )
        if after.pending:
            logger.warning(
                "cascade_quiesce_left_pending",
                pending=after.pending,
                reason="index is not a complete projection of the markdown",
            )
        return QuiesceResult(
            drained=drained,
            pending_before=before.pending,
            pending_after=after.pending,
            failed_permanent=after.failed_permanent,
        )

    # ── cascade, CLI side ──────────────────────────────────────────────────

    def ome_lock_is_free(self) -> bool:
        """Whether no other process (a server) holds this root's OME lock.

        A probe, not a reservation: it guards against the common mistake of
        running a destructive command next to a server, not against a race.
        """
        return self._ome_lock.is_free()

    @contextlib.asynccontextmanager
    async def index_session(
        self, *, verify: bool = True, ensure: bool = True, exclusive: bool = False
    ) -> AsyncIterator[None]:
        """Open the system DB and the index for one CLI command; close after.

        ``exclusive`` holds the OME lock for the whole session: a second
        process upserting the same index cannot see what the other just
        committed, so every command that writes the index holds it.
        ``verify=False`` skips the schema / provenance checks (``rebuild``
        repairs exactly what they reject); ``ensure=False`` skips the index
        migrations, which raise on the damage ``rebuild`` exists to repair.

        Raises:
            MemoryRootLockedError: ``exclusive`` and another process holds
                the lock.
            IndexSchemaMismatchError: A stored table drifted from the schema.
            EmbeddingProvenanceError: The configured embedding cannot have
                produced the stored vectors.
        """
        lock = self._ome_lock.hold() if exclusive else contextlib.nullcontext()
        with lock:
            await self._admin.create_schema()
            try:
                # Attest only while holding the index lock -- read-only
                # phases must not write the provenance record.
                await self._admin.open_index(
                    verify=verify, ensure=ensure, may_attest=exclusive
                )
                yield
            finally:
                await self._admin.close_index()
                await self._admin.dispose()

    def relative_to_root(self, path: Path) -> str:
        """``path`` relative to the memory root, as the ledger stores it.

        Raises:
            ValueError: ``path`` is outside the memory root.
        """
        absolute = path.expanduser().resolve()
        try:
            rel = absolute.relative_to(self._root.root)
        except ValueError as exc:
            raise ValueError(
                f"path {path!s} is not under memory root {self._root.root!s}"
            ) from exc
        return rel.as_posix()

    async def cascade_status(self) -> CascadeStatus:
        """The ledger summary, live index row counts and indexed live sources."""
        return CascadeStatus(
            summary=await self._admin.queue_summary(),
            index_rows=await self._admin.count_index_rows(),
            live_sources=await self._admin.count_live_sources(),
        )

    async def cascade_force_enqueue_all(self) -> list[MdChangeState]:
        self._prepare_cascade()
        return await self._cascade.force_enqueue_all()

    async def cascade_force_enqueue(self, rel_md_path: str) -> str | None:
        """Queue one markdown file; return its kind, ``None`` when none matches."""
        return await self._admin.force_enqueue(rel_md_path)

    async def cascade_sync(self) -> int:
        """Scan, then drain until the ledger holds nothing pending."""
        self._prepare_cascade()
        return await self._cascade.drain_fully()

    async def cascade_failed(self) -> list[MdChangeState]:
        return await self._admin.list_failed()

    async def cascade_fix(self) -> tuple[int, int]:
        """Re-queue retryable failures and drain once.

        Returns:
            ``(re-queued, processed)``.
        """
        moved = await self._admin.reset_retryable()
        if not moved:
            return 0, 0
        self._prepare_cascade()
        return moved, await self._cascade.drain_once()

    async def cascade_rebuild(self) -> RebuildResult:
        """Reset the ledger, drop and recreate the index, re-index every md.

        The ledger is reset FIRST so every crash window converges on "queue
        pending -> next scan re-indexes"; dropping first leaves a window in
        which a crash yields empty tables under a fully-``done`` ledger.
        """
        cleared = await self._admin.reset_ledger()
        dropped = await self._admin.drop_index_tables()
        await self._admin.ensure_index()
        self._prepare_cascade()
        processed = await self._cascade.drain_fully()
        # Attest the model that just produced every vector: a table created
        # from a pydantic model carries no field metadata, so the rebuilt
        # index would otherwise come up unattested.
        if self._capabilities.embedding:
            await self._admin.attest_embedding_model(
                self._settings.embedding.model or ""
            )
        return RebuildResult(
            ledger_rows_reset=cleared, dropped_tables=dropped, processed=processed
        )

    # ── backfill ───────────────────────────────────────────────────────────

    async def backfill(
        self,
        phase: str,
        *,
        auto_yes: bool,
        presenter: BackfillPresenter,
        summary: BackfillSummary,
    ) -> BackfillOutcome:
        """Run ``phase`` (or ``"all"``: every phase, in order) and say how it ended.

        A phase blocked by a missing provider or a running server, or one
        the user declines, stops the run: declining the first phase's
        estimate is a "no" to the phases whose estimates were never shown.
        Each phase's result lands in ``summary`` as soon as it finishes, so a
        caller interrupted mid-run still has every phase that completed.

        Raises:
            ValueError: ``phase`` is neither ``"all"`` nor a phase name.
        """
        if phase != "all" and phase not in BACKFILL_PHASES:
            raise ValueError(f"unknown backfill phase: {phase!r}")
        selected = BACKFILL_PHASES if phase == "all" else (phase,)
        for name in selected:
            presenter.phase_started(name)
            result = await _PHASE_RUNNERS[name](
                self._backfill, auto_yes=auto_yes, presenter=presenter
            )
            summary.record(name, result)
            stop = _stopping_outcome(result)
            if stop is not None:
                _log_stop(name, stop, result)
                return stop
        if summary.rows_failed:
            backfill_logger.warning(
                "cascade_backfill_completed_with_failures",
                total_failed=summary.rows_failed,
            )
            return BackfillOutcome.COMPLETED_WITH_FAILURES
        return BackfillOutcome.COMPLETED

    def _prepare_cascade(self) -> None:
        self._root.ensure()
        if self._capabilities.embedding:
            logger.info("cli_cascade_embed_available")
        else:
            logger.info(
                "cli_cascade_embed_unavailable",
                reason="embedding not configured; keyword-only mode",
            )


def _stopping_outcome(result: PhaseResult) -> BackfillOutcome | None:
    """How a phase result ends the run, or ``None`` when the next phase runs.

    A missing provider wins over a running server: the first is a permanent
    configuration gap, the second goes away when the server stops.
    """
    if result.blocked_by_capability:
        return BackfillOutcome.BLOCKED_BY_CAPABILITY
    if result.blocked_by_server:
        return BackfillOutcome.BLOCKED_BY_SERVER
    if result.aborted:
        return BackfillOutcome.ABORTED
    return None


def _log_stop(phase: str, outcome: BackfillOutcome, result: PhaseResult) -> None:
    if outcome is BackfillOutcome.BLOCKED_BY_CAPABILITY:
        backfill_logger.warning(
            "cascade_backfill_blocked_by_capability",
            phase=phase,
            provider=result.blocked_by_capability,
        )
    elif outcome is BackfillOutcome.BLOCKED_BY_SERVER:
        backfill_logger.warning("cascade_backfill_blocked_by_server", phase=phase)
    else:
        backfill_logger.info("cascade_backfill_aborted", phase=phase)

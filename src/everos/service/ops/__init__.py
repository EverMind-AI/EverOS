"""OpsService -- the operator's use cases behind one facade.

Cascade (status / sync / fix / rebuild / quiesce / backfill), OME (manual
trigger), health signals and the ``async_mode`` task readers. Each method
reaches the object that does the work through what the application
bootstrap injected -- the cascade orchestrator and its storage admin, the
OME engine and the root's lock, the memorize inbox -- so neither the HTTP
routes nor the CLI construct or look up any of them.

Presentation stays with the caller: the CLI prints and picks exit codes,
the routes shape response DTOs. Every method delegates to a private step
module -- ``_cascade`` (the server starts the orchestrator, a CLI command
drives it one pass at a time), ``_health``, ``_ome``, ``_tasks`` and
``_backfill_run`` (backfill's three phases, in order) -- which holds the
rules those surfaces used to carry.
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

from typing import TYPE_CHECKING

from . import _cascade, _ome, _tasks
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
from ._backfill_run import run_backfill
from ._backfill_vectors import check_vector_tables
from ._health import QueueHealthProbe, health_report, no_queue
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
    from contextlib import AbstractAsyncContextManager
    from pathlib import Path

    from everos.config import Settings
    from everos.infra.cascade import CascadeAdmin, CascadeOrchestrator
    from everos.infra.ome import OfflineEngine, OmeLock
    from everos.infra.persistence import MemoryRoot
    from everos.infra.persistence.sqlite import MdChangeState
    from everos.memory.ports import MemorizeInboxStore
    from everos.service.memory import TaskItem, TaskPage


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
        queue_health: QueueHealthProbe = no_queue,
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
        """Start the cascade subsystem unless ``EVEROS_DISABLE_CASCADE`` is set."""
        self._cascade_attached = await _cascade.start(
            self._cascade,
            enabled=self._run_cascade,
            embedding=self._capabilities.embedding,
        )

    async def aclose(self) -> None:
        if self._cascade_attached:
            await self._cascade.stop()
            self._cascade_attached = False

    # ── health ─────────────────────────────────────────────────────────────

    async def health(self) -> HealthReport:
        """Provider flags, disabled features, and the cascade / queue blocks."""
        return await health_report(
            admin=self._admin,
            capabilities=self._capabilities,
            cascade=self._cascade if self._cascade_attached else None,
            queue_health=self._queue_health,
        )

    # ── tasks ──────────────────────────────────────────────────────────────

    async def get_task(self, request_id: str) -> TaskItem | None:
        """One accepted request's outcome; ``None`` once it aged out (or never was)."""
        return await _tasks.get_task(self._inbox, request_id)

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
            InvalidInputError: ``status`` is not a task status.
        """
        return await _tasks.list_tasks(
            self._inbox,
            session_id=session_id,
            status=status,
            page=page,
            page_size=page_size,
            app_id=app_id,
            project_id=project_id,
        )

    # ── OME ────────────────────────────────────────────────────────────────

    async def ome_trigger(
        self, name: str, *, force: bool, wait_seconds: float
    ) -> OmeTriggerResult:
        """Trigger a strategy and wait (bounded); ``NotFoundError`` if unknown."""
        return await _ome.trigger(
            self._engine, name, force=force, wait_seconds=wait_seconds
        )

    def ome_lock_is_free(self) -> bool:
        """Whether no server holds this root's OME lock (a probe, not a lock)."""
        return self._ome_lock.is_free()

    # ── cascade, server side ───────────────────────────────────────────────

    async def cascade_quiesce(self) -> QuiesceResult | None:
        """Drain the projection queue, then stop; ``None`` if the cascade is off."""
        if not self._cascade_attached:
            return None
        return await _cascade.quiesce(self._cascade)

    # ── cascade, CLI side ──────────────────────────────────────────────────

    def index_session(
        self, *, verify: bool = True, ensure: bool = True, exclusive: bool = False
    ) -> AbstractAsyncContextManager[None]:
        """Open the system DB and the index for one CLI command; close after."""
        return _cascade.index_session(
            self._admin,
            self._ome_lock,
            verify=verify,
            ensure=ensure,
            exclusive=exclusive,
        )

    def relative_to_root(self, path: Path) -> str:
        """``path`` relative to the memory root, as the ledger stores it.

        Raises:
            ValueError: ``path`` is outside the memory root.
        """
        return _cascade.relative_to_root(self._root, path)

    async def cascade_status(self) -> CascadeStatus:
        """The ledger summary, live index row counts and indexed live sources."""
        return await _cascade.status(self._admin)

    async def cascade_force_enqueue_all(self) -> list[MdChangeState]:
        return await _cascade.force_enqueue_all(
            self._cascade, root=self._root, embedding=self._capabilities.embedding
        )

    async def cascade_force_enqueue(self, rel_md_path: str) -> str | None:
        """Queue one markdown file; return its kind, ``None`` when none matches."""
        return await self._admin.force_enqueue(rel_md_path)

    async def cascade_sync(self) -> int:
        """Scan, then drain until the ledger holds nothing pending."""
        return await _cascade.sync(
            self._cascade, root=self._root, embedding=self._capabilities.embedding
        )

    async def cascade_failed(self) -> list[MdChangeState]:
        return await self._admin.list_failed()

    async def cascade_fix(self) -> tuple[int, int]:
        """Re-queue retryable failures and drain once; ``(re-queued, processed)``."""
        return await _cascade.fix(
            self._admin,
            self._cascade,
            root=self._root,
            embedding=self._capabilities.embedding,
        )

    async def cascade_rebuild(self) -> RebuildResult:
        """Reset the ledger, drop and recreate the index, re-index every md."""
        return await _cascade.rebuild(
            self._admin,
            self._cascade,
            root=self._root,
            embedding=self._capabilities.embedding,
            embedding_model=self._settings.embedding.model or "",
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
        """Run ``phase`` (or ``"all"``) in order; stop at the first that blocks.

        Raises:
            ValueError: ``phase`` is neither ``"all"`` nor a phase name.
        """
        return await run_backfill(
            self._backfill,
            phase,
            auto_yes=auto_yes,
            presenter=presenter,
            summary=summary,
        )

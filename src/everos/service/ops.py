"""OpsService -- the operator's use cases behind one facade.

Cascade (status / sync / fix / rebuild / quiesce / backfill), OME (manual
trigger), health signals and the ``async_mode`` task readers. Each method
reaches the object that does the work through what the application
bootstrap injected -- the cascade orchestrator and its storage admin, the
OME engine, the memorize inbox -- so neither the HTTP routes nor the CLI
construct or look up any of them.

Presentation stays with the caller: the CLI prints and picks exit codes,
the routes shape response DTOs. What remains here is "which object, which
call, in which order".

Two lifecycles meet here. The server starts the cascade orchestrator through
:meth:`OpsService.start` (unless ``EVEROS_DISABLE_CASCADE`` is set) and the
health block reports it only while that is so. A CLI command never starts
it; it opens the storage itself with :meth:`OpsService.index_session` and
drives the orchestrator one pass at a time.
"""

from __future__ import annotations

__all__ = [
    "Capabilities",
    "CascadeStatus",
    "OmeRun",
    "OmeTriggerResult",
    "OpsService",
    "QuiesceResult",
    "RebuildResult",
]

import contextlib
import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING

from everos.core.errors import NotFoundError
from everos.core.observability.logging import get_logger
from everos.memory.ports import SessionKey
from everos.service.backfill import (
    BackfillContext,
    BackfillPresenter,
    PhaseResult,
    hold_ome_lock,
    ome_lock_is_free,
    run_phase,
)
from everos.service.memorize_queue import TaskItem, TaskPage, to_task_item

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from everos.config import Settings
    from everos.infra.cascade import CascadeAdmin, CascadeHealth, CascadeOrchestrator
    from everos.infra.ome import OfflineEngine
    from everos.infra.persistence import MemoryRoot
    from everos.infra.persistence.sqlite import MdChangeState, QueueSummary
    from everos.memory.ports import MemorizeInboxStore

logger = get_logger(__name__)


@dataclasses.dataclass(frozen=True)
class Capabilities:
    """Which optional providers the application was assembled with.

    A snapshot taken by :func:`everos.service.build_application`; a provider
    configured later needs a restart, as before.
    """

    llm: bool
    decider: bool
    embedding: bool
    rerank: bool
    multimodal_llm: bool
    parser: bool
    index_backend: str


@dataclasses.dataclass(frozen=True)
class CascadeStatus:
    """``everos cascade status``: the ledger, the index, and the live sources."""

    summary: QueueSummary
    index_rows: dict[str, int]
    live_sources: int


@dataclasses.dataclass(frozen=True)
class QuiesceResult:
    """What the final drain before stopping the cascade accomplished."""

    drained: int
    pending_before: int
    pending_after: int
    failed_permanent: int


@dataclasses.dataclass(frozen=True)
class RebuildResult:
    """What ``everos cascade rebuild`` reset, dropped and re-indexed."""

    ledger_rows_reset: int
    dropped_tables: list[str]
    processed: int


@dataclasses.dataclass(frozen=True)
class OmeRun:
    """One strategy run attempt within a manual trigger."""

    run_id: str
    status: str
    error: str | None


@dataclasses.dataclass(frozen=True)
class OmeTriggerResult:
    """Outcome of a manual OME trigger: ``ok`` / ``timeout`` / ``not_dispatched``."""

    status: str
    dispatched: int
    runs: list[OmeRun]


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
        inbox: The memorize inbox the task readers read.
        backfill: What the backfill phase runners run against.
        capabilities: The providers the application was assembled with.
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
        inbox: MemorizeInboxStore,
        backfill: BackfillContext,
        capabilities: Capabilities,
    ) -> None:
        self._settings = settings
        self._root = root
        self._admin = admin
        self._cascade = cascade
        self._run_cascade = run_cascade
        self._engine = engine
        self._inbox = inbox
        self._backfill = backfill
        self._capabilities = capabilities
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

    def embedding_probed(self) -> bool:
        """Whether the embedding service answered the startup dimension probe."""
        return self._admin.embedding_probed()

    async def cascade_health(self) -> CascadeHealth | None:
        """The cascade verdict; ``None`` while the server is not running it."""
        if not self._cascade_attached:
            return None
        return await self._cascade.health()

    # ── tasks ──────────────────────────────────────────────────────────────

    async def get_task(self, request_id: str) -> TaskItem | None:
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
        """Whether no other process (a server) holds this root's OME lock."""
        return ome_lock_is_free(self._root.ome_db)

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
            EngineLockHeldError: ``exclusive`` and another process holds the lock.
            IndexSchemaMismatchError: A stored table drifted from the schema.
            EmbeddingProvenanceError: The configured embedding cannot have
                produced the stored vectors.
        """
        lock = (
            hold_ome_lock(self._root.ome_db) if exclusive else contextlib.nullcontext()
        )
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

    async def backfill(
        self, phase: str, *, auto_yes: bool, presenter: BackfillPresenter
    ) -> PhaseResult:
        """Run one backfill phase (``vectors`` / ``clusters`` / ``skills``)."""
        return await run_phase(
            self._backfill, phase, auto_yes=auto_yes, presenter=presenter
        )

    def _prepare_cascade(self) -> None:
        self._root.ensure()
        if self._capabilities.embedding:
            logger.info("cli_cascade_embed_available")
        else:
            logger.info(
                "cli_cascade_embed_unavailable",
                reason="embedding not configured; keyword-only mode",
            )

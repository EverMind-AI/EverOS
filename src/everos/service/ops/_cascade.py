"""The cascade use cases: the server's lifecycle and the CLI's commands.

The server starts the orchestrator (:func:`start`) and drains it before
stopping (:func:`quiesce`). A CLI command never starts it: it opens the
storage itself (:func:`index_session`) and drives the orchestrator one pass
at a time (:func:`sync`, :func:`fix`, :func:`rebuild`).
"""

from __future__ import annotations

__all__ = [
    "fix",
    "force_enqueue_all",
    "index_session",
    "quiesce",
    "rebuild",
    "relative_to_root",
    "start",
    "status",
    "sync",
]

import contextlib
from typing import TYPE_CHECKING

from everos.core.observability.logging import get_logger

from ._models import CascadeStatus, QuiesceResult, RebuildResult

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from everos.infra.cascade import CascadeAdmin, CascadeOrchestrator
    from everos.infra.ome import OmeLock
    from everos.infra.persistence import MemoryRoot
    from everos.infra.persistence.sqlite import MdChangeState

logger = get_logger("everos.service.ops")


# ── server side ────────────────────────────────────────────────────────────


async def start(
    cascade: CascadeOrchestrator, *, enabled: bool, embedding: bool
) -> bool:
    """Start the cascade subsystem unless it is switched off; ``True`` if started.

    A read-only retrieval server does not ingest markdown, so the
    watcher / scanner / worker are pure overhead -- and the periodic scan
    re-enqueues a large store's whole markdown set, which starves search
    on a dense store. ``[cascade] enabled = false`` keeps it off.
    """
    if not enabled:
        logger.info("cascade_lifespan_disabled")
        return False
    if embedding:
        logger.info("cascade_startup_embed_available")
    else:
        logger.info(
            "cascade_startup_embed_unavailable",
            reason="embedding not configured; keyword-only mode",
        )
    await cascade.start()
    logger.info("cascade_lifespan_ready")
    return True


async def quiesce(cascade: CascadeOrchestrator) -> QuiesceResult:
    """Drain the projection queue, then stop watcher + scanner + worker.

    Drain BEFORE stopping: ``drain_fully`` is a full scan + drain to
    empty, so it also picks up markdown the watcher never saw, and it
    does not stop at the worker's per-drain bound. Stopping first would
    freeze a partial index.
    """
    before = await cascade.queue_summary()
    drained = await cascade.drain_fully()
    await cascade.stop()
    after = await cascade.queue_summary()
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


# ── CLI side ───────────────────────────────────────────────────────────────


@contextlib.asynccontextmanager
async def index_session(
    admin: CascadeAdmin,
    ome_lock: OmeLock,
    *,
    verify: bool,
    ensure: bool,
    exclusive: bool,
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
    lock = ome_lock.hold() if exclusive else contextlib.nullcontext()
    with lock:
        await admin.create_schema()
        try:
            # Attest only while holding the index lock -- read-only
            # phases must not write the provenance record.
            await admin.open_index(verify=verify, ensure=ensure, may_attest=exclusive)
            yield
        finally:
            await admin.close_index()
            await admin.dispose()


def relative_to_root(root: MemoryRoot, path: Path) -> str:
    """``path`` relative to the memory root, as the ledger stores it.

    Raises:
        ValueError: ``path`` is outside the memory root.
    """
    absolute = path.expanduser().resolve()
    try:
        rel = absolute.relative_to(root.root)
    except ValueError as exc:
        raise ValueError(
            f"path {path!s} is not under memory root {root.root!s}"
        ) from exc
    return rel.as_posix()


async def status(admin: CascadeAdmin) -> CascadeStatus:
    """The ledger summary, live index row counts and indexed live sources."""
    return CascadeStatus(
        summary=await admin.queue_summary(),
        index_rows=await admin.count_index_rows(),
        live_sources=await admin.count_live_sources(),
    )


async def force_enqueue_all(
    cascade: CascadeOrchestrator, *, root: MemoryRoot, embedding: bool
) -> list[MdChangeState]:
    """Queue every markdown file under the root."""
    _prepare(root, embedding=embedding)
    return await cascade.force_enqueue_all()


async def sync(
    cascade: CascadeOrchestrator, *, root: MemoryRoot, embedding: bool
) -> int:
    """Scan, then drain until the ledger holds nothing pending."""
    _prepare(root, embedding=embedding)
    return await cascade.drain_fully()


async def fix(
    admin: CascadeAdmin,
    cascade: CascadeOrchestrator,
    *,
    root: MemoryRoot,
    embedding: bool,
) -> tuple[int, int]:
    """Re-queue retryable failures and drain once.

    Returns:
        ``(re-queued, processed)``.
    """
    moved = await admin.reset_retryable()
    if not moved:
        return 0, 0
    _prepare(root, embedding=embedding)
    return moved, await cascade.drain_once()


async def rebuild(
    admin: CascadeAdmin,
    cascade: CascadeOrchestrator,
    *,
    root: MemoryRoot,
    embedding: bool,
    embedding_model: str,
) -> RebuildResult:
    """Reset the ledger, drop and recreate the index, re-index every md.

    The ledger is reset FIRST so every crash window converges on "queue
    pending -> next scan re-indexes"; dropping first leaves a window in
    which a crash yields empty tables under a fully-``done`` ledger.
    """
    cleared = await admin.reset_ledger()
    dropped = await admin.drop_index_tables()
    await admin.ensure_index()
    _prepare(root, embedding=embedding)
    processed = await cascade.drain_fully()
    # Attest the model that just produced every vector: a table created
    # from a pydantic model carries no field metadata, so the rebuilt
    # index would otherwise come up unattested.
    if embedding:
        await admin.attest_embedding_model(embedding_model)
    return RebuildResult(
        ledger_rows_reset=cleared, dropped_tables=dropped, processed=processed
    )


def _prepare(root: MemoryRoot, *, embedding: bool) -> None:
    root.ensure()
    if embedding:
        logger.info("cli_cascade_embed_available")
    else:
        logger.info(
            "cli_cascade_embed_unavailable",
            reason="embedding not configured; keyword-only mode",
        )

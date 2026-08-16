"""LanceDB lifespan provider (HTTP API entrypoint).

Startup:
    Acquire the shared projection lock, then serialize marker, connection,
    schema, and index bootstrap under the exclusive bootstrap lock.
    Importing :mod:`everos.infra.persistence.lancedb` also triggers the
    side-effect import of ``tables`` so business schemas are loaded
    (future: preflight registration).
    Log hint if unbackfilled (vector IS NULL) rows exist.

Shutdown:
    Close the connection (also clears the table cache), then release the
    shared projection lock.

Unbackfilled hint:
    The informational "you have unbackfilled memory rows" banner runs
    an unconditional ``count_rows(filter='vector IS NULL')`` against
    every business table on startup. An earlier "marker + limit(1)
    probe" amortisation was reverted (round-3 finding #3): the vector
    column has no scalar index, so ``limit(1)`` on ``vector IS NULL``
    costs the same full scan as ``count_rows``. On a clean state the
    probe scanned the entire empty tail before returning, matching the
    cost it was meant to avoid; on a dirty state the probe hit early
    and then the full ``count_rows`` ran anyway, doubling the scan.
    The marker's ``last_seen_count`` field was written but never read.
"""

from __future__ import annotations

from contextlib import AbstractAsyncContextManager
from typing import Any

import anyio
from fastapi import FastAPI

from everos.core.lifespan import LifespanProvider
from everos.core.observability.logging import get_logger
from everos.core.persistence import MemoryRoot
from everos.infra.persistence.lancedb import (
    BUSINESS_SCHEMAS_WITH_VECTOR,
    dispose_connection,
    ensure_business_indexes,
    get_connection,
    get_table,
    projection_bootstrap_lock,
    projection_server_lock,
    verify_business_schemas,
    verify_storage_identity_ready,
)

logger = get_logger(__name__)


async def _dispose_connection_before_unlock(
    *, preserve_active_exception: bool = False
) -> None:
    """Finish idempotent LanceDB cleanup before lifecycle locks can release.

    The shield prevents an outer shutdown cancellation from interrupting
    cleanup. A second attempt keeps a transient or explicitly injected first
    failure inside the same lock boundary instead of retrying after bootstrap
    exclusion has already been released.
    """
    with anyio.CancelScope(shield=True):
        first_error: BaseException | None = None
        for attempt in range(2):
            try:
                await dispose_connection()
                return
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
                logger.exception(
                    "lancedb_dispose_retry",
                    attempt=attempt + 1,
                )
        if first_error is not None and not preserve_active_exception:
            raise first_error


async def _log_unbackfilled_hint() -> None:
    """Warn at startup if there are unbackfilled memory rows.

    Runs an unconditional ``count_rows(filter="vector IS NULL")`` per
    business table. The vector column has no scalar index, so
    ``count_rows`` and any ``limit(1)`` probe cost the same full scan
    — a previous "marker + probe" optimisation (removed here) turned
    out to be net-zero on clean state and net-negative on dirty state
    (probe hits early, then the full count runs anyway = twice the
    scan).

    Per-table failures are logged as warnings and don't interrupt
    startup.
    """
    total_null = 0
    for schema in BUSINESS_SCHEMAS_WITH_VECTOR:
        try:
            table = await get_table(schema.TABLE_NAME, schema)
            count = await table.count_rows(filter="vector IS NULL")
        except Exception as exc:
            logger.warning(
                "unbackfilled_check_failed",
                schema=schema.__name__,
                error=repr(exc),
            )
            continue
        if count > 0:
            total_null += count

    if total_null > 0:
        banner_logger = get_logger("everos.cli.server")
        banner_logger.warning(
            "unbackfilled_memory_rows",
            count=total_null,
            hint="Run `everos cascade backfill` to include them in "
            "vector/hybrid search (optional).",
        )


class LanceDBLifespanProvider(LifespanProvider):
    """Manage the LanceDB connection + table cache for the app lifecycle.

    Startup runs seven steps:

    1. Acquire and retain the shared projection lock.
    2. Acquire the exclusive projection-bootstrap lock while retaining the
       shared lock. This serializes fresh multi-process initialization.
    3. Require the current storage-identity generation. A missing marker is
       initialized only when source markdown and LanceDB artifacts are absent;
       an existing projection must be rebuilt.
    4. ``get_connection`` — lazy-open the async connection.
    5. ``verify_business_schemas`` — fail loud if an on-disk table's
       columns drift from the current Pydantic schema. LanceDB has no
       online migration; cascade is rebuildable from md so the recovery
       is ``everos cascade rebuild`` (see ``docs/cascade_runbook.md``).
    6. ``ensure_business_indexes`` — idempotent FTS index creation, then
       release the bootstrap lock.
    7. ``_log_unbackfilled_hint`` — warn if unbackfilled rows exist.
    """

    def __init__(self, order: int = 11) -> None:
        super().__init__(name="lancedb", order=order)
        self._projection_lock: AbstractAsyncContextManager[None] | None = None

    async def startup(self, app: FastAPI) -> Any:
        # Retain the shared lock from before marker verification until after
        # shutdown has disposed every cached table handle. An offline rebuild
        # owns the exclusive side, so the marker cannot change between this
        # verification and later server writes.
        memory_root = MemoryRoot.resolve()
        lock = projection_server_lock(memory_root)
        await lock.__aenter__()
        cleanup_completed = False
        try:
            # All runtimes acquire locks in one order: projection SH, then
            # bootstrap EX. Rebuild acquires projection EX and therefore never
            # overlaps this block. The bootstrap lock is released before the
            # steady-state lifespan so multiple server processes can coexist.
            async with projection_bootstrap_lock(memory_root):
                try:
                    # This gate must run before opening LanceDB or running any
                    # migration. Otherwise startup itself could mutate a legacy
                    # projection before proving the row-id generation current.
                    await verify_storage_identity_ready()
                    conn = await get_connection()
                    await verify_business_schemas()
                    await ensure_business_indexes()
                except BaseException:
                    # Keep partial-bootstrap cleanup serialized. A waiting
                    # process must not enter while this process still owns
                    # half-initialized connection or table handles.
                    cleanup_completed = True
                    await _dispose_connection_before_unlock(
                        preserve_active_exception=True
                    )
                    raise
            await _log_unbackfilled_hint()
        except BaseException as exc:
            # Cleanup normally completed while bootstrap exclusion was still
            # held. If failure happened outside that block, finish cleanup
            # while retaining the shared projection lock.
            if not cleanup_completed:
                await _dispose_connection_before_unlock(preserve_active_exception=True)
            await lock.__aexit__(type(exc), exc, exc.__traceback__)
            raise
        self._projection_lock = lock
        logger.info("lancedb_ready", uri=conn.uri)
        return conn

    async def shutdown(self, app: FastAPI) -> None:
        lock = self._projection_lock
        try:
            # Shutdown cancellation must not release projection exclusion
            # while the process still owns cached LanceDB handles.
            await _dispose_connection_before_unlock()
        finally:
            self._projection_lock = None
            if lock is not None:
                await lock.__aexit__(None, None, None)

"""LanceDB lifespan provider (HTTP API entrypoint).

Startup:
    Acquire the shared projection lock, then open the connection via
    ``get_connection`` (lazy, idempotent).
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

from contextlib import AbstractAsyncContextManager, suppress
from typing import Any

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
    projection_server_lock,
    verify_business_schemas,
    verify_storage_identity_ready,
)

logger = get_logger(__name__)


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

    Startup runs six steps:

    1. Acquire and retain the shared projection lock.
    2. Require the current storage-identity generation. A missing marker is
       initialized only for a provably empty memory root; existing data must
       be rebuilt.
    3. ``get_connection`` — lazy-open the async connection.
    4. ``verify_business_schemas`` — fail loud if an on-disk table's
       columns drift from the current Pydantic schema. LanceDB has no
       online migration; cascade is rebuildable from md so the recovery
       is ``everos cascade rebuild`` (see ``docs/cascade_runbook.md``).
    5. ``ensure_business_indexes`` — idempotent FTS index creation.
    6. ``_log_unbackfilled_hint`` — warn if unbackfilled rows exist.
    """

    def __init__(self, order: int = 11) -> None:
        super().__init__(name="lancedb", order=order)
        self._projection_lock: AbstractAsyncContextManager[None] | None = None

    async def startup(self, app: FastAPI) -> Any:
        # Retain the shared lock from before marker verification until after
        # shutdown has disposed every cached table handle. An offline rebuild
        # owns the exclusive side, so the marker cannot change between this
        # verification and later server writes.
        lock = projection_server_lock(MemoryRoot.resolve())
        await lock.__aenter__()
        try:
            # This gate must run before opening LanceDB or running any migration.
            # Otherwise startup itself could mutate a legacy projection before it
            # has proved that the row-id generation is current.
            await verify_storage_identity_ready()
            conn = await get_connection()
            await verify_business_schemas()
            await ensure_business_indexes()
            await _log_unbackfilled_hint()
        except BaseException as exc:
            # Preserve the startup error even if best-effort connection cleanup
            # also fails. The lifecycle lock still has to be released so a
            # recovery rebuild is not stranded behind this failed process.
            with suppress(Exception):
                await dispose_connection()
            await lock.__aexit__(type(exc), exc, exc.__traceback__)
            raise
        self._projection_lock = lock
        logger.info("lancedb_ready", uri=conn.uri)
        return conn

    async def shutdown(self, app: FastAPI) -> None:
        lock = self._projection_lock
        try:
            await dispose_connection()
        finally:
            self._projection_lock = None
            if lock is not None:
                await lock.__aexit__(None, None, None)

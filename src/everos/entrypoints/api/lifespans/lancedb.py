"""LanceDB lifespan provider (HTTP API entrypoint).

Startup:
    Open the connection via ``get_connection`` (lazy, idempotent).
    Importing :mod:`everos.infra.persistence.lancedb` also triggers the
    side-effect import of ``tables`` so business schemas are loaded
    (future: preflight registration).
    Log hint if unbackfilled (vector IS NULL) rows exist.

Shutdown:
    Close the connection (also clears the table cache).

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

from typing import Any

from fastapi import FastAPI

from everos.core.lifespan import LifespanProvider
from everos.core.observability.logging import get_logger
from everos.infra.persistence.index import (
    ALL_REPOS,
    active_backend,
    is_null,
    schema_for,
)
from everos.infra.persistence.index import (
    shutdown as shutdown_index,
)
from everos.infra.persistence.index import (
    startup as startup_index,
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
    for repo in ALL_REPOS:
        logical_schema = schema_for(repo.schema)
        if not any(field.name == "vector" for field in logical_schema.vector_fields):
            continue
        try:
            count = await repo.count_where(is_null("vector"))
        except Exception as exc:
            logger.warning(
                "unbackfilled_check_failed",
                schema=repo.schema.__name__,
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

    Startup runs four steps:

    1. ``get_connection`` — lazy-open the async connection.
    2. ``verify_business_schemas`` — fail loud if an on-disk table's
       columns drift from the current Pydantic schema. LanceDB has no
       online migration; cascade is rebuildable from md so the recovery
       is ``everos cascade rebuild`` (see ``docs/cascade_runbook.md``).
    3. ``ensure_business_indexes`` — idempotent FTS index creation.
    4. ``_log_unbackfilled_hint`` — warn if unbackfilled rows exist.
    """

    def __init__(self, order: int = 11) -> None:
        super().__init__(name="lancedb", order=order)

    async def startup(self, app: FastAPI) -> Any:
        conn = await startup_index()
        if active_backend() == "lancedb":
            await _log_unbackfilled_hint()
        logger.info("derived_index_ready", backend=active_backend())
        return conn

    async def shutdown(self, app: FastAPI) -> None:
        await shutdown_index()

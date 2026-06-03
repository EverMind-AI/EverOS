"""LanceDB lifespan provider (HTTP API entrypoint).

Startup:
    Open the connection via ``get_connection`` (lazy, idempotent).
    Importing :mod:`everos.infra.persistence.lancedb` also triggers the
    side-effect import of ``tables`` so business schemas are loaded
    (future: preflight registration).

Shutdown:
    Close the connection (also clears the table cache).
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI

from everos.core.lifespan import LifespanProvider
from everos.core.observability.logging import get_logger
from everos.infra.persistence.lancedb import (
    dispose_connection,
    ensure_business_indexes,
    get_connection,
    verify_business_schemas,
)

logger = get_logger(__name__)


class LanceDBLifespanProvider(LifespanProvider):
    """Manage the LanceDB connection + table cache for the app lifecycle.

    Startup runs three steps:

    1. ``get_connection`` — lazy-open the async connection.
    2. ``verify_business_schemas`` — fail loud if an on-disk table's
       columns drift from the current Pydantic schema. LanceDB has no
       online migration; cascade is rebuildable from md so the recovery
       is documented as ``rm -rf ~/.everos/.index/lancedb``.
    3. ``ensure_business_indexes`` — idempotent FTS index creation.
    """

    def __init__(self, order: int = 11) -> None:
        super().__init__(name="lancedb", order=order)

    async def startup(self, app: FastAPI) -> Any:
        conn = await get_connection()
        await verify_business_schemas()
        await ensure_business_indexes()
        logger.info("lancedb_ready", uri=conn.uri)
        return conn

    async def shutdown(self, app: FastAPI) -> None:
        await dispose_connection()

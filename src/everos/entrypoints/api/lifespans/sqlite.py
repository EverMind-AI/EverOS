"""SQLite system-DB lifespan provider (HTTP API entrypoint).

Startup:
    1. Build the engine via ``get_engine`` (lazy, idempotent). Importing
       :mod:`everos.infra.persistence.sqlite` also triggers the side-
       effect import of ``tables`` so every business SQLModel registers
       itself in ``SQLModel.metadata``.
    2. ``SQLModel.metadata.create_all`` so every registered table exists.

Shutdown:
    Dispose the engine + connection pool.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from sqlmodel import SQLModel

from everos.core.lifespan import LifespanProvider
from everos.core.observability.logging import get_logger
from everos.infra.persistence.sqlite import dispose_engine, get_engine

logger = get_logger(__name__)


class SqliteLifespanProvider(LifespanProvider):
    """Manage the SQLite system-DB engine + schema for the app lifecycle."""

    def __init__(self, order: int = 10) -> None:
        super().__init__(name="sqlite", order=order)

    async def startup(self, app: FastAPI) -> Any:
        engine = get_engine()
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)
        logger.info(
            "sqlite_schema_ready",
            tables=len(SQLModel.metadata.tables),
        )
        return engine

    async def shutdown(self, app: FastAPI) -> None:
        await dispose_engine()

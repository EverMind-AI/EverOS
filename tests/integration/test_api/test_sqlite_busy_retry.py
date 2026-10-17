"""End-to-end: a real SQLite lock reaches the HTTP 503 mapping.

The unit tests pin the two halves separately (``session_scope`` translating
the lock, and the exception handler rendering it). This test drives the whole
chain with a second connection holding SQLite's write lock, so a refactor that
removes either half fails here — and it proves the 503 is a retryable
instruction rather than a dead end.

No EverOS lifespan is started: the route uses a session factory bound to a
temporary system DB, and ``register_handlers`` is the production handler set.
"""

from __future__ import annotations

import sqlite3
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlmodel import Field, SQLModel

from everos.config import SqliteSettings
from everos.core.persistence import (
    MemoryRoot,
    create_session_factory,
    create_system_engine,
    session_scope,
)
from everos.entrypoints.api.exception_handlers import register_handlers


class _BusyRow(SQLModel, table=True):
    """Tiny model used only by this test."""

    __tablename__ = "_api_sqlite_busy_row"  # type: ignore[assignment]
    id: int | None = Field(default=None, primary_key=True)
    note: str


def _make_app(factory: async_sessionmaker[AsyncSession]) -> FastAPI:
    """Minimal app exposing the production handler set plus one write route."""
    app = FastAPI()
    register_handlers(app)

    @app.get("/write")
    async def _write() -> dict[str, str]:
        async with session_scope(factory) as session:
            session.add(_BusyRow(note="from-http"))
            await session.commit()
        return {"status": "ok"}

    return app


@pytest.fixture
async def memory_root(tmp_path: Path) -> AsyncIterator[MemoryRoot]:
    mr = MemoryRoot(tmp_path)
    mr.ensure()
    yield mr


async def test_real_sqlite_lock_reaches_503_with_retry_after(
    memory_root: MemoryRoot,
) -> None:
    engine = create_system_engine(
        memory_root.system_db,
        SqliteSettings(busy_timeout_ms=100),
    )
    factory = create_session_factory(engine)
    app = _make_app(factory)
    holder = sqlite3.connect(memory_root.system_db, isolation_level=None)
    try:
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

        # Hold SQLite's write lock on a second connection. The route's
        # commit waits out busy_timeout_ms, then session_scope classifies it.
        holder.execute("BEGIN IMMEDIATE")
        holder.execute("INSERT INTO _api_sqlite_busy_row (note) VALUES ('held')")

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            busy = await client.get("/write")
            assert busy.status_code == 503, busy.text
            assert busy.headers["retry-after"] == "1"
            body = busy.json()
            assert body["error"]["code"] == "DATABASE_BUSY"
            assert body["error"]["path"] == "/write"

            # Release the lock: the same request must now succeed, proving
            # the 503 told the caller something true.
            holder.execute("ROLLBACK")
            retried = await client.get("/write")
            assert retried.status_code == 200, retried.text
    finally:
        if holder.in_transaction:
            holder.execute("ROLLBACK")
        holder.close()
        await engine.dispose()

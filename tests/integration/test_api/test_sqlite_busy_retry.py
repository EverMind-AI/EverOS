"""SQLite lock contention -> 503 DATABASE_BUSY -> retry succeeds.

Drives the production exception handler and the real ``session_scope``
against a real SQLite file. A raw ``BEGIN IMMEDIATE`` holds the write lock;
the route's commit fails with ``database is locked`` and must surface as a
retryable 503 with ``Retry-After``, not as a 500. Releasing the lock makes
the same request succeed.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlmodel import Field, SQLModel

from everos.config import SqliteSettings
from everos.entrypoints.api.exception_handlers import register_handlers
from everos.infra.persistence import MemoryRoot
from everos.infra.persistence.sqlite import (
    create_session_factory,
    create_system_engine,
    session_scope,
)


class _BusyRow(SQLModel, table=True):
    __tablename__ = "_busy_retry_row"  # type: ignore[assignment]
    id: int | None = Field(default=None, primary_key=True)
    note: str


def _make_app(factory: object) -> FastAPI:
    app = FastAPI()
    register_handlers(app)

    @app.get("/busy")
    async def _busy() -> dict[str, bool]:
        async with session_scope(factory) as session:  # type: ignore[arg-type]
            session.add(_BusyRow(note="row"))
            await session.commit()
        return {"ok": True}

    return app


async def test_locked_sqlite_returns_503_then_succeeds(
    tmp_path: Path,
) -> None:
    root = MemoryRoot(tmp_path)
    root.ensure()
    engine = create_system_engine(root.system_db, SqliteSettings(busy_timeout_ms=0))
    factory = create_session_factory(engine)
    try:
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

        blocker = sqlite3.connect(root.system_db, timeout=0)
        blocker.execute("BEGIN IMMEDIATE")
        try:
            async with AsyncClient(
                transport=ASGITransport(app=_make_app(factory)),
                base_url="http://test",
            ) as client:
                busy = await client.get("/busy")
                assert busy.status_code == 503
                assert busy.json()["error"]["code"] == "DATABASE_BUSY"
                assert busy.headers["Retry-After"] == "1"
        finally:
            blocker.rollback()
            blocker.close()

        async with AsyncClient(
            transport=ASGITransport(app=_make_app(factory)),
            base_url="http://test",
        ) as client:
            ok = await client.get("/busy")
            assert ok.status_code == 200
            assert ok.json() == {"ok": True}
    finally:
        await engine.dispose()

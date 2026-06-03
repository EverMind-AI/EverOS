"""SQLite + LanceDB lifespan providers — startup wires singletons, shutdown disposes."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI

from everos.entrypoints.api.lifespans import (
    LanceDBLifespanProvider,
    SqliteLifespanProvider,
)
from everos.infra.persistence.lancedb import lancedb_manager
from everos.infra.persistence.sqlite import sqlite_manager


@pytest.fixture(autouse=True)
async def _reset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Redirect both managers at an isolated memory-root."""
    monkeypatch.setenv("EVEROS_MEMORY__ROOT", str(tmp_path))
    sqlite_manager._engine = None
    sqlite_manager._session_factory = None
    lancedb_manager._conn = None
    lancedb_manager._tables.clear()
    yield
    await sqlite_manager.dispose_engine()
    await lancedb_manager.dispose_connection()


async def test_sqlite_provider_startup_builds_engine_and_creates_schema(
    tmp_path: Path,
) -> None:
    provider = SqliteLifespanProvider()
    app = FastAPI()

    engine = await provider.startup(app)

    assert engine is sqlite_manager.get_engine()  # singleton wired
    assert (
        tmp_path / ".index" / "sqlite" / "system.db"
    ).exists()  # schema create_all opened the file


async def test_sqlite_provider_shutdown_disposes_singleton() -> None:
    provider = SqliteLifespanProvider()
    app = FastAPI()
    await provider.startup(app)
    assert sqlite_manager._engine is not None

    await provider.shutdown(app)
    assert sqlite_manager._engine is None


async def test_lancedb_provider_startup_opens_connection(tmp_path: Path) -> None:
    provider = LanceDBLifespanProvider()
    app = FastAPI()

    conn = await provider.startup(app)

    assert conn is await lancedb_manager.get_connection()  # singleton wired
    assert (tmp_path / ".index" / "lancedb").is_dir()


async def test_lancedb_provider_shutdown_disposes_singleton() -> None:
    provider = LanceDBLifespanProvider()
    app = FastAPI()
    await provider.startup(app)
    assert lancedb_manager._conn is not None

    await provider.shutdown(app)
    assert lancedb_manager._conn is None

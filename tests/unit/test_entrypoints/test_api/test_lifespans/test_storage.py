"""SQLite + LanceDB lifespan providers — startup wires singletons, shutdown disposes."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI

import everos.entrypoints.api.lifespans.lancedb as lancedb_lifespan
from everos.core.persistence import MemoryRoot
from everos.entrypoints.api.lifespans import (
    LanceDBLifespanProvider,
    SqliteLifespanProvider,
)
from everos.infra.persistence.lancedb import lancedb_manager
from everos.infra.persistence.lancedb.projection_lock import (
    ProjectionLockUnavailableError,
    projection_rebuild_lock,
)
from everos.infra.persistence.sqlite import sqlite_manager


@pytest.fixture(autouse=True)
async def _reset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Redirect both managers at an isolated memory-root."""
    monkeypatch.setenv("EVEROS_ROOT", str(tmp_path))
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
    await provider.shutdown(app)


async def test_lancedb_provider_shutdown_disposes_singleton() -> None:
    provider = LanceDBLifespanProvider()
    app = FastAPI()
    await provider.startup(app)
    assert lancedb_manager._conn is not None

    await provider.shutdown(app)
    assert lancedb_manager._conn is None


async def test_lancedb_provider_holds_shared_projection_lock_through_disposal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = LanceDBLifespanProvider()
    app = FastAPI()
    real_verify = lancedb_lifespan.verify_storage_identity_ready
    real_dispose = lancedb_lifespan.dispose_connection
    observed: list[str] = []

    async def verify_while_locked() -> None:
        with pytest.raises(ProjectionLockUnavailableError):
            async with projection_rebuild_lock(MemoryRoot(tmp_path)):
                raise AssertionError("rebuild lock entered during marker verification")
        observed.append("verify_locked")
        await real_verify()

    async def dispose_while_locked() -> None:
        with pytest.raises(ProjectionLockUnavailableError):
            async with projection_rebuild_lock(MemoryRoot(tmp_path)):
                raise AssertionError("rebuild lock entered before connection disposal")
        observed.append("dispose_locked")
        await real_dispose()

    monkeypatch.setattr(
        lancedb_lifespan, "verify_storage_identity_ready", verify_while_locked
    )
    monkeypatch.setattr(lancedb_lifespan, "dispose_connection", dispose_while_locked)

    await provider.startup(app)
    await provider.shutdown(app)

    assert observed == ["verify_locked", "dispose_locked"]
    async with projection_rebuild_lock(MemoryRoot(tmp_path)):
        pass


async def test_lancedb_startup_failure_releases_projection_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail_verification() -> None:
        raise RuntimeError("marker rejected")

    monkeypatch.setattr(
        lancedb_lifespan, "verify_storage_identity_ready", fail_verification
    )
    provider = LanceDBLifespanProvider()

    with pytest.raises(RuntimeError, match="marker rejected"):
        await provider.startup(FastAPI())

    async with projection_rebuild_lock(MemoryRoot(tmp_path)):
        pass

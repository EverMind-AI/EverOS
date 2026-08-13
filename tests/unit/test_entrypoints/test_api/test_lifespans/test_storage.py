"""SQLite + LanceDB lifespan providers — startup wires singletons, shutdown disposes."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import anyio
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


async def test_lancedb_provider_serializes_only_bootstrap_in_fixed_lock_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    @asynccontextmanager
    async def server_lock(_root: MemoryRoot):  # type: ignore[no-untyped-def]
        events.append("projection_shared_enter")
        try:
            yield
        finally:
            events.append("projection_shared_exit")

    @asynccontextmanager
    async def bootstrap_lock(_root: MemoryRoot):  # type: ignore[no-untyped-def]
        assert events == ["projection_shared_enter"]
        events.append("bootstrap_exclusive_enter")
        try:
            yield
        finally:
            events.append("bootstrap_exclusive_exit")

    async def record(name: str, result=None):  # type: ignore[no-untyped-def]
        events.append(name)
        return result

    monkeypatch.setattr(lancedb_lifespan, "projection_server_lock", server_lock)
    monkeypatch.setattr(lancedb_lifespan, "projection_bootstrap_lock", bootstrap_lock)
    monkeypatch.setattr(
        lancedb_lifespan,
        "verify_storage_identity_ready",
        lambda: record("identity_gate"),
    )
    monkeypatch.setattr(
        lancedb_lifespan,
        "get_connection",
        lambda: record("connection", SimpleNamespace(uri="test://lancedb")),
    )
    monkeypatch.setattr(
        lancedb_lifespan,
        "verify_business_schemas",
        lambda: record("schema_verify"),
    )
    monkeypatch.setattr(
        lancedb_lifespan,
        "ensure_business_indexes",
        lambda: record("index_ensure"),
    )
    monkeypatch.setattr(
        lancedb_lifespan,
        "_log_unbackfilled_hint",
        lambda: record("unbackfilled_hint"),
    )
    monkeypatch.setattr(
        lancedb_lifespan,
        "dispose_connection",
        lambda: record("dispose"),
    )

    provider = LanceDBLifespanProvider()
    await provider.startup(FastAPI())

    assert events == [
        "projection_shared_enter",
        "bootstrap_exclusive_enter",
        "identity_gate",
        "connection",
        "schema_verify",
        "index_ensure",
        "bootstrap_exclusive_exit",
        "unbackfilled_hint",
    ]

    await provider.shutdown(FastAPI())
    assert events[-2:] == ["dispose", "projection_shared_exit"]


async def test_lancedb_bootstrap_failure_cleans_up_before_unlocking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    @asynccontextmanager
    async def server_lock(_root: MemoryRoot):  # type: ignore[no-untyped-def]
        events.append("projection_shared_enter")
        try:
            yield
        finally:
            events.append("projection_shared_exit")

    @asynccontextmanager
    async def bootstrap_lock(_root: MemoryRoot):  # type: ignore[no-untyped-def]
        events.append("bootstrap_exclusive_enter")
        try:
            yield
        finally:
            events.append("bootstrap_exclusive_exit")

    async def pass_gate() -> None:
        events.append("identity_gate")

    async def open_connection():
        events.append("connection")
        return SimpleNamespace(uri="test://lancedb")

    async def fail_schema() -> None:
        events.append("schema_verify")
        raise RuntimeError("schema rejected")

    dispose_calls = 0

    async def dispose() -> None:
        nonlocal dispose_calls
        dispose_calls += 1
        events.append(f"dispose_{dispose_calls}")
        if dispose_calls == 1:
            raise RuntimeError("transient cleanup failure")

    monkeypatch.setattr(lancedb_lifespan, "projection_server_lock", server_lock)
    monkeypatch.setattr(lancedb_lifespan, "projection_bootstrap_lock", bootstrap_lock)
    monkeypatch.setattr(lancedb_lifespan, "verify_storage_identity_ready", pass_gate)
    monkeypatch.setattr(lancedb_lifespan, "get_connection", open_connection)
    monkeypatch.setattr(lancedb_lifespan, "verify_business_schemas", fail_schema)
    monkeypatch.setattr(lancedb_lifespan, "dispose_connection", dispose)

    with pytest.raises(RuntimeError, match="schema rejected"):
        await LanceDBLifespanProvider().startup(FastAPI())

    assert events == [
        "projection_shared_enter",
        "bootstrap_exclusive_enter",
        "identity_gate",
        "connection",
        "schema_verify",
        "dispose_1",
        "dispose_2",
        "bootstrap_exclusive_exit",
        "projection_shared_exit",
    ]


async def test_lancedb_bootstrap_preserves_primary_error_when_cleanup_exhausts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    @asynccontextmanager
    async def server_lock(_root: MemoryRoot):  # type: ignore[no-untyped-def]
        events.append("projection_shared_enter")
        try:
            yield
        finally:
            events.append("projection_shared_exit")

    @asynccontextmanager
    async def bootstrap_lock(_root: MemoryRoot):  # type: ignore[no-untyped-def]
        events.append("bootstrap_exclusive_enter")
        try:
            yield
        finally:
            events.append("bootstrap_exclusive_exit")

    async def pass_gate() -> None:
        return None

    async def open_connection():
        return SimpleNamespace(uri="test://lancedb")

    async def fail_schema() -> None:
        raise ValueError("PRIMARY_INIT_FAILURE")

    async def fail_dispose() -> None:
        events.append("dispose_failed")
        raise RuntimeError("SECONDARY_DISPOSE_FAILURE")

    monkeypatch.setattr(lancedb_lifespan, "projection_server_lock", server_lock)
    monkeypatch.setattr(lancedb_lifespan, "projection_bootstrap_lock", bootstrap_lock)
    monkeypatch.setattr(lancedb_lifespan, "verify_storage_identity_ready", pass_gate)
    monkeypatch.setattr(lancedb_lifespan, "get_connection", open_connection)
    monkeypatch.setattr(lancedb_lifespan, "verify_business_schemas", fail_schema)
    monkeypatch.setattr(lancedb_lifespan, "dispose_connection", fail_dispose)

    with pytest.raises(ValueError, match="PRIMARY_INIT_FAILURE"):
        await LanceDBLifespanProvider().startup(FastAPI())

    assert events == [
        "projection_shared_enter",
        "bootstrap_exclusive_enter",
        "dispose_failed",
        "dispose_failed",
        "bootstrap_exclusive_exit",
        "projection_shared_exit",
    ]


async def test_lancedb_shutdown_cancellation_keeps_projection_lock_until_disposed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = LanceDBLifespanProvider()
    app = FastAPI()
    await provider.startup(app)
    real_dispose = lancedb_lifespan.dispose_connection
    dispose_started = anyio.Event()
    allow_dispose = anyio.Event()
    shutdown_done = anyio.Event()
    cancel_scope: anyio.CancelScope | None = None

    async def delayed_dispose() -> None:
        dispose_started.set()
        await allow_dispose.wait()
        await real_dispose()

    monkeypatch.setattr(lancedb_lifespan, "dispose_connection", delayed_dispose)

    async def run_shutdown() -> None:
        nonlocal cancel_scope
        with anyio.CancelScope() as scope:
            cancel_scope = scope
            try:
                await provider.shutdown(app)
            finally:
                shutdown_done.set()

    async with anyio.create_task_group() as tasks:
        tasks.start_soon(run_shutdown)
        await dispose_started.wait()
        assert cancel_scope is not None
        cancel_scope.cancel()
        await anyio.lowlevel.checkpoint()

        assert not shutdown_done.is_set()
        with pytest.raises(ProjectionLockUnavailableError):
            async with projection_rebuild_lock(MemoryRoot(tmp_path)):
                raise AssertionError("rebuild entered during shielded cleanup")

        allow_dispose.set()
        await shutdown_done.wait()

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

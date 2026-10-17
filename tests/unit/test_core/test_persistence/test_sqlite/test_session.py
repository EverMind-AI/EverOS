"""Unit tests for session_scope semantics."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from sqlmodel import Field, SQLModel

from everos.config import SqliteSettings
from everos.core.errors import DatabaseBusyError
from everos.core.persistence import (
    MemoryRoot,
    create_session_factory,
    create_system_engine,
    session_scope,
)


class _Sample(SQLModel, table=True):
    """Tiny model used only by these tests."""

    __tablename__ = "_sample_session_scope"  # type: ignore[assignment]
    id: int | None = Field(default=None, primary_key=True)
    note: str


@pytest.fixture
def memory_root(tmp_path: Path) -> MemoryRoot:
    mr = MemoryRoot(tmp_path)
    mr.ensure()
    return mr


async def test_session_scope_commits_on_success(memory_root: MemoryRoot) -> None:
    engine = create_system_engine(memory_root.system_db, SqliteSettings())
    factory = create_session_factory(engine)
    try:
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

        async with session_scope(factory) as s:
            s.add(_Sample(note="hello"))
            await s.commit()

        async with session_scope(factory) as s:
            row = (
                await s.execute(text("SELECT note FROM _sample_session_scope"))
            ).fetchone()
            assert row is not None
            assert row[0] == "hello"
    finally:
        await engine.dispose()


async def test_session_scope_rolls_back_on_exception(
    memory_root: MemoryRoot,
) -> None:
    engine = create_system_engine(memory_root.system_db, SqliteSettings())
    factory = create_session_factory(engine)
    try:
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

        with pytest.raises(RuntimeError):
            async with session_scope(factory) as s:
                s.add(_Sample(note="should rollback"))
                # No commit yet → scope must rollback on exception.
                raise RuntimeError("boom")

        async with session_scope(factory) as s:
            count = (
                await s.execute(text("SELECT COUNT(*) FROM _sample_session_scope"))
            ).fetchone()
            assert count is not None
            assert count[0] == 0
    finally:
        await engine.dispose()


async def test_session_scope_translates_lock_contention_to_database_busy(
    memory_root: MemoryRoot,
) -> None:
    """A busy SQLite writer becomes DatabaseBusyError, not a bare OperationalError.

    Reproduces the shape of issue #498 without a 10-hour soak: hold the write
    lock on a second connection past ``busy_timeout_ms`` and assert the
    persistence layer classifies the failure as retryable. The HTTP mapping
    (503 + Retry-After) is pinned separately in the API handler tests.
    """
    engine = create_system_engine(
        memory_root.system_db,
        SqliteSettings(busy_timeout_ms=100),
    )
    factory = create_session_factory(engine)
    holder = sqlite3.connect(memory_root.system_db, isolation_level=None)
    try:
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

        # Take the write lock and keep it held. isolation_level=None turns
        # off sqlite3's implicit transaction so BEGIN IMMEDIATE is legal.
        holder.execute("BEGIN IMMEDIATE")
        holder.execute("INSERT INTO _sample_session_scope (note) VALUES ('held')")

        with pytest.raises(DatabaseBusyError):
            async with session_scope(factory) as s:
                s.add(_Sample(note="blocked"))
                await s.commit()
    finally:
        if holder.in_transaction:
            holder.execute("ROLLBACK")
        holder.close()
        await engine.dispose()


async def test_session_scope_does_not_translate_unrelated_operational_errors(
    memory_root: MemoryRoot,
) -> None:
    """Only lock contention is retryable; a malformed statement stays an error."""
    engine = create_system_engine(memory_root.system_db, SqliteSettings())
    factory = create_session_factory(engine)
    try:
        with pytest.raises(OperationalError) as excinfo:
            async with session_scope(factory) as s:
                await s.execute(text("SELECT * FROM _table_that_does_not_exist"))
        assert not isinstance(excinfo.value, DatabaseBusyError)
    finally:
        await engine.dispose()

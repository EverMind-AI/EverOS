"""Unit tests for session_scope semantics."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from sqlmodel import Field, SQLModel

from everos.config import SqliteSettings
from everos.core.errors import StorageError
from everos.infra.persistence import MemoryRoot
from everos.infra.persistence.sqlite import (
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


async def test_a_locked_database_surfaces_as_a_retryable_storage_error(
    memory_root: MemoryRoot,
) -> None:
    """Another connection holds the write lock past ``busy_timeout``: the
    caller gets a ``StorageError`` (the API answers 503, retry), and its
    message carries neither the SQL nor the bound values, which are user data.
    """
    engine = create_system_engine(
        memory_root.system_db, SqliteSettings(busy_timeout_ms=50)
    )
    factory = create_session_factory(engine)
    holder = sqlite3.connect(memory_root.system_db, isolation_level=None)
    try:
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)
        holder.execute("BEGIN IMMEDIATE")

        with pytest.raises(StorageError) as caught:
            async with session_scope(factory) as s:
                s.add(_Sample(note="private words"))
                await s.commit()

        assert isinstance(caught.value.__cause__, OperationalError)
        assert "private words" not in str(caught.value)
        assert "INSERT" not in str(caught.value)
    finally:
        holder.execute("ROLLBACK")
        holder.close()
        await engine.dispose()


async def test_other_operational_errors_pass_through(
    memory_root: MemoryRoot,
) -> None:
    """Only a lock wait is retryable; a missing table is a bug, not a 503."""
    engine = create_system_engine(memory_root.system_db, SqliteSettings())
    factory = create_session_factory(engine)
    try:
        with pytest.raises(OperationalError):
            async with session_scope(factory) as s:
                await s.execute(text("SELECT * FROM no_such_table"))
    finally:
        await engine.dispose()

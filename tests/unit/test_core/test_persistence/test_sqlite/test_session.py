"""Unit tests for session_scope semantics."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import text
from sqlmodel import Field, SQLModel

from everos.config import SqliteSettings
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

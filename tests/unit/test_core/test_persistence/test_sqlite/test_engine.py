"""Unit tests for the SQLite async engine + PRAGMA listener.

Critical: verifies PRAGMAs are actually applied at the SQLite layer
(not just declared in code). The whole reason for the listener is that
PRAGMAs are per-connection and the SA pool reuses connections.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import text

from everos.config import SqliteSettings
from everos.core.persistence import (
    MemoryRoot,
    create_session_factory,
    create_system_engine,
    session_scope,
)


@pytest.fixture
def memory_root(tmp_path: Path) -> MemoryRoot:
    mr = MemoryRoot(tmp_path)
    mr.ensure()
    return mr


async def test_engine_creates_db_file(memory_root: MemoryRoot) -> None:
    engine = create_system_engine(memory_root.system_db, SqliteSettings())
    factory = create_session_factory(engine)
    async with session_scope(factory) as s:
        await s.execute(text("SELECT 1"))
    await engine.dispose()
    assert memory_root.system_db.exists()


async def test_pragmas_actually_applied_default_settings(
    memory_root: MemoryRoot,
) -> None:
    """Default PRAGMAs match what's in default.toml."""
    settings = SqliteSettings()
    engine = create_system_engine(memory_root.system_db, settings)
    factory = create_session_factory(engine)
    try:
        async with session_scope(factory) as s:
            assert _scalar(await _pragma(s, "journal_mode")) == "wal"
            # synchronous: 0=OFF 1=NORMAL 2=FULL 3=EXTRA
            assert _scalar(await _pragma(s, "synchronous")) == 1
            # foreign_keys: 1=ON 0=OFF
            assert _scalar(await _pragma(s, "foreign_keys")) == 1
            # temp_store: 0=DEFAULT 1=FILE 2=MEMORY
            assert _scalar(await _pragma(s, "temp_store")) == 2
            assert _scalar(await _pragma(s, "busy_timeout")) == 5000
            assert _scalar(await _pragma(s, "journal_size_limit")) == 64 * 1024 * 1024
            # cache_size: negative value = KB; positive = pages
            assert _scalar(await _pragma(s, "cache_size")) == -2048
    finally:
        await engine.dispose()


async def test_pragmas_respect_custom_settings(memory_root: MemoryRoot) -> None:
    """Engine reflects non-default tunables."""
    settings = SqliteSettings(
        journal_mode="DELETE",
        synchronous="FULL",
        foreign_keys=False,
        temp_store="FILE",
        busy_timeout_ms=10000,
        journal_size_limit_bytes=1024 * 1024,
        cache_size_kb=4096,
    )
    engine = create_system_engine(memory_root.system_db, settings)
    factory = create_session_factory(engine)
    try:
        async with session_scope(factory) as s:
            assert _scalar(await _pragma(s, "journal_mode")) == "delete"
            assert _scalar(await _pragma(s, "synchronous")) == 2  # FULL
            assert _scalar(await _pragma(s, "foreign_keys")) == 0
            assert _scalar(await _pragma(s, "temp_store")) == 1  # FILE
            assert _scalar(await _pragma(s, "busy_timeout")) == 10000
            assert _scalar(await _pragma(s, "cache_size")) == -4096
    finally:
        await engine.dispose()


async def test_pragmas_applied_on_each_new_connection(
    memory_root: MemoryRoot,
) -> None:
    """The listener fires on every new connection from the pool, not just once."""
    settings = SqliteSettings()
    engine = create_system_engine(memory_root.system_db, settings)
    factory = create_session_factory(engine)
    try:
        # Two independent sessions → at least two connection acquisitions
        # → both must see WAL mode.
        async with session_scope(factory) as s1:
            assert _scalar(await _pragma(s1, "journal_mode")) == "wal"
        async with session_scope(factory) as s2:
            assert _scalar(await _pragma(s2, "journal_mode")) == "wal"
    finally:
        await engine.dispose()


async def _pragma(session, name: str):  # type: ignore[no-untyped-def]
    return await session.execute(text(f"PRAGMA {name}"))


def _scalar(result):  # type: ignore[no-untyped-def]
    row = result.fetchone()
    return row[0] if row is not None else None

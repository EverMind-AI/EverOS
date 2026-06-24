"""Tests for OME storage schema migration — event_id column."""

from __future__ import annotations

from pathlib import Path

import aiosqlite
import pytest

from everos.infra.ome._stores.storage import OMEStorage


@pytest.mark.asyncio
async def test_fresh_db_has_event_id_column(tmp_path: Path) -> None:
    """A brand-new database should have the event_id column."""
    storage = OMEStorage(db_path=tmp_path / "ome.db")
    await storage.init()
    async with aiosqlite.connect(tmp_path / "ome.db") as conn:
        cur = await conn.execute("PRAGMA table_info(run_record)")
        columns = {row[1] for row in await cur.fetchall()}
    assert "event_id" in columns


@pytest.mark.asyncio
async def test_existing_db_without_event_id_gets_migrated(tmp_path: Path) -> None:
    """An existing database created before P3 should gain the event_id
    column after init() runs the migration.
    """
    db_path = tmp_path / "ome.db"
    async with aiosqlite.connect(db_path) as conn:
        await conn.execute(
            "CREATE TABLE run_record ("
            "  run_id TEXT PRIMARY KEY,"
            "  strategy_name TEXT NOT NULL,"
            "  status TEXT NOT NULL,"
            "  attempt INTEGER NOT NULL DEFAULT 0,"
            "  started_at TIMESTAMP NOT NULL,"
            "  finished_at TIMESTAMP,"
            "  error TEXT,"
            "  event_topic TEXT NOT NULL,"
            "  event_payload TEXT NOT NULL,"
            "  max_retries_snapshot INTEGER NOT NULL"
            ")"
        )
        await conn.commit()

    storage = OMEStorage(db_path=db_path)
    await storage.init()

    async with aiosqlite.connect(db_path) as conn:
        cur = await conn.execute("PRAGMA table_info(run_record)")
        columns = {row[1] for row in await cur.fetchall()}
    assert "event_id" in columns


@pytest.mark.asyncio
async def test_migration_is_idempotent(tmp_path: Path) -> None:
    """Calling init() twice on the same database must not fail."""
    storage = OMEStorage(db_path=tmp_path / "ome.db")
    await storage.init()
    await storage.init()

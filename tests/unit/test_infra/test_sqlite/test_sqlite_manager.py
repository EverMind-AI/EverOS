"""``SqliteDatabase``: one root's engine and session factory.

Building touches nothing on disk; the database directory appears with the
first connection, and ``dispose`` leaves the engine usable.
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import text

from everos.config import SqliteSettings
from everos.infra.persistence import MemoryRoot
from everos.infra.persistence.sqlite import SqliteDatabase


def _database(tmp_path: Path) -> SqliteDatabase:
    return SqliteDatabase(MemoryRoot(tmp_path / "root"), SqliteSettings())


def test_building_touches_nothing_on_disk(tmp_path: Path) -> None:
    _database(tmp_path)
    assert not (tmp_path / "root").exists()


async def test_engine_points_at_the_root_system_db(tmp_path: Path) -> None:
    database = _database(tmp_path)
    assert str(tmp_path / "root") in str(database.engine.url)
    assert database.session_factory.kw["bind"] is database.engine


async def test_first_connection_creates_the_directory(tmp_path: Path) -> None:
    database = _database(tmp_path)
    async with database.engine.connect() as conn:
        await conn.execute(text("select 1"))
    assert MemoryRoot(tmp_path / "root").system_db.exists()
    await database.dispose()


async def test_engine_is_usable_after_dispose(tmp_path: Path) -> None:
    database = _database(tmp_path)
    await database.dispose()  # nothing opened yet
    async with database.engine.connect() as conn:
        await conn.execute(text("select 1"))
    await database.dispose()
    async with database.engine.connect() as conn:
        assert (await conn.execute(text("select 1"))).scalar() == 1
    await database.dispose()

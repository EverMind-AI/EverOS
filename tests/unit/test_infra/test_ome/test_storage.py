from __future__ import annotations

from pathlib import Path

import aiosqlite
import pytest

from everos.infra.ome._stores.storage import OMEStorage


@pytest.mark.asyncio
async def test_storage_creates_db_and_tables(tmp_path: Path) -> None:
    db = tmp_path / "ome.db"
    storage = OMEStorage(db_path=db)
    await storage.init()

    assert db.exists()
    async with aiosqlite.connect(db) as conn:
        cur = await conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        names = {row[0] for row in await cur.fetchall()}
    assert {"counter_store", "idle_store", "run_record"}.issubset(names)


@pytest.mark.asyncio
async def test_storage_applies_pragmas(tmp_path: Path) -> None:
    db = tmp_path / "ome.db"
    storage = OMEStorage(db_path=db)
    await storage.init()

    async with aiosqlite.connect(db) as conn:
        cur = await conn.execute("PRAGMA journal_mode")
        mode = (await cur.fetchone())[0]
    assert mode == "wal"


@pytest.mark.asyncio
async def test_storage_init_is_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "ome.db"
    storage = OMEStorage(db_path=db)
    await storage.init()
    await storage.init()  # second call must not raise


@pytest.mark.asyncio
async def test_storage_creates_parent_dir(tmp_path: Path) -> None:
    db = tmp_path / "nested" / "dir" / "ome.db"
    storage = OMEStorage(db_path=db)
    await storage.init()
    assert db.exists()


@pytest.mark.asyncio
async def test_storage_connect_applies_per_connection_pragmas(
    tmp_path: Path,
) -> None:
    """``synchronous`` and ``cache_size`` are per-connection PRAGMAs:
    SQLite resets them to defaults on every new connection. The
    ``OMEStorage.connect`` wrapper must re-apply them or the module
    docstring's promise is silently broken.
    """
    db = tmp_path / "ome.db"
    storage = OMEStorage(db_path=db)
    await storage.init()

    async with storage.connect() as conn:
        sync_row = await (await conn.execute("PRAGMA synchronous")).fetchone()
        cache_row = await (await conn.execute("PRAGMA cache_size")).fetchone()
        busy_row = await (await conn.execute("PRAGMA busy_timeout")).fetchone()

    # synchronous: 0=OFF, 1=NORMAL, 2=FULL, 3=EXTRA
    assert sync_row[0] == 1
    # cache_size: negative value is "kibibytes of memory"
    assert cache_row[0] == -65536
    # busy_timeout: ms before SQLITE_BUSY is raised on contended writes
    assert busy_row[0] == 5000


@pytest.mark.asyncio
async def test_storage_raw_aiosqlite_connect_does_not_carry_per_conn_pragmas(
    tmp_path: Path,
) -> None:
    """Sanity check that documents why :meth:`OMEStorage.connect` exists:
    opening the same db with raw ``aiosqlite.connect`` yields a connection
    where ``synchronous`` is at SQLite's default (FULL=2), not NORMAL.
    """
    db = tmp_path / "ome.db"
    storage = OMEStorage(db_path=db)
    await storage.init()

    async with aiosqlite.connect(db) as conn:
        sync_row = await (await conn.execute("PRAGMA synchronous")).fetchone()

    assert sync_row[0] == 2  # default FULL — confirms scope is per-connection


@pytest.mark.asyncio
async def test_storage_transaction_commits_on_success(tmp_path: Path) -> None:
    db = tmp_path / "ome.db"
    storage = OMEStorage(db_path=db)
    await storage.init()

    async with storage.transaction() as conn:
        await conn.execute(
            "INSERT INTO counter_store (strategy_name, bucket_key, counter) "
            "VALUES (?, ?, ?)",
            ("s", "u1", 42),
        )

    async with storage.connect() as conn:
        cur = await conn.execute(
            "SELECT counter FROM counter_store WHERE strategy_name=? AND bucket_key=?",
            ("s", "u1"),
        )
        row = await cur.fetchone()
    assert row is not None and row[0] == 42


@pytest.mark.asyncio
async def test_storage_transaction_rolls_back_on_exception(tmp_path: Path) -> None:
    db = tmp_path / "ome.db"
    storage = OMEStorage(db_path=db)
    await storage.init()

    class _BoomError(Exception):
        pass

    with pytest.raises(_BoomError):
        async with storage.transaction() as conn:
            await conn.execute(
                "INSERT INTO counter_store (strategy_name, bucket_key, counter) "
                "VALUES (?, ?, ?)",
                ("s", "u1", 42),
            )
            raise _BoomError

    async with storage.connect() as conn:
        cur = await conn.execute(
            "SELECT counter FROM counter_store WHERE strategy_name=? AND bucket_key=?",
            ("s", "u1"),
        )
        row = await cur.fetchone()
    assert row is None

"""System-DB schema creation over a database an earlier build wrote.

An earlier 2.0 build stamped the agent-case extraction outcome on each memcell
in three nullable ``memcell.case_*`` columns. EverOS 2.0 has no agent memory:
the columns are no longer part of the table model, a fresh database is created
without them, and a database that has them keeps working -- schema creation
leaves them as they are, and the ORM reads and writes the table around them.
"""

from __future__ import annotations

from pathlib import Path

from everalgo.types import ChatMessage, MemCell
from sqlalchemy import text

from everos.config import SqliteSettings
from everos.infra.persistence import MemoryRoot
from everos.infra.persistence.sqlite import (
    MemcellRepo,
    SqliteDatabase,
    SqliteMemcellStore,
)
from everos.memory.models import MemcellRecord
from everos.memory.ports import SessionKey

_CASE_COLUMNS = {"case_extracted_at", "case_skip_reason", "case_skip_detail_json"}
_CASE_MEMCELL_DDL = """
CREATE TABLE memcell (
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL,
    memcell_id VARCHAR NOT NULL,
    app_id VARCHAR NOT NULL,
    project_id VARCHAR NOT NULL,
    session_id VARCHAR NOT NULL,
    track VARCHAR NOT NULL,
    raw_type VARCHAR NOT NULL,
    message_ids_json VARCHAR NOT NULL,
    sender_ids_json VARCHAR NOT NULL,
    payload_json VARCHAR NOT NULL,
    timestamp DATETIME NOT NULL,
    case_extracted_at DATETIME,
    case_skip_reason VARCHAR,
    case_skip_detail_json VARCHAR,
    PRIMARY KEY (memcell_id)
)
"""
_STAMPED_ROW = """
INSERT INTO memcell VALUES (
    '2026-01-01 00:00:00', '2026-01-01 00:00:00', 'mc_old', 'app', 'proj', 's1',
    'memorize', 'conversation', '[]', '[]', :payload, '2026-01-01 00:00:00',
    '2026-01-01 00:00:00', 'no_messages_after_strip', NULL
)
"""
_SESSION = SessionKey(app_id="app", project_id="proj", session_id="s1")


def _cell() -> MemCell:
    return MemCell(
        items=[
            ChatMessage(id="m1", role="user", content="hi", sender_id="u1", timestamp=1)
        ],
        timestamp=1_747_216_800_000,
    )


async def _memcell_columns(database: SqliteDatabase) -> set[str]:
    async with database.engine.connect() as conn:
        rows = (await conn.execute(text("PRAGMA table_info(memcell)"))).fetchall()
    return {row[1] for row in rows}


async def test_a_database_with_the_case_columns_keeps_working(tmp_path: Path) -> None:
    database = SqliteDatabase(MemoryRoot(tmp_path), SqliteSettings())
    async with database.engine.begin() as conn:
        await conn.execute(text(_CASE_MEMCELL_DDL))
        await conn.execute(text(_STAMPED_ROW), {"payload": _cell().model_dump_json()})

    await database.create_schema()
    await database.create_schema()  # idempotent

    assert await _memcell_columns(database) >= _CASE_COLUMNS
    store = SqliteMemcellStore(MemcellRepo(session_factory=database.session_factory))
    record = MemcellRecord(
        memcell_id="mc_new",
        cell=_cell(),
        raw_type="conversation",
        message_ids=["m1"],
        sender_ids=["u1"],
    )
    await store.insert_many(_SESSION, [record])
    assert await store.find_by_ids(["mc_old", "mc_new"]) == [_cell(), _cell()]
    async with database.engine.connect() as conn:
        stamps = (
            await conn.execute(
                text(
                    "SELECT memcell_id, case_skip_reason FROM memcell "
                    "ORDER BY memcell_id"
                )
            )
        ).fetchall()
    assert [tuple(row) for row in stamps] == [
        ("mc_new", None),
        ("mc_old", "no_messages_after_strip"),
    ]
    await database.dispose()


async def test_a_fresh_database_is_created_without_the_case_columns(
    tmp_path: Path,
) -> None:
    database = SqliteDatabase(MemoryRoot(tmp_path), SqliteSettings())

    tables = await database.create_schema()

    assert tables > 0
    assert not _CASE_COLUMNS & await _memcell_columns(database)
    await database.dispose()


async def test_an_older_ledger_gains_the_unreadable_entries_column(
    tmp_path: Path,
) -> None:
    database = SqliteDatabase(MemoryRoot(tmp_path), SqliteSettings())
    await database.create_schema()
    async with database.engine.begin() as conn:  # a system.db from before it
        await conn.execute(
            text("ALTER TABLE md_change_state DROP COLUMN unreadable_entries")
        )

    await database.create_schema()
    await database.create_schema()  # idempotent

    async with database.engine.connect() as conn:
        rows = (
            await conn.execute(text("PRAGMA table_info(md_change_state)"))
        ).fetchall()
    assert "unreadable_entries" in {row[1] for row in rows}
    await database.dispose()

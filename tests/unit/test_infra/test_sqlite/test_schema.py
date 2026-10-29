"""System-DB schema creation upgrades a database made before a column existed.

``create_all`` never alters an existing table, so a ``system.db`` written
before the ``memcell.case_*`` columns must gain them at schema creation,
keep its rows, and then read and write through the ORM as a fresh one does.
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
_PRE_CASE_MEMCELL_DDL = """
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
    PRIMARY KEY (memcell_id)
)
"""
_LEGACY_ROW = """
INSERT INTO memcell VALUES (
    '2026-01-01 00:00:00', '2026-01-01 00:00:00', 'mc_old', 'app', 'proj', 's1',
    'memorize', 'conversation', '[]', '[]', :payload, '2026-01-01 00:00:00'
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


async def test_an_older_database_gains_the_case_columns(tmp_path: Path) -> None:
    database = SqliteDatabase(MemoryRoot(tmp_path), SqliteSettings())
    async with database.engine.begin() as conn:
        await conn.execute(text(_PRE_CASE_MEMCELL_DDL))
        await conn.execute(text(_LEGACY_ROW), {"payload": _cell().model_dump_json()})
    assert not _CASE_COLUMNS & await _memcell_columns(database)

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
    await store.mark_case_extracted(
        "mc_old", skip_reason="no_messages_after_strip", skip_detail={}
    )
    assert await store.find_by_ids(["mc_old", "mc_new"]) == [_cell(), _cell()]
    async with database.engine.connect() as conn:
        reason = (
            await conn.execute(
                text("SELECT case_skip_reason FROM memcell WHERE memcell_id='mc_old'")
            )
        ).scalar()
    assert reason == "no_messages_after_strip"
    await database.dispose()


async def test_a_fresh_database_is_created_with_the_case_columns(
    tmp_path: Path,
) -> None:
    database = SqliteDatabase(MemoryRoot(tmp_path), SqliteSettings())

    tables = await database.create_schema()

    assert tables > 0
    assert await _memcell_columns(database) >= _CASE_COLUMNS
    await database.dispose()

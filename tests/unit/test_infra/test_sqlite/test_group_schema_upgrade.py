"""A ``system.db`` from before the group deliverer and the dedupe ledger opens.

Schema creation over a database an earlier 2.0 build wrote: ``group_meta``
without ``deliverer_id`` and no ``group_message`` table. What these pin
(docs/group-memory.md, sections 5.2 and 6.4; docs/storage_layout.md):

- startup adds the nullable column and creates the table, twice in a row
  without a change the second time, and nothing else needs a rebuild;
- a group mirrored before reads back with no deliverer;
- the ledger then records and answers by key, and a key recorded again moves
  to the later session.
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import text

from everos.config import SqliteSettings
from everos.infra.persistence import MemoryRoot
from everos.infra.persistence.sqlite import (
    GroupMessageRepo,
    GroupRosterRepo,
    SqliteDatabase,
    SqliteGroupMessageStore,
)
from everos.memory.ports import Scope, SessionKey

_OLD_GROUP_ROW = """
INSERT INTO group_meta (
    created_at, updated_at, app_id, project_id, group_id, name, description,
    group_created_at, created_by, history_visible_from, extra_json, md_path
) VALUES (
    '2026-11-01 00:00:00', '2026-11-01 00:00:00', 'default', 'p', 'trip',
    'Trip', '', 1772323200000, 'u_chen', 'join', '{}',
    'default_app/p/groups/trip/group.md'
)
"""


async def _columns(database: SqliteDatabase, table: str) -> set[str]:
    async with database.engine.connect() as conn:
        rows = (await conn.execute(text(f"PRAGMA table_info({table})"))).fetchall()
    return {row[1] for row in rows}


async def test_an_older_system_db_gains_the_column_and_the_ledger(
    tmp_path: Path,
) -> None:
    database = SqliteDatabase(MemoryRoot(tmp_path), SqliteSettings())
    await database.create_schema()
    # As an earlier build left it.
    async with database.engine.begin() as conn:
        await conn.execute(text("DROP TABLE group_message"))
        await conn.execute(text("ALTER TABLE group_meta DROP COLUMN deliverer_id"))
        await conn.execute(text(_OLD_GROUP_ROW))
    assert "deliverer_id" not in await _columns(database, "group_meta")
    assert not await _columns(database, "group_message")

    await database.create_schema()
    await database.create_schema()  # idempotent

    assert "deliverer_id" in await _columns(database, "group_meta")
    assert {"group_id", "key", "session_id"} <= await _columns(
        database, "group_message"
    )
    rows = await GroupRosterRepo(database.session_factory).load("default", "p", "trip")
    assert rows is not None and rows[0].deliverer_id is None

    ledger = SqliteGroupMessageStore(GroupMessageRepo(database.session_factory))
    scope = Scope(app_id="default", project_id="p", owner_id="trip", owner_type="group")
    await ledger.record(scope, SessionKey("default", "p", "s1"), ["id:m-1", "fp:x"])
    assert await ledger.held(scope, ["id:m-1", "id:m-2"]) == {"id:m-1": "s1"}
    await ledger.record(scope, SessionKey("default", "p", "s2"), ["id:m-1"])
    assert await ledger.held(scope, ["id:m-1", "fp:x"]) == {
        "id:m-1": "s2",
        "fp:x": "s1",
    }
    other = Scope(app_id="default", project_id="p", owner_id="x", owner_type="group")
    assert await ledger.held(other, ["id:m-1"]) == {}
    await database.dispose()

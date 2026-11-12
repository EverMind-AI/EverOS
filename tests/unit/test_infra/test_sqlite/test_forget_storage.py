"""The storage of a group forget: ``message_forget`` and the forget log's new columns.

docs/group-memory.md, sections 6.2 and 9.4. What these pin:

- a message is forgotten while any forget naming it is live, per session (an
  id of one session never forgets the same id in another), and a segment's
  forgotten ids are its session's;
- the forget log takes ``owner_type = group``, ``requested_by`` and the undo
  record, finds an entry by id, and says whether a live group forget took a
  picture out -- an undone one does not count;
- a user and a group with the same id never read each other's live forgets;
- a ``system.db`` written before ``requested_by`` / ``undo_json`` existed gains
  them at schema creation and keeps its rows.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from everalgo.types import ChatMessage, MemCell
from sqlalchemy import text

from everos.config import SqliteSettings
from everos.infra.persistence import MemoryRoot
from everos.infra.persistence.sqlite import (
    SqliteDatabase,
    SqliteForgetLogStore,
    SqliteMemcellStore,
    SqliteMessageForgetStore,
    SqliteRepos,
)
from everos.memory.models import MemcellRecord
from everos.memory.ports import SessionKey

_HASH = "ab" * 32
_S1 = SessionKey(app_id="app", project_id="proj", session_id="s1")
_S2 = SessionKey(app_id="app", project_id="proj", session_id="s2")


@pytest.fixture
async def repos(tmp_path: Path) -> AsyncIterator[SqliteRepos]:
    database = SqliteDatabase(MemoryRoot(tmp_path), SqliteSettings())
    await database.create_schema()
    yield SqliteRepos.over(database)
    await database.dispose()


async def _segment(repos: SqliteRepos, session: SessionKey, memcell_id: str) -> None:
    cell = MemCell(
        items=[
            ChatMessage(id="m-1", role="user", content="hi", sender_id="u", timestamp=1)
        ],
        timestamp=1_747_216_800_000,
    )
    await SqliteMemcellStore(repos.memcell).insert_many(
        session,
        [
            MemcellRecord(
                memcell_id=memcell_id,
                cell=cell,
                raw_type="conversation",
                message_ids=["gen-1"],
                sender_ids=["u"],
            )
        ],
    )


async def test_a_message_stays_forgotten_while_a_forget_of_it_is_live(
    repos: SqliteRepos,
) -> None:
    store = SqliteMessageForgetStore(repos.message_forget)
    await store.record(_S1, ["m-1", "m-2"], forget_id="fg_a")
    await store.record(_S1, ["m-1", "m-1"], forget_id="fg_b")  # a repeat adds once
    await store.record(_S1, ["m-1"], forget_id="fg_b")

    assert await store.forgotten(_S1) == {"m-1", "m-2"}
    assert await store.forgotten(_S2) == frozenset()  # the same id elsewhere
    assert await store.undo("fg_a") == 2
    assert await store.undo("fg_a") == 0  # stamped once
    assert await store.forgotten(_S1) == {"m-1"}  # fg_b still names it
    assert await store.undo("fg_b") == 1
    assert await store.forgotten(_S1) == frozenset()


async def test_a_segment_reads_its_sessions_forgotten_ids(repos: SqliteRepos) -> None:
    store = SqliteMessageForgetStore(repos.message_forget)
    await _segment(repos, _S1, "mc_1")
    await _segment(repos, _S2, "mc_2")
    await store.record(_S1, ["m-1"], forget_id="fg_a")

    found = await store.forgotten_in_segments(["mc_1", "mc_2", "mc_gone"])

    assert found == {"mc_1": frozenset({"m-1"}), "mc_2": frozenset()}


async def test_the_log_keeps_a_group_forget_and_finds_its_pictures(
    repos: SqliteRepos,
) -> None:
    log = SqliteForgetLogStore(repos.forget_log)
    undo = {"withdrawn": [{"kind": "image_memory", "image_id": _HASH}]}
    entry = await log.record(
        target="message",
        target_id="s1",
        app_id="app",
        project_id="proj",
        owner_id="trip",
        owner_type="group",
        requested_by="u_chen",
        undo=undo,
        forget_id="fg_fixed",
    )
    personal = await log.record(
        target="image",
        target_id="cd" * 32,
        app_id="app",
        project_id="proj",
        owner_id="u_chen",
        undo={"withdrawn": [{"image_id": "cd" * 32}]},
    )

    found = await log.get("fg_fixed")
    assert entry.forget_id == "fg_fixed"
    assert found is not None
    assert (found.owner_type, found.requested_by, found.undo) == (
        "group",
        "u_chen",
        undo,
    )
    assert await log.get("fg_missing") is None
    assert (await log.get(personal.forget_id)).requested_by is None  # type: ignore[union-attr]
    assert await log.holds_picture(_HASH)
    assert not await log.holds_picture("cd" * 32)  # not a group forget
    assert await log.mark_undone("fg_fixed")
    assert not await log.holds_picture(_HASH)


async def test_a_user_and_a_group_of_one_id_read_only_their_own_forgets(
    repos: SqliteRepos,
) -> None:
    """Users and groups share one id space: every lookup keys on the type too."""
    log = SqliteForgetLogStore(repos.forget_log)
    space = {"app_id": "app", "project_id": "proj", "owner_id": "trip"}
    mine = await log.record(target="image", target_id=_HASH, **space)
    group = await log.record(
        target="image",
        target_id=_HASH,
        owner_type="group",
        undo={"withdrawn": []},
        **space,
    )
    await log.record(target="image", target_id="cd" * 32, owner_type="group", **space)

    personal = await log.latest_live(
        target="image", target_id=_HASH, owner_type="user", **space
    )
    grouped = await log.latest_live(
        target="image", target_id=_HASH, owner_type="group", **space
    )
    assert personal is not None and personal.forget_id == mine.forget_id
    assert grouped is not None and grouped.forget_id == group.forget_id
    assert await log.live_targets(target="image", owner_type="user", **space) == [_HASH]
    assert [
        e.forget_id for e in await log.live_entries(owner_type="user", **space)
    ] == [mine.forget_id]
    assert await log.live_targets(target="image", owner_type="agent", **space) == []


_PRE_GROUP_FORGET_LOG_DDL = """
CREATE TABLE forget_log (
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL,
    forget_id VARCHAR NOT NULL,
    target_kind VARCHAR NOT NULL,
    target_id VARCHAR NOT NULL,
    app_id VARCHAR NOT NULL,
    project_id VARCHAR NOT NULL,
    owner_id VARCHAR NOT NULL,
    owner_type VARCHAR NOT NULL,
    reason VARCHAR,
    entry_ids_json VARCHAR NOT NULL,
    episode_ids_json VARCHAR NOT NULL,
    removed_profile_items_json VARCHAR NOT NULL,
    undone_at DATETIME,
    PRIMARY KEY (forget_id)
)
"""


async def test_an_older_forget_log_gains_the_group_columns(tmp_path: Path) -> None:
    database = SqliteDatabase(MemoryRoot(tmp_path), SqliteSettings())
    async with database.engine.begin() as conn:
        await conn.execute(text(_PRE_GROUP_FORGET_LOG_DDL))
        await conn.execute(
            text(
                "INSERT INTO forget_log VALUES ('2026-01-01 00:00:00', "
                "'2026-01-01 00:00:00', 'fg_old', 'image', :hash, 'app', 'proj', "
                "'u1', 'user', NULL, '[]', '[]', '[]', NULL)"
            ),
            {"hash": _HASH},
        )

    await database.create_schema()
    await database.create_schema()  # idempotent

    async with database.engine.connect() as conn:
        rows = (await conn.execute(text("PRAGMA table_info(forget_log)"))).fetchall()
    assert {"requested_by", "undo_json"} <= {row[1] for row in rows}
    log = SqliteForgetLogStore(SqliteRepos.over(database).forget_log)
    old = await log.get("fg_old")
    assert old is not None and old.requested_by is None and old.undo is None
    await database.dispose()

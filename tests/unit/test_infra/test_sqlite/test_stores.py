"""SQLite adapters of the memory state ports, against a real system DB."""

from __future__ import annotations

import datetime as dt
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

import pytest
from everalgo.types import ChatMessage, MemCell, ToolCallResult
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlmodel import SQLModel

from everos.config import SqliteSettings
from everos.core.utils.datetime import ensure_utc, get_utc_now
from everos.infra.persistence import MemoryRoot
from everos.infra.persistence.sqlite import (
    Memcell,
    SqliteClusterStore,
    SqliteMemcellStore,
    SqliteMemorizeInboxStore,
    SqliteSessionBufferStore,
    create_session_factory,
    create_system_engine,
)
from everos.infra.persistence.sqlite._repos.cluster import ClusterRepo
from everos.infra.persistence.sqlite._repos.memcell import MemcellRepo
from everos.infra.persistence.sqlite._repos.memorize_inbox import MemorizeInboxRepo
from everos.infra.persistence.sqlite._repos.session_cut import SessionCutRepo
from everos.infra.persistence.sqlite._repos.unprocessed_buffer import (
    UnprocessedBufferRepo,
)
from everos.memory.models import (
    CanonicalMessage,
    Cluster,
    MemcellRecord,
    ToolCall,
)
from everos.memory.ports import (
    ClusterStore,
    MemcellStore,
    MemorizeInboxStore,
    Scope,
    SessionBufferStore,
    SessionKey,
)

_SESSION = SessionKey(app_id="app", project_id="proj", session_id="s1")
_USER = Scope(app_id="app", project_id="proj", owner_id="u1", owner_type="user")
_FROZEN_MEMCELL = (
    Path(__file__).resolve().parents[3]
    / "fixtures/compatibility/head_0d3b502_core_0_6_0/memcell.json"
)


@dataclass
class _Stores:
    buffer: SqliteSessionBufferStore
    memcell: SqliteMemcellStore
    cluster: SqliteClusterStore
    inbox: SqliteMemorizeInboxStore
    factory: async_sessionmaker[AsyncSession]


@pytest.fixture
async def stores(tmp_path: Path) -> AsyncIterator[_Stores]:
    root = MemoryRoot(tmp_path)
    root.ensure()
    engine = create_system_engine(root.system_db, SqliteSettings())
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    factory: async_sessionmaker[AsyncSession] = create_session_factory(engine)
    stores = _Stores(
        buffer=SqliteSessionBufferStore(
            UnprocessedBufferRepo(session_factory=factory),
            SessionCutRepo(session_factory=factory),
        ),
        memcell=SqliteMemcellStore(MemcellRepo(session_factory=factory)),
        cluster=SqliteClusterStore(ClusterRepo(session_factory=factory)),
        inbox=SqliteMemorizeInboxStore(MemorizeInboxRepo(session_factory=factory)),
        factory=factory,
    )
    yield stores
    await engine.dispose()


def _message(message_id: str, minute: int, **extra: object) -> CanonicalMessage:
    return CanonicalMessage(
        message_id=message_id,
        session_id="s1",
        sender_id="u1",
        role="user",
        timestamp=dt.datetime(2026, 5, 14, 10, minute, tzinfo=dt.UTC),
        content_items=[{"type": "text", "text": f"hi {message_id}"}],
        text=f"hi {message_id}",
        **extra,  # type: ignore[arg-type]
    )


async def test_adapters_satisfy_their_ports(stores: _Stores) -> None:
    assert isinstance(stores.buffer, SessionBufferStore)
    assert isinstance(stores.memcell, MemcellStore)
    assert isinstance(stores.cluster, ClusterStore)
    assert isinstance(stores.inbox, MemorizeInboxStore)


async def test_session_buffer_round_trips_messages(stores: _Stores) -> None:
    tool_call = ToolCall(id="t1", function={"name": "f", "arguments": "{}"})
    messages = [_message("m2", 2, tool_calls=[tool_call]), _message("m1", 1)]

    await stores.buffer.keep(_SESSION, messages, last_message_ts=messages[0].timestamp)
    loaded = await stores.buffer.list(_SESSION)

    assert loaded == sorted(messages, key=lambda m: m.timestamp)
    other = SessionKey(app_id="app", project_id="other", session_id="s1")
    assert await stores.buffer.list(other) == []


async def test_session_buffer_quarantine_sets_the_buffer_aside(
    stores: _Stores,
) -> None:
    m1 = _message("m1", 1)
    await stores.buffer.keep(_SESSION, [m1], last_message_ts=m1.timestamp)

    moved = await stores.buffer.quarantine(_SESSION, [_message("m2", 2)])

    assert moved == 2
    assert await stores.buffer.list(_SESSION) == []


async def _status(stores: _Stores) -> tuple[object, object] | None:
    """``(last_message_ts, last_memcell_ts)`` of ``_SESSION``, or ``None``."""
    async with stores.factory() as s:
        row = (
            await s.execute(
                text(
                    "SELECT last_message_ts, last_memcell_ts FROM conversation_status"
                    " WHERE app_id = 'app' AND project_id = 'proj'"
                    " AND session_id = 's1'"
                )
            )
        ).first()
    return None if row is None else (row[0], row[1])


async def _memcell_ids(stores: _Stores) -> list[str]:
    async with stores.factory() as s:
        return list((await s.execute(text("SELECT memcell_id FROM memcell"))).scalars())


def _record(memcell_id: str, *messages: CanonicalMessage) -> MemcellRecord:
    cell = MemCell(
        items=[
            ChatMessage(
                id=m.message_id,
                role="user",
                content=m.text,
                sender_id="u1",
                timestamp=int(m.timestamp.timestamp() * 1000),
            )
            for m in messages
        ],
        timestamp=int(messages[-1].timestamp.timestamp() * 1000),
    )
    return MemcellRecord(
        memcell_id=memcell_id,
        cell=cell,
        raw_type="conversation",
        message_ids=[m.message_id for m in messages],
        sender_ids=["u1"],
    )


async def test_keep_buffers_messages_and_stamps_the_last_message(
    stores: _Stores,
) -> None:
    m1, m2 = _message("m1", 1), _message("m2", 2)

    await stores.buffer.keep(_SESSION, [m2, m1], last_message_ts=m2.timestamp)

    assert await stores.buffer.list(_SESSION) == [m1, m2]
    status = await _status(stores)
    assert status is not None
    assert status[0] is not None and status[1] is None


async def test_cut_records_the_cells_and_buffers_only_the_tail(
    stores: _Stores,
) -> None:
    m1, m2, m3 = _message("m1", 1), _message("m2", 2), _message("m3", 3)
    await stores.buffer.keep(_SESSION, [m1, m2, m3], last_message_ts=m3.timestamp)
    last_message = (await _status(stores) or (None, None))[0]

    await stores.buffer.cut(
        _SESSION, [_record("mc_1", m1, m2)], tail=[m3], last_memcell_ts=m2.timestamp
    )

    assert await _memcell_ids(stores) == ["mc_1"]
    assert await stores.buffer.list(_SESSION) == [m3]
    status = await _status(stores)
    assert status is not None
    assert status[0] == last_message  # cut leaves last_message_ts alone
    assert status[1] is not None


async def test_cut_keeps_the_date_of_a_memcell_from_before_2001(
    stores: _Stores,
) -> None:
    """A cell timestamp is epoch ms. Before 2001-09-09 that value is under the
    1e12 seconds/ms threshold, and reading it through the heuristic raised (year
    ~32,700); the memcell row must keep the instant instead."""
    when = dt.datetime(2000, 10, 6, 9, 30, tzinfo=dt.UTC)
    m1 = CanonicalMessage(
        message_id="m1",
        session_id="s1",
        sender_id="u1",
        role="user",
        timestamp=when,
        content_items=[{"type": "text", "text": "hi m1"}],
        text="hi m1",
    )
    await stores.buffer.keep(_SESSION, [m1], last_message_ts=when)

    await stores.buffer.cut(
        _SESSION, [_record("mc_1", m1)], tail=[], last_memcell_ts=when
    )

    async with stores.factory() as s:
        stored = (await s.execute(select(Memcell.timestamp))).scalar_one()
    assert ensure_utc(stored) == when


async def test_keep_leaves_the_last_memcell_stamp_alone(stores: _Stores) -> None:
    m1, m2 = _message("m1", 1), _message("m2", 2)
    await stores.buffer.keep(_SESSION, [m1], last_message_ts=m1.timestamp)
    await stores.buffer.cut(
        _SESSION, [_record("mc_1", m1)], tail=[], last_memcell_ts=m1.timestamp
    )
    last_memcell = (await _status(stores) or (None, None))[1]

    await stores.buffer.keep(_SESSION, [m2], last_message_ts=m2.timestamp)

    status = await _status(stores)
    assert status is not None and status[1] == last_memcell


async def test_a_cut_that_fails_midway_changes_nothing(stores: _Stores) -> None:
    """The tail repeats a message id, so its insert breaks the buffer's
    primary key after the cells and the status were written in the same
    transaction; all of it must roll back. Three separate commits leave an
    orphan memcell here (issue seen under SQLite lock timeouts in 1.6.0).
    """
    m1, m2 = _message("m1", 1), _message("m2", 2)
    await stores.buffer.keep(_SESSION, [m1, m2], last_message_ts=m2.timestamp)
    before = await _status(stores)

    with pytest.raises(IntegrityError):
        await stores.buffer.cut(
            _SESSION, [_record("mc_1", m1)], tail=[m2, m2], last_memcell_ts=m1.timestamp
        )

    assert await _memcell_ids(stores) == []
    assert await stores.buffer.list(_SESSION) == [m1, m2]
    assert await _status(stores) == before


async def test_a_keep_that_fails_midway_changes_nothing(stores: _Stores) -> None:
    m1, m2 = _message("m1", 1), _message("m2", 2)
    await stores.buffer.keep(_SESSION, [m1], last_message_ts=m1.timestamp)
    before = await _status(stores)

    with pytest.raises(IntegrityError):
        await stores.buffer.keep(_SESSION, [m1, m2, m2], last_message_ts=m2.timestamp)

    assert await stores.buffer.list(_SESSION) == [m1]
    assert await _status(stores) == before


async def test_memcells_round_trip(stores: _Stores) -> None:
    cell = MemCell(
        items=[
            ChatMessage(
                id="m1", role="user", content="hello", sender_id="u1", timestamp=1
            )
        ],
        timestamp=1_747_216_800_000,
    )
    record = MemcellRecord(
        memcell_id="mc_1",
        cell=cell,
        raw_type="conversation",
        message_ids=["m1"],
        sender_ids=["u1"],
    )

    await stores.memcell.insert_many(_SESSION, [record])

    assert await stores.memcell.find_by_ids(["mc_1", "missing"]) == [cell]


async def test_recent_in_session_reads_the_session_newest_first(
    stores: _Stores,
) -> None:
    """The session's cells up to a moment, newest first, without the one asked
    about: what a picture-only cell reads its conversation's language from."""
    m1, m2, m3, m4 = (_message(f"m{i}", i) for i in range(1, 5))
    await stores.memcell.insert_many(
        _SESSION,
        [
            _record("mc_1", m1),
            _record("mc_2", m2),
            _record("mc_3", m3),
            _record("mc_4", m4),
        ],
    )
    other = SessionKey(app_id="app", project_id="proj", session_id="s2")
    await stores.memcell.insert_many(other, [_record("mc_x", m2)])

    def ids(cells: list[MemCell]) -> list[str]:
        return [cell.items[0].id for cell in cells]

    until_m3 = int(m3.timestamp.timestamp() * 1000)
    assert ids(
        await stores.memcell.recent_in_session(
            _SESSION, until_ms=until_m3, exclude="mc_3", limit=8
        )
    ) == ["m2", "m1"]
    until_m4 = int(m4.timestamp.timestamp() * 1000)
    assert ids(
        await stores.memcell.recent_in_session(_SESSION, until_ms=until_m4, limit=2)
    ) == ["m4", "m3"]


async def test_memcell_reads_frozen_payload_from_sqlite(stores: _Stores) -> None:
    payload = _FROZEN_MEMCELL.read_text()
    async with stores.factory() as session:
        await session.execute(
            text(
                "INSERT INTO memcell "
                "(memcell_id, app_id, project_id, session_id, track, raw_type, "
                "message_ids_json, sender_ids_json, payload_json, timestamp, "
                "created_at, updated_at) VALUES "
                "('mc-frozen', 'app', 'proj', 's1', 'user_memory', 'conversation', "
                "'[]', '[]', :payload, '2023-11-14 22:13:20.456000', "
                "'2023-11-14 22:13:20.456000', '2023-11-14 22:13:20.456000')"
            ),
            {"payload": payload},
        )
        await session.commit()

    cells = await stores.memcell.find_by_ids(["mc-frozen", "missing"])

    assert len(cells) == 1
    assert cells[0].timestamp == 1_700_000_000_456
    assert len(cells[0].items) == 2
    message, tool_result = cells[0].items
    assert isinstance(message, ChatMessage)
    assert message.id == "message-frozen"
    assert message.timestamp == 1_700_000_000_123
    assert message.content == "Preserve this archived message."
    assert message.sender_id == "user-frozen"
    assert message.sender_name is None
    assert isinstance(tool_result, ToolCallResult)
    assert tool_result.tool_call_id == "tool-call-frozen"
    assert tool_result.content == "Archived tool output."
    assert tool_result.timestamp == 1_700_000_000_456


async def test_cluster_lifecycle(stores: _Stores) -> None:
    cluster = Cluster(
        id="cl_1",
        centroid=(0.5, -0.25),
        count=2,
        last_ts=100,
        preview=["first"],
        members=["ep_1", "ep_2"],
    )

    await stores.cluster.upsert(_USER, "user_scene", cluster, member_type="episode")

    assert await stores.cluster.list_for_owner(_USER, "user_scene") == [cluster]
    assert await stores.cluster.get("cl_1") == cluster
    assert await stores.cluster.get("cl_missing") is None
    assert (
        await stores.cluster.find_cluster_for_member(_USER, "episode", "ep_2") == "cl_1"
    )

    # A merge upserts the new snapshot: the row is restamped and the new
    # member appended after the ones the cluster already had.
    merged = cluster.model_copy(
        update={
            "centroid": (1.0, 0.0),
            "count": 3,
            "last_ts": 200,
            "preview": ["merged"],
            "members": ["ep_1", "ep_2", "ep_3"],
        }
    )
    await stores.cluster.upsert(_USER, "user_scene", merged, member_type="episode")

    assert await stores.cluster.get("cl_1") == merged
    assert await stores.cluster.list_for_owner(_USER, "user_scene") == [merged]


async def test_cluster_reverse_lookup_names_the_kind(stores: _Stores) -> None:
    """A store an earlier 2.0 build wrote can hold a profile cluster
    (``user_memory``) next to the scene's with the same episode; a kind picks
    one."""
    for cluster_id, kind in (("cl_p", "user_memory"), ("cl_s", "user_scene")):
        await stores.cluster.upsert(
            _USER,
            kind,  # type: ignore[arg-type]
            Cluster(
                id=cluster_id,
                centroid=(1.0, 0.0),
                count=1,
                last_ts=1,
                preview=["p"],
                members=["ep_1"],
            ),
            member_type="episode",
        )
    find = stores.cluster.find_cluster_for_member
    assert await find(_USER, "episode", "ep_1", kind="user_scene") == "cl_s"
    assert await find(_USER, "episode", "ep_1", kind="user_memory") == "cl_p"  # type: ignore[arg-type]
    assert await find(_USER, "episode", "ep_1", kind="group_scene") is None


async def test_memorize_inbox_queue_flow(stores: _Stores) -> None:
    task = await stores.inbox.enqueue(
        _SESSION,
        request_id="r1",
        is_final=False,
        defer_extraction=False,
        payload_json='{"session_id": "s1"}',
        message_count=1,
    )

    assert task.status == "queued"
    assert await stores.inbox.count_pending(_SESSION) == 1
    assert await stores.inbox.claim_next(exclude_sessions=[_SESSION]) is None
    claimed = await stores.inbox.claim_next()
    assert claimed is not None and claimed.id == task.id
    assert claimed.status == "processing"

    await stores.inbox.mark(task.id, status="success", attempt=1, payload_json="")

    done = await stores.inbox.get_task("r1")
    assert done is not None
    assert (done.status, done.attempt, done.payload_json) == ("success", 1, "")
    assert await stores.inbox.count_status("success") == 1
    assert await stores.inbox.count_pending() == 0
    assert await stores.inbox.oldest_pending_at() is None
    page = await stores.inbox.list_tasks(session=_SESSION)
    assert (page.total, [t.request_id for t in page.items]) == (1, ["r1"])
    assert await stores.inbox.requeue_processing() == (0, 0)
    assert await stores.inbox.fail_abandoned() == 0
    later = get_utc_now() + dt.timedelta(days=1)
    assert await stores.inbox.purge_expired(before=later) == (1, 0)


async def test_stamping_works_on_a_status_table_with_the_older_unique_key(
    stores: _Stores,
) -> None:
    """Databases created before ``conversation_status`` was scoped by app and
    project carry a ``(session_id, track)`` unique constraint; the schema
    upgrade adds columns but not constraints, so stamping must not rely on
    the newer four-column one."""
    async with stores.factory() as s:
        await s.execute(text("DROP TABLE conversation_status"))
        await s.execute(
            text(
                "CREATE TABLE conversation_status ("
                " id INTEGER PRIMARY KEY, created_at DATETIME, updated_at DATETIME,"
                " app_id VARCHAR DEFAULT 'default', project_id VARCHAR DEFAULT"
                " 'default', session_id VARCHAR NOT NULL, track VARCHAR NOT NULL,"
                " last_message_ts DATETIME, last_memcell_ts DATETIME,"
                " UNIQUE (session_id, track))"
            )
        )
        await s.commit()
    m1, m2 = _message("m1", 1), _message("m2", 2)

    await stores.buffer.keep(_SESSION, [m1], last_message_ts=m1.timestamp)
    await stores.buffer.keep(_SESSION, [m1, m2], last_message_ts=m2.timestamp)

    assert await stores.buffer.list(_SESSION) == [m1, m2]
    status = await _status(stores)
    assert status is not None and status[0] is not None

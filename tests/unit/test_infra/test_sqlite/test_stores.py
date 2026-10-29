"""SQLite adapters of the memory state ports, against a real system DB."""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

import pytest
from everalgo.types import ChatMessage, MemCell, ToolCallResult
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlmodel import SQLModel

from everos.config import SqliteSettings
from everos.core.utils.datetime import get_utc_now
from everos.infra.persistence import MemoryRoot
from everos.infra.persistence.sqlite import (
    Memcell,
    SqliteClusterStore,
    SqliteMemcellStore,
    SqliteMemorizeInboxStore,
    SqliteReflectionReportStore,
    SqliteSessionBufferStore,
    create_session_factory,
    create_system_engine,
)
from everos.infra.persistence.sqlite.repos.cluster import ClusterRepo
from everos.infra.persistence.sqlite.repos.conversation_status import (
    ConversationStatusRepo,
)
from everos.infra.persistence.sqlite.repos.memcell import MemcellRepo
from everos.infra.persistence.sqlite.repos.memorize_inbox import MemorizeInboxRepo
from everos.infra.persistence.sqlite.repos.reflection_report import (
    ReflectionReportRepo,
)
from everos.infra.persistence.sqlite.repos.unprocessed_buffer import (
    UnprocessedBufferRepo,
)
from everos.memory.models import (
    CanonicalMessage,
    Cluster,
    MemcellRecord,
    ReflectionReport,
    ToolCall,
)
from everos.memory.ports import (
    ClusterStore,
    MemcellStore,
    MemorizeInboxStore,
    ReflectionReportStore,
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
    report: SqliteReflectionReportStore
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
            ConversationStatusRepo(session_factory=factory),
        ),
        memcell=SqliteMemcellStore(MemcellRepo(session_factory=factory)),
        cluster=SqliteClusterStore(ClusterRepo(session_factory=factory)),
        inbox=SqliteMemorizeInboxStore(MemorizeInboxRepo(session_factory=factory)),
        report=SqliteReflectionReportStore(
            ReflectionReportRepo(session_factory=factory)
        ),
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
    assert isinstance(stores.report, ReflectionReportStore)


async def test_session_buffer_round_trips_messages(stores: _Stores) -> None:
    tool_call = ToolCall(id="t1", function={"name": "f", "arguments": "{}"})
    messages = [_message("m2", 2, tool_calls=[tool_call]), _message("m1", 1)]

    await stores.buffer.replace(_SESSION, messages)
    loaded = await stores.buffer.list(_SESSION)

    assert loaded == sorted(messages, key=lambda m: m.timestamp)
    other = SessionKey(app_id="app", project_id="other", session_id="s1")
    assert await stores.buffer.list(other) == []


async def test_session_buffer_quarantine_sets_the_buffer_aside(
    stores: _Stores,
) -> None:
    await stores.buffer.replace(_SESSION, [_message("m1", 1)])

    moved = await stores.buffer.quarantine(_SESSION, [_message("m2", 2)])

    assert moved == 2
    assert await stores.buffer.list(_SESSION) == []


async def test_session_status_touches_do_not_raise(stores: _Stores) -> None:
    now = get_utc_now()

    await stores.buffer.touch_last_message(_SESSION, now)
    await stores.buffer.touch_last_memcell(_SESSION, now)


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


async def test_memcell_case_extraction_outcome_is_stamped(stores: _Stores) -> None:
    cell = MemCell(
        items=[
            ChatMessage(
                id="m1", role="user", content="hello", sender_id="u1", timestamp=1
            )
        ],
        timestamp=1_747_216_800_000,
    )
    records = [
        MemcellRecord(
            memcell_id=memcell_id,
            cell=cell,
            raw_type="conversation",
            message_ids=["m1"],
            sender_ids=["u1"],
        )
        for memcell_id in ("mc_skip", "mc_case")
    ]
    await stores.memcell.insert_many(_SESSION, records)

    await stores.memcell.mark_case_extracted(
        "mc_skip",
        skip_reason="trajectory_not_closed",
        skip_detail={"last_item_kind": "tool_result", "note": "x" * 5000},
    )
    await stores.memcell.mark_case_extracted(
        "mc_case", skip_reason=None, skip_detail={}
    )
    await stores.memcell.mark_case_extracted(
        "missing", skip_reason=None, skip_detail={}
    )

    async with stores.factory() as session:
        rows = {
            row.memcell_id: row
            for row in (await session.execute(select(Memcell))).scalars()
        }
    skipped, produced = rows["mc_skip"], rows["mc_case"]
    assert skipped.case_skip_reason == "trajectory_not_closed"
    assert skipped.case_extracted_at is not None
    assert skipped.case_extracted_at.tzinfo is not None
    assert skipped.case_skip_detail_json is not None
    detail = json.loads(skipped.case_skip_detail_json)
    assert detail["last_item_kind"] == "tool_result"
    assert len(detail["note"]) == 512
    assert produced.case_skip_reason is None
    assert produced.case_skip_detail_json is None
    assert produced.case_extracted_at is not None
    assert set(rows) == {"mc_skip", "mc_case"}


async def test_cluster_lifecycle(stores: _Stores) -> None:
    cluster = Cluster(
        id="cl_1",
        centroid=(0.5, -0.25),
        count=2,
        last_ts=100,
        preview=["first"],
        members=["ep_1", "ep_2"],
    )

    await stores.cluster.upsert(_USER, "user_memory", cluster, member_type="episode")

    assert await stores.cluster.list_for_owner(_USER, "user_memory") == [cluster]
    assert await stores.cluster.get("cl_1") == cluster
    assert await stores.cluster.get("cl_missing") is None
    assert (
        await stores.cluster.find_cluster_for_member(_USER, "episode", "ep_2") == "cl_1"
    )
    assert await stores.cluster.list_distinct_owners() == [_USER]

    await stores.cluster.replace_members(
        "cl_1",
        removed={"ep_1"},
        added="ep_m",
        member_type="episode",
        centroid=[1.0, 0.0],
        count=1,
        last_ts=200,
        preview=["merged"],
    )

    assert await stores.cluster.members("cl_1") == [
        ("ep_2", "episode"),
        ("ep_m", "episode"),
    ]
    updated = await stores.cluster.get("cl_1")
    assert updated is not None
    assert (updated.centroid, updated.count, updated.last_ts, updated.preview) == (
        (1.0, 0.0),
        1,
        200,
        ["merged"],
    )


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


async def test_reflection_reports(stores: _Stores) -> None:
    report = ReflectionReport(
        id="rep_1",
        cluster_id="cl_1",
        owner_id="u1",
        app_id="app",
        project_id="proj",
        mode="init",
        source_members=[("ep_1", "episode"), ("ep_2", "episode")],
        merged_entry_id="ep_m",
        deprecated_fact_count=3,
    )

    await stores.report.create(report)

    assert await stores.report.reflected_cluster_ids(_USER) == {"cl_1"}
    other = Scope(app_id="app", project_id="proj", owner_id="u2", owner_type="user")
    assert await stores.report.reflected_cluster_ids(other) == set()

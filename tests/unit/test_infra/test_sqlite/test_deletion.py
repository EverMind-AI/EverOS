"""SQLite side of memory deletion: session purge and cluster removal."""

from __future__ import annotations

from everalgo.types import ChatMessage, MemCell
from sqlalchemy import func, select

from everos.core.utils.datetime import get_utc_now
from everos.infra.persistence.sqlite import (
    SqliteClusterStore,
    SqliteMemcellStore,
    SqliteSessionPurgeStore,
    session_scope,
)
from everos.infra.persistence.sqlite._tables import (
    ClusterMember,
    ConversationStatus,
    Memcell,
    MemorizeInbox,
    UnprocessedBuffer,
)
from everos.memory.models import Cluster, MemcellRecord
from everos.memory.ports import Scope, SessionKey
from tests.fakes.storage import Storage

_S1 = SessionKey(app_id="a", project_id="p", session_id="s1")


def _cell() -> MemCell:
    return MemCell(
        items=[
            ChatMessage(
                id="m1",
                role="user",
                content="hi",
                timestamp=1_700_000_000_000,
                sender_id="u1",
            )
        ],
        timestamp=1_700_000_000_000,
    )


async def _seed_session(storage: Storage, session: SessionKey, *, tag: str) -> None:
    await SqliteMemcellStore(storage.sqlite.memcell).insert_many(
        session,
        [
            MemcellRecord(
                memcell_id=f"mc_{tag}",
                cell=_cell(),
                raw_type="conversation",
                message_ids=["m1"],
                sender_ids=["u1", "a1"],
            )
        ],
    )
    async with session_scope(storage.database.session_factory) as s:
        for track in ("memorize", "quarantine"):
            s.add(
                UnprocessedBuffer(
                    message_id=f"{tag}_{track}",
                    app_id=session.app_id,
                    project_id=session.project_id,
                    session_id=session.session_id,
                    track=track,
                    sender_id="u1",
                    role="user",
                    timestamp=get_utc_now(),
                    content_items_json="[]",
                    text="hi",
                )
            )
        s.add(
            ConversationStatus(
                app_id=session.app_id,
                project_id=session.project_id,
                session_id=session.session_id,
                track="memorize",
            )
        )
        for status in ("queued", "failed", "success"):
            s.add(
                MemorizeInbox(
                    request_id=f"{tag}_{status}",
                    app_id=session.app_id,
                    project_id=session.project_id,
                    session_id=session.session_id,
                    payload_json='{"messages": ["secret"]}',
                    status=status,
                )
            )
        await s.commit()


async def _count(storage: Storage, table: type, session: SessionKey) -> int:
    async with session_scope(storage.database.session_factory) as s:
        return (
            await s.execute(
                select(func.count())
                .select_from(table)
                .where(table.app_id == session.app_id)
                .where(table.project_id == session.project_id)
                .where(table.session_id == session.session_id)
            )
        ).scalar_one()


async def _payloads(storage: Storage, session: SessionKey) -> dict[str, str]:
    async with session_scope(storage.database.session_factory) as s:
        rows = (
            await s.execute(
                select(MemorizeInbox.status, MemorizeInbox.payload_json)
                .where(MemorizeInbox.project_id == session.project_id)
                .where(MemorizeInbox.session_id == session.session_id)
            )
        ).all()
    return dict(rows)


async def test_purge_removes_memcell_buffer_status_and_blanks_failed_payload(
    storage: Storage,
) -> None:
    """A3: both buffer tracks, the status and the memcells go; a failed
    request keeps its row without its body; queued rows are untouched."""
    await storage.create_schema()
    other_project = SessionKey(app_id="a", project_id="p2", session_id="s1")
    await _seed_session(storage, _S1, tag="x")
    await _seed_session(storage, other_project, tag="y")

    await SqliteSessionPurgeStore(storage.sqlite.session_purge).purge(_S1)

    for table in (Memcell, UnprocessedBuffer, ConversationStatus):
        assert await _count(storage, table, _S1) == 0, table
        assert await _count(storage, table, other_project) > 0, table
    assert await _payloads(storage, _S1) == {
        "queued": '{"messages": ["secret"]}',
        "failed": "",
        "success": '{"messages": ["secret"]}',
    }
    assert (await _payloads(storage, other_project))["failed"] != ""


async def test_purge_returns_deleted_memcells_with_senders(storage: Storage) -> None:
    await storage.create_schema()
    await _seed_session(storage, _S1, tag="x")

    (record,) = await SqliteSessionPurgeStore(storage.sqlite.session_purge).purge(_S1)

    assert (record.memcell_id, record.sender_ids) == ("mc_x", ["u1", "a1"])
    assert record.cell == _cell()
    assert await SqliteSessionPurgeStore(storage.sqlite.session_purge).purge(_S1) == []


_ALICE = Scope(app_id="a", project_id="p", owner_id="u1", owner_type="user")


async def _seed_cluster(
    store: SqliteClusterStore, scope: Scope, cluster_id: str, members: list[str]
) -> None:
    await store.upsert(
        scope,
        "user_scene",
        Cluster(
            id=cluster_id,
            centroid=(0.25, 0.5, 0.75),
            count=len(members),
            last_ts=1,
            preview=["old"],
            members=members,
        ),
        member_type="episode",
    )


async def test_delete_cluster_removes_members_first(storage: Storage) -> None:
    await storage.create_schema()
    store = SqliteClusterStore(storage.sqlite.cluster)
    await _seed_cluster(store, _ALICE, "cl_1", ["e1", "e2"])

    await store.delete("cl_1")

    assert await store.get("cl_1") is None
    async with session_scope(storage.database.session_factory) as s:
        assert (
            await s.execute(select(func.count()).select_from(ClusterMember))
        ).scalar_one() == 0

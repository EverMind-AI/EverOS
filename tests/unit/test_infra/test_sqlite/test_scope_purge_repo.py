"""Deleting one purge scope's rows from ``system.db``, and nothing else."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy import func, select

from everos.core.purge_marks import PurgeScope
from everos.infra.persistence.sqlite import session_scope
from everos.infra.persistence.sqlite._tables import (
    Cluster,
    ClusterMember,
    ConversationStatus,
    KnowledgeDocumentRow,
    KnowledgeTopicRow,
    MdChangeState,
    Memcell,
    MemorizeInbox,
    ReflectionReport,
    UnprocessedBuffer,
)
from tests.fakes.storage import Storage

NOW = datetime(2026, 11, 11, tzinfo=UTC)
SCOPES = [("a1", "p1"), ("a1", "p2"), ("a2", "p1"), ("default", "default")]
TABLES = (
    "memcell",
    "unprocessed_buffer",
    "conversation_status",
    "cluster",
    "cluster_member",
    "reflection_report",
    "knowledge_documents",
    "knowledge_topics",
)


def _project(app: str, project: str) -> PurgeScope:
    return PurgeScope("project", app, project, app, project)


def _rows(app: str, project: str) -> list[object]:
    tag = f"{app}_{project}"
    where = {"app_id": app, "project_id": project}
    return [
        Memcell(
            memcell_id=f"mc_{tag}",
            session_id="s1",
            track="user_memory",
            raw_type="conversation",
            message_ids_json="[]",
            sender_ids_json="[]",
            payload_json="{}",
            timestamp=NOW,
            **where,
        ),
        UnprocessedBuffer(
            message_id=f"m_{tag}",
            session_id="s1",
            track="chat",
            sender_id="u1",
            role="user",
            timestamp=NOW,
            content_items_json="[]",
            text="t",
            **where,
        ),
        ConversationStatus(session_id="s1", track="chat", **where),
        Cluster(
            cluster_id=f"cl_{tag}",
            owner_id="u1",
            owner_type="user",
            kind="user_memory",
            centroid_blob=b"",
            count=1,
            last_ts_ms=0,
            preview_json="[]",
            **where,
        ),
        ClusterMember(
            cluster_id=f"cl_{tag}",
            member_id=f"ep_{tag}",
            member_type="episode",
            added_ts=NOW,
        ),
        ReflectionReport(
            id=f"rr_{tag}",
            cluster_id=f"cl_{tag}",
            owner_id="u1",
            mode="merge",
            source_members="[]",
            source_count=1,
            merged_entry_id="e",
            deprecated_fact_count=0,
            **where,
        ),
        KnowledgeDocumentRow(
            doc_id=f"d_{tag}",
            category_id="c",
            title="t",
            summary="s",
            md_path=f"{app}/{project}/knowledge/x",
            **where,
        ),
        KnowledgeTopicRow(
            node_id=f"d_{tag}_1",
            doc_id=f"d_{tag}",
            category_id="c",
            topic_index=1,
            topic_name="n",
            topic_path="n",
            depth=1,
            summary="s",
            content="c",
            md_path=f"{app}/{project}/knowledge/y",
            **where,
        ),
    ]


async def _seed(storage: Storage) -> None:
    await storage.create_schema()
    async with session_scope(storage.database.session_factory) as s:
        for app, project in SCOPES:
            for row in _rows(app, project):
                s.add(row)
        await s.commit()


async def _count(storage: Storage, model: object, **where: str) -> int:
    async with session_scope(storage.database.session_factory) as s:
        stmt = select(func.count()).select_from(model)  # type: ignore[arg-type]
        for column, value in where.items():
            stmt = stmt.where(getattr(model, column) == value)
        return int(await s.scalar(stmt) or 0)


# purged scope -> scopes whose rows must be gone (hand-written)
GONE = {
    "project": {("a1", "p1")},
    "app": {("a1", "p1"), ("a1", "p2")},
    "space": set(SCOPES),
}


@pytest.mark.parametrize("kind", ["project", "app", "space"])
async def test_delete_scope_matrix(storage: Storage, kind: str) -> None:
    await _seed(storage)
    scope = {
        "project": _project("a1", "p1"),
        "app": PurgeScope("app", "a1", None, "a1"),
        "space": PurgeScope("space"),
    }[kind]
    repo = storage.sqlite.scope_purge
    assert await repo.count(scope) == {t: len(GONE[kind]) for t in TABLES}
    deleted = await repo.delete(scope)
    assert deleted == {t: len(GONE[kind]) for t in TABLES}
    for app, project in SCOPES:
        left = await repo.count(_project(app, project))
        expected = 0 if (app, project) in GONE[kind] else 1
        assert left == {t: expected for t in TABLES}, (app, project)


async def test_cluster_member_follows_cluster(storage: Storage) -> None:
    await _seed(storage)
    await storage.sqlite.scope_purge.delete(_project("a1", "p1"))
    assert await _count(storage, ClusterMember, cluster_id="cl_a1_p1") == 0
    assert await _count(storage, ClusterMember, cluster_id="cl_a1_p2") == 1


LATER = datetime(2999, 1, 1, tzinfo=UTC)
"""A purge confirmed after every row in these tests."""


def _inbox(
    request_id: str, app: str, project: str, status: str, **kw: object
) -> MemorizeInbox:
    return MemorizeInbox(
        request_id=request_id,
        app_id=app,
        project_id=project,
        session_id="s1",
        payload_json='{"m": 1}',
        status=status,
        **kw,
    )


async def _seed_inbox(storage: Storage, rows: list[MemorizeInbox]) -> None:
    await storage.create_schema()
    async with session_scope(storage.database.session_factory) as s:
        s.add_all(rows)
        await s.commit()


async def _inbox_rows(storage: Storage) -> dict[str, tuple[str, str, str | None]]:
    async with session_scope(storage.database.session_factory) as s:
        rows = (await s.execute(select(MemorizeInbox))).scalars().all()
        return {r.request_id: (r.status, r.payload_json, r.last_error) for r in rows}


async def test_discard_queued_inbox(storage: Storage) -> None:
    await _seed_inbox(
        storage,
        [
            _inbox("q_s", "a1", "p1", "queued"),
            _inbox("p_s", "a1", "p1", "processing"),
            _inbox("q_o", "a1", "p2", "queued"),
        ],
    )
    n = await storage.sqlite.scope_purge.discard_queued_inbox(
        _project("a1", "p1"), job_id="j1", before=LATER
    )
    rows = await _inbox_rows(storage)
    assert n == 1
    status, payload, error = rows["q_s"]
    assert (status, payload) == ("failed", "")
    assert error is not None and error.startswith("purge_in_progress: ")
    assert "j1" in error
    assert rows["p_s"] == ("processing", '{"m": 1}', None)
    assert rows["q_o"] == ("queued", '{"m": 1}', None)


async def test_inbox_history_keeps_own_job_rows(storage: Storage) -> None:
    await _seed_inbox(
        storage,
        [
            _inbox("ok_s", "a1", "p1", "success"),
            _inbox("bad_s", "a1", "p1", "failed", last_error="LLMCallError: x"),
            _inbox("q_s", "a1", "p1", "queued"),
            _inbox("ok_o", "a1", "p2", "success"),
            # Dropped at delivery by an earlier purge: /tasks must still say so.
            _inbox(
                "dropped",
                "a1",
                "p1",
                "failed",
                last_error="purge_in_progress: discarded by purge; do not resend",
            ),
        ],
    )
    repo = storage.sqlite.scope_purge
    await repo.discard_queued_inbox(_project("a1", "p1"), job_id="j1", before=LATER)
    assert await repo.delete_inbox_history(_project("a1", "p1"), before=LATER) == 2
    assert set(await _inbox_rows(storage)) == {"q_s", "ok_o", "dropped"}


async def test_inbox_rows_taken_in_after_the_purge_are_left_alone(
    storage: Storage,
) -> None:
    """Accepted after ``T``: new writes, delivered and kept as usual."""
    t = datetime(2026, 11, 12, 10, 0, tzinfo=UTC)
    after = datetime(2026, 11, 12, 10, 0, 1, tzinfo=UTC)
    await _seed_inbox(
        storage,
        [
            _inbox("q_old", "a1", "p1", "queued", created_at=t.replace(hour=9)),
            _inbox("q_new", "a1", "p1", "queued", created_at=after),
            _inbox("ok_new", "a1", "p1", "success", created_at=after),
        ],
    )
    repo = storage.sqlite.scope_purge
    assert (
        await repo.discard_queued_inbox(_project("a1", "p1"), job_id="j1", before=t)
        == 1
    )
    assert await repo.delete_inbox_history(_project("a1", "p1"), before=t) == 0
    rows = await _inbox_rows(storage)
    assert rows["q_old"][0] == "failed"
    assert rows["q_new"] == ("queued", '{"m": 1}', None)
    assert rows["ok_new"][0] == "success"


def _state(path: str) -> MdChangeState:
    return MdChangeState(md_path=path, kind="episode", change_type="added", lsn=1)


@pytest.mark.parametrize(
    ("scope", "kept"),
    [
        (
            PurgeScope("app", "a_1", None, "a_1"),
            {"a11/p1/x.md", "a1/p1/x.md", "a1/p10/x.md"},
        ),
        (_project("a1", "p1"), {"a_1/p1/x.md", "a11/p1/x.md", "a1/p10/x.md"}),
        (PurgeScope("space"), set()),
    ],
)
async def test_change_state_prefix(
    storage: Storage, scope: PurgeScope, kept: set[str]
) -> None:
    """``a_1`` must not match ``a11`` (no LIKE) and ``a1/p1`` not ``a1/p10``."""
    await storage.create_schema()
    paths = ["a_1/p1/x.md", "a11/p1/x.md", "a1/p1/x.md", "a1/p10/x.md"]
    async with session_scope(storage.database.session_factory) as s:
        s.add_all([_state(p) for p in paths])
        await s.commit()
    await storage.sqlite.scope_purge.delete_change_state(scope)
    async with session_scope(storage.database.session_factory) as s:
        left = set((await s.execute(select(MdChangeState.md_path))).scalars())
    assert left == kept

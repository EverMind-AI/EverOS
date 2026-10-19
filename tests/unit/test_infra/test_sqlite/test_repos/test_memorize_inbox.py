"""Tests for :class:`_MemorizeInboxRepo` — the async memorize queue table.

Pins the queue contract the worker relies on:

- **FIFO by id**: ``claim_next`` hands out the smallest-id ``queued`` row
  and flips it to ``processing`` so a second claim skips it.
- **Pending = queued + processing**: terminal rows never count toward the
  per-session / global caps.
- **Crash recovery**: ``requeue_processing`` returns stale ``processing``
  rows to ``queued`` at startup.
- **Nothing lives forever**: ``purge_expired`` deletes rows older than the
  retention window regardless of status, except rows still ``processing``.
- **request_id is unique**: one HTTP request maps to at most one row.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy.exc import IntegrityError
from sqlmodel import SQLModel

from everos.component.utils.datetime import get_utc_now
from everos.config import SqliteSettings
from everos.core.persistence import (
    MemoryRoot,
    create_session_factory,
    create_system_engine,
)
from everos.infra.persistence.sqlite.repos.memorize_inbox import _MemorizeInboxRepo
from everos.infra.persistence.sqlite.tables import MemorizeInbox


@pytest.fixture
async def repo(tmp_path: Path) -> _MemorizeInboxRepo:
    mr = MemoryRoot(tmp_path)
    mr.ensure()
    engine = create_system_engine(mr.system_db, SqliteSettings())
    factory = create_session_factory(engine)
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    return _MemorizeInboxRepo(session_factory=factory)


_SEQ = iter(range(1, 10_000))


def _row(session_id: str = "s1", **overrides: object) -> MemorizeInbox:
    fields: dict[str, object] = {
        "request_id": f"req{next(_SEQ):028d}",
        "session_id": session_id,
        "payload_json": "{}",
        "message_count": 1,
    }
    fields.update(overrides)
    return MemorizeInbox(**fields)  # type: ignore[arg-type]


async def test_claim_next_returns_rows_in_insertion_order(
    repo: _MemorizeInboxRepo,
) -> None:
    first = await repo.add(_row(session_id="a"))
    second = await repo.add(_row(session_id="b"))
    third = await repo.add(_row(session_id="a"))

    claimed = [await repo.claim_next() for _ in range(3)]

    assert [r.id for r in claimed if r is not None] == [first.id, second.id, third.id]
    assert await repo.claim_next() is None


async def test_claim_next_flips_row_to_processing(repo: _MemorizeInboxRepo) -> None:
    row = await repo.add(_row())

    claimed = await repo.claim_next()

    assert claimed is not None and claimed.id == row.id
    assert claimed.status == "processing"
    stored = await repo.get_by_id(row.id)
    assert stored is not None and stored.status == "processing"


async def test_count_pending_counts_queued_and_processing_only(
    repo: _MemorizeInboxRepo,
) -> None:
    await repo.add(_row(session_id="s1"))
    await repo.add(_row(session_id="s1", status="processing"))
    await repo.add(_row(session_id="s1", status="success"))
    await repo.add(_row(session_id="s1", status="failed"))
    await repo.add(_row(session_id="other"))

    assert await repo.count_pending(session_id="s1") == 2
    assert await repo.count_pending() == 3


async def test_mark_updates_status_attempt_and_error(
    repo: _MemorizeInboxRepo,
) -> None:
    row = await repo.add(_row())
    assert row.id is not None

    await repo.mark(row.id, status="failed", attempt=3, last_error="boom")

    stored = await repo.get_by_id(row.id)
    assert stored is not None
    assert (stored.status, stored.attempt, stored.last_error) == ("failed", 3, "boom")
    assert stored.payload_json == "{}"  # untouched unless asked

    await repo.mark(row.id, status="success", payload_json="")
    stored = await repo.get_by_id(row.id)
    assert stored is not None and stored.payload_json == ""


async def test_requeue_processing_returns_stale_rows_to_queued(
    repo: _MemorizeInboxRepo,
) -> None:
    await repo.add(_row(status="processing"))
    await repo.add(_row(status="processing"))
    await repo.add(_row(status="success"))

    assert await repo.requeue_processing() == 2
    assert await repo.count_pending(session_id="s1") == 2
    assert await repo.claim_next() is not None


async def test_purge_expired_deletes_old_rows_except_processing(
    repo: _MemorizeInboxRepo,
) -> None:
    old = get_utc_now() - timedelta(hours=48)
    await repo.add(_row(status="success", created_at=old))
    await repo.add(_row(status="failed", created_at=old))
    await repo.add(_row(status="queued", created_at=old))
    kept_processing = await repo.add(_row(status="processing", created_at=old))
    kept_fresh = await repo.add(_row(status="success"))

    deleted, undelivered = await repo.purge_expired(
        before=get_utc_now() - timedelta(hours=24)
    )

    assert (deleted, undelivered) == (3, 1)
    remaining = {r.id for r in await repo.list_all()}
    assert remaining == {kept_processing.id, kept_fresh.id}


async def test_request_id_is_unique(repo: _MemorizeInboxRepo) -> None:
    await repo.add(_row(request_id="dup"))
    with pytest.raises(IntegrityError):
        await repo.add(_row(request_id="dup"))


async def test_fail_abandoned_marks_every_processing_row(
    repo: _MemorizeInboxRepo,
) -> None:
    """Called only when no row is in flight: every processing row lost its write."""
    lost = await repo.add(_row(status="processing"))
    await repo.add(_row(status="queued"))
    await repo.add(_row(status="success"))

    assert await repo.fail_abandoned() == 1

    rows = {r.id: (r.status, r.last_error) for r in await repo.list_all()}
    assert rows[lost.id] == ("failed", "status write lost; outcome unknown")
    assert await repo.count_pending() == 1  # the queued row is untouched


async def test_session_summary_reports_pending_oldest_and_failed(
    repo: _MemorizeInboxRepo,
) -> None:
    oldest = datetime(2026, 1, 1, tzinfo=UTC)
    await repo.add(_row(session_id="s1", created_at=oldest))
    await repo.add(_row(session_id="s1", status="processing"))
    await repo.add(_row(session_id="s1", status="failed"))
    await repo.add(_row(session_id="s1", status="success"))
    await repo.add(_row(session_id="other"))

    summary = await repo.session_summary(session_id="s1")

    assert summary.pending == 2
    assert summary.failed == 1
    assert summary.oldest_pending_created_at == oldest

    empty = await repo.session_summary(session_id="nobody")
    assert (empty.pending, empty.failed, empty.oldest_pending_created_at) == (
        0,
        0,
        None,
    )

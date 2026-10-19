"""Tests for :mod:`everos.service.memorize_queue` — the ``async_mode`` worker.

Runs against a real SQLite inbox on ``tmp_path`` (the repo singleton and
the buffer repo singleton are swapped for temp-bound instances) and a fake
``memorize`` so no LLM is involved. Pins:

- ``enqueue`` persists the row and reports the session's pending depth.
- the caps: 413 for an oversized body, 429 per session and globally.
- delivery order is insertion order, across sessions, and each row lands
  in ``success`` with the flags (``is_final`` / ``defer_extraction``) it
  was accepted with.
- a row that keeps failing is retried ``max_attempts`` times, then marked
  ``failed`` and skipped so the rows behind it still deliver.
- ``start`` recovers ``processing`` rows left by a crash, purges expired
  rows, and ``enqueue`` wakes the running loop without waiting for a scan.
- the two ``GET /status`` readers map rows to the public shape.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from sqlmodel import SQLModel

from everos.component.utils.datetime import get_utc_now
from everos.config import SqliteSettings
from everos.config.settings import AsyncQueueSettings
from everos.core.errors import PayloadTooLargeError, QueueSaturatedError
from everos.core.persistence import (
    MemoryRoot,
    create_session_factory,
    create_system_engine,
)
from everos.infra.persistence.sqlite.repos.memorize_inbox import _MemorizeInboxRepo
from everos.infra.persistence.sqlite.repos.unprocessed_buffer import (
    _UnprocessedBufferRepo,
)
from everos.infra.persistence.sqlite.tables import MemorizeInbox, UnprocessedBuffer
from everos.service import memorize_queue as mq
from everos.service.memorize import MemorizeResult


@pytest.fixture
async def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _MemorizeInboxRepo:
    mr = MemoryRoot(tmp_path)
    mr.ensure()
    engine = create_system_engine(mr.system_db, SqliteSettings())
    factory = create_session_factory(engine)
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    inbox = _MemorizeInboxRepo(session_factory=factory)
    monkeypatch.setattr(mq, "memorize_inbox_repo", inbox)
    monkeypatch.setattr(
        mq, "unprocessed_buffer_repo", _UnprocessedBufferRepo(session_factory=factory)
    )
    return inbox


def _settings(**overrides: Any) -> AsyncQueueSettings:
    base: dict[str, Any] = {"retry_backoff_seconds": 0.0, "scan_interval_seconds": 60.0}
    base.update(overrides)
    return AsyncQueueSettings(**base)


def _payload(session_id: str = "s1", n_messages: int = 1) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "app_id": "default",
        "project_id": "default",
        "messages": [{"content": f"m{i}"} for i in range(n_messages)],
    }


class _FakeMemorize:
    """Records calls; raises for sessions listed in ``failing``."""

    def __init__(self, failing: set[str] | None = None) -> None:
        self.calls: list[tuple[str, bool, bool]] = []
        self.failing = failing or set()
        self.called = asyncio.Event()

    async def __call__(
        self,
        payload: dict[str, Any],
        *,
        is_final: bool = False,
        defer_extraction: bool = False,
    ) -> MemorizeResult:
        self.calls.append((payload["session_id"], is_final, defer_extraction))
        self.called.set()
        if payload["session_id"] in self.failing:
            raise RuntimeError("boom")
        return MemorizeResult(
            message_count=len(payload["messages"]), status="extracted"
        )


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> _FakeMemorize:
    f = _FakeMemorize()
    monkeypatch.setattr(mq, "memorize", f)
    return f


# ── enqueue ────────────────────────────────────────────────────────────────


async def test_enqueue_persists_row_and_reports_pending_depth(
    repo: _MemorizeInboxRepo, fake: _FakeMemorize
) -> None:
    w = mq.MemorizeQueueWorker(_settings())

    a1 = await w.enqueue(
        _payload(n_messages=3), request_id="r1", is_final=False, defer_extraction=True
    )
    a2 = await w.enqueue(
        _payload(), request_id="r2", is_final=True, defer_extraction=False
    )

    assert (a1, a2) == (1, 2)  # pending depth of s1, this row included
    row = await repo.find_one(request_id="r1")
    assert row is not None
    assert (row.status, row.is_final, row.defer_extraction) == ("queued", False, True)
    assert fake.calls == []  # enqueue never runs memorize inline


async def test_enqueue_rejects_oversized_payload(
    repo: _MemorizeInboxRepo, fake: _FakeMemorize
) -> None:
    w = mq.MemorizeQueueWorker(_settings(max_payload_bytes=64))

    with pytest.raises(PayloadTooLargeError):
        await w.enqueue(
            _payload(n_messages=20),
            request_id="r1",
            is_final=False,
            defer_extraction=False,
        )
    assert await repo.count_pending() == 0


async def test_enqueue_rejects_when_session_cap_reached(
    repo: _MemorizeInboxRepo, fake: _FakeMemorize
) -> None:
    w = mq.MemorizeQueueWorker(_settings(max_pending_per_session=1))
    await w.enqueue(
        _payload("s1"), request_id="r1", is_final=False, defer_extraction=False
    )

    with pytest.raises(QueueSaturatedError):
        await w.enqueue(
            _payload("s1"), request_id="r2", is_final=False, defer_extraction=False
        )
    # Another session is unaffected by s1's cap.
    await w.enqueue(
        _payload("s2"), request_id="r3", is_final=False, defer_extraction=False
    )


async def test_enqueue_rejects_when_global_cap_reached(
    repo: _MemorizeInboxRepo, fake: _FakeMemorize
) -> None:
    w = mq.MemorizeQueueWorker(_settings(max_pending_total=1))
    await w.enqueue(
        _payload("s1"), request_id="r1", is_final=False, defer_extraction=False
    )

    with pytest.raises(QueueSaturatedError):
        await w.enqueue(
            _payload("s2"), request_id="r2", is_final=False, defer_extraction=False
        )


async def test_has_pending_reflects_queued_rows(
    repo: _MemorizeInboxRepo, fake: _FakeMemorize
) -> None:
    w = mq.MemorizeQueueWorker(_settings())
    assert await w.has_pending(session_id="s1") is False

    await w.enqueue(
        _payload("s1"), request_id="r1", is_final=False, defer_extraction=False
    )

    assert await w.has_pending(session_id="s1") is True
    assert await w.has_pending(session_id="s2") is False


# ── delivery ───────────────────────────────────────────────────────────────


async def test_drain_delivers_in_insertion_order_with_row_flags(
    repo: _MemorizeInboxRepo, fake: _FakeMemorize
) -> None:
    w = mq.MemorizeQueueWorker(_settings())
    await w.enqueue(
        _payload("a"), request_id="r1", is_final=False, defer_extraction=True
    )
    await w.enqueue(
        _payload("b"), request_id="r2", is_final=False, defer_extraction=False
    )
    await w.enqueue(
        _payload("a"), request_id="r3", is_final=True, defer_extraction=False
    )

    delivered = await w.drain_once()

    assert delivered == 3
    assert fake.calls == [("a", False, True), ("b", False, False), ("a", True, False)]
    statuses = {r.request_id: (r.status, r.attempt) for r in await repo.list_all()}
    assert statuses == {
        "r1": ("success", 1),
        "r2": ("success", 1),
        "r3": ("success", 1),
    }


async def test_failing_row_is_retried_then_marked_failed_and_skipped(
    repo: _MemorizeInboxRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeMemorize(failing={"bad"})
    monkeypatch.setattr(mq, "memorize", fake)
    w = mq.MemorizeQueueWorker(_settings(max_attempts=3))
    await w.enqueue(
        _payload("bad"), request_id="r1", is_final=False, defer_extraction=False
    )
    await w.enqueue(
        _payload("good"), request_id="r2", is_final=False, defer_extraction=False
    )

    await w.drain_once()

    assert [c[0] for c in fake.calls] == ["bad", "bad", "bad", "good"]
    bad = await repo.find_one(request_id="r1")
    assert bad is not None
    assert (bad.status, bad.attempt) == ("failed", 3)
    assert bad.last_error is not None and "boom" in bad.last_error
    good = await repo.find_one(request_id="r2")
    assert good is not None and good.status == "success"
    assert await repo.count_pending() == 0


# ── lifecycle ──────────────────────────────────────────────────────────────


async def test_start_requeues_processing_rows_and_delivers_them(
    repo: _MemorizeInboxRepo, fake: _FakeMemorize
) -> None:
    await repo.add(
        MemorizeInbox(
            request_id="r1",
            session_id="s1",
            payload_json='{"session_id": "s1", "messages": []}',
            status="processing",
        )
    )
    w = mq.MemorizeQueueWorker(_settings())

    await w.start()
    try:
        await asyncio.wait_for(fake.called.wait(), timeout=2)
        await _wait_until_status(repo, "r1", "success")
    finally:
        await w.stop()

    assert [c[0] for c in fake.calls] == ["s1"]


async def test_enqueue_wakes_running_worker_without_scan(
    repo: _MemorizeInboxRepo, fake: _FakeMemorize
) -> None:
    w = mq.MemorizeQueueWorker(_settings(scan_interval_seconds=600.0))
    await w.start()
    try:
        await w.enqueue(
            _payload("s1"), request_id="r1", is_final=False, defer_extraction=False
        )
        await asyncio.wait_for(fake.called.wait(), timeout=2)
        await _wait_until_status(repo, "r1", "success")
    finally:
        await w.stop()


async def test_start_purges_rows_past_retention(
    repo: _MemorizeInboxRepo, fake: _FakeMemorize
) -> None:
    old = get_utc_now() - timedelta(hours=48)
    await repo.add(
        MemorizeInbox(
            request_id="old",
            session_id="s1",
            payload_json="{}",
            status="success",
            created_at=old,
        )
    )
    await repo.add(
        MemorizeInbox(
            request_id="fresh", session_id="s1", payload_json="{}", status="success"
        )
    )
    w = mq.MemorizeQueueWorker(_settings(row_retention_hours=24.0))

    await w.start()
    await w.stop()

    assert await repo.find_one(request_id="old") is None
    assert await repo.find_one(request_id="fresh") is not None


async def test_stop_is_idempotent_and_safe_before_start(
    repo: _MemorizeInboxRepo, fake: _FakeMemorize
) -> None:
    w = mq.MemorizeQueueWorker(_settings())
    await w.stop()
    await w.start()
    await w.stop()
    await w.stop()


# ── status readers ─────────────────────────────────────────────────────────


async def test_get_request_status_maps_row_and_none_when_missing(
    repo: _MemorizeInboxRepo, fake: _FakeMemorize
) -> None:
    await repo.add(
        MemorizeInbox(
            request_id="r1",
            session_id="s1",
            payload_json="{}",
            message_count=4,
            status="failed",
            attempt=3,
            last_error="RuntimeError: boom",
        )
    )

    st = await mq.get_request_status("r1")

    assert st is not None
    assert (st.status, st.session_id, st.message_count, st.attempt) == (
        "failed",
        "s1",
        4,
        3,
    )
    assert st.last_error == "RuntimeError: boom"
    assert st.created_at.tzinfo is not None
    assert await mq.get_request_status("nope") is None


async def test_get_session_status_combines_inbox_and_buffer(
    repo: _MemorizeInboxRepo, fake: _FakeMemorize
) -> None:
    oldest = get_utc_now() - timedelta(seconds=30)
    await repo.add(
        MemorizeInbox(
            request_id="r1", session_id="s1", payload_json="{}", created_at=oldest
        )
    )
    await repo.add(
        MemorizeInbox(
            request_id="r2", session_id="s1", payload_json="{}", status="failed"
        )
    )
    await mq.unprocessed_buffer_repo.replace(
        "s1",
        "memorize",
        [
            UnprocessedBuffer(
                message_id="m1",
                session_id="s1",
                track="memorize",
                sender_id="u",
                role="user",
                timestamp=datetime(2026, 1, 1, tzinfo=UTC),
                content_items_json="[]",
                text="hi",
            )
        ],
    )

    st = await mq.get_session_status(session_id="s1")

    assert (st.pending, st.failed, st.buffered) == (1, 1, 1)
    assert st.oldest_pending_age_ms is not None and st.oldest_pending_age_ms >= 29_000

    empty = await mq.get_session_status(session_id="nobody")
    assert (
        empty.pending,
        empty.failed,
        empty.buffered,
        empty.oldest_pending_age_ms,
    ) == (
        0,
        0,
        0,
        None,
    )


async def _wait_until_status(
    repo: _MemorizeInboxRepo, request_id: str, status: str
) -> None:
    async with asyncio.timeout(2.0):
        while True:
            row = await repo.find_one(request_id=request_id)
            if row is not None and row.status == status:
                return
            await asyncio.sleep(0.01)


# ── graceful stop ──────────────────────────────────────────────────────────


class _BlockingMemorize(_FakeMemorize):
    """Blocks inside memorize() until ``release`` is set."""

    def __init__(self) -> None:
        super().__init__()
        self.release = asyncio.Event()

    async def __call__(
        self,
        payload: dict[str, Any],
        *,
        is_final: bool = False,
        defer_extraction: bool = False,
    ) -> MemorizeResult:
        self.called.set()
        await self.release.wait()
        return await super().__call__(
            payload, is_final=is_final, defer_extraction=defer_extraction
        )


async def test_stop_waits_for_the_row_in_flight(
    repo: _MemorizeInboxRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A normal shutdown must not leave a half-delivered row for a replay."""
    fake = _BlockingMemorize()
    monkeypatch.setattr(mq, "memorize", fake)
    w = mq.MemorizeQueueWorker(_settings())
    await w.start()
    await w.enqueue(
        _payload("s1"), request_id="r1", is_final=False, defer_extraction=False
    )
    await asyncio.wait_for(fake.called.wait(), timeout=2)

    stopper = asyncio.create_task(w.stop())
    await asyncio.sleep(0.05)
    assert not stopper.done()  # still waiting on the row
    fake.release.set()
    await asyncio.wait_for(stopper, timeout=2)

    row = await repo.find_one(request_id="r1")
    assert row is not None and row.status == "success"


async def test_stop_cancels_after_the_grace_period(
    repo: _MemorizeInboxRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _BlockingMemorize()  # never released
    monkeypatch.setattr(mq, "memorize", fake)
    monkeypatch.setattr(mq, "_STOP_GRACE_SECONDS", 0.05)
    w = mq.MemorizeQueueWorker(_settings())
    await w.start()
    await w.enqueue(
        _payload("s1"), request_id="r1", is_final=False, defer_extraction=False
    )
    await asyncio.wait_for(fake.called.wait(), timeout=2)

    await asyncio.wait_for(w.stop(), timeout=2)

    row = await repo.find_one(request_id="r1")
    assert row is not None and row.status == "processing"  # recovered on next start


# ── round-3 additions: sweep, status-write retry, accept switch, surrogates ──


async def test_enqueue_rejects_lone_surrogate_with_422_class(
    repo: _MemorizeInboxRepo, fake: _FakeMemorize
) -> None:
    from everos.core.errors import InvalidInputError

    w = mq.MemorizeQueueWorker(_settings())
    payload = _payload()
    payload["messages"][0]["content"] = "bad \ud800 text"

    with pytest.raises(InvalidInputError):
        await w.enqueue(
            payload, request_id="r1", is_final=False, defer_extraction=False
        )
    assert await repo.count_pending() == 0


async def test_sweep_fails_processing_rows_nobody_is_working_on(
    repo: _MemorizeInboxRepo, fake: _FakeMemorize
) -> None:
    """A row left ``processing`` by a lost status write must not stay pending."""
    stuck = await repo.add(
        MemorizeInbox(
            request_id="stuck", session_id="s1", payload_json="{}", status="processing"
        )
    )
    w = mq.MemorizeQueueWorker(_settings())

    await w._sweep()

    row = await repo.find_one(request_id="stuck")
    assert row is not None and row.id == stuck.id
    assert row.status == "failed"
    assert row.last_error == "status write lost; outcome unknown"
    assert await w.has_pending(session_id="s1") is False


async def test_status_write_is_retried_before_giving_up(
    repo: _MemorizeInboxRepo, fake: _FakeMemorize, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mq, "_MARK_RETRY_SECONDS", 0.0)
    real_mark = repo.mark
    failures = {"left": 2}

    async def flaky_mark(row_id: int, **values: Any) -> None:
        if failures["left"]:
            failures["left"] -= 1
            raise RuntimeError("database is locked")
        await real_mark(row_id, **values)

    monkeypatch.setattr(repo, "mark", flaky_mark)
    w = mq.MemorizeQueueWorker(_settings())
    await w.enqueue(
        _payload("s1"), request_id="r1", is_final=False, defer_extraction=False
    )

    await w.drain_once()

    row = await repo.find_one(request_id="r1")
    assert row is not None and row.status == "success"
    assert failures["left"] == 0


async def test_worker_not_accepting_async_still_delivers_backlog(
    repo: _MemorizeInboxRepo, fake: _FakeMemorize
) -> None:
    """``enabled = false`` only closes the door; rows already inside still run."""
    await repo.add(
        MemorizeInbox(
            request_id="old",
            session_id="s1",
            payload_json='{"session_id": "s1", "messages": []}',
        )
    )
    w = mq.MemorizeQueueWorker(_settings(), accept_async=False)
    assert w.accepting is False

    await w.start()
    try:
        await _wait_until_status(repo, "old", "success")
    finally:
        await w.stop()


# ── round-3 review follow-ups: body dropped on success, lost write reaped ────


async def test_delivered_row_drops_its_request_body(
    repo: _MemorizeInboxRepo, fake: _FakeMemorize
) -> None:
    """Terminal rows must not hold seven days of request bodies on disk."""
    w = mq.MemorizeQueueWorker(_settings())
    await w.enqueue(
        _payload("s1", n_messages=3),
        request_id="r1",
        is_final=False,
        defer_extraction=False,
    )

    await w.drain_once()

    row = await repo.find_one(request_id="r1")
    assert row is not None and row.status == "success"
    assert row.payload_json == ""
    assert row.message_count == 3  # the count survives for /status


async def test_failed_row_keeps_its_body_and_error_in_one_write(
    repo: _MemorizeInboxRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeMemorize(failing={"bad"})
    monkeypatch.setattr(mq, "memorize", fake)
    writes: list[dict[str, Any]] = []
    real_mark = repo.mark

    async def spy_mark(row_id: int, **values: Any) -> None:
        writes.append(values)
        await real_mark(row_id, **values)

    monkeypatch.setattr(repo, "mark", spy_mark)
    w = mq.MemorizeQueueWorker(_settings())  # max_attempts 1
    await w.enqueue(
        _payload("bad"), request_id="r1", is_final=False, defer_extraction=False
    )

    await w.drain_once()

    assert writes == [
        {"status": "failed", "attempt": 1, "last_error": "RuntimeError: boom"}
    ]
    row = await repo.find_one(request_id="r1")
    assert row is not None and row.payload_json != ""


async def test_lost_status_write_is_reaped_before_the_next_row(
    repo: _MemorizeInboxRepo, fake: _FakeMemorize, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stuck row must not wait for the backlog to drain before it is failed."""
    monkeypatch.setattr(mq, "_MARK_RETRY_SECONDS", 0.0)
    real_mark = repo.mark

    async def mark_that_loses_r1(row_id: int, **values: Any) -> None:
        row = await repo.get_by_id(row_id)
        if row is not None and row.request_id == "r1":
            raise RuntimeError("database is locked")
        await real_mark(row_id, **values)

    monkeypatch.setattr(repo, "mark", mark_that_loses_r1)
    w = mq.MemorizeQueueWorker(_settings())
    await w.enqueue(
        _payload("s1"), request_id="r1", is_final=False, defer_extraction=False
    )
    await w.enqueue(
        _payload("s1"), request_id="r2", is_final=False, defer_extraction=False
    )

    await w.drain_once()

    r1 = await repo.find_one(request_id="r1")
    r2 = await repo.find_one(request_id="r2")
    assert r1 is not None and r1.status == "failed"
    assert r1.last_error == "status write lost; outcome unknown"
    assert r2 is not None and r2.status == "success"
    assert w._status_write_lost is False


async def test_purge_failure_does_not_stop_delivery(
    repo: _MemorizeInboxRepo, fake: _FakeMemorize, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def broken_purge(*, before: object) -> tuple[int, int]:
        raise RuntimeError("database or disk is full")

    monkeypatch.setattr(repo, "purge_expired", broken_purge)
    w = mq.MemorizeQueueWorker(_settings())
    await w.enqueue(
        _payload("s1"), request_id="r1", is_final=False, defer_extraction=False
    )

    await w.start()  # start() sweeps too; it must not raise
    try:
        await _wait_until_status(repo, "r1", "success")
    finally:
        await w.stop()

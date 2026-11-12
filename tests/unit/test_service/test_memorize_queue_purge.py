"""The async memorize queue during and after a purge of its root.

Claiming never pauses. A row accepted before the latest purge of its scope
is failed without being delivered, checked before every attempt, its body
dropped; a delivery runs with the row's acceptance time as its origin and is
tracked while it runs.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from sqlmodel import SQLModel

from everos.config import SqliteSettings
from everos.config.settings import AsyncQueueSettings
from everos.core.purge_marks import MARKS, ORIGIN, PurgeMarks, PurgeScope, Tombstone
from everos.core.utils.datetime import get_utc_now
from everos.infra.persistence import MemoryRoot
from everos.infra.persistence.sqlite import (
    SqliteMemorizeInboxStore,
    create_session_factory,
    create_system_engine,
)
from everos.infra.persistence.sqlite._repos.memorize_inbox import MemorizeInboxRepo
from everos.memory.models import MemorizeTask
from everos.service import MemorizeResult
from everos.service.memory import _queue as mq
from everos.service.memory._queue_models import to_task_item

S = PurgeScope("project", "a1", "p1", "a1", "p1")


@pytest.fixture(autouse=True)
def _clean() -> Iterator[None]:
    MARKS.reset()
    yield
    MARKS.reset()


@pytest.fixture
async def inbox(tmp_path: Path) -> SqliteMemorizeInboxStore:
    root = MemoryRoot(tmp_path)
    root.ensure()
    engine = create_system_engine(root.system_db, SqliteSettings())
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    return SqliteMemorizeInboxStore(
        MemorizeInboxRepo(session_factory=create_session_factory(engine))
    )


@pytest.fixture
def marks(tmp_path: Path) -> PurgeMarks:
    return MARKS.for_root(tmp_path)


class _Deliver:
    def __init__(self, fail_first: int = 0) -> None:
        self.calls: list[str] = []
        self.fail_first = fail_first
        self.on_call: Any = None

    async def __call__(self, payload: dict[str, Any], **_: Any) -> MemorizeResult:
        self.calls.append(payload["session_id"])
        if self.on_call is not None:
            self.on_call()
        if len(self.calls) <= self.fail_first:
            raise RuntimeError("boom")
        return MemorizeResult(message_count=1, status="extracted")


def _worker(
    inbox: SqliteMemorizeInboxStore, deliver: _Deliver, marks: PurgeMarks, **cfg: Any
) -> mq.MemorizeQueueWorker:
    settings = AsyncQueueSettings(
        retry_backoff_seconds=0.0, scan_interval_seconds=60.0, **cfg
    )
    return mq.MemorizeQueueWorker(settings, inbox=inbox, deliver=deliver, marks=marks)


async def _enqueue(
    worker: mq.MemorizeQueueWorker, session: str, app: str = "a1", project: str = "p1"
) -> None:
    await worker.enqueue(
        {
            "session_id": session,
            "app_id": app,
            "project_id": project,
            "messages": [{"content": "m"}],
        },
        request_id=f"req-{session}",
        is_final=False,
        defer_extraction=False,
    )


async def _row(inbox: SqliteMemorizeInboxStore, session: str) -> MemorizeTask:
    task = await inbox.get_task(f"req-{session}")
    assert task is not None
    return task


async def test_claims_while_a_purge_runs(
    inbox: SqliteMemorizeInboxStore, marks: PurgeMarks
) -> None:
    deliver = _Deliver()
    worker = _worker(inbox, deliver, marks)
    await _enqueue(worker, "s-other", app="a2")
    await _enqueue(worker, "s-new")  # S itself, accepted with no tombstone
    marks.raise_mark("j1", S)
    assert await worker.drain_once() == 2
    assert sorted(deliver.calls) == ["s-new", "s-other"]


async def test_delivery_carries_the_row_origin_and_is_tracked(
    inbox: SqliteMemorizeInboxStore, marks: PurgeMarks
) -> None:
    seen: list[tuple[Any, int]] = []
    deliver = _Deliver()
    worker = _worker(inbox, deliver, marks)
    await _enqueue(worker, "s1")
    accepted = (await _row(inbox, "s1")).created_at
    later = accepted + timedelta(seconds=1)
    deliver.on_call = lambda: seen.append(
        (ORIGIN.get(), marks.tracked_before(later, S))
    )
    stale = ORIGIN.set(accepted - timedelta(days=1))  # e.g. inherited by the loop
    try:
        await worker.drain_once()
    finally:
        ORIGIN.reset(stale)
    assert seen == [(accepted, 1)]
    assert marks.tracked_before(later, S) == 0


async def test_replayed_row_older_than_tombstone_is_discarded(
    inbox: SqliteMemorizeInboxStore, marks: PurgeMarks
) -> None:
    deliver = _Deliver()
    worker = _worker(inbox, deliver, marks)
    await _enqueue(worker, "s1")
    marks.add_tombstone(Tombstone(S, get_utc_now() + timedelta(seconds=1)))
    await worker.drain_once()
    assert deliver.calls == []
    row = await _row(inbox, "s1")
    assert (row.status, row.payload_json) == ("failed", "")
    assert row.last_error is not None
    assert row.last_error.startswith("purge_in_progress: ")
    assert to_task_item(row).error_code == "purge_in_progress"


async def test_rows_after_tombstone_deliver(
    inbox: SqliteMemorizeInboxStore, marks: PurgeMarks
) -> None:
    deliver = _Deliver()
    worker = _worker(inbox, deliver, marks)
    marks.add_tombstone(Tombstone(S, get_utc_now() - timedelta(seconds=1)))
    await _enqueue(worker, "s1")
    await worker.drain_once()
    assert deliver.calls == ["s1"]
    assert (await _row(inbox, "s1")).status == "success"


async def test_discard_checked_before_each_attempt(
    inbox: SqliteMemorizeInboxStore, marks: PurgeMarks
) -> None:
    deliver = _Deliver(fail_first=1)
    worker = _worker(inbox, deliver, marks, max_attempts=2)
    await _enqueue(worker, "s1")
    # The first attempt fails; a purge of the scope lands before the retry.
    deliver.on_call = lambda: marks.add_tombstone(
        Tombstone(S, get_utc_now() + timedelta(seconds=1))
    )
    await worker.drain_once()
    assert deliver.calls == ["s1"]
    row = await _row(inbox, "s1")
    assert (row.status, row.payload_json, row.attempt) == ("failed", "", 1)
    assert row.last_error is not None
    assert row.last_error.startswith("purge_in_progress: ")

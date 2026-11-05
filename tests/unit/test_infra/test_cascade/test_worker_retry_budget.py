"""#549: the inline retry budget must survive a restart during the ladder.

``CascadeWorker._process_one`` retries an ``ExternalServiceError`` inline; the
attempt it consumed used to live only in memory until the final
``mark_failed``, so a process that died inside the ladder left the row
``processing`` with the pre-drain ``retry_count``.
``recover_orphan_processing`` rolled the row back to ``pending`` without
touching the count, and the next drain started the ladder at attempt 0 -- so
recurring restarts never spent ``MAX_TOTAL_RETRIES``, ``cascade fix`` never
listed the file, and every restart re-ran the handler on the same row.

These tests drive the real :class:`CascadeWorker` against the real
:class:`MdChangeStateRepo` on a temp SQLite file (no LanceDB: a handler without
``index_repo`` makes the optimize scheduler a no-op). A process death is
modelled by a handler raising a ``BaseException`` after the persisted attempt,
the same technique ``test_worker_stateful.py`` uses for its ``crash`` outcome:
the SQLite row is exactly what a killed process leaves behind.
"""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from sqlmodel import SQLModel

from everos.config import SqliteSettings
from everos.core.errors import EmbeddingServiceError
from everos.infra.cascade._projections import Handler
from everos.infra.cascade._worker import CascadeWorker
from everos.infra.cascade._worker import _tuning as tuning
from everos.infra.cascade.models import HandlerOutcome
from everos.infra.persistence import MemoryRoot
from everos.infra.persistence.sqlite import create_session_factory, create_system_engine
from everos.infra.persistence.sqlite._repos.md_change_state import MdChangeStateRepo

PATH = "users/u/episodes/episode-2026-05-14.md"


class _Crash(BaseException):
    """The process dies: nothing in the worker catches it.

    Not an ``Exception`` on purpose -- the worker's handler try/except blocks
    must not turn a process death into a ``mark_failed`` verdict.
    """


class _FailsThenCrashHandler(Handler):
    """Attempt 1 fails transiently; attempt 2 is the process dying."""

    def __init__(self) -> None:
        self.calls = 0

    async def handle_added_or_modified(self, md_path: str) -> HandlerOutcome:
        self.calls += 1
        if self.calls == 1:
            raise EmbeddingServiceError("embedding 503")
        raise _Crash

    async def handle_deleted(self, md_path: str) -> HandlerOutcome:
        raise AssertionError("the row enqueued an add, not a delete")


@pytest.fixture
async def ledger(tmp_path: Path) -> AsyncIterator[MdChangeStateRepo]:
    """A real ledger on a temp SQLite file with the schema applied."""
    root = MemoryRoot(tmp_path)
    root.ensure()
    engine = create_system_engine(root.system_db, SqliteSettings())
    factory = create_session_factory(engine)
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    try:
        yield MdChangeStateRepo(session_factory=factory)
    finally:
        await engine.dispose()


def _worker(ledger: MdChangeStateRepo, handler: Handler) -> CascadeWorker:
    """One worker run; the inline ladder is what the tests kill."""
    return CascadeWorker({"episode": handler}, ledger, retry_backoff_seconds=0.0)


async def _drain_and_die(ledger: MdChangeStateRepo) -> None:
    """Run one drain and assert the scripted process death escapes it."""
    with pytest.raises(_Crash):
        await _worker(ledger, _FailsThenCrashHandler()).drain_once()


async def test_restart_during_the_ladder_keeps_the_retry_budget(
    ledger: MdChangeStateRepo,
) -> None:
    """A restart between inline attempts resumes the count, it does not reset it."""
    await ledger.upsert(PATH, kind="episode", change_type="added", mtime=1.0)

    await _drain_and_die(ledger)

    row = await ledger.get_by_id(PATH)
    assert row is not None
    assert row.status == "processing", "the terminal write never happened"
    assert row.retry_count == 1, "the consumed attempt must survive the sleep"

    # The restart: orphan recovery rolls the claim back but keeps the budget.
    assert await ledger.recover_orphan_processing() == 1
    row = await ledger.get_by_id(PATH)
    assert row is not None
    assert (row.status, row.retry_count) == ("pending", 1)

    # The next process continues the ladder at 1, so its first consumed
    # attempt lands the persisted count on 2 rather than 1 again.
    await _drain_and_die(ledger)

    row = await ledger.get_by_id(PATH)
    assert row is not None
    assert row.status == "processing"
    assert row.retry_count == 2, "the restart resumed the budget, it did not reset it"


async def test_recurring_restarts_still_reach_the_retry_budget(
    ledger: MdChangeStateRepo,
) -> None:
    """Recurring restarts must end in ``failed`` so ``cascade fix`` lists it.

    One persisted attempt per restart adds up to ``MAX_TOTAL_RETRIES``; the
    next drain short-circuits the row to a permanent failure before invoking
    the handler. Without persistence each restart began at 0 and the budget
    bounded nothing (#549).
    """
    await ledger.upsert(PATH, kind="episode", change_type="added", mtime=1.0)

    for expected in range(1, tuning.MAX_TOTAL_RETRIES + 1):
        await _drain_and_die(ledger)
        row = await ledger.get_by_id(PATH)
        assert row is not None
        assert row.retry_count == expected, f"restart {expected} forgot the budget"
        assert await ledger.recover_orphan_processing() == 1

    row = await ledger.get_by_id(PATH)
    assert row is not None
    assert (row.status, row.retry_count) == ("pending", tuning.MAX_TOTAL_RETRIES)

    handler = _FailsThenCrashHandler()
    assert await _worker(ledger, handler).drain_once() == 1
    assert handler.calls == 0, "the budget check must precede the handler"
    row = await ledger.get_by_id(PATH)
    assert row is not None
    assert row.status == "failed"
    assert row.retryable is False
    assert row.retry_count == tuning.MAX_TOTAL_RETRIES
    assert [r.md_path for r in await ledger.list_failed()] == [PATH]


class _LockedAttemptHandler(Handler):
    """Attempt 1 takes a real SQLite write lock and fails transiently.

    The lock is held by a second connection, as the watcher, the scanner or a
    CLI ``cascade sync`` would, and released before attempt 2, which succeeds.
    """

    def __init__(self, db: Path, hold_seconds: float) -> None:
        self.db = db
        self.hold_seconds = hold_seconds
        self.calls = 0
        self._con: sqlite3.Connection | None = None

    def release(self) -> None:
        if self._con is not None:
            self._con.execute("ROLLBACK")
            self._con.close()
            self._con = None

    async def handle_added_or_modified(self, md_path: str) -> HandlerOutcome:
        self.calls += 1
        if self.calls == 1:
            self._con = sqlite3.connect(self.db, isolation_level=None, timeout=0)
            self._con.execute("BEGIN IMMEDIATE")
            asyncio.get_running_loop().call_later(self.hold_seconds, self.release)
            raise EmbeddingServiceError("embedding 503")
        return HandlerOutcome(
            md_path=md_path, kind="episode", upserted=0, deleted=0, skipped=1
        )

    async def handle_deleted(self, md_path: str) -> HandlerOutcome:
        raise AssertionError("the row enqueued an add, not a delete")


async def test_a_locked_attempt_write_does_not_strand_the_row(tmp_path: Path) -> None:
    """``record_attempt`` hitting ``database is locked`` must not leave the row
    ``processing`` (the #548 symptom); the ladder goes on and the row lands
    ``done`` once the provider recovers.
    """
    root = MemoryRoot(tmp_path)
    root.ensure()
    engine = create_system_engine(root.system_db, SqliteSettings(busy_timeout_ms=50))
    factory = create_session_factory(engine)
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    ledger = MdChangeStateRepo(session_factory=factory)
    handler = _LockedAttemptHandler(root.system_db, hold_seconds=0.2)
    try:
        await ledger.upsert(PATH, kind="episode", change_type="added", mtime=1.0)
        worker = CascadeWorker(
            {"episode": handler}, ledger, max_retry=3, retry_backoff_seconds=0.3
        )
        await worker.drain_once()
    finally:
        handler.release()
        await engine.dispose()

    assert handler.calls == 2
    row = await ledger.get_by_id(PATH)
    assert row is not None
    assert row.status == "done"

"""An index write that finds the disk full is waited out, never given up on.

docs/cascade_runbook.md, "Disk full". The worker takes no inline retry (the
disk is still full) and spends no retry budget: it parks the row as a
retryable failure with a ``retry_at``, the scanner leaves it alone until then,
and each further full disk doubles the wait up to the cap -- one attempt and
one warning per wait. Once space is freed the next attempt lands the row,
with no ``cascade fix``. Meanwhile ``cascade status`` counts the row
(``waiting_for_space``) and ``/health`` reads the cascade unhealthy.

Driven against the real ledger on a temp SQLite file and the real reconciler;
the handler stands in for a LanceDB write on a full disk.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import timedelta
from pathlib import Path

import pytest
import structlog
from sqlmodel import SQLModel

from everos.config import SqliteSettings
from everos.core.errors import IndexDiskFullError
from everos.core.utils.datetime import get_utc_now
from everos.infra.cascade._projections import Handler
from everos.infra.cascade._worker import CascadeWorker
from everos.infra.cascade._worker import _tuning as tuning
from everos.infra.cascade.models import HandlerOutcome, ScanInput
from everos.infra.cascade.reconciler import PriorState, reconcile
from everos.infra.persistence import MemoryRoot
from everos.infra.persistence.sqlite import create_session_factory, create_system_engine
from everos.infra.persistence.sqlite._repos.md_change_state import MdChangeStateRepo

PATH = "users/u/episodes/episode-2026-05-14.md"


class _FullDisk(Handler):
    """Every write fails for a full disk until ``full`` is cleared."""

    def __init__(self) -> None:
        self.full = True
        self.calls = 0

    async def handle_added_or_modified(self, md_path: str) -> HandlerOutcome:
        self.calls += 1
        if self.full:
            raise IndexDiskFullError(
                "upsert on table 'episode': the disk is full: "
                "No space left on device (os error 28)"
            )
        return HandlerOutcome(
            md_path=md_path, kind="episode", upserted=1, deleted=0, skipped=0
        )

    async def handle_deleted(self, md_path: str) -> HandlerOutcome:
        raise AssertionError("the row enqueued an add, not a delete")


@pytest.fixture
async def ledger(tmp_path: Path) -> AsyncIterator[MdChangeStateRepo]:
    root = MemoryRoot(tmp_path)
    root.ensure()
    engine = create_system_engine(root.system_db, SqliteSettings())
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    try:
        yield MdChangeStateRepo(session_factory=create_session_factory(engine))
    finally:
        await engine.dispose()


def _worker(ledger: MdChangeStateRepo, handler: Handler) -> CascadeWorker:
    return CascadeWorker(
        {"episode": handler},
        ledger,
        retry_backoff_seconds=0.0,
        storage_full_retry_seconds=30.0,
        storage_full_retry_max_seconds=120.0,
    )


async def _sweep(ledger: MdChangeStateRepo, *, at_offset: float) -> bool:
    """One scanner sweep ``at_offset`` seconds from now over the unchanged
    file; whether it re-enqueued the row."""
    row = await ledger.get_by_id(PATH)
    assert row is not None
    prior = PriorState(
        md_path=row.md_path,
        kind=row.kind,
        mtime=row.mtime,
        status=row.status,
        change_type=row.change_type,
        retryable=row.retryable,
        retry_count=row.retry_count,
        retry_at=row.retry_at,
    )
    decisions = reconcile(
        [ScanInput(md_path=PATH, kind="episode", mtime=row.mtime)],
        {PATH: prior},
        now=get_utc_now() + timedelta(seconds=at_offset),
    )
    for decision in decisions:
        await ledger.upsert(
            decision.md_path,
            kind=decision.kind,
            change_type=decision.change_type,
            mtime=decision.mtime,
        )
    return bool(decisions)


async def test_a_full_disk_is_waited_out_with_a_capped_back_off(
    ledger: MdChangeStateRepo,
) -> None:
    handler = _FullDisk()
    worker = _worker(ledger, handler)
    await ledger.upsert(PATH, kind="episode", change_type="added", mtime=1.0)
    waits: list[float] = []

    with structlog.testing.capture_logs() as logs:
        # Well past the budget that files other failures as permanent.
        for _ in range(tuning.MAX_TOTAL_RETRIES + 3):
            calls = handler.calls
            started = get_utc_now()
            await worker.drain_once()
            assert handler.calls == calls + 1, "no inline retry on a full disk"
            row = await ledger.get_by_id(PATH)
            assert row is not None and row.retry_at is not None
            assert (row.status, row.retryable, row.retry_count) == ("failed", True, 0)
            waits.append(round((row.retry_at - started).total_seconds()))
            # Not before its time; then the scanner hands it back.
            assert not await _sweep(ledger, at_offset=waits[-1] - 5)
            assert await _sweep(ledger, at_offset=waits[-1] + 1)

    assert waits[:4] == [30, 60, 120, 120]
    assert set(waits[3:]) == {120}
    warned = [e for e in logs if e["event"] == "cascade_worker_disk_full"]
    assert len(warned) == len(waits), "one warning per wait, not per attempt"
    assert [e["waits"] for e in warned] == list(range(1, len(waits) + 1))

    # Space is freed: the next attempt lands the row, no `cascade fix`.
    handler.full = False
    await worker.drain_once()
    row = await ledger.get_by_id(PATH)
    assert row is not None
    assert (row.status, row.retry_at, row.wait_steps) == ("done", None, None)
    assert (await ledger.queue_summary()).waiting_for_space == 0


async def test_waiting_rows_are_counted_and_cascade_fix_retries_them_at_once(
    ledger: MdChangeStateRepo,
) -> None:
    await ledger.upsert(PATH, kind="episode", change_type="added", mtime=1.0)
    await _worker(ledger, _FullDisk()).drain_once()

    summary = await ledger.queue_summary()
    assert (summary.failed_retryable, summary.waiting_for_space) == (1, 1)
    assert summary.pending == 0

    assert await ledger.reset_retryable_to_pending() == 1
    row = await ledger.get_by_id(PATH)
    assert row is not None
    assert (row.status, row.retry_at, row.wait_steps) == ("pending", None, None)


async def test_an_edited_file_is_tried_at_once_and_starts_the_back_off_again(
    ledger: MdChangeStateRepo,
) -> None:
    handler = _FullDisk()
    worker = _worker(ledger, handler)
    await ledger.upsert(PATH, kind="episode", change_type="added", mtime=1.0)
    await worker.drain_once()
    assert await _sweep(ledger, at_offset=31)
    await worker.drain_once()
    row = await ledger.get_by_id(PATH)
    assert row is not None and row.wait_steps == 2

    # The watcher reports an edit: the new content is tried now, from the
    # first wait again.
    await ledger.upsert(PATH, kind="episode", change_type="modified", mtime=2.0)
    row = await ledger.get_by_id(PATH)
    assert row is not None
    assert (row.status, row.retry_at, row.wait_steps) == ("pending", None, None)
    await worker.drain_once()
    row = await ledger.get_by_id(PATH)
    assert row is not None and row.wait_steps == 1


async def test_the_cascade_is_unhealthy_while_a_file_waits_for_space(
    ledger: MdChangeStateRepo,
) -> None:
    from everos.infra.cascade.orchestrator import CascadeOrchestrator

    await ledger.upsert(PATH, kind="episode", change_type="added", mtime=1.0)
    worker = _worker(ledger, _FullDisk())
    await worker.drain_once()
    orchestrator = CascadeOrchestrator.__new__(CascadeOrchestrator)
    orchestrator._worker = worker  # type: ignore[attr-defined]
    orchestrator._ledger = ledger  # type: ignore[attr-defined]
    orchestrator._index_rows = {}  # type: ignore[attr-defined]
    orchestrator._index_emptied = False  # type: ignore[attr-defined]

    health = await orchestrator.health()

    assert not health.healthy
    assert health.waiting_for_space == 1 and health.failed_retryable == 1
    assert any("wait for disk space" in reason for reason in health.reasons)

"""Tests for :class:`_MdChangeStateRepo` — cascade work-queue persistence.

Builds a fresh tmp-file SQLite engine per test (the in-memory ``sqlite``
driver can't share schema across concurrent connections), wires a
private repo instance to its session factory, then exercises every
public method against the live database — no mocks, no in-memory
shortcuts.

Covers the unit-test matrix from
``16_cascade_impl_design.md`` §14 for this commit:

- ``upsert`` — LSN monotonic across the same path, retry_count resets.
- ``claim_one`` — atomic; concurrent racers split 1 winner / N losers.
- ``reset_retryable_to_pending`` — only ``retryable=TRUE`` rows move.

Plus the rest of the repo surface (mark_done / mark_failed /
queue_summary / list_failed / force_enqueue / claim_pending_batch).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from sqlmodel import SQLModel

from everos.config import SqliteSettings
from everos.core.persistence import (
    MemoryRoot,
    create_session_factory,
    create_system_engine,
)
from everos.infra.persistence.sqlite.repos.md_change_state import (
    _MdChangeStateRepo,
)


@pytest.fixture
async def repo(tmp_path: Path) -> _MdChangeStateRepo:
    """Per-test repo wired to a fresh tmp SQLite DB with schema applied."""
    mr = MemoryRoot(tmp_path)
    mr.ensure()
    engine = create_system_engine(mr.system_db, SqliteSettings())
    factory = create_session_factory(engine)
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    return _MdChangeStateRepo(session_factory=factory)


# ── upsert ──────────────────────────────────────────────────────────────


async def test_upsert_assigns_monotonic_lsn(repo: _MdChangeStateRepo) -> None:
    """Two distinct paths get strictly increasing LSNs."""
    lsn_a = await repo.upsert(
        "users/u/episodes/episode-2026-05-12.md",
        kind="episode",
        change_type="added",
        mtime=1.0,
    )
    lsn_b = await repo.upsert(
        "users/u/episodes/episode-2026-05-13.md",
        kind="episode",
        change_type="added",
        mtime=2.0,
    )
    assert lsn_a == 1
    assert lsn_b == 2


async def test_upsert_same_path_bumps_lsn_and_resets_retry(
    repo: _MdChangeStateRepo,
) -> None:
    """Re-enqueueing the same path bumps LSN and clears prior failure state."""
    path = "users/u/episodes/episode-2026-05-12.md"
    await repo.upsert(path, kind="episode", change_type="added", mtime=1.0)
    # Simulate a worker run that failed (retryable): claim then fail.
    await repo.claim_one(path)
    await repo.mark_failed(path, retryable=True, error="503", new_retry_count=3)

    lsn_after = await repo.upsert(
        path, kind="episode", change_type="modified", mtime=2.0
    )
    row = await repo.get_by_id(path)
    assert row is not None
    assert row.lsn == lsn_after
    assert lsn_after > 1
    # State reset back to pending; failure metadata cleared.
    assert row.status == "pending"
    assert row.retry_count == 0
    assert row.error is None
    assert row.retryable is None
    # Re-enqueue refreshes change_type / mtime to the new event.
    assert row.change_type == "modified"
    assert row.mtime == 2.0


# ── force_enqueue ───────────────────────────────────────────────────────


async def test_force_enqueue_resurrects_done_row(
    repo: _MdChangeStateRepo,
) -> None:
    """`cascade sync --path` re-enqueues even a row that already landed."""
    path = "users/u/episodes/episode-2026-05-12.md"
    await repo.upsert(path, kind="episode", change_type="added", mtime=1.0)
    await repo.claim_one(path)
    await repo.mark_done(path)

    lsn = await repo.force_enqueue(path, "episode")
    row = await repo.get_by_id(path)
    assert row is not None
    assert row.lsn == lsn
    assert row.status == "pending"
    assert row.change_type == "modified"


# ── claim_one ───────────────────────────────────────────────────────────


async def test_claim_one_returns_row_when_pending(
    repo: _MdChangeStateRepo,
) -> None:
    path = "users/u/episodes/episode-2026-05-12.md"
    await repo.upsert(path, kind="episode", change_type="added", mtime=1.0)
    row = await repo.claim_one(path)
    assert row is not None
    assert row.md_path == path
    assert row.status == "processing"
    assert row.last_attempt_at is not None


async def test_claim_one_returns_none_when_already_processing(
    repo: _MdChangeStateRepo,
) -> None:
    """Second claim of the same row returns None — claim is one-shot."""
    path = "users/u/episodes/episode-2026-05-12.md"
    await repo.upsert(path, kind="episode", change_type="added", mtime=1.0)
    first = await repo.claim_one(path)
    assert first is not None
    second = await repo.claim_one(path)
    assert second is None


async def test_claim_one_race_only_one_winner(
    repo: _MdChangeStateRepo,
) -> None:
    """Three concurrent claims on the same row: exactly one wins."""
    path = "users/u/episodes/episode-2026-05-12.md"
    await repo.upsert(path, kind="episode", change_type="added", mtime=1.0)
    results = await asyncio.gather(
        repo.claim_one(path),
        repo.claim_one(path),
        repo.claim_one(path),
    )
    winners = [r for r in results if r is not None]
    assert len(winners) == 1


# ── claim_pending_batch ─────────────────────────────────────────────────


async def test_claim_pending_batch_returns_in_lsn_order(
    repo: _MdChangeStateRepo,
) -> None:
    paths = [f"users/u/episodes/e-{i}.md" for i in range(3)]
    for p in paths:
        await repo.upsert(p, kind="episode", change_type="added", mtime=0.0)

    batch = await repo.claim_pending_batch(limit=10)
    assert [r.md_path for r in batch] == paths
    assert all(r.status == "processing" for r in batch)


async def test_claim_pending_batch_skips_already_claimed(
    repo: _MdChangeStateRepo,
) -> None:
    """Already-processing rows are not re-claimed."""
    await repo.upsert("a.md", kind="episode", change_type="added", mtime=0.0)
    await repo.upsert("b.md", kind="episode", change_type="added", mtime=0.0)
    await repo.claim_one("a.md")

    batch = await repo.claim_pending_batch(limit=10)
    assert [r.md_path for r in batch] == ["b.md"]


async def test_claim_pending_batch_zero_limit_returns_empty(
    repo: _MdChangeStateRepo,
) -> None:
    await repo.upsert("a.md", kind="episode", change_type="added", mtime=0.0)
    assert await repo.claim_pending_batch(limit=0) == []


# ── mark_done / mark_failed ─────────────────────────────────────────────


async def test_mark_done_transitions_processing_to_done(
    repo: _MdChangeStateRepo,
) -> None:
    """``processing → done`` lands a clean terminal row (no error fields)."""
    path = "users/u/episodes/episode-2026-05-12.md"
    await repo.upsert(path, kind="episode", change_type="added", mtime=1.0)
    await repo.claim_one(path)
    await repo.mark_done(path)

    row = await repo.get_by_id(path)
    assert row is not None
    assert row.status == "done"
    assert row.error is None
    assert row.retryable is None


async def test_mark_failed_records_retryable_flag(
    repo: _MdChangeStateRepo,
) -> None:
    path = "users/u/episodes/episode-2026-05-12.md"
    await repo.upsert(path, kind="episode", change_type="added", mtime=1.0)
    await repo.claim_one(path)
    await repo.mark_failed(
        path, retryable=False, error="YAML parse: line 5", new_retry_count=0
    )

    row = await repo.get_by_id(path)
    assert row is not None
    assert row.status == "failed"
    assert row.retryable is False
    assert row.error == "YAML parse: line 5"
    assert row.retry_count == 0


# ── Race: re-enqueue during processing must win over stale mark_xxx ─────
#
# Reproduces the Bug A scenario:
#   T0  watcher upsert       → status=pending,    lsn=1
#   T1  worker claim_one     → status=processing, lsn=1
#   T2  watcher upsert again → status=pending,    lsn=2  (on_conflict_do_update)
#   T3  worker (stale claim) finishes handler
#   T4  worker mark_xxx      → must no-op because status != processing
#
# Without the guard, T4 overwrites T2's pending and the worker never
# re-processes the latest md state.


async def test_mark_done_noop_when_row_reverted_to_pending(
    repo: _MdChangeStateRepo,
) -> None:
    """Concurrent upsert during processing → mark_done must not overwrite it."""
    path = "users/u/episodes/episode-2026-05-12.md"
    # T0: watcher enqueues.
    lsn_1 = await repo.upsert(path, kind="episode", change_type="added", mtime=1.0)
    # T1: worker claims.
    claimed = await repo.claim_one(path)
    assert claimed is not None
    # T2: watcher re-enqueues — row flipped back to pending with a fresh lsn.
    lsn_2 = await repo.upsert(path, kind="episode", change_type="modified", mtime=2.0)
    assert lsn_2 > lsn_1
    # T4: stale mark_done — guard must make this a no-op.
    await repo.mark_done(path)

    row = await repo.get_by_id(path)
    assert row is not None
    assert row.status == "pending"  # not "done"
    assert row.lsn == lsn_2  # upsert's lsn survives
    assert row.change_type == "modified"  # upsert's payload survives
    assert row.mtime == 2.0


async def test_mark_failed_noop_when_row_reverted_to_pending(
    repo: _MdChangeStateRepo,
) -> None:
    """Concurrent upsert during processing → mark_failed must not overwrite it."""
    path = "users/u/episodes/episode-2026-05-12.md"
    lsn_1 = await repo.upsert(path, kind="episode", change_type="added", mtime=1.0)
    claimed = await repo.claim_one(path)
    assert claimed is not None
    lsn_2 = await repo.upsert(path, kind="episode", change_type="modified", mtime=2.0)
    assert lsn_2 > lsn_1
    await repo.mark_failed(path, retryable=True, error="503", new_retry_count=2)

    row = await repo.get_by_id(path)
    assert row is not None
    assert row.status == "pending"  # not "failed"
    assert row.lsn == lsn_2
    assert row.error is None  # upsert cleared the error fields
    assert row.retryable is None
    assert row.retry_count == 0


async def test_mark_done_concurrent_with_upsert_preserves_reenqueue(
    repo: _MdChangeStateRepo,
) -> None:
    """asyncio.gather(upsert, mark_done): final state never loses the upsert.

    Two valid commit orderings:
      * upsert first → mark_done sees status != processing → no-op
        → final = pending(lsn=2)
      * mark_done first → row=done(lsn=1) → upsert flips back to pending(lsn=2)
        → final = pending(lsn=2)

    Both orderings converge on the same invariant: the re-enqueue wins.
    """
    path = "users/u/episodes/episode-2026-05-12.md"
    lsn_1 = await repo.upsert(path, kind="episode", change_type="added", mtime=1.0)
    await repo.claim_one(path)

    # Race the two writes. SQLite WAL serialises commits, so one is
    # ordered before the other — but the test does not pin which.
    await asyncio.gather(
        repo.upsert(path, kind="episode", change_type="modified", mtime=2.0),
        repo.mark_done(path),
    )

    row = await repo.get_by_id(path)
    assert row is not None
    assert row.status == "pending"
    assert row.lsn > lsn_1
    assert row.change_type == "modified"
    assert row.mtime == 2.0


# ── reset_retryable_to_pending ──────────────────────────────────────────


async def test_reset_retryable_to_pending_moves_only_retryable(
    repo: _MdChangeStateRepo,
) -> None:
    """`cascade fix --apply` semantics: only retryable=TRUE rows move."""
    await repo.upsert("a.md", kind="episode", change_type="added", mtime=0.0)
    await repo.upsert("b.md", kind="episode", change_type="added", mtime=0.0)
    await repo.upsert("c.md", kind="episode", change_type="added", mtime=0.0)

    await repo.claim_one("a.md")
    await repo.mark_failed("a.md", retryable=True, error="503", new_retry_count=3)
    await repo.claim_one("b.md")
    await repo.mark_failed("b.md", retryable=False, error="YAML", new_retry_count=0)
    # c.md remains pending.

    moved = await repo.reset_retryable_to_pending()
    assert moved == 1

    a = await repo.get_by_id("a.md")
    b = await repo.get_by_id("b.md")
    assert a is not None and a.status == "pending"
    assert a.retry_count == 0
    assert a.retryable is None
    assert a.error is None
    assert b is not None and b.status == "failed"
    assert b.retryable is False


async def test_reset_retryable_to_pending_zero_when_none_eligible(
    repo: _MdChangeStateRepo,
) -> None:
    await repo.upsert("a.md", kind="episode", change_type="added", mtime=0.0)
    await repo.claim_one("a.md")
    await repo.mark_failed("a.md", retryable=False, error="YAML", new_retry_count=0)
    assert await repo.reset_retryable_to_pending() == 0


# ── list_failed ─────────────────────────────────────────────────────────


async def test_list_failed_orders_by_lsn(repo: _MdChangeStateRepo) -> None:
    for path in ("a.md", "b.md", "c.md"):
        await repo.upsert(path, kind="episode", change_type="added", mtime=0.0)
    await repo.claim_one("a.md")
    await repo.mark_failed("a.md", retryable=True, error="x", new_retry_count=3)
    await repo.claim_one("c.md")
    await repo.mark_failed("c.md", retryable=False, error="y", new_retry_count=0)

    rows = await repo.list_failed()
    assert [r.md_path for r in rows] == ["a.md", "c.md"]


# ── queue_summary ───────────────────────────────────────────────────────


async def test_queue_summary_aggregates_all_states(
    repo: _MdChangeStateRepo,
) -> None:
    # Pending: 2 (one rolled through processing)
    await repo.upsert("p1.md", kind="episode", change_type="added", mtime=0.0)
    await repo.upsert("p2.md", kind="episode", change_type="added", mtime=0.0)
    await repo.claim_one("p2.md")  # → processing, still counts as pending.
    # Done: 1 (full claim → mark_done path matches production flow).
    await repo.upsert("d.md", kind="episode", change_type="added", mtime=0.0)
    await repo.claim_one("d.md")
    await repo.mark_done("d.md")
    # Failed retryable: 1
    await repo.upsert("fr.md", kind="episode", change_type="added", mtime=0.0)
    await repo.claim_one("fr.md")
    await repo.mark_failed("fr.md", retryable=True, error="503", new_retry_count=3)
    # Failed permanent: 1
    await repo.upsert("fp.md", kind="episode", change_type="added", mtime=0.0)
    await repo.claim_one("fp.md")
    await repo.mark_failed("fp.md", retryable=False, error="YAML", new_retry_count=0)

    summary = await repo.queue_summary()
    assert summary.pending == 2
    assert summary.done == 1
    assert summary.failed_retryable == 1
    assert summary.failed_permanent == 1
    # 5 upserts → max LSN 5; last_processed = max among done/failed.
    assert summary.max_lsn == 5
    assert summary.last_processed_lsn == 5


async def test_queue_summary_empty_table(repo: _MdChangeStateRepo) -> None:
    summary = await repo.queue_summary()
    assert summary == _empty_summary()


def _empty_summary() -> object:
    from everos.infra.persistence.sqlite import QueueSummary

    return QueueSummary(
        pending=0,
        done=0,
        failed_retryable=0,
        failed_permanent=0,
        max_lsn=0,
        last_processed_lsn=0,
    )


# ── recover_orphan_processing ───────────────────────────────────────────


async def test_recover_orphan_processing_resets_stale_rows(
    repo: _MdChangeStateRepo,
) -> None:
    """Crash recovery: every ``processing`` row goes back to ``pending``."""
    await repo.upsert("a.md", kind="episode", change_type="added", mtime=0.0)
    await repo.upsert("b.md", kind="episode", change_type="added", mtime=0.0)
    # Simulate a worker that claimed both but died before mark_done/failed.
    await repo.claim_one("a.md")
    await repo.claim_one("b.md")

    moved = await repo.recover_orphan_processing()
    assert moved == 2
    a = await repo.get_by_id("a.md")
    b = await repo.get_by_id("b.md")
    assert a is not None and a.status == "pending"
    assert b is not None and b.status == "pending"
    # last_attempt_at cleared so the next claim records the new attempt.
    assert a.last_attempt_at is None
    assert b.last_attempt_at is None


async def test_recover_orphan_processing_zero_when_clean(
    repo: _MdChangeStateRepo,
) -> None:
    """No rows in ``processing`` → returns 0, leaves the rest alone."""
    await repo.upsert("a.md", kind="episode", change_type="added", mtime=0.0)
    await repo.claim_one("a.md")
    await repo.mark_done("a.md")
    assert await repo.recover_orphan_processing() == 0
    row = await repo.get_by_id("a.md")
    assert row is not None
    assert row.status == "done"


async def test_recover_orphan_processing_only_touches_processing_rows(
    repo: _MdChangeStateRepo,
) -> None:
    """Pending / done / failed rows are untouched."""
    await repo.upsert("p.md", kind="episode", change_type="added", mtime=0.0)
    await repo.upsert("d.md", kind="episode", change_type="added", mtime=0.0)
    await repo.upsert("f.md", kind="episode", change_type="added", mtime=0.0)
    await repo.upsert("proc.md", kind="episode", change_type="added", mtime=0.0)
    await repo.claim_one("d.md")
    await repo.mark_done("d.md")
    await repo.claim_one("f.md")
    await repo.mark_failed("f.md", retryable=True, error="x", new_retry_count=1)
    await repo.claim_one("proc.md")

    moved = await repo.recover_orphan_processing()
    assert moved == 1
    p = await repo.get_by_id("p.md")
    d = await repo.get_by_id("d.md")
    f = await repo.get_by_id("f.md")
    proc = await repo.get_by_id("proc.md")
    assert p is not None and p.status == "pending"
    assert d is not None and d.status == "done"
    assert f is not None and f.status == "failed"
    assert proc is not None and proc.status == "pending"


# ── Partial indexes (smoke) ─────────────────────────────────────────────


async def test_partial_indexes_are_created(repo: _MdChangeStateRepo) -> None:
    """The three partial / mtime indexes from the schema land in sqlite_master."""
    async with repo.session_factory() as s:
        from sqlalchemy import text

        result = await s.execute(
            text("SELECT name FROM sqlite_master WHERE type='index'")
        )
        names = {row[0] for row in result.all()}
    for expected in (
        "idx_md_change_pending",
        "idx_md_change_retryable",
        "idx_md_change_mtime",
        "idx_md_change_kind",
    ):
        assert expected in names, f"missing index {expected!r}; got {names!r}"

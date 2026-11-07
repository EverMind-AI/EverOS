"""An unrecoverable cascade row is terminal: the same bytes are not retried."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from sqlmodel import SQLModel

from everos.config import SqliteSettings
from everos.infra.cascade._projections import Handler
from everos.infra.cascade._worker import CascadeWorker
from everos.infra.cascade.models import HandlerOutcome
from everos.infra.cascade.projection import settle_written
from everos.infra.persistence import MemoryRoot
from everos.infra.persistence.sqlite import (
    MdChangeStateRepo,
    create_session_factory,
    create_system_engine,
)

_EPISODE = "default/default/users/u/episodes/episode-2026-11-07.md"


class _Boom(Handler):
    def __init__(self) -> None:
        self.calls = 0

    async def handle_added_or_modified(self, md_path: str) -> HandlerOutcome:
        self.calls += 1
        raise RuntimeError("lance schema drift")

    async def handle_deleted(self, md_path: str) -> HandlerOutcome:
        return HandlerOutcome(
            md_path=md_path, kind="episode", upserted=0, deleted=1, skipped=0
        )


@pytest.fixture
async def ledger(tmp_path: Path) -> tuple[MdChangeStateRepo, Path]:
    root = MemoryRoot(tmp_path)
    root.ensure()
    engine = create_system_engine(root.system_db, SqliteSettings())
    factory = create_session_factory(engine)
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    return MdChangeStateRepo(session_factory=factory), root.root


async def test_same_mtime_is_not_retried_after_an_unrecoverable_failure(
    ledger: tuple[MdChangeStateRepo, Path],
) -> None:
    repo, _root = ledger
    handler = _Boom()
    worker = CascadeWorker(
        {"episode": handler}, repo, retry_backoff_seconds=0, max_retry=0
    )
    await repo.upsert(_EPISODE, kind="episode", change_type="added", mtime=10.0)

    await worker.drain_once()

    row = await repo.read_one(_EPISODE)
    assert row is not None
    assert row.status == "failed"
    assert row.retryable is False
    assert row.retry_count == 1
    assert row.error is not None and "lance schema drift" in row.error
    assert handler.calls == 1

    for _ in range(5):
        lsn = await repo.upsert(
            _EPISODE, kind="episode", change_type="modified", mtime=10.0
        )
        assert lsn == row.lsn
        await worker.drain_once()

    assert handler.calls == 1
    again = await repo.read_one(_EPISODE)
    assert again is not None
    assert again.status == "failed"
    assert again.retry_count == 1


async def test_changed_mtime_and_operator_force_each_get_another_attempt(
    ledger: tuple[MdChangeStateRepo, Path],
) -> None:
    repo, _root = ledger
    handler = _Boom()
    worker = CascadeWorker(
        {"episode": handler}, repo, retry_backoff_seconds=0, max_retry=0
    )
    await repo.upsert(_EPISODE, kind="episode", change_type="added", mtime=10.0)
    await worker.drain_once()
    assert handler.calls == 1

    await repo.upsert(_EPISODE, kind="episode", change_type="modified", mtime=20.0)
    await worker.drain_once()
    assert handler.calls == 2
    edited = await repo.read_one(_EPISODE)
    assert edited is not None and edited.status == "failed"

    await repo.force_enqueue(_EPISODE, "episode", mtime=20.0)
    forced = await repo.read_one(_EPISODE)
    assert forced is not None and forced.status == "pending" and forced.retry_count == 0
    await worker.drain_once()
    assert handler.calls == 3


async def test_deleting_a_terminal_file_is_still_queued(
    ledger: tuple[MdChangeStateRepo, Path],
) -> None:
    repo, _root = ledger
    await repo.upsert(_EPISODE, kind="episode", change_type="added", mtime=10.0)
    await repo.claim_one(_EPISODE)
    await repo.mark_failed(_EPISODE, retryable=False, error="schema", new_retry_count=1)

    await repo.upsert(_EPISODE, kind="episode", change_type="deleted", mtime=10.0)

    row = await repo.read_one(_EPISODE)
    assert row is not None and row.status == "pending" and row.change_type == "deleted"


async def test_settle_reports_a_permanent_failure_instead_of_success(
    ledger: tuple[MdChangeStateRepo, Path],
) -> None:
    repo, root = ledger
    full = root / _EPISODE
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text("episode\n", encoding="utf-8")
    mtime = full.stat().st_mtime
    await repo.upsert(_EPISODE, kind="episode", change_type="added", mtime=mtime)
    await repo.claim_one(_EPISODE)
    await repo.mark_failed(_EPISODE, retryable=False, error="schema", new_retry_count=1)

    verdict = await settle_written(
        repo, root, [_EPISODE], started=True, budget_seconds=1.0, poll=0.01
    )

    assert verdict.observed is True
    assert verdict.failed_permanent == 1
    assert verdict.failed_retryable == 0
    assert verdict.pending == 0
    row = await repo.read_one(_EPISODE)
    assert row is not None and row.status == "failed"


async def test_settle_waits_until_a_processing_row_fails(
    ledger: tuple[MdChangeStateRepo, Path],
) -> None:
    repo, root = ledger
    full = root / _EPISODE
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text("episode\n", encoding="utf-8")
    mtime = full.stat().st_mtime
    await repo.upsert(_EPISODE, kind="episode", change_type="added", mtime=mtime)
    await repo.claim_one(_EPISODE)

    async def _fail_soon() -> None:
        await asyncio.sleep(0.05)
        await repo.mark_failed(
            _EPISODE, retryable=False, error="schema", new_retry_count=1
        )

    task = asyncio.create_task(_fail_soon())
    verdict = await settle_written(
        repo, root, [_EPISODE], started=True, budget_seconds=2.0, poll=0.01
    )
    await task

    assert verdict.failed_permanent == 1
    assert verdict.pending == 0


async def test_settle_keeps_a_still_pending_row_out_of_the_failure_counts(
    ledger: tuple[MdChangeStateRepo, Path],
) -> None:
    repo, root = ledger
    full = root / _EPISODE
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text("episode\n", encoding="utf-8")
    mtime = full.stat().st_mtime
    await repo.upsert(_EPISODE, kind="episode", change_type="added", mtime=mtime)

    verdict = await settle_written(
        repo, root, [_EPISODE], started=True, budget_seconds=0.05, poll=0.01
    )

    assert verdict.observed is True
    assert verdict.pending == 1
    assert verdict.failed_permanent == 0
    assert verdict.failed_retryable == 0


async def test_settle_is_unobserved_when_cascade_is_not_running(
    ledger: tuple[MdChangeStateRepo, Path],
) -> None:
    repo, root = ledger
    verdict = await settle_written(
        repo, root, [_EPISODE], started=False, budget_seconds=0.01, poll=0.01
    )
    assert verdict.observed is False

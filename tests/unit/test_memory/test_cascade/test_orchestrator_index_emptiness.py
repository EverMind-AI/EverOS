"""The cascade tells an emptied index apart from a healthy one (issue #501).

Deleting ``.index/lancedb`` leaves the sqlite ledger behind, so every file
stays ``done``, the scanner enqueues nothing, and every health surface reports
a fully drained pipeline over a store that holds nothing. The verdict needs
all three facts at once -- files were indexed, nothing is waiting, and every
table is empty -- and each of the three cases below removes exactly one of
them, so a verdict that ignored any single fact would go green here.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import patch

import pytest
from sqlmodel import SQLModel

from everos.component.tokenizer import build_tokenizer
from everos.core.persistence import MemoryRoot
from everos.infra.persistence.lancedb import (
    dispose_connection,
    ensure_business_indexes,
)
from everos.infra.persistence.sqlite import dispose_engine, get_engine
from everos.infra.persistence.sqlite.repos.md_change_state import QueueSummary
from everos.memory.cascade import CascadeConfig, CascadeOrchestrator

_ORCH = "everos.memory.cascade.orchestrator"


@pytest.fixture
async def runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[MemoryRoot]:
    monkeypatch.setenv("EVEROS_ROOT", str(tmp_path))
    monkeypatch.setenv("EVEROS_EMBEDDING__MODEL", "stub-model")
    monkeypatch.setenv("EVEROS_EMBEDDING__BASE_URL", "http://stub.invalid/v1")
    monkeypatch.setenv("EVEROS_EMBEDDING__API_KEY", "stub-key")
    await dispose_connection()
    await dispose_engine()
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    await ensure_business_indexes()
    yield MemoryRoot.resolve()
    await dispose_connection()
    await dispose_engine()


def _orchestrator(memory_root: MemoryRoot) -> CascadeOrchestrator:
    return CascadeOrchestrator(
        memory_root=memory_root,
        tokenizer=build_tokenizer(),
        config=CascadeConfig(
            scan_interval_seconds=60.0,
            worker_batch_size=10,
            worker_max_retry=1,
            worker_poll_interval_seconds=0.05,
            worker_retry_backoff_seconds=0.0,
        ),
    )


def _summary(*, pending: int, done: int) -> QueueSummary:
    """A queue summary built through the real type, not a stand-in.

    ``QueueSummary`` carries fields this check never reads; constructing it
    properly means a field added to it later cannot quietly change what this
    test is asserting about.
    """
    return QueueSummary(
        pending=pending,
        done=done,
        failed_retryable=0,
        failed_permanent=0,
        max_lsn=done,
        last_processed_lsn=done,
    )


async def _verdict(
    memory_root: MemoryRoot,
    *,
    rows: dict[str, int],
    summary: QueueSummary,
    live: int | None = None,
) -> tuple[bool, dict[str, int]]:
    """Run the check against a stated ledger and index state.

    ``live`` is the count of ``done`` rows that are **not** completed
    deletions, which is the number the verdict is allowed to use. It
    defaults to ``summary.done`` so a case that does not care about the
    distinction reads as before; the deletion case sets them apart.
    """
    orch = _orchestrator(memory_root)
    with (
        patch(f"{_ORCH}.count_business_rows", return_value=rows),
        patch(f"{_ORCH}.md_change_state_repo.queue_summary", return_value=summary),
        patch(
            f"{_ORCH}.md_change_state_repo.count_done_live_sources",
            return_value=summary.done if live is None else live,
        ),
    ):
        await orch._snapshot_index_rows()
    return orch._index_emptied, orch._index_rows


async def test_ledger_done_over_empty_tables_is_reported(
    runtime: MemoryRoot,
) -> None:
    """The #501 state itself: 106 files done, nothing pending, zero rows."""
    emptied, rows = await _verdict(
        runtime,
        rows={"episode": 0, "atomic_fact": 0},
        summary=_summary(pending=0, done=106),
    )
    assert emptied is True
    assert rows == {"episode": 0, "atomic_fact": 0}


async def test_rows_present_is_not_reported(runtime: MemoryRoot) -> None:
    """A healthy index: same ledger, but the tables hold something."""
    emptied, _ = await _verdict(
        runtime,
        rows={"episode": 1060, "atomic_fact": 0},
        summary=_summary(pending=0, done=106),
    )
    assert emptied is False


async def test_work_still_pending_is_not_reported(runtime: MemoryRoot) -> None:
    """Mid-drain, or right after a reconcile re-armed: empty but catching up.

    ``done`` is deliberately non-zero. An earlier version of this case used
    ``done=0`` and so was already excluded by the "anything indexed at all"
    condition -- removing the pending condition left it green, which is how
    it was caught. Here only the pending condition can keep the verdict
    false.
    """
    emptied, _ = await _verdict(
        runtime,
        rows={"episode": 0},
        summary=_summary(pending=94, done=12),
    )
    assert emptied is False


async def test_fresh_root_is_not_reported(runtime: MemoryRoot) -> None:
    """Nothing indexed yet: empty tables are the correct state, not a fault."""
    emptied, _ = await _verdict(
        runtime,
        rows={"episode": 0},
        summary=_summary(pending=0, done=0),
    )
    assert emptied is False


async def test_all_sources_deleted_is_not_reported(runtime: MemoryRoot) -> None:
    """The user deleted their last markdown file; the index is correctly empty.

    ``QueueSummary.done`` still counts that file -- the reconciler keeps a
    finished delete as ``done`` with ``change_type='deleted'`` so the scanner
    does not re-enqueue it every pass. Reading ``done`` as "sources were
    indexed" makes this state indistinguishable from a wiped index, and the
    warning would stand forever while pointing at a rebuild that has nothing
    to rebuild from. Only the live-source count separates the two, which is
    why ``live`` is zero here while ``done`` is not.
    """
    emptied, _ = await _verdict(
        runtime,
        rows={"episode": 0},
        summary=_summary(pending=0, done=7),
        live=0,
    )
    assert emptied is False


async def test_a_backend_that_cannot_count_does_not_stop_startup(
    runtime: MemoryRoot,
) -> None:
    """Counting is diagnostics; it must not take the cascade down with it."""
    orch = _orchestrator(runtime)
    with patch(f"{_ORCH}.count_business_rows", side_effect=RuntimeError("no")):
        await orch._snapshot_index_rows()
    assert orch._index_emptied is False
    assert orch._index_rows == {}

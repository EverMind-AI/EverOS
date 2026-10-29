"""Round-2 finding M8: partial success is its own exit code.

A vectors phase that wrote most rows but could not write some leaves data
behind for another pass. Automation must tell that apart from a clean run:
the run ends ``COMPLETED_WITH_FAILURES`` and ``run_backfill`` exits ``4``
with a hint naming the per-row log event; a clean run exits ``0`` without
it. Only the vectors phase counts rows, so a summary without it counts zero.
"""

from __future__ import annotations

import pytest

from everos.entrypoints.cli.commands._backfill_cmd import run_backfill
from everos.service import BackfillSummary, OpsService, VectorsPhaseResult
from tests.fakes import (
    FakeEmbeddingProvider,
    InMemoryVectorTable,
    backfill_deps,
    episode_row,
    make_ops,
)


def _ops_over(table: InMemoryVectorTable) -> OpsService:
    deps = backfill_deps(embedding=FakeEmbeddingProvider(), tables={"episode": table})
    return make_ops(backfill=deps)


def _episodes(n: int) -> InMemoryVectorTable:
    return InMemoryVectorTable(
        "episode", [episode_row(f"ep_{i}", id=f"row_{i}") for i in range(n)]
    )


def test_a_summary_without_the_vectors_phase_counts_no_failed_rows() -> None:
    assert BackfillSummary().rows_failed == 0


def test_the_summary_counts_the_vectors_phase_failures() -> None:
    summary = BackfillSummary(
        vectors=VectorsPhaseResult(rows_processed=27, rows_failed=3)
    )
    assert summary.rows_failed == 3


async def test_run_backfill_returns_four_when_rows_failed(
    capsys: pytest.CaptureFixture[str],
) -> None:
    table = _episodes(5)
    table.fail_update = {"row_0", "row_1", "row_2"}

    code = await run_backfill(_ops_over(table), phase="vectors", auto_yes=True)
    out = capsys.readouterr().out

    assert code == 4
    assert "COMPLETED_WITH_FAILURES" in out
    assert "3 rows failed embedding" in out
    assert "cascade_backfill_row_embed_failed" in out


async def test_run_backfill_returns_zero_when_every_row_is_written(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = await run_backfill(_ops_over(_episodes(5)), phase="vectors", auto_yes=True)
    out = capsys.readouterr().out

    assert code == 0
    assert "Exit: SUCCESS" in out
    assert "COMPLETED_WITH_FAILURES" not in out
    assert "rows failed embedding" not in out

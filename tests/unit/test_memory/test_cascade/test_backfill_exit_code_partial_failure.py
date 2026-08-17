"""Round-2 finding M8: partial-success exit code (COMPLETED_WITH_FAILURES).

Before M8, ``run_backfill`` returned exit ``0`` on every path that
didn't raise — including the case where Phase 1 wrote most rows but
tallied some as failed. Automation could not tell "clean success"
from "partial success — data left behind that needs another pass".

Round-2 adds exit code ``4`` (COMPLETED_WITH_FAILURES). At the end of
a run, if the summary carries ``rows_failed > 0`` across any phase
(currently only Phase 1 exposes a row-level counter), ``run_backfill``
returns ``4`` instead of ``0`` and the printed summary shows the
"COMPLETED_WITH_FAILURES" label plus a hint pointing at the per-row
log events.

This file pins:

- Exit 4 fires when ``vectors.rows_failed > 0``.
- Exit 0 stays when every phase's ``rows_failed`` is zero.
- The ``_EXIT_LABELS`` mapping carries 4 → ``"COMPLETED_WITH_FAILURES"``.
- ``_count_failed_rows`` sums Phase 1's counter today; a missing phase
  contributes zero (single-``--phase`` invocations don't inflate the
  count).
"""

from __future__ import annotations

import pytest

from everos.entrypoints.cli.commands._backfill_cmd import _EXIT_LABELS, run_backfill
from everos.memory.cascade._backfill import (
    _BackfillSummary,
    _count_failed_rows,
    _PhaseResult,
)


def test_exit_label_four_is_completed_with_failures() -> None:
    """The new label must map to the human-readable string automation
    and operators will look for in logs and CI output."""
    assert _EXIT_LABELS[4] == "COMPLETED_WITH_FAILURES"
    # Existing codes stay intact — the change only adds a new key.
    assert _EXIT_LABELS[0] == "SUCCESS"
    assert _EXIT_LABELS[1] == "ABORTED"
    assert _EXIT_LABELS[2] == "FAILED"
    assert _EXIT_LABELS[3] == "SERVER_RUNNING"
    assert _EXIT_LABELS[130] == "INTERRUPTED"


def test_count_failed_rows_returns_zero_when_no_phase_ran() -> None:
    """A fresh, empty summary has no ``vectors`` field yet — the sum
    must fall to zero rather than raise on the ``None`` access."""
    assert _count_failed_rows(_BackfillSummary()) == 0


def test_count_failed_rows_sums_phase_one_rows_failed() -> None:
    """Phase 1 is the only phase with a row-level failure counter today.
    ``_count_failed_rows`` must surface it verbatim so ``run_backfill``
    can pick the right exit code."""
    summary = _BackfillSummary(vectors=_PhaseResult(rows_processed=27, rows_failed=3))
    assert _count_failed_rows(summary) == 3


async def test_run_backfill_returns_four_when_rows_failed_gt_zero(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """End-to-end: stub ``_run_phase_vectors`` to return a phase result
    with 3 rows failed. ``run_backfill`` must (a) return ``4``, (b) print
    the ``COMPLETED_WITH_FAILURES`` label, (c) print the "N rows failed
    embedding" hint pointing at the log event name."""
    from everos.memory.cascade import _backfill as backfill_mod

    async def fake_phase(*, auto_yes: bool, presenter: object) -> _PhaseResult:
        return _PhaseResult(rows_processed=29, rows_failed=3, tokens_embedded=123)

    monkeypatch.setattr(backfill_mod, "_run_phase_vectors", fake_phase)

    code = await run_backfill(phase="vectors", auto_yes=True)
    out = capsys.readouterr().out

    assert code == 4
    assert "COMPLETED_WITH_FAILURES" in out
    assert "3 rows failed embedding" in out
    assert "cascade_backfill_row_embed_failed" in out


async def test_run_backfill_returns_zero_when_all_rows_succeed(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Regression: exit code 0 stays reserved for a clean Phase 1 run
    (``rows_failed == 0``). The COMPLETED_WITH_FAILURES hint must NOT
    appear on this path."""
    from everos.memory.cascade import _backfill as backfill_mod

    async def fake_phase(*, auto_yes: bool, presenter: object) -> _PhaseResult:
        return _PhaseResult(rows_processed=10, rows_failed=0, tokens_embedded=42)

    monkeypatch.setattr(backfill_mod, "_run_phase_vectors", fake_phase)

    code = await run_backfill(phase="vectors", auto_yes=True)
    out = capsys.readouterr().out

    assert code == 0
    assert "Exit: SUCCESS" in out
    assert "COMPLETED_WITH_FAILURES" not in out
    assert "rows failed embedding" not in out

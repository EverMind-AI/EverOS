"""Round-2 finding M9: optimize() after Phase 1 backfill loop.

Phase 1 writes each row's ``vector`` (and Episode's ``subject_vector``)
through a partial-column ``update`` — one LanceDB transaction, one
manifest version per row. A 50k-row backfill therefore opens 50k
fragments; without a compaction pass the on-disk directory grows
unbounded. Same failure mode as v1.1.3's FTS optimize gap
(#336 / lance-format/lance#7653): eventual query slowdown, disk bloat,
and eventual ``optimize()`` breakage.

Round-2 adds a single ``optimize()`` call at the end of
``_backfill_table``, gated on ``rows_processed > 0`` so no-op backlogs
don't pay the compact cost. The per-row update loop stays intact —
Batch 2b's per-row failure isolation depends on per-row atomicity, so
Option A (read-modify-upsert batching) was rejected in favour of
Option B (leave writes, compact after).

This file pins:

- ``optimize()`` fires exactly once when any row was written, and its
  ``cascade_backfill_table_optimized`` event names the table.
- ``optimize()`` does NOT fire when the backlog processed zero rows
  (empty backlog / every row failed).
- ``optimize()`` raising does not abort the phase — the returned
  counters stay accurate and a ``cascade_backfill_table_optimize_failed``
  warning event fires, naming the table so an operator can grep.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import pytest

from everos.core.observability.logging import configure_logging
from everos.memory.cascade._backfill import (
    NullBackfillPresenter,
    _backfill_table,
    _NullVectorRow,
    _TableBacklog,
    _TableSpec,
)
from everos.memory.cascade.worker import (
    DEFAULT_OPTIMIZE_PRUNE_RETENTION_SECONDS,
)


@pytest.fixture(autouse=True)
def _wire_structlog_to_stdlib() -> None:
    """Unit tests bypass the CLI entry, so the structlog → stdlib bridge
    is uninitialised. Wire it up at INFO so caplog sees both the INFO
    (optimize success) and WARNING (optimize failure) events emitted by
    the optimize path."""
    configure_logging(level="INFO")


class _FakeSchema:
    """Stand-in for ``BaseLanceTable`` — only ``TABLE_NAME`` is read."""

    TABLE_NAME = "fake_table"


class _FakeRepo:
    """Records ``update`` / ``optimize`` / ``prune`` calls; can be told to
    fail an operation to model poison writes / failing maintenance.

    The ``optimize`` / ``prune`` signatures MUST mirror the real
    ``LanceRepoBase`` exactly. A stale double here previously kept a removed
    ``optimize(cleanup_older_than=…)`` kwarg after the repo API split
    compact (``optimize()``) from reclaim (``prune()``); the fake happily
    accepted the old call while the real ``optimize`` raised ``TypeError``,
    hiding a silent backfill regression from CI (review P0-1).
    """

    def __init__(
        self,
        *,
        update_fails: bool = False,
        optimize_fails: bool = False,
        prune_fails: bool = False,
    ) -> None:
        self.update_fails = update_fails
        self.optimize_fails = optimize_fails
        self.prune_fails = prune_fails
        self.update_calls: list[tuple[dict[str, Any], str]] = []
        self.optimize_calls = 0
        self.prune_calls = 0
        self.last_prune_older_than: dt.timedelta | None = None

    async def update(self, values: dict[str, Any], *, where: str) -> None:
        self.update_calls.append((values, where))
        if self.update_fails:
            raise RuntimeError("simulated per-row write failure")

    async def optimize(self) -> None:
        self.optimize_calls += 1
        if self.optimize_fails:
            raise RuntimeError("simulated optimize failure (e.g. lock contention)")

    async def prune(self, older_than: dt.timedelta) -> None:
        self.prune_calls += 1
        self.last_prune_older_than = older_than
        if self.prune_fails:
            raise RuntimeError("simulated prune failure (e.g. lock contention)")


class _HappyProvider:
    """``embed_batch`` returns deterministic vectors. Never falls back."""

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[float(i)] * 3 for i, _ in enumerate(texts)]

    async def embed(self, text: str) -> list[float]:  # pragma: no cover
        return [0.0, 0.0, 0.0]


class _AlwaysFailProvider:
    """Both batch and per-row embed always raise — every row fails."""

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("simulated batch failure")

    async def embed(self, text: str) -> list[float]:
        raise RuntimeError("simulated per-row failure")


def _row(row_id: str, text: str) -> _NullVectorRow:
    return _NullVectorRow(
        id=row_id,
        text=text,
        subject_text=None,
        tokens=len(text.split()),
        needs_primary=True,
        needs_subject=False,
    )


def _backlog(repo: _FakeRepo, rows: list[_NullVectorRow]) -> _TableBacklog:
    spec = _TableSpec(
        schema=_FakeSchema,  # type: ignore[arg-type]
        repo=repo,  # type: ignore[arg-type]
        text_of=lambda r: r["text"],
        subject_of=None,
    )
    return _TableBacklog(spec=spec, rows=rows)


async def test_backfill_table_calls_optimize_when_rows_processed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Happy path: rows are embedded and written; optimize fires exactly
    once at the end and does not fire the failure warning.

    (The ``cascade_backfill_table_optimized`` INFO log itself is emitted
    inside the try-block but is not asserted here — structlog's
    ``cache_logger_on_first_use=True`` can leave the module-level logger
    proxy pinned to a WARNING wrapper by a prior test in the same
    process, silently dropping INFO. The load-bearing assertion is
    ``repo.optimize_calls == 1``.)"""
    repo = _FakeRepo()
    backlog = _backlog(repo, [_row(f"r{i}", f"text {i}") for i in range(3)])

    with caplog.at_level("WARNING", logger="everos.memory.cascade._backfill"):
        result = await _backfill_table(  # type: ignore[arg-type]
            backlog, _HappyProvider(), presenter=NullBackfillPresenter()
        )

    assert result.rows_processed == 3
    assert result.rows_failed == 0
    assert len(repo.update_calls) == 3
    assert repo.optimize_calls == 1
    # Compact then reclaim: prune fires once — this is what the removed
    # ``cleanup_older_than`` kwarg used to do inline (review P0-1).
    assert repo.prune_calls == 1
    # It must pass the daemon's retention window, NOT zero: backfill runs in a
    # separate process, so the in-process write lock cannot fence a daemon
    # /search that still holds a reference to a just-superseded version.
    # Reclaiming at zero age can delete files out from under that read.
    assert repo.last_prune_older_than == dt.timedelta(
        seconds=DEFAULT_OPTIMIZE_PRUNE_RETENTION_SECONDS
    )
    # Happy path must not fire the failure log.
    assert "cascade_backfill_table_optimize_failed" not in caplog.text


async def test_backfill_table_skips_optimize_when_no_rows_written() -> None:
    """No rows advanced (every embed failed) → optimize must not fire.
    The compact cost is not worth paying on a no-op backlog."""
    repo = _FakeRepo()
    backlog = _backlog(repo, [_row(f"r{i}", f"text {i}") for i in range(2)])

    result = await _backfill_table(  # type: ignore[arg-type]
        backlog, _AlwaysFailProvider(), presenter=NullBackfillPresenter()
    )

    assert result.rows_processed == 0
    assert result.rows_failed == 2
    assert len(repo.update_calls) == 0
    assert repo.optimize_calls == 0
    assert repo.prune_calls == 0


async def test_backfill_table_skips_optimize_when_backlog_is_empty() -> None:
    """Defensive: an empty backlog also short-circuits optimize. The
    scan filter already drops these, but the gate must hold if a caller
    hands us one anyway."""
    repo = _FakeRepo()
    backlog = _backlog(repo, [])

    result = await _backfill_table(  # type: ignore[arg-type]
        backlog, _HappyProvider(), presenter=NullBackfillPresenter()
    )

    assert result.rows_processed == 0
    assert repo.optimize_calls == 0
    assert repo.prune_calls == 0


async def test_backfill_table_optimize_failure_does_not_abort(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """optimize() is best-effort maintenance — a raising ``optimize`` must
    not invalidate the writes or lose the row counters. Failure logs a
    warning event that names the table so operators can grep for it."""
    repo = _FakeRepo(optimize_fails=True)
    backlog = _backlog(repo, [_row(f"r{i}", f"text {i}") for i in range(3)])

    with caplog.at_level("WARNING", logger="everos.memory.cascade._backfill"):
        result = await _backfill_table(  # type: ignore[arg-type]
            backlog, _HappyProvider(), presenter=NullBackfillPresenter()
        )

    # Writes stay committed; counters reflect the real work.
    assert result.rows_processed == 3
    assert result.rows_failed == 0
    assert len(repo.update_calls) == 3
    # Optimize was attempted and raised — the failure log names the table.
    assert repo.optimize_calls == 1
    # optimize() raised before prune() could run.
    assert repo.prune_calls == 0
    assert "cascade_backfill_table_optimize_failed" in caplog.text
    assert "fake_table" in caplog.text


async def test_backfill_table_prune_failure_does_not_abort(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """prune() is best-effort maintenance too — a raising ``prune`` (after a
    clean ``optimize``) must not invalidate the writes or lose the counters;
    it logs the same failure warning naming the table."""
    repo = _FakeRepo(prune_fails=True)
    backlog = _backlog(repo, [_row(f"r{i}", f"text {i}") for i in range(3)])

    with caplog.at_level("WARNING", logger="everos.memory.cascade._backfill"):
        result = await _backfill_table(  # type: ignore[arg-type]
            backlog, _HappyProvider(), presenter=NullBackfillPresenter()
        )

    assert result.rows_processed == 3
    assert result.rows_failed == 0
    assert len(repo.update_calls) == 3
    assert repo.optimize_calls == 1
    assert repo.prune_calls == 1
    assert "cascade_backfill_table_optimize_failed" in caplog.text
    assert "fake_table" in caplog.text

"""Unit tests for the unbackfilled-memory-rows startup hint.

Pins the shape of
:func:`everos.entrypoints.api.lifespans.lancedb._log_unbackfilled_hint`
after the round-3 revert (finding #3): the earlier "marker + limit(1)
probe" optimisation was net-zero on clean state and net-negative on
dirty state (probe hits early, then the full count runs anyway =
twice the scan), so the module now runs an unconditional
``count_rows(filter='vector IS NULL')`` per business table on startup.

Contracts pinned:

- With any NULL-vector rows across the business tables, one
  ``unbackfilled_memory_rows`` warning is emitted with the summed
  count and the ``everos cascade backfill`` hint.
- With zero NULL-vector rows across every table, the hint is
  silent — no marker, no side effects.
- A per-table ``count_rows`` failure logs
  ``unbackfilled_check_failed`` and does not interrupt startup;
  remaining tables still contribute.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import structlog.testing

from everos.entrypoints.api.lifespans import lancedb as lancedb_lifespan
from everos.infra.persistence.index import ALL_REPOS
from everos.infra.persistence.index.schema import schema_for


class _FakeRepo:
    """Minimal backend-neutral repository used by the hint."""

    def __init__(self, schema: Any, null_count: int) -> None:
        self.schema = schema
        self._null_count = null_count
        self.count_where_calls = 0

    async def count_where(self, predicate: Any) -> int:
        self.count_where_calls += 1
        return self._null_count


@pytest.fixture
def _isolated_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("EVEROS_ROOT", str(tmp_path))
    (tmp_path / ".index").mkdir(parents=True, exist_ok=True)
    return tmp_path


def _wire_tables(
    monkeypatch: pytest.MonkeyPatch, *, null_count: int
) -> dict[str, _FakeRepo]:
    """Replace the facade registry with per-schema repository stubs."""
    repos: dict[str, _FakeRepo] = {
        repo.schema.TABLE_NAME: _FakeRepo(repo.schema, null_count)
        for repo in ALL_REPOS
        if schema_for(repo.schema).vector_fields
    }
    monkeypatch.setattr(lancedb_lifespan, "ALL_REPOS", tuple(repos.values()))
    return repos


async def test_hint_fires_when_null_vectors_exist(
    _isolated_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tables = _wire_tables(monkeypatch, null_count=3)

    with structlog.testing.capture_logs() as captured:
        await lancedb_lifespan._log_unbackfilled_hint()

    emissions = [e for e in captured if e.get("event") == "unbackfilled_memory_rows"]
    assert len(emissions) == 1
    expected_total = 3 * len(tables)
    assert emissions[0]["count"] == expected_total
    # Every business table contributes exactly one ``count_rows`` call.
    assert all(t.count_where_calls == 1 for t in tables.values())


async def test_hint_silent_when_no_null_vectors(
    _isolated_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tables = _wire_tables(monkeypatch, null_count=0)

    with structlog.testing.capture_logs() as captured:
        await lancedb_lifespan._log_unbackfilled_hint()

    emissions = [e for e in captured if e.get("event") == "unbackfilled_memory_rows"]
    assert emissions == []
    # Every table was scanned (unconditional count) — none had rows to
    # report, so no banner. No marker involved.
    assert all(t.count_where_calls == 1 for t in tables.values())


async def test_per_table_failure_is_swallowed_and_logged(
    _isolated_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tables: dict[str, _FakeRepo] = {
        repo.schema.TABLE_NAME: _FakeRepo(repo.schema, null_count=1)
        for repo in ALL_REPOS
        if schema_for(repo.schema).vector_fields
    }
    poisoned = next(iter(tables))

    async def _poisoned_count(_predicate: Any) -> int:
        raise RuntimeError("simulated index hiccup")

    tables[poisoned].count_where = _poisoned_count  # type: ignore[method-assign]
    monkeypatch.setattr(lancedb_lifespan, "ALL_REPOS", tuple(tables.values()))

    with structlog.testing.capture_logs() as captured:
        await lancedb_lifespan._log_unbackfilled_hint()

    check_failed = [
        e for e in captured if e.get("event") == "unbackfilled_check_failed"
    ]
    assert len(check_failed) == 1
    # Remaining tables still contribute (1 row each) — poisoned one drops
    # out silently.
    emissions = [e for e in captured if e.get("event") == "unbackfilled_memory_rows"]
    assert len(emissions) == 1
    assert emissions[0]["count"] == len(tables) - 1

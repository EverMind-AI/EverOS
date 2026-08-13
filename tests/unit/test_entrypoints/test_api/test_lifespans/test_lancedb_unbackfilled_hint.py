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
from everos.infra.persistence.lancedb import BUSINESS_SCHEMAS_WITH_VECTOR


class _FakeTable:
    """Minimal stand-in for :class:`BaseLanceTable` used by the hint."""

    def __init__(self, null_count: int) -> None:
        self._null_count = null_count
        self.count_rows_calls = 0

    async def count_rows(self, filter: str) -> int:
        self.count_rows_calls += 1
        return self._null_count


@pytest.fixture
def _isolated_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("EVEROS_ROOT", str(tmp_path))
    (tmp_path / ".index").mkdir(parents=True, exist_ok=True)
    return tmp_path


def _wire_tables(
    monkeypatch: pytest.MonkeyPatch, *, null_count: int
) -> dict[str, _FakeTable]:
    """Replace ``get_table`` in the lifespan module with a per-table stub."""
    tables: dict[str, _FakeTable] = {
        schema.TABLE_NAME: _FakeTable(null_count)
        for schema in BUSINESS_SCHEMAS_WITH_VECTOR
    }

    async def _fake_get_table(name: str, _schema: Any) -> _FakeTable:
        return tables[name]

    monkeypatch.setattr(lancedb_lifespan, "get_table", _fake_get_table)
    return tables


async def test_hint_fires_when_null_vectors_exist(
    _isolated_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tables = _wire_tables(monkeypatch, null_count=3)

    with structlog.testing.capture_logs() as captured:
        await lancedb_lifespan._log_unbackfilled_hint()

    emissions = [e for e in captured if e.get("event") == "unbackfilled_memory_rows"]
    assert len(emissions) == 1
    expected_total = 3 * len(BUSINESS_SCHEMAS_WITH_VECTOR)
    assert emissions[0]["count"] == expected_total
    # Every business table contributes exactly one ``count_rows`` call.
    assert all(t.count_rows_calls == 1 for t in tables.values())


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
    assert all(t.count_rows_calls == 1 for t in tables.values())


async def test_per_table_failure_is_swallowed_and_logged(
    _isolated_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tables: dict[str, _FakeTable] = {
        schema.TABLE_NAME: _FakeTable(null_count=1)
        for schema in BUSINESS_SCHEMAS_WITH_VECTOR
    }
    poisoned = BUSINESS_SCHEMAS_WITH_VECTOR[0].TABLE_NAME

    async def _fake_get_table(name: str, _schema: Any) -> _FakeTable:
        if name == poisoned:
            raise RuntimeError("simulated LanceDB hiccup")
        return tables[name]

    monkeypatch.setattr(lancedb_lifespan, "get_table", _fake_get_table)

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
    assert emissions[0]["count"] == len(BUSINESS_SCHEMAS_WITH_VECTOR) - 1

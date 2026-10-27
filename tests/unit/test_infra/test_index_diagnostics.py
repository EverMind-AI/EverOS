"""Duplicate-key scan used by ``everos cascade status --duplicates``."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from everos.infra.persistence import index as index_mod


class _Repo:
    def __init__(self, table_name: str, rows: list[SimpleNamespace]) -> None:
        self.table_name = table_name
        self._rows = rows

    async def scan(self) -> list[SimpleNamespace]:
        return self._rows


def _row(key: str) -> SimpleNamespace:
    return SimpleNamespace(app_id="default", project_id="default", id=key)


async def test_count_duplicate_keys_reports_extra_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A raw write can leave two rows for one logical key; the scan reports
    # the extra row, so a table with two unique keys is not "1 duplicate".
    rows = [_row("a"), _row("a"), _row("b")]
    monkeypatch.setattr(index_mod, "ALL_REPOS", (_Repo("episode", rows),))

    assert await index_mod.count_duplicate_keys() == {"episode": 1}


async def test_count_duplicate_keys_reports_zero_for_unique_tables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [_row("a"), _row("b")]
    monkeypatch.setattr(index_mod, "ALL_REPOS", (_Repo("episode", rows),))

    assert await index_mod.count_duplicate_keys() == {"episode": 0}

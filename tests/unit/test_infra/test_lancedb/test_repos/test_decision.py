"""Decision LanceDB repo can open an empty table."""

from __future__ import annotations

from pathlib import Path

import pytest

from everos.infra.persistence.lancedb import decision_repo, lancedb_manager


@pytest.fixture
async def _real_lancedb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Spin up a clean LanceDB rooted under ``tmp_path`` for one test."""
    monkeypatch.setenv("EVEROS_ROOT", str(tmp_path))
    lancedb_manager._conn = None
    lancedb_manager._tables.clear()
    yield
    await lancedb_manager.dispose_connection()


async def test_repo_opens_empty_table(_real_lancedb: None) -> None:
    table = await decision_repo._table()
    assert await table.count_rows() == 0
    assert decision_repo.schema.TABLE_NAME == "decision"

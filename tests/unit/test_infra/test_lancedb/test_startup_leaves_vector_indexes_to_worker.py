"""``ensure_business_indexes`` runs in every process that opens the root — the
server lifespan and the CLI's ``_runtime`` — so it must never train an ANN
index: a CLI command doing that races the running server's commits (soak:
``cascade status`` storms failed with ``Retryable commit conflict`` the moment
a table crossed the row threshold). Vector indexes are the cascade worker's:
first rebuild sweep at server start, then the heavy beat.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from everos.core.persistence import BaseLanceTable
from everos.infra.persistence.lancedb import ensure_business_indexes, lancedb_manager


@pytest.fixture(autouse=True)
async def _isolated_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("EVEROS_ROOT", str(tmp_path))
    lancedb_manager._conn = None
    lancedb_manager._tables.clear()
    yield
    await lancedb_manager.dispose_connection()


async def test_startup_pass_never_trains_a_vector_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    async def spy(cls, table, *, min_rows):  # type: ignore[no-untyped-def]
        calls.append(cls.__name__)
        return []

    monkeypatch.setattr(BaseLanceTable, "ensure_vector_indexes", classmethod(spy))
    await ensure_business_indexes()
    assert calls == [], f"startup pass trained vector indexes on {calls}"

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

from everos.infra.persistence.lancedb._base import indexes
from tests.fakes.storage import Storage


@pytest.fixture(autouse=True)
async def _isolated_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, storage: Storage
):
    monkeypatch.setenv("EVEROS_ROOT", str(tmp_path))
    yield
    await storage.lance.close()


async def test_startup_pass_never_trains_a_vector_index(
    monkeypatch: pytest.MonkeyPatch, storage: Storage
) -> None:
    calls: list[str] = []

    async def spy(table, schema, *, min_rows):  # type: ignore[no-untyped-def]
        calls.append(schema.__name__)
        return []

    monkeypatch.setattr(indexes, "ensure_vector_indexes", spy)
    await storage.index.ensure_business_indexes()
    assert calls == [], f"startup pass trained vector indexes on {calls}"

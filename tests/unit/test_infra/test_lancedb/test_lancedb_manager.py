"""LanceDB manager singletons.

Verifies ``get_connection`` / ``get_table`` / ``dispose_connection``
are idempotent and rebuild after dispose.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from lancedb.pydantic import Vector

from everos.core.persistence import BaseLanceTable
from everos.infra.persistence.lancedb import lancedb_manager


class _DemoVec(BaseLanceTable):
    """Demo schema — only used by this test module."""

    text: str
    vector: Vector(3)  # type: ignore[valid-type]


@pytest.fixture(autouse=True)
async def _reset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point the singleton at an isolated memory-root and reset module state."""
    monkeypatch.setenv("EVEROS_MEMORY__ROOT", str(tmp_path))
    lancedb_manager._conn = None
    lancedb_manager._tables.clear()
    yield
    await lancedb_manager.dispose_connection()


async def test_get_connection_is_singleton() -> None:
    c1 = await lancedb_manager.get_connection()
    c2 = await lancedb_manager.get_connection()
    assert c1 is c2


async def test_get_table_creates_then_caches() -> None:
    t1 = await lancedb_manager.get_table("demo", _DemoVec)
    t2 = await lancedb_manager.get_table("demo", _DemoVec)
    assert t1 is t2
    assert "demo" in lancedb_manager._tables


async def test_get_table_reopens_existing() -> None:
    """A second connection cycle must reopen (not recreate) the table."""
    await lancedb_manager.get_table("demo", _DemoVec)
    await lancedb_manager.dispose_connection()

    t = await lancedb_manager.get_table("demo", _DemoVec)
    assert t is not None
    # Round-trip a row to prove the schema survived.
    await t.add([_DemoVec(text="hello", vector=[0.1, 0.2, 0.3])])
    assert await t.count_rows() == 1


async def test_dispose_resets_state() -> None:
    await lancedb_manager.get_connection()
    await lancedb_manager.get_table("demo", _DemoVec)
    await lancedb_manager.dispose_connection()
    assert lancedb_manager._conn is None
    assert lancedb_manager._tables == {}


async def test_dispose_is_idempotent() -> None:
    await lancedb_manager.dispose_connection()  # nothing built yet
    await lancedb_manager.get_connection()
    await lancedb_manager.dispose_connection()
    await lancedb_manager.dispose_connection()  # second call must not raise

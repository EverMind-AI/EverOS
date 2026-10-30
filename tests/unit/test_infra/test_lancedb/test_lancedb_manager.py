"""``LanceDbConnection``: one root's connection and table cache.

Building opens nothing; the connection and tables open on first use, are
cached, and come back after ``close``.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pytest

from everos.config import LanceDBSettings
from everos.infra.persistence import MemoryRoot
from everos.infra.persistence.lancedb import LanceDbConnection
from everos.infra.persistence.records import DenseVector, IndexRecord
from tests.fakes import arrow_schema


class _DemoVec(IndexRecord):
    """Demo schema — only used by this test module."""

    PRIMARY_KEY: ClassVar[tuple[str, ...]] = ("text",)

    text: str
    vector: DenseVector


_SCHEMA = arrow_schema(_DemoVec, dim=3)


@pytest.fixture
async def connection(tmp_path: Path):  # type: ignore[no-untyped-def]
    built = LanceDbConnection(MemoryRoot(tmp_path / "root"), LanceDBSettings())
    yield built
    await built.close()


def test_building_touches_nothing_on_disk(tmp_path: Path) -> None:
    LanceDbConnection(MemoryRoot(tmp_path / "root"), LanceDBSettings())
    assert not (tmp_path / "root").exists()


async def test_connect_returns_the_same_connection(
    connection: LanceDbConnection,
) -> None:
    assert await connection.connect() is await connection.connect()


async def test_table_creates_then_caches(connection: LanceDbConnection) -> None:
    t1 = await connection.table("demo", _SCHEMA)
    t2 = await connection.table("demo", _SCHEMA)
    assert t1 is t2


async def test_table_reopens_existing_after_close(
    connection: LanceDbConnection,
) -> None:
    """A second connection cycle must reopen (not recreate) the table."""
    await connection.table("demo", _SCHEMA)
    await connection.close()

    t = await connection.table("demo", _SCHEMA)
    # Round-trip a row to prove the schema survived.
    await t.add([_DemoVec(text="hello", vector=[0.1, 0.2, 0.3]).model_dump()])
    assert await t.count_rows() == 1


async def test_close_evicts_the_cache(connection: LanceDbConnection) -> None:
    t1 = await connection.table("demo", _SCHEMA)
    await connection.close()
    assert await connection.table("demo", _SCHEMA) is not t1


async def test_close_is_idempotent(connection: LanceDbConnection) -> None:
    await connection.close()  # nothing opened yet
    await connection.connect()
    await connection.close()
    await connection.close()  # second call must not raise

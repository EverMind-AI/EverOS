"""merge_insert keys on the declared PRIMARY_KEY, not on ``id`` alone.

Real temporary LanceDB, no fake repo: the fix lives in the merge key the
repository hands to ``merge_insert``, and a fake would model whichever key
semantics its author assumed.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pyarrow as pa
import pytest

from everos.config import LanceDBSettings
from everos.infra.persistence import MemoryRoot
from everos.infra.persistence.index import primary_key
from everos.infra.persistence.lancedb import (
    LanceIndexContract,
    LanceRepoBase,
    open_lancedb_connection,
)
from everos.infra.persistence.records import IndexRecord
from tests.fakes import TEST_DIM, arrow_schema


class _Scoped(IndexRecord):
    TABLE_NAME: ClassVar[str] = "_scoped"
    PRIMARY_KEY: ClassVar[tuple[str, ...]] = ("app_id", "project_id", "id")
    id: str
    app_id: str = "default"
    project_id: str = "default"
    text: str


class _ScopedRepo(LanceIndexContract[_Scoped], LanceRepoBase[_Scoped]):
    schema = _Scoped


class _Undeclared(IndexRecord):
    TABLE_NAME: ClassVar[str] = "_undeclared"
    id: str


class _UndeclaredRepo(LanceIndexContract[_Undeclared], LanceRepoBase[_Undeclared]):
    schema = _Undeclared


@pytest.fixture(autouse=True)
def _reset_locks() -> None:
    # pytest-asyncio makes a fresh loop per test; the class-level lock pool
    # must not carry a lock bound to the previous loop.
    LanceRepoBase._reset_locks_for_tests()


async def _repo(tmp_path: Path, schema, repo_cls):  # type: ignore[no-untyped-def]
    mr = MemoryRoot(tmp_path)
    mr.ensure()
    conn = await open_lancedb_connection(mr.lancedb_dir, LanceDBSettings())
    # ``_Undeclared`` has no neutral schema to derive from: that is the point.
    arrow = (
        pa.schema([pa.field("id", pa.string(), nullable=False)])
        if schema is _Undeclared
        else arrow_schema(schema, dim=TEST_DIM)
    )
    table = await conn.create_table(schema.TABLE_NAME, schema=arrow)
    return repo_cls(table=table, dim=TEST_DIM)


async def test_same_id_in_two_projects_keeps_two_rows(tmp_path: Path) -> None:
    repo = await _repo(tmp_path, _Scoped, _ScopedRepo)
    await repo.upsert([_Scoped(id="u1_ep_1", project_id="alpha", text="rust")])
    await repo.upsert([_Scoped(id="u1_ep_1", project_id="beta", text="go")])
    assert await repo.count() == 2
    alpha = await repo.find_one_where(primary_key("default", "alpha", "u1_ep_1"))
    assert alpha is not None and alpha.text == "rust"


async def test_same_key_twice_updates_in_place(tmp_path: Path) -> None:
    repo = await _repo(tmp_path, _Scoped, _ScopedRepo)
    await repo.upsert([_Scoped(id="u1_ep_1", project_id="alpha", text="v1")])
    await repo.upsert([_Scoped(id="u1_ep_1", project_id="alpha", text="v2")])
    assert await repo.count() == 1
    row = await repo.find_one_where(primary_key("default", "alpha", "u1_ep_1"))
    assert row is not None and row.text == "v2"


async def test_upsert_rejects_undeclared_primary_key(tmp_path: Path) -> None:
    repo = await _repo(tmp_path, _Undeclared, _UndeclaredRepo)
    with pytest.raises(ValueError, match="_Undeclared has no PRIMARY_KEY"):
        await repo.upsert([_Undeclared(id="x")])

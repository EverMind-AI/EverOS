"""``verify_business_schemas`` startup guard.

Regression coverage for EverOS #337: the guard must catch a column
whose on-disk Arrow *type* drifted from the current schema (not only a
missing/extra column name), and must NOT false-positive on a healthy
table freshly built from the current schema.

White-box surfaces: builds LanceDB tables directly on disk under an
isolated ``EVEROS_ROOT`` and drives ``verify_business_schemas`` /
``get_table`` against them.
"""

from __future__ import annotations

from pathlib import Path

import lancedb
import pyarrow as pa
import pytest

from everos.core.persistence import MemoryRoot
from everos.infra.persistence.lancedb import (
    LanceDBSchemaMismatchError,
    drop_business_tables,
    ensure_business_indexes,
    get_connection,
    get_table,
    lancedb_manager,
    verify_business_schemas,
)
from everos.infra.persistence.lancedb.tables.episode import Episode


@pytest.fixture(autouse=True)
async def _isolated_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point the manager singleton at an isolated memory root."""
    monkeypatch.setenv("EVEROS_ROOT", str(tmp_path))
    lancedb_manager._conn = None
    lancedb_manager._tables.clear()
    yield
    await lancedb_manager.dispose_connection()


async def _create_episode_table_on_disk(arrow_schema: pa.Schema) -> None:
    """Create the ``episode`` table straight on disk, then drop the
    connection so the manager reopens it from disk on next use."""
    root = MemoryRoot.resolve()
    root.ensure()
    conn = await lancedb.connect_async(str(root.lancedb_dir))
    await conn.create_table("episode", schema=arrow_schema)
    conn.close()
    lancedb_manager._conn = None
    lancedb_manager._tables.clear()


def _episode_schema_with_subject_vector_as(dtype: pa.DataType) -> pa.Schema:
    """Current episode Arrow schema, but subject_vector forced to ``dtype``."""
    return pa.schema(
        [
            pa.field("subject_vector", dtype, nullable=True)
            if f.name == "subject_vector"
            else f
            for f in Episode.to_arrow_schema()
        ]
    )


async def test_verify_passes_on_healthy_tables() -> None:
    """A table built from the current schema must NOT trip the guard.

    Guards against a type comparison that false-positives (e.g. on the
    fixed_size_list item name or the datetime tz rewrite)."""
    # verify_business_schemas creates every business table fresh via
    # get_table, then re-reads and compares — so a clean run proves no
    # false positive across ALL business schemas.
    await verify_business_schemas()
    # A second run over the now-existing tables must also pass.
    await verify_business_schemas()


async def test_verify_raises_on_subject_vector_string_drift() -> None:
    """#337: subject_vector left as `string` must be caught by type."""
    await _create_episode_table_on_disk(
        _episode_schema_with_subject_vector_as(pa.string())
    )
    with pytest.raises(LanceDBSchemaMismatchError) as exc:
        await verify_business_schemas()
    msg = str(exc.value)
    assert "episode" in msg
    assert "type_drift" in msg
    assert "subject_vector" in msg


async def test_verify_raises_on_subject_vector_null_drift() -> None:
    """A `null`-typed subject_vector (data-inferred all-None) is also drift."""
    await _create_episode_table_on_disk(
        _episode_schema_with_subject_vector_as(pa.null())
    )
    with pytest.raises(LanceDBSchemaMismatchError) as exc:
        await verify_business_schemas()
    assert "subject_vector" in str(exc.value)


async def test_verify_raises_on_missing_column() -> None:
    """A table missing a current column is caught by name (unchanged)."""
    reduced = pa.schema(
        [f for f in Episode.to_arrow_schema() if f.name != "subject_vector"]
    )
    await _create_episode_table_on_disk(reduced)
    with pytest.raises(LanceDBSchemaMismatchError) as exc:
        await verify_business_schemas()
    assert "missing" in str(exc.value)
    assert "subject_vector" in str(exc.value)


async def test_drop_business_tables_removes_then_recreatable() -> None:
    """drop_business_tables drops existing business tables + clears the cache;
    the next get_table recreates a fresh, current-schema table."""
    # Materialise a drifted episode table + the rest of the business set.
    await _create_episode_table_on_disk(
        _episode_schema_with_subject_vector_as(pa.string())
    )
    await ensure_business_indexes()  # create the remaining business tables
    conn = await get_connection()
    assert "episode" in set((await conn.list_tables()).tables)

    dropped = await drop_business_tables()

    assert "episode" in dropped
    conn = await get_connection()
    assert "episode" not in set((await conn.list_tables()).tables)
    assert "episode" not in lancedb_manager._tables  # cache evicted
    # Recreated fresh from the current schema → correct vector type, not string.
    tbl = await get_table("episode", Episode)
    assert (
        (await tbl.schema())
        .field("subject_vector")
        .type.equals(Episode.to_arrow_schema().field("subject_vector").type)
    )

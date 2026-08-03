"""LanceDB business table schema migration (v2: vector nullability).

Exercises real LanceDB: builds a pre-migration ("v1-shape") physical
table with a non-nullable ``vector`` column, runs
``migrate_table_schemas``, and asserts the on-disk pyarrow schema is
altered in place plus the ``.table_schema_version`` marker is written.
Also proves idempotency — a second call must not re-issue
``alter_columns`` once the marker is at the target version.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pytest

from everos.infra.persistence.lancedb import (
    _TABLE_SCHEMA_VERSION,
    BUSINESS_SCHEMAS_WITH_VECTOR,
    AtomicFact,
    Episode,
    LanceDBMigrationError,
    lancedb_manager,
    migrate_table_schemas,
)


def _v1_shape_schema() -> pa.Schema:
    """Minimal pre-migration physical shape: ``vector`` is non-nullable."""
    return pa.schema(
        [
            pa.field("id", pa.string(), nullable=False),
            pa.field(
                "vector",
                pa.list_(pa.field("item", pa.float32(), nullable=True), 1024),
                nullable=False,
            ),
        ]
    )


@pytest.fixture(autouse=True)
async def _reset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point the singleton connection at an isolated memory-root."""
    monkeypatch.setenv("EVEROS_ROOT", str(tmp_path))
    lancedb_manager._conn = None
    lancedb_manager._tables.clear()
    yield
    await lancedb_manager.dispose_connection()


async def _create_v1_table(name: str) -> None:
    conn = await lancedb_manager.get_connection()
    await conn.create_table(name, schema=_v1_shape_schema())


async def test_migration_alters_all_business_tables_to_nullable() -> None:
    for schema in BUSINESS_SCHEMAS_WITH_VECTOR:
        await _create_v1_table(schema.TABLE_NAME)

    await migrate_table_schemas()

    for schema in BUSINESS_SCHEMAS_WITH_VECTOR:
        table = await lancedb_manager.get_table(schema.TABLE_NAME, schema)
        arrow_schema = await table.schema()
        assert arrow_schema.field("vector").nullable is True


async def test_migration_writes_marker_with_target_version(tmp_path: Path) -> None:
    await _create_v1_table(Episode.TABLE_NAME)

    await migrate_table_schemas()

    marker = tmp_path / ".index" / "lancedb" / ".table_schema_version"
    assert marker.exists()
    assert int(marker.read_text().strip()) == _TABLE_SCHEMA_VERSION


async def test_migration_idempotent_second_call_skips_alter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Second call short-circuits at the marker; ``alter_columns`` is not
    re-issued."""
    await _create_v1_table(Episode.TABLE_NAME)
    await migrate_table_schemas()

    table = await lancedb_manager.get_table(Episode.TABLE_NAME, Episode)
    calls: list[object] = []
    original_alter_columns = table.alter_columns

    async def _spy_alter_columns(*alterations: object) -> object:
        calls.append(alterations)
        return await original_alter_columns(*alterations)

    monkeypatch.setattr(table, "alter_columns", _spy_alter_columns)

    await migrate_table_schemas()

    assert calls == []


async def test_migrate_table_schemas_raises_on_alter_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Any table's failing ``alter_columns`` must fail startup loudly:
    :class:`LanceDBMigrationError` is raised, the version marker stays
    unwritten, and the message names the failed table plus escalating
    recovery hints (restart, then ``cascade rebuild``). Cascade would
    otherwise write
    NULL vectors into a still-NOT-NULL column and silently drop every
    row."""
    for schema in BUSINESS_SCHEMAS_WITH_VECTOR:
        await _create_v1_table(schema.TABLE_NAME)

    episode_table = await lancedb_manager.get_table(Episode.TABLE_NAME, Episode)

    async def _raise_alter_columns(*alterations: object) -> object:
        raise RuntimeError("simulated alter_columns failure")

    monkeypatch.setattr(episode_table, "alter_columns", _raise_alter_columns)

    with pytest.raises(LanceDBMigrationError) as excinfo:
        await migrate_table_schemas()

    message = str(excinfo.value)
    assert Episode.TABLE_NAME in message
    lancedb_dir = tmp_path / ".index" / "lancedb"
    assert str(lancedb_dir) in message
    # Escalating recovery: restart first, `cascade rebuild` second. Never
    # "delete the index dir" — that leaves the queue done and the index empty.
    restart_idx = message.find("restart the process")
    rebuild_idx = message.find("everos cascade rebuild")
    assert restart_idx != -1 and rebuild_idx != -1
    assert restart_idx < rebuild_idx
    assert "wipe the index directory" not in message

    marker = lancedb_dir / ".table_schema_version"
    assert not marker.exists()

    # Sibling tables reached in the loop before the failure keep the
    # per-table alter that already succeeded — only the marker write
    # (i.e. "the migration as a whole is done") is gated.
    atomic_fact_table = await lancedb_manager.get_table(
        AtomicFact.TABLE_NAME, AtomicFact
    )
    atomic_fact_schema = await atomic_fact_table.schema()
    assert atomic_fact_schema.field("vector").nullable is True

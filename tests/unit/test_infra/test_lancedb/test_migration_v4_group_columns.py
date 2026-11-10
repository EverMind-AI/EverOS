"""LanceDB v4: an existing store gains the group-memory columns in place.

A 2.0 store built before group support opens as before: ``episode`` and
``image_memory`` get ``subject_ids`` / ``origin`` / ``source_start`` /
``source_end`` (and ``image_memory.sender_ids``) as nullable columns, its rows
keep their values and read the empty defaults, and the schema check passes
afterwards (docs/group-memory.md, section 6.4).
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import pyarrow as pa

from everos.infra.persistence import MemoryRoot
from everos.infra.persistence.records import Episode, ImageMemory
from tests.fakes.storage import arrow_schema, make_storage

_NEW = {
    Episode: ("subject_ids", "origin", "source_start", "source_end"),
    ImageMemory: ("sender_ids", "subject_ids", "origin", "source_start", "source_end"),
}


def _without(schema: pa.Schema, names: tuple[str, ...]) -> pa.Schema:
    return pa.schema([f for f in schema if f.name not in names])


async def test_v4_adds_the_group_columns_to_a_v3_store(tmp_path: Path) -> None:
    root = MemoryRoot(tmp_path)
    root.ensure()
    storage = make_storage(root)
    try:
        conn = await storage.lance.connect()
        for record, names in _NEW.items():
            await conn.create_table(
                record.TABLE_NAME, schema=_without(arrow_schema(record), names)
            )
        episode = await conn.open_table(Episode.TABLE_NAME)
        old_shape = _without(arrow_schema(Episode), _NEW[Episode])
        await episode.add(
            pa.Table.from_pylist(
                [
                    {
                        "created_at": _dt.datetime(2026, 1, 1, tzinfo=_dt.UTC),
                        "updated_at": _dt.datetime(2026, 1, 1, tzinfo=_dt.UTC),
                        "id": "u1_ep_20260101_00000001",
                        "entry_id": "ep_20260101_00000001",
                        "owner_id": "u1",
                        "owner_type": "user",
                        "app_id": "default",
                        "project_id": "default",
                        "timestamp": _dt.datetime(2026, 1, 1, tzinfo=_dt.UTC),
                        "parent_type": "memcell",
                        "parent_id": "mc_1",
                        "sender_ids": ["u1"],
                        "image_ids": [],
                        "episode": "hello",
                        "episode_tokens": "hello",
                        "md_path": "x.md",
                        "content_sha256": "0",
                    }
                ],
                schema=old_shape,
            )
        )
        (tmp_path / ".index" / "lancedb" / ".table_schema_version").write_text("3")

        await storage.backend.migrate_table_schemas()

        for record, names in _NEW.items():
            table = await conn.open_table(record.TABLE_NAME)
            on_disk = await table.schema()
            for name in names:
                assert name in on_disk.names, (record.TABLE_NAME, name)
                assert on_disk.field(name).nullable
        await storage.backend.verify_business_schemas()
        rows = await (await conn.open_table(Episode.TABLE_NAME)).query().to_list()
        assert rows[0]["episode"] == "hello"
        assert rows[0]["subject_ids"] is None
        assert rows[0]["source_start"] is None
    finally:
        await storage.aclose()

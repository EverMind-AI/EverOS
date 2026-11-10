"""LanceDB v5: an existing store gains the rest of the group columns in place.

docs/group-memory.md, sections 6.3 and 9.1: member-in-group profiles are rows
of the ``user_profile`` table with the member in ``subject_id``, and an agent
case learned in a group carries ``source_group_id``. A store at v4 opens as
before: both columns are added nullable, existing rows keep their values and
read ``None``, and the schema check passes afterwards.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import pyarrow as pa

from everos.infra.persistence import MemoryRoot
from everos.infra.persistence.records import AgentCase, UserProfile
from tests.fakes.storage import arrow_schema, make_storage


async def test_v5_adds_subject_id_to_a_v4_store(tmp_path: Path) -> None:
    root = MemoryRoot(tmp_path)
    root.ensure()
    storage = make_storage(root)
    try:
        conn = await storage.lance.connect()
        old_shape = pa.schema(
            [f for f in arrow_schema(UserProfile) if f.name != "subject_id"]
        )
        await conn.create_table(UserProfile.TABLE_NAME, schema=old_shape)
        await conn.create_table(
            AgentCase.TABLE_NAME,
            schema=pa.schema(
                [f for f in arrow_schema(AgentCase) if f.name != "source_group_id"]
            ),
        )
        table = await conn.open_table(UserProfile.TABLE_NAME)
        at = _dt.datetime(2026, 1, 1, tzinfo=_dt.UTC)
        await table.add(
            pa.Table.from_pylist(
                [
                    {
                        "created_at": at,
                        "updated_at": at,
                        "id": "u1",
                        "owner_id": "u1",
                        "owner_type": "user",
                        "app_id": "default",
                        "project_id": "default",
                        "summary": "likes tea",
                        "explicit_info_json": "[]",
                        "implicit_traits_json": "[]",
                        "profile_timestamp_ms": 1,
                        "md_path": "x.md",
                        "content_sha256": "0",
                    }
                ],
                schema=old_shape,
            )
        )
        (tmp_path / ".index" / "lancedb" / ".table_schema_version").write_text("4")

        await storage.backend.migrate_table_schemas()

        on_disk = await (await conn.open_table(UserProfile.TABLE_NAME)).schema()
        assert "subject_id" in on_disk.names
        assert on_disk.field("subject_id").nullable
        cases = await (await conn.open_table(AgentCase.TABLE_NAME)).schema()
        assert cases.field("source_group_id").nullable
        await storage.backend.verify_business_schemas()
        rows = await (await conn.open_table(UserProfile.TABLE_NAME)).query().to_list()
        assert rows[0]["summary"] == "likes tea"
        assert rows[0]["subject_id"] is None
    finally:
        await storage.aclose()

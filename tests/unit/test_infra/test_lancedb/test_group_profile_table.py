"""A store opened before the group profile existed gains its table, empty.

docs/group-memory.md, section 6.3: ``group_profile_item`` is a new table, not
new columns, so it takes no step of ``migrate_table_schemas``. A store whose
markers already say "current" (table schema 5, FTS 2) must open as it is:
verification creates the missing table from its record and finds no drift,
the FTS pass builds the table's BM25 index, the markers stay, and the rows the
store already had are untouched.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

from everos.infra.persistence import MemoryRoot
from everos.infra.persistence.index import BUSINESS_SCHEMAS
from everos.infra.persistence.records import Episode, GroupProfileItem
from tests.fakes.storage import arrow_schema, make_storage

_TABLE_SCHEMA = "5"
"""The table-schema marker group memory step 1 left; no step follows it."""
_FTS = "2"


async def test_a_current_store_opens_with_the_new_empty_table(tmp_path: Path) -> None:
    root = MemoryRoot(tmp_path)
    root.ensure()
    storage = make_storage(root)
    try:
        conn = await storage.lance.connect()
        for schema in BUSINESS_SCHEMAS:
            if schema is not GroupProfileItem:
                await conn.create_table(schema.TABLE_NAME, schema=arrow_schema(schema))
        await storage.repos.episode.upsert([_episode()])
        lancedb_dir = tmp_path / ".index" / "lancedb"
        (lancedb_dir / ".table_schema_version").write_text(_TABLE_SCHEMA)
        (lancedb_dir / ".fts_index_version").write_text(_FTS)
        assert GroupProfileItem.TABLE_NAME not in (await conn.list_tables()).tables

        await storage.backend.verify_business_schemas()
        await storage.backend.ensure_business_indexes()

        assert GroupProfileItem.TABLE_NAME in (await conn.list_tables()).tables
        assert await storage.repos.group_profile_item.count() == 0
        table = await conn.open_table(GroupProfileItem.TABLE_NAME)
        indexed = {
            column for index in await table.list_indices() for column in index.columns
        }
        assert "statement_tokens" in indexed
        assert (lancedb_dir / ".table_schema_version").read_text() == _TABLE_SCHEMA
        assert await storage.repos.episode.count() == 1
        await storage.backend.verify_business_schemas()  # still no drift
    finally:
        await storage.aclose()


def _episode() -> Episode:
    at = _dt.datetime(2026, 1, 1, tzinfo=_dt.UTC)
    return Episode(
        id="u1_ep_20260101_00000001",
        entry_id="ep_20260101_00000001",
        owner_id="u1",
        owner_type="user",
        timestamp=at,
        parent_id="mc_1",
        sender_ids=["u1"],
        episode="lunch with Betty",
        episode_tokens="lunch betty",
        md_path="default_app/default_project/users/u1/episodes/episode-2026-01-01.md",
        content_sha256="0",
    )

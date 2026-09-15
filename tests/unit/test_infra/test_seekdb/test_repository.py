"""Pin generated write and search SQL without requiring a SeekDB server."""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

import pytest

from everos.config import load_settings
from everos.infra.persistence.index import Episode, eq
from everos.infra.persistence.index.schema import schema_for
from everos.infra.persistence.seekdb import episode_repo
from everos.infra.persistence.seekdb import repository as repository_module
from everos.infra.persistence.seekdb.errors import SeekdbSchemaMismatchError
from everos.infra.persistence.seekdb.repository import SeekdbRepoBase
from everos.infra.persistence.seekdb.schema import (
    build_create_table,
    physical_columns,
    physical_indexes,
)


class _FakeSession:
    def __init__(self) -> None:
        self.sql: list[str] = []
        self.rows: list[dict[str, Any]] = []
        self.row_batches: list[list[dict[str, Any]]] = []

    def execute(self, sql: str, *, table: str | None = None) -> None:
        assert table == "unit_episode"
        self.sql.append(sql)

    def fetch_all(
        self,
        sql: str,
        columns: Sequence[str],
        *,
        table: str | None = None,
    ) -> list[dict[str, Any]]:
        assert table == "unit_episode"
        self.sql.append(sql)
        if self.row_batches:
            return self.row_batches.pop(0)
        return self.rows

    def fetch_scalar(self, sql: str, *, table: str | None = None) -> int:
        assert table == "unit_episode"
        self.sql.append(sql)
        return 0


def _episode(**overrides: object) -> Episode:
    values: dict[str, object] = {
        "id": "u1_ep1",
        "entry_id": "ep1",
        "owner_id": "u1",
        "owner_type": "user",
        "session_id": "session",
        "timestamp": dt.datetime(2026, 1, 1, tzinfo=dt.UTC),
        "parent_id": "mc1",
        "sender_ids": ["u1"],
        "episode": "red apple memory",
        "episode_tokens": "red apple memory",
        "md_path": "users/u1/episodes/day.md",
        "content_sha256": "a" * 64,
        "vector": [1.0] + [0.0] * 1023,
    }
    values.update(overrides)
    return Episode(**values)  # type: ignore[arg-type]


@pytest.fixture(autouse=True)
def _fake_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> _FakeSession:
    monkeypatch.setenv("EVEROS_SEEKDB__MODE", "remote")
    monkeypatch.setenv("EVEROS_SEEKDB__HOST", "db.example")
    monkeypatch.setenv("EVEROS_SEEKDB__TABLE_PREFIX", "unit")
    load_settings.cache_clear()
    SeekdbRepoBase._reset_locks_for_tests()
    fake = _FakeSession()

    async def get_fake_session() -> _FakeSession:
        return fake

    async def run_inline(fn: Any, /, *args: Any, **kwargs: Any) -> Any:
        return fn(*args, **kwargs)

    monkeypatch.setattr(repository_module, "get_session", get_fake_session)
    monkeypatch.setattr(repository_module, "run", run_inline)
    SeekdbRepoBase._ready_tables.add(episode_repo.physical_table_name)
    yield fake
    SeekdbRepoBase._reset_locks_for_tests()
    load_settings.cache_clear()


async def test_upsert_uses_on_duplicate_key_update(_fake_backend: _FakeSession) -> None:
    await episode_repo.upsert([_episode()])
    sql = _fake_backend.sql[-1]
    assert sql.startswith("INSERT INTO `unit_episode`")
    assert "ON DUPLICATE KEY UPDATE" in sql
    assert "`id`=VALUES(`id`)" not in sql
    assert "`vector`=VALUES(`vector`)" in sql


async def test_add_is_insert_only_and_batches_at_64(
    _fake_backend: _FakeSession,
) -> None:
    await episode_repo.add([_episode(id=f"u1_ep_{number}") for number in range(65)])
    inserts = [sql for sql in _fake_backend.sql if sql.startswith("INSERT INTO")]
    assert len(inserts) == 2
    assert all("ON DUPLICATE KEY UPDATE" not in sql for sql in inserts)


async def test_dense_search_is_filtered_clamped_and_null_safe(
    _fake_backend: _FakeSession,
) -> None:
    await episode_repo.dense_search(
        [1.0] + [0.0] * 1023,
        eq("owner_id", "u1"),
        limit=100_000,
        vector_field="subject_vector",
    )
    sql = _fake_backend.sql[-1]
    assert "cosine_distance(`subject_vector`" in sql
    assert "`subject_vector` IS NOT NULL" in sql
    assert "`owner_id` = 'u1'" in sql
    assert "APPROXIMATE LIMIT 16384" in sql


async def test_sparse_search_queries_each_column_and_keeps_best_score(
    _fake_backend: _FakeSession,
) -> None:
    _fake_backend.rows = [{"id": "u1_ep1", "_score": 2.5}]
    rows = await episode_repo.sparse_search(
        ["red", "apple"], None, columns=["episode_tokens"], limit=10
    )
    assert "MATCH(`episode_tokens`) AGAINST" in _fake_backend.sql[-1]
    assert rows == [{"id": "u1_ep1", "_score": 2.5}]


async def test_update_and_native_pagination_render_one_statement_each(
    _fake_backend: _FakeSession,
) -> None:
    await episode_repo.update({"subject": "updated"}, where=eq("id", "u1_ep1"))
    assert _fake_backend.sql[-1] == (
        "UPDATE `unit_episode` SET `subject` = 'updated' WHERE `id` = 'u1_ep1'"
    )

    _fake_backend.rows = []
    await episode_repo.find_where_paginated(
        eq("owner_id", "u1"), sort_by="timestamp", page=2, page_size=20
    )
    assert (
        "ORDER BY `timestamp_ms` DESC, `id` ASC LIMIT 20 OFFSET 20"
        in (_fake_backend.sql[-1])
    )


async def test_update_none_and_mutation_guards_are_explicit(
    _fake_backend: _FakeSession,
) -> None:
    await episode_repo.update({"subject": None}, where=eq("id", "u1_ep1"))
    assert "SET `subject` = NULL" in _fake_backend.sql[-1]
    with pytest.raises(TypeError, match="update requires"):
        await episode_repo.update({"subject": "x"}, where=None)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="delete requires"):
        await episode_repo.delete(None)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "sort_by", ["unknown", "vector", "sender_ids", "episode_tokens"]
)
async def test_pagination_rejects_unknown_and_non_scalar_sort_fields(
    _fake_backend: _FakeSession,
    sort_by: str,
) -> None:
    with pytest.raises(ValueError):
        await episode_repo.find_where_paginated(eq("owner_id", "u1"), sort_by=sort_by)


async def test_scan_uses_keyset_batches_beyond_one_thousand(
    _fake_backend: _FakeSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batches = [
        [{"id": f"id-{number:04d}"} for number in range(1000)],
        [{"id": "id-1000"}],
    ]

    def fetch_all(
        sql: str,
        columns: Sequence[str],
        *,
        table: str | None = None,
    ) -> list[dict[str, Any]]:
        assert table == "unit_episode"
        _fake_backend.sql.append(sql)
        return batches.pop(0)

    monkeypatch.setattr(_fake_backend, "fetch_all", fetch_all)
    monkeypatch.setattr(
        episode_repo,
        "_model",
        lambda row: SimpleNamespace(model_dump=lambda mode: {"id": row["id"]}),
    )
    rows = await episode_repo.scan()
    assert len(rows) == 1001
    assert "`id` > 'id-0999'" in _fake_backend.sql[-1]
    assert batches == []


async def test_verify_table_accepts_catalog_spelling_and_rejects_collation_drift(
    _fake_backend: _FakeSession,
) -> None:
    logical = schema_for(Episode)
    columns = []
    for column in physical_columns(logical):
        columns.append(
            {
                "COLUMN_NAME": column.name,
                "COLUMN_TYPE": (
                    "bigint(20)" if column.sql_type == "BIGINT" else column.sql_type
                ),
                "IS_NULLABLE": (
                    "YES" if column.nullable and not column.primary else "NO"
                ),
                "COLUMN_KEY": "PRI" if column.primary else "",
                "CHARACTER_SET_NAME": column.character_set,
                "COLLATION_NAME": column.collation,
            }
        )
    indexes = []
    for index in physical_indexes(logical):
        prefixes = index.prefix_lengths or (None,) * len(index.columns)
        for position, (name, prefix) in enumerate(
            zip(index.columns, prefixes, strict=True), start=1
        ):
            indexes.append(
                {
                    "INDEX_NAME": index.name,
                    "COLUMN_NAME": name,
                    "INDEX_TYPE": {
                        "btree": "BTREE",
                        "fulltext": "FULLTEXT",
                        "vector": "VECTOR",
                    }[index.kind],
                    "SEQ_IN_INDEX": position,
                    "SUB_PART": prefix,
                }
            )
    ddl = build_create_table("unit_episode", logical)
    _fake_backend.row_batches = [
        columns,
        indexes,
        [{"Table": "unit_episode", "Create Table": ddl}],
    ]
    await episode_repo.verify_table()

    next(row for row in columns if row["COLUMN_NAME"] == "id")["COLLATION_NAME"] = (
        "utf8mb4_general_ci"
    )
    _fake_backend.row_batches = [
        columns,
        indexes,
        [{"Table": "unit_episode", "Create Table": ddl}],
    ]
    with pytest.raises(SeekdbSchemaMismatchError, match="collation"):
        await episode_repo.verify_table()

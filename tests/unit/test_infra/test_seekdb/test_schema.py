"""Verify SeekDB physical schemas and drift comparison for all seven tables."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from everos.infra.persistence.index import ALL_REPOS, Episode, UserProfile
from everos.infra.persistence.index.schema import schema_for
from everos.infra.persistence.seekdb.schema import (
    PhysicalColumn,
    PhysicalIndex,
    build_create_table,
    column_drift,
    index_drift,
    normalize_sql_type,
    physical_columns,
    physical_indexes,
)

_DDL_GOLDEN_SHA256 = {
    "episode": "b4300451b98fc7d61d65f86d92c11f74ebebb2bb5f0c57ce49e5e9291554433f",
    "atomic_fact": "3cabf5316b1e994c02172ace1f2f251debc3ed634942438cecb3db48b4f6309c",
    "foresight": "e6c16cf6566a6d285f4d1db00e02b55037dc900228d9fe30deaf58cbcf0b7e6e",
    "agent_case": "0f92b68f53c09a84ac815411c3577b94380301dd56ccaae3c63dc8b52a5ecc78",
    "agent_skill": "c1a79ea2a7a2e68dc36a4e9ec2c20da88660854c1234eddac6595dc35b3f2b0f",
    "user_profile": "364d7e68a065d52e7c0f60cc6f0fa8daf9860df6254a8bebf525b9ee41813bb1",
    "knowledge_topic": (
        "ee501b9d525c5d357a49b7944ec3b6113f2fd670e5a67f9b48b78ee3f3cae4c8"
    ),
}


def test_every_logical_field_has_one_seekdb_column() -> None:
    for repo in ALL_REPOS:
        logical = schema_for(repo.schema)
        physical = physical_columns(logical)
        assert len(physical) == len(logical.fields)
        assert sum(column.primary for column in physical) == 1
        assert {index.name for index in physical_indexes(logical)} >= {
            f"ft_{name}" for name in logical.bm25_fields
        }


def test_all_seven_create_table_statements_match_reviewed_golden() -> None:
    actual = {}
    for repo in ALL_REPOS:
        logical = schema_for(repo.schema)
        ddl = build_create_table(f"unit_{logical.table_name}", logical)
        actual[logical.table_name] = hashlib.sha256(ddl.encode()).hexdigest()
    assert actual == _DDL_GOLDEN_SHA256


def test_episode_ddl_has_bounded_keys_collation_and_search_indexes() -> None:
    ddl = build_create_table("unit_episode", schema_for(Episode))
    assert "`timestamp_ms` BIGINT NOT NULL" in ddl
    assert "`sender_ids` JSON NOT NULL" in ddl
    assert "INDEX `ix_owner_scope` (`owner_id`(128), `app_id`(128)," in ddl
    assert "FULLTEXT INDEX `ft_episode_tokens`" in ddl
    assert ddl.count("VECTOR INDEX") == 2
    assert ddl.count("SYNC_MODE=immediate") == 2
    assert "DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_bin" in ddl
    assert ddl.endswith("ORGANIZATION = HEAP;")


def test_vector_sync_mode_is_part_of_ddl_and_drift_contract() -> None:
    logical = schema_for(Episode)
    ddl = build_create_table("unit_episode", logical, "async")
    assert ddl.count("SYNC_MODE=async") == 2
    missing, stale, incompatible = index_drift(
        physical_indexes(logical, "immediate"), (), create_sql=ddl
    )
    assert missing == []
    assert stale == []
    assert any("sync_mode 'async' != 'immediate'" in item for item in incompatible)


def test_vectorless_profile_does_not_get_a_dummy_vector() -> None:
    ddl = build_create_table("unit_profile", schema_for(UserProfile))
    assert "VECTOR(" not in ddl
    assert "VECTOR INDEX" not in ddl


def test_only_character_columns_carry_collation_expectations() -> None:
    columns = {column.name: column for column in physical_columns(schema_for(Episode))}
    assert columns["id"].collation == "utf8mb4_bin"
    assert columns["episode_tokens"].collation == "utf8mb4_bin"
    assert columns["sender_ids"].collation is None
    assert columns["timestamp_ms"].collation is None


def test_integer_display_width_is_not_schema_drift() -> None:
    assert normalize_sql_type(" BIGINT ( 20 ) ") == "bigint"
    expected = physical_columns(schema_for(UserProfile))
    reported = [_catalog_column(column, integer_width=True) for column in expected]
    assert column_drift(expected, reported) == ([], [], [])

    for row in reported:
        for key in (
            "COLUMN_NAME",
            "COLUMN_TYPE",
            "IS_NULLABLE",
            "COLUMN_KEY",
            "CHARACTER_SET_NAME",
            "COLLATION_NAME",
        ):
            if isinstance(row[key], str):
                row[key] = row[key].encode()
    assert column_drift(expected, reported) == ([], [], [])


def test_column_drift_reports_missing_stale_type_and_collation() -> None:
    expected = physical_columns(schema_for(UserProfile))
    reported = [
        _catalog_column(column) for column in expected if column.name != "summary"
    ]
    id_row = next(row for row in reported if row["COLUMN_NAME"] == "id")
    id_row["COLUMN_TYPE"] = "BIGINT"
    id_row["COLLATION_NAME"] = "utf8mb4_general_ci"
    reported.append(
        {
            "COLUMN_NAME": "stale",
            "COLUMN_TYPE": "BIGINT",
            "IS_NULLABLE": "YES",
            "COLUMN_KEY": "",
            "CHARACTER_SET_NAME": None,
            "COLLATION_NAME": None,
        }
    )
    missing, stale, incompatible = column_drift(expected, reported)
    assert missing == ["summary"]
    assert stale == ["stale"]
    assert any("id: type" in item for item in incompatible)
    assert any("id: collation" in item for item in incompatible)


def test_index_drift_checks_stats_ddl_parameters_and_stale_indexes() -> None:
    logical = schema_for(Episode)
    expected = physical_indexes(logical)
    reported = _catalog_indexes(expected)
    ddl = build_create_table("unit_episode", logical)
    assert index_drift(expected, reported, create_sql=ddl) == ([], [], [])

    wrong = ddl.replace("DISTANCE=cosine", "DISTANCE=l2", 1)
    missing, stale, incompatible = index_drift(expected, reported, create_sql=wrong)
    assert missing == []
    assert stale == []
    assert any("distance 'l2' != 'cosine'" in item for item in incompatible)

    wrong = ddl.replace(
        "FULLTEXT INDEX `ft_episode_tokens`",
        "INDEX `ft_episode_tokens`",
    )
    _, _, incompatible = index_drift(expected, reported, create_sql=wrong)
    assert any("DDL type 'btree' != 'fulltext'" in item for item in incompatible)

    with_stale = [
        *reported,
        {
            "INDEX_NAME": "ix_obsolete",
            "COLUMN_NAME": "id",
            "INDEX_TYPE": "BTREE",
            "SEQ_IN_INDEX": 1,
            "SUB_PART": None,
        },
    ]
    _, stale, _ = index_drift(expected, with_stale, create_sql=ddl)
    assert stale == ["ix_obsolete"]


def test_search_catalog_internal_columns_are_not_logical_schema_drift() -> None:
    logical = schema_for(Episode)
    expected = physical_indexes(logical)
    ddl = build_create_table("unit_episode", logical)
    reported = _catalog_indexes(expected)
    for row in reported:
        name = str(row["INDEX_NAME"])
        if name.startswith(("ft_", "vec_")):
            row["COLUMN_NAME"] = "__pk_increment"
            row["INDEX_TYPE"] = "BTREE"
    for index in expected:
        if index.kind == "vector":
            for suffix in ("_index_id_table", "_index_snapshot_data_table"):
                reported.append(
                    {
                        "INDEX_NAME": index.name + suffix,
                        "COLUMN_NAME": "__vid",
                        "INDEX_TYPE": "BTREE",
                        "SEQ_IN_INDEX": 1,
                        "SUB_PART": None,
                    }
                )
    assert index_drift(expected, reported, create_sql=ddl) == ([], [], [])
    # Still reject real logical option drift and undeclared search indexes.
    wrong = ddl.replace("WITH PARSER space", "WITH PARSER ngram")
    assert any(
        "parser" in item
        for item in index_drift(expected, reported, create_sql=wrong)[2]
    )
    missing_ddl = ddl.replace("VECTOR INDEX `vec_vector`", "INDEX `vec_vector`")
    _, stale, incompatible = index_drift(expected, reported, create_sql=missing_ddl)
    assert "vec_vector_index_id_table" in stale
    assert any("vec_vector: DDL type" in item for item in incompatible)
    # A user-created index with an auxiliary-looking name is not hidden.
    extra = ddl.replace(
        "PRIMARY KEY (`id`)",
        "INDEX `vec_vector_index_id_table` (`id`), PRIMARY KEY (`id`)",
    )
    assert (
        "vec_vector_index_id_table"
        in index_drift(expected, reported, create_sql=extra)[1]
    )


def _catalog_column(
    column: PhysicalColumn, *, integer_width: bool = False
) -> dict[str, object]:
    sql_type = column.sql_type
    if integer_width and sql_type == "BIGINT":
        sql_type = "bigint(20)"
    return {
        "COLUMN_NAME": column.name,
        "COLUMN_TYPE": sql_type,
        "IS_NULLABLE": ("YES" if column.nullable and not column.primary else "NO"),
        "COLUMN_KEY": "PRI" if column.primary else "",
        "CHARACTER_SET_NAME": column.character_set,
        "COLLATION_NAME": column.collation,
    }


def _catalog_indexes(indexes: Sequence[PhysicalIndex]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in indexes:
        prefixes = index.prefix_lengths or (None,) * len(index.columns)
        index_type = {
            "btree": "BTREE",
            "fulltext": "FULLTEXT",
            "vector": "VECTOR",
        }[index.kind]
        rows.extend(
            {
                "INDEX_NAME": index.name,
                "COLUMN_NAME": column,
                "INDEX_TYPE": index_type,
                "SEQ_IN_INDEX": position,
                "SUB_PART": prefix,
            }
            for position, (column, prefix) in enumerate(
                zip(index.columns, prefixes, strict=True), start=1
            )
        )
    return rows

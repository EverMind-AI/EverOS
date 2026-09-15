"""Map portable index schemas to SeekDB columns, indexes, and DDL.

The same immutable descriptors drive creation and drift verification. This
keeps catalog spelling differences separate from real semantic differences.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from typing import Any, Literal

from everos.infra.persistence.index.schema import (
    IndexField,
    IndexFieldKind,
    IndexSchema,
)

from .sql import quote_identifier

_SHORT_STRING_FIELDS = frozenset(
    {"md_path", "content_sha256", "deprecated_by", "cluster_id", "entry_id"}
)
_INTEGER_DISPLAY_WIDTH = re.compile(
    r"^(tinyint|smallint|mediumint|int|integer|bigint)\(\d+\)"
)
_INDEX_HEADER = re.compile(
    r"^\s*(?:(?P<kind>VECTOR|FULLTEXT)\s+)?(?:INDEX|KEY)\s+"
    r"`?(?P<name>[A-Za-z_][A-Za-z0-9_]*)`?\s*\(",
    re.IGNORECASE,
)
_INDEX_COLUMN = re.compile(
    r"^\s*`?(?P<name>[A-Za-z_][A-Za-z0-9_]*)`?"
    r"(?:\s*\(\s*(?P<prefix>\d+)\s*\))?\s*$"
)
_VECTOR_OPTION = re.compile(
    r"\b(DISTANCE|TYPE|LIB|SYNC_MODE)\s*=\s*([A-Za-z0-9_]+)",
    re.IGNORECASE,
)
_TEXT_SQL_TYPES = ("VARCHAR(", "LONGTEXT")
_INDEX_PREFIX_LENGTH = 128


@dataclass(frozen=True)
class PhysicalColumn:
    """One physical column as declared in SeekDB."""

    name: str
    sql_type: str
    nullable: bool
    primary: bool = False
    dimension: int | None = None
    character_set: str | None = None
    collation: str | None = None

    def ddl(self) -> str:
        nullability = "NULL" if self.nullable and not self.primary else "NOT NULL"
        return f"{quote_identifier(self.name)} {self.sql_type} {nullability}"

    def mismatches(self, reported: Mapping[str, Any]) -> list[str]:
        actual_type = normalize_sql_type(_as_text(_get(reported, "COLUMN_TYPE", "")))
        expected_type = normalize_sql_type(self.sql_type)
        actual_nullable = _as_text(_get(reported, "IS_NULLABLE", "NO")).upper() == "YES"
        actual_primary = _as_text(_get(reported, "COLUMN_KEY", "")).upper() == "PRI"
        out: list[str] = []
        if actual_type != expected_type:
            out.append(f"{self.name}: type {actual_type!r} != {expected_type!r}")
        if actual_nullable != (self.nullable and not self.primary):
            out.append(
                f"{self.name}: nullable {actual_nullable} != "
                f"{self.nullable and not self.primary}"
            )
        if actual_primary != self.primary:
            out.append(f"{self.name}: primary {actual_primary} != {self.primary}")
        _append_text_metadata_mismatches(out, self, reported)
        return out


@dataclass(frozen=True)
class PhysicalIndex:
    """One named secondary, full-text, or vector index."""

    name: str
    columns: tuple[str, ...]
    kind: Literal["btree", "fulltext", "vector"]
    prefix_lengths: tuple[int | None, ...] = ()
    parser: str | None = None
    distance: str | None = None
    vector_type: str | None = None
    library: str | None = None
    sync_mode: Literal["immediate", "async"] | None = None

    def ddl(self) -> str:
        prefixes = self.prefix_lengths or (None,) * len(self.columns)
        columns = ", ".join(
            quote_identifier(column) + (f"({prefix})" if prefix else "")
            for column, prefix in zip(self.columns, prefixes, strict=True)
        )
        name = quote_identifier(self.name)
        if self.kind == "fulltext":
            return f"FULLTEXT INDEX {name} ({columns}) WITH PARSER {self.parser}"
        if self.kind == "vector":
            return (
                f"VECTOR INDEX {name} ({columns}) WITH "
                f"(DISTANCE={self.distance}, TYPE={self.vector_type}, "
                f"LIB={self.library}, SYNC_MODE={self.sync_mode})"
            )
        return f"INDEX {name} ({columns})"


@dataclass(frozen=True)
class _ParsedIndex:
    name: str
    kind: Literal["btree", "fulltext", "vector"]
    columns: tuple[str, ...]
    prefix_lengths: tuple[int | None, ...]
    parser: str | None = None
    options: tuple[tuple[str, str], ...] = ()


def normalize_sql_type(value: str) -> str:
    """Collapse non-semantic MySQL/OceanBase catalog spelling differences."""
    lowered = value.strip().lower().replace(" ", "")
    return _INTEGER_DISPLAY_WIDTH.sub(r"\1", lowered)


@cache
def physical_columns(schema: IndexSchema) -> tuple[PhysicalColumn, ...]:
    """Return the complete physical column set in deterministic order."""
    return tuple(_physical_column(field, schema) for field in schema.fields)


def physical_column(schema: IndexSchema, name: str) -> PhysicalColumn:
    """Return the physical descriptor for one logical field."""
    field = schema.field(name)
    return next(
        column
        for column in physical_columns(schema)
        if column.name == _storage_name(field)
    )


@cache
def physical_indexes(
    schema: IndexSchema,
    vector_sync_mode: Literal["immediate", "async"] = "immediate",
) -> tuple[PhysicalIndex, ...]:
    """Return indexes useful to EverOS reads plus every search index."""
    names = {field.name for field in schema.fields}
    columns = {column.name: column for column in physical_columns(schema)}
    out: list[PhysicalIndex] = []
    _append_index(
        out,
        names,
        columns,
        "ix_owner_scope",
        "owner_id",
        "app_id",
        "project_id",
    )
    _append_index(out, names, columns, "ix_md_path", "md_path")
    _append_index(out, names, columns, "ix_owner_entry", "owner_id", "entry_id")
    _append_index(out, names, columns, "ix_owner_cluster", "owner_id", "cluster_id")
    _append_index(out, names, columns, "ix_parent", "parent_type", "parent_id")
    out.extend(
        PhysicalIndex(f"ft_{name}", (name,), "fulltext", parser="space")
        for name in schema.bm25_fields
    )
    out.extend(
        PhysicalIndex(
            f"vec_{field.name}",
            (field.name,),
            "vector",
            distance="cosine",
            vector_type="hnsw",
            library="vsag",
            sync_mode=vector_sync_mode,
        )
        for field in schema.vector_fields
    )
    return tuple(out)


def build_create_table(
    table: str,
    schema: IndexSchema,
    vector_sync_mode: Literal["immediate", "async"] = "immediate",
) -> str:
    """Build the complete idempotent CREATE TABLE statement."""
    columns = physical_columns(schema)
    definitions = [column.ddl() for column in columns]
    primary = next(column for column in columns if column.primary)
    definitions.append(f"PRIMARY KEY ({quote_identifier(primary.name)})")
    definitions.extend(
        index.ddl() for index in physical_indexes(schema, vector_sync_mode)
    )
    body = ",\n    ".join(definitions)
    return (
        f"CREATE TABLE IF NOT EXISTS {quote_identifier(table)} (\n"
        f"    {body}\n"
        ") DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_bin "
        "ORGANIZATION = HEAP;"
    )


def column_drift(
    expected: Sequence[PhysicalColumn], reported: Sequence[Mapping[str, Any]]
) -> tuple[list[str], list[str], list[str]]:
    """Return missing, stale, and incompatible column details."""
    wanted = {column.name: column for column in expected}
    actual = {_as_text(_get(row, "COLUMN_NAME", "")): row for row in reported}
    missing = sorted(set(wanted) - set(actual))
    stale = sorted(set(actual) - set(wanted))
    incompatible: list[str] = []
    for name, column in wanted.items():
        if name in actual:
            incompatible.extend(column.mismatches(actual[name]))
    return missing, stale, incompatible


def index_drift(
    expected: Sequence[PhysicalIndex],
    reported: Sequence[Mapping[str, Any]],
    *,
    create_sql: str = "",
) -> tuple[list[str], list[str], list[str]]:
    """Return missing, stale, and incompatible index details."""
    by_name: dict[str, list[Mapping[str, Any]]] = {}
    for row in reported:
        name = _as_text(_get(row, "INDEX_NAME", ""))
        if name.upper() != "PRIMARY":
            by_name.setdefault(name, []).append(row)
    parsed = _parse_indexes(create_sql)
    wanted = {index.name: index for index in expected}
    actual_names = set(by_name) | set(parsed)
    missing: list[str] = []
    incompatible: list[str] = []
    for name, index in wanted.items():
        rows = by_name.get(name, [])
        declaration = parsed.get(name)
        if not rows and declaration is None:
            missing.append(name)
            continue
        if rows:
            incompatible.extend(_statistics_mismatches(index, rows))
        if declaration is not None:
            incompatible.extend(_ddl_index_mismatches(index, declaration))
        elif index.kind in {"fulltext", "vector"}:
            incompatible.append(f"{name}: SHOW CREATE TABLE declaration is missing")
    stale = sorted(actual_names - set(wanted))
    return sorted(missing), stale, incompatible


def _physical_column(field: IndexField, schema: IndexSchema) -> PhysicalColumn:
    if field.kind is IndexFieldKind.STRING:
        short = field.primary or field.name.endswith(("_id", "_type"))
        sql_type = (
            "LONGTEXT"
            if field.name in schema.bm25_fields
            or not (short or field.name in _SHORT_STRING_FIELDS)
            else "VARCHAR(512)"
        )
    elif field.kind is IndexFieldKind.STRING_ARRAY:
        sql_type = "JSON"
    elif field.kind is IndexFieldKind.FLOAT:
        sql_type = "DOUBLE"
    elif field.kind is IndexFieldKind.INTEGER:
        sql_type = "BIGINT"
    elif field.kind is IndexFieldKind.DATETIME:
        return PhysicalColumn(f"{field.name}_ms", "BIGINT", field.nullable)
    elif field.kind is IndexFieldKind.DENSE_VECTOR:
        sql_type = f"VECTOR({field.dimension})"
    else:  # pragma: no cover - enum exhaustiveness guard
        raise TypeError(f"unsupported index field kind: {field.kind}")
    is_text = sql_type.startswith(_TEXT_SQL_TYPES)
    return PhysicalColumn(
        field.name,
        sql_type,
        field.nullable,
        primary=field.primary,
        dimension=field.dimension,
        character_set="utf8mb4" if is_text else None,
        collation="utf8mb4_bin" if is_text else None,
    )


def _storage_name(field: IndexField) -> str:
    return f"{field.name}_ms" if field.kind is IndexFieldKind.DATETIME else field.name


def _append_index(
    indexes: list[PhysicalIndex],
    available: set[str],
    physical: Mapping[str, PhysicalColumn],
    name: str,
    *columns: str,
) -> None:
    if not set(columns) <= available:
        return
    use_prefixes = len(columns) > 1
    prefixes = tuple(
        _INDEX_PREFIX_LENGTH
        if use_prefixes and physical[column].sql_type.startswith("VARCHAR(")
        else None
        for column in columns
    )
    indexes.append(PhysicalIndex(name, tuple(columns), "btree", prefixes))


def _append_text_metadata_mismatches(
    out: list[str], column: PhysicalColumn, reported: Mapping[str, Any]
) -> None:
    checks = (
        ("CHARACTER_SET_NAME", column.character_set, "character set"),
        ("COLLATION_NAME", column.collation, "collation"),
    )
    for key, expected, label in checks:
        if expected is None or not _has_key(reported, key):
            continue
        actual = _get(reported, key, None)
        actual_text = None if actual is None else _as_text(actual).casefold()
        if actual_text != expected.casefold():
            out.append(f"{column.name}: {label} {actual_text!r} != {expected!r}")


def _statistics_mismatches(
    expected: PhysicalIndex, rows: Sequence[Mapping[str, Any]]
) -> list[str]:
    ordered = sorted(rows, key=_sequence_in_index)
    columns = tuple(_as_text(_get(row, "COLUMN_NAME", "")) for row in ordered)
    prefixes = tuple(_optional_int(_get(row, "SUB_PART", None)) for row in ordered)
    expected_prefixes = expected.prefix_lengths or (None,) * len(expected.columns)
    out: list[str] = []
    if columns != expected.columns:
        out.append(f"{expected.name}: columns {columns!r} != {expected.columns!r}")
    if prefixes != expected_prefixes:
        out.append(f"{expected.name}: prefixes {prefixes!r} != {expected_prefixes!r}")
    actual_types = {_as_text(_get(row, "INDEX_TYPE", "")).upper() for row in ordered}
    if not _statistics_kind_matches(expected.kind, actual_types):
        out.append(
            f"{expected.name}: type {sorted(actual_types)!r} != {expected.kind!r}"
        )
    return out


def _ddl_index_mismatches(expected: PhysicalIndex, actual: _ParsedIndex) -> list[str]:
    out: list[str] = []
    if actual.kind != expected.kind:
        out.append(f"{expected.name}: DDL type {actual.kind!r} != {expected.kind!r}")
    if actual.columns != expected.columns:
        out.append(
            f"{expected.name}: DDL columns {actual.columns!r} != {expected.columns!r}"
        )
    expected_prefixes = expected.prefix_lengths or (None,) * len(expected.columns)
    if actual.prefix_lengths != expected_prefixes:
        out.append(
            f"{expected.name}: DDL prefixes {actual.prefix_lengths!r} "
            f"!= {expected_prefixes!r}"
        )
    if expected.kind == "fulltext" and actual.parser != expected.parser:
        out.append(f"{expected.name}: parser {actual.parser!r} != {expected.parser!r}")
    if expected.kind == "vector":
        options = dict(actual.options)
        expected_options = {
            "distance": expected.distance,
            "type": expected.vector_type,
            "lib": expected.library,
            "sync_mode": expected.sync_mode,
        }
        for key, wanted in expected_options.items():
            if options.get(key) != wanted:
                out.append(f"{expected.name}: {key} {options.get(key)!r} != {wanted!r}")
    return out


def _parse_indexes(create_sql: str) -> dict[str, _ParsedIndex]:
    parsed: dict[str, _ParsedIndex] = {}
    for definition in _table_definitions(create_sql):
        match = _INDEX_HEADER.match(definition)
        if match is None:
            continue
        opening = match.end() - 1
        closing = _matching_parenthesis(definition, opening)
        if closing is None:
            continue
        columns, prefixes = _parse_index_columns(definition[opening + 1 : closing])
        raw_kind = (match.group("kind") or "").casefold()
        kind: Literal["btree", "fulltext", "vector"] = (
            "fulltext"
            if raw_kind == "fulltext"
            else "vector"
            if raw_kind == "vector"
            else "btree"
        )
        tail = definition[closing + 1 :]
        parser_match = re.search(
            r"\bWITH\s+PARSER\s+([A-Za-z0-9_]+)", tail, re.IGNORECASE
        )
        options = tuple(
            (key.casefold(), value.casefold())
            for key, value in _VECTOR_OPTION.findall(tail)
        )
        name = match.group("name")
        parsed[name] = _ParsedIndex(
            name,
            kind,
            columns,
            prefixes,
            parser_match.group(1).casefold() if parser_match else None,
            options,
        )
    return parsed


def _table_definitions(create_sql: str) -> tuple[str, ...]:
    if not create_sql:
        return ()
    opening = create_sql.find("(")
    closing = create_sql.rfind(")")
    if opening < 0 or closing <= opening:
        return ()
    body = create_sql[opening + 1 : closing]
    out: list[str] = []
    start = 0
    depth = 0
    quote: str | None = None
    for position, char in enumerate(body):
        if char in {"'", '"', "`"}:
            quote = None if quote == char else char if quote is None else quote
        elif quote is None and char == "(":
            depth += 1
        elif quote is None and char == ")":
            depth -= 1
        elif quote is None and char == "," and depth == 0:
            out.append(body[start:position].strip())
            start = position + 1
    out.append(body[start:].strip())
    return tuple(item for item in out if item)


def _parse_index_columns(value: str) -> tuple[tuple[str, ...], tuple[int | None, ...]]:
    columns: list[str] = []
    prefixes: list[int | None] = []
    for item in value.split(","):
        match = _INDEX_COLUMN.match(item)
        if match is None:
            return (), ()
        columns.append(match.group("name"))
        prefixes.append(_optional_int(match.group("prefix")))
    return tuple(columns), tuple(prefixes)


def _matching_parenthesis(value: str, opening: int) -> int | None:
    depth = 0
    for position in range(opening, len(value)):
        if value[position] == "(":
            depth += 1
        elif value[position] == ")":
            depth -= 1
            if depth == 0:
                return position
    return None


def _statistics_kind_matches(
    kind: Literal["btree", "fulltext", "vector"], actual_types: set[str]
) -> bool:
    if kind == "fulltext":
        return actual_types == {"FULLTEXT"}
    if kind == "vector":
        return any("VECTOR" in value or "HNSW" in value for value in actual_types)
    return not actual_types or actual_types == {"BTREE"}


def _get(row: Mapping[str, Any], name: str, default: Any) -> Any:
    if name in row:
        return row[name]
    folded = name.casefold()
    return next(
        (value for key, value in row.items() if _as_text(key).casefold() == folded),
        default,
    )


def _has_key(row: Mapping[str, Any], name: str) -> bool:
    folded = name.casefold()
    return any(_as_text(key).casefold() == folded for key in row)


def _as_text(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _sequence_in_index(row: Mapping[str, Any]) -> int:
    return int(_get(row, "SEQ_IN_INDEX", 0) or 0)


def _optional_int(value: Any) -> int | None:
    return None if value in {None, ""} else int(value)


__all__ = [
    "PhysicalColumn",
    "PhysicalIndex",
    "build_create_table",
    "column_drift",
    "index_drift",
    "normalize_sql_type",
    "physical_column",
    "physical_columns",
    "physical_indexes",
]

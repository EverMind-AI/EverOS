"""Backend-neutral schemas for rebuildable business indexes.

Each table has one explicit field description consumed by every backend
adapter. The model-field parity check fails immediately when a domain field is
added without a storage decision, avoiding fallback coercions and silently
divergent schemas.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from enum import StrEnum
from functools import cache
from typing import Any, get_args, get_origin

_DEFAULT_STRING_LENGTH = 65_535
_ID_LENGTH = 512
_ARRAY_CAPACITY = 256
_VECTOR_DIMENSION = 1024


class IndexFieldKind(StrEnum):
    STRING = "string"
    STRING_ARRAY = "string_array"
    FLOAT = "float"
    INTEGER = "integer"
    DATETIME = "datetime"
    DENSE_VECTOR = "dense_vector"


@dataclass(frozen=True)
class IndexField:
    name: str
    kind: IndexFieldKind
    nullable: bool = False
    primary: bool = False
    max_length: int | None = None
    max_capacity: int | None = None
    dimension: int | None = None


@dataclass(frozen=True)
class IndexSchema:
    table_name: str
    model: type[Any]
    fields: tuple[IndexField, ...]
    bm25_fields: tuple[str, ...]

    def field(self, name: str) -> IndexField:
        for field in self.fields:
            if field.name == name:
                return field
        raise KeyError(name)

    @property
    def vector_fields(self) -> tuple[IndexField, ...]:
        return tuple(
            field for field in self.fields if field.kind is IndexFieldKind.DENSE_VECTOR
        )

    @property
    def datetime_fields(self) -> frozenset[str]:
        return frozenset(
            field.name for field in self.fields if field.kind is IndexFieldKind.DATETIME
        )


def _s(
    name: str,
    *,
    nullable: bool = False,
    primary: bool = False,
    max_length: int = _DEFAULT_STRING_LENGTH,
) -> IndexField:
    return IndexField(
        name,
        IndexFieldKind.STRING,
        nullable=nullable,
        primary=primary,
        max_length=max_length,
    )


def _id(name: str = "id") -> IndexField:
    return _s(name, primary=True, max_length=_ID_LENGTH)


def _a(name: str, *, nullable: bool = False) -> IndexField:
    return IndexField(
        name,
        IndexFieldKind.STRING_ARRAY,
        nullable=nullable,
        max_length=_ID_LENGTH,
        max_capacity=_ARRAY_CAPACITY,
    )


def _f(name: str, *, nullable: bool = False) -> IndexField:
    return IndexField(name, IndexFieldKind.FLOAT, nullable=nullable)


def _i(name: str, *, nullable: bool = False) -> IndexField:
    return IndexField(name, IndexFieldKind.INTEGER, nullable=nullable)


def _d(name: str, *, nullable: bool = False) -> IndexField:
    return IndexField(name, IndexFieldKind.DATETIME, nullable=nullable)


def _v(name: str, *, nullable: bool = True) -> IndexField:
    return IndexField(
        name,
        IndexFieldKind.DENSE_VECTOR,
        nullable=nullable,
        dimension=_VECTOR_DIMENSION,
    )


_FIELDS: dict[str, tuple[IndexField, ...]] = {
    "episode": (
        _id(),
        _s("entry_id"),
        _s("owner_id"),
        _s("owner_type"),
        _s("app_id"),
        _s("project_id"),
        _s("session_id", nullable=True),
        _d("timestamp"),
        _s("parent_type"),
        _s("parent_id"),
        _a("sender_ids"),
        _s("subject", nullable=True),
        _s("summary", nullable=True),
        _s("episode"),
        _s("episode_tokens"),
        _s("md_path"),
        _s("content_sha256"),
        _s("deprecated_by", nullable=True),
        _v("vector"),
        _v("subject_vector"),
        _d("created_at"),
        _d("updated_at"),
    ),
    "atomic_fact": (
        _id(),
        _s("entry_id"),
        _s("owner_id"),
        _s("owner_type"),
        _s("app_id"),
        _s("project_id"),
        _s("session_id", nullable=True),
        _d("timestamp"),
        _s("parent_type"),
        _s("parent_id"),
        _a("sender_ids"),
        _s("fact"),
        _s("fact_tokens"),
        _s("md_path"),
        _s("content_sha256"),
        _s("deprecated_by", nullable=True),
        _v("vector"),
        _d("created_at"),
        _d("updated_at"),
    ),
    "foresight": (
        _id(),
        _s("entry_id"),
        _s("owner_id"),
        _s("owner_type"),
        _s("app_id"),
        _s("project_id"),
        _s("session_id", nullable=True),
        _d("timestamp"),
        _d("start_time", nullable=True),
        _d("end_time", nullable=True),
        _i("duration_days", nullable=True),
        _s("parent_type"),
        _s("parent_id"),
        _a("sender_ids"),
        _s("foresight"),
        _s("foresight_tokens"),
        _s("evidence", nullable=True),
        _s("evidence_tokens", nullable=True),
        _s("md_path"),
        _s("content_sha256"),
        _v("vector"),
        _d("created_at"),
        _d("updated_at"),
    ),
    "agent_case": (
        _id(),
        _s("entry_id"),
        _s("owner_id"),
        _s("owner_type"),
        _s("app_id"),
        _s("project_id"),
        _s("session_id"),
        _d("timestamp"),
        _s("parent_type"),
        _s("parent_id"),
        _f("quality_score"),
        _s("task_intent"),
        _s("task_intent_tokens"),
        _s("approach"),
        _s("approach_tokens"),
        _s("key_insight", nullable=True),
        _s("md_path"),
        _s("content_sha256"),
        _v("vector"),
        _d("created_at"),
        _d("updated_at"),
    ),
    "agent_skill": (
        _id(),
        _s("owner_id"),
        _s("owner_type"),
        _s("app_id"),
        _s("project_id"),
        _s("name"),
        _s("description"),
        _s("description_tokens"),
        _s("content"),
        _s("content_tokens"),
        _f("confidence"),
        _f("maturity_score"),
        _a("source_case_ids"),
        _s("cluster_id", nullable=True),
        _s("md_path"),
        _s("content_sha256"),
        _v("vector"),
        _d("created_at"),
        _d("updated_at"),
    ),
    "user_profile": (
        _id(),
        _s("owner_id"),
        _s("owner_type"),
        _s("app_id"),
        _s("project_id"),
        _s("summary"),
        _s("explicit_info_json"),
        _s("implicit_traits_json"),
        _i("profile_timestamp_ms"),
        _s("md_path"),
        _s("content_sha256"),
        _d("created_at"),
        _d("updated_at"),
    ),
    "knowledge_topic": (
        _id(),
        _s("doc_id"),
        _s("category_id"),
        _s("app_id"),
        _s("project_id"),
        _s("topic_name"),
        _s("topic_path"),
        _i("depth"),
        _s("parent_node_id"),
        _s("summary"),
        _s("summary_tokens"),
        _s("content_tokens"),
        _a("content_labels"),
        _s("md_path"),
        _s("content_sha256"),
        _v("vector"),
        _d("created_at"),
        _d("updated_at"),
    ),
}


@cache
def schema_for(model: type[Any]) -> IndexSchema:
    """Return and validate the explicit neutral schema for a model."""
    table_name = model.TABLE_NAME
    try:
        fields = _FIELDS[table_name]
    except KeyError as exc:
        raise ValueError(f"no derived-index schema for {table_name!r}") from exc

    declared = {field.name for field in fields}
    actual = set(model.model_fields)
    if declared != actual:
        raise ValueError(
            f"derived-index schema drift for {table_name!r}: "
            f"missing={sorted(actual - declared)}, stale={sorted(declared - actual)}"
        )
    for field in fields:
        _validate_model_field(table_name, field, model.model_fields[field.name])
    bm25_fields = tuple(model.BM25_FIELDS)
    unknown_bm25 = set(bm25_fields) - declared
    if unknown_bm25:
        raise ValueError(
            f"derived-index schema {table_name!r} has unknown BM25 fields: "
            f"{sorted(unknown_bm25)}"
        )
    return IndexSchema(table_name, model, fields, bm25_fields)


def _validate_model_field(table_name: str, field: IndexField, model_field: Any) -> None:
    annotation = model_field.annotation
    args = get_args(annotation)
    optional = type(None) in args
    candidates = (
        tuple(arg for arg in args if arg is not type(None))
        if optional
        else (annotation,)
    )
    if optional != field.nullable:
        raise ValueError(
            f"derived-index schema {table_name}.{field.name} nullable drift: "
            f"model={optional}, schema={field.nullable}"
        )

    valid = False
    if field.kind is IndexFieldKind.STRING:
        valid = candidates == (str,)
    elif field.kind is IndexFieldKind.STRING_ARRAY:
        valid = len(candidates) == 1 and (
            get_origin(candidates[0]) is list and get_args(candidates[0]) == (str,)
        )
    elif field.kind is IndexFieldKind.FLOAT:
        valid = candidates == (float,)
    elif field.kind is IndexFieldKind.INTEGER:
        valid = candidates == (int,)
    elif field.kind is IndexFieldKind.DATETIME:
        valid = candidates == (dt.datetime,)
    elif field.kind is IndexFieldKind.DENSE_VECTOR:
        dimension = getattr(candidates[0], "dim", None) if candidates else None
        if callable(dimension):
            dimension = dimension()
        valid = len(candidates) == 1 and dimension == field.dimension
    if not valid:
        raise ValueError(
            f"derived-index schema {table_name}.{field.name} type drift: "
            f"model={annotation!r}, schema={field.kind.value}"
        )


__all__ = ["IndexField", "IndexFieldKind", "IndexSchema", "schema_for"]

"""Logical schema descriptions shared by derived-index adapters.

The record models remain the source of truth for logical fields. This module
normalizes their supported types into a small backend-neutral vocabulary and
fails loudly when a new field has no portable representation. Physical limits
and index options remain adapter responsibilities.
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


@cache
def schema_for(model: type[Any]) -> IndexSchema:
    """Build and validate the portable logical schema for ``model``."""
    table_name = _class_var(model, "TABLE_NAME")
    bm25_fields = tuple(_class_var(model, "BM25_FIELDS"))
    fields = tuple(
        _normalize_field(name, model_field.annotation)
        for name, model_field in model.model_fields.items()
    )
    declared = {field.name for field in fields}
    unknown_bm25 = set(bm25_fields) - declared
    if unknown_bm25:
        raise ValueError(
            f"derived-index schema {table_name!r} has unknown BM25 fields: "
            f"{sorted(unknown_bm25)}"
        )
    return IndexSchema(table_name, model, fields, bm25_fields)


def _normalize_field(name: str, annotation: Any) -> IndexField:
    args = get_args(annotation)
    optional = type(None) in args
    candidates = tuple(arg for arg in args if arg is not type(None)) if optional else ()
    value_type = candidates[0] if len(candidates) == 1 else annotation

    if value_type is str:
        kind = IndexFieldKind.STRING
        max_length = _ID_LENGTH if name == "id" else _DEFAULT_STRING_LENGTH
        max_capacity = None
        dimension = None
    elif value_type is float:
        kind = IndexFieldKind.FLOAT
        max_length = None
        max_capacity = None
        dimension = None
    elif value_type is int:
        kind = IndexFieldKind.INTEGER
        max_length = None
        max_capacity = None
        dimension = None
    elif value_type is dt.datetime:
        kind = IndexFieldKind.DATETIME
        max_length = None
        max_capacity = None
        dimension = None
    elif get_origin(value_type) is list and get_args(value_type) == (str,):
        kind = IndexFieldKind.STRING_ARRAY
        max_length = _ID_LENGTH
        max_capacity = _ARRAY_CAPACITY
        dimension = None
    else:
        dimension = getattr(value_type, "dim", None)
        if callable(dimension):
            dimension = dimension()
        if not isinstance(dimension, int) or dimension <= 0:
            raise ValueError(
                f"derived-index field {name!r} has no portable type mapping: "
                f"{annotation!r}"
            )
        kind = IndexFieldKind.DENSE_VECTOR
        max_length = None
        max_capacity = None

    return IndexField(
        name=name,
        kind=kind,
        nullable=optional,
        primary=name == "id",
        max_length=max_length,
        max_capacity=max_capacity,
        dimension=dimension,
    )


def _class_var(model: type[Any], name: str) -> Any:
    value = getattr(model, name, None)
    if value is None:
        raise ValueError(f"derived-index model {model.__name__} has no {name}")
    return value


__all__ = ["IndexField", "IndexFieldKind", "IndexSchema", "schema_for"]

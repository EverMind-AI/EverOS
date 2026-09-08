"""Render backend-neutral predicates as Milvus filter expressions."""

from __future__ import annotations

import datetime as dt
import json
import re
from collections.abc import Collection
from typing import Final

from everos.component.utils.datetime import ensure_utc, to_timestamp_ms
from everos.infra.persistence.predicate import (
    All,
    AnyOf,
    Comparison,
    Contains,
    In,
    IsNull,
    Predicate,
    Scalar,
)

_FIELD_NAME: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_OPERATORS: Final[dict[str, str]] = {
    "eq": "==",
    "ne": "!=",
    "gt": ">",
    "gte": ">=",
    "lt": "<",
    "lte": "<=",
}


def render_predicate(
    predicate: Predicate | None,
    *,
    datetime_fields: Collection[str] = (),
    vector_fields: Collection[str] = (),
) -> str:
    """Render a predicate using Milvus operators and physical field names."""
    if predicate is None:
        return ""
    if isinstance(predicate, Comparison):
        return (
            f"{_field(predicate.field, datetime_fields)} "
            f"{_OPERATORS[predicate.operator]} {_literal(predicate.value)}"
        )
    if isinstance(predicate, In):
        values = ", ".join(_literal(value) for value in predicate.values)
        return f"{_field(predicate.field, datetime_fields)} in [{values}]"
    if isinstance(predicate, Contains):
        return (
            f"array_contains({_field(predicate.field, datetime_fields)}, "
            f"{_literal(predicate.value)})"
        )
    if isinstance(predicate, IsNull):
        if predicate.field in vector_fields:
            return f"{_field(predicate.field, ())}__present == false"
        return f"{_field(predicate.field, datetime_fields)} is null"
    if isinstance(predicate, All):
        return _render_group(predicate.children, "and", datetime_fields, vector_fields)
    if isinstance(predicate, AnyOf):
        return _render_group(predicate.children, "or", datetime_fields, vector_fields)
    raise TypeError(f"unsupported predicate: {type(predicate).__name__}")


def _render_group(
    children: tuple[Predicate, ...],
    operator: str,
    datetime_fields: Collection[str],
    vector_fields: Collection[str],
) -> str:
    rendered = [
        render_predicate(
            child,
            datetime_fields=datetime_fields,
            vector_fields=vector_fields,
        )
        for child in children
    ]
    rendered = [item for item in rendered if item]
    if not rendered:
        return ""
    if len(rendered) == 1:
        return rendered[0]
    return "(" + f" {operator} ".join(f"({item})" for item in rendered) + ")"


def _field(value: str, datetime_fields: Collection[str]) -> str:
    if not _FIELD_NAME.fullmatch(value):
        raise ValueError(f"invalid predicate field: {value!r}")
    if value in datetime_fields:
        return f"{value}_ms"
    return value


def _literal(value: Scalar) -> str:
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, dt.datetime):
        aware = ensure_utc(value)
        assert aware is not None
        return str(to_timestamp_ms(aware))
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


__all__ = ["render_predicate"]

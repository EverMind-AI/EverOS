"""Render backend-neutral predicates into the SeekDB MySQL dialect."""

from __future__ import annotations

import datetime as dt
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

from .sql import json_literal, literal, quote_identifier

_OPERATORS: Final[dict[str, str]] = {
    "eq": "=",
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
    """Render ``predicate`` without a leading ``WHERE`` keyword."""
    if predicate is None:
        return ""
    if not isinstance(predicate, Predicate):
        raise TypeError(
            "SeekDB predicates must use the neutral Predicate AST, "
            f"got {type(predicate).__name__}"
        )
    if isinstance(predicate, Comparison):
        return _comparison(predicate, datetime_fields)
    if isinstance(predicate, In):
        if not predicate.values:
            raise ValueError("SeekDB IN predicates require at least one value")
        values = ", ".join(
            _field_literal(predicate.field, value, datetime_fields)
            for value in predicate.values
        )
        return f"{_field(predicate.field, datetime_fields)} IN ({values})"
    if isinstance(predicate, Contains):
        column = _field(predicate.field, datetime_fields)
        return f"JSON_CONTAINS({column}, {json_literal(predicate.value)})"
    if isinstance(predicate, IsNull):
        # SeekDB 1.4 permits nullable VECTOR columns; vector_fields remains an
        # explicit argument so a future presence-marker fallback is contained.
        _ = vector_fields
        return f"{_field(predicate.field, datetime_fields)} IS NULL"
    if isinstance(predicate, All):
        return _render_group(predicate.children, "AND", datetime_fields, vector_fields)
    if isinstance(predicate, AnyOf):
        return _render_group(predicate.children, "OR", datetime_fields, vector_fields)
    raise TypeError(f"unsupported predicate: {type(predicate).__name__}")


def _comparison(node: Comparison, datetime_fields: Collection[str]) -> str:
    return (
        f"{_field(node.field, datetime_fields)} {_OPERATORS[node.operator]} "
        f"{_field_literal(node.field, node.value, datetime_fields)}"
    )


def _field(name: str, datetime_fields: Collection[str]) -> str:
    physical = f"{name}_ms" if name in datetime_fields else name
    return quote_identifier(physical)


def _field_literal(name: str, value: Scalar, datetime_fields: Collection[str]) -> str:
    if name in datetime_fields and isinstance(value, dt.datetime):
        aware = ensure_utc(value)
        assert aware is not None
        return str(to_timestamp_ms(aware))
    return literal(value)


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
    if len(rendered) == 1:
        return rendered[0]
    return "(" + f" {operator} ".join(f"({item})" for item in rendered) + ")"


__all__ = ["render_predicate"]

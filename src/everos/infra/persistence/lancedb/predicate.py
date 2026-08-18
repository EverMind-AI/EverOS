"""Render backend-neutral predicates as LanceDB DataFusion expressions."""

from __future__ import annotations

import datetime as dt
import re
from typing import Final

from everos.component.utils.datetime import ensure_utc, to_iso_format
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
    "eq": "=",
    "ne": "!=",
    "gt": ">",
    "gte": ">=",
    "lt": "<",
    "lte": "<=",
}


def render_predicate(predicate: Predicate | None) -> str:
    """Render a predicate with LanceDB escaping and timestamp literals."""
    if predicate is None:
        return ""
    if isinstance(predicate, Comparison):
        return (
            f"{_field(predicate.field)} {_OPERATORS[predicate.operator]} "
            f"{_literal(predicate.value)}"
        )
    if isinstance(predicate, In):
        values = ", ".join(_literal(value) for value in predicate.values)
        return f"{_field(predicate.field)} IN ({values})"
    if isinstance(predicate, Contains):
        return f"array_has({_field(predicate.field)}, {_literal(predicate.value)})"
    if isinstance(predicate, IsNull):
        return f"{_field(predicate.field)} IS NULL"
    if isinstance(predicate, All):
        return _render_group(predicate.children, "AND")
    if isinstance(predicate, AnyOf):
        return _render_group(predicate.children, "OR")
    raise TypeError(f"unsupported predicate: {type(predicate).__name__}")


def _render_group(children: tuple[Predicate, ...], operator: str) -> str:
    rendered = [render_predicate(child) for child in children]
    rendered = [item for item in rendered if item]
    if not rendered:
        return ""
    if len(rendered) == 1:
        return rendered[0]
    return "(" + f" {operator} ".join(f"({item})" for item in rendered) + ")"


def _field(value: str) -> str:
    if not _FIELD_NAME.fullmatch(value):
        raise ValueError(f"invalid predicate field: {value!r}")
    return value


def _literal(value: Scalar) -> str:
    if isinstance(value, str):
        return f"'{value.replace(chr(39), chr(39) * 2)}'"
    if isinstance(value, dt.datetime):
        aware = ensure_utc(value)
        assert aware is not None
        return f"TIMESTAMP '{to_iso_format(aware)}'"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    return str(value)


__all__ = ["render_predicate"]

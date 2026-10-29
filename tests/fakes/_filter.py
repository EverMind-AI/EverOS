"""Evaluate a backend-neutral ``Filter`` against one record's fields."""

from __future__ import annotations

__all__ = ["matches"]

import operator
from collections.abc import Callable, Mapping
from typing import Any

from everos.memory.ports import All, AnyOf, Comparison, Contains, Filter, In, IsNull

_OPERATORS: dict[str, Callable[[Any, Any], bool]] = {
    "eq": operator.eq,
    "ne": operator.ne,
    "gt": operator.gt,
    "gte": operator.ge,
    "lt": operator.lt,
    "lte": operator.le,
}


def matches(where: Filter, fields: Mapping[str, Any]) -> bool:
    """Whether ``fields`` satisfies ``where`` (missing fields read as ``None``)."""
    match where:
        case All(children=children):
            return all(matches(c, fields) for c in children)
        case AnyOf(children=children):
            return any(matches(c, fields) for c in children)
        case Comparison(field=field, operator=op, value=value):
            actual = fields.get(field)
            return actual is not None and _OPERATORS[op](actual, value)
        case In(field=field, values=values):
            return fields.get(field) in values
        case Contains(field=field, value=value):
            return value in (fields.get(field) or "")
        case IsNull(field=field):
            return fields.get(field) is None
    raise TypeError(f"unsupported filter node: {type(where).__name__}")

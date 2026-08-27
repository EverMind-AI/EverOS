"""Validate the public filter DSL and build a backend-neutral predicate.

Storage-specific syntax is intentionally absent from this module. Adapters
render the resulting predicate in their own persistence packages.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any, Final, Literal

from everos.component.utils.datetime import ensure_utc, from_iso_format, from_timestamp
from everos.core.errors import FilterError as FilterError
from everos.infra.persistence.index import (
    Predicate,
    Scalar,
    all_of,
    any_of,
    compare,
    contains,
    eq,
    is_null,
    one_of,
)

from .dto import FilterNode

_FieldKind = Literal["str", "ts", "array_str"]


@dataclass(frozen=True)
class _FieldSpec:
    column: str
    kind: _FieldKind


ALLOWED_FIELDS: Final[dict[str, _FieldSpec]] = {
    "session_id": _FieldSpec("session_id", "str"),
    "parent_type": _FieldSpec("parent_type", "str"),
    "parent_id": _FieldSpec("parent_id", "str"),
    "timestamp": _FieldSpec("timestamp", "ts"),
    "sender_id": _FieldSpec("sender_ids", "array_str"),
}

RESERVED_FIELDS: Final[frozenset[str]] = frozenset(
    {"owner_id", "owner_type", "app_id", "project_id"}
)


def compile_filters(
    node: FilterNode | None,
    *,
    owner_id: str,
    owner_type: str,
    app_id: str = "default",
    project_id: str = "default",
) -> Predicate:
    """Validate request filters and return one normalized predicate tree."""
    base: list[Predicate] = [
        eq("owner_id", owner_id),
        eq("owner_type", owner_type),
        eq("app_id", app_id),
        eq("project_id", project_id),
    ]
    if owner_type == "user":
        base.append(is_null("deprecated_by"))
    if node is not None:
        compiled = _compile_node(node.model_dump(exclude_none=True))
        if compiled is not None:
            base.append(compiled)
    return all_of(*base)


def _compile_node(raw: dict[str, Any]) -> Predicate | None:
    raw = dict(raw)
    parts: list[Predicate] = []

    if (and_list := raw.pop("AND", None)) is not None:
        combinator = _compile_combinator(and_list, "AND")
        if combinator is not None:
            parts.append(combinator)
    if (or_list := raw.pop("OR", None)) is not None:
        combinator = _compile_combinator(or_list, "OR")
        if combinator is not None:
            parts.append(combinator)

    for field, value in raw.items():
        if field in RESERVED_FIELDS:
            raise FilterError(
                f"filter field {field!r} is reserved; pass it at the top of the request"
            )
        if field not in ALLOWED_FIELDS:
            raise FilterError(f"unsupported filter field: {field!r}")
        parts.append(compile_predicate(field, value))

    return all_of(*parts) if parts else None


def _compile_combinator(
    children: list[dict[str, Any]], op: Literal["AND", "OR"]
) -> Predicate | None:
    if not isinstance(children, list):
        raise FilterError(f"{op} expects an array of nodes")
    fragments: list[Predicate] = []
    for child in children:
        if not isinstance(child, dict):
            raise FilterError(f"{op} children must be objects")
        compiled = _compile_node(child)
        if compiled is not None:
            fragments.append(compiled)
    if not fragments:
        return None
    return all_of(*fragments) if op == "AND" else any_of(*fragments)


def compile_predicate(field: str, value: Any) -> Predicate:
    """Validate and normalize one field clause into the neutral AST."""
    spec = ALLOWED_FIELDS[field]
    if isinstance(value, dict):
        if not value:
            raise FilterError(f"empty operator map for field {field!r}")
        return all_of(
            *(
                _compile_op_clause(spec, field, op, op_value)
                for op, op_value in value.items()
            )
        )
    return _compile_op_clause(spec, field, "eq", value)


def _compile_op_clause(spec: _FieldSpec, field: str, op: str, value: Any) -> Predicate:
    if op not in {"eq", "ne", "gt", "gte", "lt", "lte", "in"}:
        raise FilterError(f"unsupported operator {op!r} on field {field!r}")

    if spec.kind == "array_str":
        if op == "eq":
            return contains(spec.column, _require_str(value, field))
        if op == "in":
            values = _require_list(value, field)
            return any_of(
                *(contains(spec.column, _require_str(item, field)) for item in values)
            )
        raise FilterError(f"operator {op!r} is not supported on array field {field!r}")

    if op == "in":
        values = _require_list(value, field)
        return one_of(
            spec.column,
            [_normalize_literal(item, spec.kind, field) for item in values],
        )
    return compare(
        spec.column,
        op,  # type: ignore[arg-type]
        _normalize_literal(value, spec.kind, field),
    )


def _normalize_literal(value: Any, kind: _FieldKind, field: str) -> Scalar:
    if kind == "str":
        return _require_str(value, field)
    if kind == "ts":
        return _normalize_timestamp(value, field)
    raise FilterError(f"unsupported field kind {kind!r} for field {field!r}")


def _normalize_timestamp(value: Any, field: str) -> dt.datetime:
    if isinstance(value, bool):
        raise FilterError(f"timestamp value for {field!r} must be ms or ISO string")
    try:
        if isinstance(value, (int, float)):
            return from_timestamp(int(value))
        if isinstance(value, str):
            if "'" in value:
                raise FilterError(f"timestamp string for {field!r} contains a quote")
            parsed = ensure_utc(from_iso_format(value))
            assert parsed is not None
            return parsed
        if isinstance(value, dt.datetime):
            parsed = ensure_utc(value)
            assert parsed is not None
            return parsed
    except (TypeError, ValueError) as exc:
        raise FilterError(
            f"timestamp value for {field!r} must be ms or ISO string"
        ) from exc
    raise FilterError(f"timestamp value for {field!r} must be ms or ISO string")


def _require_str(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise FilterError(f"value for {field!r} must be a string")
    return value


def _require_list(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise FilterError(f"value for {field!r} with 'in' must be a non-empty list")
    return value


__all__ = [
    "ALLOWED_FIELDS",
    "RESERVED_FIELDS",
    "compile_filters",
    "compile_predicate",
]

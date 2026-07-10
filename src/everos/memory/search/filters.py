"""Filters DSL → derived-index filter compiler.

The Filters DSL is intentionally permissive at the JSON layer (so callers
can pass whatever they like and get a clean 400 if it is not supported)
and rigid at compile time. Field names are validated against a small
allow-list; operators against a closed enum; string literals are
single-quote-escaped. Timestamps are accepted as epoch milliseconds and
rendered as DataFusion ``TIMESTAMP '<iso>'`` literals.

``owner_id`` and ``owner_type`` are the hard partition keys; they are
not part of the DSL at all. :func:`compile_filters` injects them at the
top of the compiled string from :class:`SearchRequest` and rejects any
attempt to override them inside ``filters``.

Public surface
--------------

The compiler exposes three primitives so adjacent subpackages
(notably ``memory.get``) can build narrower DSLs without forking the
field allow-list:

* :data:`ALLOWED_FIELDS` — mapping ``field_name → _FieldSpec`` (column +
  kind). Iterate / membership-test only; do not mutate.
* :data:`RESERVED_FIELDS` — names rejected inside any ``filters`` block.
* :func:`compile_predicate` — render one ``{field: value}`` clause to
  a backend-specific predicate. Operator-map and equality-shorthand are
  both handled.

The high-level :func:`compile_filters` remains the entry point for
``/search`` (combinator-aware).
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import json
from typing import Any, Final

from everos.component.utils.datetime import (
    ensure_utc,
    from_iso_format,
    from_timestamp,
    to_iso_format,
    to_timestamp_ms,
)
from everos.core.errors import FilterError as FilterError

from .dto import FilterNode

# ── Allow-lists ──────────────────────────────────────────────────────────

_OP_MAP: Final[dict[str, str]] = {
    "eq": "=",
    "ne": "!=",
    "gt": ">",
    "gte": ">=",
    "lt": "<",
    "lte": "<=",
    "in": "IN",
}
_MILVUS_OP_MAP: Final[dict[str, str]] = {
    "eq": "==",
    "ne": "!=",
    "gt": ">",
    "gte": ">=",
    "lt": "<",
    "lte": "<=",
    "in": "in",
}

# Field kinds: ``str`` rendered as ``'<escaped>'``; ``ts`` rendered as
# ``TIMESTAMP '<iso>'`` (DataFusion timestamp literal); ``array_str``
# uses DataFusion's ``array_has`` on a list column.
_FieldKind = str  # one of: "str" | "ts" | "array_str"


class _FieldSpec:
    __slots__ = ("column", "kind")

    def __init__(self, column: str, kind: _FieldKind) -> None:
        self.column = column
        self.kind = kind


ALLOWED_FIELDS: Final[dict[str, _FieldSpec]] = {
    "session_id": _FieldSpec("session_id", "str"),
    "parent_type": _FieldSpec("parent_type", "str"),
    "parent_id": _FieldSpec("parent_id", "str"),
    "timestamp": _FieldSpec("timestamp", "ts"),
    "sender_id": _FieldSpec("sender_ids", "array_str"),
}

# Fields the caller is explicitly **not** allowed to place inside
# ``filters``; they live at the top of :class:`SearchRequest` and are
# injected by :func:`compile_filters`. Rejecting them here turns a
# silent override into a 400.
RESERVED_FIELDS: Final[frozenset[str]] = frozenset(
    {"owner_id", "owner_type", "app_id", "project_id"}
)


@dataclasses.dataclass(frozen=True)
class BackendFilters:
    """Compiled filters for every supported derived index backend."""

    lancedb: str
    milvus: str

    def __str__(self) -> str:
        return self.lancedb


# ── Public API ───────────────────────────────────────────────────────────


def compile_filters(
    node: FilterNode | None,
    *,
    owner_id: str,
    owner_type: str,
    app_id: str = "default",
    project_id: str = "default",
) -> str:
    """Compile a request's filters into a single LanceDB ``where`` string.

    The base clause always pins the hard partition keys (``owner_id`` /
    ``owner_type`` and the ``app_id`` / ``project_id`` scope segments) to
    the request's top-level values; anything in ``node`` is appended with
    an ``AND``. Pinning app/project here is what isolates one space's rows
    from another — omitting it would let a query bleed across spaces. Both
    ``/search`` and ``/get`` share this compile path.
    """
    return _compile_filters_backend(
        node,
        owner_id=owner_id,
        owner_type=owner_type,
        app_id=app_id,
        project_id=project_id,
        backend="lancedb",
    )


def compile_filters_for_backends(
    node: FilterNode | None,
    *,
    owner_id: str,
    owner_type: str,
    app_id: str = "default",
    project_id: str = "default",
) -> BackendFilters:
    """Compile request filters for every supported derived index backend."""
    return BackendFilters(
        lancedb=_compile_filters_backend(
            node,
            owner_id=owner_id,
            owner_type=owner_type,
            app_id=app_id,
            project_id=project_id,
            backend="lancedb",
        ),
        milvus=_compile_filters_backend(
            node,
            owner_id=owner_id,
            owner_type=owner_type,
            app_id=app_id,
            project_id=project_id,
            backend="milvus",
        ),
    )


def _compile_filters_backend(
    node: FilterNode | None,
    *,
    owner_id: str,
    owner_type: str,
    app_id: str,
    project_id: str,
    backend: str,
) -> str:
    base = _base_clauses(
        owner_id=owner_id,
        owner_type=owner_type,
        app_id=app_id,
        project_id=project_id,
        backend=backend,
    )
    # Only episode / atomic_fact tables carry the ``deprecated_by`` column
    # (Reflection V1 marks superseded entries). Agent tables don't have it.
    if owner_type == "user":
        base.append(
            "deprecated_by IS NULL" if backend == "lancedb" else "deprecated_by is null"
        )
    if node is None:
        return " AND ".join(base)
    compiled = _compile_node(node.model_dump(exclude_none=True), backend=backend)
    if not compiled:
        return " AND ".join(base)
    return " AND ".join([*base, compiled])


# ── Internals ────────────────────────────────────────────────────────────


def _compile_node(raw: dict[str, Any], *, backend: str = "lancedb") -> str:
    """Walk one DSL node; return the matching SQL fragment (no leading parens).

    Empty nodes yield ``""`` so :func:`compile_filters` can skip the
    trailing ``AND``.
    """
    raw = dict(raw)  # never mutate the caller's dict
    parts: list[str] = []

    if (and_list := raw.pop("AND", None)) is not None:
        parts.append(_compile_combinator(and_list, "AND", backend=backend))
    if (or_list := raw.pop("OR", None)) is not None:
        parts.append(_compile_combinator(or_list, "OR", backend=backend))

    for field, value in raw.items():
        if field in RESERVED_FIELDS:
            raise FilterError(
                f"filter field {field!r} is reserved; pass it at the top of the request"
            )
        if field not in ALLOWED_FIELDS:
            raise FilterError(f"unsupported filter field: {field!r}")
        parts.append(compile_predicate(field, value, backend=backend))

    # Drop empty fragments coming from empty AND/OR arrays.
    parts = [p for p in parts if p]
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return " AND ".join(parts)


def _compile_combinator(
    children: list[dict[str, Any]], op: str, *, backend: str = "lancedb"
) -> str:
    """Render an ``AND`` / ``OR`` array of child nodes."""
    if not isinstance(children, list):
        raise FilterError(f"{op} expects an array of nodes")
    fragments: list[str] = []
    for child in children:
        if not isinstance(child, dict):
            raise FilterError(f"{op} children must be objects")
        compiled = _compile_node(child, backend=backend)
        if compiled:
            fragments.append(f"({compiled})")
    if not fragments:
        return ""
    if len(fragments) == 1:
        # No need for the surrounding combinator when only one effective child.
        return fragments[0]
    glue = f" {op} "
    return "(" + glue.join(fragments) + ")"


def compile_predicate(field: str, value: Any, *, backend: str = "lancedb") -> str:
    """Render one ``"<field>": <value>`` clause to SQL.

    Public primitive — :mod:`memory.get` builds a flat (no AND/OR)
    DSL on top of it. Callers must pre-validate ``field`` against
    :data:`ALLOWED_FIELDS` and :data:`RESERVED_FIELDS`; this function
    will ``KeyError`` on unknown fields.

    ``value`` is either a scalar (equality shorthand) or an
    ``{"<op>": <scalar | list>}`` map. Mixing multiple operators in one
    dict is allowed and folds with ``AND``::

        "timestamp": {"gte": 1, "lt": 2}
        →  (timestamp >= TIMESTAMP '...' AND timestamp < TIMESTAMP '...')
    """
    spec = ALLOWED_FIELDS[field]
    if isinstance(value, dict):
        if not value:
            raise FilterError(f"empty operator map for field {field!r}")
        clauses = [
            _compile_op_clause(spec, field, op, op_val, backend=backend)
            for op, op_val in value.items()
        ]
        if len(clauses) == 1:
            return clauses[0]
        return "(" + " AND ".join(clauses) + ")"
    # Equality shorthand.
    return _compile_op_clause(spec, field, "eq", value, backend=backend)


def _compile_op_clause(
    spec: _FieldSpec, field: str, op: str, value: Any, *, backend: str = "lancedb"
) -> str:
    """Render a single ``<field> <op> <value>`` clause."""
    if op not in _OP_MAP:
        raise FilterError(f"unsupported operator {op!r} on field {field!r}")
    op_map = _MILVUS_OP_MAP if backend == "milvus" else _OP_MAP
    sql_op = op_map[op]
    column = _column_for_backend(spec, backend)

    if spec.kind == "array_str":
        # Only equality / membership make sense on a list column.
        fn = "array_contains" if backend == "milvus" else "array_has"
        if op == "eq":
            literal = _render_str_literal(_require_str(value, field), backend)
            return f"{fn}({column}, {literal})"
        if op == "in":
            items = _require_list(value, field)
            literals = [
                _render_str_literal(_require_str(v, field), backend) for v in items
            ]
            inner = " OR ".join(f"{fn}({column}, {lit})" for lit in literals)
            return f"({inner})"
        raise FilterError(f"operator {op!r} is not supported on array field {field!r}")

    if op == "in":
        items = _require_list(value, field)
        literals = [
            _render_literal(v, spec.kind, field, backend=backend) for v in items
        ]
        if backend == "milvus":
            return f"{column} in [{', '.join(literals)}]"
        return f"{column} IN ({', '.join(literals)})"

    literal = _render_literal(value, spec.kind, field, backend=backend)
    return f"{column} {sql_op} {literal}"


# ── Literal rendering ────────────────────────────────────────────────────


def _render_literal(
    value: Any, kind: _FieldKind, field: str, *, backend: str = "lancedb"
) -> str:
    if kind == "str":
        return _render_str_literal(_require_str(value, field), backend)
    if kind == "ts":
        if backend == "milvus":
            return str(_render_ts_ms(value, field))
        return f"TIMESTAMP '{_render_ts(value, field)}'"
    raise FilterError(f"unsupported field kind {kind!r} for field {field!r}")


def _render_ts(value: Any, field: str) -> str:
    """Accept epoch ms (int / float) or an ISO 8601 string; emit ISO."""
    if isinstance(value, bool):  # bools subclass int — reject early
        raise FilterError(f"timestamp value for {field!r} must be ms or ISO string")
    if isinstance(value, (int, float)):
        return to_iso_format(from_timestamp(int(value)))
    if isinstance(value, str):
        # Trust the caller-supplied ISO string but escape quotes defensively.
        if "'" in value:
            raise FilterError(f"timestamp string for {field!r} contains a quote")
        return value
    if isinstance(value, _dt.datetime):
        return to_iso_format(value)
    raise FilterError(f"timestamp value for {field!r} must be ms or ISO string")


def _render_ts_ms(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise FilterError(f"timestamp value for {field!r} must be ms or ISO string")
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        if "'" in value:
            raise FilterError(f"timestamp string for {field!r} contains a quote")
        try:
            aware = ensure_utc(from_iso_format(value))
        except (TypeError, ValueError) as exc:
            raise FilterError(
                f"timestamp value for {field!r} must be ms or ISO string"
            ) from exc
        assert aware is not None
        return to_timestamp_ms(aware)
    if isinstance(value, _dt.datetime):
        aware = ensure_utc(value)
        assert aware is not None
        return to_timestamp_ms(aware)
    raise FilterError(f"timestamp value for {field!r} must be ms or ISO string")


def _render_str_literal(value: str, backend: str) -> str:
    if backend == "milvus":
        return json.dumps(value, ensure_ascii=False)
    return f"'{_escape_str(value)}'"


def _base_clauses(
    *,
    owner_id: str,
    owner_type: str,
    app_id: str,
    project_id: str,
    backend: str,
) -> list[str]:
    eq = "==" if backend == "milvus" else "="
    return [
        f"owner_id {eq} {_render_str_literal(owner_id, backend)}",
        f"owner_type {eq} {_render_str_literal(owner_type, backend)}",
        f"app_id {eq} {_render_str_literal(app_id, backend)}",
        f"project_id {eq} {_render_str_literal(project_id, backend)}",
    ]


def _column_for_backend(spec: _FieldSpec, backend: str) -> str:
    if backend == "milvus" and spec.kind == "ts":
        return f"{spec.column}_ms"
    return spec.column


def _escape_str(value: str) -> str:
    """Double single quotes — SQL-standard escape for a single-quoted literal."""
    return value.replace("'", "''")


def _require_str(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise FilterError(f"value for {field!r} must be a string")
    return value


def _require_list(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise FilterError(f"value for {field!r} with 'in' must be a non-empty list")
    return value

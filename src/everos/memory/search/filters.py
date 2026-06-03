"""Filters DSL → LanceDB ``where`` string compiler.

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
  SQL. Operator-map and equality-shorthand are both handled.

The high-level :func:`compile_filters` remains the entry point for
``/search`` (combinator-aware).
"""

from __future__ import annotations

import datetime as _dt
from typing import Any, Final

from everos.component.utils.datetime import from_timestamp, to_iso_format

from .dto import FilterNode


class FilterError(ValueError):
    """Raised when the DSL contains a disallowed field, operator, or value."""


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
    base = [
        f"owner_id = '{_escape_str(owner_id)}'",
        f"owner_type = '{owner_type}'",
        f"app_id = '{_escape_str(app_id)}'",
        f"project_id = '{_escape_str(project_id)}'",
    ]
    if node is None:
        return " AND ".join(base)
    compiled = _compile_node(node.model_dump(exclude_none=True))
    if not compiled:
        return " AND ".join(base)
    return " AND ".join([*base, compiled])


# ── Internals ────────────────────────────────────────────────────────────


def _compile_node(raw: dict[str, Any]) -> str:
    """Walk one DSL node; return the matching SQL fragment (no leading parens).

    Empty nodes yield ``""`` so :func:`compile_filters` can skip the
    trailing ``AND``.
    """
    raw = dict(raw)  # never mutate the caller's dict
    parts: list[str] = []

    if (and_list := raw.pop("AND", None)) is not None:
        parts.append(_compile_combinator(and_list, "AND"))
    if (or_list := raw.pop("OR", None)) is not None:
        parts.append(_compile_combinator(or_list, "OR"))

    for field, value in raw.items():
        if field in RESERVED_FIELDS:
            raise FilterError(
                f"filter field {field!r} is reserved; pass it at the top of the request"
            )
        if field not in ALLOWED_FIELDS:
            raise FilterError(f"unsupported filter field: {field!r}")
        parts.append(compile_predicate(field, value))

    # Drop empty fragments coming from empty AND/OR arrays.
    parts = [p for p in parts if p]
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return " AND ".join(parts)


def _compile_combinator(children: list[dict[str, Any]], op: str) -> str:
    """Render an ``AND`` / ``OR`` array of child nodes."""
    if not isinstance(children, list):
        raise FilterError(f"{op} expects an array of nodes")
    fragments: list[str] = []
    for child in children:
        if not isinstance(child, dict):
            raise FilterError(f"{op} children must be objects")
        compiled = _compile_node(child)
        if compiled:
            fragments.append(f"({compiled})")
    if not fragments:
        return ""
    if len(fragments) == 1:
        # No need for the surrounding combinator when only one effective child.
        return fragments[0]
    glue = f" {op} "
    return "(" + glue.join(fragments) + ")"


def compile_predicate(field: str, value: Any) -> str:
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
            _compile_op_clause(spec, field, op, op_val) for op, op_val in value.items()
        ]
        if len(clauses) == 1:
            return clauses[0]
        return "(" + " AND ".join(clauses) + ")"
    # Equality shorthand.
    return _compile_op_clause(spec, field, "eq", value)


def _compile_op_clause(spec: _FieldSpec, field: str, op: str, value: Any) -> str:
    """Render a single ``<field> <op> <value>`` clause."""
    if op not in _OP_MAP:
        raise FilterError(f"unsupported operator {op!r} on field {field!r}")
    sql_op = _OP_MAP[op]

    if spec.kind == "array_str":
        # Only equality / membership make sense on a list column.
        if op == "eq":
            literal = _escape_str(_require_str(value, field))
            return f"array_has({spec.column}, '{literal}')"
        if op == "in":
            items = _require_list(value, field)
            literals = [f"'{_escape_str(_require_str(v, field))}'" for v in items]
            inner = " OR ".join(f"array_has({spec.column}, {lit})" for lit in literals)
            return f"({inner})"
        raise FilterError(f"operator {op!r} is not supported on array field {field!r}")

    if op == "in":
        items = _require_list(value, field)
        literals = [_render_literal(v, spec.kind, field) for v in items]
        return f"{spec.column} IN ({', '.join(literals)})"

    return f"{spec.column} {sql_op} {_render_literal(value, spec.kind, field)}"


# ── Literal rendering ────────────────────────────────────────────────────


def _render_literal(value: Any, kind: _FieldKind, field: str) -> str:
    if kind == "str":
        return f"'{_escape_str(_require_str(value, field))}'"
    if kind == "ts":
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

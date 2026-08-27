"""Tests for ``memory.get.filters_adapter.compile_filters_for_get``.

The adapter is a thin wrapper around
:func:`everos.memory.search.compile_filters` — these tests pin the
behaviour /get callers depend on:

* base clause shape (owner / owner_type / app / project / deprecated_by)
* flat multi-field → implicit conjunction
* reserved field (``owner_id`` / ``owner_type`` inside ``filters``)
  → :class:`FilterError`
* unknown field → :class:`FilterError`
* top-level ``AND`` / ``OR`` combinators are accepted (parity with
  ``/search`` — the wiki §附录 C restriction was dropped 2026-05-16)
* ``timestamp`` range (multi-op map) folds into the conjunction
* ``sender_id`` is an array column → ``Contains`` nodes

Assertions target the backend-neutral predicate tree; rendered syntax is an
adapter concern and is pinned in ``tests/unit/test_infra``.
"""

from __future__ import annotations

import pytest

from everos.component.utils.datetime import from_iso_format, from_timestamp
from everos.infra.persistence.index import (
    All,
    AnyOf,
    Comparison,
    Contains,
    Predicate,
    contains,
    eq,
    is_null,
)
from everos.memory.get.filters_adapter import compile_filters_for_get
from everos.memory.search import FilterError, FilterNode


def _clauses(predicate: Predicate) -> list[Predicate]:
    """Flatten a compiled tree into its leaf clauses, descending into groups."""
    if isinstance(predicate, All | AnyOf):
        return [leaf for child in predicate.children for leaf in _clauses(child)]
    return [predicate]


def _groups(predicate: Predicate, kind: type[Predicate]) -> list[Predicate]:
    """Every group node of ``kind`` in the tree, outermost first."""
    found: list[Predicate] = [predicate] if isinstance(predicate, kind) else []
    if isinstance(predicate, All | AnyOf):
        for child in predicate.children:
            found.extend(_groups(child, kind))
    return found


def _u1_filter(raw: dict[str, object]) -> Predicate:
    node = FilterNode.model_validate(raw)
    return compile_filters_for_get(node, owner_id="u1", owner_type="user")


def test_no_filters_emits_base_clause() -> None:
    """``filters=None`` → owner + app/project scope clauses, AND-joined."""
    where = compile_filters_for_get(None, owner_id="u1", owner_type="user")
    assert where == All(
        (
            eq("owner_id", "u1"),
            eq("owner_type", "user"),
            eq("app_id", "default"),
            eq("project_id", "default"),
            is_null("deprecated_by"),
        )
    )


def test_no_filters_agent_omits_deprecated_by() -> None:
    """Agent tables lack ``deprecated_by`` — clause must be absent."""
    where = compile_filters_for_get(None, owner_id="bot", owner_type="agent")
    assert is_null("deprecated_by") not in _clauses(where)


def test_owner_id_is_carried_verbatim() -> None:
    """The compiler does not pre-quote — escaping belongs to the adapter."""
    where = compile_filters_for_get(None, owner_id="o'reilly", owner_type="user")
    assert eq("owner_id", "o'reilly") in _clauses(where)


def test_flat_multi_field_renders_implicit_and() -> None:
    """Multiple top-level fields → implicit conjunction with the base clauses."""
    where = _u1_filter({"session_id": "sess_a", "parent_id": "mc_x"})
    # 5 base scope clauses + 2 filter fields, flattened into one conjunction.
    assert where == All(
        (
            eq("owner_id", "u1"),
            eq("owner_type", "user"),
            eq("app_id", "default"),
            eq("project_id", "default"),
            is_null("deprecated_by"),
            eq("session_id", "sess_a"),
            eq("parent_id", "mc_x"),
        )
    )


def test_reserved_owner_id_in_filters_raises() -> None:
    """``owner_id`` inside ``filters`` is a hard error (must be top level)."""
    node = FilterNode.model_validate({"owner_id": "u1"})
    with pytest.raises(FilterError, match="reserved"):
        compile_filters_for_get(node, owner_id="u1", owner_type="user")


def test_reserved_owner_type_in_filters_raises() -> None:
    """``owner_type`` inside ``filters`` is also reserved."""
    node = FilterNode.model_validate({"owner_type": "user"})
    with pytest.raises(FilterError, match="reserved"):
        compile_filters_for_get(node, owner_id="u1", owner_type="user")


def test_unsupported_field_raises() -> None:
    """Any field outside the shared allow-list → :class:`FilterError`."""
    node = FilterNode.model_validate({"random_attr": "x"})
    with pytest.raises(FilterError, match="unsupported"):
        compile_filters_for_get(node, owner_id="u1", owner_type="user")


def test_timestamp_range_folds_into_conjunction() -> None:
    """Multi-op map on one field folds with AND (reused from /search)."""
    where = _u1_filter({"timestamp": {"gte": 1704067200000, "lt": 1735689600000}})
    clauses = _clauses(where)
    assert Comparison("timestamp", "gte", from_timestamp(1704067200000)) in clauses
    assert Comparison("timestamp", "lt", from_timestamp(1735689600000)) in clauses
    assert _groups(where, AnyOf) == []


def test_sender_id_in_list_becomes_contains() -> None:
    """``sender_id`` is an array column — ``in`` → OR of ``Contains``."""
    where = _u1_filter({"sender_id": {"in": ["alice", "bob"]}})
    disjunctions = _groups(where, AnyOf)
    assert len(disjunctions) == 1
    assert disjunctions[0].children == (  # type: ignore[attr-defined]
        Contains("sender_ids", "alice"),
        Contains("sender_ids", "bob"),
    )


def test_sender_id_eq_shorthand_becomes_contains() -> None:
    """Equality shorthand on an array column → a single ``Contains``."""
    where = _u1_filter({"sender_id": "alice"})
    assert contains("sender_ids", "alice") in _clauses(where)


def test_parent_id_eq_shorthand_stays_scalar_eq() -> None:
    """``parent_id`` is a scalar string column → plain equality."""
    where = _u1_filter({"parent_id": "mc_42"})
    assert eq("parent_id", "mc_42") in _clauses(where)


def test_top_level_and_folds_into_conjunction() -> None:
    """``AND`` combinator compiles like /search."""
    where = _u1_filter({"AND": [{"session_id": "sess_a"}, {"parent_id": "mc_x"}]})
    clauses = _clauses(where)
    assert eq("owner_id", "u1") in clauses
    assert eq("owner_type", "user") in clauses
    assert eq("session_id", "sess_a") in clauses
    assert eq("parent_id", "mc_x") in clauses
    assert _groups(where, AnyOf) == []


def test_top_level_or_emits_disjunction() -> None:
    """``OR`` combinator emits a disjunction between sibling predicates."""
    where = _u1_filter({"OR": [{"session_id": "sess_a"}, {"session_id": "sess_b"}]})
    disjunctions = _groups(where, AnyOf)
    assert len(disjunctions) == 1
    branches = [_clauses(child) for child in disjunctions[0].children]  # type: ignore[attr-defined]
    assert branches == [[eq("session_id", "sess_a")], [eq("session_id", "sess_b")]]


def test_ne_operator_kept_as_ne() -> None:
    """``ne`` op compiles to a ``ne`` comparison on str fields."""
    where = _u1_filter({"session_id": {"ne": "sess_internal"}})
    assert Comparison("session_id", "ne", "sess_internal") in _clauses(where)


def test_timestamp_iso_string_normalized_to_datetime() -> None:
    """ISO 8601 strings are accepted alongside epoch ms and normalized."""
    where = _u1_filter({"timestamp": {"gte": "2026-01-04T00:00:00+00:00"}})
    expected = from_iso_format("2026-01-04T00:00:00+00:00")
    assert Comparison("timestamp", "gte", expected) in _clauses(where)


def test_nested_and_inside_or() -> None:
    """``AND`` nested inside ``OR`` — combinators compose recursively."""
    where = _u1_filter(
        {
            "OR": [
                {"AND": [{"session_id": "sess_a"}, {"parent_id": "mc_x"}]},
                {"session_id": "sess_b"},
            ]
        }
    )
    disjunctions = _groups(where, AnyOf)
    assert len(disjunctions) == 1
    branches = [_clauses(child) for child in disjunctions[0].children]  # type: ignore[attr-defined]
    assert branches == [
        [eq("session_id", "sess_a"), eq("parent_id", "mc_x")],
        [eq("session_id", "sess_b")],
    ]


# ── Malformed value shapes ──────────────────────────────────────────────


def test_in_op_with_non_list_rejected() -> None:
    """``in`` requires a non-empty list — a scalar is a hard error."""
    node = FilterNode.model_validate({"session_id": {"in": "not_a_list"}})
    with pytest.raises(FilterError, match="non-empty list"):
        compile_filters_for_get(node, owner_id="u1", owner_type="user")


def test_in_op_with_empty_list_rejected() -> None:
    """``in: []`` is invalid — must contain at least one value."""
    node = FilterNode.model_validate({"session_id": {"in": []}})
    with pytest.raises(FilterError, match="non-empty list"):
        compile_filters_for_get(node, owner_id="u1", owner_type="user")


def test_empty_operator_map_rejected() -> None:
    """``{}`` as a field value (no op) is a hard error."""
    node = FilterNode.model_validate({"timestamp": {}})
    with pytest.raises(FilterError, match="empty operator map"):
        compile_filters_for_get(node, owner_id="u1", owner_type="user")


def test_unknown_op_rejected() -> None:
    """``between`` / other non-allow-listed ops surface as :class:`FilterError`."""
    node = FilterNode.model_validate({"timestamp": {"between": [1, 2]}})
    with pytest.raises(FilterError, match="operator"):
        compile_filters_for_get(node, owner_id="u1", owner_type="user")


def test_sender_id_gt_rejected() -> None:
    """``gt`` on an ``array_str`` column is not supported (semantics unclear)."""
    node = FilterNode.model_validate({"sender_id": {"gt": "alice"}})
    with pytest.raises(FilterError, match="not supported on array"):
        compile_filters_for_get(node, owner_id="u1", owner_type="user")


def test_non_string_in_str_field_rejected() -> None:
    """``session_id`` is a str field — passing an int is a typed error."""
    node = FilterNode.model_validate({"session_id": {"in": [1, 2]}})
    with pytest.raises(FilterError, match="must be a string"):
        compile_filters_for_get(node, owner_id="u1", owner_type="user")

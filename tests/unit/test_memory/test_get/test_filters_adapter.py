"""Tests for ``memory.get.filters_adapter.compile_filters_for_get``.

The adapter is a thin wrapper around
:func:`everos.memory.search.compile_filters` — these tests pin the
behaviour /get callers depend on:

* base clause shape (``owner_id = '...' AND owner_type = '...'``)
* flat multi-field → implicit ``AND``
* reserved field (``owner_id`` / ``owner_type`` inside ``filters``)
  → :class:`FilterError`
* unknown field → :class:`FilterError`
* top-level ``AND`` / ``OR`` combinators are accepted (parity with
  ``/search`` — the wiki §附录 C restriction was dropped 2026-05-16)
* ``timestamp`` range (multi-op map) renders ``AND``-folded clauses
* ``sender_id`` is an array column → ``array_has(...)`` rendering
"""

from __future__ import annotations

import pytest

from everos.memory.get.filters_adapter import compile_filters_for_get
from everos.memory.search import FilterError, FilterNode


def test_no_filters_emits_base_clause() -> None:
    """``filters=None`` → owner + app/project scope clauses AND-joined."""
    where = compile_filters_for_get(None, owner_id="u1", owner_type="user")
    assert where == (
        "owner_id = 'u1' AND owner_type = 'user' "
        "AND app_id = 'default' AND project_id = 'default' "
        "AND deprecated_by IS NULL"
    )


def test_no_filters_agent_omits_deprecated_by() -> None:
    """Agent tables lack ``deprecated_by`` — clause must be absent."""
    where = compile_filters_for_get(None, owner_id="bot", owner_type="agent")
    assert "deprecated_by" not in where


def test_owner_id_quote_is_escaped() -> None:
    """SQL-standard double-quote escape on ``owner_id``."""
    where = compile_filters_for_get(None, owner_id="o'reilly", owner_type="user")
    assert where == (
        "owner_id = 'o''reilly' AND owner_type = 'user' "
        "AND app_id = 'default' AND project_id = 'default' "
        "AND deprecated_by IS NULL"
    )


def test_flat_multi_field_renders_implicit_and() -> None:
    """Multiple top-level fields → implicit ``AND`` between predicates."""
    node = FilterNode.model_validate({"session_id": "sess_a", "parent_id": "mc_x"})
    where = compile_filters_for_get(node, owner_id="u1", owner_type="user")
    # Field iteration order follows insertion order, so both are present.
    assert "owner_id = 'u1'" in where
    assert "owner_type = 'user'" in where
    assert "session_id = 'sess_a'" in where
    assert "parent_id = 'mc_x'" in where
    # 5 base scope clauses + 2 filter fields = 7 clauses → 6 ' AND ' joins.
    assert where.count(" AND ") == 6


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


def test_timestamp_range_renders_and_folded() -> None:
    """Multi-op map on one field folds with ``AND`` (reused from /search)."""
    node = FilterNode.model_validate(
        {"timestamp": {"gte": 1704067200000, "lt": 1735689600000}}
    )
    where = compile_filters_for_get(node, owner_id="u1", owner_type="user")
    assert "timestamp >= TIMESTAMP '" in where
    assert "timestamp < TIMESTAMP '" in where
    # The two clauses are AND-joined inside one parenthesised group.
    assert "(timestamp >= TIMESTAMP" in where
    assert " AND timestamp < TIMESTAMP" in where


def test_sender_id_in_list_renders_array_has() -> None:
    """``sender_id`` is an array column — ``in`` → ``array_has(...) OR ...``."""
    node = FilterNode.model_validate({"sender_id": {"in": ["alice", "bob"]}})
    where = compile_filters_for_get(node, owner_id="u1", owner_type="user")
    assert "array_has(sender_ids, 'alice')" in where
    assert "array_has(sender_ids, 'bob')" in where


def test_sender_id_eq_shorthand_renders_array_has() -> None:
    """Equality shorthand on an array column → single ``array_has``."""
    node = FilterNode.model_validate({"sender_id": "alice"})
    where = compile_filters_for_get(node, owner_id="u1", owner_type="user")
    assert "array_has(sender_ids, 'alice')" in where


def test_parent_id_eq_shorthand_renders_scalar_eq() -> None:
    """``parent_id`` is a scalar string column → plain ``=``."""
    node = FilterNode.model_validate({"parent_id": "mc_42"})
    where = compile_filters_for_get(node, owner_id="u1", owner_type="user")
    assert "parent_id = 'mc_42'" in where


def test_top_level_and_renders_grouped_clause() -> None:
    """``AND`` combinator compiles like /search — parens-grouped fragments."""
    node = FilterNode.model_validate(
        {"AND": [{"session_id": "sess_a"}, {"parent_id": "mc_x"}]}
    )
    where = compile_filters_for_get(node, owner_id="u1", owner_type="user")
    # Base clause is always first; combinator output appended.
    assert where.startswith("owner_id = 'u1' AND owner_type = 'user' AND ")
    assert "session_id = 'sess_a'" in where
    assert "parent_id = 'mc_x'" in where


def test_top_level_or_renders_grouped_clause() -> None:
    """``OR`` combinator emits parens-grouped ``OR`` between sibling preds."""
    node = FilterNode.model_validate(
        {"OR": [{"session_id": "sess_a"}, {"session_id": "sess_b"}]}
    )
    where = compile_filters_for_get(node, owner_id="u1", owner_type="user")
    assert "session_id = 'sess_a'" in where
    assert "session_id = 'sess_b'" in where
    assert " OR " in where


def test_ne_operator_renders_not_equal() -> None:
    """``ne`` op compiles to ``!=`` on str fields."""
    node = FilterNode.model_validate({"session_id": {"ne": "sess_internal"}})
    where = compile_filters_for_get(node, owner_id="u1", owner_type="user")
    assert "session_id != 'sess_internal'" in where


def test_timestamp_iso_string_renders_literal() -> None:
    """ISO 8601 string is accepted as a timestamp literal (alongside epoch ms)."""
    node = FilterNode.model_validate(
        {"timestamp": {"gte": "2026-01-04T00:00:00+00:00"}}
    )
    where = compile_filters_for_get(node, owner_id="u1", owner_type="user")
    assert "timestamp >= TIMESTAMP '2026-01-04T00:00:00+00:00'" in where


def test_nested_and_inside_or() -> None:
    """``AND`` nested inside ``OR`` — combinators compose recursively."""
    node = FilterNode.model_validate(
        {
            "OR": [
                {"AND": [{"session_id": "sess_a"}, {"parent_id": "mc_x"}]},
                {"session_id": "sess_b"},
            ]
        }
    )
    where = compile_filters_for_get(node, owner_id="u1", owner_type="user")
    assert "session_id = 'sess_a'" in where
    assert "parent_id = 'mc_x'" in where
    assert "session_id = 'sess_b'" in where
    assert " OR " in where
    assert " AND " in where


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

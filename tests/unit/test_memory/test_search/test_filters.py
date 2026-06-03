"""Unit tests for the Filters DSL compiler."""

from __future__ import annotations

import pytest

from everos.memory.search import (
    FilterError,
    FilterNode,
    compile_filters,
)

# ── Base injection ───────────────────────────────────────────────────────


def test_no_filters_emits_base_clause() -> None:
    where = compile_filters(None, owner_id="alice", owner_type="user")
    assert where == (
        "owner_id = 'alice' AND owner_type = 'user' "
        "AND app_id = 'default' AND project_id = 'default'"
    )


def test_owner_type_agent_pinned() -> None:
    where = compile_filters(None, owner_id="alice", owner_type="agent")
    assert "owner_type = 'agent'" in where


def test_app_project_scope_pinned() -> None:
    where = compile_filters(
        None,
        owner_id="alice",
        owner_type="user",
        app_id="claude_code",
        project_id="oss",
    )
    assert "app_id = 'claude_code'" in where
    assert "project_id = 'oss'" in where


def test_owner_id_with_quote_is_escaped() -> None:
    where = compile_filters(None, owner_id="al'ice", owner_type="user")
    assert "owner_id = 'al''ice'" in where


# ── Equality / shorthand ────────────────────────────────────────────────


def test_flat_equality_shorthand() -> None:
    node = FilterNode(session_id="sess_a")  # type: ignore[call-arg]
    where = compile_filters(node, owner_id="alice", owner_type="user")
    assert "session_id = 'sess_a'" in where


def test_multiple_flat_fields_join_with_and() -> None:
    node = FilterNode.model_validate({"session_id": "sess_a", "parent_type": "memcell"})
    where = compile_filters(node, owner_id="alice", owner_type="user")
    assert "session_id = 'sess_a'" in where
    assert "parent_type = 'memcell'" in where


# ── Operators ───────────────────────────────────────────────────────────


def test_timestamp_gte_renders_timestamp_literal() -> None:
    node = FilterNode.model_validate({"timestamp": {"gte": 1704067200000}})
    where = compile_filters(node, owner_id="alice", owner_type="user")
    assert "timestamp >= TIMESTAMP '" in where


def test_timestamp_range_folds_with_and() -> None:
    node = FilterNode.model_validate(
        {"timestamp": {"gte": 1704067200000, "lt": 1740614399000}}
    )
    where = compile_filters(node, owner_id="alice", owner_type="user")
    assert "timestamp >= TIMESTAMP '" in where
    assert "timestamp < TIMESTAMP '" in where
    # Operators on the same field are wrapped in a single group.
    assert " AND " in where


def test_in_operator_string_field() -> None:
    node = FilterNode.model_validate({"parent_type": {"in": ["memcell", "episode"]}})
    where = compile_filters(node, owner_id="alice", owner_type="user")
    assert "parent_type IN ('memcell', 'episode')" in where


def test_in_operator_requires_non_empty_list() -> None:
    node = FilterNode.model_validate({"parent_type": {"in": []}})
    with pytest.raises(FilterError):
        compile_filters(node, owner_id="alice", owner_type="user")


def test_invalid_operator_rejected() -> None:
    node = FilterNode.model_validate({"timestamp": {"between": [1, 2]}})
    with pytest.raises(FilterError, match="operator"):
        compile_filters(node, owner_id="alice", owner_type="user")


# ── Combinators ─────────────────────────────────────────────────────────


def test_and_combinator() -> None:
    node = FilterNode.model_validate(
        {
            "AND": [
                {"timestamp": {"gte": 1704067200000}},
                {"timestamp": {"lt": 1740614399000}},
            ]
        }
    )
    where = compile_filters(node, owner_id="alice", owner_type="user")
    assert "timestamp >= TIMESTAMP '" in where
    assert "timestamp < TIMESTAMP '" in where
    assert " AND " in where


def test_or_combinator() -> None:
    node = FilterNode.model_validate(
        {
            "OR": [
                {"parent_type": "memcell"},
                {"parent_type": "episode"},
            ]
        }
    )
    where = compile_filters(node, owner_id="alice", owner_type="user")
    assert " OR " in where
    assert "parent_type = 'memcell'" in where
    assert "parent_type = 'episode'" in where


def test_nested_and_inside_or() -> None:
    node = FilterNode.model_validate(
        {
            "OR": [
                {"AND": [{"parent_type": "memcell"}, {"session_id": "sa"}]},
                {"parent_type": "episode"},
            ]
        }
    )
    where = compile_filters(node, owner_id="alice", owner_type="user")
    assert "parent_type = 'memcell'" in where
    assert "session_id = 'sa'" in where
    assert "parent_type = 'episode'" in where
    assert " OR " in where
    assert " AND " in where


def test_flat_field_alongside_and_combinator() -> None:
    node = FilterNode.model_validate(
        {
            "session_id": "sess_a",
            "AND": [{"timestamp": {"gte": 1}}],
        }
    )
    where = compile_filters(node, owner_id="alice", owner_type="user")
    assert "session_id = 'sess_a'" in where
    assert "timestamp >= TIMESTAMP '" in where


# ── Array field (sender_id → sender_ids) ────────────────────────────────


def test_sender_id_eq_uses_array_has() -> None:
    node = FilterNode.model_validate({"sender_id": "u_jason"})
    where = compile_filters(node, owner_id="alice", owner_type="user")
    assert "array_has(sender_ids, 'u_jason')" in where


def test_sender_id_in_expands_to_or_array_has() -> None:
    node = FilterNode.model_validate({"sender_id": {"in": ["u_a", "u_b"]}})
    where = compile_filters(node, owner_id="alice", owner_type="user")
    assert "array_has(sender_ids, 'u_a')" in where
    assert "array_has(sender_ids, 'u_b')" in where
    assert " OR " in where


def test_sender_id_gt_rejected() -> None:
    node = FilterNode.model_validate({"sender_id": {"gt": "x"}})
    with pytest.raises(FilterError, match="not supported on array"):
        compile_filters(node, owner_id="alice", owner_type="user")


# ── Safety ──────────────────────────────────────────────────────────────


def test_unknown_field_rejected() -> None:
    node = FilterNode.model_validate({"secret_field": "x"})
    with pytest.raises(FilterError, match="unsupported filter field"):
        compile_filters(node, owner_id="alice", owner_type="user")


def test_owner_id_in_filters_rejected() -> None:
    node = FilterNode.model_validate({"owner_id": "mallory"})
    with pytest.raises(FilterError, match="reserved"):
        compile_filters(node, owner_id="alice", owner_type="user")


def test_owner_type_in_filters_rejected() -> None:
    node = FilterNode.model_validate({"owner_type": "agent"})
    with pytest.raises(FilterError, match="reserved"):
        compile_filters(node, owner_id="alice", owner_type="user")


def test_string_with_single_quote_escaped() -> None:
    node = FilterNode.model_validate({"session_id": "ses's"})
    where = compile_filters(node, owner_id="alice", owner_type="user")
    assert "session_id = 'ses''s'" in where


def test_timestamp_string_with_quote_rejected() -> None:
    """ISO strings with embedded quotes can break the literal — reject loudly."""
    node = FilterNode.model_validate({"timestamp": {"gte": "2024-01'-01T00:00:00"}})
    with pytest.raises(FilterError, match="contains a quote"):
        compile_filters(node, owner_id="alice", owner_type="user")


def test_in_value_type_check() -> None:
    node = FilterNode.model_validate({"parent_type": {"in": [1, 2]}})
    with pytest.raises(FilterError, match="must be a string"):
        compile_filters(node, owner_id="alice", owner_type="user")


def test_bool_for_timestamp_rejected() -> None:
    node = FilterNode.model_validate({"timestamp": {"gte": True}})
    with pytest.raises(FilterError, match="timestamp value"):
        compile_filters(node, owner_id="alice", owner_type="user")


def test_empty_operator_map_rejected() -> None:
    node = FilterNode.model_validate({"timestamp": {}})
    with pytest.raises(FilterError, match="empty operator map"):
        compile_filters(node, owner_id="alice", owner_type="user")


def test_empty_and_array_skips_combinator() -> None:
    """Empty AND/OR arrays compile to no clauses — only the base remains."""
    node = FilterNode.model_validate({"AND": []})
    where = compile_filters(node, owner_id="alice", owner_type="user")
    assert where == (
        "owner_id = 'alice' AND owner_type = 'user' "
        "AND app_id = 'default' AND project_id = 'default'"
    )

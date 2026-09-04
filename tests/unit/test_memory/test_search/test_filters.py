"""Unit tests for the Filters DSL compiler.

The compiler's output is a backend-neutral :class:`Predicate` tree, so these
assertions are made against that tree rather than against any adapter's
rendered syntax. Quoting, literal formats and array-containment syntax belong
to the adapters and are pinned in ``tests/unit/test_infra`` instead.
"""

from __future__ import annotations

import datetime as dt

import pytest

from everos.component.utils.datetime import from_timestamp
from everos.infra.persistence.index import (
    All,
    AnyOf,
    Comparison,
    Contains,
    Predicate,
    contains,
    eq,
    is_null,
    one_of,
)
from everos.memory.search import (
    FilterError,
    FilterNode,
    compile_filters,
)

_BASE_USER = (
    eq("owner_id", "alice"),
    eq("owner_type", "user"),
    eq("app_id", "default"),
    eq("project_id", "default"),
    is_null("deprecated_by"),
)


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


def _user_filter(raw: dict[str, object]) -> Predicate:
    node = FilterNode.model_validate(raw)
    return compile_filters(node, owner_id="alice", owner_type="user")


# ── Base injection ───────────────────────────────────────────────────────


def test_no_filters_emits_base_clause() -> None:
    where = compile_filters(None, owner_id="alice", owner_type="user")
    assert where == All(_BASE_USER)


def test_no_filters_agent_omits_deprecated_by() -> None:
    where = compile_filters(None, owner_id="bot_42", owner_type="agent")
    assert where == All(
        (
            eq("owner_id", "bot_42"),
            eq("owner_type", "agent"),
            eq("app_id", "default"),
            eq("project_id", "default"),
        )
    )
    assert is_null("deprecated_by") not in _clauses(where)


def test_owner_type_agent_pinned() -> None:
    where = compile_filters(None, owner_id="alice", owner_type="agent")
    assert eq("owner_type", "agent") in _clauses(where)


def test_app_project_scope_pinned() -> None:
    where = compile_filters(
        None,
        owner_id="alice",
        owner_type="user",
        app_id="claude_code",
        project_id="oss",
    )
    assert eq("app_id", "claude_code") in _clauses(where)
    assert eq("project_id", "oss") in _clauses(where)


def test_owner_id_is_carried_verbatim() -> None:
    """The compiler must not pre-escape — quoting is the adapter's job.

    Escaping itself is asserted against the LanceDB renderer in
    ``tests/unit/test_infra/test_index_contract.py``.
    """
    where = compile_filters(None, owner_id="al'ice", owner_type="user")
    assert eq("owner_id", "al'ice") in _clauses(where)


# ── Equality / shorthand ────────────────────────────────────────────────


def test_flat_equality_shorthand() -> None:
    where = _user_filter({"session_id": "sess_a"})
    assert eq("session_id", "sess_a") in _clauses(where)


def test_multiple_flat_fields_join_with_and() -> None:
    where = _user_filter({"session_id": "sess_a", "parent_type": "memcell"})
    assert where == All(
        (*_BASE_USER, eq("session_id", "sess_a"), eq("parent_type", "memcell"))
    )


# ── Operators ───────────────────────────────────────────────────────────


def test_timestamp_gte_normalized_to_datetime() -> None:
    """Epoch ms become a real ``datetime`` in the AST, not a rendered literal."""
    where = _user_filter({"timestamp": {"gte": 1704067200000}})
    clause = Comparison("timestamp", "gte", from_timestamp(1704067200000))
    assert clause in _clauses(where)
    assert isinstance(clause.value, dt.datetime)


def test_timestamp_range_folds_with_and() -> None:
    where = _user_filter({"timestamp": {"gte": 1704067200000, "lt": 1740614399000}})
    clauses = _clauses(where)
    assert Comparison("timestamp", "gte", from_timestamp(1704067200000)) in clauses
    assert Comparison("timestamp", "lt", from_timestamp(1740614399000)) in clauses
    # Operators on the same field sit in the conjunction, not a disjunction.
    assert _groups(where, AnyOf) == []


def test_in_operator_string_field() -> None:
    where = _user_filter({"parent_type": {"in": ["memcell", "episode"]}})
    assert one_of("parent_type", ["memcell", "episode"]) in _clauses(where)


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
    where = _user_filter(
        {
            "AND": [
                {"timestamp": {"gte": 1704067200000}},
                {"timestamp": {"lt": 1740614399000}},
            ]
        }
    )
    clauses = _clauses(where)
    assert Comparison("timestamp", "gte", from_timestamp(1704067200000)) in clauses
    assert Comparison("timestamp", "lt", from_timestamp(1740614399000)) in clauses
    assert _groups(where, AnyOf) == []


def test_or_combinator() -> None:
    where = _user_filter(
        {"OR": [{"parent_type": "memcell"}, {"parent_type": "episode"}]}
    )
    disjunctions = _groups(where, AnyOf)
    assert len(disjunctions) == 1
    branches = [_clauses(child) for child in disjunctions[0].children]  # type: ignore[attr-defined]
    assert branches == [[eq("parent_type", "memcell")], [eq("parent_type", "episode")]]


def test_nested_and_inside_or() -> None:
    where = _user_filter(
        {
            "OR": [
                {"AND": [{"parent_type": "memcell"}, {"session_id": "sa"}]},
                {"parent_type": "episode"},
            ]
        }
    )
    disjunctions = _groups(where, AnyOf)
    assert len(disjunctions) == 1
    branches = [_clauses(child) for child in disjunctions[0].children]  # type: ignore[attr-defined]
    assert branches == [
        [eq("parent_type", "memcell"), eq("session_id", "sa")],
        [eq("parent_type", "episode")],
    ]


def test_flat_field_alongside_and_combinator() -> None:
    where = _user_filter({"session_id": "sess_a", "AND": [{"timestamp": {"gte": 1}}]})
    clauses = _clauses(where)
    assert eq("session_id", "sess_a") in clauses
    assert Comparison("timestamp", "gte", from_timestamp(1)) in clauses


# ── Array field (sender_id → sender_ids) ────────────────────────────────


def test_sender_id_eq_becomes_contains() -> None:
    where = _user_filter({"sender_id": "u_jason"})
    assert contains("sender_ids", "u_jason") in _clauses(where)


def test_sender_id_in_expands_to_or_of_contains() -> None:
    where = _user_filter({"sender_id": {"in": ["u_a", "u_b"]}})
    disjunctions = _groups(where, AnyOf)
    assert len(disjunctions) == 1
    assert disjunctions[0].children == (  # type: ignore[attr-defined]
        Contains("sender_ids", "u_a"),
        Contains("sender_ids", "u_b"),
    )


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


def test_string_with_single_quote_is_carried_verbatim() -> None:
    """Filter values reach the AST unmodified; the adapter does the quoting."""
    where = _user_filter({"session_id": "ses's"})
    assert eq("session_id", "ses's") in _clauses(where)


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
    assert where == compile_filters(None, owner_id="alice", owner_type="user")


# ── Deprecated exclusion ──────────────────────────────────────────────


def test_compile_filters_excludes_deprecated_by_for_user() -> None:
    where = compile_filters(None, owner_id="u_a", owner_type="user")
    assert is_null("deprecated_by") in _clauses(where)


def test_compile_filters_omits_deprecated_by_for_agent() -> None:
    where = compile_filters(None, owner_id="agent_1", owner_type="agent")
    assert is_null("deprecated_by") not in _clauses(where)

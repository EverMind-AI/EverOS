"""Unit tests for MongoFilterParser.

Tests cover:
- eq / in / gt / gte / lt / lte operators
- AND / OR combinators (including nested)
- Timestamp epoch ms -> datetime conversion
- Unknown fields silently ignored (allowlist behavior)
- Scope extraction (user_id, group_id)
- Edge cases: empty filters, None values
"""

from datetime import datetime, timezone

from agentic_layer.filter_parser import MongoFilterParser


class TestEqOperator:
    """Test implicit eq operator (plain value)."""

    def test_user_id_eq(self):
        mq, uid, gids = MongoFilterParser.parse({"user_id": "u1"})
        assert mq == {"user_id": "u1"}
        assert uid == "u1"
        assert gids is None

    def test_group_id_eq(self):
        mq, uid, gids = MongoFilterParser.parse({"group_id": "g1"})
        assert mq == {"group_id": "g1"}
        assert uid is None
        assert gids == ["g1"]

    def test_session_id_eq(self):
        mq, uid, gids = MongoFilterParser.parse({"user_id": "u1", "session_id": "s1"})
        assert mq == {"user_id": "u1", "session_id": "s1"}

    def test_user_and_group(self):
        mq, uid, gids = MongoFilterParser.parse({"user_id": "u1", "group_id": "g1"})
        assert mq == {"user_id": "u1", "group_id": "g1"}
        assert uid == "u1"
        assert gids == ["g1"]


class TestInOperator:
    """Test 'in' operator."""

    def test_group_id_in(self):
        mq, uid, gids = MongoFilterParser.parse({"group_id": {"in": ["g1", "g2"]}})
        assert mq == {"group_id": {"$in": ["g1", "g2"]}}
        assert gids == ["g1", "g2"]

    def test_user_id_in(self):
        mq, uid, gids = MongoFilterParser.parse({"user_id": {"in": ["u1", "u2"]}})
        assert mq == {"user_id": {"$in": ["u1", "u2"]}}
        assert uid == "u1"

    def test_session_id_in(self):
        mq, _, _ = MongoFilterParser.parse(
            {"user_id": "u1", "session_id": {"in": ["s1", "s2"]}}
        )
        assert mq["session_id"] == {"$in": ["s1", "s2"]}

    def test_user_id_in_empty_list(self):
        mq, uid, gids = MongoFilterParser.parse({"user_id": {"in": []}})
        assert mq == {"user_id": {"$in": []}}
        assert uid is None


class TestComparisonOperators:
    """Test gt, gte, lt, lte operators on allowed fields."""

    def test_session_id_gt(self):
        mq, _, _ = MongoFilterParser.parse(
            {"user_id": "u1", "session_id": {"gt": "s5"}}
        )
        assert mq["session_id"] == {"$gt": "s5"}

    def test_session_id_gte_lte(self):
        mq, _, _ = MongoFilterParser.parse(
            {"user_id": "u1", "session_id": {"gte": "s1", "lte": "s5"}}
        )
        assert mq["session_id"] == {"$gte": "s1", "$lte": "s5"}


class TestTimestampHandling:
    """Test timestamp field parsing and epoch ms/s conversion."""

    def test_timestamp_epoch_millis(self):
        ts_ms = 1768471200000  # 2026-01-15T10:00:00Z
        mq, _, _ = MongoFilterParser.parse({"user_id": "u1", "timestamp": ts_ms})
        expected = datetime(2026, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        assert mq["timestamp"] == expected

    def test_timestamp_epoch_seconds(self):
        ts_s = 1768471200  # 2026-01-15T10:00:00Z
        mq, _, _ = MongoFilterParser.parse({"user_id": "u1", "timestamp": ts_s})
        expected = datetime(2026, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        assert mq["timestamp"] == expected

    def test_timestamp_gte_lt_range(self):
        t_gte = 1768469400000  # 2026-01-15T09:30:00Z
        t_lt = 1768473000000  # 2026-01-15T10:30:00Z
        mq, _, _ = MongoFilterParser.parse(
            {
                "user_id": "u1",
                "AND": [{"timestamp": {"gte": t_gte}}, {"timestamp": {"lt": t_lt}}],
            }
        )
        and_clauses = mq["$and"]
        assert len(and_clauses) == 2
        assert "$gte" in and_clauses[0]["timestamp"]
        assert "$lt" in and_clauses[1]["timestamp"]

    def test_timestamp_iso_string(self):
        mq, _, _ = MongoFilterParser.parse(
            {"user_id": "u1", "timestamp": "2026-01-15T10:00:00+00:00"}
        )
        expected = datetime(2026, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        assert mq["timestamp"] == expected

    def test_timestamp_operators_dict(self):
        mq, _, _ = MongoFilterParser.parse(
            {"user_id": "u1", "timestamp": {"gte": 1768471200000, "lt": 1768474800000}}
        )
        ts = mq["timestamp"]
        assert "$gte" in ts
        assert "$lt" in ts
        assert isinstance(ts["$gte"], datetime)
        assert isinstance(ts["$lt"], datetime)


class TestCombinators:
    """Test AND / OR combinators."""

    def test_and_combinator(self):
        mq, _, _ = MongoFilterParser.parse(
            {"user_id": "u1", "AND": [{"session_id": "s1"}, {"group_id": "g1"}]}
        )
        assert "$and" in mq
        assert len(mq["$and"]) == 2
        assert {"session_id": "s1"} in mq["$and"]
        assert {"group_id": "g1"} in mq["$and"]

    def test_or_combinator(self):
        mq, _, _ = MongoFilterParser.parse(
            {"user_id": "u1", "OR": [{"session_id": "s1"}, {"session_id": "s2"}]}
        )
        assert "$or" in mq
        assert len(mq["$or"]) == 2

    def test_nested_and_or(self):
        mq, _, _ = MongoFilterParser.parse(
            {
                "user_id": "u1",
                "AND": [
                    {"OR": [{"session_id": "s1"}, {"session_id": "s2"}]},
                    {"group_id": "g1"},
                ],
            }
        )
        assert "$and" in mq
        assert len(mq["$and"]) == 2
        or_clause = mq["$and"][0]
        assert "$or" in or_clause

    def test_empty_and_items_ignored(self):
        mq, _, _ = MongoFilterParser.parse(
            {"user_id": "u1", "AND": [{}, {"session_id": "s1"}]}
        )
        # Empty dict {} is falsy for `if item` check, should be filtered
        assert "$and" in mq


class TestAllowlistSecurity:
    """Test that unknown fields are silently ignored."""

    def test_unknown_field_ignored(self):
        mq, uid, gids = MongoFilterParser.parse(
            {"user_id": "u1", "unknown_field": "should_be_ignored"}
        )
        assert "unknown_field" not in mq
        assert mq == {"user_id": "u1"}

    def test_multiple_unknown_fields(self):
        mq, _, _ = MongoFilterParser.parse(
            {"user_id": "u1", "password": "hack", "$where": "evil()", "admin": True}
        )
        assert mq == {"user_id": "u1"}
        assert "password" not in mq
        assert "$where" not in mq
        assert "admin" not in mq

    def test_unknown_field_in_and(self):
        mq, _, _ = MongoFilterParser.parse(
            {"user_id": "u1", "AND": [{"unknown": "val"}, {"session_id": "s1"}]}
        )
        and_items = mq["$and"]
        # First item should be empty dict (unknown filtered), second should have session_id
        assert {"session_id": "s1"} in and_items

    def test_unknown_operator_ignored(self):
        mq, _, _ = MongoFilterParser.parse(
            {"user_id": "u1", "session_id": {"regex": ".*hack.*"}}
        )
        # 'regex' is not in _OPERATOR_MAP, so session_id should not appear
        assert "session_id" not in mq


class TestScopeExtraction:
    """Test user_id and group_id scope extraction."""

    def test_no_scope(self):
        mq, uid, gids = MongoFilterParser.parse({"session_id": "s1"})
        assert uid is None
        assert gids is None

    def test_user_id_only(self):
        _, uid, gids = MongoFilterParser.parse({"user_id": "u1"})
        assert uid == "u1"
        assert gids is None

    def test_group_id_only(self):
        _, uid, gids = MongoFilterParser.parse({"group_id": "g1"})
        assert uid is None
        assert gids == ["g1"]

    def test_both_scopes(self):
        _, uid, gids = MongoFilterParser.parse({"user_id": "u1", "group_id": "g1"})
        assert uid == "u1"
        assert gids == ["g1"]

    def test_group_id_in_extraction(self):
        _, uid, gids = MongoFilterParser.parse({"group_id": {"in": ["g1", "g2", "g3"]}})
        assert gids == ["g1", "g2", "g3"]


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_filters(self):
        mq, uid, gids = MongoFilterParser.parse({})
        assert mq == {}
        assert uid is None
        assert gids is None

    def test_empty_and_array(self):
        mq, _, _ = MongoFilterParser.parse({"user_id": "u1", "AND": []})
        # Empty AND should not produce $and key
        assert "$and" not in mq

    def test_empty_or_array(self):
        mq, _, _ = MongoFilterParser.parse({"user_id": "u1", "OR": []})
        assert "$or" not in mq

    def test_all_fields_combined(self):
        mq, uid, gids = MongoFilterParser.parse(
            {
                "user_id": "u1",
                "group_id": {"in": ["g1", "g2"]},
                "session_id": "s1",
                "timestamp": {"gte": 1768471200000},
            }
        )
        assert uid == "u1"
        assert gids == ["g1", "g2"]
        assert mq["user_id"] == "u1"
        assert mq["group_id"] == {"$in": ["g1", "g2"]}
        assert mq["session_id"] == "s1"
        assert "$gte" in mq["timestamp"]

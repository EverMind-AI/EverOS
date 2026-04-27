"""
MongoDB Filters DSL Parser

Parses a filters object into a MongoDB query dict for use with Beanie find().

Supported top-level keys: user_id, group_id, session_id, timestamp
Supported operators: eq (implicit), in, gt, gte, lt, lte
Supported combinators: AND, OR
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


class MongoFilterParser:
    """Parse filters DSL into MongoDB query dict.

    Allowlist-based design: only fields in ALLOWED_FIELDS and
    TIMESTAMP_FIELDS are processed. Unknown fields are silently ignored.

    Usage:
        mongo_filter, user_id, group_ids = MongoFilterParser.parse(filters)
    """

    # --- Whitelist Configuration ---
    # Add new filterable fields here. No logic changes needed.
    ALLOWED_FIELDS = {"user_id", "group_id", "session_id"}
    TIMESTAMP_FIELDS = {"timestamp"}
    COMBINATOR_KEYS = {"AND", "OR"}

    # Mapping from DSL operators to MongoDB operators
    _OPERATOR_MAP = {
        "gt": "$gt",
        "gte": "$gte",
        "lt": "$lt",
        "lte": "$lte",
        "in": "$in",
    }

    @staticmethod
    def _parse_timestamp_value(value: Any) -> datetime:
        """Convert a timestamp value (epoch millis or seconds) to datetime."""
        if isinstance(value, (int, float)):
            # Heuristic: if > 1e12, treat as milliseconds
            if value > 1e12:
                return datetime.fromtimestamp(value / 1000, tz=timezone.utc)
            return datetime.fromtimestamp(value, tz=timezone.utc)
        if isinstance(value, str):
            return datetime.fromisoformat(value)
        return value

    @classmethod
    def _parse_field_condition(
        cls, field: str, condition: Any, mongo_query: Dict[str, Any]
    ) -> None:
        """Parse a single field condition and merge into mongo_query."""
        if field in cls.TIMESTAMP_FIELDS:
            if isinstance(condition, dict):
                ts_filter: Dict[str, Any] = {}
                for op, val in condition.items():
                    mongo_op = cls._OPERATOR_MAP.get(op)
                    if mongo_op:
                        ts_filter[mongo_op] = cls._parse_timestamp_value(val)
                if ts_filter:
                    mongo_query.setdefault(field, {}).update(ts_filter)
            else:
                mongo_query[field] = cls._parse_timestamp_value(condition)
        elif field in cls.ALLOWED_FIELDS:
            if isinstance(condition, dict):
                if "in" in condition:
                    mongo_query[field] = {"$in": condition["in"]}
                else:
                    field_filter: Dict[str, Any] = {}
                    for op, val in condition.items():
                        mongo_op = cls._OPERATOR_MAP.get(op)
                        if mongo_op:
                            field_filter[mongo_op] = val
                    if field_filter:
                        mongo_query[field] = field_filter
            else:
                mongo_query[field] = condition
        # else: unknown field silently ignored (security: allowlist only)

    @classmethod
    def _parse_single_filter(cls, filter_item: Dict[str, Any]) -> Dict[str, Any]:
        """Parse a single filter dict (used inside AND/OR arrays)."""
        result: Dict[str, Any] = {}
        for key, value in filter_item.items():
            if key in cls.COMBINATOR_KEYS:
                cls._parse_combinator(key, value, result)
            else:
                cls._parse_field_condition(key, value, result)
        return result

    @classmethod
    def _parse_combinator(
        cls, combinator: str, items: List[Dict[str, Any]], mongo_query: Dict[str, Any]
    ) -> None:
        """Parse AND/OR combinator arrays."""
        mongo_op = "$and" if combinator == "AND" else "$or"
        parsed_items = [cls._parse_single_filter(item) for item in items if item]
        if parsed_items:
            mongo_query.setdefault(mongo_op, []).extend(parsed_items)

    @classmethod
    def parse(
        cls, filters: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Optional[str], Optional[List[str]]]:
        """Parse filters object into MongoDB query.

        Returns:
            Tuple of (mongo_filter_dict, user_id, group_ids)
        """
        mongo_query: Dict[str, Any] = {}
        user_id: Optional[str] = None
        group_ids: Optional[List[str]] = None

        for key, value in filters.items():
            if key == "user_id":
                if isinstance(value, str):
                    user_id = value
                    mongo_query["user_id"] = value
                elif isinstance(value, dict) and "in" in value:
                    user_id = value["in"][0] if value["in"] else None
                    mongo_query["user_id"] = {"$in": value["in"]}

            elif key == "group_id":
                if isinstance(value, str):
                    group_ids = [value]
                    mongo_query["group_id"] = value
                elif isinstance(value, dict) and "in" in value:
                    group_ids = value["in"]
                    mongo_query["group_id"] = {"$in": value["in"]}

            elif key in cls.COMBINATOR_KEYS:
                cls._parse_combinator(key, value, mongo_query)

            else:
                cls._parse_field_condition(key, value, mongo_query)

        return mongo_query, user_id, group_ids


def parse_mongo_filters(
    filters: Dict[str, Any]
) -> Tuple[Dict[str, Any], Optional[str], Optional[List[str]]]:
    """Convenience function wrapping MongoFilterParser.parse()."""
    return MongoFilterParser.parse(filters)

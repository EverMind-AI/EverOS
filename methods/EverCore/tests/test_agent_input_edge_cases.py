"""Tests for agent input format edge cases and robustness.

Covers scenarios NOT tested in test_agent_post_content.py:
- Content array with non-dict items, unknown types, missing fields
- Tool call edge cases: empty tool_calls, missing function/arguments
- Timestamp edge cases: zero, negative, float, string
- Role edge cases: uppercase, empty, whitespace-only
- Sender ID conflicts and edge cases
- _normalize_content_items edge cases
- _unwrap_messages edge cases
- get_text_from_content_items edge cases

Usage:
    PYTHONPATH=src pytest tests/test_api/test_agent_input_edge_cases.py -v
"""

import pytest

from api_specs.request_converter import (
    convert_agent_add_to_memorize_request,
    _normalize_content_items,
)
from api_specs.memory_types import get_text_from_content_items


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE_TS = 1710835200000


def _agent_request(messages, user_id="user_01", session_id="sess_01"):
    return {"user_id": user_id, "session_id": session_id, "messages": messages}


def _msg(role, content, ts_offset=0, **kwargs):
    m = {"role": role, "timestamp": BASE_TS + ts_offset, "content": content}
    m.update(kwargs)
    return m


# ===========================================================================
# Content array edge cases
# ===========================================================================


class TestContentArrayEdgeCases:
    """Content items array with non-standard elements."""

    def test_non_dict_items_in_array_passed_through(self):
        """Non-dict items (int, string, None) in content array are passed through by normalize."""
        items = [123, "raw_string", None, {"type": "text", "text": "valid"}]
        result = _normalize_content_items(items)
        assert result == [123, "raw_string", None, {"type": "text", "text": "valid"}]

    def test_unknown_content_type_passed_through(self):
        """Content items with unknown types (e.g. 'video') are preserved as-is."""
        content = [
            {"type": "video", "url": "https://example.com/vid.mp4"},
            {"type": "text", "text": "See video above"},
        ]
        req = convert_agent_add_to_memorize_request(
            _agent_request([_msg("user", content)])
        )
        raw = req.new_raw_data_list[0]
        assert raw.content["content"][0]["type"] == "video"
        assert raw.content["content"][1]["type"] == "text"

    def test_text_item_missing_text_and_content_fields(self):
        """Text type item with neither 'text' nor 'content' field — passed through as-is."""
        content = [{"type": "text", "metadata": "something"}]
        result = _normalize_content_items(content)
        assert result == [{"type": "text", "metadata": "something"}]

    def test_item_without_type_field(self):
        """Dict item missing 'type' field is preserved unchanged."""
        content = [{"text": "orphan text", "extra": 1}]
        result = _normalize_content_items(content)
        assert result == [{"text": "orphan text", "extra": 1}]

    def test_empty_content_items_list_normalize(self):
        """Empty list passes through normalization unchanged."""
        assert _normalize_content_items([]) == []

    def test_content_as_dict_raises(self):
        """Content passed as a dict (not string or array) raises ValueError."""
        msg = {"role": "user", "timestamp": BASE_TS, "content": {"text": "hello"}}
        with pytest.raises(ValueError, match="content"):
            convert_agent_add_to_memorize_request(_agent_request([msg]))

    def test_content_as_integer_raises(self):
        """Content passed as an integer raises ValueError."""
        msg = {"role": "user", "timestamp": BASE_TS, "content": 42}
        with pytest.raises(ValueError, match="content"):
            convert_agent_add_to_memorize_request(_agent_request([msg]))

    def test_content_as_boolean_raises(self):
        """Content passed as a boolean raises ValueError."""
        msg = {"role": "user", "timestamp": BASE_TS, "content": True}
        with pytest.raises(ValueError, match="content"):
            convert_agent_add_to_memorize_request(_agent_request([msg]))


# ===========================================================================
# Tool call edge cases
# ===========================================================================


class TestToolCallEdgeCases:
    """Edge cases for tool_calls field in assistant messages."""

    def test_assistant_with_empty_tool_calls_array(self):
        """Assistant message with empty tool_calls=[] — treated as no tool calls."""
        req = convert_agent_add_to_memorize_request(
            _agent_request([_msg("assistant", "No tools needed.", tool_calls=[])])
        )
        raw = req.new_raw_data_list[0]
        # Empty list is falsy, so tool_calls may or may not be in content
        assert raw.content["role"] == "assistant"

    def test_assistant_with_none_tool_calls(self):
        """Assistant message with tool_calls=None — treated as no tool calls."""
        req = convert_agent_add_to_memorize_request(
            _agent_request([_msg("assistant", "Just a response.", tool_calls=None)])
        )
        raw = req.new_raw_data_list[0]
        assert raw.content["role"] == "assistant"

    def test_tool_call_missing_function_key(self):
        """Tool call dict without 'function' key — still stored in RawData."""
        calls = [{"id": "call_01", "type": "function"}]
        req = convert_agent_add_to_memorize_request(
            _agent_request([_msg("assistant", "Calling tool.", tool_calls=calls)])
        )
        raw = req.new_raw_data_list[0]
        assert raw.content["tool_calls"] == calls

    def test_tool_call_missing_arguments(self):
        """Tool call with function but no arguments — still accepted."""
        calls = [
            {"id": "call_02", "type": "function", "function": {"name": "list_files"}}
        ]
        req = convert_agent_add_to_memorize_request(
            _agent_request([_msg("assistant", "Listing files.", tool_calls=calls)])
        )
        raw = req.new_raw_data_list[0]
        assert raw.content["tool_calls"][0]["function"]["name"] == "list_files"

    def test_tool_call_id_empty_string_raises(self):
        """Tool role with empty string tool_call_id raises."""
        with pytest.raises(ValueError, match="tool_call_id"):
            convert_agent_add_to_memorize_request(
                _agent_request([_msg("tool", "result", tool_call_id="")])
            )


# ===========================================================================
# Timestamp edge cases
# ===========================================================================


class TestTimestampEdgeCases:
    """Edge cases for message timestamps."""

    def test_zero_timestamp_raises(self):
        """Timestamp of 0 is falsy — raises missing timestamp error."""
        msg = {"role": "user", "content": "hello", "timestamp": 0}
        with pytest.raises(ValueError, match="timestamp"):
            convert_agent_add_to_memorize_request(_agent_request([msg]))

    def test_float_timestamp_accepted(self):
        """Float timestamp (e.g., from JS Date.now()) should be accepted."""
        req = convert_agent_add_to_memorize_request(
            _agent_request([_msg("user", "hi", ts_offset=0.5)])
        )
        assert len(req.new_raw_data_list) == 1

    def test_negative_timestamp(self):
        """Negative timestamp (before epoch) — accepted or raises depends on implementation."""
        msg = {"role": "user", "content": "hello", "timestamp": -1000}
        # Negative timestamps are truthy, so validation passes;
        # _unix_ms_to_datetime may handle or error — verify no crash
        try:
            req = convert_agent_add_to_memorize_request(_agent_request([msg]))
            assert len(req.new_raw_data_list) == 1
        except (ValueError, OverflowError, OSError):
            pass  # acceptable to reject negative timestamps

    def test_very_large_timestamp(self):
        """Extremely large timestamp — verify no crash."""
        msg = {"role": "user", "content": "hello", "timestamp": 99999999999999}
        try:
            req = convert_agent_add_to_memorize_request(_agent_request([msg]))
            assert len(req.new_raw_data_list) == 1
        except (ValueError, OverflowError, OSError):
            pass  # acceptable to reject out-of-range timestamps

    def test_string_timestamp_raises(self):
        """String timestamp should raise or be rejected."""
        msg = {"role": "user", "content": "hello", "timestamp": "2024-03-19"}
        with pytest.raises((ValueError, TypeError)):
            convert_agent_add_to_memorize_request(_agent_request([msg]))


# ===========================================================================
# Role edge cases
# ===========================================================================


class TestRoleEdgeCases:
    """Edge cases for message role validation."""

    def test_uppercase_role_raises(self):
        """Role is case-sensitive — 'USER' should raise."""
        with pytest.raises(ValueError, match="role"):
            convert_agent_add_to_memorize_request(
                _agent_request([_msg("USER", "hello")])
            )

    def test_mixed_case_role_raises(self):
        """'Assistant' (capitalized) should raise."""
        with pytest.raises(ValueError, match="role"):
            convert_agent_add_to_memorize_request(
                _agent_request([_msg("Assistant", "hello")])
            )

    def test_empty_string_role_raises(self):
        """Empty string role raises missing role error."""
        msg = {"role": "", "content": "hello", "timestamp": BASE_TS}
        with pytest.raises(ValueError, match="role"):
            convert_agent_add_to_memorize_request(_agent_request([msg]))

    def test_whitespace_role_raises(self):
        """Whitespace-only role raises."""
        msg = {"role": "  ", "content": "hello", "timestamp": BASE_TS}
        with pytest.raises(ValueError, match="role"):
            convert_agent_add_to_memorize_request(_agent_request([msg]))

    def test_role_function_raises(self):
        """'function' role (OpenAI legacy) should raise."""
        with pytest.raises(ValueError, match="role"):
            convert_agent_add_to_memorize_request(
                _agent_request([_msg("function", "result")])
            )


# ===========================================================================
# Sender ID edge cases
# ===========================================================================


class TestSenderIdEdgeCases:
    """Edge cases for sender_id assignment and validation."""

    def test_user_role_with_mismatched_sender_id_raises(self):
        """User role with sender_id != user_id raises."""
        with pytest.raises(ValueError, match="sender_id"):
            convert_agent_add_to_memorize_request(
                _agent_request(
                    [_msg("user", "hello", sender_id="wrong_id")], user_id="user_01"
                )
            )

    def test_assistant_role_with_user_id_as_sender_id_raises(self):
        """Assistant role using user_id as sender_id raises conflict."""
        with pytest.raises(ValueError, match="sender_id"):
            convert_agent_add_to_memorize_request(
                _agent_request(
                    [_msg("assistant", "hello", sender_id="user_01")], user_id="user_01"
                )
            )

    def test_tool_auto_sender_id_is_deterministic(self):
        """Same user_id + tool_call_id always produces the same sender_id."""
        req1 = convert_agent_add_to_memorize_request(
            _agent_request([_msg("tool", "res", tool_call_id="call_x")])
        )
        req2 = convert_agent_add_to_memorize_request(
            _agent_request([_msg("tool", "res", tool_call_id="call_x")])
        )
        sid1 = req1.new_raw_data_list[0].content["sender_id"]
        sid2 = req2.new_raw_data_list[0].content["sender_id"]
        assert sid1 == sid2

    def test_different_tool_call_ids_get_different_sender_ids(self):
        """Different tool_call_ids produce different auto sender_ids."""
        req1 = convert_agent_add_to_memorize_request(
            _agent_request([_msg("tool", "res1", tool_call_id="call_a")])
        )
        req2 = convert_agent_add_to_memorize_request(
            _agent_request([_msg("tool", "res2", tool_call_id="call_b")])
        )
        sid1 = req1.new_raw_data_list[0].content["sender_id"]
        sid2 = req2.new_raw_data_list[0].content["sender_id"]
        assert sid1 != sid2

    def test_custom_sender_name_propagated(self):
        """Custom sender_name is stored in RawData."""
        req = convert_agent_add_to_memorize_request(
            _agent_request(
                [_msg("assistant", "hello", sender_id="bot_1", sender_name="MyBot")]
            )
        )
        raw = req.new_raw_data_list[0]
        assert raw.content.get("sender_name") == "MyBot"


# ===========================================================================
# Request-level edge cases
# ===========================================================================


class TestRequestEdgeCases:
    """Request-level input edge cases."""

    def test_no_messages_key_raises(self):
        """Request without messages key raises."""
        with pytest.raises(ValueError, match="messages"):
            convert_agent_add_to_memorize_request({"user_id": "u1"})

    def test_none_user_id_raises(self):
        """user_id=None raises."""
        with pytest.raises(ValueError, match="user_id"):
            convert_agent_add_to_memorize_request(
                {"user_id": None, "messages": [_msg("user", "hello")]}
            )

    def test_empty_string_user_id_raises(self):
        """user_id='' raises."""
        with pytest.raises(ValueError, match="user_id"):
            convert_agent_add_to_memorize_request(
                {"user_id": "", "messages": [_msg("user", "hello")]}
            )

    def test_multiple_messages_preserves_order(self):
        """Multiple messages maintain insertion order."""
        msgs = [
            _msg("user", "first", ts_offset=0),
            _msg("assistant", "second", ts_offset=1000),
            _msg("user", "third", ts_offset=2000),
        ]
        req = convert_agent_add_to_memorize_request(_agent_request(msgs))
        texts = []
        for rd in req.new_raw_data_list:
            for item in rd.content["content"]:
                if item.get("type") == "text":
                    texts.append(item["text"])
        assert texts == ["first", "second", "third"]

    def test_session_id_defaults_when_missing(self):
        """Missing session_id uses default value."""
        req = convert_agent_add_to_memorize_request(
            {"user_id": "u1", "messages": [_msg("user", "hello")]}
        )
        assert req.session_id is not None


# ===========================================================================
# get_text_from_content_items edge cases
# ===========================================================================


class TestGetTextFromContentItems:
    """Edge cases for get_text_from_content_items utility."""

    def test_plain_string_passthrough(self):
        assert get_text_from_content_items("hello world") == "hello world"

    def test_not_list_or_string_returns_empty(self):
        assert get_text_from_content_items(123) == ""
        assert get_text_from_content_items(None) == ""
        assert get_text_from_content_items({"type": "text"}) == ""

    def test_empty_list_returns_empty(self):
        assert get_text_from_content_items([]) == ""

    def test_non_dict_items_skipped(self):
        items = [123, "raw", None, {"type": "text", "text": "valid"}]
        assert get_text_from_content_items(items) == "valid"

    def test_non_text_type_included_with_tag(self):
        items = [
            {"type": "image", "uri": "https://example.com/img.png"},
            {"type": "text", "text": "caption"},
        ]
        assert get_text_from_content_items(items) == "[IMAGE] caption"

    def test_multiple_text_items_joined_with_space(self):
        items = [{"type": "text", "text": "line 1"}, {"type": "text", "text": "line 2"}]
        assert get_text_from_content_items(items) == "line 1 line 2"

    def test_legacy_content_field_fallback(self):
        items = [{"type": "text", "content": "legacy"}]
        assert get_text_from_content_items(items) == "legacy"

    def test_text_field_preferred_over_content(self):
        items = [{"type": "text", "text": "canonical", "content": "legacy"}]
        assert get_text_from_content_items(items) == "canonical"

    def test_empty_text_items_skipped(self):
        items = [{"type": "text", "text": ""}, {"type": "text", "text": "real"}]
        assert get_text_from_content_items(items) == "real"

    def test_item_missing_text_and_content_skipped(self):
        items = [{"type": "text", "metadata": "something"}]
        assert get_text_from_content_items(items) == ""

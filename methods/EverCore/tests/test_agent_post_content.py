"""Tests for agent POST endpoint content format handling.

Covers convert_agent_add_to_memorize_request with various content formats:
- Plain string content
- Content items array (text, image, multi-item)
- Legacy 'content' field in ContentItem dicts
- Tool call messages (assistant with tool_calls, tool role responses)
- Empty/missing content edge cases
- Mixed content formats in a single request

Usage:
    PYTHONPATH=src pytest tests/test_api/test_agent_post_content.py -v
"""

import pytest

from api_specs.request_converter import convert_agent_add_to_memorize_request
from api_specs.memory_types import RawDataType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE_TS = 1710835200000  # 2024-03-19 arbitrary base timestamp


def _agent_request(messages, user_id="user_agent_01", session_id="sess_01"):
    """Build a minimal agent add request dict."""
    return {"user_id": user_id, "session_id": session_id, "messages": messages}


def _msg(role, content, ts_offset=0, **kwargs):
    """Build a single agent message dict."""
    m = {"role": role, "timestamp": BASE_TS + ts_offset, "content": content}
    m.update(kwargs)
    return m


# ===========================================================================
# Plain string content
# ===========================================================================


class TestPlainStringContent:
    """Agent messages where content is a plain string."""

    def test_user_plain_string(self):
        req = convert_agent_add_to_memorize_request(
            _agent_request([_msg("user", "Hello, what can you do?")])
        )
        assert len(req.new_raw_data_list) == 1
        raw = req.new_raw_data_list[0]
        assert raw.content["content"] == [
            {"type": "text", "text": "Hello, what can you do?"}
        ]
        assert raw.content["role"] == "user"

    def test_assistant_plain_string(self):
        req = convert_agent_add_to_memorize_request(
            _agent_request([_msg("assistant", "I can help with many tasks.")])
        )
        raw = req.new_raw_data_list[0]
        assert raw.content["content"] == [
            {"type": "text", "text": "I can help with many tasks."}
        ]
        assert raw.content["role"] == "assistant"

    def test_tool_plain_string(self):
        req = convert_agent_add_to_memorize_request(
            _agent_request(
                [_msg("tool", "Weather: 22C sunny", tool_call_id="call_001")]
            )
        )
        raw = req.new_raw_data_list[0]
        assert raw.content["content"] == [
            {"type": "text", "text": "Weather: 22C sunny"}
        ]
        assert raw.content["role"] == "tool"
        assert raw.content["tool_call_id"] == "call_001"

    def test_multiline_string(self):
        text = "Line one\nLine two\nLine three"
        req = convert_agent_add_to_memorize_request(
            _agent_request([_msg("user", text)])
        )
        raw = req.new_raw_data_list[0]
        assert raw.content["content"] == [{"type": "text", "text": text}]


# ===========================================================================
# Content items array
# ===========================================================================


class TestContentItemsArray:
    """Agent messages where content is an array of content items."""

    def test_single_text_item(self):
        content = [{"type": "text", "text": "Hello agent"}]
        req = convert_agent_add_to_memorize_request(
            _agent_request([_msg("user", content)])
        )
        raw = req.new_raw_data_list[0]
        assert raw.content["content"] == [{"type": "text", "text": "Hello agent"}]

    def test_multiple_text_items(self):
        content = [
            {"type": "text", "text": "First paragraph"},
            {"type": "text", "text": "Second paragraph"},
        ]
        req = convert_agent_add_to_memorize_request(
            _agent_request([_msg("user", content)])
        )
        raw = req.new_raw_data_list[0]
        assert len(raw.content["content"]) == 2
        assert raw.content["content"][0]["text"] == "First paragraph"
        assert raw.content["content"][1]["text"] == "Second paragraph"

    def test_image_content_item(self):
        content = [
            {
                "type": "image",
                "uri": "https://example.com/photo.png",
                "name": "photo.png",
            }
        ]
        req = convert_agent_add_to_memorize_request(
            _agent_request([_msg("user", content)])
        )
        raw = req.new_raw_data_list[0]
        assert raw.content["content"][0]["type"] == "image"
        assert raw.content["content"][0]["uri"] == "https://example.com/photo.png"

    def test_mixed_text_and_image(self):
        content = [
            {"type": "text", "text": "Check this image"},
            {"type": "image", "uri": "https://example.com/img.jpg"},
        ]
        req = convert_agent_add_to_memorize_request(
            _agent_request([_msg("user", content)])
        )
        raw = req.new_raw_data_list[0]
        assert len(raw.content["content"]) == 2
        assert raw.content["content"][0]["type"] == "text"
        assert raw.content["content"][1]["type"] == "image"

    def test_content_item_with_extras(self):
        content = [
            {
                "type": "text",
                "text": "data",
                "source": "notion",
                "extras": {"page_id": "abc"},
            }
        ]
        req = convert_agent_add_to_memorize_request(
            _agent_request([_msg("user", content)])
        )
        raw = req.new_raw_data_list[0]
        item = raw.content["content"][0]
        assert item["text"] == "data"
        assert item["source"] == "notion"
        assert item["extras"] == {"page_id": "abc"}


# ===========================================================================
# Legacy 'content' field compatibility
# ===========================================================================


class TestLegacyContentField:
    """ContentItem dicts using legacy 'content' key instead of 'text'."""

    def test_legacy_content_field_normalized_to_text(self):
        content = [{"type": "text", "content": "legacy format"}]
        req = convert_agent_add_to_memorize_request(
            _agent_request([_msg("user", content)])
        )
        raw = req.new_raw_data_list[0]
        item = raw.content["content"][0]
        assert item["text"] == "legacy format"
        assert "content" not in item

    def test_mixed_legacy_and_canonical(self):
        content = [
            {"type": "text", "content": "from legacy"},
            {"type": "text", "text": "from canonical"},
        ]
        req = convert_agent_add_to_memorize_request(
            _agent_request([_msg("user", content)])
        )
        raw = req.new_raw_data_list[0]
        assert raw.content["content"][0]["text"] == "from legacy"
        assert raw.content["content"][1]["text"] == "from canonical"


# ===========================================================================
# Tool call messages
# ===========================================================================


class TestToolCallMessages:
    """Agent messages involving tool_calls and tool role responses."""

    def _weather_tool_call(self):
        return {
            "id": "call_weather_01",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city": "Tokyo"}'},
        }

    def test_assistant_with_tool_calls_and_text(self):
        req = convert_agent_add_to_memorize_request(
            _agent_request(
                [
                    _msg(
                        "assistant",
                        "Let me check the weather.",
                        tool_calls=[self._weather_tool_call()],
                    )
                ]
            )
        )
        raw = req.new_raw_data_list[0]
        assert raw.content["content"] == [
            {"type": "text", "text": "Let me check the weather."}
        ]
        assert raw.content["tool_calls"] == [self._weather_tool_call()]
        assert "tool_call_id" not in raw.content

    def test_assistant_with_tool_calls_content_array(self):
        req = convert_agent_add_to_memorize_request(
            _agent_request(
                [
                    _msg(
                        "assistant",
                        [{"type": "text", "text": "Checking weather..."}],
                        tool_calls=[self._weather_tool_call()],
                    )
                ]
            )
        )
        raw = req.new_raw_data_list[0]
        assert raw.content["content"] == [
            {"type": "text", "text": "Checking weather..."}
        ]
        assert raw.content["tool_calls"] == [self._weather_tool_call()]

    def test_tool_response_with_text_content(self):
        req = convert_agent_add_to_memorize_request(
            _agent_request(
                [
                    _msg(
                        "tool",
                        [{"type": "text", "text": "Tokyo: 22C, partly cloudy"}],
                        tool_call_id="call_weather_01",
                    )
                ]
            )
        )
        raw = req.new_raw_data_list[0]
        assert raw.content["tool_call_id"] == "call_weather_01"
        assert raw.content["content"] == [
            {"type": "text", "text": "Tokyo: 22C, partly cloudy"}
        ]

    def test_tool_response_empty_string_content_raises(self):
        """Tool messages must have non-empty content (execution result)."""
        with pytest.raises(ValueError, match="messages\\[\\]\\.content"):
            convert_agent_add_to_memorize_request(
                _agent_request([_msg("tool", "", tool_call_id="call_noop_01")])
            )

    def test_tool_response_empty_array_content_raises(self):
        """Tool messages with empty array content should raise."""
        with pytest.raises(ValueError, match="messages\\[\\]\\.content"):
            convert_agent_add_to_memorize_request(
                _agent_request([_msg("tool", [], tool_call_id="call_noop_02")])
            )

    def test_tool_response_none_content_raises(self):
        """Tool messages with None/missing content should raise."""
        msg = {"role": "tool", "timestamp": BASE_TS, "tool_call_id": "call_noop_03"}
        with pytest.raises(ValueError, match="messages\\[\\]\\.content"):
            convert_agent_add_to_memorize_request(_agent_request([msg]))

    def test_assistant_with_tool_calls_empty_string_content(self):
        """Assistant message with tool_calls and empty string content should succeed."""
        tool_call = self._weather_tool_call()
        req = convert_agent_add_to_memorize_request(
            _agent_request([_msg("assistant", "", tool_calls=[tool_call])])
        )
        raw = req.new_raw_data_list[0]
        assert raw.content["content"] == [{"type": "text", "text": ""}]
        assert raw.content["tool_calls"] == [tool_call]

    def test_assistant_with_tool_calls_none_content(self):
        """Assistant message with tool_calls and content=None should succeed."""
        tool_call = self._weather_tool_call()
        msg = {
            "role": "assistant",
            "timestamp": BASE_TS,
            "content": None,
            "tool_calls": [tool_call],
        }
        req = convert_agent_add_to_memorize_request(_agent_request([msg]))
        raw = req.new_raw_data_list[0]
        assert raw.content["content"] == [{"type": "text", "text": ""}]
        assert raw.content["tool_calls"] == [tool_call]

    def test_assistant_with_tool_calls_missing_content(self):
        """Assistant message with tool_calls and no content key should succeed."""
        tool_call = self._weather_tool_call()
        msg = {"role": "assistant", "timestamp": BASE_TS, "tool_calls": [tool_call]}
        req = convert_agent_add_to_memorize_request(_agent_request([msg]))
        raw = req.new_raw_data_list[0]
        assert raw.content["content"] == [{"type": "text", "text": ""}]
        assert raw.content["tool_calls"] == [tool_call]

    def test_assistant_with_tool_calls_empty_array_content(self):
        """Assistant message with tool_calls and content=[] should succeed."""
        tool_call = self._weather_tool_call()
        req = convert_agent_add_to_memorize_request(
            _agent_request([_msg("assistant", [], tool_calls=[tool_call])])
        )
        raw = req.new_raw_data_list[0]
        assert raw.content["content"] == [{"type": "text", "text": ""}]
        assert raw.content["tool_calls"] == [tool_call]

    def test_assistant_without_tool_calls_empty_content_still_raises(self):
        """Assistant without tool_calls and empty content should still raise."""
        with pytest.raises(ValueError, match="content"):
            convert_agent_add_to_memorize_request(
                _agent_request([_msg("assistant", "")])
            )

    def test_assistant_without_tool_calls_none_content_still_raises(self):
        """Assistant without tool_calls and None content should still raise."""
        msg = {"role": "assistant", "timestamp": BASE_TS, "content": None}
        with pytest.raises(ValueError, match="content"):
            convert_agent_add_to_memorize_request(_agent_request([msg]))

    def test_multiple_tool_calls_in_one_assistant_message(self):
        calls = [
            {
                "id": "call_01",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "Tokyo"}'},
            },
            {
                "id": "call_02",
                "type": "function",
                "function": {
                    "name": "get_time",
                    "arguments": '{"timezone": "Asia/Tokyo"}',
                },
            },
        ]
        req = convert_agent_add_to_memorize_request(
            _agent_request([_msg("assistant", "Let me check both.", tool_calls=calls)])
        )
        raw = req.new_raw_data_list[0]
        assert len(raw.content["tool_calls"]) == 2
        assert raw.content["tool_calls"][0]["id"] == "call_01"
        assert raw.content["tool_calls"][1]["id"] == "call_02"


# ===========================================================================
# Full conversation trajectory
# ===========================================================================


class TestFullTrajectory:
    """Full agent conversation with user -> assistant -> tool -> assistant flow."""

    def test_complete_tool_use_conversation(self):
        messages = [
            _msg("user", "What is the weather in Tokyo?", ts_offset=0),
            _msg(
                "assistant",
                "Let me check.",
                ts_offset=1000,
                tool_calls=[
                    {
                        "id": "call_w1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Tokyo"}',
                        },
                    }
                ],
            ),
            _msg(
                "tool",
                [{"type": "text", "text": "Tokyo: 18C, partly cloudy"}],
                ts_offset=2000,
                tool_call_id="call_w1",
            ),
            _msg(
                "assistant",
                "The weather in Tokyo is 18C and partly cloudy.",
                ts_offset=3000,
            ),
        ]
        req = convert_agent_add_to_memorize_request(_agent_request(messages))

        assert len(req.new_raw_data_list) == 4
        assert req.raw_data_type == RawDataType.AGENTCONVERSATION

        # User message
        assert req.new_raw_data_list[0].content["role"] == "user"
        # Assistant with tool call
        assert req.new_raw_data_list[1].content["role"] == "assistant"
        assert "tool_calls" in req.new_raw_data_list[1].content
        # Tool response
        assert req.new_raw_data_list[2].content["role"] == "tool"
        assert req.new_raw_data_list[2].content["tool_call_id"] == "call_w1"
        # Final assistant
        assert req.new_raw_data_list[3].content["role"] == "assistant"
        assert "tool_calls" not in req.new_raw_data_list[3].content

    def test_mixed_content_formats_in_trajectory(self):
        """Different messages use different content formats in the same request."""
        messages = [
            _msg("user", "Analyze this image", ts_offset=0),  # plain string
            _msg(
                "user",
                [
                    {"type": "text", "text": "Here is the image:"},
                    {"type": "image", "uri": "https://example.com/chart.png"},
                ],
                ts_offset=1000,
            ),  # array with mixed types
            _msg(
                "assistant",
                [{"type": "text", "content": "I see a bar chart"}],
                ts_offset=2000,
            ),  # legacy content field
        ]
        req = convert_agent_add_to_memorize_request(_agent_request(messages))
        assert len(req.new_raw_data_list) == 3

        # Plain string was coerced
        assert req.new_raw_data_list[0].content["content"] == [
            {"type": "text", "text": "Analyze this image"}
        ]
        # Array preserved
        assert len(req.new_raw_data_list[1].content["content"]) == 2
        # Legacy field normalized
        assert (
            req.new_raw_data_list[2].content["content"][0]["text"]
            == "I see a bar chart"
        )


# ===========================================================================
# Validation / error cases
# ===========================================================================


class TestValidationErrors:
    """Error cases for invalid agent post content."""

    def test_empty_string_content_non_tool_raises(self):
        with pytest.raises(ValueError, match="content"):
            convert_agent_add_to_memorize_request(_agent_request([_msg("user", "")]))

    def test_empty_array_content_non_tool_raises(self):
        with pytest.raises(ValueError, match="content"):
            convert_agent_add_to_memorize_request(
                _agent_request([_msg("assistant", [])])
            )

    def test_none_content_non_tool_raises(self):
        msg = {"role": "user", "timestamp": BASE_TS}
        with pytest.raises(ValueError, match="content"):
            convert_agent_add_to_memorize_request(_agent_request([msg]))

    def test_missing_role_raises(self):
        msg = {"timestamp": BASE_TS, "content": "hello"}
        with pytest.raises(ValueError, match="role"):
            convert_agent_add_to_memorize_request(_agent_request([msg]))

    def test_invalid_role_raises(self):
        with pytest.raises(ValueError, match="role"):
            convert_agent_add_to_memorize_request(
                _agent_request([_msg("system", "hello")])
            )

    def test_tool_without_tool_call_id_raises(self):
        with pytest.raises(ValueError, match="tool_call_id"):
            convert_agent_add_to_memorize_request(
                _agent_request([_msg("tool", "response")])
            )

    def test_missing_timestamp_raises(self):
        msg = {"role": "user", "content": "hello"}
        with pytest.raises(ValueError, match="timestamp"):
            convert_agent_add_to_memorize_request(_agent_request([msg]))

    def test_missing_user_id_raises(self):
        with pytest.raises(ValueError, match="user_id"):
            convert_agent_add_to_memorize_request({"messages": [_msg("user", "hello")]})

    def test_empty_messages_raises(self):
        with pytest.raises(ValueError, match="messages"):
            convert_agent_add_to_memorize_request(
                {"user_id": "user_01", "messages": []}
            )


# ===========================================================================
# Sender ID and metadata
# ===========================================================================


class TestSenderAndMetadata:
    """Verify sender_id assignment and metadata for different roles."""

    def test_user_sender_id_equals_user_id(self):
        req = convert_agent_add_to_memorize_request(
            _agent_request([_msg("user", "hi")], user_id="u_123")
        )
        raw = req.new_raw_data_list[0]
        assert raw.content["sender_id"] == "u_123"

    def test_assistant_gets_auto_sender_id(self):
        req = convert_agent_add_to_memorize_request(
            _agent_request([_msg("assistant", "hello")])
        )
        raw = req.new_raw_data_list[0]
        assert raw.content["sender_id"].endswith("_assistant")

    def test_tool_gets_auto_sender_id(self):
        req = convert_agent_add_to_memorize_request(
            _agent_request([_msg("tool", "result", tool_call_id="call_01")])
        )
        raw = req.new_raw_data_list[0]
        assert raw.content["sender_id"].endswith("_tool")

    def test_custom_sender_id_for_assistant(self):
        req = convert_agent_add_to_memorize_request(
            _agent_request([_msg("assistant", "hello", sender_id="my_bot")])
        )
        raw = req.new_raw_data_list[0]
        assert raw.content["sender_id"] == "my_bot"

    def test_session_id_propagated(self):
        req = convert_agent_add_to_memorize_request(
            _agent_request([_msg("user", "hi")], session_id="custom_session")
        )
        assert req.session_id == "custom_session"

    def test_raw_data_type_is_agent_conversation(self):
        req = convert_agent_add_to_memorize_request(
            _agent_request([_msg("user", "hi")])
        )
        assert req.raw_data_type == RawDataType.AGENTCONVERSATION

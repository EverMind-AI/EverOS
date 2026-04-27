"""
AgentCaseExtractor Unit Tests

Tests for:
- _should_skip: pre-filter heuristics
- _truncate_text: token-based head/tail truncation
- _heuristic_trim_tool_outputs: bulk tool output trimming
- _strip_before_first_user: system prompt removal
- _has_tool_calls / _count_tool_call_rounds: tool call detection

Usage:
    PYTHONPATH=src pytest tests/test_agent_case_extractor.py -v
"""

import copy
import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock, AsyncMock
from typing import Any, Dict, List

from memory_layer.memory_extractor.agent_case_extractor import (
    AgentCaseExtractor,
    HIGH_MESSAGE_COUNT_THRESHOLD,
    MAX_TOOL_OUTPUT_TOKENS,
    MAX_TOOL_ARGS_TOKENS,
    MAX_ASSISTANT_RESPONSE_TOKENS,
)
from api_specs.memory_types import RawDataType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _user(content: str) -> Dict[str, Any]:
    return {"role": "user", "content": content}


def _assistant(content: str) -> Dict[str, Any]:
    return {"role": "assistant", "content": content}


def _tool_call(
    content: str = "",
    arguments: str = "{}",
    name: str = "search",
) -> Dict[str, Any]:
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [
            {
                "id": "call_1",
                "function": {"name": name, "arguments": arguments},
            }
        ],
    }


def _tool_response(content: str = "tool result") -> Dict[str, Any]:
    return {"role": "tool", "content": content, "tool_call_id": "call_1"}


def _long_text(token_count: int) -> str:
    """Generate a text that is approximately token_count tokens long."""
    # Each "word_N " is roughly 2 tokens; overshoot then let truncation handle it.
    return " ".join(f"word_{i}" for i in range(token_count))


# ---------------------------------------------------------------------------
# Mock tokenizer — use real tiktoken so token counts are accurate
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _mock_tokenizer_factory():
    """Patch get_bean_by_type to return a real tiktoken tokenizer factory."""
    import tiktoken

    encoding = tiktoken.get_encoding("o200k_base")
    mock_factory = MagicMock()
    mock_factory.get_tokenizer_from_tiktoken.return_value = encoding

    with patch(
        "memory_layer.memory_extractor.agent_case_extractor.get_bean_by_type",
        return_value=mock_factory,
    ):
        yield


# ===========================================================================
# _should_skip tests
# ===========================================================================


class TestShouldSkip:
    """Tests for AgentCaseExtractor._should_skip."""

    def test_no_user_messages(self):
        msgs = [_assistant("hello")]
        assert AgentCaseExtractor._should_skip(msgs) is not None

    def test_no_assistant_messages(self):
        msgs = [_user("hello")]
        assert AgentCaseExtractor._should_skip(msgs) is not None

    def test_incomplete_trajectory_last_is_tool_call(self):
        msgs = [_user("do something"), _tool_call("thinking")]
        assert "Incomplete" in AgentCaseExtractor._should_skip(msgs)

    def test_incomplete_trajectory_last_is_tool_response(self):
        msgs = [_user("do something"), _tool_call(), _tool_response()]
        assert AgentCaseExtractor._should_skip(msgs) is not None

    def test_single_turn_no_tools(self):
        msgs = [_user("hi"), _assistant("hello")]
        assert AgentCaseExtractor._should_skip(msgs) is not None

    def test_multi_turn_no_tools_passes(self):
        # Must have > FILTER_NO_TOOL_MAX_MESSAGES (4) messages
        # AND sufficient assistant token count to pass heuristic
        long_response = (
            "The TypeError on line 5 is caused by passing a string where an integer "
            "is expected. You need to wrap the input with int() before passing it to "
            "the calculate function. Here is the corrected code: change "
            "calculate(user_input) to calculate(int(user_input)). This happens because "
            "input() always returns a string in Python 3, even when the user types a "
            "number. The int() call converts it properly. You should also add error "
            "handling with a try/except ValueError block around the int() conversion."
        )
        msgs = [
            _user("help me debug"),
            _assistant(long_response),
            _user("TypeError on line 5"),
            _assistant(long_response),
            _user("thanks"),
            _assistant("glad it worked"),
        ]
        assert AgentCaseExtractor._should_skip(msgs) is None

    def test_single_turn_with_tools_passes(self):
        msgs = [
            _user("search for python docs"),
            _tool_call("let me search"),
            _tool_response("found docs"),
            _assistant("here are the results"),
        ]
        assert AgentCaseExtractor._should_skip(msgs) is None

    def test_empty_messages(self):
        assert AgentCaseExtractor._should_skip([]) is not None


# ===========================================================================
# _truncate_text tests
# ===========================================================================


class TestTruncateText:
    """Tests for AgentCaseExtractor._truncate_text."""

    def test_short_text_unchanged(self):
        text = "hello world"
        result = AgentCaseExtractor._truncate_text(text, max_tokens=100)
        assert result == text

    def test_empty_text(self):
        assert AgentCaseExtractor._truncate_text("", max_tokens=10) == ""

    def test_none_text(self):
        assert AgentCaseExtractor._truncate_text(None, max_tokens=10) is None

    def test_non_string_returns_as_is(self):
        assert AgentCaseExtractor._truncate_text(123, max_tokens=10) == 123

    def test_long_text_truncated(self):
        text = _long_text(500)
        result = AgentCaseExtractor._truncate_text(text, max_tokens=50)
        assert "[... trimmed" in result
        assert "tokens ...]" in result

    def test_truncated_has_head_and_tail(self):
        text = _long_text(500)
        result = AgentCaseExtractor._truncate_text(
            text, max_tokens=100, head_ratio=0.7
        )
        assert result.startswith("word_0")
        assert "tokens ...]" in result
        # Tail should contain later words
        parts = result.split("[... trimmed")
        assert len(parts) == 2

    def test_head_ratio_respected(self):
        text = _long_text(500)
        import tiktoken

        enc = tiktoken.get_encoding("o200k_base")

        result_70 = AgentCaseExtractor._truncate_text(
            text, max_tokens=100, head_ratio=0.7
        )
        result_50 = AgentCaseExtractor._truncate_text(
            text, max_tokens=100, head_ratio=0.5
        )
        # With 0.7 ratio, head portion should be longer
        head_70 = result_70.split("\n[... trimmed")[0]
        head_50 = result_50.split("\n[... trimmed")[0]
        assert len(enc.encode(head_70)) > len(enc.encode(head_50))

    def test_exact_limit_unchanged(self):
        import tiktoken

        enc = tiktoken.get_encoding("o200k_base")
        # Build text that is exactly 50 tokens
        words = []
        for i in range(100):
            words.append(f"w{i}")
            if len(enc.encode(" ".join(words))) >= 50:
                break
        text = " ".join(words)
        token_count = len(enc.encode(text))
        result = AgentCaseExtractor._truncate_text(text, max_tokens=token_count)
        assert result == text


# ===========================================================================
# _heuristic_trim_tool_outputs tests
# ===========================================================================


class TestHeuristicTrimToolOutputs:
    """Tests for AgentCaseExtractor._heuristic_trim_tool_outputs."""

    def test_short_content_unchanged(self):
        msgs = [
            _user("hi"),
            _tool_call("thinking", arguments='{"query": "test"}'),
            _tool_response("short result"),
            _assistant("done"),
        ]
        result = AgentCaseExtractor._heuristic_trim_tool_outputs(msgs, 1500, 800)
        assert result[2]["content"] == "short result"
        assert result[1]["tool_calls"][0]["function"]["arguments"] == '{"query": "test"}'

    def test_long_tool_output_trimmed(self):
        long_content = _long_text(3000)
        msgs = [
            _user("search"),
            _tool_call(),
            _tool_response(long_content),
            _assistant("done"),
        ]
        result = AgentCaseExtractor._heuristic_trim_tool_outputs(msgs, 100, 800)
        assert "[... trimmed" in result[2]["content"]
        assert len(result[2]["content"]) < len(long_content)

    def test_long_tool_args_trimmed(self):
        long_args = _long_text(2000)
        msgs = [
            _user("do it"),
            _tool_call(arguments=long_args),
            _tool_response("ok"),
            _assistant("done"),
        ]
        result = AgentCaseExtractor._heuristic_trim_tool_outputs(msgs, 1500, 100)
        trimmed_args = result[1]["tool_calls"][0]["function"]["arguments"]
        assert "[... trimmed" in trimmed_args
        assert len(trimmed_args) < len(long_args)

    def test_does_not_mutate_original(self):
        long_content = _long_text(3000)
        msgs = [_tool_response(long_content)]
        original_content = msgs[0]["content"]
        AgentCaseExtractor._heuristic_trim_tool_outputs(msgs, 100, 800)
        assert msgs[0]["content"] == original_content

    def test_user_content_untouched(self):
        long_text = _long_text(3000)
        msgs = [_user(long_text), _assistant("short")]
        result = AgentCaseExtractor._heuristic_trim_tool_outputs(msgs, 100, 100)
        assert result[0]["content"] == long_text

    def test_long_assistant_response_trimmed(self):
        long_text = _long_text(5000)
        msgs = [_user("hi"), _assistant(long_text)]
        result = AgentCaseExtractor._heuristic_trim_tool_outputs(
            msgs, 100, 100, max_assistant_response_tokens=100
        )
        assert "[... trimmed" in result[1]["content"]
        assert len(result[1]["content"]) < len(long_text)

    def test_short_assistant_response_untouched(self):
        msgs = [_user("hi"), _assistant("short reply")]
        result = AgentCaseExtractor._heuristic_trim_tool_outputs(msgs, 100, 100)
        assert result[1]["content"] == "short reply"

    def test_multiple_tool_calls_all_trimmed(self):
        long_content = _long_text(3000)
        msgs = [
            _user("do many things"),
            _tool_call(),
            _tool_response(long_content),
            _tool_call(arguments='{"x": 1}'),
            _tool_response(long_content),
            _assistant("all done"),
        ]
        result = AgentCaseExtractor._heuristic_trim_tool_outputs(msgs, 100, 800)
        assert "[... trimmed" in result[2]["content"]
        assert "[... trimmed" in result[4]["content"]

    def test_missing_function_key_skipped(self):
        msgs = [
            _user("test"),
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call_1"}],  # no "function" key
            },
            _assistant("done"),
        ]
        # Should not raise
        result = AgentCaseExtractor._heuristic_trim_tool_outputs(msgs, 1500, 800)
        assert len(result) == 3

    def test_non_string_tool_content_untouched(self):
        msgs = [{"role": "tool", "content": {"key": "value"}, "tool_call_id": "c1"}]
        result = AgentCaseExtractor._heuristic_trim_tool_outputs(msgs, 100, 800)
        assert result[0]["content"] == {"key": "value"}

    def test_empty_messages_list(self):
        result = AgentCaseExtractor._heuristic_trim_tool_outputs([], 1500, 800)
        assert result == []


# ===========================================================================
# _strip_before_first_user tests
# ===========================================================================


class TestStripBeforeFirstUser:
    """Tests for AgentCaseExtractor._strip_before_first_user."""

    def test_removes_system_prompt(self):
        msgs = [
            {"role": "system", "content": "You are helpful"},
            _user("hello"),
            _assistant("hi"),
        ]
        result = AgentCaseExtractor._strip_before_first_user(msgs)
        assert len(result) == 2
        assert result[0]["role"] == "user"

    def test_no_user_returns_empty(self):
        msgs = [{"role": "system", "content": "setup"}, _assistant("hi")]
        assert AgentCaseExtractor._strip_before_first_user(msgs) == []

    def test_already_starts_with_user(self):
        msgs = [_user("hi"), _assistant("hello")]
        result = AgentCaseExtractor._strip_before_first_user(msgs)
        assert len(result) == 2


# ===========================================================================
# _has_tool_calls / _count_tool_call_rounds tests
# ===========================================================================


class TestToolCallHelpers:
    """Tests for tool call detection helpers."""

    def test_has_tool_calls_true(self):
        msgs = [_user("x"), _tool_call(), _tool_response(), _assistant("done")]
        assert AgentCaseExtractor._has_tool_calls(msgs) is True

    def test_has_tool_calls_false(self):
        msgs = [_user("hi"), _assistant("hello")]
        assert AgentCaseExtractor._has_tool_calls(msgs) is False

    def test_count_tool_call_rounds(self):
        msgs = [
            _user("x"),
            _tool_call(),
            _tool_response(),
            _tool_call(),
            _tool_response(),
            _assistant("done"),
        ]
        assert AgentCaseExtractor._count_tool_call_rounds(msgs) == 2

    def test_count_zero_rounds(self):
        msgs = [_user("hi"), _assistant("hello")]
        assert AgentCaseExtractor._count_tool_call_rounds(msgs) == 0

    def test_has_tool_calls_tool_response_only(self):
        """A tool response without tool_calls in assistant still counts."""
        msgs = [_user("x"), _tool_response("result"), _assistant("done")]
        assert AgentCaseExtractor._has_tool_calls(msgs) is True

    def test_has_tool_calls_empty(self):
        assert AgentCaseExtractor._has_tool_calls([]) is False


# ===========================================================================
# _calc_tool_content_size tests
# ===========================================================================


class TestCalcToolContentSize:
    """Tests for AgentCaseExtractor._calc_tool_content_size."""

    def test_tool_message_counts_content(self):
        msg = _tool_response("some tool output text here")
        size = AgentCaseExtractor._calc_tool_content_size(msg)
        assert size > 0

    def test_assistant_with_tool_calls_counts_arguments(self):
        msg = _tool_call(arguments='{"query": "test search", "limit": 10}')
        size = AgentCaseExtractor._calc_tool_content_size(msg)
        assert size > 0

    def test_user_message_returns_zero(self):
        msg = _user("some text")
        assert AgentCaseExtractor._calc_tool_content_size(msg) == 0

    def test_plain_assistant_returns_zero(self):
        msg = _assistant("some text")
        assert AgentCaseExtractor._calc_tool_content_size(msg) == 0

    def test_tool_empty_content(self):
        msg = {"role": "tool", "content": "", "tool_call_id": "c1"}
        assert AgentCaseExtractor._calc_tool_content_size(msg) == 0

    def test_assistant_empty_arguments(self):
        msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "function": {"name": "f", "arguments": ""}}],
        }
        assert AgentCaseExtractor._calc_tool_content_size(msg) == 0

    def test_assistant_multiple_tool_calls_sums(self):
        msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "c1", "function": {"name": "f1", "arguments": '{"a": 1}'}},
                {"id": "c2", "function": {"name": "f2", "arguments": '{"b": 2}'}},
            ],
        }
        size = AgentCaseExtractor._calc_tool_content_size(msg)
        single_size = AgentCaseExtractor._calc_tool_content_size(
            _tool_call(arguments='{"a": 1}')
        )
        assert size > single_size  # two calls should be larger than one


# ===========================================================================
# _json_default tests
# ===========================================================================


class TestJsonDefault:
    """Tests for AgentCaseExtractor._json_default."""

    def test_datetime_to_isoformat(self):
        from datetime import datetime

        dt = datetime(2025, 3, 1, 12, 0, 0)
        result = AgentCaseExtractor._json_default(dt)
        assert result == "2025-03-01T12:00:00"

    def test_non_serializable_to_str(self):
        result = AgentCaseExtractor._json_default(set([1, 2, 3]))
        assert isinstance(result, str)

    def test_bytes_to_str(self):
        result = AgentCaseExtractor._json_default(b"hello")
        assert result == "b'hello'"


# ===========================================================================
# _clamp_quality_score tests
# ===========================================================================


class TestClampQualityScore:
    """Tests for AgentCaseExtractor._clamp_quality_score."""

    def test_valid_float_in_range(self):
        assert AgentCaseExtractor._clamp_quality_score(0.5) == 0.5

    def test_clamps_above_one(self):
        assert AgentCaseExtractor._clamp_quality_score(1.5) == 1.0

    def test_clamps_below_zero(self):
        assert AgentCaseExtractor._clamp_quality_score(-0.3) == 0.0

    def test_boundary_zero(self):
        assert AgentCaseExtractor._clamp_quality_score(0.0) == 0.0

    def test_boundary_one(self):
        assert AgentCaseExtractor._clamp_quality_score(1.0) == 1.0

    def test_none_returns_none(self):
        assert AgentCaseExtractor._clamp_quality_score(None) is None

    def test_string_number(self):
        assert AgentCaseExtractor._clamp_quality_score("0.7") == 0.7

    def test_invalid_string_returns_none(self):
        assert AgentCaseExtractor._clamp_quality_score("abc") is None

    def test_int_value(self):
        assert AgentCaseExtractor._clamp_quality_score(1) == 1.0


# ===========================================================================
# _unwrap_messages tests
# ===========================================================================


class TestUnwrapMessages:
    """Tests for AgentCaseExtractor._unwrap_messages."""

    def test_wrapped_format(self):
        data = [
            {"message": {"role": "user", "content": "hello"}, "parse_info": {}},
            {"message": {"role": "assistant", "content": "hi"}, "parse_info": {}},
        ]
        result = AgentCaseExtractor._unwrap_messages(data)
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "hello"

    def test_unwrapped_format_passthrough(self):
        data = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        result = AgentCaseExtractor._unwrap_messages(data)
        assert len(result) == 2
        assert result[0]["content"] == "hello"

    def test_content_list_normalized_to_string(self):
        data = [
            {
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": "hello world"}],
                },
            },
        ]
        result = AgentCaseExtractor._unwrap_messages(data)
        assert isinstance(result[0]["content"], str)
        assert "hello" in result[0]["content"]

    def test_non_dict_items_skipped(self):
        data = [
            {"message": {"role": "user", "content": "ok"}},
            "not a dict",
            42,
            None,
        ]
        result = AgentCaseExtractor._unwrap_messages(data)
        assert len(result) == 1

    def test_empty_list(self):
        assert AgentCaseExtractor._unwrap_messages([]) == []

    def test_assistant_with_tool_calls_preserved(self):
        data = [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": "c1", "function": {"name": "f"}}],
                },
            },
        ]
        result = AgentCaseExtractor._unwrap_messages(data)
        assert result[0]["tool_calls"] is not None
        assert result[0]["content"] is None


# ===========================================================================
# _collect_tool_call_groups tests
# ===========================================================================


class TestCollectToolCallGroups:
    """Tests for AgentCaseExtractor._collect_tool_call_groups."""

    def _build_extractor(self):
        from unittest.mock import MagicMock
        mock_llm = MagicMock()
        return AgentCaseExtractor(
            llm_provider=mock_llm,
            tool_pre_compress_prompt="",
            filter_prompt="",
            experience_compress_prompt="",
        )

    def test_single_group(self):
        ext = self._build_extractor()
        items = [
            _user("hi"),
            _tool_call(),
            _tool_response("r1"),
            _assistant("done"),
        ]
        groups = ext._collect_tool_call_groups(items)
        assert len(groups) == 1
        assert groups[0] == [1, 2]

    def test_multiple_groups(self):
        ext = self._build_extractor()
        items = [
            _user("hi"),
            _tool_call(),
            _tool_response("r1"),
            _tool_call(),
            _tool_response("r2"),
            _tool_response("r3"),
            _assistant("done"),
        ]
        groups = ext._collect_tool_call_groups(items)
        assert len(groups) == 2
        assert groups[0] == [1, 2]
        assert groups[1] == [3, 4, 5]

    def test_no_tool_calls(self):
        ext = self._build_extractor()
        items = [_user("hi"), _assistant("hello")]
        groups = ext._collect_tool_call_groups(items)
        assert groups == []

    def test_empty_items(self):
        ext = self._build_extractor()
        assert ext._collect_tool_call_groups([]) == []

    def test_tool_call_without_response(self):
        ext = self._build_extractor()
        items = [_user("hi"), _tool_call(), _assistant("done")]
        groups = ext._collect_tool_call_groups(items)
        assert len(groups) == 1
        assert groups[0] == [1]  # just the assistant with tool_calls, no tool response


# ===========================================================================
# _calc_group_size tests
# ===========================================================================


class TestCalcGroupSize:
    """Tests for AgentCaseExtractor._calc_group_size."""

    def _build_extractor(self):
        from unittest.mock import MagicMock
        mock_llm = MagicMock()
        return AgentCaseExtractor(
            llm_provider=mock_llm,
            tool_pre_compress_prompt="",
            filter_prompt="",
            experience_compress_prompt="",
        )

    def test_group_size_sum(self):
        ext = self._build_extractor()
        items = [
            _tool_call(arguments='{"query": "test search"}'),
            _tool_response("this is the tool response content"),
        ]
        group = [0, 1]
        size = ext._calc_group_size(items, group)
        # Both items contribute tokens
        assert size > 0
        # Should equal sum of individual sizes
        expected = sum(AgentCaseExtractor._calc_tool_content_size(items[i]) for i in group)
        assert size == expected

    def test_empty_group(self):
        ext = self._build_extractor()
        items = [_tool_call(), _tool_response()]
        assert ext._calc_group_size(items, []) == 0


# ===========================================================================
# _filter_conversation tests
# ===========================================================================


class TestFilterConversation:
    """Tests for AgentCaseExtractor._filter_conversation (async)."""

    def _build_extractor(self, llm_response: str):
        from unittest.mock import MagicMock, AsyncMock
        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(return_value=llm_response)
        return AgentCaseExtractor(
            llm_provider=mock_llm,
            tool_pre_compress_prompt="",
            filter_prompt="{messages}",
            experience_compress_prompt="",
        )

    @pytest.mark.asyncio
    async def test_worth_extracting_true(self):
        ext = self._build_extractor('{"worth_extracting": true}')
        assert await ext._filter_conversation("[]") is True

    @pytest.mark.asyncio
    async def test_worth_extracting_false(self):
        ext = self._build_extractor('{"worth_extracting": false, "reason": "trivial"}')
        assert await ext._filter_conversation("[]") is False

    @pytest.mark.asyncio
    async def test_llm_error_defaults_to_true(self):
        from unittest.mock import MagicMock, AsyncMock
        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(side_effect=Exception("LLM down"))
        ext = AgentCaseExtractor(
            llm_provider=mock_llm,
            tool_pre_compress_prompt="",
            filter_prompt="{messages}",
            experience_compress_prompt="",
        )
        assert await ext._filter_conversation("[]") is True

    @pytest.mark.asyncio
    async def test_invalid_json_defaults_to_true(self):
        ext = self._build_extractor("not json")
        assert await ext._filter_conversation("[]") is True


# ===========================================================================
# _compress_experience tests
# ===========================================================================


class TestCompressExperience:
    """Tests for AgentCaseExtractor._compress_experience (async)."""

    def _build_extractor(self, llm_response: str):
        from unittest.mock import MagicMock, AsyncMock
        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(return_value=llm_response)
        return AgentCaseExtractor(
            llm_provider=mock_llm,
            tool_pre_compress_prompt="",
            filter_prompt="",
            experience_compress_prompt="{messages}",
        )

    @pytest.mark.asyncio
    async def test_valid_extraction(self):
        resp = '{"task_intent": "Build API", "approach": "Design then implement", "quality_score": 0.8}'
        ext = self._build_extractor(resp)
        result = await ext._compress_experience("[]")
        assert result["task_intent"] == "Build API"
        assert result["approach"] == "Design then implement"

    @pytest.mark.asyncio
    async def test_empty_task_intent_returns_none(self):
        resp = '{"task_intent": "", "approach": "some approach"}'
        ext = self._build_extractor(resp)
        assert await ext._compress_experience("[]") is None

    @pytest.mark.asyncio
    async def test_empty_approach_returns_none(self):
        resp = '{"task_intent": "Do X", "approach": ""}'
        ext = self._build_extractor(resp)
        assert await ext._compress_experience("[]") is None

    @pytest.mark.asyncio
    async def test_missing_task_intent_retries_then_none(self):
        resp = '{"approach": "some steps"}'
        ext = self._build_extractor(resp)
        result = await ext._compress_experience("[]")
        assert result is None
        # Should have been called twice (retry)
        assert ext.llm_provider.generate.call_count == 2

    @pytest.mark.asyncio
    async def test_with_key_insight(self):
        resp = '{"task_intent": "Deploy", "approach": "CI/CD pipeline", "key_insight": "Blue-green deploy", "quality_score": 0.9}'
        ext = self._build_extractor(resp)
        result = await ext._compress_experience("[]")
        assert result["key_insight"] == "Blue-green deploy"

    @pytest.mark.asyncio
    async def test_llm_exception_retries(self):
        from unittest.mock import MagicMock, AsyncMock
        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(side_effect=Exception("timeout"))
        ext = AgentCaseExtractor(
            llm_provider=mock_llm,
            tool_pre_compress_prompt="",
            filter_prompt="",
            experience_compress_prompt="{messages}",
        )
        result = await ext._compress_experience("[]")
        assert result is None
        assert mock_llm.generate.call_count == 2


# ===========================================================================
# _compress_tool_chunk tests
# ===========================================================================


class TestCompressToolChunk:
    """Tests for AgentCaseExtractor._compress_tool_chunk (async)."""

    def _build_extractor(self, llm_response: str):
        from unittest.mock import MagicMock, AsyncMock
        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(return_value=llm_response)
        return AgentCaseExtractor(
            llm_provider=mock_llm,
            tool_pre_compress_prompt="{messages_json}{new_count}",
            filter_prompt="",
            experience_compress_prompt="",
        )

    @pytest.mark.asyncio
    async def test_valid_compression(self):
        msgs = [_tool_response("long output"), _tool_response("another output")]
        resp = '{"compressed_messages": [{"role": "tool", "content": "short1"}, {"role": "tool", "content": "short2"}]}'
        ext = self._build_extractor(resp)
        result = await ext._compress_tool_chunk(msgs)
        assert len(result) == 2
        assert result[0]["content"] == "short1"

    @pytest.mark.asyncio
    async def test_wrong_count_returns_none(self):
        msgs = [_tool_response("output")]
        resp = '{"compressed_messages": [{"role": "tool", "content": "a"}, {"role": "tool", "content": "b"}]}'
        ext = self._build_extractor(resp)
        result = await ext._compress_tool_chunk(msgs)
        assert result is None

    @pytest.mark.asyncio
    async def test_missing_key_returns_none(self):
        msgs = [_tool_response("output")]
        resp = '{"result": "bad format"}'
        ext = self._build_extractor(resp)
        result = await ext._compress_tool_chunk(msgs)
        assert result is None

    @pytest.mark.asyncio
    async def test_llm_exception_retries_then_none(self):
        from unittest.mock import MagicMock, AsyncMock
        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(side_effect=Exception("fail"))
        ext = AgentCaseExtractor(
            llm_provider=mock_llm,
            tool_pre_compress_prompt="{messages_json}{new_count}",
            filter_prompt="",
            experience_compress_prompt="",
        )
        result = await ext._compress_tool_chunk([_tool_response("x")])
        assert result is None
        assert mock_llm.generate.call_count == 2


# ===========================================================================
# _pre_compress_to_list tests
# ===========================================================================


class TestPreCompressToList:
    """Tests for AgentCaseExtractor._pre_compress_to_list (async)."""

    @pytest.mark.asyncio
    async def test_no_tool_calls_returns_unchanged(self):
        from unittest.mock import MagicMock, AsyncMock
        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock()
        ext = AgentCaseExtractor(
            llm_provider=mock_llm,
            tool_pre_compress_prompt="",
            filter_prompt="",
            experience_compress_prompt="",
        )
        items = [_user("hi"), _assistant("hello")]
        result = await ext._pre_compress_to_list(items)
        assert len(result) == 2
        mock_llm.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_small_tool_content_no_compression(self):
        from unittest.mock import MagicMock, AsyncMock
        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock()
        ext = AgentCaseExtractor(
            llm_provider=mock_llm,
            tool_pre_compress_prompt="",
            filter_prompt="",
            experience_compress_prompt="",
            pre_compress_chunk_size=999999,  # very large threshold
        )
        items = [_user("hi"), _tool_call(), _tool_response("short"), _assistant("done")]
        result = await ext._pre_compress_to_list(items)
        assert len(result) == 4
        mock_llm.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_does_not_mutate_original(self):
        from unittest.mock import MagicMock, AsyncMock
        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock()
        ext = AgentCaseExtractor(
            llm_provider=mock_llm,
            tool_pre_compress_prompt="",
            filter_prompt="",
            experience_compress_prompt="",
        )
        items = [_user("hi"), _tool_call(), _tool_response("data"), _assistant("ok")]
        original = copy.deepcopy(items)
        await ext._pre_compress_to_list(items)
        assert items == original


# ===========================================================================
# extract_memory tests
# ===========================================================================


class TestExtractMemory:
    """Tests for AgentCaseExtractor.extract_memory (async)."""

    def _build_extractor(self, filter_resp="", compress_resp=""):
        from unittest.mock import MagicMock, AsyncMock
        mock_llm = MagicMock()
        responses = []
        if filter_resp:
            responses.append(filter_resp)
        if compress_resp:
            responses.append(compress_resp)
        if responses:
            mock_llm.generate = AsyncMock(side_effect=responses)
        else:
            mock_llm.generate = AsyncMock(return_value='{}')
        return AgentCaseExtractor(
            llm_provider=mock_llm,
            tool_pre_compress_prompt="{messages_json}{new_count}",
            filter_prompt="{messages}",
            experience_compress_prompt="{messages}",
        )

    def _make_request(self, messages=None):
        from types import SimpleNamespace
        memcell = SimpleNamespace(
            type=RawDataType.AGENTCONVERSATION,
            timestamp=datetime(2025, 3, 1),
            original_data=messages or [],
            participants=["user1"],
            sender_ids=["user1"],
            event_id="evt_test_001",
        )
        return SimpleNamespace(
            memcell=memcell,
            user_id="user_001",
            group_id="group_001",
        )

    @pytest.mark.asyncio
    async def test_none_memcell_returns_none(self):
        ext = self._build_extractor()
        from types import SimpleNamespace
        req = SimpleNamespace(memcell=None, user_id="u", group_id="g")
        assert await ext.extract_memory(req) is None

    @pytest.mark.asyncio
    async def test_wrong_type_returns_none(self):
        ext = self._build_extractor()
        from types import SimpleNamespace
        memcell = SimpleNamespace(type="OTHER_TYPE", original_data=[], timestamp=None, participants=[], sender_ids=[])
        req = SimpleNamespace(memcell=memcell, user_id="u", group_id="g")
        assert await ext.extract_memory(req) is None

    @pytest.mark.asyncio
    async def test_empty_conversation_returns_none(self):
        ext = self._build_extractor()
        req = self._make_request(messages=[])
        assert await ext.extract_memory(req) is None

    @pytest.mark.asyncio
    @patch("memory_layer.memory_extractor.agent_case_extractor.get_vectorize_service")
    async def test_successful_extraction(self, mock_vs):
        import numpy as np
        mock_service = MagicMock()
        mock_service.get_embedding = AsyncMock(return_value=np.array([0.1, 0.2, 0.3]))
        mock_service.get_model_name.return_value = "test-model"
        mock_vs.return_value = mock_service

        compress_resp = '{"task_intent": "Build API", "approach": "Step 1, Step 2", "quality_score": 0.85, "key_insight": "Use caching"}'
        ext = self._build_extractor(compress_resp=compress_resp)
        # Multi-round tool conversation skips filter
        messages = [
            {"message": _user("build an API")},
            {"message": _tool_call("searching", arguments='{"q":"api"}')},
            {"message": _tool_response("found frameworks")},
            {"message": _tool_call("creating", arguments='{"name":"api"}')},
            {"message": _tool_response("created project")},
            {"message": _assistant("Done! I built the API for you.")},
        ]
        req = self._make_request(messages=messages)
        result = await ext.extract_memory(req)
        assert result is not None
        assert result.task_intent == "Build API"
        assert result.approach == "Step 1, Step 2"
        assert result.key_insight == "Use caching"
        assert result.quality_score == 0.85
        assert result.vector is not None

# Adaptive trim limit scaling tests
# ===========================================================================


class TestAdaptiveTrimScaling:
    """Tests for scale-based trim limit calculation when tokens exceed 2x threshold.

    Replicates the scaling logic in extract_memory to verify the math
    and floor behaviour without going through the full async pipeline.
    """

    def _make_extractor(self, chunk_size: int = 1000) -> AgentCaseExtractor:
        e = object.__new__(AgentCaseExtractor)
        e.pre_compress_chunk_size = chunk_size
        e.max_tool_output_tokens = MAX_TOOL_OUTPUT_TOKENS
        e.max_tool_args_tokens = MAX_TOOL_ARGS_TOKENS
        e.max_assistant_response_tokens = MAX_ASSISTANT_RESPONSE_TOKENS
        return e

    def _compute_limits(self, extractor, total_tokens):
        """Mirror the scaling logic from extract_memory."""
        scale_trigger = extractor.pre_compress_chunk_size // 2
        if total_tokens > scale_trigger:
            scale = scale_trigger / total_tokens
            return (
                max(100, int(extractor.max_tool_output_tokens * scale)),
                max(100, int(extractor.max_tool_args_tokens * scale)),
                max(200, int(extractor.max_assistant_response_tokens * scale)),
            )
        return (
            extractor.max_tool_output_tokens,
            extractor.max_tool_args_tokens,
            extractor.max_assistant_response_tokens,
        )

    def test_normal_limits_when_under_trigger(self):
        e = self._make_extractor(chunk_size=1000)
        # scale_trigger = 500; total=400 is NOT over
        out, args, asst = self._compute_limits(e, 400)
        assert out == MAX_TOOL_OUTPUT_TOKENS
        assert args == MAX_TOOL_ARGS_TOKENS
        assert asst == MAX_ASSISTANT_RESPONSE_TOKENS

    def test_normal_limits_when_exactly_at_trigger(self):
        e = self._make_extractor(chunk_size=1000)
        # scale_trigger = 500; total=500 is NOT over (strict >)
        out, args, asst = self._compute_limits(e, 500)
        assert out == MAX_TOOL_OUTPUT_TOKENS
        assert args == MAX_TOOL_ARGS_TOKENS
        assert asst == MAX_ASSISTANT_RESPONSE_TOKENS

    def test_limits_reduced_when_over_trigger(self):
        e = self._make_extractor(chunk_size=1000)
        # scale_trigger = 500; total=501 is over
        out, args, asst = self._compute_limits(e, 501)
        assert out < MAX_TOOL_OUTPUT_TOKENS
        assert args < MAX_TOOL_ARGS_TOKENS
        assert asst < MAX_ASSISTANT_RESPONSE_TOKENS

    def test_limits_at_1x_chunk_size(self):
        """At 1x chunk_size (=2x trigger), scale=0.5, limits should be halved."""
        e = self._make_extractor(chunk_size=1000)
        # scale_trigger=500; total=1000 -> scale=0.5
        out, args, asst = self._compute_limits(e, 1000)
        assert out == max(100, MAX_TOOL_OUTPUT_TOKENS // 2)
        assert args == max(100, MAX_TOOL_ARGS_TOKENS // 2)
        assert asst == max(200, MAX_ASSISTANT_RESPONSE_TOKENS // 2)

    def test_limits_decrease_as_tokens_increase(self):
        e = self._make_extractor(chunk_size=1000)
        out_3x, _, _ = self._compute_limits(e, 3000)
        out_6x, _, _ = self._compute_limits(e, 6000)
        assert out_3x > out_6x

    def test_tool_output_floor_at_100(self):
        """Tool output limit must never drop below 100."""
        e = self._make_extractor(chunk_size=1000)
        # scale_trigger=500; scale=500/200000=0.0025; int(1000*0.0025)=2 < 100 -> floor
        out, _, _ = self._compute_limits(e, 200000)
        assert out == 100

    def test_tool_args_floor_at_100(self):
        e = self._make_extractor(chunk_size=1000)
        _, args, _ = self._compute_limits(e, 200000)
        assert args == 100

    def test_assistant_floor_at_200(self):
        e = self._make_extractor(chunk_size=1000)
        _, _, asst = self._compute_limits(e, 200000)
        assert asst == 200

    # -- floor NOT applied (moderate scale, computed value stays above floor) --

    def test_tool_output_floor_not_applied_at_1x_chunk(self):
        """At 1x chunk_size scale=0.5; int(1000*0.5)=500 > 100, floor should not kick in."""
        e = self._make_extractor(chunk_size=1000)
        # scale_trigger=500; total=1000 -> scale=0.5
        out, _, _ = self._compute_limits(e, 1000)
        assert out == int(MAX_TOOL_OUTPUT_TOKENS * (500 / 1000))
        assert out > 100

    def test_tool_args_floor_not_applied_at_1x_chunk(self):
        e = self._make_extractor(chunk_size=1000)
        _, args, _ = self._compute_limits(e, 1000)
        assert args == int(MAX_TOOL_ARGS_TOKENS * (500 / 1000))
        assert args > 100

    def test_assistant_floor_not_applied_at_1x_chunk(self):
        e = self._make_extractor(chunk_size=1000)
        _, _, asst = self._compute_limits(e, 1000)
        assert asst == int(MAX_ASSISTANT_RESPONSE_TOKENS * (500 / 1000))
        assert asst > 200

    # -- each limit hits its own floor independently --

    def test_only_tool_args_hits_floor(self):
        """Craft a scale where tool_args hits floor but tool_output does not."""
        e = object.__new__(AgentCaseExtractor)
        e.pre_compress_chunk_size = 1000
        e.max_tool_output_tokens = 10000  # 10000 * 0.025 = 250 >> 100
        e.max_tool_args_tokens = 50       # 50 * 0.025 = 1 < 100 -> floor
        e.max_assistant_response_tokens = MAX_ASSISTANT_RESPONSE_TOKENS

        # scale_trigger=500; total=20000 -> scale=500/20000=0.025
        out, args, _ = self._compute_limits(e, 20000)
        assert out == int(10000 * (500 / 20000))  # 250, no floor
        assert args == 100                         # floor applied

    def test_only_assistant_hits_floor(self):
        """Craft a scale where assistant hits floor but tool limits do not."""
        e = object.__new__(AgentCaseExtractor)
        e.pre_compress_chunk_size = 1000
        e.max_tool_output_tokens = 10000   # 10000 * 0.0125 = 125 >> 100
        e.max_tool_args_tokens = 10000     # same
        e.max_assistant_response_tokens = 100  # 100 * 0.0125 = 1 < 200 -> floor

        # scale_trigger=500; total=40000 -> scale=500/40000=0.0125
        out, args, asst = self._compute_limits(e, 40000)
        assert out > 100    # no floor
        assert args > 100   # no floor
        assert asst == 200  # floor applied

    # -- boundary: one token over scale_trigger --

    def test_one_token_over_trigger_uses_scaled_limits(self):
        """total_tokens = scale_trigger + 1 should trigger scaled path."""
        e = self._make_extractor(chunk_size=1000)
        trigger = 500  # pre_compress_chunk_size // 2
        total = trigger + 1
        out, args, asst = self._compute_limits(e, total)
        scale = trigger / total
        assert out == max(100, int(MAX_TOOL_OUTPUT_TOKENS * scale))
        assert args == max(100, int(MAX_TOOL_ARGS_TOKENS * scale))
        assert asst == max(200, int(MAX_ASSISTANT_RESPONSE_TOKENS * scale))


# ===========================================================================
# Post-trim token guard in extract_memory
# ===========================================================================


def _make_memcell_request():
    """Return a minimal (memcell, request) pair suitable for extract_memory."""
    memcell = MagicMock()
    memcell.type = RawDataType.AGENTCONVERSATION
    memcell.original_data = []
    memcell.timestamp = None
    memcell.participants = []
    memcell.sender_ids = []

    request = MagicMock()
    request.memcell = memcell
    request.user_id = "u1"
    request.group_id = "g1"
    return memcell, request


_STUB_MSGS = [
    {"role": "user", "content": "hi"},
    {"role": "assistant", "tool_calls": [{"id": "c1", "function": {"name": "f", "arguments": "{}"}}]},
    {"role": "tool", "content": "result", "tool_call_id": "c1"},
    {"role": "assistant", "content": "done"},
]


class TestExtractMemoryPostTrimSkip:
    """Tests for the post-trim token guard in extract_memory."""

    @pytest.mark.asyncio
    async def test_skips_when_trimmed_tokens_still_over_threshold(self):
        """extract_memory returns None if tokens are still > scale_threshold after trim."""
        extractor = object.__new__(AgentCaseExtractor)
        extractor.pre_compress_chunk_size = 100  # threshold = 100
        extractor.max_tool_output_tokens = MAX_TOOL_OUTPUT_TOKENS
        extractor.max_tool_args_tokens = MAX_TOOL_ARGS_TOKENS
        extractor.max_assistant_response_tokens = MAX_ASSISTANT_RESPONSE_TOKENS
        extractor.memory_type = MagicMock()
        extractor.filter_prompt = "{messages}"
        extractor.experience_compress_prompt = "{messages}"
        extractor.tool_pre_compress_prompt = "{messages_json}{new_count}"
        extractor.llm_provider = MagicMock()

        _, request = _make_memcell_request()

        # scale_trigger=50; first call 250 > 50 -> scaled trim
        # skip threshold=200; second call 210 > 200 -> should skip
        token_returns = iter([250, 210])

        with (
            patch.object(AgentCaseExtractor, "_unwrap_messages", return_value=_STUB_MSGS),
            patch.object(AgentCaseExtractor, "_strip_before_first_user", return_value=_STUB_MSGS),
            patch.object(AgentCaseExtractor, "_should_skip", return_value=None),
            patch.object(AgentCaseExtractor, "_count_tokens", side_effect=lambda _: next(token_returns)),
            patch.object(AgentCaseExtractor, "_heuristic_trim_tool_outputs", return_value=_STUB_MSGS),
        ):
            result = await extractor.extract_memory(request)

        assert result is None

    @pytest.mark.asyncio
    async def test_proceeds_when_trimmed_tokens_under_threshold(self):
        """extract_memory calls _pre_compress_to_list when post-trim tokens are within limit."""
        extractor = object.__new__(AgentCaseExtractor)
        extractor.pre_compress_chunk_size = 100  # threshold = 100
        extractor.max_tool_output_tokens = MAX_TOOL_OUTPUT_TOKENS
        extractor.max_tool_args_tokens = MAX_TOOL_ARGS_TOKENS
        extractor.max_assistant_response_tokens = MAX_ASSISTANT_RESPONSE_TOKENS
        extractor.memory_type = MagicMock()
        extractor.filter_prompt = "{messages}"
        extractor.experience_compress_prompt = "{messages}"
        extractor.tool_pre_compress_prompt = "{messages_json}{new_count}"
        extractor.llm_provider = MagicMock()

        _, request = _make_memcell_request()

        # scale_trigger=50; first call 250 > 50 -> scaled trim
        # skip threshold=200; second call 150 <= 200 -> proceed
        token_returns = iter([250, 150])

        mock_pre_compress = AsyncMock(return_value=_STUB_MSGS)

        with (
            patch.object(AgentCaseExtractor, "_unwrap_messages", return_value=_STUB_MSGS),
            patch.object(AgentCaseExtractor, "_strip_before_first_user", return_value=_STUB_MSGS),
            patch.object(AgentCaseExtractor, "_should_skip", return_value=None),
            patch.object(AgentCaseExtractor, "_count_tokens", side_effect=lambda _: next(token_returns)),
            patch.object(AgentCaseExtractor, "_heuristic_trim_tool_outputs", return_value=_STUB_MSGS),
            patch.object(AgentCaseExtractor, "_pre_compress_to_list", mock_pre_compress),
            patch.object(AgentCaseExtractor, "_filter_conversation", AsyncMock(return_value=False)),
        ):
            await extractor.extract_memory(request)

        mock_pre_compress.assert_called_once()

    @pytest.mark.asyncio
    async def test_scaled_limits_passed_to_heuristic_trim(self):
        """When total_tokens > scale_trigger, heuristic_trim receives scaled-down limits."""
        chunk_size = 1000  # _STUB_MSGS has 4 msgs < HIGH_MESSAGE_COUNT_THRESHOLD, so scale_trigger = 1000
        total = 4000  # scale = 1000/4000 = 0.25

        extractor = object.__new__(AgentCaseExtractor)
        extractor.pre_compress_chunk_size = chunk_size
        extractor.max_tool_output_tokens = MAX_TOOL_OUTPUT_TOKENS
        extractor.max_tool_args_tokens = MAX_TOOL_ARGS_TOKENS
        extractor.max_assistant_response_tokens = MAX_ASSISTANT_RESPONSE_TOKENS
        extractor.memory_type = MagicMock()
        extractor.filter_prompt = "{messages}"
        extractor.experience_compress_prompt = "{messages}"
        extractor.tool_pre_compress_prompt = "{messages_json}{new_count}"
        extractor.llm_provider = MagicMock()

        _, request = _make_memcell_request()

        # After trim, return under skip threshold (chunk_size*2=2000) so we don't skip
        token_returns = iter([total, chunk_size])
        mock_trim = MagicMock(return_value=_STUB_MSGS)

        with (
            patch.object(AgentCaseExtractor, "_unwrap_messages", return_value=_STUB_MSGS),
            patch.object(AgentCaseExtractor, "_strip_before_first_user", return_value=_STUB_MSGS),
            patch.object(AgentCaseExtractor, "_should_skip", return_value=None),
            patch.object(AgentCaseExtractor, "_count_tokens", side_effect=lambda _: next(token_returns)),
            patch.object(AgentCaseExtractor, "_heuristic_trim_tool_outputs", mock_trim),
            patch.object(AgentCaseExtractor, "_pre_compress_to_list", AsyncMock(return_value=_STUB_MSGS)),
            patch.object(AgentCaseExtractor, "_filter_conversation", AsyncMock(return_value=False)),
        ):
            await extractor.extract_memory(request)

        scale = chunk_size / total  # scale_trigger/total = 1000/4000 = 0.25
        expected_tool_out = max(200, int(MAX_TOOL_OUTPUT_TOKENS * scale))
        expected_tool_args = max(200, int(MAX_TOOL_ARGS_TOKENS * scale))
        expected_assistant = max(500, int(MAX_ASSISTANT_RESPONSE_TOKENS * scale))

        mock_trim.assert_called_once_with(
            _STUB_MSGS,
            expected_tool_out,
            expected_tool_args,
            expected_assistant,
        )

    @pytest.mark.asyncio
    async def test_normal_limits_passed_to_trim_when_under_threshold(self):
        """When total_tokens <= pre_compress_chunk_size, full (unscaled) limits reach _heuristic_trim."""
        chunk_size = 1000  # threshold = 1000

        extractor = object.__new__(AgentCaseExtractor)
        extractor.pre_compress_chunk_size = chunk_size
        extractor.max_tool_output_tokens = MAX_TOOL_OUTPUT_TOKENS
        extractor.max_tool_args_tokens = MAX_TOOL_ARGS_TOKENS
        extractor.max_assistant_response_tokens = MAX_ASSISTANT_RESPONSE_TOKENS
        extractor.memory_type = MagicMock()
        extractor.filter_prompt = "{messages}"
        extractor.experience_compress_prompt = "{messages}"
        extractor.tool_pre_compress_prompt = "{messages_json}{new_count}"
        extractor.llm_provider = MagicMock()

        _, request = _make_memcell_request()

        # scale_trigger=500; both counts 400 <= 500 -> normal limits
        token_returns = iter([400, 400])
        mock_trim = MagicMock(return_value=_STUB_MSGS)

        with (
            patch.object(AgentCaseExtractor, "_unwrap_messages", return_value=_STUB_MSGS),
            patch.object(AgentCaseExtractor, "_strip_before_first_user", return_value=_STUB_MSGS),
            patch.object(AgentCaseExtractor, "_should_skip", return_value=None),
            patch.object(AgentCaseExtractor, "_count_tokens", side_effect=lambda _: next(token_returns)),
            patch.object(AgentCaseExtractor, "_heuristic_trim_tool_outputs", mock_trim),
            patch.object(AgentCaseExtractor, "_pre_compress_to_list", AsyncMock(return_value=_STUB_MSGS)),
            patch.object(AgentCaseExtractor, "_filter_conversation", AsyncMock(return_value=False)),
        ):
            await extractor.extract_memory(request)

        mock_trim.assert_called_once_with(
            _STUB_MSGS,
            MAX_TOOL_OUTPUT_TOKENS,
            MAX_TOOL_ARGS_TOKENS,
            MAX_ASSISTANT_RESPONSE_TOKENS,
        )

    @pytest.mark.asyncio
    async def test_post_trim_exactly_at_threshold_proceeds(self):
        """trimmed_tokens == scale_threshold should NOT skip (condition is strictly >)."""
        extractor = object.__new__(AgentCaseExtractor)
        extractor.pre_compress_chunk_size = 100  # threshold = 100
        extractor.max_tool_output_tokens = MAX_TOOL_OUTPUT_TOKENS
        extractor.max_tool_args_tokens = MAX_TOOL_ARGS_TOKENS
        extractor.max_assistant_response_tokens = MAX_ASSISTANT_RESPONSE_TOKENS
        extractor.memory_type = MagicMock()
        extractor.filter_prompt = "{messages}"
        extractor.experience_compress_prompt = "{messages}"
        extractor.tool_pre_compress_prompt = "{messages_json}{new_count}"
        extractor.llm_provider = MagicMock()

        _, request = _make_memcell_request()

        # scale_trigger=50; first count 300 > 50 -> scaled trim
        # skip threshold=200; second count exactly 200 -> NOT over (strict >), should proceed
        token_returns = iter([300, 200])
        mock_pre_compress = AsyncMock(return_value=_STUB_MSGS)

        with (
            patch.object(AgentCaseExtractor, "_unwrap_messages", return_value=_STUB_MSGS),
            patch.object(AgentCaseExtractor, "_strip_before_first_user", return_value=_STUB_MSGS),
            patch.object(AgentCaseExtractor, "_should_skip", return_value=None),
            patch.object(AgentCaseExtractor, "_count_tokens", side_effect=lambda _: next(token_returns)),
            patch.object(AgentCaseExtractor, "_heuristic_trim_tool_outputs", return_value=_STUB_MSGS),
            patch.object(AgentCaseExtractor, "_pre_compress_to_list", mock_pre_compress),
            patch.object(AgentCaseExtractor, "_filter_conversation", AsyncMock(return_value=False)),
        ):
            await extractor.extract_memory(request)

        mock_pre_compress.assert_called_once()

    @pytest.mark.asyncio
    async def test_floor_limits_passed_to_trim_at_extreme_token_count(self):
        """At very high token counts all three limits should hit their floors."""
        chunk_size = 100  # threshold = 100
        total = 1_000_000  # massively over -> scale ≈ 0

        extractor = object.__new__(AgentCaseExtractor)
        extractor.pre_compress_chunk_size = chunk_size
        extractor.max_tool_output_tokens = MAX_TOOL_OUTPUT_TOKENS
        extractor.max_tool_args_tokens = MAX_TOOL_ARGS_TOKENS
        extractor.max_assistant_response_tokens = MAX_ASSISTANT_RESPONSE_TOKENS
        extractor.memory_type = MagicMock()
        extractor.filter_prompt = "{messages}"
        extractor.experience_compress_prompt = "{messages}"
        extractor.tool_pre_compress_prompt = "{messages_json}{new_count}"
        extractor.llm_provider = MagicMock()

        _, request = _make_memcell_request()

        # After trim return under skip threshold (chunk_size*2=200) so we don't skip
        token_returns = iter([total, chunk_size])
        mock_trim = MagicMock(return_value=_STUB_MSGS)

        with (
            patch.object(AgentCaseExtractor, "_unwrap_messages", return_value=_STUB_MSGS),
            patch.object(AgentCaseExtractor, "_strip_before_first_user", return_value=_STUB_MSGS),
            patch.object(AgentCaseExtractor, "_should_skip", return_value=None),
            patch.object(AgentCaseExtractor, "_count_tokens", side_effect=lambda _: next(token_returns)),
            patch.object(AgentCaseExtractor, "_heuristic_trim_tool_outputs", mock_trim),
            patch.object(AgentCaseExtractor, "_pre_compress_to_list", AsyncMock(return_value=_STUB_MSGS)),
            patch.object(AgentCaseExtractor, "_filter_conversation", AsyncMock(return_value=False)),
        ):
            await extractor.extract_memory(request)

        mock_trim.assert_called_once_with(_STUB_MSGS, 200, 200, 500)

    @pytest.mark.asyncio
    async def test_count_tokens_called_once_when_no_scaling_needed(self):
        """When total_tokens <= scale_trigger, the second _count_tokens call is skipped."""
        extractor = object.__new__(AgentCaseExtractor)
        extractor.pre_compress_chunk_size = 1000  # scale_trigger = 500
        extractor.max_tool_output_tokens = MAX_TOOL_OUTPUT_TOKENS
        extractor.max_tool_args_tokens = MAX_TOOL_ARGS_TOKENS
        extractor.max_assistant_response_tokens = MAX_ASSISTANT_RESPONSE_TOKENS
        extractor.memory_type = MagicMock()
        extractor.filter_prompt = "{messages}"
        extractor.experience_compress_prompt = "{messages}"
        extractor.tool_pre_compress_prompt = "{messages_json}{new_count}"
        extractor.llm_provider = MagicMock()

        _, request = _make_memcell_request()

        mock_count = MagicMock(return_value=400)  # 400 <= 500 (scale_trigger), no scaling

        with (
            patch.object(AgentCaseExtractor, "_unwrap_messages", return_value=_STUB_MSGS),
            patch.object(AgentCaseExtractor, "_strip_before_first_user", return_value=_STUB_MSGS),
            patch.object(AgentCaseExtractor, "_should_skip", return_value=None),
            patch.object(AgentCaseExtractor, "_count_tokens", mock_count),
            patch.object(AgentCaseExtractor, "_heuristic_trim_tool_outputs", return_value=_STUB_MSGS),
            patch.object(AgentCaseExtractor, "_pre_compress_to_list", AsyncMock(return_value=_STUB_MSGS)),
            patch.object(AgentCaseExtractor, "_filter_conversation", AsyncMock(return_value=False)),
        ):
            await extractor.extract_memory(request)

        assert mock_count.call_count == 1

    @pytest.mark.asyncio
    async def test_count_tokens_called_twice_when_scaling_needed(self):
        """When total_tokens > scale_trigger, _count_tokens is called twice (before and after trim)."""
        extractor = object.__new__(AgentCaseExtractor)
        extractor.pre_compress_chunk_size = 1000  # _STUB_MSGS has 4 msgs < HIGH_MESSAGE_COUNT_THRESHOLD, so scale_trigger = 1000, skip = 2000
        extractor.max_tool_output_tokens = MAX_TOOL_OUTPUT_TOKENS
        extractor.max_tool_args_tokens = MAX_TOOL_ARGS_TOKENS
        extractor.max_assistant_response_tokens = MAX_ASSISTANT_RESPONSE_TOKENS
        extractor.memory_type = MagicMock()
        extractor.filter_prompt = "{messages}"
        extractor.experience_compress_prompt = "{messages}"
        extractor.tool_pre_compress_prompt = "{messages_json}{new_count}"
        extractor.llm_provider = MagicMock()

        _, request = _make_memcell_request()

        # First: 1200 > 1000 (scaling triggered); Second: 600 <= 2000 (no skip)
        token_returns = iter([1200, 600])
        mock_count = MagicMock(side_effect=lambda _: next(token_returns))

        with (
            patch.object(AgentCaseExtractor, "_unwrap_messages", return_value=_STUB_MSGS),
            patch.object(AgentCaseExtractor, "_strip_before_first_user", return_value=_STUB_MSGS),
            patch.object(AgentCaseExtractor, "_should_skip", return_value=None),
            patch.object(AgentCaseExtractor, "_count_tokens", mock_count),
            patch.object(AgentCaseExtractor, "_heuristic_trim_tool_outputs", return_value=_STUB_MSGS),
            patch.object(AgentCaseExtractor, "_pre_compress_to_list", AsyncMock(return_value=_STUB_MSGS)),
            patch.object(AgentCaseExtractor, "_filter_conversation", AsyncMock(return_value=False)),
        ):
            await extractor.extract_memory(request)

        assert mock_count.call_count == 2


def _make_msgs(n: int) -> list:
    """Build a list of n alternating user/assistant messages for threshold tests."""
    msgs = []
    for i in range(n):
        if i % 2 == 0:
            msgs.append({"role": "user", "content": f"msg {i}"})
        else:
            msgs.append({"role": "assistant", "content": f"reply {i}"})
    return msgs


def _make_extractor(chunk_size: int) -> AgentCaseExtractor:
    extractor = object.__new__(AgentCaseExtractor)
    extractor.pre_compress_chunk_size = chunk_size
    extractor.max_tool_output_tokens = MAX_TOOL_OUTPUT_TOKENS
    extractor.max_tool_args_tokens = MAX_TOOL_ARGS_TOKENS
    extractor.max_assistant_response_tokens = MAX_ASSISTANT_RESPONSE_TOKENS
    extractor.memory_type = MagicMock()
    extractor.filter_prompt = "{messages}"
    extractor.experience_compress_prompt = "{messages}"
    extractor.tool_pre_compress_prompt = "{messages_json}{new_count}"
    extractor.llm_provider = MagicMock()
    return extractor


class TestHighMessageCountThreshold:
    """Tests for the HIGH_MESSAGE_COUNT_THRESHOLD branch in extract_memory.

    When len(messages) > HIGH_MESSAGE_COUNT_THRESHOLD the scale_trigger is halved,
    causing more aggressive trim. When len(messages) <= threshold the full
    pre_compress_chunk_size is used.
    """

    @pytest.mark.asyncio
    async def test_exactly_at_threshold_uses_full_scale_trigger(self):
        """len == HIGH_MESSAGE_COUNT_THRESHOLD (100) should NOT halve the trigger."""
        chunk_size = 1000
        msgs = _make_msgs(HIGH_MESSAGE_COUNT_THRESHOLD)  # exactly 100 messages
        extractor = _make_extractor(chunk_size)
        _, request = _make_memcell_request()

        # scale_trigger = chunk_size = 1000; token 800 <= 1000 -> no scaling
        mock_trim = MagicMock(return_value=msgs)
        with (
            patch.object(AgentCaseExtractor, "_unwrap_messages", return_value=msgs),
            patch.object(AgentCaseExtractor, "_strip_before_first_user", return_value=msgs),
            patch.object(AgentCaseExtractor, "_should_skip", return_value=None),
            patch.object(AgentCaseExtractor, "_count_tokens", return_value=800),
            patch.object(AgentCaseExtractor, "_heuristic_trim_tool_outputs", mock_trim),
            patch.object(AgentCaseExtractor, "_pre_compress_to_list", AsyncMock(return_value=msgs)),
            patch.object(AgentCaseExtractor, "_filter_conversation", AsyncMock(return_value=False)),
        ):
            await extractor.extract_memory(request)

        # No scaling -> full limits passed
        mock_trim.assert_called_once_with(msgs, MAX_TOOL_OUTPUT_TOKENS, MAX_TOOL_ARGS_TOKENS, MAX_ASSISTANT_RESPONSE_TOKENS)

    @pytest.mark.asyncio
    async def test_one_above_threshold_halves_scale_trigger(self):
        """len == HIGH_MESSAGE_COUNT_THRESHOLD + 1 (101) SHOULD halve the trigger."""
        chunk_size = 1000
        msgs = _make_msgs(HIGH_MESSAGE_COUNT_THRESHOLD + 1)  # 101 messages
        extractor = _make_extractor(chunk_size)
        _, request = _make_memcell_request()

        # scale_trigger = 500; token 800 > 500 -> scaling triggered
        # 800 would NOT trigger scaling if msg count were low (800 < 1000)
        total = 800
        token_returns = iter([total, chunk_size])
        mock_trim = MagicMock(return_value=msgs)
        with (
            patch.object(AgentCaseExtractor, "_unwrap_messages", return_value=msgs),
            patch.object(AgentCaseExtractor, "_strip_before_first_user", return_value=msgs),
            patch.object(AgentCaseExtractor, "_should_skip", return_value=None),
            patch.object(AgentCaseExtractor, "_count_tokens", side_effect=lambda _: next(token_returns)),
            patch.object(AgentCaseExtractor, "_heuristic_trim_tool_outputs", mock_trim),
            patch.object(AgentCaseExtractor, "_pre_compress_to_list", AsyncMock(return_value=msgs)),
            patch.object(AgentCaseExtractor, "_filter_conversation", AsyncMock(return_value=False)),
        ):
            await extractor.extract_memory(request)

        scale_trigger = chunk_size // 2  # 500
        scale = scale_trigger / total    # 500/800 = 0.625
        mock_trim.assert_called_once_with(
            msgs,
            max(200, int(MAX_TOOL_OUTPUT_TOKENS * scale)),
            max(200, int(MAX_TOOL_ARGS_TOKENS * scale)),
            max(500, int(MAX_ASSISTANT_RESPONSE_TOKENS * scale)),
        )

    @pytest.mark.asyncio
    async def test_gray_zone_tokens_trigger_scaling_only_for_high_msg_count(self):
        """Tokens between chunk_size//2 and chunk_size: scaling fires for >100 msgs but not for <=100."""
        chunk_size = 1000
        gray_zone_tokens = 700  # chunk_size//2 < 700 < chunk_size

        # HIGH message count: scale_trigger=500, 700 > 500 -> scaling
        high_msgs = _make_msgs(HIGH_MESSAGE_COUNT_THRESHOLD + 1)
        extractor_high = _make_extractor(chunk_size)
        _, request_high = _make_memcell_request()

        token_returns_high = iter([gray_zone_tokens, chunk_size])
        mock_trim_high = MagicMock(return_value=high_msgs)
        with (
            patch.object(AgentCaseExtractor, "_unwrap_messages", return_value=high_msgs),
            patch.object(AgentCaseExtractor, "_strip_before_first_user", return_value=high_msgs),
            patch.object(AgentCaseExtractor, "_should_skip", return_value=None),
            patch.object(AgentCaseExtractor, "_count_tokens", side_effect=lambda _: next(token_returns_high)),
            patch.object(AgentCaseExtractor, "_heuristic_trim_tool_outputs", mock_trim_high),
            patch.object(AgentCaseExtractor, "_pre_compress_to_list", AsyncMock(return_value=high_msgs)),
            patch.object(AgentCaseExtractor, "_filter_conversation", AsyncMock(return_value=False)),
        ):
            await extractor_high.extract_memory(request_high)

        # High msg count: scaling was applied
        call_args_high = mock_trim_high.call_args[0]
        assert call_args_high[1] < MAX_TOOL_OUTPUT_TOKENS or call_args_high[2] < MAX_TOOL_ARGS_TOKENS, \
            "Expected scaled-down limits for high message count"

        # LOW message count: scale_trigger=1000, 700 <= 1000 -> no scaling
        low_msgs = _make_msgs(HIGH_MESSAGE_COUNT_THRESHOLD)
        extractor_low = _make_extractor(chunk_size)
        _, request_low = _make_memcell_request()

        mock_trim_low = MagicMock(return_value=low_msgs)
        with (
            patch.object(AgentCaseExtractor, "_unwrap_messages", return_value=low_msgs),
            patch.object(AgentCaseExtractor, "_strip_before_first_user", return_value=low_msgs),
            patch.object(AgentCaseExtractor, "_should_skip", return_value=None),
            patch.object(AgentCaseExtractor, "_count_tokens", return_value=gray_zone_tokens),
            patch.object(AgentCaseExtractor, "_heuristic_trim_tool_outputs", mock_trim_low),
            patch.object(AgentCaseExtractor, "_pre_compress_to_list", AsyncMock(return_value=low_msgs)),
            patch.object(AgentCaseExtractor, "_filter_conversation", AsyncMock(return_value=False)),
        ):
            await extractor_low.extract_memory(request_low)

        # Low msg count: no scaling, full limits
        mock_trim_low.assert_called_once_with(low_msgs, MAX_TOOL_OUTPUT_TOKENS, MAX_TOOL_ARGS_TOKENS, MAX_ASSISTANT_RESPONSE_TOKENS)

    @pytest.mark.asyncio
    async def test_high_msg_count_tokens_below_halved_trigger_no_scaling(self):
        """Even with >100 messages, tokens <= chunk_size//2 should not trigger scaling."""
        chunk_size = 1000
        msgs = _make_msgs(HIGH_MESSAGE_COUNT_THRESHOLD + 1)  # 101 messages -> scale_trigger = 500
        extractor = _make_extractor(chunk_size)
        _, request = _make_memcell_request()

        mock_trim = MagicMock(return_value=msgs)
        with (
            patch.object(AgentCaseExtractor, "_unwrap_messages", return_value=msgs),
            patch.object(AgentCaseExtractor, "_strip_before_first_user", return_value=msgs),
            patch.object(AgentCaseExtractor, "_should_skip", return_value=None),
            patch.object(AgentCaseExtractor, "_count_tokens", return_value=400),  # 400 <= 500, no scaling
            patch.object(AgentCaseExtractor, "_heuristic_trim_tool_outputs", mock_trim),
            patch.object(AgentCaseExtractor, "_pre_compress_to_list", AsyncMock(return_value=msgs)),
            patch.object(AgentCaseExtractor, "_filter_conversation", AsyncMock(return_value=False)),
        ):
            await extractor.extract_memory(request)

        mock_trim.assert_called_once_with(msgs, MAX_TOOL_OUTPUT_TOKENS, MAX_TOOL_ARGS_TOKENS, MAX_ASSISTANT_RESPONSE_TOKENS)

    @pytest.mark.asyncio
    async def test_high_msg_count_scaled_limits_use_halved_trigger_for_scale(self):
        """With >100 messages, the scale factor is computed from chunk_size//2, not chunk_size."""
        chunk_size = 1000
        total = 2000
        msgs = _make_msgs(HIGH_MESSAGE_COUNT_THRESHOLD + 1)  # 101 messages
        extractor = _make_extractor(chunk_size)
        _, request = _make_memcell_request()

        token_returns = iter([total, chunk_size])
        mock_trim = MagicMock(return_value=msgs)
        with (
            patch.object(AgentCaseExtractor, "_unwrap_messages", return_value=msgs),
            patch.object(AgentCaseExtractor, "_strip_before_first_user", return_value=msgs),
            patch.object(AgentCaseExtractor, "_should_skip", return_value=None),
            patch.object(AgentCaseExtractor, "_count_tokens", side_effect=lambda _: next(token_returns)),
            patch.object(AgentCaseExtractor, "_heuristic_trim_tool_outputs", mock_trim),
            patch.object(AgentCaseExtractor, "_pre_compress_to_list", AsyncMock(return_value=msgs)),
            patch.object(AgentCaseExtractor, "_filter_conversation", AsyncMock(return_value=False)),
        ):
            await extractor.extract_memory(request)

        # scale = (chunk_size // 2) / total = 500/2000 = 0.25
        scale_trigger = chunk_size // 2
        scale = scale_trigger / total
        mock_trim.assert_called_once_with(
            msgs,
            max(200, int(MAX_TOOL_OUTPUT_TOKENS * scale)),
            max(200, int(MAX_TOOL_ARGS_TOKENS * scale)),
            max(500, int(MAX_ASSISTANT_RESPONSE_TOKENS * scale)),
        )

    @pytest.mark.asyncio
    async def test_high_msg_count_floor_limits_at_extreme_token_count(self):
        """With >100 messages at extreme token counts, floor values still apply (200, 200, 500)."""
        chunk_size = 1000
        msgs = _make_msgs(HIGH_MESSAGE_COUNT_THRESHOLD + 1)  # 101 messages -> scale_trigger = 500
        extractor = _make_extractor(chunk_size)
        _, request = _make_memcell_request()

        token_returns = iter([1_000_000, chunk_size])
        mock_trim = MagicMock(return_value=msgs)
        with (
            patch.object(AgentCaseExtractor, "_unwrap_messages", return_value=msgs),
            patch.object(AgentCaseExtractor, "_strip_before_first_user", return_value=msgs),
            patch.object(AgentCaseExtractor, "_should_skip", return_value=None),
            patch.object(AgentCaseExtractor, "_count_tokens", side_effect=lambda _: next(token_returns)),
            patch.object(AgentCaseExtractor, "_heuristic_trim_tool_outputs", mock_trim),
            patch.object(AgentCaseExtractor, "_pre_compress_to_list", AsyncMock(return_value=msgs)),
            patch.object(AgentCaseExtractor, "_filter_conversation", AsyncMock(return_value=False)),
        ):
            await extractor.extract_memory(request)

        mock_trim.assert_called_once_with(msgs, 200, 200, 500)

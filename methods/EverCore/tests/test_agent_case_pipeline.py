"""
AgentCase Full Pipeline Unit Tests

Tests for:
- AgentCaseExtractor.extract_memory: end-to-end extraction flow
- _convert_agent_case_to_doc: BO-to-document conversion
- _extract_user_id_from_memcell: user ID extraction from MemCell
- _clamp_quality_score: quality score validation
- _unwrap_messages: MemCell original_data unwrapping + content normalization
- _should_skip: heuristic pre-filtering (skips no-tool short conversations)
- _filter_conversation: LLM-based filtering (for remaining low-tool-call cases)
- _compress_experience: LLM-based extraction
- _pre_compress_to_list: tool content compression

Usage:
    PYTHONPATH=src pytest tests/test_agent_case_pipeline.py -v
"""

import json
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any, Dict, List

from api_specs.memory_types import MemCell, RawDataType, AgentCase
from api_specs.memory_models import MemoryType
from memory_layer.memory_extractor.agent_case_extractor import (
    AgentCaseExtractor,
    AgentCaseExtractRequest,
)
from memory_layer.memory_extractor.agent_skill_extractor import (
    AgentSkillExtractor,
    SkillExtractionResult,
)
from memory_layer.memory_extractor.base_memory_extractor import MemoryExtractRequest
from biz_layer.mem_db_operations import (
    _convert_agent_case_to_doc,
    _extract_user_id_from_memcell,
)
from biz_layer.mem_memorize import _is_agent_case_quality_sufficient
from biz_layer.memorize_config import MemorizeConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wrap_msg(msg: Dict[str, Any]) -> Dict[str, Any]:
    """Wrap a message in MemCell original_data format."""
    return {"message": msg, "parse_info": None}


def _user_msg(content: str, sender_id: str = "user_001") -> Dict[str, Any]:
    return {"role": "user", "content": content, "sender_id": sender_id}


def _assistant_msg(content: str) -> Dict[str, Any]:
    return {"role": "assistant", "content": content}


def _tool_call_msg(
    content: str = "", name: str = "search", arguments: str = "{}"
) -> Dict[str, Any]:
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [{"id": "call_1", "function": {"name": name, "arguments": arguments}}],
    }


def _tool_response_msg(content: str = "result") -> Dict[str, Any]:
    return {"role": "tool", "content": content, "tool_call_id": "call_1"}


def _make_agent_memcell(
    messages: List[Dict[str, Any]],
    event_id: str = "evt_001",
    group_id: str = "group_001",
) -> MemCell:
    """Create a MemCell with wrapped messages for agent conversation."""
    original_data = [_wrap_msg(m) for m in messages]
    return MemCell(
        user_id_list=["user_001"],
        original_data=original_data,
        timestamp=datetime(2025, 3, 1, 12, 0, 0),
        event_id=event_id,
        group_id=group_id,
        participants=["user_001"],
        sender_ids=["user_001"],
        type=RawDataType.AGENTCONVERSATION,
    )


@pytest.fixture(autouse=True)
def _mock_tokenizer():
    """Patch get_bean_by_type to return a real tiktoken tokenizer."""
    import tiktoken

    encoding = tiktoken.get_encoding("o200k_base")
    mock_factory = MagicMock()
    mock_factory.get_tokenizer_from_tiktoken.return_value = encoding

    with patch(
        "memory_layer.memory_extractor.agent_case_extractor.get_bean_by_type",
        return_value=mock_factory,
    ):
        yield


def _build_case_extractor(
    llm_responses: List[str] = None,
) -> AgentCaseExtractor:
    """Build extractor with mocked LLM and vectorize service."""
    mock_llm = MagicMock()
    if llm_responses:
        mock_llm.generate = AsyncMock(side_effect=llm_responses)
    else:
        mock_llm.generate = AsyncMock(
            return_value=json.dumps({
                "task_intent": "Build a REST API endpoint",
                "approach": "1. Define the route\n2. Implement handler\n3. Add validation",
                "quality_score": 0.8,
            })
        )

    return AgentCaseExtractor(
        llm_provider=mock_llm,
        filter_prompt="{messages}",
        experience_compress_prompt="{messages}",
        tool_pre_compress_prompt="{messages_json}{new_count}",
    )


# ===========================================================================
# _unwrap_messages tests
# ===========================================================================


class TestUnwrapMessages:
    """Tests for AgentCaseExtractor._unwrap_messages."""

    def test_unwrap_standard_format(self):
        data = [
            _wrap_msg(_user_msg("hello")),
            _wrap_msg(_assistant_msg("hi")),
        ]
        result = AgentCaseExtractor._unwrap_messages(data)
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"

    def test_unwrap_content_list_normalized(self):
        """v1 API content[] list is converted to plain string."""
        msg = {
            "role": "user",
            "content": [
                {"type": "text", "content": "hello "},
                {"type": "text", "content": "world"},
            ],
        }
        data = [_wrap_msg(msg)]
        result = AgentCaseExtractor._unwrap_messages(data)
        assert isinstance(result[0]["content"], str)
        assert "hello" in result[0]["content"]
        assert "world" in result[0]["content"]

    def test_unwrap_bare_messages(self):
        """Messages without wrapping are passed through."""
        data = [_user_msg("hello")]
        result = AgentCaseExtractor._unwrap_messages(data)
        assert len(result) == 1
        assert result[0]["content"] == "hello"

    def test_non_dict_items_skipped(self):
        data = ["not_a_dict", None, _wrap_msg(_user_msg("hello"))]
        result = AgentCaseExtractor._unwrap_messages(data)
        assert len(result) == 1

    def test_empty_list(self):
        result = AgentCaseExtractor._unwrap_messages([])
        assert result == []


# ===========================================================================
# _clamp_quality_score tests
# ===========================================================================


class TestClampQualityScore:
    """Tests for AgentCaseExtractor._clamp_quality_score."""

    def test_normal_value(self):
        assert AgentCaseExtractor._clamp_quality_score(0.5) == 0.5

    def test_above_one_clamped(self):
        assert AgentCaseExtractor._clamp_quality_score(1.5) == 1.0

    def test_below_zero_clamped(self):
        assert AgentCaseExtractor._clamp_quality_score(-0.5) == 0.0

    def test_none_returns_none(self):
        assert AgentCaseExtractor._clamp_quality_score(None) is None

    def test_string_number(self):
        assert AgentCaseExtractor._clamp_quality_score("0.7") == 0.7

    def test_invalid_string_returns_none(self):
        assert AgentCaseExtractor._clamp_quality_score("abc") is None

    def test_boundary_values(self):
        assert AgentCaseExtractor._clamp_quality_score(0.0) == 0.0
        assert AgentCaseExtractor._clamp_quality_score(1.0) == 1.0


# ===========================================================================
# _should_skip heuristic filter tests
# ===========================================================================


class TestHeuristicFilter:
    """Tests for _should_skip heuristic that skips no-tool short conversations."""

    def test_single_turn_no_tools_skipped(self):
        """Single user+assistant turn without tools is skipped."""
        messages = [
            _user_msg("What time is it?"),
            _assistant_msg("3 PM"),
        ]
        result = AgentCaseExtractor._should_skip(messages)
        assert result is not None

    def test_no_tool_four_messages_skipped(self):
        """No-tool conversation with exactly 4 messages is skipped."""
        messages = [
            _user_msg("What is X?"),
            _assistant_msg("X is Y."),
            _user_msg("Thanks"),
            _assistant_msg("OK."),
        ]
        result = AgentCaseExtractor._should_skip(messages)
        assert result is not None
        assert "No-tool conversation" in result

    def test_no_tool_brief_assistant_skipped(self):
        """No-tool conversation with > 4 messages but brief assistant response is skipped."""
        messages = [
            _user_msg("My service drops connections"),
            _assistant_msg("Can you share the config?"),
            _user_msg("Here: timeout=30"),
            _assistant_msg("Change timeout to 60"),
            _user_msg("That fixed it, thanks!"),
            _assistant_msg("Glad it worked."),
        ]
        result = AgentCaseExtractor._should_skip(messages)
        assert result is not None
        assert "brief assistant response" in result

    def test_no_tool_substantial_assistant_passes(self):
        """No-tool conversation with substantial assistant response passes to LLM filter."""
        long_response_1 = (
            "I can see the issue. Let me explain in detail. The problem is that your "
            "configuration file has an incorrect parameter setting. You need to change "
            "the timeout value from 30 to 60 seconds, and also update the retry count "
            "from 1 to 3. Here are the exact steps you should follow: First, open the "
            "config.yaml file in your project root directory. Second, locate the network "
            "section under the server block. Third, update the timeout field to 60."
        )
        long_response_2 = (
            "Now for the second part of the fix. You also need to update the retries "
            "field to 3, and then save and restart the service with the command "
            "systemctl restart myservice. This should resolve the connection drops you "
            "are experiencing during peak hours. The root cause is that the default "
            "timeout of 30 seconds is too aggressive for your upstream dependencies "
            "which have P99 latency around 45 seconds. After making these changes, "
            "monitor the Grafana dashboard for at least two hours to confirm stability."
        )
        messages = [
            _user_msg("My service keeps dropping connections during peak hours"),
            _assistant_msg(long_response_1),
            _user_msg("OK, what about the retries?"),
            _assistant_msg(long_response_2),
            _user_msg("That fixed it, thanks!"),
            _assistant_msg("Glad it worked. Keep monitoring the dashboard."),
        ]
        result = AgentCaseExtractor._should_skip(messages)
        assert result is None

    def test_single_tool_round_passes(self):
        """Conversations with tool calls pass regardless of message count."""
        messages = [
            _user_msg("Build an API"),
            _tool_call_msg("writing code"),
            _tool_response_msg("done"),
            _assistant_msg("API is ready"),
        ]
        result = AgentCaseExtractor._should_skip(messages)
        assert result is None

    def test_multi_tool_rounds_passes(self):
        """Conversations with > 1 tool round pass."""
        messages = [
            _user_msg("Fix the bug"),
            _tool_call_msg("reading file"),
            _tool_response_msg("content"),
            _tool_call_msg("editing file"),
            _tool_response_msg("saved"),
            _assistant_msg("Done"),
        ]
        result = AgentCaseExtractor._should_skip(messages)
        assert result is None


# ===========================================================================
# _filter_conversation tests
# ===========================================================================


class TestFilterConversation:
    """Tests for AgentCaseExtractor._filter_conversation (LLM-based)."""

    @pytest.mark.asyncio
    async def test_worth_extracting_true(self):
        extractor = _build_case_extractor()
        extractor.llm_provider.generate = AsyncMock(
            return_value=json.dumps({"worth_extracting": True, "reason": "complex task"})
        )
        result = await extractor._filter_conversation("messages json")
        assert result is True

    @pytest.mark.asyncio
    async def test_worth_extracting_false(self):
        extractor = _build_case_extractor()
        extractor.llm_provider.generate = AsyncMock(
            return_value=json.dumps({"worth_extracting": False, "reason": "trivial"})
        )
        result = await extractor._filter_conversation("messages json")
        assert result is False

    @pytest.mark.asyncio
    async def test_llm_failure_defaults_to_true(self):
        extractor = _build_case_extractor()
        extractor.llm_provider.generate = AsyncMock(side_effect=Exception("LLM error"))
        result = await extractor._filter_conversation("messages json")
        assert result is True

    @pytest.mark.asyncio
    async def test_invalid_json_defaults_to_true(self):
        extractor = _build_case_extractor()
        extractor.llm_provider.generate = AsyncMock(return_value="not json")
        result = await extractor._filter_conversation("messages json")
        assert result is True


# ===========================================================================
# _compress_experience tests
# ===========================================================================


class TestCompressExperience:
    """Tests for AgentCaseExtractor._compress_experience."""

    @pytest.mark.asyncio
    async def test_successful_extraction(self):
        extractor = _build_case_extractor()
        extractor.llm_provider.generate = AsyncMock(
            return_value=json.dumps({
                "task_intent": "Setup CI/CD pipeline",
                "approach": "1. Configure GitHub Actions\n2. Add tests",
                "quality_score": 0.9,
            })
        )
        result = await extractor._compress_experience("messages")
        assert result["task_intent"] == "Setup CI/CD pipeline"
        assert result["quality_score"] == 0.9

    @pytest.mark.asyncio
    async def test_empty_task_intent_returns_none(self):
        extractor = _build_case_extractor()
        extractor.llm_provider.generate = AsyncMock(
            return_value=json.dumps({
                "task_intent": "",
                "approach": "steps",
                "quality_score": 0.5,
            })
        )
        result = await extractor._compress_experience("messages")
        assert result is None

    @pytest.mark.asyncio
    async def test_empty_approach_returns_none(self):
        extractor = _build_case_extractor()
        extractor.llm_provider.generate = AsyncMock(
            return_value=json.dumps({
                "task_intent": "Something",
                "approach": "",
                "quality_score": 0.5,
            })
        )
        result = await extractor._compress_experience("messages")
        assert result is None

    @pytest.mark.asyncio
    async def test_retries_on_failure_returns_none(self):
        extractor = _build_case_extractor()
        extractor.llm_provider.generate = AsyncMock(side_effect=Exception("fail"))
        result = await extractor._compress_experience("messages")
        assert result is None
        assert extractor.llm_provider.generate.call_count == 2


# ===========================================================================
# extract_memory end-to-end tests
# ===========================================================================


class TestExtractMemoryE2E:
    """End-to-end tests for AgentCaseExtractor.extract_memory."""

    @pytest.mark.asyncio
    @patch("memory_layer.memory_extractor.agent_case_extractor.get_vectorize_service")
    async def test_successful_extraction(self, mock_vs_factory):
        mock_vs = MagicMock()
        mock_vs.get_embedding = AsyncMock(return_value=MagicMock(tolist=lambda: [0.1, 0.2]))
        mock_vs.get_model_name.return_value = "test-embed"
        mock_vs_factory.return_value = mock_vs

        extractor = _build_case_extractor()
        memcell = _make_agent_memcell([
            _user_msg("Build me a REST API for user management"),
            _tool_call_msg("Let me create the code", "write_file", '{"path": "api.py"}'),
            _tool_response_msg("File created successfully"),
            _assistant_msg("I've built the REST API with user CRUD endpoints"),
        ])
        request = MemoryExtractRequest(memcell=memcell, user_id="user_001", group_id="group_001")
        result = await extractor.extract_memory(request)

        assert result is not None
        assert isinstance(result, AgentCase)
        assert result.task_intent == "Build a REST API endpoint"
        assert result.vector == [0.1, 0.2]
        assert result.memory_type == MemoryType.AGENT_CASE

    @pytest.mark.asyncio
    async def test_non_agent_memcell_returns_none(self):
        extractor = _build_case_extractor()
        memcell = MemCell(
            user_id_list=["u1"],
            original_data=[_wrap_msg(_user_msg("hi"))],
            timestamp=datetime(2025, 1, 1),
            type=RawDataType.CONVERSATION,
        )
        request = MemoryExtractRequest(memcell=memcell)
        result = await extractor.extract_memory(request)
        assert result is None

    @pytest.mark.asyncio
    async def test_empty_memcell_returns_none(self):
        extractor = _build_case_extractor()
        request = MemoryExtractRequest(memcell=None)
        result = await extractor.extract_memory(request)
        assert result is None

    @pytest.mark.asyncio
    async def test_single_turn_no_tools_skipped(self):
        """Single user-assistant turn without tools is skipped."""
        extractor = _build_case_extractor()
        memcell = _make_agent_memcell([
            _user_msg("What is Python?"),
            _assistant_msg("Python is a programming language."),
        ])
        request = MemoryExtractRequest(memcell=memcell)
        result = await extractor.extract_memory(request)
        assert result is None

    @pytest.mark.asyncio
    async def test_incomplete_trajectory_skipped(self):
        """Last message is tool call, not final response."""
        extractor = _build_case_extractor()
        memcell = _make_agent_memcell([
            _user_msg("Do something"),
            _tool_call_msg("calling"),
        ])
        request = MemoryExtractRequest(memcell=memcell)
        result = await extractor.extract_memory(request)
        assert result is None

    @pytest.mark.asyncio
    @patch("memory_layer.memory_extractor.agent_case_extractor.get_vectorize_service")
    async def test_multi_turn_no_tools_passes(self, mock_vs_factory):
        """Multi-turn conversation without tools passes heuristic + LLM filter."""
        mock_vs = MagicMock()
        mock_vs.get_embedding = AsyncMock(return_value=MagicMock(tolist=lambda: [0.1]))
        mock_vs.get_model_name.return_value = "test"
        mock_vs_factory.return_value = mock_vs

        # LLM filter returns True, then compress_experience returns data
        filter_resp = json.dumps({"worth_extracting": True})
        extract_resp = json.dumps({
            "task_intent": "Debug the app",
            "approach": "1. Check logs\n2. Fix error",
            "quality_score": 0.7,
        })

        extractor = _build_case_extractor()
        extractor.llm_provider.generate = AsyncMock(side_effect=[filter_resp, extract_resp])

        # Need > 4 messages AND > 200 assistant tokens to pass heuristic
        memcell = _make_agent_memcell([
            _user_msg("Help me debug this TypeError"),
            _assistant_msg(
                "I can see the issue. The TypeError on line 5 is caused by passing "
                "a string where an integer is expected. You need to wrap the input "
                "with int() before passing it to the calculate function. Here is the "
                "corrected code: change calculate(user_input) to calculate(int(user_input)). "
                "This happens because input() always returns a string in Python 3, even "
                "when the user types a number. The int() call converts it properly. "
                "You should also add error handling with a try/except ValueError block "
                "around the int() conversion to gracefully handle non-numeric input."
            ),
            _user_msg("That worked, but now KeyError on line 12"),
            _assistant_msg(
                "The second error is a KeyError because the dictionary key config_value "
                "does not exist in your settings dict. You should use settings.get("
                "'config_value', default_value) instead of settings['config_value']. "
                "The .get() method returns the default when the key is missing, preventing "
                "the KeyError. This pattern is common when dealing with optional configuration "
                "parameters that may not be present in all environments. Additionally, you "
                "might want to add type hints to your function signatures so that your IDE "
                "can catch these kinds of type mismatches earlier during development."
            ),
            _user_msg("Both fixes worked, thanks!"),
            _assistant_msg("Great, glad both issues are resolved now."),
        ])
        request = MemoryExtractRequest(memcell=memcell, user_id="user_001")
        result = await extractor.extract_memory(request)
        assert result is not None
        assert result.task_intent == "Debug the app"

    @pytest.mark.asyncio
    async def test_filter_rejects_trivial_conversation(self):
        """LLM filter says not worth extracting -> returns None."""
        extractor = _build_case_extractor()
        extractor.llm_provider.generate = AsyncMock(
            return_value=json.dumps({"worth_extracting": False, "reason": "trivial"})
        )
        memcell = _make_agent_memcell([
            _user_msg("What time is it?"),
            _tool_call_msg("checking time"),
            _tool_response_msg("3:00 PM"),
            _assistant_msg("It's 3:00 PM"),
        ])
        request = MemoryExtractRequest(memcell=memcell)
        result = await extractor.extract_memory(request)
        assert result is None

    @pytest.mark.asyncio
    async def test_heuristic_rejects_short_no_tool_conversation(self):
        """Heuristic skips no-tool conversation with <= 4 messages (no LLM call)."""
        extractor = _build_case_extractor()
        memcell = _make_agent_memcell([
            _user_msg("What time is it?"),
            _assistant_msg("It's 3 PM"),
        ])
        request = MemoryExtractRequest(memcell=memcell)
        result = await extractor.extract_memory(request)
        assert result is None
        # No LLM call should have been made
        extractor.llm_provider.generate.assert_not_called()

    @pytest.mark.asyncio
    @patch("memory_layer.memory_extractor.agent_case_extractor.get_vectorize_service")
    async def test_embedding_failure_still_returns_case(self, mock_vs_factory):
        """If embedding fails, AgentCase is still returned without vector."""
        mock_vs_factory.side_effect = Exception("Vectorize service down")

        extractor = _build_case_extractor()
        memcell = _make_agent_memcell([
            _user_msg("Build an API"),
            _tool_call_msg("writing code"),
            _tool_response_msg("done"),
            _assistant_msg("API is ready"),
        ])
        request = MemoryExtractRequest(memcell=memcell, user_id="user_001")
        result = await extractor.extract_memory(request)
        assert result is not None
        assert result.vector is None

    @pytest.mark.asyncio
    @patch("memory_layer.memory_extractor.agent_case_extractor.get_vectorize_service")
    async def test_system_messages_stripped(self, mock_vs_factory):
        """System messages before first user message are stripped."""
        mock_vs = MagicMock()
        mock_vs.get_embedding = AsyncMock(return_value=MagicMock(tolist=lambda: [0.1]))
        mock_vs.get_model_name.return_value = "test"
        mock_vs_factory.return_value = mock_vs

        extractor = _build_case_extractor()
        memcell = _make_agent_memcell([
            {"role": "system", "content": "You are a helpful assistant"},
            _user_msg("Build an API"),
            _tool_call_msg("writing"),
            _tool_response_msg("ok"),
            _assistant_msg("Done"),
        ])
        request = MemoryExtractRequest(memcell=memcell, user_id="u1")
        result = await extractor.extract_memory(request)
        assert result is not None


# ===========================================================================
# _convert_agent_case_to_doc tests
# ===========================================================================


class TestConvertAgentCaseToDoc:
    """Tests for _convert_agent_case_to_doc.

    AgentCaseRecord is a Beanie document that requires MongoDB initialization,
    so we mock its constructor to capture the kwargs and verify field mapping.
    """

    @patch("biz_layer.mem_db_operations.AgentCaseRecord")
    def test_basic_conversion(self, mock_record_cls):
        mock_record_cls.return_value = MagicMock()
        agent_case = AgentCase(
            memory_type=MemoryType.AGENT_CASE,
            user_id="user_001",
            timestamp=datetime(2025, 3, 1),
            task_intent="Build API",
            approach="1. Design\n2. Implement",
            quality_score=0.8,
            vector=[0.1, 0.2],
            vector_model="test-model",
        )
        memcell = _make_agent_memcell([
            _user_msg("Build an API", sender_id="agent_user_001"),
            _assistant_msg("Done"),
        ])
        _convert_agent_case_to_doc(agent_case, memcell)
        kwargs = mock_record_cls.call_args.kwargs
        assert kwargs["task_intent"] == "Build API"
        assert kwargs["approach"] == "1. Design\n2. Implement"
        assert kwargs["quality_score"] == 0.8
        assert kwargs["parent_type"] == "memcell"
        assert kwargs["parent_id"] == "evt_001"
        assert kwargs["vector"] == [0.1, 0.2]
        assert kwargs["user_id"] == "agent_user_001"

    @patch("biz_layer.mem_db_operations.AgentCaseRecord")
    def test_no_user_message_user_id_none(self, mock_record_cls):
        mock_record_cls.return_value = MagicMock()
        agent_case = AgentCase(
            memory_type=MemoryType.AGENT_CASE,
            user_id="",
            timestamp=datetime(2025, 1, 1),
            task_intent="task",
            approach="approach",
        )
        memcell = _make_agent_memcell([_assistant_msg("hello")])
        _convert_agent_case_to_doc(agent_case, memcell)
        kwargs = mock_record_cls.call_args.kwargs
        assert kwargs["user_id"] is None

    @patch("biz_layer.mem_db_operations.AgentCaseRecord")
    def test_uses_memcell_timestamp(self, mock_record_cls):
        mock_record_cls.return_value = MagicMock()
        agent_case = AgentCase(
            memory_type=MemoryType.AGENT_CASE,
            user_id="u1",
            timestamp=datetime(2025, 1, 1),
            task_intent="t",
            approach="a",
        )
        memcell = _make_agent_memcell([_user_msg("hi"), _assistant_msg("hey")])
        _convert_agent_case_to_doc(agent_case, memcell)
        kwargs = mock_record_cls.call_args.kwargs
        assert kwargs["timestamp"] == datetime(2025, 3, 1, 12, 0, 0)

    @patch("biz_layer.mem_db_operations.AgentCaseRecord")
    def test_fallback_to_current_time(self, mock_record_cls):
        mock_record_cls.return_value = MagicMock()
        agent_case = AgentCase(
            memory_type=MemoryType.AGENT_CASE,
            user_id="u1",
            timestamp=None,
            task_intent="t",
            approach="a",
        )
        memcell = MemCell(
            user_id_list=["u1"],
            original_data=[_wrap_msg(_user_msg("hi")), _wrap_msg(_assistant_msg("hey"))],
            timestamp=None,
            type=RawDataType.AGENTCONVERSATION,
        )
        current_time = datetime(2025, 6, 15, 10, 0, 0)
        _convert_agent_case_to_doc(agent_case, memcell, current_time=current_time)
        kwargs = mock_record_cls.call_args.kwargs
        assert kwargs["timestamp"] == current_time


# ===========================================================================
# _is_agent_case_quality_sufficient tests
# ===========================================================================


class TestIsAgentCaseQualitySufficient:
    """Tests for _is_agent_case_quality_sufficient.

    Verifies that skill extraction is gated by the AgentCase quality_score
    against the configurable skill_min_quality_score threshold.
    """

    def _make_case(self, quality_score):
        return AgentCase(
            memory_type=MemoryType.AGENT_CASE,
            user_id="user_001",
            timestamp=datetime(2025, 3, 1),
            task_intent="task",
            approach="approach",
            quality_score=quality_score,
        )

    def test_score_above_threshold_passes(self):
        config = MemorizeConfig(skill_min_quality_score=0.1)
        assert _is_agent_case_quality_sufficient(self._make_case(0.5), config) is True

    def test_score_equal_to_threshold_passes(self):
        config = MemorizeConfig(skill_min_quality_score=0.1)
        assert _is_agent_case_quality_sufficient(self._make_case(0.1), config) is True

    def test_score_below_threshold_rejected(self):
        config = MemorizeConfig(skill_min_quality_score=0.1)
        assert _is_agent_case_quality_sufficient(self._make_case(0.05), config) is False

    def test_score_zero_below_default_threshold(self):
        config = MemorizeConfig(skill_min_quality_score=0.1)
        assert _is_agent_case_quality_sufficient(self._make_case(0.0), config) is False

    def test_score_none_rejected(self):
        """None quality_score should block skill extraction (unknown quality)."""
        config = MemorizeConfig(skill_min_quality_score=0.1)
        assert _is_agent_case_quality_sufficient(self._make_case(None), config) is False

    def test_custom_threshold(self):
        config = MemorizeConfig(skill_min_quality_score=0.5)
        assert _is_agent_case_quality_sufficient(self._make_case(0.3), config) is False
        assert _is_agent_case_quality_sufficient(self._make_case(0.5), config) is True
        assert _is_agent_case_quality_sufficient(self._make_case(0.8), config) is True

    def test_threshold_zero_allows_all(self):
        """When threshold is 0.0, all scores (including 0.0) should pass."""
        config = MemorizeConfig(skill_min_quality_score=0.0)
        assert _is_agent_case_quality_sufficient(self._make_case(0.0), config) is True
        assert _is_agent_case_quality_sufficient(self._make_case(0.01), config) is True


# ===========================================================================
# _extract_user_id_from_memcell tests
# ===========================================================================


class TestExtractUserIdFromMemcell:
    """Tests for _extract_user_id_from_memcell."""

    def test_extracts_first_user_sender_id(self):
        memcell = _make_agent_memcell([
            _user_msg("hello", sender_id="user_abc"),
            _assistant_msg("hi"),
        ])
        assert _extract_user_id_from_memcell(memcell) == "user_abc"

    def test_no_user_messages_returns_none(self):
        memcell = _make_agent_memcell([_assistant_msg("hello")])
        assert _extract_user_id_from_memcell(memcell) is None

    def test_user_without_sender_id_returns_none(self):
        msg = {"role": "user", "content": "hello"}  # no sender_id
        memcell = MemCell(
            user_id_list=["u1"],
            original_data=[_wrap_msg(msg)],
            timestamp=datetime(2025, 1, 1),
            type=RawDataType.AGENTCONVERSATION,
        )
        assert _extract_user_id_from_memcell(memcell) is None

    def test_empty_original_data(self):
        memcell = MemCell(
            user_id_list=["u1"],
            original_data=[_wrap_msg(_user_msg("dummy"))],
            timestamp=datetime(2025, 1, 1),
        )
        # Override original_data to empty after construction
        memcell.original_data = []
        assert _extract_user_id_from_memcell(memcell) is None

    def test_skips_non_user_messages(self):
        memcell = _make_agent_memcell([
            {"role": "system", "content": "system prompt"},
            _assistant_msg("thinking"),
            _user_msg("actual question", sender_id="real_user"),
        ])
        assert _extract_user_id_from_memcell(memcell) == "real_user"


# ===========================================================================
# _pre_compress_to_list tests
# ===========================================================================


class TestPreCompressToList:
    """Tests for AgentCaseExtractor._pre_compress_to_list."""

    @pytest.mark.asyncio
    async def test_no_tool_calls_returns_as_is(self):
        extractor = _build_case_extractor()
        messages = [
            _user_msg("hello"),
            _assistant_msg("world"),
        ]
        result = await extractor._pre_compress_to_list(messages)
        assert len(result) == 2
        assert result[0]["content"] == "hello"

    @pytest.mark.asyncio
    async def test_small_tool_content_no_compression(self):
        extractor = _build_case_extractor()
        messages = [
            _user_msg("search"),
            _tool_call_msg("searching", "search", '{"q": "test"}'),
            _tool_response_msg("result"),
            _assistant_msg("found it"),
        ]
        result = await extractor._pre_compress_to_list(messages)
        assert len(result) == 4
        # No LLM call for compression
        extractor.llm_provider.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_does_not_mutate_original(self):
        extractor = _build_case_extractor()
        original = [_user_msg("hello"), _assistant_msg("world")]
        import copy
        original_copy = copy.deepcopy(original)
        await extractor._pre_compress_to_list(original)
        assert original == original_copy


# ===========================================================================
# _collect_tool_call_groups tests
# ===========================================================================


class TestCollectToolCallGroups:
    """Tests for AgentCaseExtractor._collect_tool_call_groups."""

    def test_single_group(self):
        extractor = _build_case_extractor()
        items = [
            _user_msg("search"),
            _tool_call_msg("searching"),
            _tool_response_msg("result"),
            _assistant_msg("done"),
        ]
        groups = extractor._collect_tool_call_groups(items)
        assert len(groups) == 1
        assert groups[0] == [1, 2]

    def test_multiple_groups(self):
        extractor = _build_case_extractor()
        items = [
            _user_msg("do things"),
            _tool_call_msg("first"),
            _tool_response_msg("r1"),
            _tool_call_msg("second"),
            _tool_response_msg("r2"),
            _assistant_msg("done"),
        ]
        groups = extractor._collect_tool_call_groups(items)
        assert len(groups) == 2
        assert groups[0] == [1, 2]
        assert groups[1] == [3, 4]

    def test_no_tool_calls(self):
        extractor = _build_case_extractor()
        items = [_user_msg("hi"), _assistant_msg("hello")]
        groups = extractor._collect_tool_call_groups(items)
        assert groups == []

    def test_tool_call_with_multiple_responses(self):
        extractor = _build_case_extractor()
        items = [
            _tool_call_msg("batch"),
            _tool_response_msg("r1"),
            _tool_response_msg("r2"),
            _assistant_msg("done"),
        ]
        groups = extractor._collect_tool_call_groups(items)
        assert len(groups) == 1
        assert groups[0] == [0, 1, 2]

    def test_empty_items(self):
        extractor = _build_case_extractor()
        groups = extractor._collect_tool_call_groups([])
        assert groups == []


# ===========================================================================
# MemCell.conversation_data tests (agent conversation filtering)
# ===========================================================================


class TestMemCellConversationData:
    """Tests for MemCell.conversation_data property filtering."""

    def test_agent_conversation_filters_tool_messages(self):
        memcell = _make_agent_memcell([
            _user_msg("search for X"),
            _tool_call_msg("searching"),
            _tool_response_msg("found X"),
            _assistant_msg("Here is X"),
        ])
        conv_data = memcell.conversation_data
        # Should only have user and final assistant (tool messages filtered)
        roles = []
        for item in conv_data:
            msg = item.get("message", item)
            roles.append(msg.get("role"))
        assert "tool" not in roles
        assert len(conv_data) == 2

    def test_regular_conversation_not_filtered(self):
        memcell = MemCell(
            user_id_list=["u1"],
            original_data=[_wrap_msg(_user_msg("hi")), _wrap_msg(_assistant_msg("hello"))],
            timestamp=datetime(2025, 1, 1),
            type=RawDataType.CONVERSATION,
        )
        assert len(memcell.conversation_data) == 2

    def test_original_data_preserved(self):
        """original_data is not affected by conversation_data filtering."""
        memcell = _make_agent_memcell([
            _user_msg("query"),
            _tool_call_msg("calling"),
            _tool_response_msg("result"),
            _assistant_msg("answer"),
        ])
        _ = memcell.conversation_data  # trigger filtering
        assert len(memcell.original_data) == 4


# ===========================================================================
# _compress_tool_chunk tests
# ===========================================================================


class TestCompressToolChunk:
    """Tests for AgentCaseExtractor._compress_tool_chunk."""

    @pytest.mark.asyncio
    async def test_successful_compression(self):
        compressed = [
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "function": {"name": "search", "arguments": "{}"}}]},
            {"role": "tool", "content": "compressed result", "tool_call_id": "c1"},
        ]
        extractor = _build_case_extractor()
        extractor.llm_provider.generate = AsyncMock(
            return_value=json.dumps({"compressed_messages": compressed})
        )
        messages = [
            _tool_call_msg("searching", "search", '{"q": "long query"}'),
            _tool_response_msg("very long result " * 100),
        ]
        result = await extractor._compress_tool_chunk(messages)
        assert result is not None
        assert len(result) == 2
        assert result[1]["content"] == "compressed result"

    @pytest.mark.asyncio
    async def test_wrong_count_returns_none(self):
        """If LLM returns different number of messages, return None."""
        extractor = _build_case_extractor()
        extractor.llm_provider.generate = AsyncMock(
            return_value=json.dumps({"compressed_messages": [{"role": "tool", "content": "only one"}]})
        )
        messages = [
            _tool_call_msg("searching"),
            _tool_response_msg("result"),
        ]
        result = await extractor._compress_tool_chunk(messages)
        assert result is None

    @pytest.mark.asyncio
    async def test_missing_key_returns_none(self):
        """If LLM response lacks compressed_messages key, return None."""
        extractor = _build_case_extractor()
        extractor.llm_provider.generate = AsyncMock(
            return_value=json.dumps({"wrong_key": []})
        )
        result = await extractor._compress_tool_chunk([_tool_response_msg("x")])
        assert result is None

    @pytest.mark.asyncio
    async def test_llm_exception_retries_then_none(self):
        """On repeated LLM errors, retries twice then returns None."""
        extractor = _build_case_extractor()
        extractor.llm_provider.generate = AsyncMock(
            side_effect=Exception("LLM down")
        )
        result = await extractor._compress_tool_chunk([_tool_response_msg("x")])
        assert result is None
        assert extractor.llm_provider.generate.call_count == 2

    @pytest.mark.asyncio
    async def test_first_fail_second_success(self):
        """Retries on first failure, succeeds on second attempt."""
        compressed = [{"role": "tool", "content": "ok", "tool_call_id": "c1"}]
        extractor = _build_case_extractor()
        extractor.llm_provider.generate = AsyncMock(
            side_effect=[
                "invalid json!!!",
                json.dumps({"compressed_messages": compressed}),
            ]
        )
        result = await extractor._compress_tool_chunk([_tool_response_msg("x")])
        assert result is not None
        assert len(result) == 1


# ===========================================================================
# _pre_compress_to_list with large content tests
# ===========================================================================


class TestPreCompressLargeContent:
    """Tests for _pre_compress_to_list when tool content exceeds threshold."""

    @pytest.mark.asyncio
    async def test_large_content_triggers_compression(self):
        """When tool content exceeds threshold, LLM compression is called."""
        compressed_msgs = [
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "function": {"name": "search", "arguments": "{}"}}]},
            {"role": "tool", "content": "compressed", "tool_call_id": "c1"},
        ]
        extractor = _build_case_extractor()
        # Set a very low threshold to force compression
        extractor.pre_compress_chunk_size = 10
        extractor.llm_provider.generate = AsyncMock(
            return_value=json.dumps({"compressed_messages": compressed_msgs})
        )
        messages = [
            _user_msg("search"),
            _tool_call_msg("searching", "search", '{"q": "test"}'),
            _tool_response_msg("a]very long result " * 200),
            _assistant_msg("done"),
        ]
        result = await extractor._pre_compress_to_list(messages)
        # LLM should have been called for compression
        assert extractor.llm_provider.generate.call_count >= 1
        assert len(result) == 4

    @pytest.mark.asyncio
    async def test_compression_failure_keeps_originals(self):
        """When LLM compression fails, original messages are preserved."""
        extractor = _build_case_extractor()
        extractor.pre_compress_chunk_size = 10
        extractor.llm_provider.generate = AsyncMock(
            side_effect=Exception("LLM error")
        )
        messages = [
            _user_msg("search"),
            _tool_call_msg("searching", "search", '{"q": "test"}'),
            _tool_response_msg("long result " * 200),
            _assistant_msg("done"),
        ]
        result = await extractor._pre_compress_to_list(messages)
        # Should still return all messages (originals preserved)
        assert len(result) == 4

    @pytest.mark.asyncio
    async def test_does_not_mutate_original_messages(self):
        """_pre_compress_to_list should not modify the input list."""
        import copy as copy_mod

        extractor = _build_case_extractor()
        extractor.pre_compress_chunk_size = 10
        compressed_msgs = [
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "function": {"name": "s", "arguments": "{}"}}]},
            {"role": "tool", "content": "compressed", "tool_call_id": "c1"},
        ]
        extractor.llm_provider.generate = AsyncMock(
            return_value=json.dumps({"compressed_messages": compressed_msgs})
        )
        messages = [
            _user_msg("go"),
            _tool_call_msg("calling", "search", '{"q": "x"}'),
            _tool_response_msg("long " * 200),
            _assistant_msg("done"),
        ]
        original_copy = copy_mod.deepcopy(messages)
        await extractor._pre_compress_to_list(messages)
        assert messages == original_copy


# ===========================================================================
# _compute_embedding tests
# ===========================================================================


class TestComputeEmbedding:
    """Tests for AgentCaseExtractor._compute_embedding."""

    @pytest.mark.asyncio
    @patch("memory_layer.memory_extractor.agent_case_extractor.get_vectorize_service")
    async def test_successful_embedding(self, mock_vs_factory):
        mock_vs = MagicMock()
        mock_vs.get_embedding = AsyncMock(
            return_value=MagicMock(tolist=lambda: [0.1, 0.2, 0.3])
        )
        mock_vs.get_model_name.return_value = "test-model"
        mock_vs_factory.return_value = mock_vs

        extractor = _build_case_extractor()
        result = await extractor._compute_embedding("Build a REST API")
        assert result is not None
        assert result["embedding"] == [0.1, 0.2, 0.3]
        assert result["vector_model"] == "test-model"

    @pytest.mark.asyncio
    async def test_empty_text_returns_none(self):
        extractor = _build_case_extractor()
        result = await extractor._compute_embedding("")
        assert result is None

    @pytest.mark.asyncio
    @patch("memory_layer.memory_extractor.agent_case_extractor.get_vectorize_service")
    async def test_embedding_exception_returns_none(self, mock_vs_factory):
        mock_vs_factory.side_effect = Exception("Service unavailable")
        extractor = _build_case_extractor()
        result = await extractor._compute_embedding("some text")
        assert result is None

    @pytest.mark.asyncio
    @patch("memory_layer.memory_extractor.agent_case_extractor.get_vectorize_service")
    async def test_embedding_without_tolist(self, mock_vs_factory):
        """Embedding result that is a plain list (no tolist method)."""
        mock_vs = MagicMock()
        mock_vs.get_embedding = AsyncMock(return_value=[0.5, 0.6])
        mock_vs.get_model_name.return_value = "plain-model"
        mock_vs_factory.return_value = mock_vs

        extractor = _build_case_extractor()
        result = await extractor._compute_embedding("test text")
        assert result is not None
        assert result["embedding"] == [0.5, 0.6]


# ===========================================================================
# _unwrap_messages edge cases
# ===========================================================================


class TestUnwrapMessagesEdgeCases:
    """Additional edge case tests for _unwrap_messages."""

    def test_assistant_with_content_list_normalized(self):
        """Assistant message with content[] list is also normalized."""
        msg = {
            "role": "assistant",
            "content": [
                {"type": "text", "content": "Here is "},
                {"type": "text", "content": "the answer"},
            ],
        }
        data = [_wrap_msg(msg)]
        result = AgentCaseExtractor._unwrap_messages(data)
        assert isinstance(result[0]["content"], str)
        assert "Here is" in result[0]["content"]
        assert "the answer" in result[0]["content"]

    def test_tool_message_with_content_list_normalized(self):
        """Tool message with content[] list is normalized to string."""
        msg = {
            "role": "tool",
            "content": [{"type": "text", "content": "tool output"}],
            "tool_call_id": "c1",
        }
        data = [_wrap_msg(msg)]
        result = AgentCaseExtractor._unwrap_messages(data)
        assert isinstance(result[0]["content"], str)
        assert "tool output" in result[0]["content"]

    def test_assistant_with_tool_calls_and_null_content(self):
        """Assistant message with tool_calls and None content stays None."""
        msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "c1", "function": {"name": "f", "arguments": "{}"}}],
        }
        data = [_wrap_msg(msg)]
        result = AgentCaseExtractor._unwrap_messages(data)
        assert result[0]["content"] is None
        assert result[0]["tool_calls"] is not None


# ===========================================================================
# extract_memory additional edge cases
# ===========================================================================


class TestExtractMemoryEdgeCases:
    """Additional edge case tests for extract_memory."""

    @pytest.mark.asyncio
    async def test_llm_extraction_all_retries_fail(self):
        """When LLM extraction fails after all retries, returns None."""
        extractor = _build_case_extractor()
        extractor.llm_provider.generate = AsyncMock(
            side_effect=Exception("LLM unavailable")
        )
        memcell = _make_agent_memcell([
            _user_msg("Build me a complex API"),
            _tool_call_msg("writing", "write_file", '{"path": "api.py"}'),
            _tool_response_msg("done"),
            _tool_call_msg("testing", "run_test", '{"file": "test.py"}'),
            _tool_response_msg("passed"),
            _assistant_msg("All done"),
        ])
        request = MemoryExtractRequest(memcell=memcell, user_id="u1")
        result = await extractor.extract_memory(request)
        assert result is None

    @pytest.mark.asyncio
    async def test_only_system_messages_returns_none(self):
        """MemCell with only system messages (no user) returns None."""
        extractor = _build_case_extractor()
        memcell = _make_agent_memcell([
            {"role": "system", "content": "You are a helper"},
        ])
        request = MemoryExtractRequest(memcell=memcell)
        result = await extractor.extract_memory(request)
        assert result is None
        extractor.llm_provider.generate.assert_not_called()

    @pytest.mark.asyncio
    @patch("memory_layer.memory_extractor.agent_case_extractor.get_vectorize_service")
    async def test_quality_score_clamped_in_result(self, mock_vs_factory):
        """quality_score > 1.0 from LLM is clamped to 1.0 in final AgentCase."""
        mock_vs = MagicMock()
        mock_vs.get_embedding = AsyncMock(return_value=MagicMock(tolist=lambda: [0.1]))
        mock_vs.get_model_name.return_value = "test"
        mock_vs_factory.return_value = mock_vs

        extractor = _build_case_extractor()
        extractor.llm_provider.generate = AsyncMock(
            return_value=json.dumps({
                "task_intent": "Deploy service",
                "approach": "1. Build\n2. Deploy",
                "quality_score": 1.5,
            })
        )
        memcell = _make_agent_memcell([
            _user_msg("Deploy the service"),
            _tool_call_msg("deploying"),
            _tool_response_msg("deployed"),
            _assistant_msg("Service is live"),
        ])
        request = MemoryExtractRequest(memcell=memcell, user_id="u1")
        result = await extractor.extract_memory(request)
        assert result is not None
        assert result.quality_score == 1.0


# ---------------------------------------------------------------------------
# _content_change_ratio tests
# ---------------------------------------------------------------------------

class TestContentChangeRatio:
    """Tests for AgentSkillExtractor._content_change_ratio."""

    def test_both_empty(self):
        assert AgentSkillExtractor._content_change_ratio("", "") == 0.0

    def test_old_empty(self):
        assert AgentSkillExtractor._content_change_ratio("", "new content") == 1.0

    def test_new_empty(self):
        assert AgentSkillExtractor._content_change_ratio("old content", "") == 1.0

    def test_identical(self):
        text = "## Steps\n1. Do something\n2. Check result"
        assert AgentSkillExtractor._content_change_ratio(text, text) == 0.0

    def test_minor_edit(self):
        old = "## Steps\n1. Run the query\n2. Check the result\n3. Return output"
        new = "## Steps\n1. Run the query\n2. Check the result\n3. Return the output"
        ratio = AgentSkillExtractor._content_change_ratio(old, new)
        assert ratio < 0.1, f"Minor edit should have small ratio, got {ratio}"

    def test_major_rewrite(self):
        old = "## Steps\n1. Use curl to call API\n2. Parse JSON response"
        new = "## Steps\n1. Use Python requests library\n2. Handle pagination\n3. Retry on failure\n4. Cache results"
        ratio = AgentSkillExtractor._content_change_ratio(old, new)
        assert ratio >= 0.3, f"Major rewrite should have ratio >= 0.3, got {ratio}"

    def test_completely_different(self):
        ratio = AgentSkillExtractor._content_change_ratio("aaaaaa", "zzzzzz")
        assert ratio >= 0.9, f"Completely different should be near 1.0, got {ratio}"


# ---------------------------------------------------------------------------
# Maturity re-evaluation skip/trigger logic in _apply_update
# ---------------------------------------------------------------------------

def _make_skill_record(
    content="## Steps\n1. Step one\n  - How: do it\n2. Step two\n  - How: do that\n3. Done\n  - Check: ok",
    name="Test Skill",
    description="A test skill",
    confidence=0.7,
    maturity_score=0.8,
    source_case_ids=None,
):
    """Create a mock skill record for _apply_update tests."""
    record = MagicMock()
    record.id = "skill_001"
    record.content = content
    record.name = name
    record.description = description
    record.confidence = confidence
    record.maturity_score = maturity_score
    record.source_case_ids = source_case_ids or []
    return record


def _build_skill_extractor():
    """Build an AgentSkillExtractor with mocked LLM provider."""
    llm = MagicMock()
    llm.generate = AsyncMock(return_value="")
    extractor = AgentSkillExtractor(
        llm_provider=llm,
        maturity_threshold=0.6,
        retire_confidence=0.1,
    )
    extractor._compute_embedding = AsyncMock(return_value=None)
    return extractor


class TestMaturityReevalLogic:
    """Tests that maturity re-evaluation triggers correctly based on
    content change ratio, confidence direction, and existing maturity."""

    @pytest.mark.asyncio
    async def test_skip_when_trivial_change(self):
        """Content change < 10% should always skip maturity, even if immature."""
        extractor = _build_skill_extractor()
        extractor._evaluate_maturity = AsyncMock(return_value=0.9)

        record = _make_skill_record(maturity_score=0.4, confidence=0.7)
        # Trivial edit: append a tiny note
        new_content = record.content + "\n  - Note: ok"
        ratio = AgentSkillExtractor._content_change_ratio(record.content, new_content)
        assert ratio < 0.1, f"Test setup: expected trivial change, got {ratio}"

        op = {"index": 0, "data": {"content": new_content, "confidence": 0.7}}
        repo = MagicMock()
        repo.update_skill_by_id = AsyncMock(return_value=True)
        result = SkillExtractionResult()

        await extractor._apply_update(op, [record], repo, result)

        extractor._evaluate_maturity.assert_not_called()

    @pytest.mark.asyncio
    async def test_skip_when_trivial_change_even_confidence_drops(self):
        """Trivial content change should skip maturity even if confidence drops."""
        extractor = _build_skill_extractor()
        extractor._evaluate_maturity = AsyncMock(return_value=0.5)

        record = _make_skill_record(maturity_score=0.8, confidence=0.7)
        new_content = record.content + "\n  - Note: ok"
        ratio = AgentSkillExtractor._content_change_ratio(record.content, new_content)
        assert ratio < 0.1, f"Test setup: expected trivial change, got {ratio}"

        op = {"index": 0, "data": {"content": new_content, "confidence": 0.3}}
        repo = MagicMock()
        repo.update_skill_by_id = AsyncMock(return_value=True)
        result = SkillExtractionResult()

        await extractor._apply_update(op, [record], repo, result)

        extractor._evaluate_maturity.assert_not_called()

    @pytest.mark.asyncio
    async def test_skip_when_minor_edit_and_mature(self):
        """Change 10-30% on a mature skill with stable confidence should skip."""
        extractor = _build_skill_extractor()
        extractor._evaluate_maturity = AsyncMock(return_value=0.9)

        record = _make_skill_record(maturity_score=0.8, confidence=0.7)
        # Moderate edit (~15% change) but still below 30%
        new_content = record.content + "\n4. Extra verification step\n  - How: run checks\n  - Check: all green"
        ratio = AgentSkillExtractor._content_change_ratio(record.content, new_content)
        assert 0.1 <= ratio < 0.3, f"Test setup: expected 10-30% change, got {ratio}"

        op = {"index": 0, "data": {"content": new_content, "confidence": 0.7}}
        repo = MagicMock()
        repo.update_skill_by_id = AsyncMock(return_value=True)
        result = SkillExtractionResult()

        await extractor._apply_update(op, [record], repo, result)

        extractor._evaluate_maturity.assert_not_called()

    @pytest.mark.asyncio
    async def test_trigger_when_confidence_drops(self):
        """Mature skill + moderate change + confidence dropping: LLM re-evaluation."""
        extractor = _build_skill_extractor()
        extractor._evaluate_maturity = AsyncMock(return_value=0.65)

        record = _make_skill_record(maturity_score=0.8, confidence=0.7)
        # Non-trivial edit (>= 10%) + confidence drops
        new_content = record.content + "\n4. Extra verification step\n  - How: run checks\n  - Check: all green"
        ratio = AgentSkillExtractor._content_change_ratio(record.content, new_content)
        assert ratio >= 0.1, f"Test setup: expected non-trivial change, got {ratio}"

        op = {"index": 0, "data": {"content": new_content, "confidence": 0.4}}
        repo = MagicMock()
        repo.update_skill_by_id = AsyncMock(return_value=True)
        result = SkillExtractionResult()

        await extractor._apply_update(op, [record], repo, result)

        extractor._evaluate_maturity.assert_called_once()
        update_args = repo.update_skill_by_id.call_args[0][1]
        assert "maturity_score" in update_args
        assert update_args["maturity_score"] == 0.65

    @pytest.mark.asyncio
    async def test_trigger_when_immature(self):
        """Immature skill with moderate change and decent source quality: LLM re-evaluation."""
        extractor = _build_skill_extractor()
        extractor._evaluate_maturity = AsyncMock(return_value=0.7)

        record = _make_skill_record(maturity_score=0.4, confidence=0.7)
        new_content = record.content + "\n4. Extra verification step\n  - How: run checks\n  - Check: all green"
        ratio = AgentSkillExtractor._content_change_ratio(record.content, new_content)
        assert ratio >= 0.1, f"Test setup: expected non-trivial change, got {ratio}"

        op = {"index": 0, "data": {"content": new_content, "confidence": 0.7}}
        repo = MagicMock()
        repo.update_skill_by_id = AsyncMock(return_value=True)
        result = SkillExtractionResult()

        await extractor._apply_update(
            op, [record], repo, result, source_quality=0.7,
        )

        extractor._evaluate_maturity.assert_called_once()
        update_args = repo.update_skill_by_id.call_args[0][1]
        assert "maturity_score" in update_args
        assert update_args["maturity_score"] == 0.7

    @pytest.mark.asyncio
    async def test_trigger_when_major_content_change(self):
        """Content change >= 40% should always trigger LLM re-evaluation."""
        extractor = _build_skill_extractor()
        extractor._evaluate_maturity = AsyncMock(return_value=0.85)

        record = _make_skill_record(maturity_score=0.8, confidence=0.7)
        # Completely rewrite the content
        new_content = "## Steps\n1. Totally new approach\n  - How: different method\n2. New step\n  - How: new way\n3. Another step\n  - Check: new check\n4. Final\n  - Check: done"
        op = {"index": 0, "data": {"content": new_content, "confidence": 0.7}}
        repo = MagicMock()
        repo.update_skill_by_id = AsyncMock(return_value=True)
        result = SkillExtractionResult()

        # Verify the change ratio is actually >= 0.4
        ratio = AgentSkillExtractor._content_change_ratio(record.content, new_content)
        assert ratio >= 0.4, f"Test setup: expected change ratio >= 0.4, got {ratio}"

        await extractor._apply_update(op, [record], repo, result)

        extractor._evaluate_maturity.assert_called_once()
        update_args = repo.update_skill_by_id.call_args[0][1]
        assert "maturity_score" in update_args
        assert update_args["maturity_score"] == 0.85

    @pytest.mark.asyncio
    async def test_no_content_change_skips_maturity_block(self):
        """When only metadata changes (no content/name/desc), skip maturity entirely."""
        extractor = _build_skill_extractor()
        extractor._evaluate_maturity = AsyncMock(return_value=0.9)

        record = _make_skill_record(maturity_score=0.4, confidence=0.5)
        # Only confidence changes, no content/name/desc
        op = {"index": 0, "data": {"confidence": 0.8}}
        repo = MagicMock()
        repo.update_skill_by_id = AsyncMock(return_value=True)
        result = SkillExtractionResult()

        await extractor._apply_update(op, [record], repo, result)

        extractor._evaluate_maturity.assert_not_called()


# ===========================================================================
# mem_memorize._extract_agent_case (wrapper function)
# ===========================================================================


def _make_extraction_state_for_pipeline(**overrides):
    """Create a minimal ExtractionState-like object for pipeline tests."""
    from types import SimpleNamespace
    memcell = overrides.pop("memcell", MemCell(
        user_id_list=["u1"],
        original_data=[
            {"message": {"role": "user", "content": "Deploy the app", "sender_id": "u1"}},
            {"message": {"role": "assistant", "content": "Done deploying."}},
        ],
        timestamp=datetime(2025, 6, 1, 10, 0, 0),
        event_id="evt_100",
        group_id="g1",
        type=RawDataType.AGENTCONVERSATION,
    ))
    agent_case_bo = overrides.pop("agent_case", AgentCase(
        memory_type=MemoryType.AGENT_CASE,
        user_id="u1",
        timestamp=datetime(2025, 6, 1, 10, 0, 0),
        task_intent="Deploy the application to production",
        approach="1. Build docker image\n2. Push to registry\n3. Deploy to k8s",
        quality_score=0.85,
    ))
    request = overrides.pop("request", SimpleNamespace(
        group_id="g1",
        session_id="sess_001",
    ))
    current_time = overrides.pop("current_time", datetime(2025, 6, 1, 10, 0, 0))

    return SimpleNamespace(
        memcell=memcell,
        agent_case=agent_case_bo,
        request=request,
        current_time=current_time,
        **overrides,
    )


class TestMemorizeExtractAgentCaseWrapper:
    """Tests for _extract_agent_case wrapper in mem_memorize.py."""

    @pytest.mark.asyncio
    async def test_success_stores_on_state(self):
        """Successful extraction stores agent_case on state."""
        expected_case = AgentCase(
            memory_type=MemoryType.AGENT_CASE,
            user_id="u1",
            timestamp=datetime(2025, 6, 1),
            task_intent="Deploy",
            approach="Steps",
            quality_score=0.85,
        )
        mock_manager = AsyncMock()
        mock_manager.extract_memory = AsyncMock(return_value=expected_case)

        state = _make_extraction_state_for_pipeline(agent_case=None)

        from biz_layer.mem_memorize import _extract_agent_case
        result = await _extract_agent_case(state, mock_manager)

        assert result is expected_case
        assert state.agent_case is expected_case
        mock_manager.extract_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_none_result_returns_none(self):
        """When extractor returns None, state.agent_case is not set."""
        mock_manager = AsyncMock()
        mock_manager.extract_memory = AsyncMock(return_value=None)

        state = _make_extraction_state_for_pipeline(agent_case=None)

        from biz_layer.mem_memorize import _extract_agent_case
        result = await _extract_agent_case(state, mock_manager)

        assert result is None

    @pytest.mark.asyncio
    async def test_exception_result_returns_none(self):
        """When extractor returns an Exception, result should be None."""
        mock_manager = AsyncMock()
        mock_manager.extract_memory = AsyncMock(return_value=Exception("LLM error"))

        state = _make_extraction_state_for_pipeline(agent_case=None)

        from biz_layer.mem_memorize import _extract_agent_case
        result = await _extract_agent_case(state, mock_manager)

        assert result is None


# ===========================================================================
# mem_memorize._save_agent_case
# ===========================================================================


class TestMemorizeSaveAgentCase:
    """Tests for _save_agent_case in mem_memorize.py."""

    @pytest.mark.asyncio
    async def test_success_returns_one(self):
        """Successful save returns 1."""
        state = _make_extraction_state_for_pipeline()
        mock_doc = MagicMock()

        with patch("biz_layer.mem_memorize._convert_agent_case_to_doc", return_value=mock_doc) as mock_conv, \
             patch("biz_layer.mem_memorize.save_memory_docs", new_callable=AsyncMock) as mock_save:
            mock_save.return_value = {}

            from biz_layer.mem_memorize import _save_agent_case
            result = await _save_agent_case(state)

            assert result == 1
            mock_conv.assert_called_once()
            mock_save.assert_called_once()

    @pytest.mark.asyncio
    async def test_conversion_error_returns_zero(self):
        """If conversion raises, returns 0."""
        state = _make_extraction_state_for_pipeline()

        with patch("biz_layer.mem_memorize._convert_agent_case_to_doc", side_effect=ValueError("bad data")):
            from biz_layer.mem_memorize import _save_agent_case
            result = await _save_agent_case(state)

            assert result == 0

    @pytest.mark.asyncio
    async def test_save_error_returns_zero(self):
        """If save_memory_docs raises, returns 0."""
        state = _make_extraction_state_for_pipeline()
        mock_doc = MagicMock()

        with patch("biz_layer.mem_memorize._convert_agent_case_to_doc", return_value=mock_doc), \
             patch("biz_layer.mem_memorize.save_memory_docs", new_callable=AsyncMock, side_effect=Exception("DB down")):
            from biz_layer.mem_memorize import _save_agent_case
            result = await _save_agent_case(state)

            assert result == 0

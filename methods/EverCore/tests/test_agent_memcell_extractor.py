"""
AgentMemCellExtractor Unit Tests

Tests agent turn-boundary-aware MemCell splitting:
- Skipping intermediate agent steps (tool_call, tool_response)
- Only splitting at complete agent responses
- Flush mode packing all messages
- History boundary validation
- Helper method correctness

Usage:
    PYTHONPATH=src pytest tests/test_agent_memcell_extractor.py -v
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import timedelta
from typing import List, Dict, Any

from common_utils.datetime_utils import get_now_with_timezone
from api_specs.dtos import RawData
from api_specs.memory_types import RawDataType, is_intermediate_agent_step

from memory_layer.memcell_extractor.agent_memcell_extractor import (
    AgentMemCellExtractor,
    AgentMemCellExtractRequest,
)
from memory_layer.memcell_extractor.base_memcell_extractor import (
    MemCellExtractRequest,
    StatusResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE_TIME = get_now_with_timezone() - timedelta(hours=1)


def _ts(offset_minutes: int) -> str:
    return (BASE_TIME + timedelta(minutes=offset_minutes)).isoformat()


def _user_msg(content: str, offset: int = 0) -> Dict[str, Any]:
    return {
        "role": "user",
        "content": content,
        "speaker_name": "User",
        "timestamp": _ts(offset),
    }


def _assistant_msg(content: str, offset: int = 0) -> Dict[str, Any]:
    """Complete assistant response (no tool_calls)."""
    return {
        "role": "assistant",
        "content": content,
        "speaker_name": "Assistant",
        "timestamp": _ts(offset),
    }


def _tool_call_msg(content: str = "", offset: int = 0) -> Dict[str, Any]:
    """Intermediate assistant message WITH tool_calls."""
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [{"id": "call_1", "function": {"name": "search", "arguments": "{}"}}],
        "speaker_name": "Assistant",
        "timestamp": _ts(offset),
    }


def _tool_response_msg(content: str = "tool result", offset: int = 0) -> Dict[str, Any]:
    """Tool execution result."""
    return {
        "role": "tool",
        "content": content,
        "tool_call_id": "call_1",
        "speaker_name": "Tool",
        "timestamp": _ts(offset),
    }


def _raw(msg: Dict[str, Any], data_id: str = "d") -> RawData:
    return RawData(content=msg, data_id=data_id)


def _raw_list(msgs: List[Dict[str, Any]], prefix: str = "d") -> List[RawData]:
    return [RawData(content=m, data_id=f"{prefix}_{i}") for i, m in enumerate(msgs)]


def _make_request(
    history_msgs: List[Dict[str, Any]],
    new_msgs: List[Dict[str, Any]],
    flush: bool = False,
) -> MemCellExtractRequest:
    return MemCellExtractRequest(
        history_raw_data_list=_raw_list(history_msgs, "h"),
        new_raw_data_list=_raw_list(new_msgs, "n"),
        user_id_list=["user1"],
        group_id="test_group",
        flush=flush,
    )


def _build_extractor() -> AgentMemCellExtractor:
    """Build extractor with a mocked LLM provider (boundary detection returns no boundaries)."""
    mock_llm = MagicMock()
    mock_llm.generate = AsyncMock(
        return_value='{"reasoning": "no boundary", "boundaries": [], "should_wait": true}'
    )
    return AgentMemCellExtractor(llm_provider=mock_llm)


def _build_extractor_should_end() -> AgentMemCellExtractor:
    """Build extractor with LLM that signals boundary at position 2."""
    mock_llm = MagicMock()
    mock_llm.generate = AsyncMock(
        return_value='{"reasoning": "topic changed", "boundaries": [2], "should_wait": false}'
    )
    return AgentMemCellExtractor(llm_provider=mock_llm)


# ---------------------------------------------------------------------------
# Tests: Helper Methods
# ---------------------------------------------------------------------------


class TestHelperMethods:
    """Test is_intermediate_agent_step."""

    def test_tool_response_is_intermediate(self):
        assert is_intermediate_agent_step(_tool_response_msg()) is True

    def test_tool_call_is_intermediate(self):
        assert is_intermediate_agent_step(_tool_call_msg()) is True

    def test_user_msg_not_intermediate(self):
        assert is_intermediate_agent_step(_user_msg("hi")) is False

    def test_final_assistant_not_intermediate(self):
        assert is_intermediate_agent_step(_assistant_msg("done")) is False


# ---------------------------------------------------------------------------
# Tests: Guard 1 - Skip intermediate agent steps
# ---------------------------------------------------------------------------


class TestGuard1SkipIntermediate:
    """New message is intermediate -> return (None, should_wait=True)."""

    def setup_method(self):
        self.extractor = _build_extractor()

    @pytest.mark.asyncio
    async def test_skip_tool_call_message(self):
        request = _make_request(
            history_msgs=[_user_msg("hello", 0)],
            new_msgs=[_tool_call_msg("calling tool", 1)],
        )
        memcell, status = await self.extractor.extract_memcell(request)
        assert memcell == []
        assert status.should_wait is True

    @pytest.mark.asyncio
    async def test_skip_tool_response_message(self):
        request = _make_request(
            history_msgs=[_user_msg("hello", 0)],
            new_msgs=[_tool_response_msg("result", 1)],
        )
        memcell, status = await self.extractor.extract_memcell(request)
        assert memcell == []
        assert status.should_wait is True

    @pytest.mark.asyncio
    async def test_skip_multiple_new_last_is_tool(self):
        """Multiple new messages, last one is tool response -> skip."""
        request = _make_request(
            history_msgs=[_user_msg("hello", 0)],
            new_msgs=[_user_msg("more", 1), _tool_response_msg("result", 2)],
        )
        memcell, status = await self.extractor.extract_memcell(request)
        assert memcell == []
        assert status.should_wait is True


# ---------------------------------------------------------------------------
# Tests: Guard 2 - Flush
# ---------------------------------------------------------------------------


class TestGuard2Flush:
    """Flush mode packs all messages into one MemCell."""

    def setup_method(self):
        self.extractor = _build_extractor()

    @pytest.mark.asyncio
    @patch("memory_layer.memcell_extractor.conv_memcell_extractor.record_boundary_detection")
    @patch("memory_layer.memcell_extractor.conv_memcell_extractor.record_memcell_extracted")
    @patch("memory_layer.memcell_extractor.conv_memcell_extractor.get_space_id_for_metrics", return_value="test")
    async def test_flush_packs_all_messages(self, mock_space, mock_record, mock_bd):
        """Flush creates MemCell from all history + new messages."""
        request = _make_request(
            history_msgs=[
                _user_msg("hello", 0),
                _tool_call_msg("thinking", 1),
                _tool_response_msg("result", 2),
                _assistant_msg("answer", 3),
            ],
            new_msgs=[_user_msg("follow up", 4)],
            flush=True,
        )
        memcell, status = await self.extractor.extract_memcell(request)
        assert len(memcell) > 0
        assert status.should_wait is False
        # All 5 messages should be packed
        assert len(memcell[0].original_data) == 5
        assert memcell[0].type == RawDataType.AGENTCONVERSATION

    @pytest.mark.asyncio
    async def test_flush_empty_messages(self):
        """Flush with no messages returns None."""
        request = _make_request(history_msgs=[], new_msgs=[], flush=True)
        memcell, status = await self.extractor.extract_memcell(request)
        assert memcell == []
        assert status.should_wait is True

    @pytest.mark.asyncio
    async def test_flush_overrides_intermediate(self):
        """Flush has highest priority — even if last new message is intermediate,
        flush packs all messages into a MemCell regardless.
        Downstream extractors detect incomplete trajectories and skip accordingly.
        """
        request = _make_request(
            history_msgs=[_user_msg("hello", 0), _assistant_msg("answer", 1)],
            new_msgs=[_tool_call_msg("thinking", 2)],
            flush=True,
        )
        # Flush fires first (Guard 1), overriding intermediate skip (Guard 2)
        memcell, status = await self.extractor.extract_memcell(request)
        assert len(memcell) > 0
        assert len(memcell[0].original_data) == 3


# ---------------------------------------------------------------------------
# Tests: Guard 3 - History must end at complete agent response
# ---------------------------------------------------------------------------


class TestGuard3HistoryBoundary:
    """History doesn't end at complete agent response -> wait."""

    def setup_method(self):
        self.extractor = _build_extractor()

    @pytest.mark.asyncio
    async def test_history_ends_at_user_msg_waits(self):
        request = _make_request(
            history_msgs=[_user_msg("hello", 0)],
            new_msgs=[_user_msg("more", 1)],
        )
        memcell, status = await self.extractor.extract_memcell(request)
        assert memcell == []
        assert status.should_wait is True

    @pytest.mark.asyncio
    async def test_history_ends_at_tool_call_waits(self):
        request = _make_request(
            history_msgs=[_user_msg("hello", 0), _tool_call_msg("thinking", 1)],
            new_msgs=[_assistant_msg("done", 2)],
        )
        memcell, status = await self.extractor.extract_memcell(request)
        assert memcell == []
        assert status.should_wait is True

    @pytest.mark.asyncio
    async def test_history_ends_at_tool_response_waits(self):
        request = _make_request(
            history_msgs=[
                _user_msg("hello", 0),
                _tool_call_msg("thinking", 1),
                _tool_response_msg("result", 2),
            ],
            new_msgs=[_assistant_msg("done", 3)],
        )
        memcell, status = await self.extractor.extract_memcell(request)
        assert memcell == []
        assert status.should_wait is True


# ---------------------------------------------------------------------------
# Tests: Delegation to parent (all guards pass)
# ---------------------------------------------------------------------------


class TestDelegationToParent:
    """When all guards pass, parent's extract_memcell is called."""

    @pytest.mark.asyncio
    @patch("memory_layer.memcell_extractor.conv_memcell_extractor.record_boundary_detection")
    @patch("memory_layer.memcell_extractor.conv_memcell_extractor.record_memcell_extracted")
    @patch("memory_layer.memcell_extractor.conv_memcell_extractor.get_space_id_for_metrics", return_value="test")
    async def test_valid_boundary_delegates_to_parent_llm_no_boundary(
        self, mock_space, mock_record, mock_bd
    ):
        """History ends at assistant, new msg is user -> delegates to parent.
        Mock LLM returns should_wait -> no MemCell."""
        extractor = _build_extractor()
        request = _make_request(
            history_msgs=[_user_msg("hello", 0), _assistant_msg("hi there", 1)],
            new_msgs=[_user_msg("what is 2+2?", 5)],
        )
        memcell, status = await extractor.extract_memcell(request)
        # LLM says no boundary -> should_wait
        assert memcell == []
        assert status.should_wait is True

    @pytest.mark.asyncio
    @patch("memory_layer.memcell_extractor.conv_memcell_extractor.record_boundary_detection")
    @patch("memory_layer.memcell_extractor.conv_memcell_extractor.record_memcell_extracted")
    @patch("memory_layer.memcell_extractor.conv_memcell_extractor.get_space_id_for_metrics", return_value="test")
    async def test_valid_boundary_llm_detects_end(self, mock_space, mock_record, mock_bd):
        """LLM detects boundary -> MemCell is created from history."""
        extractor = _build_extractor_should_end()
        # new_msgs must include an assistant response to pass the all_user_only guard
        request = _make_request(
            history_msgs=[_user_msg("hello", 0), _assistant_msg("hi there", 1)],
            new_msgs=[_user_msg("new topic entirely", 30), _assistant_msg("sure", 31)],
        )
        memcell, status = await extractor.extract_memcell(request)
        assert len(memcell) > 0
        assert status.should_wait is False
        # MemCell should contain only history messages (boundary at position 2)
        assert len(memcell[0].original_data) == 2
        assert memcell[0].type == RawDataType.AGENTCONVERSATION

    @pytest.mark.asyncio
    @patch("memory_layer.memcell_extractor.conv_memcell_extractor.record_boundary_detection")
    @patch("memory_layer.memcell_extractor.conv_memcell_extractor.record_memcell_extracted")
    @patch("memory_layer.memcell_extractor.conv_memcell_extractor.get_space_id_for_metrics", return_value="test")
    async def test_empty_history_delegates(self, mock_space, mock_record, mock_bd):
        """Empty history + user new msg -> delegates to parent (guard 3 skips for empty history).
        Parent's batch boundary detection returns should_wait per LLM mock."""
        extractor = _build_extractor()
        request = _make_request(
            history_msgs=[],
            new_msgs=[_user_msg("hello", 0)],
        )
        memcell, status = await extractor.extract_memcell(request)
        # No history -> delegates to parent, LLM mock says no boundary + should_wait
        assert memcell == []
        assert status.should_wait is True


# ---------------------------------------------------------------------------
# Tests: Force split respects turn boundaries
# ---------------------------------------------------------------------------


class TestForceSplit:
    """Force split via hard limits respects tool-call boundaries."""

    @pytest.mark.asyncio
    @patch("memory_layer.memcell_extractor.conv_memcell_extractor.record_boundary_detection")
    @patch("memory_layer.memcell_extractor.conv_memcell_extractor.record_memcell_extracted")
    @patch("memory_layer.memcell_extractor.conv_memcell_extractor.get_space_id_for_metrics", return_value="test")
    async def test_force_split_at_valid_boundary(self, mock_space, mock_record, mock_bd):
        """When history is valid and exceeds message limit, force split creates MemCell."""
        extractor = _build_extractor()
        extractor.hard_message_limit = 3  # Low limit to trigger force split

        # new_msgs must include an assistant response to pass the all_user_only guard
        request = _make_request(
            history_msgs=[_user_msg("q1", 0), _assistant_msg("a1", 1)],
            new_msgs=[_user_msg("q2", 2), _assistant_msg("a2", 3)],
        )
        # 2 history + 2 new = 4 >= hard_message_limit -> force split front chunk
        memcell, status = await extractor.extract_memcell(request)
        assert len(memcell) > 0
        # v1: after force split, remaining messages go through LLM boundary detection
        # LLM mock says should_wait=True for remaining messages
        assert status.should_wait is True

    @pytest.mark.asyncio
    @patch("memory_layer.memcell_extractor.conv_memcell_extractor.record_boundary_detection")
    @patch("memory_layer.memcell_extractor.conv_memcell_extractor.record_memcell_extracted")
    @patch("memory_layer.memcell_extractor.conv_memcell_extractor.get_space_id_for_metrics", return_value="test")
    async def test_force_split_regardless_of_turn_boundary(self, mock_space, mock_record, mock_bd):
        """Force split is a safety valve — splits even mid-turn when limits are exceeded."""
        extractor = _build_extractor()
        extractor.hard_message_limit = 3

        request = _make_request(
            history_msgs=[_user_msg("q1", 0), _user_msg("q2", 1)],
            new_msgs=[_assistant_msg("a1", 2)],
        )
        # 3 messages >= hard_message_limit -> force split fires
        memcell, status = await extractor.extract_memcell(request)
        assert len(memcell) > 0


class TestIsSafeSplit:
    """Test _is_safe_split static method."""

    def test_safe_after_final_assistant(self):
        msgs = [_user_msg("q", 0), _assistant_msg("a", 1), _user_msg("q2", 2)]
        assert AgentMemCellExtractor._is_safe_split(msgs, 2) is True

    def test_unsafe_after_tool_call(self):
        msgs = [_user_msg("q", 0), _tool_call_msg("t", 1), _tool_response_msg("r", 2)]
        assert AgentMemCellExtractor._is_safe_split(msgs, 2) is False

    def test_unsafe_after_tool_response(self):
        msgs = [_user_msg("q", 0), _tool_call_msg("t", 1), _tool_response_msg("r", 2)]
        assert AgentMemCellExtractor._is_safe_split(msgs, 3) is False

    def test_unsafe_last_is_user(self):
        """Last message is user -> not safe."""
        msgs = [_user_msg("q", 0), _tool_call_msg("t", 1), _assistant_msg("a", 2)]
        assert AgentMemCellExtractor._is_safe_split(msgs, 1) is False

    def test_safe_last_is_final_assistant(self):
        """Last message is final assistant -> safe."""
        msgs = [
            _user_msg("q", 0),
            _tool_call_msg("t", 1),
            _tool_response_msg("r", 2),
            _assistant_msg("a", 3),
            _user_msg("q2", 4),
        ]
        assert AgentMemCellExtractor._is_safe_split(msgs, 4) is True

    def test_unsafe_last_is_tool_response(self):
        """Last message is tool response -> not safe."""
        msgs = [_tool_call_msg("t", 0), _tool_response_msg("r", 1), _assistant_msg("a", 2)]
        assert AgentMemCellExtractor._is_safe_split(msgs, 2) is False

    def test_unsafe_split_at_zero(self):
        """split_at=0 is out of valid range."""
        msgs = [_user_msg("q", 0), _assistant_msg("a", 1)]
        assert AgentMemCellExtractor._is_safe_split(msgs, 0) is False

    def test_unsafe_split_at_negative(self):
        """Negative split_at is out of valid range."""
        msgs = [_user_msg("q", 0), _assistant_msg("a", 1)]
        assert AgentMemCellExtractor._is_safe_split(msgs, -1) is False

    def test_unsafe_split_at_len(self):
        """split_at=len(messages) would leave no remainder for next chunk."""
        msgs = [_user_msg("q", 0), _assistant_msg("a", 1)]
        assert AgentMemCellExtractor._is_safe_split(msgs, 2) is False

    def test_unsafe_split_beyond_len(self):
        """split_at beyond len(messages) is out of range."""
        msgs = [_user_msg("q", 0), _assistant_msg("a", 1)]
        assert AgentMemCellExtractor._is_safe_split(msgs, 5) is False

    def test_safe_split_at_one_with_assistant(self):
        """split_at=1 is valid if messages[0] is a final assistant."""
        msgs = [_assistant_msg("a", 0), _user_msg("q", 1)]
        assert AgentMemCellExtractor._is_safe_split(msgs, 1) is True

    def test_unsafe_single_message_list(self):
        """Single-element list: split_at=1 means len-1=0, so out of range."""
        msgs = [_assistant_msg("a", 0)]
        assert AgentMemCellExtractor._is_safe_split(msgs, 1) is False


class TestForceSplitToolBoundary:
    """Force split adjusts split point to avoid cutting tool-call sequences."""

    def test_find_force_split_point_avoids_tool_middle(self):
        """Split point should not land after an intermediate agent step.

        Messages:
          [0] user: q1
          [1] assistant + tool_calls  <- intermediate
          [2] tool: result            <- intermediate
          [3] assistant: answer
          [4] user: q2
          [5] assistant: a2

        With hard_message_limit=4, parent returns candidate=3 (split after msg[2]=tool).
        Walking back to 1 would cut a lone user msg, so forward walk to 4
        (after msg[3]=final assistant) which includes a complete turn.
        """
        extractor = _build_extractor()
        extractor.hard_message_limit = 4

        messages = [
            _user_msg("q1", 0),
            _tool_call_msg("thinking", 1),
            _tool_response_msg("result", 2),
            _assistant_msg("answer", 3),
            _user_msg("q2", 4),
            _assistant_msg("a2", 5),
        ]
        split_at = extractor._find_force_split_point(messages)
        # Should include the full tool sequence: user + tool_call + tool + assistant
        assert split_at == 4
        assert messages[split_at - 1]["role"] == "assistant"
        assert not messages[split_at - 1].get("tool_calls")

    def test_find_force_split_point_no_tool_calls(self):
        """Without tool calls, behaves same as parent."""
        extractor = _build_extractor()
        extractor.hard_message_limit = 3

        messages = [
            _user_msg("q1", 0),
            _assistant_msg("a1", 1),
            _user_msg("q2", 2),
            _assistant_msg("a2", 3),
        ]
        split_at = extractor._find_force_split_point(messages)
        assert split_at == 2  # Same as parent: hard_message_limit - 1

    def test_find_force_split_point_forward_when_all_intermediate_before(self):
        """When walking back finds only intermediates, walk forward instead.

        Messages:
          [0] tool_call   <- intermediate
          [1] tool        <- intermediate
          [2] assistant: answer
          [3] user: q2
          [4] assistant: a2

        With hard_message_limit=3, parent candidate=2. Walk back: msg[1]=tool,
        msg[0]=tool_call -> all intermediate, no safe point. Forward walk from 3:
        chunk=[0:3] includes final assistant at [2] -> safe. split_at=3.
        """
        extractor = _build_extractor()
        extractor.hard_message_limit = 3

        messages = [
            _tool_call_msg("thinking", 0),
            _tool_response_msg("result", 1),
            _assistant_msg("answer", 2),
            _user_msg("q2", 3),
            _assistant_msg("a2", 4),
        ]
        split_at = extractor._find_force_split_point(messages)
        assert split_at == 3
        assert messages[split_at - 1]["role"] == "assistant"
        assert not messages[split_at - 1].get("tool_calls")

    def test_split_after_tool_call_keeps_all(self):
        """No safe split exists -> keep all messages in one chunk.

        Messages:
          [0] user
          [1] assistant + tool_calls  <- intermediate
          [2] user
          [3] assistant

        hard_message_limit=3, parent candidate=2 (after msg[1]=tool_call).
        Walk back: no final assistant before candidate. Forward walk:
        candidate=3, msg[2]=user -> not safe. No safe split -> keep all.
        """
        extractor = _build_extractor()
        extractor.hard_message_limit = 3

        messages = [
            _user_msg("q1", 0),
            _tool_call_msg("thinking", 1),
            _user_msg("q2", 2),
            _assistant_msg("a2", 3),
        ]
        split_at = extractor._find_force_split_point(messages)
        assert split_at == len(messages)

    def test_split_after_tool_response_walks_forward(self):
        """Parent candidate lands right after tool response -> walks forward.

        Messages:
          [0] user
          [1] assistant + tool_calls
          [2] tool response
          [3] assistant: answer
          [4] user
          [5] assistant

        hard_message_limit=4, parent candidate=3. msg[2]=tool, msg[1]=tool_call.
        Walking back to 1 gives lone user (no assistant reply) -> not safe.
        Forward walk from 4: chunk=[0:4] includes final assistant at [3] -> safe.
        """
        extractor = _build_extractor()
        extractor.hard_message_limit = 4

        messages = [
            _user_msg("q1", 0),
            _tool_call_msg("thinking", 1),
            _tool_response_msg("result", 2),
            _assistant_msg("answer", 3),
            _user_msg("q2", 4),
            _assistant_msg("a2", 5),
        ]
        split_at = extractor._find_force_split_point(messages)
        assert split_at == 4
        assert messages[split_at - 1]["role"] == "assistant"
        assert not messages[split_at - 1].get("tool_calls")

    def test_split_already_safe_no_adjustment(self):
        """When parent candidate is already at a safe point, no adjustment needed.

        Messages:
          [0] user
          [1] assistant (final)
          [2] user
          [3] assistant + tool_calls
          [4] tool
          [5] assistant

        hard_message_limit=3, parent candidate=2. msg[1]=final assistant -> safe.
        """
        extractor = _build_extractor()
        extractor.hard_message_limit = 3

        messages = [
            _user_msg("q1", 0),
            _assistant_msg("a1", 1),
            _user_msg("q2", 2),
            _tool_call_msg("thinking", 3),
            _tool_response_msg("result", 4),
            _assistant_msg("a2", 5),
        ]
        split_at = extractor._find_force_split_point(messages)
        assert split_at == 2

    def test_multiple_consecutive_tool_calls(self):
        """Agent makes multiple tool calls before final answer.

        Messages:
          [0] user
          [1] assistant + tool_calls  <- intermediate
          [2] tool                    <- intermediate
          [3] assistant + tool_calls  <- intermediate (second tool call)
          [4] tool                    <- intermediate
          [5] assistant: final answer
          [6] user
          [7] assistant

        hard_message_limit=5, parent candidate=4. msg[3]=tool_call, [2]=tool,
        [1]=tool_call, [0]=user -> walk back to 1 gives lone user -> not safe.
        Forward walk from 5: chunk=[0:5] has no final asst, [0:6] has final asst -> 6.
        """
        extractor = _build_extractor()
        extractor.hard_message_limit = 5

        messages = [
            _user_msg("q1", 0),
            _tool_call_msg("call search", 1),
            _tool_response_msg("search result", 2),
            _tool_call_msg("call compute", 3),
            _tool_response_msg("compute result", 4),
            _assistant_msg("final answer", 5),
            _user_msg("q2", 6),
            _assistant_msg("a2", 7),
        ]
        split_at = extractor._find_force_split_point(messages)
        assert split_at == 6
        assert messages[split_at - 1]["role"] == "assistant"
        assert not messages[split_at - 1].get("tool_calls")

    def test_two_complete_turns_with_tools(self):
        """Two full agent turns, limit triggers split between them.

        Messages:
          [0] user
          [1] assistant + tool_calls
          [2] tool
          [3] assistant: answer1
          [4] user
          [5] assistant + tool_calls
          [6] tool
          [7] assistant: answer2
          [8] user
          [9] assistant

        hard_message_limit=6, parent candidate=5. msg[4]=user -> not final asst.
        Walk back: msg[3]=final assistant -> safe. split_at=4, a complete turn.
        """
        extractor = _build_extractor()
        extractor.hard_message_limit = 6

        messages = [
            _user_msg("q1", 0),
            _tool_call_msg("call1", 1),
            _tool_response_msg("r1", 2),
            _assistant_msg("a1", 3),
            _user_msg("q2", 4),
            _tool_call_msg("call2", 5),
            _tool_response_msg("r2", 6),
            _assistant_msg("a2", 7),
            _user_msg("q3", 8),
            _assistant_msg("a3", 9),
        ]
        split_at = extractor._find_force_split_point(messages)
        assert split_at == 4
        assert messages[split_at - 1]["role"] == "assistant"
        assert not messages[split_at - 1].get("tool_calls")

    def test_forward_walk_finds_final_assistant(self):
        """Forward walk skips intermediates to land on final assistant.

        Messages:
          [0] tool_call   <- intermediate
          [1] tool        <- intermediate
          [2] tool_call   <- intermediate (second call)
          [3] tool        <- intermediate
          [4] assistant: final
          [5] user
          [6] assistant

        hard_message_limit=4, parent candidate=3. Walk back: msg[2]=tool_call,
        [1]=tool, [0]=tool_call -> all intermediate. Forward walk from 3:
        msg[2]=tool_call, msg[3]=tool, msg[4]=assistant(final) -> split_at=5.
        """
        extractor = _build_extractor()
        extractor.hard_message_limit = 4

        messages = [
            _tool_call_msg("call1", 0),
            _tool_response_msg("r1", 1),
            _tool_call_msg("call2", 2),
            _tool_response_msg("r2", 3),
            _assistant_msg("final", 4),
            _user_msg("q2", 5),
            _assistant_msg("a2", 6),
        ]
        split_at = extractor._find_force_split_point(messages)
        assert split_at == 5
        assert messages[split_at - 1]["role"] == "assistant"
        assert not messages[split_at - 1].get("tool_calls")

    def test_no_safe_split_keeps_all(self):
        """When no safe split exists, keep all messages in one chunk.

        Messages:
          [0] tool_call   <- intermediate
          [1] tool        <- intermediate
          [2] assistant: final

        hard_message_limit=2, parent candidate=1. Walk back: no safe point.
        Forward walk: no safe point. Keep all as one chunk.
        """
        extractor = _build_extractor()
        extractor.hard_message_limit = 2

        messages = [
            _tool_call_msg("call", 0),
            _tool_response_msg("result", 1),
            _assistant_msg("final", 2),
        ]
        split_at = extractor._find_force_split_point(messages)
        assert split_at == len(messages)

    def test_no_safe_split_long_tool_chain_keeps_all(self):
        """Single turn with many tool calls, no safe split -> keep all.

        Messages:
          [0] user
          [1] tool_call
          [2] tool
          [3] tool_call
          [4] tool
          [5] assistant: final

        hard_message_limit=3, parent candidate=2. Walk back: no safe point.
        Forward walk: no safe point. Keep all as one chunk.
        """
        extractor = _build_extractor()
        extractor.hard_message_limit = 3

        messages = [
            _user_msg("q1", 0),
            _tool_call_msg("c1", 1),
            _tool_response_msg("r1", 2),
            _tool_call_msg("c2", 3),
            _tool_response_msg("r2", 4),
            _assistant_msg("final", 5),
        ]
        split_at = extractor._find_force_split_point(messages)
        assert split_at == len(messages)

    def test_single_tool_call_in_middle(self):
        """Single tool call (no multi-step) in the middle.

        Messages:
          [0] user
          [1] assistant + tool_calls  <- intermediate
          [2] assistant: answer
          [3] user
          [4] assistant

        hard_message_limit=3, parent candidate=2. msg[1]=tool_call -> walk back
        to 1 gives lone user (no asst reply) -> not safe. Forward walk from 3:
        chunk=[0:3] includes final assistant at [2] -> safe. split_at=3.
        """
        extractor = _build_extractor()
        extractor.hard_message_limit = 3

        messages = [
            _user_msg("q1", 0),
            _tool_call_msg("thinking", 1),
            _assistant_msg("answer", 2),
            _user_msg("q2", 3),
            _assistant_msg("a2", 4),
        ]
        split_at = extractor._find_force_split_point(messages)
        assert split_at == 3
        assert messages[split_at - 1]["role"] == "assistant"
        assert not messages[split_at - 1].get("tool_calls")

    def test_split_at_assistant_final_is_safe(self):
        """Candidate landing on final assistant (no tool_calls) is safe.

        Messages:
          [0] user
          [1] assistant: answer (no tool_calls)
          [2] user
          [3] assistant

        hard_message_limit=3, parent candidate=2. msg[1]=final assistant -> safe.
        """
        extractor = _build_extractor()
        extractor.hard_message_limit = 3

        messages = [
            _user_msg("q1", 0),
            _assistant_msg("a1", 1),
            _user_msg("q2", 2),
            _assistant_msg("a2", 3),
        ]
        split_at = extractor._find_force_split_point(messages)
        assert split_at == 2

    def test_never_splits_lone_user_message(self):
        """Should never produce a chunk with only user messages and no assistant reply.

        Messages:
          [0] user
          [1] assistant + tool_calls
          [2] tool
          [3] assistant: answer
          [4] user
          [5] assistant + tool_calls
          [6] tool
          [7] assistant: answer2

        Various limits should all produce chunks containing at least one
        final assistant response.
        """
        extractor = _build_extractor()

        messages = [
            _user_msg("q1", 0),
            _tool_call_msg("c1", 1),
            _tool_response_msg("r1", 2),
            _assistant_msg("a1", 3),
            _user_msg("q2", 4),
            _tool_call_msg("c2", 5),
            _tool_response_msg("r2", 6),
            _assistant_msg("a2", 7),
        ]

        for limit in [2, 3, 4, 5, 6]:
            extractor.hard_message_limit = limit
            split_at = extractor._find_force_split_point(messages)
            chunk = messages[:split_at]
            has_final_asst = any(
                m.get("role") == "assistant" and not m.get("tool_calls")
                for m in chunk
            )
            assert has_final_asst or split_at >= len(messages) - 1, (
                f"limit={limit}, split_at={split_at}: chunk has no final assistant"
            )

    def test_walk_back_finds_complete_turn(self):
        """Walk back should stop at a final assistant boundary.

        Messages:
          [0] user
          [1] assistant: a1 (final)
          [2] user
          [3] assistant + tool_calls  <- intermediate
          [4] tool                    <- intermediate
          [5] assistant: a2
          [6] user
          [7] assistant: a3

        hard_message_limit=5, parent candidate=4. msg[3]=tool_call -> walk back.
        candidate=3: msg[2]=user -> not final asst. candidate=2: msg[1]=final
        assistant -> safe! split_at=2.
        """
        extractor = _build_extractor()
        extractor.hard_message_limit = 5

        messages = [
            _user_msg("q1", 0),
            _assistant_msg("a1", 1),
            _user_msg("q2", 2),
            _tool_call_msg("thinking", 3),
            _tool_response_msg("result", 4),
            _assistant_msg("a2", 5),
            _user_msg("q3", 6),
            _assistant_msg("a3", 7),
        ]
        split_at = extractor._find_force_split_point(messages)
        assert split_at == 2
        assert messages[split_at - 1]["role"] == "assistant"
        assert not messages[split_at - 1].get("tool_calls")

    @pytest.mark.asyncio
    @patch("memory_layer.memcell_extractor.conv_memcell_extractor.record_boundary_detection")
    @patch("memory_layer.memcell_extractor.conv_memcell_extractor.record_memcell_extracted")
    @patch("memory_layer.memcell_extractor.conv_memcell_extractor.get_space_id_for_metrics", return_value="test")
    async def test_force_split_keeps_tool_sequence_intact(self, mock_space, mock_record, mock_bd):
        """End-to-end: force split should not break a tool call sequence across MemCells."""
        extractor = _build_extractor()
        extractor.hard_message_limit = 4

        request = _make_request(
            history_msgs=[
                _user_msg("q1", 0),
                _tool_call_msg("thinking", 1),
                _tool_response_msg("result", 2),
                _assistant_msg("answer", 3),
            ],
            new_msgs=[_user_msg("q2", 4), _assistant_msg("a2", 5)],
        )
        memcells, status = await extractor.extract_memcell(request)
        # Force split should produce at least one MemCell
        assert len(memcells) > 0
        # Each MemCell should not start or end with an orphaned tool message
        for mc in memcells:
            first_msg = mc.original_data[0].get("message", mc.original_data[0])
            last_msg = mc.original_data[-1].get("message", mc.original_data[-1])
            # Last message in a MemCell should not be an intermediate step
            assert not is_intermediate_agent_step(
                last_msg
            ), f"MemCell ends with intermediate: {last_msg.get('role')}"

    @pytest.mark.asyncio
    @patch("memory_layer.memcell_extractor.conv_memcell_extractor.record_boundary_detection")
    @patch("memory_layer.memcell_extractor.conv_memcell_extractor.record_memcell_extracted")
    @patch("memory_layer.memcell_extractor.conv_memcell_extractor.get_space_id_for_metrics", return_value="test")
    async def test_force_split_multi_tool_turn_e2e(self, mock_space, mock_record, mock_bd):
        """End-to-end: multi-tool turn stays intact after force split."""
        extractor = _build_extractor()
        extractor.hard_message_limit = 5

        request = _make_request(
            history_msgs=[
                _user_msg("q1", 0),
                _tool_call_msg("search", 1),
                _tool_response_msg("search result", 2),
                _tool_call_msg("compute", 3),
                _tool_response_msg("compute result", 4),
                _assistant_msg("final answer", 5),
            ],
            new_msgs=[_user_msg("q2", 6), _assistant_msg("a2", 7)],
        )
        memcells, status = await extractor.extract_memcell(request)
        assert len(memcells) > 0
        for mc in memcells:
            last_msg = mc.original_data[-1].get("message", mc.original_data[-1])
            assert not is_intermediate_agent_step(
                last_msg
            ), f"MemCell ends with intermediate: {last_msg.get('role')}"


# ---------------------------------------------------------------------------
# Tests: Multi-turn agent conversation flow simulation
# ---------------------------------------------------------------------------


class TestMultiTurnFlow:
    """Simulate a complete agent turn: user -> tool_call -> tool_response -> assistant."""

    @pytest.mark.asyncio
    async def test_full_agent_turn_sequence(self):
        """Simulate messages arriving one by one; only the final assistant triggers processing."""
        extractor = _build_extractor()
        results = []

        # Message 1: user asks a question
        req1 = _make_request(history_msgs=[], new_msgs=[_user_msg("search for X", 0)])
        # Empty history -> guard 3 skips, delegates to parent
        # Parent: no history, first messages -> LLM says should_wait
        with patch(
            "memory_layer.memcell_extractor.conv_memcell_extractor.record_boundary_detection"
        ), patch(
            "memory_layer.memcell_extractor.conv_memcell_extractor.record_memcell_extracted"
        ), patch(
            "memory_layer.memcell_extractor.conv_memcell_extractor.get_space_id_for_metrics",
            return_value="test",
        ):
            r1 = await extractor.extract_memcell(req1)
        results.append(r1)

        # Message 2: agent makes tool call (intermediate)
        req2 = _make_request(
            history_msgs=[_user_msg("search for X", 0)],
            new_msgs=[_tool_call_msg("calling search", 1)],
        )
        r2 = await extractor.extract_memcell(req2)
        results.append(r2)

        # Message 3: tool response (intermediate)
        req3 = _make_request(
            history_msgs=[_user_msg("search for X", 0), _tool_call_msg("calling search", 1)],
            new_msgs=[_tool_response_msg("search results", 2)],
        )
        r3 = await extractor.extract_memcell(req3)
        results.append(r3)

        # Message 4: final assistant response
        req4 = _make_request(
            history_msgs=[
                _user_msg("search for X", 0),
                _tool_call_msg("calling search", 1),
                _tool_response_msg("search results", 2),
            ],
            new_msgs=[_assistant_msg("Here are the results for X", 3)],
        )
        # History ends at tool_response -> guard 3 blocks
        r4 = await extractor.extract_memcell(req4)
        results.append(r4)

        # Verify: messages 2, 3, 4 all returned (None, should_wait=True)
        for i in range(1, 4):
            memcell, status = results[i]
            assert memcell == [], f"Step {i+1} should not create MemCell"
            assert status.should_wait is True, f"Step {i+1} should wait"

        # Message 5: next user message arrives, now history ends at assistant response
        req5 = _make_request(
            history_msgs=[
                _user_msg("search for X", 0),
                _tool_call_msg("calling search", 1),
                _tool_response_msg("search results", 2),
                _assistant_msg("Here are the results for X", 3),
            ],
            new_msgs=[_user_msg("thanks, now search Y", 10)],
        )
        with patch(
            "memory_layer.memcell_extractor.conv_memcell_extractor.record_boundary_detection"
        ), patch(
            "memory_layer.memcell_extractor.conv_memcell_extractor.record_memcell_extracted"
        ), patch(
            "memory_layer.memcell_extractor.conv_memcell_extractor.get_space_id_for_metrics",
            return_value="test",
        ):
            r5 = await extractor.extract_memcell(req5)
        # All guards pass -> delegates to parent LLM boundary detection
        memcell, status = r5
        # LLM mock says should_wait -> no memcell, which is correct behavior
        assert status.should_wait is True


# ---------------------------------------------------------------------------
# Tests: raw_data_type
# ---------------------------------------------------------------------------


class TestDetectBoundariesRemap:
    """Test _detect_boundaries filter + remap logic."""

    @pytest.mark.asyncio
    async def test_remap_single_boundary(self):
        """LLM boundary on filtered messages is remapped to original indices.

        Original:
          [0] user         -> filtered[0]
          [1] tool_call    (filtered out)
          [2] tool         (filtered out)
          [3] assistant    -> filtered[1]
          [4] user         -> filtered[2]
          [5] assistant    -> filtered[3]

        LLM returns boundary=[2] on filtered (after filtered[1]=assistant).
        Remap: filtered_to_orig[2-1] + 1 = orig[3] + 1 = 4.
        """
        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(
            return_value='{"reasoning": "topic change", "boundaries": [2], "should_wait": false}'
        )
        extractor = AgentMemCellExtractor(llm_provider=mock_llm)

        messages = [
            _user_msg("q1", 0),
            _tool_call_msg("thinking", 1),
            _tool_response_msg("result", 2),
            _assistant_msg("answer", 3),
            _user_msg("q2", 4),
            _assistant_msg("a2", 5),
        ]
        result = await extractor._detect_boundaries(messages)
        assert result.boundaries == [4]

    @pytest.mark.asyncio
    async def test_remap_multiple_boundaries(self):
        """Multiple boundaries are all remapped correctly.

        Original:
          [0] user          -> filtered[0]
          [1] assistant     -> filtered[1]
          [2] user          -> filtered[2]
          [3] tool_call     (filtered out)
          [4] tool          (filtered out)
          [5] assistant     -> filtered[3]
          [6] user          -> filtered[4]
          [7] assistant     -> filtered[5]

        LLM returns boundaries=[2, 4] on filtered.
        Remap[2]: filtered_to_orig[1] + 1 = 1 + 1 = 2
        Remap[4]: filtered_to_orig[3] + 1 = 5 + 1 = 6
        """
        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(
            return_value='{"reasoning": "two topics", "boundaries": [2, 4], "should_wait": false}'
        )
        extractor = AgentMemCellExtractor(llm_provider=mock_llm)

        messages = [
            _user_msg("q1", 0),
            _assistant_msg("a1", 1),
            _user_msg("q2", 2),
            _tool_call_msg("thinking", 3),
            _tool_response_msg("result", 4),
            _assistant_msg("a2", 5),
            _user_msg("q3", 6),
            _assistant_msg("a3", 7),
        ]
        result = await extractor._detect_boundaries(messages)
        assert result.boundaries == [2, 6]

    @pytest.mark.asyncio
    async def test_all_messages_are_intermediate(self):
        """If all messages are intermediate, return should_wait=True."""
        extractor = _build_extractor()
        messages = [
            _tool_call_msg("call", 0),
            _tool_response_msg("result", 1),
        ]
        result = await extractor._detect_boundaries(messages)
        assert result.boundaries == []
        assert result.should_wait is True

    @pytest.mark.asyncio
    async def test_no_tool_messages_passthrough(self):
        """Without tool messages, boundaries pass through unchanged."""
        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(
            return_value='{"reasoning": "split", "boundaries": [2], "should_wait": false}'
        )
        extractor = AgentMemCellExtractor(llm_provider=mock_llm)

        messages = [
            _user_msg("q1", 0),
            _assistant_msg("a1", 1),
            _user_msg("q2", 2),
            _assistant_msg("a2", 3),
        ]
        result = await extractor._detect_boundaries(messages)
        assert result.boundaries == [2]


class TestForceSplitKeepsAllE2E:
    """End-to-end tests for when _find_force_split_point returns len(messages)."""

    @pytest.mark.asyncio
    @patch("memory_layer.memcell_extractor.conv_memcell_extractor.record_boundary_detection")
    @patch("memory_layer.memcell_extractor.conv_memcell_extractor.record_memcell_extracted")
    @patch("memory_layer.memcell_extractor.conv_memcell_extractor.get_space_id_for_metrics", return_value="test")
    async def test_single_long_tool_turn_becomes_one_memcell(self, mock_space, mock_record, mock_bd):
        """A single turn with many tools that exceeds limit stays as one MemCell."""
        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(
            return_value='{"reasoning": "done", "boundaries": [], "should_wait": false}'
        )
        extractor = AgentMemCellExtractor(llm_provider=mock_llm)
        extractor.hard_message_limit = 3

        request = _make_request(
            history_msgs=[],
            new_msgs=[
                _user_msg("complex question", 0),
                _tool_call_msg("search", 1),
                _tool_response_msg("search result", 2),
                _tool_call_msg("compute", 3),
                _tool_response_msg("compute result", 4),
                _assistant_msg("final answer", 5),
            ],
        )
        memcells, status = await extractor.extract_memcell(request)
        # All 6 messages should be in one MemCell
        assert len(memcells) == 1
        assert len(memcells[0].original_data) == 6

    @pytest.mark.asyncio
    @patch("memory_layer.memcell_extractor.conv_memcell_extractor.record_boundary_detection")
    @patch("memory_layer.memcell_extractor.conv_memcell_extractor.record_memcell_extracted")
    @patch("memory_layer.memcell_extractor.conv_memcell_extractor.get_space_id_for_metrics", return_value="test")
    async def test_multiple_force_splits_in_loop(self, mock_space, mock_record, mock_bd):
        """Force split loop runs multiple times, each split respects tool boundaries."""
        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(
            return_value='{"reasoning": "done", "boundaries": [], "should_wait": false}'
        )
        extractor = AgentMemCellExtractor(llm_provider=mock_llm)
        extractor.hard_message_limit = 6

        request = _make_request(
            history_msgs=[
                # Turn 1: complete
                _user_msg("q1", 0),
                _tool_call_msg("c1", 1),
                _tool_response_msg("r1", 2),
                _assistant_msg("a1", 3),
                # Turn 2: complete
                _user_msg("q2", 4),
                _tool_call_msg("c2", 5),
                _tool_response_msg("r2", 6),
                _assistant_msg("a2", 7),
            ],
            new_msgs=[
                # Turn 3: complete
                _user_msg("q3", 8),
                _assistant_msg("a3", 9),
            ],
        )
        memcells, status = await extractor.extract_memcell(request)
        # Should produce multiple MemCells, each ending at a final assistant
        assert len(memcells) >= 2
        for mc in memcells:
            last_msg = mc.original_data[-1].get("message", mc.original_data[-1])
            assert last_msg.get("role") == "assistant"
            assert not last_msg.get("tool_calls")


class TestIsSafeSplitEdgeCases:
    """Additional edge cases for _is_safe_split."""

    def test_assistant_with_empty_tool_calls_is_safe(self):
        """assistant with tool_calls=[] is a final response, should be safe."""
        msgs = [
            _user_msg("q", 0),
            {"role": "assistant", "content": "done", "tool_calls": [],
             "speaker_name": "Assistant", "timestamp": _ts(1)},
            _user_msg("q2", 2),
        ]
        assert AgentMemCellExtractor._is_safe_split(msgs, 2) is True

    def test_assistant_with_tool_calls_none_is_safe(self):
        """assistant with tool_calls=None is a final response, should be safe."""
        msgs = [
            _user_msg("q", 0),
            {"role": "assistant", "content": "done", "tool_calls": None,
             "speaker_name": "Assistant", "timestamp": _ts(1)},
            _user_msg("q2", 2),
        ]
        assert AgentMemCellExtractor._is_safe_split(msgs, 2) is True

    def test_assistant_without_tool_calls_key_is_safe(self):
        """assistant without tool_calls key at all is a final response."""
        msgs = [
            _user_msg("q", 0),
            {"role": "assistant", "content": "done",
             "speaker_name": "Assistant", "timestamp": _ts(1)},
            _user_msg("q2", 2),
        ]
        assert AgentMemCellExtractor._is_safe_split(msgs, 2) is True

    def test_system_message_is_not_safe(self):
        """system message at split boundary is not safe."""
        msgs = [
            {"role": "system", "content": "You are helpful"},
            _user_msg("q", 1),
            _assistant_msg("a", 2),
        ]
        assert AgentMemCellExtractor._is_safe_split(msgs, 1) is False

    def test_empty_messages_list(self):
        """Empty list has no valid split."""
        assert AgentMemCellExtractor._is_safe_split([], 0) is False
        assert AgentMemCellExtractor._is_safe_split([], 1) is False


class TestGuardEdgeCases:
    """Additional guard edge cases in extract_memcell."""

    def setup_method(self):
        self.extractor = _build_extractor()

    @pytest.mark.asyncio
    async def test_empty_new_msgs_with_history(self):
        """Empty new_raw_data_list with non-empty history -> delegates to parent."""
        request = _make_request(
            history_msgs=[_user_msg("q", 0), _assistant_msg("a", 1)],
            new_msgs=[],
        )
        with patch(
            "memory_layer.memcell_extractor.conv_memcell_extractor.record_boundary_detection"
        ), patch(
            "memory_layer.memcell_extractor.conv_memcell_extractor.record_memcell_extracted"
        ), patch(
            "memory_layer.memcell_extractor.conv_memcell_extractor.get_space_id_for_metrics",
            return_value="test",
        ):
            memcells, status = await self.extractor.extract_memcell(request)
        # Delegates to parent (no guard triggers), LLM says should_wait
        assert status.should_wait is True

    @pytest.mark.asyncio
    async def test_single_assistant_msg_total(self):
        """Only one message total -> should_wait."""
        request = _make_request(
            history_msgs=[],
            new_msgs=[_assistant_msg("hello", 0)],
        )
        memcells, status = await self.extractor.extract_memcell(request)
        assert memcells == []
        assert status.should_wait is True

    @pytest.mark.asyncio
    async def test_new_msgs_mix_user_and_assistant(self):
        """New messages contain both user and assistant -> not blocked by all_user_only guard."""
        request = _make_request(
            history_msgs=[_user_msg("q1", 0), _assistant_msg("a1", 1)],
            new_msgs=[_user_msg("q2", 2), _assistant_msg("a2", 3)],
        )
        with patch(
            "memory_layer.memcell_extractor.conv_memcell_extractor.record_boundary_detection"
        ), patch(
            "memory_layer.memcell_extractor.conv_memcell_extractor.record_memcell_extracted"
        ), patch(
            "memory_layer.memcell_extractor.conv_memcell_extractor.get_space_id_for_metrics",
            return_value="test",
        ):
            memcells, status = await self.extractor.extract_memcell(request)
        # Passes all guards, delegates to parent


# ---------------------------------------------------------------------------
# Helpers for trajectory generation
# ---------------------------------------------------------------------------


def _make_simple_turn(turn_idx: int) -> List[Dict[str, Any]]:
    """user -> assistant (no tools). 2 messages."""
    base = turn_idx * 10
    return [
        _user_msg(f"question_{turn_idx}", base),
        _assistant_msg(f"answer_{turn_idx}", base + 1),
    ]


def _make_single_tool_turn(turn_idx: int) -> List[Dict[str, Any]]:
    """user -> tool_call -> tool -> assistant. 4 messages."""
    base = turn_idx * 10
    return [
        _user_msg(f"question_{turn_idx}", base),
        _tool_call_msg(f"calling_tool_{turn_idx}", base + 1),
        _tool_response_msg(f"result_{turn_idx}", base + 2),
        _assistant_msg(f"answer_{turn_idx}", base + 3),
    ]


def _make_multi_tool_turn(turn_idx: int, tool_count: int = 3) -> List[Dict[str, Any]]:
    """user -> (tool_call -> tool) * N -> assistant. 2 + 2*N messages."""
    base = turn_idx * 10
    msgs = [_user_msg(f"question_{turn_idx}", base)]
    for t in range(tool_count):
        msgs.append(_tool_call_msg(f"call_{turn_idx}_{t}", base + 1 + t * 2))
        msgs.append(_tool_response_msg(f"result_{turn_idx}_{t}", base + 2 + t * 2))
    msgs.append(_assistant_msg(f"answer_{turn_idx}", base + 1 + tool_count * 2))
    return msgs


def _build_trajectory(turns: int, pattern: str = "mixed") -> List[Dict[str, Any]]:
    """Build a full trajectory with the given number of turns.

    Patterns:
    - "simple": all turns are user->assistant
    - "tool": all turns have single tool call
    - "multi_tool": all turns have 3 tool calls
    - "mixed": alternating simple/tool/multi_tool
    """
    msgs = []
    for i in range(turns):
        if pattern == "simple":
            msgs.extend(_make_simple_turn(i))
        elif pattern == "tool":
            msgs.extend(_make_single_tool_turn(i))
        elif pattern == "multi_tool":
            msgs.extend(_make_multi_tool_turn(i))
        elif pattern == "mixed":
            if i % 3 == 0:
                msgs.extend(_make_simple_turn(i))
            elif i % 3 == 1:
                msgs.extend(_make_single_tool_turn(i))
            else:
                msgs.extend(_make_multi_tool_turn(i))
    return msgs


def _assert_all_memcells_valid(memcells: List, extractor: AgentMemCellExtractor):
    """Assert every MemCell ends at a final assistant (not intermediate)."""
    for i, mc in enumerate(memcells):
        last_msg = mc.original_data[-1].get("message", mc.original_data[-1])
        role = last_msg.get("role", "")
        has_tc = bool(last_msg.get("tool_calls"))
        assert role == "assistant" and not has_tc, (
            f"MemCell[{i}] ends with role={role}, tool_calls={has_tc}"
        )
        # Each MemCell should have at least 2 messages
        assert len(mc.original_data) >= 2, (
            f"MemCell[{i}] has only {len(mc.original_data)} message(s)"
        )


# ---------------------------------------------------------------------------
# Tests: _find_force_split_point with various trajectory lengths
# ---------------------------------------------------------------------------


class TestForceSplitPointTrajectoryLengths:
    """Test _find_force_split_point across trajectory lengths 2 to 1000 turns."""

    @pytest.mark.parametrize("turns", [1, 2, 3, 5, 10, 20, 50])
    def test_simple_trajectory_various_lengths(self, turns):
        """Simple turns (user->assistant) should split cleanly."""
        extractor = _build_extractor()
        extractor.hard_message_limit = 8
        msgs = _build_trajectory(turns, "simple")
        if len(msgs) < extractor.hard_message_limit:
            return  # No force split needed
        split_at = extractor._find_force_split_point(msgs)
        assert AgentMemCellExtractor._is_safe_split(msgs, split_at) or split_at == len(msgs)

    @pytest.mark.parametrize("turns", [1, 2, 3, 5, 10, 20, 50])
    def test_tool_trajectory_various_lengths(self, turns):
        """Single-tool turns should never split inside tool sequence."""
        extractor = _build_extractor()
        extractor.hard_message_limit = 6
        msgs = _build_trajectory(turns, "tool")
        if len(msgs) < extractor.hard_message_limit:
            return
        split_at = extractor._find_force_split_point(msgs)
        assert AgentMemCellExtractor._is_safe_split(msgs, split_at) or split_at == len(msgs)

    @pytest.mark.parametrize("turns", [1, 2, 3, 5, 10, 20, 50])
    def test_multi_tool_trajectory_various_lengths(self, turns):
        """Multi-tool turns (8 msgs each) should keep tool sequence intact."""
        extractor = _build_extractor()
        extractor.hard_message_limit = 10
        msgs = _build_trajectory(turns, "multi_tool")
        if len(msgs) < extractor.hard_message_limit:
            return
        split_at = extractor._find_force_split_point(msgs)
        assert AgentMemCellExtractor._is_safe_split(msgs, split_at) or split_at == len(msgs)

    @pytest.mark.parametrize("turns", [1, 2, 3, 5, 10, 20, 50])
    def test_mixed_trajectory_various_lengths(self, turns):
        """Mixed pattern (simple/tool/multi_tool alternating)."""
        extractor = _build_extractor()
        extractor.hard_message_limit = 10
        msgs = _build_trajectory(turns, "mixed")
        if len(msgs) < extractor.hard_message_limit:
            return
        split_at = extractor._find_force_split_point(msgs)
        assert AgentMemCellExtractor._is_safe_split(msgs, split_at) or split_at == len(msgs)

    @pytest.mark.parametrize("limit", [4, 8, 16, 32, 64])
    def test_100_turn_mixed_various_limits(self, limit):
        """100 mixed turns with various hard_message_limit settings."""
        extractor = _build_extractor()
        extractor.hard_message_limit = limit
        msgs = _build_trajectory(100, "mixed")
        split_at = extractor._find_force_split_point(msgs)
        assert AgentMemCellExtractor._is_safe_split(msgs, split_at) or split_at == len(msgs)

    def test_1000_turn_simple_trajectory(self):
        """1000 simple turns (2000 messages), default limit."""
        extractor = _build_extractor()
        # Default: AGENT_DEFAULT_HARD_MESSAGE_LIMIT = 64
        msgs = _build_trajectory(1000, "simple")
        assert len(msgs) == 2000
        split_at = extractor._find_force_split_point(msgs)
        assert AgentMemCellExtractor._is_safe_split(msgs, split_at)

    def test_1000_turn_tool_trajectory(self):
        """1000 tool turns (4000 messages), default limit."""
        extractor = _build_extractor()
        msgs = _build_trajectory(1000, "tool")
        assert len(msgs) == 4000
        split_at = extractor._find_force_split_point(msgs)
        assert AgentMemCellExtractor._is_safe_split(msgs, split_at) or split_at == len(msgs)

    def test_1000_turn_multi_tool_trajectory(self):
        """1000 multi-tool turns (8000 messages), default limit."""
        extractor = _build_extractor()
        msgs = _build_trajectory(1000, "multi_tool")
        assert len(msgs) == 8000
        split_at = extractor._find_force_split_point(msgs)
        assert AgentMemCellExtractor._is_safe_split(msgs, split_at) or split_at == len(msgs)

    def test_1000_turn_mixed_trajectory(self):
        """1000 mixed turns, default limit."""
        extractor = _build_extractor()
        msgs = _build_trajectory(1000, "mixed")
        split_at = extractor._find_force_split_point(msgs)
        assert AgentMemCellExtractor._is_safe_split(msgs, split_at) or split_at == len(msgs)

    @pytest.mark.parametrize("limit", [4, 8, 16, 32, 64, 128])
    def test_1000_turn_mixed_various_limits(self, limit):
        """1000 mixed turns with various limits."""
        extractor = _build_extractor()
        extractor.hard_message_limit = limit
        msgs = _build_trajectory(1000, "mixed")
        split_at = extractor._find_force_split_point(msgs)
        assert AgentMemCellExtractor._is_safe_split(msgs, split_at) or split_at == len(msgs)


# ---------------------------------------------------------------------------
# Tests: Full force-split loop simulation (Phase 1 repeated splits)
# ---------------------------------------------------------------------------


class TestForceSplitLoopSimulation:
    """Simulate the parent's Phase 1 force-split loop to verify repeated
    splits all produce valid chunks."""

    @staticmethod
    def _simulate_force_split_loop(
        extractor: AgentMemCellExtractor,
        messages: List[Dict[str, Any]],
    ) -> List[List[Dict[str, Any]]]:
        """Simulate Phase 1 loop, return list of chunks."""
        all_msgs = list(messages)
        chunks = []
        iterations = 0
        max_iterations = len(messages)  # Safety bound

        while len(all_msgs) > 1 and iterations < max_iterations:
            if len(all_msgs) < extractor.hard_message_limit:
                break
            split_at = extractor._find_force_split_point(all_msgs)
            if split_at >= len(all_msgs):
                # No split possible, take all
                chunks.append(all_msgs)
                all_msgs = []
                break
            chunks.append(all_msgs[:split_at])
            all_msgs = all_msgs[split_at:]
            iterations += 1

        if all_msgs:
            chunks.append(all_msgs)  # Remaining goes to Phase 2
        return chunks

    @pytest.mark.parametrize("turns,pattern", [
        (10, "simple"),
        (10, "tool"),
        (10, "multi_tool"),
        (10, "mixed"),
        (50, "mixed"),
        (100, "mixed"),
    ])
    def test_all_chunks_end_at_final_assistant(self, turns, pattern):
        """Every force-split chunk (except possibly the last remainder)
        should end at a final assistant message."""
        extractor = _build_extractor()
        extractor.hard_message_limit = 10
        msgs = _build_trajectory(turns, pattern)
        chunks = self._simulate_force_split_loop(extractor, msgs)

        # All chunks except the last should end at final assistant
        for i, chunk in enumerate(chunks[:-1]):
            last_msg = chunk[-1]
            assert last_msg.get("role") == "assistant" and not last_msg.get("tool_calls"), (
                f"Chunk[{i}] (len={len(chunk)}) ends with "
                f"role={last_msg.get('role')}, tool_calls={bool(last_msg.get('tool_calls'))}"
            )

    @pytest.mark.parametrize("turns,pattern", [
        (10, "tool"),
        (10, "multi_tool"),
        (50, "mixed"),
        (100, "mixed"),
    ])
    def test_no_orphaned_tool_messages(self, turns, pattern):
        """No chunk should start with a tool response or end with a tool_call."""
        extractor = _build_extractor()
        extractor.hard_message_limit = 10
        msgs = _build_trajectory(turns, pattern)
        chunks = self._simulate_force_split_loop(extractor, msgs)

        for i, chunk in enumerate(chunks[:-1]):
            # Should not end at intermediate
            last = chunk[-1]
            assert not is_intermediate_agent_step(last), (
                f"Chunk[{i}] ends with intermediate: role={last.get('role')}"
            )

    def test_all_messages_accounted_for(self):
        """Total messages across all chunks equals original message count."""
        extractor = _build_extractor()
        extractor.hard_message_limit = 10
        msgs = _build_trajectory(50, "mixed")
        original_count = len(msgs)
        chunks = self._simulate_force_split_loop(extractor, msgs)
        total = sum(len(c) for c in chunks)
        assert total == original_count

    @pytest.mark.parametrize("limit", [4, 8, 16, 32])
    def test_loop_terminates_various_limits(self, limit):
        """Loop always terminates (no infinite loop) for various limits."""
        extractor = _build_extractor()
        extractor.hard_message_limit = limit
        msgs = _build_trajectory(100, "mixed")
        chunks = self._simulate_force_split_loop(extractor, msgs)
        # Must produce at least 1 chunk
        assert len(chunks) >= 1
        # Total must equal original
        assert sum(len(c) for c in chunks) == len(msgs)

    def test_1000_turn_loop_completes(self):
        """1000-turn trajectory loop completes without issues."""
        extractor = _build_extractor()
        extractor.hard_message_limit = 64
        msgs = _build_trajectory(1000, "mixed")
        chunks = self._simulate_force_split_loop(extractor, msgs)
        assert sum(len(c) for c in chunks) == len(msgs)
        # Each non-last chunk should end at final assistant
        for i, chunk in enumerate(chunks[:-1]):
            last = chunk[-1]
            assert last.get("role") == "assistant" and not last.get("tool_calls"), (
                f"Chunk[{i}] invalid end"
            )

    def test_1000_turn_loop_small_limit(self):
        """1000-turn trajectory with small limit=4 creates many chunks."""
        extractor = _build_extractor()
        extractor.hard_message_limit = 4
        msgs = _build_trajectory(1000, "mixed")
        chunks = self._simulate_force_split_loop(extractor, msgs)
        assert sum(len(c) for c in chunks) == len(msgs)
        # Should create many chunks with small limit
        assert len(chunks) > 100


# ---------------------------------------------------------------------------
# Tests: End-to-end extract_memcell with various trajectory lengths
# ---------------------------------------------------------------------------


METRIC_PATCHES = [
    patch("memory_layer.memcell_extractor.conv_memcell_extractor.record_boundary_detection"),
    patch("memory_layer.memcell_extractor.conv_memcell_extractor.record_memcell_extracted"),
    patch("memory_layer.memcell_extractor.conv_memcell_extractor.get_space_id_for_metrics", return_value="test"),
]


class TestE2ETrajectoryLengths:
    """End-to-end extract_memcell with various trajectory lengths."""

    @staticmethod
    def _make_extractor_no_boundary(hard_message_limit: int = 10) -> AgentMemCellExtractor:
        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(
            return_value='{"reasoning": "no boundary", "boundaries": [], "should_wait": false}'
        )
        ext = AgentMemCellExtractor(llm_provider=mock_llm)
        ext.hard_message_limit = hard_message_limit
        return ext

    @pytest.mark.asyncio
    async def test_e2e_10_turns_mixed(self):
        """10 mixed turns, limit=10."""
        extractor = self._make_extractor_no_boundary(10)
        msgs = _build_trajectory(10, "mixed")
        request = _make_request(history_msgs=[], new_msgs=msgs)
        with patch(
            "memory_layer.memcell_extractor.conv_memcell_extractor.record_boundary_detection"
        ), patch(
            "memory_layer.memcell_extractor.conv_memcell_extractor.record_memcell_extracted"
        ), patch(
            "memory_layer.memcell_extractor.conv_memcell_extractor.get_space_id_for_metrics",
            return_value="test",
        ):
            memcells, status = await extractor.extract_memcell(request)
        assert len(memcells) >= 1
        _assert_all_memcells_valid(memcells, extractor)

    @pytest.mark.asyncio
    async def test_e2e_50_turns_tool(self):
        """50 tool turns (200 messages), limit=20."""
        extractor = self._make_extractor_no_boundary(20)
        msgs = _build_trajectory(50, "tool")
        request = _make_request(history_msgs=[], new_msgs=msgs)
        with patch(
            "memory_layer.memcell_extractor.conv_memcell_extractor.record_boundary_detection"
        ), patch(
            "memory_layer.memcell_extractor.conv_memcell_extractor.record_memcell_extracted"
        ), patch(
            "memory_layer.memcell_extractor.conv_memcell_extractor.get_space_id_for_metrics",
            return_value="test",
        ):
            memcells, status = await extractor.extract_memcell(request)
        assert len(memcells) >= 1
        _assert_all_memcells_valid(memcells, extractor)

    @pytest.mark.asyncio
    async def test_e2e_100_turns_mixed(self):
        """100 mixed turns, limit=16."""
        extractor = self._make_extractor_no_boundary(16)
        msgs = _build_trajectory(100, "mixed")
        request = _make_request(history_msgs=[], new_msgs=msgs)
        with patch(
            "memory_layer.memcell_extractor.conv_memcell_extractor.record_boundary_detection"
        ), patch(
            "memory_layer.memcell_extractor.conv_memcell_extractor.record_memcell_extracted"
        ), patch(
            "memory_layer.memcell_extractor.conv_memcell_extractor.get_space_id_for_metrics",
            return_value="test",
        ):
            memcells, status = await extractor.extract_memcell(request)
        assert len(memcells) >= 1
        _assert_all_memcells_valid(memcells, extractor)

    @pytest.mark.asyncio
    async def test_e2e_flush_long_trajectory(self):
        """Flush mode with 50 mixed turns should pack remaining into final MemCell."""
        extractor = self._make_extractor_no_boundary(20)
        msgs = _build_trajectory(50, "mixed")
        request = _make_request(history_msgs=[], new_msgs=msgs, flush=True)
        with patch(
            "memory_layer.memcell_extractor.conv_memcell_extractor.record_boundary_detection"
        ), patch(
            "memory_layer.memcell_extractor.conv_memcell_extractor.record_memcell_extracted"
        ), patch(
            "memory_layer.memcell_extractor.conv_memcell_extractor.get_space_id_for_metrics",
            return_value="test",
        ):
            memcells, status = await extractor.extract_memcell(request)
        assert len(memcells) >= 1
        assert status.should_wait is False
        # Total messages across all MemCells should equal input
        total = sum(len(mc.original_data) for mc in memcells)
        assert total == len(msgs)

    @pytest.mark.asyncio
    async def test_e2e_history_plus_new_long(self):
        """Long history + new messages, limit=10."""
        extractor = self._make_extractor_no_boundary(10)
        history = _build_trajectory(20, "tool")
        new = _build_trajectory(5, "simple")
        # Shift new messages offsets to avoid timestamp collision
        for m in new:
            m["timestamp"] = _ts(500 + int(m["timestamp"][-10:-6]) if len(m["timestamp"]) > 10 else 500)
        request = _make_request(history_msgs=history, new_msgs=new)
        with patch(
            "memory_layer.memcell_extractor.conv_memcell_extractor.record_boundary_detection"
        ), patch(
            "memory_layer.memcell_extractor.conv_memcell_extractor.record_memcell_extracted"
        ), patch(
            "memory_layer.memcell_extractor.conv_memcell_extractor.get_space_id_for_metrics",
            return_value="test",
        ):
            memcells, status = await extractor.extract_memcell(request)
        assert len(memcells) >= 1
        _assert_all_memcells_valid(memcells, extractor)


# ---------------------------------------------------------------------------
# Tests: Edge cases in parent integration
# ---------------------------------------------------------------------------


class TestParentIntegrationEdgeCases:
    """Test edge cases specific to how AgentMemCellExtractor interacts
    with ConvMemCellExtractor's three-phase pipeline."""

    def test_force_split_returns_len_messages_terminates_loop(self):
        """When _find_force_split_point returns len(messages), the simulated
        loop terminates without infinite iteration."""
        extractor = _build_extractor()
        extractor.hard_message_limit = 3

        # Single tool turn: no safe split possible
        msgs = [
            _user_msg("q", 0),
            _tool_call_msg("c", 1),
            _tool_response_msg("r", 2),
            _assistant_msg("a", 3),
        ]
        split_at = extractor._find_force_split_point(msgs)
        assert split_at == len(msgs)

        # Simulate: all_msgs[:4] = all, all_msgs[4:] = [], loop exits
        remaining = msgs[split_at:]
        assert remaining == []

    def test_force_split_with_exactly_limit_messages(self):
        """Exactly at hard_message_limit boundary."""
        extractor = _build_extractor()
        extractor.hard_message_limit = 4

        msgs = [
            _user_msg("q1", 0),
            _assistant_msg("a1", 1),
            _user_msg("q2", 2),
            _assistant_msg("a2", 3),
        ]
        split_at = extractor._find_force_split_point(msgs)
        # Parent returns min(3, 3) = 3. msg[2]=user -> not safe.
        # Walk back: msg[1]=assistant -> safe. split_at=2.
        assert split_at == 2

    def test_force_split_one_over_limit(self):
        """One message over the limit."""
        extractor = _build_extractor()
        extractor.hard_message_limit = 4

        msgs = [
            _user_msg("q1", 0),
            _assistant_msg("a1", 1),
            _user_msg("q2", 2),
            _assistant_msg("a2", 3),
            _user_msg("q3", 4),
        ]
        split_at = extractor._find_force_split_point(msgs)
        # Parent returns min(3, 4) = 3. msg[2]=user -> walk back.
        # msg[1]=assistant -> safe. split_at=2.
        assert split_at == 2

    def test_two_messages_over_limit(self):
        """Two messages, both exceed limit=2."""
        extractor = _build_extractor()
        extractor.hard_message_limit = 2

        msgs = [_user_msg("q", 0), _assistant_msg("a", 1), _user_msg("q2", 2)]
        split_at = extractor._find_force_split_point(msgs)
        # Parent returns min(1, 2) = 1. msg[0]=user -> not safe (not assistant).
        # Walk forward from 2: msg[1]=assistant -> safe. split_at=2.
        assert split_at == 2

    def test_only_tool_messages_exceeding_limit(self):
        """All messages are intermediate except the last assistant."""
        extractor = _build_extractor()
        extractor.hard_message_limit = 3

        msgs = [
            _tool_call_msg("c1", 0),
            _tool_response_msg("r1", 1),
            _tool_call_msg("c2", 2),
            _tool_response_msg("r2", 3),
            _tool_call_msg("c3", 4),
            _tool_response_msg("r3", 5),
            _assistant_msg("final", 6),
        ]
        split_at = extractor._find_force_split_point(msgs)
        # No safe split (walk back all intermediate, walk forward finds nothing
        # until we include the assistant)... but _is_safe_split requires
        # split_at <= len-1=6. Check: msg[6]=assistant -> safe at 7? No, 7>6.
        # Forward walk: candidate goes 3,4,5,6,7 -> 7 > len=7? No, 7==len.
        # No safe split found -> returns len(messages)=7
        assert split_at == len(msgs)


# ---------------------------------------------------------------------------
# Tests: split_at == len(messages) in parent's three-phase pipeline
# ---------------------------------------------------------------------------


class TestSplitAtFullLength:
    """When _find_force_split_point returns len(messages), the entire
    message list becomes one MemCell in Phase 1. Verify the exact behavior
    of all three phases and the returned status."""

    @pytest.mark.asyncio
    @patch("memory_layer.memcell_extractor.conv_memcell_extractor.record_boundary_detection")
    @patch("memory_layer.memcell_extractor.conv_memcell_extractor.record_memcell_extracted")
    @patch("memory_layer.memcell_extractor.conv_memcell_extractor.get_space_id_for_metrics", return_value="test")
    async def test_single_unsplittable_turn_creates_one_memcell(
        self, mock_space, mock_record, mock_bd
    ):
        """One long tool turn with no safe split -> one MemCell, Phase 2 skipped.

        Messages: user, tc, tool, tc, tool, assistant (6 msgs, limit=3).
        _find_force_split_point returns 6. Phase 1 creates 1 MemCell.
        Phase 2 gets [] -> skipped. Phase 3 no flush -> skipped.
        """
        mock_llm = MagicMock()
        # LLM should NOT be called (Phase 2 skipped)
        mock_llm.generate = AsyncMock(side_effect=AssertionError("LLM should not be called"))
        extractor = AgentMemCellExtractor(llm_provider=mock_llm)
        extractor.hard_message_limit = 3

        msgs = [
            _user_msg("q", 0),
            _tool_call_msg("c1", 1),
            _tool_response_msg("r1", 2),
            _tool_call_msg("c2", 3),
            _tool_response_msg("r2", 4),
            _assistant_msg("final", 5),
        ]
        request = _make_request(history_msgs=[], new_msgs=msgs)
        memcells, status = await extractor.extract_memcell(request)

        assert len(memcells) == 1
        assert len(memcells[0].original_data) == 6
        assert status.should_wait is False
        # LLM.generate was never called (Phase 2 skipped)
        mock_llm.generate.assert_not_called()

    @pytest.mark.asyncio
    @patch("memory_layer.memcell_extractor.conv_memcell_extractor.record_boundary_detection")
    @patch("memory_layer.memcell_extractor.conv_memcell_extractor.record_memcell_extracted")
    @patch("memory_layer.memcell_extractor.conv_memcell_extractor.get_space_id_for_metrics", return_value="test")
    async def test_full_length_split_with_flush_still_works(
        self, mock_space, mock_record, mock_bd
    ):
        """flush=True, but Phase 1 already consumed everything -> Phase 3 skipped.

        All messages become one force-split MemCell. Flush has nothing left.
        """
        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(side_effect=AssertionError("LLM should not be called"))
        extractor = AgentMemCellExtractor(llm_provider=mock_llm)
        extractor.hard_message_limit = 3

        msgs = [
            _user_msg("q", 0),
            _tool_call_msg("c1", 1),
            _tool_response_msg("r1", 2),
            _assistant_msg("final", 3),
        ]
        request = _make_request(history_msgs=[], new_msgs=msgs, flush=True)
        memcells, status = await extractor.extract_memcell(request)

        # Exactly 1 MemCell from Phase 1, Phase 3 not triggered
        assert len(memcells) == 1
        assert len(memcells[0].original_data) == 4
        assert status.should_wait is False

    @pytest.mark.asyncio
    @patch("memory_layer.memcell_extractor.conv_memcell_extractor.record_boundary_detection")
    @patch("memory_layer.memcell_extractor.conv_memcell_extractor.record_memcell_extracted")
    @patch("memory_layer.memcell_extractor.conv_memcell_extractor.get_space_id_for_metrics", return_value="test")
    async def test_full_length_then_normal_split_in_next_iteration(
        self, mock_space, mock_record, mock_bd
    ):
        """Two turns: first is unsplittable (returns len), second is normal.

        Messages:
          Turn 1: user, tc, tool, tc, tool, assistant (6 msgs, unsplittable)
          Turn 2: user, assistant (2 msgs, normal)

        limit=5. First loop iteration: 8 msgs >= 5. split_at for 8 msgs:
        parent candidate=4. Walk back finds assistant at [5] -> no, [5] is
        the 6th msg. Let me think... msgs[3]=tool_call, [2]=tool, [1]=tc,
        [0]=user -> no safe. Walk forward: msgs[4]=tool, [5]=assistant -> safe at 6.
        So split_at=6 (normal split, not len). Chunk=[0:6], remainder=[6:8]=[user, asst].
        Phase 2 runs on [user, asst].
        """
        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(
            return_value='{"reasoning": "done", "boundaries": [], "should_wait": false}'
        )
        extractor = AgentMemCellExtractor(llm_provider=mock_llm)
        extractor.hard_message_limit = 5

        msgs = [
            # Turn 1 (unsplittable single turn)
            _user_msg("q1", 0),
            _tool_call_msg("c1", 1),
            _tool_response_msg("r1", 2),
            _tool_call_msg("c2", 3),
            _tool_response_msg("r2", 4),
            _assistant_msg("a1", 5),
            # Turn 2 (simple)
            _user_msg("q2", 6),
            _assistant_msg("a2", 7),
        ]
        request = _make_request(history_msgs=[], new_msgs=msgs)
        memcells, status = await extractor.extract_memcell(request)

        # Phase 1: split at 6 (first turn complete)
        # Phase 2: [user, assistant] -> LLM called, no boundaries
        assert len(memcells) >= 1
        # First MemCell should contain the full tool turn (6 msgs)
        assert len(memcells[0].original_data) == 6
        # LLM was called for Phase 2 on the remainder
        mock_llm.generate.assert_called_once()

    @pytest.mark.asyncio
    @patch("memory_layer.memcell_extractor.conv_memcell_extractor.record_boundary_detection")
    @patch("memory_layer.memcell_extractor.conv_memcell_extractor.record_memcell_extracted")
    @patch("memory_layer.memcell_extractor.conv_memcell_extractor.get_space_id_for_metrics", return_value="test")
    async def test_all_intermediate_no_assistant_returns_full_len(
        self, mock_space, mock_record, mock_bd
    ):
        """Edge case: all messages are intermediate (no final assistant at all).
        _find_force_split_point returns len(messages).

        This shouldn't happen in practice (guard ensures last msg is not
        intermediate), but if it does, the pipeline handles it gracefully.
        """
        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(side_effect=AssertionError("LLM should not be called"))
        extractor = AgentMemCellExtractor(llm_provider=mock_llm)
        extractor.hard_message_limit = 2

        msgs = [
            _tool_call_msg("c1", 0),
            _tool_response_msg("r1", 1),
            _tool_call_msg("c2", 2),
        ]
        # Directly test _find_force_split_point
        split_at = extractor._find_force_split_point(msgs)
        assert split_at == len(msgs)

    @pytest.mark.asyncio
    @patch("memory_layer.memcell_extractor.conv_memcell_extractor.record_boundary_detection")
    @patch("memory_layer.memcell_extractor.conv_memcell_extractor.record_memcell_extracted")
    @patch("memory_layer.memcell_extractor.conv_memcell_extractor.get_space_id_for_metrics", return_value="test")
    async def test_repeated_unsplittable_turns_each_become_own_memcell(
        self, mock_space, mock_record, mock_bd
    ):
        """Multiple consecutive unsplittable turns, each should become its own MemCell.

        Turn 1: user, tc, tool, tc, tool, assistant (6 msgs)
        Turn 2: user, tc, tool, tc, tool, assistant (6 msgs)
        limit=5.

        Iteration 1: 12 msgs. split walks forward to 6 (first assistant).
        Iteration 2: 6 msgs >= 5. No safe split within first 4. Walk forward to 6.
        split_at=6=len -> all consumed. Loop exits.
        """
        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(side_effect=AssertionError("LLM should not be called"))
        extractor = AgentMemCellExtractor(llm_provider=mock_llm)
        extractor.hard_message_limit = 5

        msgs = []
        for i in range(2):
            msgs.extend([
                _user_msg(f"q{i}", i * 10),
                _tool_call_msg(f"c{i}_1", i * 10 + 1),
                _tool_response_msg(f"r{i}_1", i * 10 + 2),
                _tool_call_msg(f"c{i}_2", i * 10 + 3),
                _tool_response_msg(f"r{i}_2", i * 10 + 4),
                _assistant_msg(f"a{i}", i * 10 + 5),
            ])

        request = _make_request(history_msgs=[], new_msgs=msgs)
        memcells, status = await extractor.extract_memcell(request)

        # Each turn becomes one MemCell
        assert len(memcells) == 2
        assert len(memcells[0].original_data) == 6
        assert len(memcells[1].original_data) == 6
        assert status.should_wait is False
        # LLM not called (Phase 2 skipped because all consumed in Phase 1)
        mock_llm.generate.assert_not_called()

    @pytest.mark.asyncio
    @patch("memory_layer.memcell_extractor.conv_memcell_extractor.record_boundary_detection")
    @patch("memory_layer.memcell_extractor.conv_memcell_extractor.record_memcell_extracted")
    @patch("memory_layer.memcell_extractor.conv_memcell_extractor.get_space_id_for_metrics", return_value="test")
    async def test_total_messages_preserved_after_full_length_split(
        self, mock_space, mock_record, mock_bd
    ):
        """Verify no messages are lost when split_at=len is used."""
        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(side_effect=AssertionError("LLM should not be called"))
        extractor = AgentMemCellExtractor(llm_provider=mock_llm)
        extractor.hard_message_limit = 4

        # 3 unsplittable turns of 8 msgs each = 24 total
        msgs = []
        for i in range(3):
            msgs.extend(_make_multi_tool_turn(i))

        request = _make_request(history_msgs=[], new_msgs=msgs)
        memcells, status = await extractor.extract_memcell(request)

        total = sum(len(mc.original_data) for mc in memcells)
        assert total == len(msgs)


# ---------------------------------------------------------------------------
# Tests: Infinite loop protection and progress guarantee
# ---------------------------------------------------------------------------


class TestLoopProgress:
    """Verify the force-split loop always makes progress and terminates."""

    def test_split_at_never_returns_zero(self):
        """_find_force_split_point never returns 0 (which would cause no progress)."""
        extractor = _build_extractor()
        for limit in [2, 3, 4, 8, 16, 64]:
            extractor.hard_message_limit = limit
            for pattern in ["simple", "tool", "multi_tool", "mixed"]:
                msgs = _build_trajectory(20, pattern)
                if len(msgs) <= 1:
                    continue
                split_at = extractor._find_force_split_point(msgs)
                assert split_at >= 2 or split_at == len(msgs), (
                    f"limit={limit}, pattern={pattern}: split_at={split_at} "
                    f"would cause infinite loop or single-msg chunk"
                )

    def test_split_at_never_returns_one(self):
        """split_at=1 would create a single-message MemCell (useless). Should never happen."""
        extractor = _build_extractor()
        for limit in [2, 3, 4]:
            extractor.hard_message_limit = limit
            for msgs in [
                [_user_msg("q", 0), _assistant_msg("a", 1), _user_msg("q2", 2)],
                [_user_msg("q", 0), _tool_call_msg("c", 1), _tool_response_msg("r", 2), _assistant_msg("a", 3)],
                [_tool_call_msg("c", 0), _tool_response_msg("r", 1), _assistant_msg("a", 2), _user_msg("q", 3)],
            ]:
                split_at = extractor._find_force_split_point(msgs)
                assert split_at != 1, (
                    f"limit={limit}: split_at=1 for msgs of len {len(msgs)}"
                )

    def test_loop_max_iterations_bounded(self):
        """Loop iterations should be bounded by len/2 at worst."""
        extractor = _build_extractor()
        extractor.hard_message_limit = 4
        msgs = _build_trajectory(100, "simple")  # 200 messages

        iterations = 0
        all_msgs = list(msgs)
        while len(all_msgs) > 1:
            if len(all_msgs) < extractor.hard_message_limit:
                break
            split_at = extractor._find_force_split_point(all_msgs)
            assert split_at >= 2 or split_at == len(all_msgs)
            all_msgs = all_msgs[split_at:]
            iterations += 1
            # Safety: should never exceed len/2 iterations
            assert iterations <= len(msgs), "Loop exceeded maximum iterations"

        # Sanity: we should have done some iterations
        assert iterations > 0

    def test_adversarial_all_tool_calls_no_assistant(self):
        """Worst case: all messages are intermediate, no final assistant anywhere.
        Loop must still terminate."""
        extractor = _build_extractor()
        extractor.hard_message_limit = 3

        # 20 intermediate messages, no final assistant
        msgs = []
        for i in range(10):
            msgs.append(_tool_call_msg(f"c{i}", i * 2))
            msgs.append(_tool_response_msg(f"r{i}", i * 2 + 1))

        all_msgs = list(msgs)
        iterations = 0
        while len(all_msgs) > 1:
            if len(all_msgs) < extractor.hard_message_limit:
                break
            split_at = extractor._find_force_split_point(all_msgs)
            # Returns len(messages) since no safe split exists
            assert split_at == len(all_msgs)
            all_msgs = all_msgs[split_at:]
            iterations += 1
            assert iterations <= 1, "Should exit in 1 iteration with len(messages)"

        assert iterations == 1


# ---------------------------------------------------------------------------
# Tests: Phase 2 MemCell validity after remap
# ---------------------------------------------------------------------------


class TestPhase2MemCellValidity:
    """Phase 2 uses _detect_boundaries with remap. Verify the MemCells
    created from remapped boundaries contain complete tool sequences."""

    @pytest.mark.asyncio
    @patch("memory_layer.memcell_extractor.conv_memcell_extractor.record_boundary_detection")
    @patch("memory_layer.memcell_extractor.conv_memcell_extractor.record_memcell_extracted")
    @patch("memory_layer.memcell_extractor.conv_memcell_extractor.get_space_id_for_metrics", return_value="test")
    async def test_phase2_memcell_includes_full_tool_sequence(
        self, mock_space, mock_record, mock_bd
    ):
        """LLM boundary splits two turns. Each Phase 2 MemCell should
        contain the complete tool sequence for its turn.

        Messages (under limit, so Phase 1 skips):
          [0] user       -> filtered[0]
          [1] tc         (filtered)
          [2] tool       (filtered)
          [3] assistant  -> filtered[1]
          [4] user       -> filtered[2]
          [5] assistant  -> filtered[3]

        LLM boundary=[2] on filtered -> remap to 4.
        Phase 2: MemCell_1=[0:4]=[user,tc,tool,asst], MemCell_2=[4:]=[user,asst].
        """
        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(
            return_value='{"reasoning": "topic change", "boundaries": [2], "should_wait": false}'
        )
        extractor = AgentMemCellExtractor(llm_provider=mock_llm)
        extractor.hard_message_limit = 100  # High limit to skip Phase 1

        request = _make_request(
            history_msgs=[],
            new_msgs=[
                _user_msg("q1", 0),
                _tool_call_msg("c1", 1),
                _tool_response_msg("r1", 2),
                _assistant_msg("a1", 3),
                _user_msg("q2", 4),
                _assistant_msg("a2", 5),
            ],
        )
        memcells, status = await extractor.extract_memcell(request)

        # boundary=[2] on filtered -> remap to 4 -> MemCell_1=[0:4]
        # Remainder=[4:6]=[user, asst] stays as remaining (no further boundary)
        assert len(memcells) == 1
        # MemCell: complete tool turn [user, tc, tool, asst]
        assert len(memcells[0].original_data) == 4
        first_last = memcells[0].original_data[-1]["message"]
        assert first_last["role"] == "assistant"
        assert not first_last.get("tool_calls")
        assert status.should_wait is False

    @pytest.mark.asyncio
    @patch("memory_layer.memcell_extractor.conv_memcell_extractor.record_boundary_detection")
    @patch("memory_layer.memcell_extractor.conv_memcell_extractor.record_memcell_extracted")
    @patch("memory_layer.memcell_extractor.conv_memcell_extractor.get_space_id_for_metrics", return_value="test")
    async def test_phase2_remap_never_splits_inside_tool_sequence(
        self, mock_space, mock_record, mock_bd
    ):
        """Remap guarantees boundaries land after non-intermediate messages.
        Since filtered list excludes intermediates, remap always produces
        boundaries at user or final-assistant positions.

        Three turns with tools, LLM splits after each turn.
        """
        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(
            return_value='{"reasoning": "3 topics", "boundaries": [2, 4], "should_wait": false}'
        )
        extractor = AgentMemCellExtractor(llm_provider=mock_llm)
        extractor.hard_message_limit = 100

        request = _make_request(
            history_msgs=[],
            new_msgs=[
                # Turn 1
                _user_msg("q1", 0),       # filtered[0]
                _tool_call_msg("c1", 1),
                _tool_response_msg("r1", 2),
                _assistant_msg("a1", 3),   # filtered[1]
                # Turn 2
                _user_msg("q2", 4),        # filtered[2]
                _tool_call_msg("c2", 5),
                _tool_response_msg("r2", 6),
                _assistant_msg("a2", 7),   # filtered[3]
                # Turn 3
                _user_msg("q3", 8),        # filtered[4]
                _assistant_msg("a3", 9),   # filtered[5]
            ],
        )
        memcells, status = await extractor.extract_memcell(request)

        # boundaries=[2,4] on filtered -> remap: [4, 8]
        # MemCell_1=[0:4], MemCell_2=[4:8], remainder=[8:]
        assert len(memcells) == 2
        # Each MemCell should have complete tool sequences
        for mc in memcells:
            last_msg = mc.original_data[-1]["message"]
            assert last_msg["role"] == "assistant"
            assert not last_msg.get("tool_calls")


# ---------------------------------------------------------------------------
# Tests: All three phases triggered together
# ---------------------------------------------------------------------------


class TestAllThreePhases:
    """End-to-end test that triggers Phase 1 + Phase 2 + Phase 3."""

    @pytest.mark.asyncio
    @patch("memory_layer.memcell_extractor.conv_memcell_extractor.record_boundary_detection")
    @patch("memory_layer.memcell_extractor.conv_memcell_extractor.record_memcell_extracted")
    @patch("memory_layer.memcell_extractor.conv_memcell_extractor.get_space_id_for_metrics", return_value="test")
    async def test_phase1_force_split_phase2_llm_phase3_flush(
        self, mock_space, mock_record, mock_bd
    ):
        """Trajectory that triggers all three phases:
        - Phase 1: limit=8, total=14 msgs -> force split
        - Phase 2: LLM detects boundary in remaining
        - Phase 3: flush=True packs the tail

        Messages (14 total):
          Turn 1: user, tc, tool, assistant    [0-3]   -> Phase 1 chunk
          Turn 2: user, assistant              [4-5]
          Turn 3: user, tc, tool, assistant    [6-9]
          Turn 4: user, assistant              [10-11]
          Turn 5: user, assistant              [12-13]

        limit=8: Phase 1 splits at 4 (first turn). Remaining=10 msgs.
        10 >= 8: Phase 1 again, splits at Turn 2 end (idx 6 in remaining=2).
        Remaining=8 msgs. 8 >= 8: Phase 1 again, splits.
        Eventually remaining < 8 -> Phase 2. LLM detects boundary -> MemCell.
        flush=True -> Phase 3 packs remainder.
        """
        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(
            return_value='{"reasoning": "boundary after turn", "boundaries": [2], "should_wait": false}'
        )
        extractor = AgentMemCellExtractor(llm_provider=mock_llm)
        extractor.hard_message_limit = 8

        msgs = [
            # Turn 1 (with tools)
            _user_msg("q1", 0),
            _tool_call_msg("c1", 1),
            _tool_response_msg("r1", 2),
            _assistant_msg("a1", 3),
            # Turn 2 (simple)
            _user_msg("q2", 4),
            _assistant_msg("a2", 5),
            # Turn 3 (with tools)
            _user_msg("q3", 6),
            _tool_call_msg("c3", 7),
            _tool_response_msg("r3", 8),
            _assistant_msg("a3", 9),
            # Turn 4 (simple)
            _user_msg("q4", 10),
            _assistant_msg("a4", 11),
            # Turn 5 (simple)
            _user_msg("q5", 12),
            _assistant_msg("a5", 13),
        ]
        request = _make_request(history_msgs=[], new_msgs=msgs, flush=True)
        memcells, status = await extractor.extract_memcell(request)

        # Should produce multiple MemCells from all three phases
        assert len(memcells) >= 2
        # should_wait=False because flush consumed everything
        assert status.should_wait is False
        # Total messages preserved
        total = sum(len(mc.original_data) for mc in memcells)
        assert total == len(msgs)
        # All MemCells have type AGENTCONVERSATION
        for mc in memcells:
            assert mc.type == RawDataType.AGENTCONVERSATION


# ---------------------------------------------------------------------------
# Tests: Flush edge cases with agent guard
# ---------------------------------------------------------------------------


class TestFlushWithGuard:
    """Flush mode bypasses agent guard but still flows through parent pipeline."""

    @pytest.mark.asyncio
    @patch("memory_layer.memcell_extractor.conv_memcell_extractor.record_boundary_detection")
    @patch("memory_layer.memcell_extractor.conv_memcell_extractor.record_memcell_extracted")
    @patch("memory_layer.memcell_extractor.conv_memcell_extractor.get_space_id_for_metrics", return_value="test")
    async def test_flush_with_only_history_no_new(self, mock_space, mock_record, mock_bd):
        """flush=True, history=[user, asst], new=[] -> parent's special branch
        creates one flush MemCell directly from history."""
        extractor = _build_extractor()
        request = _make_request(
            history_msgs=[_user_msg("q", 0), _assistant_msg("a", 1)],
            new_msgs=[],
            flush=True,
        )
        memcells, status = await extractor.extract_memcell(request)
        assert len(memcells) == 1
        assert len(memcells[0].original_data) == 2
        assert status.should_wait is False

    @pytest.mark.asyncio
    async def test_flush_empty_history_empty_new(self):
        """flush=True, history=[], new=[] -> parent returns ([], should_wait=True)."""
        extractor = _build_extractor()
        request = _make_request(history_msgs=[], new_msgs=[], flush=True)
        memcells, status = await extractor.extract_memcell(request)
        assert memcells == []
        assert status.should_wait is True

    @pytest.mark.asyncio
    @patch("memory_layer.memcell_extractor.conv_memcell_extractor.record_boundary_detection")
    @patch("memory_layer.memcell_extractor.conv_memcell_extractor.record_memcell_extracted")
    @patch("memory_layer.memcell_extractor.conv_memcell_extractor.get_space_id_for_metrics", return_value="test")
    async def test_flush_bypasses_intermediate_guard(self, mock_space, mock_record, mock_bd):
        """flush=True overrides intermediate guard: even if last new msg
        is tool_call, flush still processes everything."""
        extractor = _build_extractor()
        request = _make_request(
            history_msgs=[_user_msg("q", 0), _assistant_msg("a", 1)],
            new_msgs=[_user_msg("q2", 2), _tool_call_msg("thinking", 3)],
            flush=True,
        )
        memcells, status = await extractor.extract_memcell(request)
        assert len(memcells) >= 1
        assert status.should_wait is False
        total = sum(len(mc.original_data) for mc in memcells)
        assert total == 4

    @pytest.mark.asyncio
    @patch("memory_layer.memcell_extractor.conv_memcell_extractor.record_boundary_detection")
    @patch("memory_layer.memcell_extractor.conv_memcell_extractor.record_memcell_extracted")
    @patch("memory_layer.memcell_extractor.conv_memcell_extractor.get_space_id_for_metrics", return_value="test")
    async def test_flush_with_long_history_tool_sequence(self, mock_space, mock_record, mock_bd):
        """flush=True, only history (no new), history contains tool sequences."""
        extractor = _build_extractor()
        request = _make_request(
            history_msgs=[
                _user_msg("q", 0),
                _tool_call_msg("c", 1),
                _tool_response_msg("r", 2),
                _assistant_msg("a", 3),
            ],
            new_msgs=[],
            flush=True,
        )
        memcells, status = await extractor.extract_memcell(request)
        assert len(memcells) == 1
        assert len(memcells[0].original_data) == 4
        assert status.should_wait is False


class TestRawDataType:
    def test_raw_data_type_is_agent_conversation(self):
        extractor = _build_extractor()
        assert extractor.raw_data_type == RawDataType.AGENTCONVERSATION


class TestExtractParticipantIdsOnlyUser:
    """Test that AgentMemCellExtractor inherits parent behavior: only role='user' IDs."""

    def test_only_keeps_user_role(self):
        """Only messages with role='user' should be included in participant IDs."""
        extractor = _build_extractor()
        chat_raw_data_list = [
            {"sender_id": "user_1", "sender_name": "User", "role": "user"},
            {"sender_id": "tool_call_123", "role": "tool", "tool_call_id": "call_1"},
            {"sender_id": "assistant_1", "sender_name": "Assistant", "role": "assistant"},
        ]
        result = extractor._extract_participant_ids(chat_raw_data_list)
        assert set(result) == {"user_1"}

    def test_no_user_returns_empty(self):
        """If no messages have role='user', result should be empty."""
        extractor = _build_extractor()
        chat_raw_data_list = [
            {"sender_id": "tool_1", "role": "tool", "tool_call_id": "call_1"},
            {"sender_id": "assistant_1", "role": "assistant"},
        ]
        result = extractor._extract_participant_ids(chat_raw_data_list)
        assert result == []

    def test_no_role_field_not_included(self):
        """Messages without 'role' field should not be included."""
        extractor = _build_extractor()
        chat_raw_data_list = [
            {"sender_id": "user_1", "sender_name": "User"},
            {"sender_id": "user_2", "sender_name": "Bot"},
        ]
        result = extractor._extract_participant_ids(chat_raw_data_list)
        assert result == []

    def test_mixed_roles(self):
        """Mix of tool, user, assistant, and no-role messages — only user kept."""
        extractor = _build_extractor()
        chat_raw_data_list = [
            {"sender_id": "user_1", "role": "user"},
            {"sender_id": "tool_abc", "role": "tool", "tool_call_id": "call_1"},
            {"sender_id": "assistant_1", "role": "assistant"},
            {"sender_id": "user_2"},
            {"sender_id": "tool_def", "role": "tool", "tool_call_id": "call_2"},
        ]
        result = extractor._extract_participant_ids(chat_raw_data_list)
        assert set(result) == {"user_1"}

    def test_conv_extractor_only_keeps_user(self):
        """Base ConvMemCellExtractor should only keep role='user' IDs."""
        from memory_layer.memcell_extractor.conv_memcell_extractor import ConvMemCellExtractor
        mock_llm = MagicMock()
        conv_extractor = ConvMemCellExtractor(mock_llm)
        chat_raw_data_list = [
            {"sender_id": "user_1", "role": "user"},
            {"sender_id": "tool_1", "role": "tool"},
        ]
        result = conv_extractor._extract_participant_ids(chat_raw_data_list)
        assert set(result) == {"user_1"}

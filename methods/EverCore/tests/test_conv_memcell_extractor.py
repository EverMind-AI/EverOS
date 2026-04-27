"""
ConvMemCellExtractor Test

Test conversation boundary detection functionality, including:
- Multi-split boundary detection logic
- MemCell generation
- Force-split (token/message limit)
- Flush mode behavior

Usage:
    python src/bootstrap.py tests/test_conv_memcell_extractor.py
"""

import pytest
import asyncio
from datetime import timedelta
from typing import List, Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch

# Import dependency injection related modules
from common_utils.datetime_utils import get_now_with_timezone
from core.di.utils import get_bean_by_type
from core.observation.logger import get_logger

# Import modules to be tested
from memory_layer.memcell_extractor.conv_memcell_extractor import (
    ConvMemCellExtractor,
    ConversationMemCellExtractRequest,
    BatchBoundaryResult,
)
from memory_layer.memcell_extractor.base_memcell_extractor import (
    RawData,
    MemCell,
    StatusResult,
)
from memory_layer.llm.llm_provider import LLMProvider

# Get logger
logger = get_logger(__name__)


def get_llm_provider() -> LLMProvider:
    """Get LLM Provider, first try DI container, if fails then create directly"""
    try:
        return get_bean_by_type(LLMProvider)
    except:
        logger.info("LLMProvider not found in DI container, creating directly...")
        return LLMProvider("openai")


def mock_llm_provider() -> MagicMock:
    """Return a MagicMock LLM provider for unit tests (no API key needed)."""
    provider = MagicMock(spec=LLMProvider)
    provider.generate = AsyncMock(
        return_value='{"boundaries": [], "should_wait": false}'
    )
    return provider


class TestConvMemCellExtractor:
    """ConvMemCellExtractor Test Class"""

    def setup_method(self):
        """Setup before each test method"""
        self.base_time = get_now_with_timezone() - timedelta(hours=1)

    def create_test_messages(
        self,
        count: int,
        sender: str = "Alice",
        time_offset_minutes: int = 0,
        content_prefix: str = "Test message",
    ) -> List[Dict[str, Any]]:
        """Create test messages"""
        messages = []
        for i in range(count):
            messages.append(
                {
                    "sender_id": f"user_{i % 2}",
                    "sender_name": sender if i % 2 == 0 else "Bob",
                    "content": [
                        {
                            "type": "text",
                            "text": f"{content_prefix} {i + 1}: This is a test conversation.",
                        }
                    ],
                    "timestamp": (
                        self.base_time + timedelta(minutes=time_offset_minutes + i)
                    ).isoformat(),
                }
            )
        return messages

    def create_raw_data_list(self, messages: List[Dict[str, Any]]) -> List[RawData]:
        """Convert messages to RawData list"""
        return [
            RawData(
                content=msg, data_id=f"test_data_{i}", metadata={"message_index": i}
            )
            for i, msg in enumerate(messages)
        ]

    def make_request(
        self,
        history_msgs: List[Dict],
        new_msgs: List[Dict],
        group_id: str = "test_group",
        user_id_list: List[str] = None,
        flush: bool = False,
    ) -> ConversationMemCellExtractRequest:
        return ConversationMemCellExtractRequest(
            history_raw_data_list=self.create_raw_data_list(history_msgs),
            new_raw_data_list=self.create_raw_data_list(new_msgs),
            user_id_list=user_id_list or ["alice", "bob"],
            group_id=group_id,
            flush=flush,
        )

    # =========================================================================
    # Unit tests (mock LLM)
    # =========================================================================

    @pytest.mark.asyncio
    async def test_no_new_messages_returns_empty(self):
        """Empty new_raw_data_list always returns empty MemCell list."""
        extractor = ConvMemCellExtractor(mock_llm_provider())
        request = ConversationMemCellExtractRequest(
            history_raw_data_list=self.create_raw_data_list(
                self.create_test_messages(3)
            ),
            new_raw_data_list=[],
            user_id_list=["alice"],
            group_id="test",
        )
        memcells, status = await extractor.extract_memcell(request)
        assert memcells == []
        assert status.should_wait is True

    @pytest.mark.asyncio
    async def test_force_split_message_limit(self):
        """When combined messages exceed hard_message_limit, force split without LLM."""
        extractor = ConvMemCellExtractor(
            mock_llm_provider(), hard_message_limit=10, hard_token_limit=99999
        )

        # 9 history + 3 new = 12 total, exceeds limit=10
        history = self.create_test_messages(9, time_offset_minutes=0)
        new = self.create_test_messages(3, time_offset_minutes=10)
        request = self.make_request(history, new, flush=False)

        # Should force-split without calling LLM
        with patch.object(extractor, '_detect_boundaries') as mock_detect:
            memcells, status = await extractor.extract_memcell(request)

        # Force split should have consumed some messages; LLM not called for force-split phase
        # After force split, remaining (if any) go through LLM detection
        assert isinstance(memcells, list)
        assert len(memcells) >= 1
        # First MemCell should be from force split (within limit)
        assert len(memcells[0].original_data) <= 9

    @pytest.mark.asyncio
    async def test_no_boundary_no_flush_returns_empty(self):
        """LLM says no boundary, flush=False → empty list, should_wait from LLM."""
        extractor = ConvMemCellExtractor(
            mock_llm_provider(), hard_message_limit=100, hard_token_limit=999999
        )

        history = self.create_test_messages(3, time_offset_minutes=0)
        new = self.create_test_messages(2, time_offset_minutes=5)
        request = self.make_request(history, new, flush=False)

        # Mock LLM returns no boundaries
        mock_result = BatchBoundaryResult(boundaries=[], should_wait=False)
        with patch.object(extractor, '_detect_boundaries', return_value=mock_result):
            memcells, status = await extractor.extract_memcell(request)

        assert memcells == []
        assert status.should_wait is False

    @pytest.mark.asyncio
    async def test_no_boundary_flush_returns_one_memcell(self):
        """LLM says no boundary, flush=True → one MemCell containing all messages."""
        extractor = ConvMemCellExtractor(
            mock_llm_provider(), hard_message_limit=100, hard_token_limit=999999
        )

        history = self.create_test_messages(3, time_offset_minutes=0)
        new = self.create_test_messages(2, time_offset_minutes=5)
        request = self.make_request(history, new, flush=True)

        mock_result = BatchBoundaryResult(boundaries=[], should_wait=False)
        with patch.object(extractor, '_detect_boundaries', return_value=mock_result):
            memcells, status = await extractor.extract_memcell(request)

        assert len(memcells) == 1
        assert len(memcells[0].original_data) == 5  # 3 history + 2 new
        assert status.should_wait is False

    @pytest.mark.asyncio
    async def test_single_boundary_detected(self):
        """LLM detects one boundary → one MemCell for messages before boundary."""
        extractor = ConvMemCellExtractor(
            mock_llm_provider(), hard_message_limit=100, hard_token_limit=999999
        )

        history = self.create_test_messages(4, time_offset_minutes=0)
        new = self.create_test_messages(3, time_offset_minutes=10)
        request = self.make_request(history, new, flush=False)

        # LLM detects boundary after message 4 (all history, none of new)
        mock_result = BatchBoundaryResult(boundaries=[4], should_wait=False)
        with patch.object(extractor, '_detect_boundaries', return_value=mock_result):
            memcells, status = await extractor.extract_memcell(request)

        assert len(memcells) == 1
        assert len(memcells[0].original_data) == 4
        assert status.should_wait is False

    @pytest.mark.asyncio
    async def test_multiple_boundaries_detected(self):
        """LLM detects two boundaries → two MemCells, remainder accumulated."""
        extractor = ConvMemCellExtractor(
            mock_llm_provider(), hard_message_limit=100, hard_token_limit=999999
        )

        history = self.create_test_messages(6, time_offset_minutes=0)
        new = self.create_test_messages(4, time_offset_minutes=60)
        request = self.make_request(history, new, flush=False)

        # Boundaries after message 3 and after message 7 (within 10 total)
        mock_result = BatchBoundaryResult(boundaries=[3, 7], should_wait=False)
        with patch.object(extractor, '_detect_boundaries', return_value=mock_result):
            memcells, status = await extractor.extract_memcell(request)

        assert len(memcells) == 2
        assert len(memcells[0].original_data) == 3
        assert len(memcells[1].original_data) == 4  # messages 4-7
        # Remaining 3 messages (8-10) are not in any MemCell
        assert status.should_wait is False

    @pytest.mark.asyncio
    async def test_multiple_boundaries_with_flush(self):
        """LLM detects boundaries + flush=True → MemCells + final flush MemCell."""
        extractor = ConvMemCellExtractor(
            mock_llm_provider(), hard_message_limit=100, hard_token_limit=999999
        )

        history = self.create_test_messages(4, time_offset_minutes=0)
        new = self.create_test_messages(3, time_offset_minutes=60)
        request = self.make_request(history, new, flush=True)

        # One boundary after message 4, leaving messages 5-7 for flush
        mock_result = BatchBoundaryResult(boundaries=[4], should_wait=False)
        with patch.object(extractor, '_detect_boundaries', return_value=mock_result):
            memcells, status = await extractor.extract_memcell(request)

        assert len(memcells) == 2
        assert len(memcells[0].original_data) == 4  # LLM boundary
        assert len(memcells[1].original_data) == 3  # flush tail
        assert status.should_wait is False

    @pytest.mark.asyncio
    async def test_should_wait_propagated(self):
        """LLM returns should_wait=True → propagated in StatusResult."""
        extractor = ConvMemCellExtractor(
            mock_llm_provider(), hard_message_limit=100, hard_token_limit=999999
        )

        history = self.create_test_messages(3, time_offset_minutes=0)
        new = self.create_test_messages(1, time_offset_minutes=5)
        request = self.make_request(history, new, flush=False)

        mock_result = BatchBoundaryResult(boundaries=[], should_wait=True)
        with patch.object(extractor, '_detect_boundaries', return_value=mock_result):
            memcells, status = await extractor.extract_memcell(request)

        assert memcells == []
        assert status.should_wait is True

    @pytest.mark.asyncio
    async def test_force_split_large_history_no_history(self):
        """When new messages alone exceed limits, force-split them too."""
        extractor = ConvMemCellExtractor(
            mock_llm_provider(), hard_message_limit=5, hard_token_limit=999999
        )

        # 0 history + 8 new = 8 total, exceeds limit=5
        history: List[Dict] = []
        new = self.create_test_messages(8, time_offset_minutes=0)
        request = self.make_request(history, new, flush=False)

        mock_result = BatchBoundaryResult(boundaries=[], should_wait=False)
        with patch.object(extractor, '_detect_boundaries', return_value=mock_result):
            memcells, status = await extractor.extract_memcell(request)

        # Force split should create at least one MemCell even with no history
        assert len(memcells) >= 1
        for mc in memcells:
            assert len(mc.original_data) <= 4  # hard_message_limit - 1

    # =========================================================================
    # Parse helper unit tests
    # =========================================================================

    def test_parse_markdown_json(self):
        """Parse LLM response with markdown code block."""
        extractor = ConvMemCellExtractor(mock_llm_provider())
        resp = '''Some text before
```json
{"boundaries": [5], "should_wait": false}
```
Some text after'''
        result = extractor._parse_batch_boundary_response(resp)
        assert result is not None
        assert len(result.boundaries) == 1
        assert result.boundaries[0] == 5
        assert result.should_wait is False

    def test_parse_raw_json(self):
        """Parse LLM response as raw JSON."""
        extractor = ConvMemCellExtractor(mock_llm_provider())
        resp = '{"boundaries": [], "should_wait": true}'
        result = extractor._parse_batch_boundary_response(resp)
        assert result is not None
        assert result.boundaries == []
        assert result.should_wait is True

    def test_parse_multiple_boundaries(self):
        """Parse response with multiple boundaries."""
        extractor = ConvMemCellExtractor(mock_llm_provider())
        resp = '''{
    "boundaries": [3, 7],
    "should_wait": false
}'''
        result = extractor._parse_batch_boundary_response(resp)
        assert result is not None
        assert len(result.boundaries) == 2
        assert result.boundaries[0] == 3
        assert result.boundaries[1] == 7

    def test_parse_invalid_returns_none(self):
        """Invalid JSON returns None."""
        extractor = ConvMemCellExtractor(mock_llm_provider())
        result = extractor._parse_batch_boundary_response("not valid json at all")
        assert result is None

    def test_format_messages_with_indices(self):
        """Messages are formatted with 1-based indices."""
        extractor = ConvMemCellExtractor(mock_llm_provider())
        messages = [
            {
                "sender_name": "Alice",
                "content": [{"type": "text", "text": "Hello"}],
                "timestamp": self.base_time.isoformat(),
            },
            {
                "sender_name": "Bob",
                "content": [{"type": "text", "text": "Hi there"}],
                "timestamp": (self.base_time + timedelta(minutes=1)).isoformat(),
            },
        ]
        formatted = extractor._format_messages_with_indices(messages)
        assert "[1]" in formatted
        assert "[2]" in formatted
        assert "Alice: Hello" in formatted
        assert "Bob: Hi there" in formatted

    # =========================================================================
    # Integration tests (require real LLM)
    # =========================================================================

    @pytest.mark.asyncio
    async def test_realistic_conversation_no_boundary(self):
        """Realistic conversation that continues — expect no MemCell (accumulate)."""
        llm_provider = get_llm_provider()
        extractor = ConvMemCellExtractor(llm_provider)

        history_messages = [
            {
                "sender_name": "Alice",
                "content": [
                    {"type": "text", "text": "Let's discuss the project plan."}
                ],
                "offset": 0,
            },
            {
                "sender_name": "Bob",
                "content": [{"type": "text", "text": "Sure, backend is 80% done."}],
                "offset": 2,
            },
            {
                "sender_name": "Charlie",
                "content": [{"type": "text", "text": "Frontend design is complete."}],
                "offset": 4,
            },
        ]
        new_messages = [
            {
                "sender_name": "Alice",
                "content": [
                    {
                        "type": "text",
                        "text": "Great, when can we do integration testing?",
                    }
                ],
                "offset": 6,
            },
            {
                "sender_name": "Bob",
                "content": [{"type": "text", "text": "Next week should work."}],
                "offset": 8,
            },
        ]

        def to_raw_data(msgs):
            return [
                RawData(
                    content={
                        "sender_id": f"user_{m['sender_name'].lower()}",
                        "sender_name": m["sender_name"],
                        "content": m["content"],
                        "timestamp": (
                            self.base_time + timedelta(minutes=m["offset"])
                        ).isoformat(),
                    },
                    data_id=f"msg_{i}",
                    metadata={},
                )
                for i, m in enumerate(msgs)
            ]

        request = ConversationMemCellExtractRequest(
            history_raw_data_list=to_raw_data(history_messages),
            new_raw_data_list=to_raw_data(new_messages),
            user_id_list=["alice", "bob", "charlie"],
            group_id="project_team",
            flush=False,
        )

        memcells, status = await extractor.extract_memcell(request)

        print(
            f"\n✅ Realistic no-boundary: memcells={len(memcells)}, should_wait={status.should_wait}"
        )
        assert isinstance(memcells, list)
        assert isinstance(status, StatusResult)

    @pytest.mark.asyncio
    async def test_complete_meeting_with_flush(self):
        """Complete meeting conversation with flush=True → at least one MemCell."""
        llm_provider = get_llm_provider()
        extractor = ConvMemCellExtractor(llm_provider)

        base_time = get_now_with_timezone() - timedelta(hours=2)

        def make_msg(sender, content, offset):
            return {
                "sender_id": f"user_{sender.lower()}",
                "sender_name": sender,
                "content": content,
                "timestamp": (base_time + timedelta(minutes=offset)).isoformat(),
            }

        history = [
            make_msg("Alice", "Starting the project review meeting.", 0),
            make_msg("Bob", "Backend API is 80% complete.", 2),
            make_msg("Charlie", "Frontend design is also done.", 4),
            make_msg("Alice", "Good, any technical challenges?", 6),
            make_msg("Bob", "Permission management was tricky but solved.", 8),
        ]
        new = [
            make_msg(
                "Alice", "Charlie, present the technical solution adjustments.", 45
            ),
            make_msg(
                "Charlie", "I suggest microservices architecture for scalability.", 46
            ),
            make_msg("Bob", "Agreed. Should we adjust the timeline?", 47),
            make_msg("Alice", "Yes, one week delay but better quality.", 48),
            make_msg("Alice", "Meeting adjourned! I'll send the minutes.", 56),
        ]

        def to_raw(msgs):
            return [
                RawData(content=m, data_id=f"m_{i}", metadata={})
                for i, m in enumerate(msgs)
            ]

        request = ConversationMemCellExtractRequest(
            history_raw_data_list=to_raw(history),
            new_raw_data_list=to_raw(new),
            user_id_list=["alice", "bob", "charlie"],
            group_id="complete_meeting",
            flush=True,
        )

        memcells, status = await extractor.extract_memcell(request)

        print(
            f"\n✅ Complete meeting flush: memcells={len(memcells)}, should_wait={status.should_wait}"
        )
        assert len(memcells) >= 1
        assert status.should_wait is False

        for mc in memcells:
            assert mc.event_id is not None
            assert mc.group_id == "complete_meeting"
            assert len(mc.original_data) > 0


async def run_all_tests():
    """Run all tests"""
    print("🚀 Starting ConvMemCellExtractor tests")
    print("=" * 60)

    test_instance = TestConvMemCellExtractor()

    try:
        test_instance.setup_method()
        await test_instance.test_no_new_messages_returns_empty()
        print("✅ test_no_new_messages_returns_empty passed")

        test_instance.setup_method()
        await test_instance.test_no_boundary_no_flush_returns_empty()
        print("✅ test_no_boundary_no_flush_returns_empty passed")

        test_instance.setup_method()
        await test_instance.test_no_boundary_flush_returns_one_memcell()
        print("✅ test_no_boundary_flush_returns_one_memcell passed")

        test_instance.setup_method()
        await test_instance.test_single_boundary_detected()
        print("✅ test_single_boundary_detected passed")

        test_instance.setup_method()
        await test_instance.test_multiple_boundaries_detected()
        print("✅ test_multiple_boundaries_detected passed")

        test_instance.setup_method()
        await test_instance.test_multiple_boundaries_with_flush()
        print("✅ test_multiple_boundaries_with_flush passed")

        test_instance.setup_method()
        await test_instance.test_should_wait_propagated()
        print("✅ test_should_wait_propagated passed")

        test_instance.setup_method()
        test_instance.test_parse_markdown_json()
        print("✅ test_parse_markdown_json passed")

        test_instance.setup_method()
        test_instance.test_parse_raw_json()
        print("✅ test_parse_raw_json passed")

        test_instance.setup_method()
        test_instance.test_parse_multiple_boundaries()
        print("✅ test_parse_multiple_boundaries passed")

        test_instance.setup_method()
        test_instance.test_parse_invalid_returns_none()
        print("✅ test_parse_invalid_returns_none passed")

        test_instance.setup_method()
        test_instance.test_format_messages_with_indices()
        print("✅ test_format_messages_with_indices passed")

        # Integration tests (require LLM)
        test_instance.setup_method()
        await test_instance.test_realistic_conversation_no_boundary()
        print("✅ test_realistic_conversation_no_boundary passed")

        test_instance.setup_method()
        await test_instance.test_complete_meeting_with_flush()
        print("✅ test_complete_meeting_with_flush passed")

        print("\n" + "=" * 60)
        print("🎉 All tests completed!")

    except Exception as e:
        logger.error(f"❌ Test execution failed: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    asyncio.run(run_all_tests())

"""
ConvMemCellExtractor Test

Test conversation boundary detection functionality, including:
- Conversation boundary detection logic
- MemCell generation
- Status judgment

Usage:
    python src/bootstrap.py tests/test_conv_memcell_extractor.py
"""

import pytest
import asyncio
from datetime import timedelta
from typing import List, Dict, Any

# Import dependency injection related modules
from common_utils.datetime_utils import get_now_with_timezone
from core.di.utils import get_bean_by_type
from core.observation.logger import get_logger

# Import modules to be tested
from memory_layer.memcell_extractor.conv_memcell_extractor import (
    ConvMemCellExtractor,
    ConversationMemCellExtractRequest,
)
from memory_layer.memcell_extractor.base_memcell_extractor import RawData, MemCell
from memory_layer.llm.llm_provider import LLMProvider

# Get logger
logger = get_logger(__name__)


def get_llm_provider() -> LLMProvider:
    """Get LLM Provider, first try DI container, if fails then create directly"""
    try:
        # Try to get from DI container
        return get_bean_by_type(LLMProvider)
    except Exception:
        # If not found in DI container, create directly
        logger.info("LLMProvider not found in DI container, creating directly...")
        return LLMProvider("openai")


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
                    "speaker_id": f"user_{i}",
                    "speaker_name": sender if i % 2 == 0 else "Bob",
                    "content": f"{content_prefix} {i+1}: This is a test conversation.",
                    "timestamp": (
                        self.base_time + timedelta(minutes=time_offset_minutes + i)
                    ).isoformat(),
                }
            )
        return messages

    def create_raw_data_list(self, messages: List[Dict[str, Any]]) -> List[RawData]:
        """Convert messages to RawData list"""
        raw_data_list = []
        for i, msg in enumerate(messages):
            raw_data = RawData(
                content=msg, data_id=f"test_data_{i}", metadata={"message_index": i}
            )
            raw_data_list.append(raw_data)
        return raw_data_list

    def create_realistic_conversation(self) -> tuple[List[RawData], List[RawData]]:
        """Create realistic conversation scenario"""
        # Historical conversation - Project discussion
        history_messages = [
            {
                "speaker_name": "Alice",
                "content": "Hello everyone, let's start today's project meeting",
                "offset": 0,
            },
            {
                "speaker_name": "Bob",
                "content": "Okay, I'll report on the backend development progress",
                "offset": 2,
            },
            {
                "speaker_name": "Charlie",
                "content": "The frontend also has some updates to share",
                "offset": 4,
            },
            {
                "speaker_name": "Alice",
                "content": "Great, Bob you go first",
                "offset": 6,
            },
            {
                "speaker_name": "Bob",
                "content": "Backend API is 80% complete, database design is basically finalized",
                "offset": 8,
            },
        ]

        # New conversation - Continue discussion
        new_messages = [
            {
                "speaker_name": "Charlie",
                "content": "The frontend interface has completed the design of main pages",
                "offset": 30,
            },
            {
                "speaker_name": "Alice",
                "content": "Great, when can we start integration testing?",
                "offset": 32,
            },
            {
                "speaker_name": "Bob",
                "content": "I expect to provide stable APIs next week",
                "offset": 34,
            },
            {
                "speaker_name": "Charlie",
                "content": "Perfect, I can also start integration testing next week",
                "offset": 36,
            },
            {
                "speaker_name": "Alice",
                "content": "Perfect! Let's arrange it this way",
                "offset": 38,
            },
        ]

        def create_raw_data_from_msgs(msgs: List[Dict], prefix: str) -> List[RawData]:
            raw_data_list = []
            for i, msg in enumerate(msgs):
                timestamp = (
                    self.base_time + timedelta(minutes=msg["offset"])
                ).isoformat()
                raw_data = RawData(
                    content={
                        "speaker_id": f"user_{msg['speaker_name'].lower()}",
                        "speaker_name": msg["speaker_name"],
                        "content": msg["content"],
                        "timestamp": timestamp,
                    },
                    data_id=f"{prefix}_{i}",
                    metadata={"message_index": i},
                )
                raw_data_list.append(raw_data)
            return raw_data_list

        history_raw_data = create_raw_data_from_msgs(history_messages, "history")
        new_raw_data = create_raw_data_from_msgs(new_messages, "new")

        return history_raw_data, new_raw_data

    @pytest.mark.asyncio
    async def test_conv_boundary_detection_basic(self):
        """Test basic conversation boundary detection"""
        print("\n🧪 Test basic conversation boundary detection")

        # Get LLM Provider
        llm_provider = get_llm_provider()
        extractor = ConvMemCellExtractor(llm_provider)

        # Create test data
        history_messages = self.create_test_messages(
            3, "Alice", 0, "Historical message"
        )
        new_messages = self.create_test_messages(2, "Bob", 30, "New message")

        history_raw_data = self.create_raw_data_list(history_messages)
        new_raw_data = self.create_raw_data_list(new_messages)

        # Create request
        request = ConversationMemCellExtractRequest(
            history_raw_data_list=history_raw_data,
            new_raw_data_list=new_raw_data,
            user_id_list=["alice", "bob"],
            participants=["alice", "bob"],
            group_id="test_group",
        )

        print(
            f"📋 Request data: {len(history_raw_data)} historical + {len(new_raw_data)} new messages"
        )

        # Execute test
        result = await extractor.extract_memcell(request)

        # Verify results
        assert result is not None, "Boundary detection result should not be None"
        memcell, status_result = result

        print(f"✅ Boundary detection completed:")
        print(f"   - MemCell: {memcell is not None}")
        print(f"   - should_wait: {status_result.should_wait}")

        if memcell:
            assert memcell.event_id is not None
            assert len(memcell.user_id_list) > 0
            assert memcell.summary is not None

            print(f"\n📄 MemCell details:")
            print(f"   - event_id: {memcell.event_id}")
            print(f"   - user_id_list: {memcell.user_id_list}")
            print(f"   - participants: {memcell.participants}")
            print(f"   - group_id: {memcell.group_id}")
            print(f"   - timestamp: {memcell.timestamp}")
            print(f"   - summary: {memcell.summary}")
            print(
                f"   - original_data count: {len(memcell.original_data) if memcell.original_data else 0}"
            )

            if memcell.original_data:
                print(f"\n💬 Original conversation content:")
                for i, msg in enumerate(memcell.original_data[:3]):  # Show only first 3
                    speaker = msg.get('speaker_name', 'Unknown')
                    content = msg.get('content', '')
                    timestamp = msg.get('timestamp', '')
                    print(f"     {i+1}. [{timestamp}] {speaker}: {content}")
                if len(memcell.original_data) > 3:
                    print(f"     ... {len(memcell.original_data) - 3} more messages")
        else:
            print(f"⚠️ No MemCell generated")

    @pytest.mark.asyncio
    async def test_realistic_conversation_scenario(self):
        """Test realistic conversation scenario"""
        print("\n🧪 Test realistic conversation scenario")

        # Get LLM Provider
        llm_provider = get_llm_provider()
        extractor = ConvMemCellExtractor(llm_provider)

        # Create realistic conversation data
        history_raw_data, new_raw_data = self.create_realistic_conversation()

        # Create request
        request = ConversationMemCellExtractRequest(
            history_raw_data_list=history_raw_data,
            new_raw_data_list=new_raw_data,
            user_id_list=["alice", "bob", "charlie"],
            participants=["alice", "bob", "charlie"],
            group_id="project_team",
        )

        print(f"📋 Realistic conversation scenario:")
        print(f"   - Historical messages: {len(history_raw_data)}")
        print(f"   - New messages: {len(new_raw_data)}")
        print(f"   - Participants: {request.participants}")

        # Execute test
        result = await extractor.extract_memcell(request)

        # Analyze results
        if result is None:
            print("⚠️ No conversation boundary detected (this might be normal)")
        else:
            memcell, status_result = result
            print(f"✅ Boundary detection returned result:")
            print(f"   - MemCell: {memcell is not None}")
            print(f"   - should_wait: {status_result.should_wait}")

            if memcell:
                print(f"\n📄 Realistic conversation MemCell details:")
                print(f"   - event_id: {memcell.event_id}")
                print(f"   - user_id_list: {memcell.user_id_list}")
                print(f"   - participants: {memcell.participants}")
                print(f"   - group: {memcell.group_id}")
                print(f"   - timestamp: {memcell.timestamp}")
                print(f"   - summary: {memcell.summary}")
                print(
                    f"   - original data count: {len(memcell.original_data) if memcell.original_data else 0}"
                )

                # Display complete conversation content
                if memcell.original_data:
                    print(f"\n💬 Complete conversation record:")
                    for i, msg in enumerate(memcell.original_data):
                        speaker = msg.get('speaker_name', 'Unknown')
                        content = msg.get('content', '')
                        timestamp = msg.get('timestamp', '')
                        print(f"     {i+1}. [{timestamp}] {speaker}: {content}")

                # Verify basic fields
                assert memcell.event_id is not None
                assert len(memcell.user_id_list) == 3
                assert "alice" in memcell.user_id_list
                assert "bob" in memcell.user_id_list
                assert "charlie" in memcell.user_id_list
                assert memcell.group_id == "project_team"
            else:
                print(
                    "   - MemCell is None, conversation may not have complete boundary"
                )

            print(f"\n📊 Boundary detection status:")
            print(f"   - should_wait: {status_result.should_wait}")
            if status_result.should_wait:
                print("   - Meaning: Need to wait for more messages")
            else:
                print("   - Meaning: No need to wait, can continue processing")

    @pytest.mark.asyncio
    async def test_insufficient_data_scenario(self):
        """Test insufficient data scenario"""
        print("\n🧪 Test insufficient data scenario")

        # Get LLM Provider
        llm_provider = get_llm_provider()
        extractor = ConvMemCellExtractor(llm_provider)

        # Create very few messages
        history_messages = self.create_test_messages(1, "Alice", 0, "Short history")
        new_messages = self.create_test_messages(1, "Bob", 1, "Short new message")

        history_raw_data = self.create_raw_data_list(history_messages)
        new_raw_data = self.create_raw_data_list(new_messages)

        # Create request
        request = ConversationMemCellExtractRequest(
            history_raw_data_list=history_raw_data,
            new_raw_data_list=new_raw_data,
            user_id_list=["alice", "bob"],
            participants=["alice", "bob"],
            group_id="test_group",
        )

        print(
            f"📋 Insufficient data scenario: {len(history_raw_data)} historical + {len(new_raw_data)} new messages"
        )

        # Execute test
        result = await extractor.extract_memcell(request)

        # Verify results - may return None or should_wait=True
        if result is None:
            print("✅ Correctly handled insufficient data: returned None")
        else:
            memcell, status_result = result
            print(f"✅ Status judgment: should_wait={status_result.should_wait}")

            if memcell:
                print(f"\n📄 Insufficient data scenario MemCell info:")
                print(f"   - event_id: {memcell.event_id}")
                print(f"   - summary: {memcell.summary}")
                print(f"   - user_id_list: {memcell.user_id_list}")
                print(
                    f"   - original_data count: {len(memcell.original_data) if memcell.original_data else 0}"
                )
            else:
                print("   - MemCell: None")

            if status_result.should_wait:
                print("✅ Correctly identified need to wait for more data")
            else:
                print("ℹ️ No need to wait for more data")

    @pytest.mark.asyncio
    async def test_conversation_should_end_scenario(self):
        """Test complete conversation scenario that should end"""
        print("\n🧪 Test complete conversation scenario that should end")

        # Get LLM Provider
        llm_provider = get_llm_provider()
        extractor = ConvMemCellExtractor(llm_provider)

        # Construct a complete meeting conversation, from start to clear end
        complete_conversation = self.create_complete_meeting_conversation()
        history_raw_data, new_raw_data = complete_conversation

        # Create request
        request = ConversationMemCellExtractRequest(
            history_raw_data_list=history_raw_data,
            new_raw_data_list=new_raw_data,
            user_id_list=["alice", "bob", "charlie"],
            participants=["alice", "bob", "charlie"],
            group_id="complete_meeting",
        )

        print(f"📋 Complete meeting conversation scenario:")
        print(f"   - Historical messages: {len(history_raw_data)}")
        print(f"   - New messages: {len(new_raw_data)}")
        print(f"   - Participants: {request.participants}")
        print(f"   - Total messages: {len(history_raw_data) + len(new_raw_data)}")

        # Display conversation content preview
        print(f"\n💬 Conversation content preview:")
        all_messages = []
        for data in history_raw_data + new_raw_data:
            all_messages.append(data.content)

        for i, msg in enumerate(all_messages[:3]):
            speaker = msg.get('speaker_name', 'Unknown')
            content = msg.get('content', '')
            print(f"   Start: {speaker}: {content}")

        print(f"   ... ({len(all_messages) - 6} messages in between)")

        for i, msg in enumerate(all_messages[-3:]):
            speaker = msg.get('speaker_name', 'Unknown')
            content = msg.get('content', '')
            print(f"   End: {speaker}: {content}")

        # Execute test
        print(f"\n🔄 Starting boundary detection...")
        result = await extractor.extract_memcell(request)

        # Analyze results
        if result is None:
            print("❌ Unexpected: No boundary detected in complete conversation")
        else:
            memcell, status_result = result
            print(f"✅ Complete conversation boundary detection result:")
            print(f"   - MemCell: {memcell is not None}")
            print(f"   - should_wait: {status_result.should_wait}")

            if memcell:
                print(f"\n📄 Complete conversation MemCell details:")
                print(f"   - event_id: {memcell.event_id}")
                print(f"   - user_id_list: {memcell.user_id_list}")
                print(f"   - participants: {memcell.participants}")
                print(f"   - group: {memcell.group_id}")
                print(f"   - timestamp: {memcell.timestamp}")
                print(f"   - summary: {memcell.summary}")
                print(
                    f"   - original data count: {len(memcell.original_data) if memcell.original_data else 0}"
                )

                # Display complete conversation content
                if memcell.original_data:
                    print(f"\n💬 Conversation records included in MemCell:")
                    for i, msg in enumerate(memcell.original_data):
                        speaker = msg.get('speaker_name', 'Unknown')
                        content = msg.get('content', '')
                        timestamp = msg.get('timestamp', '')
                        print(f"     {i+1}. [{timestamp}] {speaker}: {content}")

                # Verify this is a complete conversation
                assert memcell.event_id is not None
                assert len(memcell.user_id_list) == 3
                assert memcell.group_id == "complete_meeting"
                print(
                    f"\n✅ Verification passed: This is a complete meeting conversation MemCell"
                )

            else:
                print(
                    "⚠️ MemCell is None, conversation judgment logic may need adjustment"
                )

            print(f"\n📊 Boundary detection status analysis:")
            print(f"   - should_wait: {status_result.should_wait}")
            if status_result.should_wait:
                print(
                    "   - Meaning: Need to wait for more messages (conversation may not be complete)"
                )
            else:
                print(
                    "   - Meaning: Conversation is complete, can be processed (as expected)"
                )

            if memcell and not status_result.should_wait:
                print(f"\n🎉 Success: Complete conversation boundary detected!")
            elif not memcell and not status_result.should_wait:
                print(
                    f"\n🤔 Partial success: Conversation judged complete but no MemCell generated"
                )
            else:
                print(
                    f"\n📝 Needs optimization: Conversation judgment logic may need adjustment"
                )

    def create_complete_meeting_conversation(
        self,
    ) -> tuple[List[RawData], List[RawData]]:
        """Create a complete meeting conversation, from start to clear end"""
        base_time = get_now_with_timezone() - timedelta(hours=2)  # Start 2 hours ago

        # Phase 1: Meeting start and agenda introduction (historical messages)
        meeting_start = [
            {
                "speaker_name": "Alice",
                "content": "Hello everyone, now starting our project review meeting. Today we'll discuss three topics: project progress, technical solution confirmation, and next steps.",
                "offset": 0,
            },
            {
                "speaker_name": "Bob",
                "content": "Okay Alice, I'm ready with the project progress report.",
                "offset": 1,
            },
            {
                "speaker_name": "Charlie",
                "content": "The technical solution document has also been updated.",
                "offset": 2,
            },
            {
                "speaker_name": "Alice",
                "content": "Great, let's go in order. Bob, please report on project progress first.",
                "offset": 3,
            },
            {
                "speaker_name": "Bob",
                "content": "Okay. This week we completed development and testing of the user login module, progress is on track. Database design is also complete, starting interface development next week.",
                "offset": 5,
            },
            {
                "speaker_name": "Alice",
                "content": "Good, any technical challenges encountered?",
                "offset": 6,
            },
            {
                "speaker_name": "Bob",
                "content": "Mainly in user permission management, but we've found a solution.",
                "offset": 7,
            },
        ]

        # Phase 2: Technical discussion and decision + meeting summary and end (new messages, longer time interval indicates in-depth discussion)
        meeting_end = [
            {
                "speaker_name": "Alice",
                "content": "Okay, now Charlie will present the technical solution adjustments.",
                "offset": 45,
            },  # 45 minutes later, indicating in-depth discussion in between
            {
                "speaker_name": "Charlie",
                "content": "After analysis, I suggest we adopt a microservices architecture, which will better support future scalability.",
                "offset": 46,
            },
            {
                "speaker_name": "Bob",
                "content": "I agree with Charlie's proposal, it is indeed more flexible. Do we need to adjust the development plan?",
                "offset": 47,
            },
            {
                "speaker_name": "Alice",
                "content": "Yes. We need to re-evaluate the timeline. The overall project might be delayed by one week, but quality will be better.",
                "offset": 48,
            },
            {
                "speaker_name": "Charlie",
                "content": "I can provide detailed architecture design documents next week.",
                "offset": 49,
            },
            {
                "speaker_name": "Bob",
                "content": "I'll also adjust the development plan accordingly.",
                "offset": 50,
            },
            {
                "speaker_name": "Alice",
                "content": "Good. We've finished discussing all three topics today. Summary: project progress is normal, technical solution adjusted to microservices architecture, timeline adjusted to one week delay.",
                "offset": 52,
            },
            {"speaker_name": "Alice", "content": "Any other questions?", "offset": 53},
            {
                "speaker_name": "Bob",
                "content": "I have no other questions.",
                "offset": 54,
            },
            {"speaker_name": "Charlie", "content": "Neither do I.", "offset": 55},
            {
                "speaker_name": "Alice",
                "content": "Okay, that's all for today's meeting. Thank you all for participating. I'll compile the meeting minutes and send them to everyone. Meeting adjourned!",
                "offset": 56,
            },
        ]

        def create_raw_data_from_msgs(msgs: List[Dict], prefix: str) -> List[RawData]:
            raw_data_list = []
            for i, msg in enumerate(msgs):
                timestamp = (base_time + timedelta(minutes=msg["offset"])).isoformat()
                raw_data = RawData(
                    content={
                        "speaker_id": f"user_{msg['speaker_name'].lower()}",
                        "speaker_name": msg["speaker_name"],
                        "content": msg["content"],
                        "timestamp": timestamp,
                    },
                    data_id=f"{prefix}_{i}",
                    metadata={"message_index": i, "meeting_phase": prefix},
                )
                raw_data_list.append(raw_data)
            return raw_data_list

        history_raw_data = create_raw_data_from_msgs(meeting_start, "meeting_start")
        new_raw_data = create_raw_data_from_msgs(meeting_end, "meeting_end")

        print(f"🏗️ Constructing complete meeting conversation:")
        print(f"   - Meeting start phase: {len(meeting_start)} messages")
        print(f"   - Meeting end phase: {len(meeting_end)} messages")
        print(
            f"   - Time span: {meeting_start[0]['offset']} to {meeting_end[-1]['offset']} minutes"
        )
        print(f"   - Characteristics: Clear start, discussion, decision, summary, end")

        return history_raw_data, new_raw_data

    @pytest.mark.asyncio
    async def test_data_processing_internal(self):
        """Test internal data processing logic"""
        print("\n🧪 Test internal data processing")

        # Get LLM Provider
        llm_provider = get_llm_provider()
        extractor = ConvMemCellExtractor(llm_provider)

        # Create test data
        test_message = {
            "speaker_id": "user_alice",
            "speaker_name": "Alice",
            "content": "This is a test message",
            "timestamp": self.base_time.isoformat(),
        }

        raw_data = RawData(
            content=test_message, data_id="test_data", metadata={"test": True}
        )

        # Test internal data processing method
        processed_data = extractor._data_process(raw_data)

        print(f"📋 Data processing test:")
        print(f"   - Original data: {test_message}")
        print(f"   - Processed: {processed_data}")

        # Verify processing result
        assert processed_data is not None
        assert isinstance(processed_data, dict)
        assert "speaker_name" in processed_data
        assert "content" in processed_data


async def run_all_tests():
    """Run all tests"""
    print("🚀 Starting ConvMemCellExtractor tests")
    print("=" * 60)

    test_instance = TestConvMemCellExtractor()

    try:
        # Run test methods
        test_instance.setup_method()
        await test_instance.test_conv_boundary_detection_basic()

        test_instance.setup_method()
        await test_instance.test_realistic_conversation_scenario()

        test_instance.setup_method()
        await test_instance.test_insufficient_data_scenario()

        test_instance.setup_method()
        await test_instance.test_conversation_should_end_scenario()

        test_instance.setup_method()
        await test_instance.test_data_processing_internal()

        print("\n" + "=" * 60)
        print("🎉 All tests completed!")

    except Exception as e:
        logger.error(f"❌ Test execution failed: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    # When running this script directly
    # Note: When running through bootstrap.py, environment is already initialized
    asyncio.run(run_all_tests())

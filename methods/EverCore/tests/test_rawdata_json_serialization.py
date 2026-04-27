"""
Test JSON serialization and deserialization functionality of RawData
Includes basic functionality tests and improved field name heuristic judgment tests
"""

import pytest
from datetime import datetime
from memory_layer.memcell_extractor.base_memcell_extractor import RawData
from common_utils.datetime_utils import get_now_with_timezone


class TestRawDataJsonSerialization:
    """RawData JSON serialization test class"""

    def test_basic_serialization(self):
        """Test basic serialization and deserialization"""
        # Create test data
        original_data = RawData(
            content={
                "sender_id": "user_001",
                "sender_name": "Zhang San",
                "content": [{"type": "text", "text": "This is a test message"}],
                "msgType": 1,
                "roomId": "room_123",
            },
            data_id="msg_001",
            data_type="Conversation",
            metadata={"source": "test", "version": "1.0"},
        )

        # Serialize
        json_str = original_data.to_json()
        assert isinstance(json_str, str)
        assert len(json_str) > 0

        # Deserialize
        restored_data = RawData.from_json_str(json_str)

        # Verify data consistency
        assert restored_data.content == original_data.content
        assert restored_data.data_id == original_data.data_id
        assert restored_data.data_type == original_data.data_type
        assert restored_data.metadata == original_data.metadata

    def test_datetime_serialization(self):
        """Test serialization containing datetime objects"""
        # Create test data with timestamp
        test_time = get_now_with_timezone()

        original_data = RawData(
            content={
                "timestamp": test_time,
                "createTime": test_time,
                "updateTime": test_time,
                "content": [{"type": "text", "text": "Message containing time"}],
            },
            data_id="msg_datetime",
            data_type="Conversation",
            metadata={"created_at": test_time, "processed_at": test_time},
        )

        # Serialize
        json_str = original_data.to_json()

        # Deserialize
        restored_data = RawData.from_json_str(json_str)

        # Verify timestamp fields are correctly restored as datetime objects
        assert isinstance(restored_data.content["timestamp"], datetime)
        assert isinstance(restored_data.content["createTime"], datetime)
        assert isinstance(restored_data.content["updateTime"], datetime)
        assert isinstance(restored_data.metadata["created_at"], datetime)
        assert isinstance(restored_data.metadata["processed_at"], datetime)

        # Verify timestamp accuracy (allow minor precision differences)
        assert abs((restored_data.content["timestamp"] - test_time).total_seconds()) < 1
        assert (
            abs((restored_data.metadata["created_at"] - test_time).total_seconds()) < 1
        )

    def test_nested_structure_serialization(self):
        """Test serialization of nested structures"""
        original_data = RawData(
            content={
                "replyInfo": {
                    "originalMessage": "Original message",
                    "timestamp": get_now_with_timezone(),
                    "author": {
                        "id": "user_001",
                        "name": "Zhang San",
                        "settings": {"notify": True, "theme": "dark"},
                    },
                },
                "attachments": [
                    {"type": "image", "url": "http://example.com/img1.jpg"},
                    {"type": "file", "name": "document.pdf", "size": 1024},
                ],
            },
            data_id="complex_msg",
            data_type="Conversation",
        )

        # Serialize and deserialize
        json_str = original_data.to_json()
        restored_data = RawData.from_json_str(json_str)

        # Verify nested structure
        assert (
            restored_data.content["replyInfo"]["originalMessage"] == "Original message"
        )
        assert isinstance(restored_data.content["replyInfo"]["timestamp"], datetime)
        assert restored_data.content["replyInfo"]["author"]["id"] == "user_001"
        assert (
            restored_data.content["replyInfo"]["author"]["settings"]["notify"] is True
        )
        assert len(restored_data.content["attachments"]) == 2
        assert restored_data.content["attachments"][0]["type"] == "image"

    def test_email_data_serialization(self):
        """Test serialization of email data (simulating output from email_mapper)"""
        original_data = RawData(
            content={
                'user_id_list': ["user_001", "user_002"],
                'id': 'email_123',
                'source': 'gmail',
                'mail_address': 'test@example.com',
                'thread_id': 'thread_456',
                'is_delete': False,
                'is_read': True,
                'is_draft': False,
                'importance': 'high',
                'sent_timestamp': get_now_with_timezone(),
                'received_timestamp': get_now_with_timezone(),
                'labels': ['inbox', 'important'],
                'sender_name': 'Sender Name',
                'sender_address': 'sender@example.com',
                'receiver': ['receiver@example.com'],
                'cc': [],
                'bcc': [],
                'subject': 'Email Subject',
                'body_type': 'html',
                'body_content': '<p>Email content</p>',
                'attachments': [],
                'create_timestamp': get_now_with_timezone(),
                'last_update_timestamp': get_now_with_timezone(),
                'message_id': 'msg_789',
            },
            data_id="email_123",
            data_type="Email",
            metadata={'original_id': 'email_123', 'source': 'email_mapper'},
        )

        # Serialize and deserialize
        json_str = original_data.to_json()
        restored_data = RawData.from_json_str(json_str)

        # Verify email-specific fields
        assert restored_data.content['user_id_list'] == ["user_001", "user_002"]
        assert restored_data.content['source'] == 'gmail'
        assert restored_data.content['is_read'] is True
        assert isinstance(restored_data.content['sent_timestamp'], datetime)
        assert isinstance(restored_data.content['received_timestamp'], datetime)
        assert restored_data.content['labels'] == ['inbox', 'important']
        assert restored_data.data_type == "Email"

    def test_linkdoc_data_serialization(self):
        """Test serialization of document data (simulating output from linkdoc_mapper)"""
        original_data = RawData(
            content={
                'user_id_list': ["user_001"],
                'title': 'Document Title',
                'content': 'Document content',
                'is_delete': False,
                'download_url': 'https://example.com/doc.pdf',
                'participants': ["user_001", "user_002"],
                'modify_timestamp': get_now_with_timezone(),
                'file_type': "pdf",
                'source_type': 'notion',
            },
            data_id="doc_456",
            data_type="LinkDoc",
            metadata={'original_id': 'doc_456', 'source': 'linkdoc_mapper'},
        )

        # Serialize and deserialize
        json_str = original_data.to_json()
        restored_data = RawData.from_json_str(json_str)

        # Verify document-specific fields
        assert restored_data.content['title'] == 'Document Title'
        assert restored_data.content['source_type'] == 'notion'
        assert isinstance(restored_data.content['modify_timestamp'], datetime)
        assert restored_data.content['participants'] == ["user_001", "user_002"]
        assert restored_data.data_type == "LinkDoc"

    def test_conversation_data_serialization(self):
        """Test serialization of conversation data (simulating output from format_transfer)"""
        test_time = get_now_with_timezone()

        original_data = RawData(
            content={
                "sender_name": "Zhang San",
                "receiverId": "room_123",
                "roomId": "room_123",
                "groupName": "Project Discussion Group",
                "userIdList": ["user_001", "user_002", "user_003"],
                "referList": [],
                "content": [
                    {
                        "type": "text",
                        "content": "Hello everyone, let's discuss the project progress today",
                    }
                ],
                "timestamp": test_time,
                "createBy": "user_001",
                "updateTime": test_time,
                "orgId": "org_456",
                "sender_id": "user_001",
                "msgType": 1,
            },
            data_id="conv_789",
            data_type="Conversation",
            metadata={
                "original_id": "conv_789",
                "createTime": test_time,
                "updateTime": test_time,
                "createBy": "user_001",
                "orgId": "org_456",
            },
        )

        # Serialize and deserialize
        json_str = original_data.to_json()
        restored_data = RawData.from_json_str(json_str)

        # Verify conversation-specific fields
        assert restored_data.content["sender_name"] == "Zhang San"
        assert restored_data.content["groupName"] == "Project Discussion Group"
        assert restored_data.content["userIdList"] == [
            "user_001",
            "user_002",
            "user_003",
        ]
        assert isinstance(restored_data.content["timestamp"], datetime)
        assert isinstance(restored_data.content["updateTime"], datetime)
        assert isinstance(restored_data.metadata["createTime"], datetime)

    def test_empty_and_none_values(self):
        """Test handling of empty and None values"""
        original_data = RawData(
            content={
                "required_field": "has value",
                "empty_string": "",
                "empty_list": [],
                "empty_dict": {},
                "none_value": None,
                "zero_value": 0,
                "false_value": False,
            },
            data_id="empty_test",
            data_type=None,  # Test optional field as None
            metadata=None,  # Test optional field as None
        )

        # Serialize and deserialize
        json_str = original_data.to_json()
        restored_data = RawData.from_json_str(json_str)

        # Verify correct handling of various empty values
        assert restored_data.content["required_field"] == "has value"
        assert restored_data.content["empty_string"] == ""
        assert restored_data.content["empty_list"] == []
        assert restored_data.content["empty_dict"] == {}
        assert restored_data.content["none_value"] is None
        assert restored_data.content["zero_value"] == 0
        assert restored_data.content["false_value"] is False
        assert restored_data.data_type is None
        assert restored_data.metadata is None

    def test_error_handling(self):
        """Test error handling"""
        # Test invalid JSON
        with pytest.raises(ValueError, match="JSON format error"):
            RawData.from_json_str("invalid json")

        # Test non-object JSON
        with pytest.raises(ValueError, match="JSON must be an object"):
            RawData.from_json_str('"string"')

        # Test missing required fields
        with pytest.raises(ValueError, match="JSON missing required field"):
            RawData.from_json_str('{"data_id": "test"}')  # missing content

        with pytest.raises(ValueError, match="JSON missing required field"):
            RawData.from_json_str('{"content": {}}')  # missing data_id

    def test_round_trip_consistency(self):
        """Test consistency of multiple serializations and deserializations"""
        # Create complex test data
        original_data = RawData(
            content={
                "mixed_types": {
                    "string": "text",
                    "number": 42,
                    "float": 3.14,
                    "boolean": True,
                    "null": None,
                    "datetime": get_now_with_timezone(),
                    "list": [1, "two", {"three": 3}],
                    "nested": {
                        "deep": {
                            "value": "deep nesting",
                            "timestamp": get_now_with_timezone(),
                        }
                    },
                }
            },
            data_id="round_trip_test",
            data_type="Test",
            metadata={
                "test_metadata": {"created": get_now_with_timezone(), "version": 1.0}
            },
        )

        # Perform multiple serializations and deserializations
        data = original_data
        for _ in range(3):
            json_str = data.to_json()
            data = RawData.from_json_str(json_str)

        # Verify final result matches original data
        assert data.content["mixed_types"]["string"] == "text"
        assert data.content["mixed_types"]["number"] == 42
        assert data.content["mixed_types"]["boolean"] is True
        assert isinstance(data.content["mixed_types"]["datetime"], datetime)
        assert data.content["mixed_types"]["nested"]["deep"]["value"] == "deep nesting"
        assert isinstance(
            data.content["mixed_types"]["nested"]["deep"]["timestamp"], datetime
        )
        assert isinstance(data.metadata["test_metadata"]["created"], datetime)

    # ==================== Improved field name heuristic judgment tests ====================

    def test_datetime_field_recognition(self):
        """Test time field recognition logic"""
        raw_data = RawData(content={}, data_id="test")

        # Test field names that should be recognized as time fields
        datetime_fields = [
            'timestamp',
            'createTime',
            'updateTime',
            'create_time',
            'update_time',
            'sent_timestamp',
            'received_timestamp',
            'create_timestamp',
            'last_update_timestamp',
            'modify_timestamp',
            'created_at',
            'updated_at',
            'joinTime',
            'leaveTime',
            'lastOnlineTime',
            'sync_time',
            'processed_at',
            'custom_time',
            'event_timestamp',
            'process_at',
            'end_date',
            'datetime',
            'created',
            'updated',
            'start_time',
            'end_time',
            'event_time',
            'build_timestamp',
        ]

        for field in datetime_fields:
            # Use protected method for testing - necessary for testing internal logic
            assert raw_data._is_datetime_field(
                field
            ), f"Field '{field}' should be recognized as a time field"  # pylint: disable=protected-access

        # Test field names that should not be recognized as time fields
        non_datetime_fields = [
            'content',
            'message',
            'user_id',
            'room_id',
            'title',
            'description',
            'count',
            'size',
            'type',
            'status',
            'version',
            'id',
            'name',
            'timeout',
            'runtime',
            'timeline',
            'timestamp_format',
            'time_zone',
            'time_limit',
            'timestamp_count',
            'timestamp_enabled',
            'time_sync',
        ]

        for field in non_datetime_fields:
            # Use protected method for testing - necessary for testing internal logic
            assert not raw_data._is_datetime_field(
                field
            ), f"Field '{field}' should not be recognized as a time field"  # pylint: disable=protected-access

    def test_datetime_content_vs_field_name(self):
        """Test time judgment based on field name rather than content"""
        test_time = get_now_with_timezone()
        iso_time_str = test_time.isoformat()

        # Create data containing ISO format strings but field names are not time fields
        original_data = RawData(
            content={
                "timestamp": test_time,  # Time field, should be converted
                "createTime": test_time,  # Time field, should be converted
                "message_content": iso_time_str,  # Content is time format but field name is not a time field, should not be converted
                "description": f"Event occurred at {iso_time_str}",  # Description containing time format, should not be converted
                "user_id": "2024-01-01T10:00:00Z",  # Looks like time but not a time field, should not be converted
                "version": "2024-01-01T10:00:00.123456+00:00",  # Version number happens to be in time format, should not be converted
            },
            data_id="field_test",
            data_type="Test",
        )

        # Serialize and deserialize
        json_str = original_data.to_json()
        restored_data = RawData.from_json_str(json_str)

        # Verify time fields are correctly converted
        assert isinstance(restored_data.content["timestamp"], datetime)
        assert isinstance(restored_data.content["createTime"], datetime)

        # Verify non-time fields remain as strings
        assert isinstance(restored_data.content["message_content"], str)
        assert isinstance(restored_data.content["description"], str)
        assert isinstance(restored_data.content["user_id"], str)
        assert isinstance(restored_data.content["version"], str)

        # Verify string content remains unchanged
        assert restored_data.content["message_content"] == iso_time_str
        assert (
            restored_data.content["description"] == f"Event occurred at {iso_time_str}"
        )
        assert restored_data.content["user_id"] == "2024-01-01T10:00:00Z"
        assert restored_data.content["version"] == "2024-01-01T10:00:00.123456+00:00"

    def test_real_world_conversation_data_improved(self):
        """Test real-world conversation data scenario (improved version)"""
        test_time = get_now_with_timezone()

        # Simulate real output from format_transfer.py
        original_data = RawData(
            content={
                "sender_name": "Zhang San",
                "receiverId": "room_123",
                "roomId": "room_123",
                "groupName": "Technical Discussion Group",
                "userIdList": ["user_001", "user_002"],
                "referList": [],
                "content": [
                    {
                        "type": "text",
                        "content": "Meeting time is set for 2024-01-01T10:00:00Z, remember to attend",
                    }
                ],  # Message content containing time format
                "timestamp": test_time,  # Actual time field
                "createBy": "user_001",
                "updateTime": test_time,  # Actual time field
                "orgId": "org_456",
                "sender_id": "user_001",
                "msgType": 1,
            },
            data_id="conv_001",
            data_type="Conversation",
            metadata={
                "original_id": "conv_001",
                "createTime": test_time,  # Actual time field
                "updateTime": test_time,  # Actual time field
                "createBy": "user_001",
                "orgId": "org_456",
            },
        )

        # Serialize and deserialize
        json_str = original_data.to_json()
        restored_data = RawData.from_json_str(json_str)

        # Verify time fields are correctly converted
        assert isinstance(restored_data.content["timestamp"], datetime)
        assert isinstance(restored_data.content["updateTime"], datetime)
        assert isinstance(restored_data.metadata["createTime"], datetime)
        assert isinstance(restored_data.metadata["updateTime"], datetime)

        # Verify time format strings in message content remain unchanged
        assert isinstance(restored_data.content["content"], list)
        assert "2024-01-01T10:00:00Z" in restored_data.content["content"][0]["content"]

        # Verify other field types are correct
        assert isinstance(restored_data.content["sender_name"], str)
        assert isinstance(restored_data.content["userIdList"], list)
        assert isinstance(restored_data.content["msgType"], int)

    def test_real_world_email_data_improved(self):
        """Test real-world email data scenario (improved version)"""
        test_time = get_now_with_timezone()

        # Simulate real output from email_mapper.py
        original_data = RawData(
            content={
                'user_id_list': ["user_001"],
                'id': 'email_123',
                'source': 'gmail',
                'subject': 'Meeting schedule: starting at 2024-01-01T14:00:00Z',  # Subject contains time format
                'body_content': 'The meeting will start at 2024-01-01T14:00:00+00:00, please attend on time',  # Body contains time format
                'sent_timestamp': test_time,  # Actual time field
                'received_timestamp': test_time,  # Actual time field
                'create_timestamp': test_time,  # Actual time field
                'last_update_timestamp': test_time,  # Actual time field
                'sender_name': 'Li Si',
                'sender_address': 'lisi@company.com',
            },
            data_id="email_123",
            data_type="Email",
        )

        # Serialize and deserialize
        json_str = original_data.to_json()
        restored_data = RawData.from_json_str(json_str)

        # Verify time fields are correctly converted
        assert isinstance(restored_data.content['sent_timestamp'], datetime)
        assert isinstance(restored_data.content['received_timestamp'], datetime)
        assert isinstance(restored_data.content['create_timestamp'], datetime)
        assert isinstance(restored_data.content['last_update_timestamp'], datetime)

        # Verify time format strings in text content remain unchanged
        assert isinstance(restored_data.content['subject'], str)
        assert isinstance(restored_data.content['body_content'], str)
        assert '2024-01-01T14:00:00Z' in restored_data.content['subject']
        assert '2024-01-01T14:00:00+00:00' in restored_data.content['body_content']

    def test_real_world_document_data_improved(self):
        """Test real-world document data scenario (improved version)"""
        test_time = get_now_with_timezone()

        # Simulate real output from linkdoc_mapper.py
        original_data = RawData(
            content={
                'user_id_list': ["user_001"],
                'title': 'Project Plan - Deadline 2024-12-31T23:59:59Z',  # Title contains time format
                'content': '# Project Plan\n\nStart time: 2024-01-01T09:00:00+00:00\nEnd time: 2024-12-31T18:00:00+00:00',  # Content contains time format
                'modify_timestamp': test_time,  # Actual time field
                'last_update_timestamp': test_time,  # Actual time field
                'source_type': 'notion',
                'file_type': 'markdown',
            },
            data_id="doc_123",
            data_type="LinkDoc",
        )

        # Serialize and deserialize
        json_str = original_data.to_json()
        restored_data = RawData.from_json_str(json_str)

        # Verify time fields are correctly converted
        assert isinstance(restored_data.content['modify_timestamp'], datetime)
        assert isinstance(restored_data.content['last_update_timestamp'], datetime)

        # Verify time format strings in text content remain unchanged
        assert isinstance(restored_data.content['title'], str)
        assert isinstance(restored_data.content['content'], str)
        assert '2024-12-31T23:59:59Z' in restored_data.content['title']
        assert '2024-01-01T09:00:00+00:00' in restored_data.content['content']
        assert '2024-12-31T18:00:00+00:00' in restored_data.content['content']

    def test_nested_structure_with_mixed_content(self):
        """Test nested structures with mixed content"""
        test_time = get_now_with_timezone()

        original_data = RawData(
            content={
                "message_info": {
                    "content": [
                        {
                            "type": "text",
                            "content": "Scheduled task will execute at 2024-01-01T10:00:00Z",
                        }
                    ],  # Non-time field but contains time format
                    "timestamp": test_time,  # Time field
                    "createTime": test_time,  # Time field
                    "author": {
                        "name": "Zhang San",
                        "last_login_time": test_time,  # Time field
                        "profile_description": "User registration time: 2024-01-01T08:00:00+00:00",  # Non-time field but contains time format
                    },
                },
                "system_info": {
                    "version": "2024.01.01T10.00.00",  # Version number, not a time field
                    "build_timestamp": test_time,  # Time field
                    "config": {
                        "timeout": "2024-01-01T10:00:00Z",  # Timeout configuration, not a time field
                        "start_time": test_time,  # Time field
                        "schedule": "Execute daily at 2024-01-01T10:00:00Z",  # Schedule description, not a time field
                    },
                },
            },
            data_id="nested_test",
            data_type="Test",
        )

        # Serialize and deserialize
        json_str = original_data.to_json()
        restored_data = RawData.from_json_str(json_str)

        # Verify time fields are correctly converted
        assert isinstance(restored_data.content["message_info"]["timestamp"], datetime)
        assert isinstance(restored_data.content["message_info"]["createTime"], datetime)
        assert isinstance(
            restored_data.content["message_info"]["author"]["last_login_time"], datetime
        )
        assert isinstance(
            restored_data.content["system_info"]["build_timestamp"], datetime
        )
        assert isinstance(
            restored_data.content["system_info"]["config"]["start_time"], datetime
        )

        # Verify non-time fields remain as their original types
        assert isinstance(restored_data.content["message_info"]["content"], list)
        assert isinstance(
            restored_data.content["message_info"]["author"]["profile_description"], str
        )
        assert isinstance(restored_data.content["system_info"]["version"], str)
        assert isinstance(
            restored_data.content["system_info"]["config"]["timeout"], str
        )
        assert isinstance(
            restored_data.content["system_info"]["config"]["schedule"], str
        )

        # Verify string content remains unchanged
        assert (
            "2024-01-01T10:00:00Z" in restored_data.content["message_info"]["content"][0]["content"]
        )
        assert (
            "2024-01-01T08:00:00+00:00"
            in restored_data.content["message_info"]["author"]["profile_description"]
        )
        assert restored_data.content["system_info"]["version"] == "2024.01.01T10.00.00"
        assert (
            restored_data.content["system_info"]["config"]["timeout"]
            == "2024-01-01T10:00:00Z"
        )
        assert (
            "2024-01-01T10:00:00Z"
            in restored_data.content["system_info"]["config"]["schedule"]
        )

    def test_edge_cases_improved(self):
        """Test edge cases (improved version)"""
        test_time = get_now_with_timezone()

        original_data = RawData(
            content={
                # Empty strings and None values
                "timestamp": test_time,
                "empty_timestamp": "",
                "null_timestamp": None,
                # Numeric field names containing time keywords
                "timestamp_count": 5,
                "time_limit": 3600,
                # Boolean field names containing time keywords
                "timestamp_enabled": True,
                "time_sync": False,
                # Time fields in lists
                "events": [
                    {
                        "name": "Event 1",
                        "timestamp": test_time,  # Should be converted
                        "description": "Happens at 2024-01-01T10:00:00Z",  # Should not be converted
                    },
                    {
                        "name": "Event 2",
                        "event_time": test_time,  # Should be converted
                        "note": "Scheduled to execute at 2024-01-02T14:00:00+00:00",  # Should not be converted
                    },
                ],
            },
            data_id="edge_test",
            data_type="Test",
        )

        # Serialize and deserialize
        json_str = original_data.to_json()
        restored_data = RawData.from_json_str(json_str)

        # Verify time fields are correctly handled
        assert isinstance(restored_data.content["timestamp"], datetime)

        # Verify empty values remain unchanged
        assert restored_data.content["empty_timestamp"] == ""
        assert restored_data.content["null_timestamp"] is None

        # Verify non-string values retain their original types
        assert isinstance(restored_data.content["timestamp_count"], int)
        assert isinstance(restored_data.content["time_limit"], int)
        assert isinstance(restored_data.content["timestamp_enabled"], bool)
        assert isinstance(restored_data.content["time_sync"], bool)

        # Verify nested handling in lists
        assert isinstance(restored_data.content["events"][0]["timestamp"], datetime)
        assert isinstance(restored_data.content["events"][1]["event_time"], datetime)
        assert isinstance(restored_data.content["events"][0]["description"], str)
        assert isinstance(restored_data.content["events"][1]["note"], str)

        # Verify string content remains unchanged
        assert (
            "2024-01-01T10:00:00Z" in restored_data.content["events"][0]["description"]
        )
        assert "2024-01-02T14:00:00+00:00" in restored_data.content["events"][1]["note"]


if __name__ == "__main__":
    # Run tests
    test_instance = TestRawDataJsonSerialization()

    print("Starting to test RawData JSON serialization functionality...")

    try:
        # Basic functionality tests
        test_instance.test_basic_serialization()
        print("✅ Basic serialization test passed")

        test_instance.test_datetime_serialization()
        print("✅ Datetime serialization test passed")

        test_instance.test_nested_structure_serialization()
        print("✅ Nested structure serialization test passed")

        test_instance.test_email_data_serialization()
        print("✅ Email data serialization test passed")

        test_instance.test_linkdoc_data_serialization()
        print("✅ Document data serialization test passed")

        test_instance.test_conversation_data_serialization()
        print("✅ Conversation data serialization test passed")

        test_instance.test_empty_and_none_values()
        print("✅ Empty value handling test passed")

        test_instance.test_error_handling()
        print("✅ Error handling test passed")

        test_instance.test_round_trip_consistency()
        print("✅ Round-trip consistency test passed")

        # Improved field name heuristic judgment tests
        print("\n--- Improved field name heuristic judgment tests ---")

        test_instance.test_datetime_field_recognition()
        print("✅ Time field recognition test passed")

        test_instance.test_datetime_content_vs_field_name()
        print("✅ Field name vs content judgment test passed")

        test_instance.test_real_world_conversation_data_improved()
        print("✅ Real-world conversation data test (improved version) passed")

        test_instance.test_real_world_email_data_improved()
        print("✅ Real-world email data test (improved version) passed")

        test_instance.test_real_world_document_data_improved()
        print("✅ Real-world document data test (improved version) passed")

        test_instance.test_nested_structure_with_mixed_content()
        print("✅ Nested structure mixed content test passed")

        test_instance.test_edge_cases_improved()
        print("✅ Edge case test (improved version) passed")

        print(
            "\n🎉 All tests passed! RawData JSON serialization functionality is working correctly."
        )
        print("\nMain features and improvements:")
        print("- ✅ Complete JSON serialization and deserialization support")
        print(
            "- ✅ Intelligent time field recognition based on field names rather than content"
        )
        print(
            "- ✅ Avoids mistakenly converting time format strings in message content to datetime"
        )
        print("- ✅ Supports all common time field naming patterns in the project")
        print("- ✅ Correctly handles mixed content types in nested structures")
        print("- ✅ Comprehensive error handling and edge case handling")

    except (AssertionError, ValueError, TypeError) as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()

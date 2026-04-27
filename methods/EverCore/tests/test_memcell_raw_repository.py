#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test the functionality of MemCellRawRepository

Test contents include:
1. CRUD operations based on event_id
2. Queries based on user_id
3. Queries based on time range (including segmented queries)
4. Queries based on group_id
5. Queries based on participants
6. Queries based on keywords
7. Batch deletion operations
8. Statistical and aggregation queries
"""

import asyncio
from common_utils.datetime_utils import get_now_with_timezone
from datetime import timedelta, datetime
from bson import ObjectId
from pydantic import BaseModel, Field

from core.di import get_bean_by_type
from infra_layer.adapters.out.persistence.repository.memcell_raw_repository import (
    MemCellRawRepository,
)
from infra_layer.adapters.out.persistence.document.memory.memcell import (
    MemCell,
    DataTypeEnum,
)
from core.observation.logger import get_logger

logger = get_logger(__name__)


# ==================== Projection Model Definition ====================
class MemCellProjection(BaseModel):
    """
    MemCell projection model - used to test field projection functionality
    Includes only partial fields, excluding large fields such as original_data
    """

    id: ObjectId = Field(alias="_id")
    user_id: str
    timestamp: datetime
    summary: str
    type: DataTypeEnum

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True


async def test_basic_crud_operations():
    """Test basic CRUD operations based on event_id"""
    logger.info("Starting test of basic CRUD operations based on event_id...")

    repo = get_bean_by_type(MemCellRawRepository)
    user_id = "test_user_001"

    try:
        # First clean up any existing test data (使用硬删除清理测试数据)
        await repo.hard_delete_by_user_id(user_id)
        logger.info("✅ Cleaned up existing test data")

        # Test creating a new MemCell
        now = get_now_with_timezone()
        memcell = MemCell(
            user_id=user_id,
            timestamp=now,
            summary="This is a test memory: discussed the project's technical solution",
            type=DataTypeEnum.CONVERSATION,
            keywords=["technical solution", "project discussion"],
            participants=["Zhang San", "Li Si"],
        )

        created = await repo.append_memcell(memcell)
        assert created is not None
        assert created.user_id == user_id
        assert created.summary == "This is a test memory: discussed the project's technical solution"
        assert created.event_id is not None
        logger.info("✅ Test creating new MemCell succeeded, event_id=%s", created.event_id)

        event_id = str(created.event_id)

        # Test querying by event_id
        queried = await repo.get_by_event_id(event_id)
        assert queried is not None
        assert queried.user_id == user_id
        assert str(queried.event_id) == event_id
        logger.info("✅ Test querying by event_id succeeded")

        # Test updating MemCell
        update_data = {
            "summary": "Updated summary: project technical solution has been confirmed",
            "keywords": ["technical solution", "project discussion", "confirmed"],
        }

        updated = await repo.update_by_event_id(event_id, update_data)
        assert updated is not None
        assert updated.summary == "Updated summary: project technical solution has been confirmed"
        assert len(updated.keywords) == 3
        logger.info("✅ Test updating MemCell succeeded")

        # Test deleting MemCell
        deleted = await repo.delete_by_event_id(event_id)
        assert deleted is True
        logger.info("✅ Test deleting MemCell succeeded")

        # Verify deletion
        final_check = await repo.get_by_event_id(event_id)
        assert final_check is None, "Record should have been deleted"
        logger.info("✅ Verified deletion succeeded")

    except Exception as e:
        logger.error("❌ Basic CRUD operations test failed: %s", e)
        raise

    logger.info("✅ Basic CRUD operations test completed")


async def test_find_by_user_id():
    """Test queries based on user_id"""
    logger.info("Starting test of queries based on user_id...")

    repo = get_bean_by_type(MemCellRawRepository)
    user_id = "test_user_002"

    try:
        # First clean up (使用硬删除清理测试数据)
        await repo.hard_delete_by_user_id(user_id)
        logger.info("✅ Cleaned up existing test data")

        # Create multiple records
        now = get_now_with_timezone()
        for i in range(5):
            memcell = MemCell(
                user_id=user_id,
                timestamp=now - timedelta(hours=i),
                summary=f"Test memory {i+1}",
                type=DataTypeEnum.CONVERSATION,
            )
            await repo.append_memcell(memcell)

        logger.info("✅ Created 5 test records")

        # Test querying all records (descending)
        results = await repo.find_by_user_id(user_id, sort_desc=True)
        assert len(results) == 5
        assert results[0].summary == "Test memory 1"  # Latest
        logger.info("✅ Test querying all records (descending) succeeded")

        # Test querying all records (ascending)
        results_asc = await repo.find_by_user_id(user_id, sort_desc=False)
        assert len(results_asc) == 5
        assert results_asc[0].summary == "Test memory 5"  # Earliest
        logger.info("✅ Test querying all records (ascending) succeeded")

        # Test limiting number
        limited_results = await repo.find_by_user_id(user_id, limit=2)
        assert len(limited_results) == 2
        logger.info("✅ Test limiting number succeeded")

        # Test skip and limit
        skip_results = await repo.find_by_user_id(user_id, skip=2, limit=2)
        assert len(skip_results) == 2
        logger.info("✅ Test skip and limit succeeded")

        # Clean up (使用硬删除清理测试数据)
        await repo.hard_delete_by_user_id(user_id)
        logger.info("✅ Cleaned up test data successfully")

    except Exception as e:
        logger.error("❌ Test based on user_id query failed: %s", e)
        raise

    logger.info("✅ Queries based on user_id test completed")


async def test_find_by_time_range():
    """Test queries based on time range (including segmented queries)"""
    logger.info("Starting test of queries based on time range...")

    repo = get_bean_by_type(MemCellRawRepository)
    user_id = "test_user_003"

    try:
        # First clean up (使用硬删除清理测试数据)
        await repo.hard_delete_by_user_id(user_id)
        logger.info("✅ Cleaned up existing test data")

        # Create test data with a large span (10 days)
        # Use time from 1990 to avoid conflicts with existing data
        # Note: Must use timezone-aware time, otherwise it will not match the timezone stored in MongoDB
        from common_utils.datetime_utils import get_timezone

        tz = get_timezone()
        start_time = datetime(1990, 1, 1, 0, 0, 0, tzinfo=tz)

        # Create one record per day
        created_timestamps = []
        for i in range(10):
            ts = start_time + timedelta(days=i)
            created_timestamps.append(ts)
            memcell = MemCell(
                user_id=user_id,
                timestamp=ts,
                summary=f"Day {i+1} memory",
                type=DataTypeEnum.CONVERSATION,
            )
            await repo.append_memcell(memcell)

        logger.info("✅ Created 10 days of test data")
        logger.info(
            "   Timestamp range: %s to %s", created_timestamps[0], created_timestamps[-1]
        )

        # Test small range query (3 days, does not trigger segmentation)
        # Query day 0, 1, 2 (total 3 records)
        small_start = start_time  # 1990-01-01 00:00:00
        small_end = start_time + timedelta(days=3)  # 1990-01-04 00:00:00 (exclusive)
        small_results = await repo.find_by_time_range(small_start, small_end)
        logger.info("   Small range query returned %d records (expected 3)", len(small_results))
        assert (
            len(small_results) == 3
        ), f"Expected 3 records, got {len(small_results)}"
        logger.info("✅ Test small range query (3 days) succeeded, found %d records", len(small_results))

        # Test large range query (10 days, triggers segmented query)
        # Query day 0-9 (total 10 records)
        # The last record is 1990-01-10 00:00:00, query uses $lt, so end time must be > 1990-01-10
        large_start = start_time  # 1990-01-01 00:00:00
        large_end = start_time + timedelta(
            days=10, seconds=1
        )  # 1990-01-11 00:00:01 (ensure day 9 is included)
        logger.info("   Query time range: %s to %s", large_start, large_end)
        large_results = await repo.find_by_time_range(large_start, large_end)
        logger.info("   Large range query returned %d records (expected 10)", len(large_results))

        # Print returned record timestamps for debugging
        logger.info("   Returned record details:")
        for idx, mc in enumerate(large_results):
            logger.info("     [%d] %s - %s", idx, mc.timestamp, mc.summary)

        if len(large_results) != 10:
            logger.warning("   ⚠️ Record count mismatch!")
            logger.warning("   Expected timestamps:")
            for idx, ts in enumerate(created_timestamps):
                logger.warning("     [%d] %s", idx, ts)

            # Find missing records
            returned_timestamps = {mc.timestamp for mc in large_results}
            missing = [ts for ts in created_timestamps if ts not in returned_timestamps]
            if missing:
                logger.error("   ❌ Missing timestamps:")
                for ts in missing:
                    logger.error("     - %s", ts)

        assert (
            len(large_results) == 10
        ), f"Expected 10 records, got {len(large_results)}"
        logger.info("✅ Test large range query (10 days) succeeded, found %d records", len(large_results))

        # Test descending query
        desc_results = await repo.find_by_time_range(
            large_start, large_end, sort_desc=True
        )
        assert len(desc_results) == 10
        assert "Day 10" in desc_results[0].summary  # Latest first
        logger.info("✅ Test descending query succeeded")

        # Test ascending query
        asc_results = await repo.find_by_time_range(
            large_start, large_end, sort_desc=False
        )
        assert len(asc_results) == 10
        assert "Day 1" in asc_results[0].summary  # Earliest first
        logger.info("✅ Test ascending query succeeded")

        # Test pagination
        page_results = await repo.find_by_time_range(large_start, large_end, limit=5)
        assert len(page_results) == 5
        logger.info("✅ Test pagination succeeded")

        # Clean up (使用硬删除清理测试数据)
        await repo.hard_delete_by_user_id(user_id)
        logger.info("✅ Cleaned up test data successfully")

    except Exception as e:
        logger.error("❌ Test time range query failed: %s", e)
        raise

    logger.info("✅ Time range query test completed")


async def test_find_by_user_and_time_range():
    """Test queries based on user and time range"""
    logger.info("Starting test of queries based on user and time range...")

    repo = get_bean_by_type(MemCellRawRepository)
    user_id = "test_user_004"

    try:
        # First clean up (使用硬删除清理测试数据)
        await repo.hard_delete_by_user_id(user_id)
        logger.info("✅ Cleaned up existing test data")

        # Create test data
        now = get_now_with_timezone()
        start_time = now - timedelta(days=5)

        for i in range(5):
            memcell = MemCell(
                user_id=user_id,
                timestamp=start_time + timedelta(days=i),
                summary=f"User memory {i+1}",
                type=DataTypeEnum.CONVERSATION,
            )
            await repo.append_memcell(memcell)

        logger.info("✅ Created 5 test records")

        # Test querying data for middle 3 days
        query_start = start_time + timedelta(days=1)
        query_end = start_time + timedelta(days=4)
        results = await repo.find_by_user_and_time_range(
            user_id, query_start, query_end
        )

        assert len(results) == 3
        logger.info("✅ Test user and time range query succeeded, found %d records", len(results))

        # Clean up (使用硬删除清理测试数据)
        await repo.hard_delete_by_user_id(user_id)
        logger.info("✅ Cleaned up test data successfully")

    except Exception as e:
        logger.error("❌ Test user and time range query failed: %s", e)
        raise

    logger.info("✅ User and time range query test completed")


async def test_find_by_group_id():
    """Test queries based on group_id"""
    logger.info("Starting test of queries based on group_id...")

    repo = get_bean_by_type(MemCellRawRepository)
    user_id = "test_user_005"
    group_id = "test_group_001"

    try:
        # First clean up (使用硬删除清理测试数据)
        await repo.hard_delete_by_user_id(user_id)
        logger.info("✅ Cleaned up existing test data")

        # Create group records
        now = get_now_with_timezone()
        for i in range(3):
            memcell = MemCell(
                user_id=user_id,
                group_id=group_id,
                timestamp=now - timedelta(hours=i),
                summary=f"Group memory {i+1}",
                type=DataTypeEnum.CONVERSATION,
            )
            await repo.append_memcell(memcell)

        logger.info("✅ Created 3 group records")

        # Test query
        results = await repo.find_by_group_id(group_id)
        assert len(results) == 3
        logger.info("✅ Test querying by group_id succeeded, found %d records", len(results))

        # Clean up (使用硬删除清理测试数据)
        await repo.hard_delete_by_user_id(user_id)
        logger.info("✅ Cleaned up test data successfully")

    except Exception as e:
        logger.error("❌ Test group_id query failed: %s", e)
        raise

    logger.info("✅ group_id query test completed")


async def test_find_by_participants():
    """Test queries based on participants"""
    logger.info("Starting test of queries based on participants...")

    repo = get_bean_by_type(MemCellRawRepository)
    user_id = "test_user_006"

    try:
        # First clean up (使用硬删除清理测试数据)
        await repo.hard_delete_by_user_id(user_id)
        logger.info("✅ Cleaned up existing test data")

        # Create test data
        now = get_now_with_timezone()

        # Record 1: Zhang San, Li Si
        memcell1 = MemCell(
            user_id=user_id,
            timestamp=now - timedelta(hours=1),
            summary="Record 1: Conversation between Zhang San and Li Si",
            participants=["Zhang San", "Li Si"],
        )
        await repo.append_memcell(memcell1)

        # Record 2: Zhang San, Wang Wu
        memcell2 = MemCell(
            user_id=user_id,
            timestamp=now - timedelta(hours=2),
            summary="Record 2: Conversation between Zhang San and Wang Wu",
            participants=["Zhang San", "Wang Wu"],
        )
        await repo.append_memcell(memcell2)

        # Record 3: Li Si, Wang Wu
        memcell3 = MemCell(
            user_id=user_id,
            timestamp=now - timedelta(hours=3),
            summary="Record 3: Conversation between Li Si and Wang Wu",
            participants=["Li Si", "Wang Wu"],
        )
        await repo.append_memcell(memcell3)

        logger.info("✅ Created 3 test records")

        # Test matching any participant (containing "Zhang San")
        results_any = await repo.find_by_participants(["Zhang San"], match_all=False)
        assert len(results_any) == 2
        logger.info("✅ Test matching any participant succeeded, found %d records", len(results_any))

        # Test matching all participants (containing both "Zhang San" and "Li Si")
        results_all = await repo.find_by_participants(["Zhang San", "Li Si"], match_all=True)
        assert len(results_all) == 1
        logger.info("✅ Test matching all participants succeeded, found %d records", len(results_all))

        # Clean up (使用硬删除清理测试数据)
        await repo.hard_delete_by_user_id(user_id)
        logger.info("✅ Cleaned up test data successfully")

    except Exception as e:
        logger.error("❌ Test participant query failed: %s", e)
        raise

    logger.info("✅ Participant query test completed")


async def test_search_by_keywords():
    """Test queries based on keywords"""
    logger.info("Starting test of queries based on keywords...")

    repo = get_bean_by_type(MemCellRawRepository)
    user_id = "test_user_007"

    try:
        # First clean up (使用硬删除清理测试数据)
        await repo.hard_delete_by_user_id(user_id)
        logger.info("✅ Cleaned up existing test data")

        # Create test data
        now = get_now_with_timezone()

        # Record 1: technology, Python
        memcell1 = MemCell(
            user_id=user_id,
            timestamp=now - timedelta(hours=1),
            summary="Record 1: Python technology discussion",
            keywords=["technology", "Python"],
        )
        await repo.append_memcell(memcell1)

        # Record 2: technology, Java
        memcell2 = MemCell(
            user_id=user_id,
            timestamp=now - timedelta(hours=2),
            summary="Record 2: Java technology discussion",
            keywords=["technology", "Java"],
        )
        await repo.append_memcell(memcell2)

        # Record 3: design, architecture
        memcell3 = MemCell(
            user_id=user_id,
            timestamp=now - timedelta(hours=3),
            summary="Record 3: Architecture design discussion",
            keywords=["design", "architecture"],
        )
        await repo.append_memcell(memcell3)

        logger.info("✅ Created 3 test records")

        # Test matching any keyword (containing "technology")
        results_any = await repo.search_by_keywords(["technology"], match_all=False)
        assert len(results_any) == 2
        logger.info("✅ Test matching any keyword succeeded, found %d records", len(results_any))

        # Test matching all keywords (containing both "technology" and "Python")
        results_all = await repo.search_by_keywords(["technology", "Python"], match_all=True)
        assert len(results_all) == 1
        logger.info("✅ Test matching all keywords succeeded, found %d records", len(results_all))

        # Clean up (使用硬删除清理测试数据)
        await repo.hard_delete_by_user_id(user_id)
        logger.info("✅ Cleaned up test data successfully")

    except Exception as e:
        logger.error("❌ Test keyword query failed: %s", e)
        raise

    logger.info("✅ Keyword query test completed")


async def test_batch_delete_operations():
    """Test batch deletion operations (now soft delete by default)"""
    logger.info("Starting test of batch deletion operations...")

    repo = get_bean_by_type(MemCellRawRepository)
    user_id = "test_user_008"

    try:
        # First clean up (使用硬删除清理测试数据)
        await repo.hard_delete_by_user_id(user_id)
        logger.info("✅ Cleaned up existing test data")

        # Create test data
        now = get_now_with_timezone()
        for i in range(10):
            memcell = MemCell(
                user_id=user_id,
                timestamp=now - timedelta(days=i),
                summary=f"Test memory {i+1}",
                type=DataTypeEnum.CONVERSATION,
            )
            await repo.append_memcell(memcell)

        logger.info("✅ Created 10 test records")

        # Test soft deleting records within a time range (first 5 days)
        delete_start = now - timedelta(days=5)
        delete_end = now
        deleted_count = await repo.delete_by_time_range(
            delete_start, delete_end, user_id=user_id, deleted_by="test"
        )
        assert deleted_count == 5
        logger.info("✅ Test soft deleting records within time range succeeded, deleted %d records", deleted_count)

        # Verify remaining records (常规查询只返回未删除的)
        remaining = await repo.find_by_user_id(user_id)
        assert len(remaining) == 5
        logger.info("✅ Verified remaining records successfully, %d records left", len(remaining))

        # Test soft deleting all remaining user records
        total_deleted = await repo.delete_by_user_id(user_id, deleted_by="test")
        assert total_deleted == 5
        logger.info("✅ Test soft deleting all user records succeeded, deleted %d records", total_deleted)

        # Verify all soft deleted (常规查询找不到)
        final_check = await repo.find_by_user_id(user_id)
        assert len(final_check) == 0
        logger.info("✅ Verified all soft deleted successfully (not visible in regular queries)")
        
        # Verify using hard_find_many can still find them
        hard_check = await MemCell.hard_find_many({"user_id": user_id}).to_list()
        assert len(hard_check) == 10
        logger.info("✅ Verified all 10 records still exist (soft deleted)")
        
        # Final cleanup with hard delete
        await repo.hard_delete_by_user_id(user_id)
        logger.info("✅ Hard deleted all records for cleanup")

    except Exception as e:
        logger.error("❌ Test batch deletion operations failed: %s", e)
        raise

    logger.info("✅ Batch deletion operations test completed")


async def test_statistics_and_aggregation():
    """Test statistical and aggregation queries"""
    logger.info("Starting test of statistical and aggregation queries...")

    repo = get_bean_by_type(MemCellRawRepository)
    user_id = "test_user_009"

    try:
        # First clean up (使用硬删除清理测试数据)
        await repo.hard_delete_by_user_id(user_id)
        logger.info("✅ Cleaned up existing test data")

        # Create test data of different types
        now = get_now_with_timezone()
        start_time = now - timedelta(days=7)

        # Create 6 conversation memories (Note: Originally 3 conversations, 2 emails, 1 document, but now only CONVERSATION type)
        for i in range(3):
            memcell = MemCell(
                user_id=user_id,
                timestamp=start_time + timedelta(days=i),
                summary=f"Conversation memory {i+1}",
                type=DataTypeEnum.CONVERSATION,
            )
            await repo.append_memcell(memcell)

        for i in range(2):
            memcell = MemCell(
                user_id=user_id,
                timestamp=start_time + timedelta(days=i + 3),
                summary=f"Email memory {i+1}",
                type=DataTypeEnum.CONVERSATION,
            )
            await repo.append_memcell(memcell)

        memcell = MemCell(
            user_id=user_id,
            timestamp=start_time + timedelta(days=5),
            summary="Document memory",
            type=DataTypeEnum.CONVERSATION,
        )
        await repo.append_memcell(memcell)

        logger.info("✅ Created 6 test records (all CONVERSATION type)")

        # Test counting total user records
        total_count = await repo.count_by_user_id(user_id)
        assert total_count == 6
        logger.info("✅ Test counting total user records succeeded, total %d records", total_count)

        # Test counting records within a time range
        range_start = start_time
        range_end = start_time + timedelta(days=4)
        range_count = await repo.count_by_time_range(
            range_start, range_end, user_id=user_id
        )
        assert range_count == 4  # Records from first 4 days (3 conversation memories + 1 email memory)
        logger.info("✅ Test counting records within time range succeeded, total %d records", range_count)

        # Test getting user's latest records
        latest = await repo.get_latest_by_user(user_id, limit=3)
        assert len(latest) == 3
        assert latest[0].summary == "Document memory"  # Latest
        logger.info("✅ Test getting user's latest records succeeded")

        # Clean up (使用硬删除清理测试数据)
        await repo.hard_delete_by_user_id(user_id)
        logger.info("✅ Cleaned up test data successfully")

    except Exception as e:
        logger.error("❌ Test statistical and aggregation queries failed: %s", e)
        raise

    logger.info("✅ Statistical and aggregation queries test completed")


async def test_get_by_event_ids():
    """Test batch query by event_ids"""
    logger.info("Starting test of batch query by event_ids...")

    repo = get_bean_by_type(MemCellRawRepository)
    user_id = "test_user_010"

    try:
        # First clean up (使用硬删除清理测试数据)
        await repo.hard_delete_by_user_id(user_id)
        logger.info("✅ Cleaned up existing test data")

        # Create test data
        now = get_now_with_timezone()
        created_memcells = []

        for i in range(5):
            memcell = MemCell(
                user_id=user_id,
                timestamp=now - timedelta(hours=i),
                summary=f"Test memory {i+1}",
                episode=f"This is the detailed content of test memory {i+1}",
                type=DataTypeEnum.CONVERSATION,
                keywords=[f"keyword{i+1}", "test"],
            )
            created = await repo.append_memcell(memcell)
            created_memcells.append(created)

        logger.info("✅ Created 5 test records")

        # Prepare event_ids
        event_ids = [str(mc.event_id) for mc in created_memcells[:3]]
        logger.info("   Preparing to query event_ids: %s", event_ids)

        # Test 1: Batch query (without projection)
        results = await repo.get_by_event_ids(event_ids)
        assert isinstance(results, dict), "Return result should be a dictionary"
        assert len(results) == 3, f"Should return 3 records, got {len(results)}"

        # Verify returned is a dictionary, key is event_id
        for event_id in event_ids:
            assert event_id in results, f"event_id {event_id} should be in results"
            memcell = results[event_id]
            assert memcell.user_id == user_id
            assert memcell.episode is not None

        logger.info("✅ Test batch query (without projection) succeeded, returned %d records", len(results))

        # Test 2: Batch query (with field projection)
        # Use Pydantic projection model to return only specified fields, excluding large fields like original_data
        results_with_projection = await repo.get_by_event_ids(
            event_ids, projection_model=MemCellProjection
        )

        assert isinstance(results_with_projection, dict), "Return result should be a dictionary"
        assert (
            len(results_with_projection) == 3
        ), f"Should return 3 records, got {len(results_with_projection)}"

        # Verify projection effect: returned should be MemCellProjection instances
        for event_id, memcell_projection in results_with_projection.items():
            assert isinstance(
                memcell_projection, MemCellProjection
            ), "Returned should be MemCellProjection instance"
            assert memcell_projection.summary is not None, "summary field should exist"
            assert memcell_projection.timestamp is not None, "timestamp field should exist"
            assert memcell_projection.type is not None, "type field should exist"
            assert memcell_projection.user_id == user_id, "user_id should match"
            # Verify fields not defined in projection model are not included
            assert not hasattr(
                memcell_projection, 'original_data'
            ), "original_data field should not exist"
            assert not hasattr(memcell_projection, 'episode'), "episode field should not exist"

        logger.info(
            "✅ Test batch query (with field projection) succeeded, returned %d records",
            len(results_with_projection),
        )

        # Test 3: Query partially valid event_ids (including an invalid one)
        mixed_event_ids = event_ids[:2] + ["invalid_id_123", "507f1f77bcf86cd799439011"]
        results_mixed = await repo.get_by_event_ids(mixed_event_ids)

        # Should only return 2 valid records
        assert (
            len(results_mixed) == 2
        ), f"Should return 2 records, got {len(results_mixed)}"
        assert event_ids[0] in results_mixed
        assert event_ids[1] in results_mixed
        assert "invalid_id_123" not in results_mixed
        assert "507f1f77bcf86cd799439011" not in results_mixed

        logger.info(
            "✅ Test querying partially valid event_ids succeeded, returned %d records", len(results_mixed)
        )

        # Test 4: Empty list input
        results_empty = await repo.get_by_event_ids([])
        assert isinstance(results_empty, dict), "Return result should be a dictionary"
        assert len(results_empty) == 0, "Empty list should return empty dictionary"
        logger.info("✅ Test empty list input succeeded")

        # Test 5: Query non-existent event_ids
        non_existent_ids = ["507f1f77bcf86cd799439011", "507f1f77bcf86cd799439012"]
        results_non_existent = await repo.get_by_event_ids(non_existent_ids)
        assert isinstance(results_non_existent, dict), "Return result should be a dictionary"
        assert len(results_non_existent) == 0, "Non-existent event_ids should return empty dictionary"
        logger.info("✅ Test querying non-existent event_ids succeeded")

        # Test 6: Verify returned data integrity
        first_event_id = event_ids[0]
        first_memcell = results[first_event_id]
        original_memcell = created_memcells[0]

        assert str(first_memcell.event_id) == str(original_memcell.event_id)
        assert first_memcell.summary == original_memcell.summary
        assert first_memcell.user_id == original_memcell.user_id
        logger.info("✅ Verified returned data integrity succeeded")

        # Clean up (使用硬删除清理测试数据)
        await repo.hard_delete_by_user_id(user_id)
        logger.info("✅ Cleaned up test data successfully")

    except Exception as e:
        logger.error("❌ Test batch query by event_ids failed: %s", e)
        import traceback

        logger.error("Detailed error: %s", traceback.format_exc())
        raise

    logger.info("✅ Batch query by event_ids test completed")


async def test_soft_delete_single():
    """测试单个软删除功能"""
    logger.info("Starting test of soft delete single record...")
    
    repo = get_bean_by_type(MemCellRawRepository)
    user_id = "test_user_soft_delete_001"
    
    try:
        # 清理旧数据
        await repo.hard_delete_by_user_id(user_id)
        logger.info("✅ Cleaned up existing test data")
        
        # 创建测试数据
        now = get_now_with_timezone()
        memcell = MemCell(
            user_id=user_id,
            timestamp=now,
            summary="测试软删除的记录",
            type=DataTypeEnum.CONVERSATION,
        )
        created = await repo.append_memcell(memcell)
        event_id = str(created.event_id)
        logger.info("✅ Created test record, event_id=%s", event_id)
        
        # 验证可以查到
        found = await repo.get_by_event_id(event_id)
        assert found is not None, "Should be able to find the record"
        assert not found.is_deleted(), "Record should not be marked as deleted"
        logger.info("✅ Verified: Can query the record before deletion")
        
        # 执行软删除
        deleted = await repo.delete_by_event_id(event_id, deleted_by="test_admin")
        assert deleted is True, "Soft delete should succeed"
        logger.info("✅ Soft delete succeeded")
        
        # 验证常规查询找不到
        not_found = await repo.get_by_event_id(event_id)
        assert not_found is None, "Regular query should not find soft-deleted record"
        logger.info("✅ Verified: Regular query cannot find soft-deleted record")
        
        # 使用 hard_find_one 可以找到
        hard_found = await MemCell.hard_find_one({"_id": created.id})
        assert hard_found is not None, "hard_find_one should find deleted record"
        assert hard_found.is_deleted(), "Record should be marked as deleted"
        assert hard_found.deleted_by == "test_admin", "Should record who deleted it"
        assert hard_found.deleted_at is not None, "Should have deletion timestamp"
        assert hard_found.deleted_id != 0, "deleted_id should be set"
        logger.info("✅ Verified: hard_find_one can find deleted record")
        logger.info("   - deleted_by: %s", hard_found.deleted_by)
        logger.info("   - deleted_at: %s", hard_found.deleted_at)
        logger.info("   - deleted_id: %s", hard_found.deleted_id)
        
        # 恢复记录
        restored = await repo.restore_by_event_id(event_id)
        assert restored is True, "Restore should succeed"
        logger.info("✅ Restore succeeded")
        
        # 验证恢复后可以查到
        restored_memcell = await repo.get_by_event_id(event_id)
        assert restored_memcell is not None, "Should find record after restore"
        assert not restored_memcell.is_deleted(), "Should not be marked as deleted after restore"
        assert restored_memcell.deleted_at is None, "deleted_at should be cleared"
        assert restored_memcell.deleted_by is None, "deleted_by should be cleared"
        assert restored_memcell.deleted_id == 0, "deleted_id should be reset to 0"
        logger.info("✅ Verified: Record is normal after restore")
        
        # 清理
        await repo.hard_delete_by_user_id(user_id)
        logger.info("✅ Test completed, cleaned up test data")
        
    except Exception as e:
        logger.error("❌ Soft delete single test failed: %s", e)
        import traceback
        logger.error(traceback.format_exc())
        raise
    
    logger.info("✅ Soft delete single test completed")


async def test_soft_delete_batch():
    """测试批量软删除功能"""
    logger.info("Starting test of soft delete batch...")
    
    repo = get_bean_by_type(MemCellRawRepository)
    user_id = "test_user_soft_delete_002"
    
    try:
        # 清理旧数据
        await repo.hard_delete_by_user_id(user_id)
        logger.info("✅ Cleaned up existing test data")
        
        # 创建5条测试数据
        now = get_now_with_timezone()
        for i in range(5):
            memcell = MemCell(
                user_id=user_id,
                timestamp=now - timedelta(hours=i),
                summary=f"测试记录 {i+1}",
                type=DataTypeEnum.CONVERSATION,
            )
            await repo.append_memcell(memcell)
        logger.info("✅ Created 5 test records")
        
        # 验证可以查到5条
        results = await repo.find_by_user_id(user_id)
        assert len(results) == 5, f"Should have 5 records, got {len(results)}"
        logger.info("✅ Verified: Can query 5 records")
        
        # 批量软删除
        deleted_count = await repo.delete_by_user_id(user_id, deleted_by="batch_admin")
        assert deleted_count == 5, f"Should soft delete 5 records, got {deleted_count}"
        logger.info("✅ Batch soft delete succeeded, deleted %d records", deleted_count)
        
        # 验证常规查询找不到
        results_after_delete = await repo.find_by_user_id(user_id)
        assert len(results_after_delete) == 0, "Regular query should not find soft-deleted records"
        logger.info("✅ Verified: Regular query cannot find soft-deleted records")
        
        # 使用 hard_find_many 可以找到
        hard_results = await MemCell.hard_find_many({"user_id": user_id}).to_list()
        assert len(hard_results) == 5, f"hard_find_many should find 5 records, got {len(hard_results)}"
        logger.info("✅ Verified: hard_find_many can find 5 deleted records")
        
        # 验证所有记录都被标记为已删除
        for mc in hard_results:
            assert mc.is_deleted(), "All records should be marked as deleted"
            assert mc.deleted_at is not None, "Should have deletion timestamp"
        logger.info("✅ Verified: All records are correctly marked as deleted")
        
        # 批量恢复
        restored_count = await repo.restore_by_user_id(user_id)
        assert restored_count == 5, f"Should restore 5 records, got {restored_count}"
        logger.info("✅ Batch restore succeeded, restored %d records", restored_count)
        
        # 验证恢复后可以查到
        results_after_restore = await repo.find_by_user_id(user_id)
        assert len(results_after_restore) == 5, f"Should have 5 records after restore, got {len(results_after_restore)}"
        logger.info("✅ Verified: Can query 5 records after restore")
        
        # 清理
        await repo.hard_delete_by_user_id(user_id)
        logger.info("✅ Test completed, cleaned up test data")
        
    except Exception as e:
        logger.error("❌ Soft delete batch test failed: %s", e)
        import traceback
        logger.error(traceback.format_exc())
        raise
    
    logger.info("✅ Soft delete batch test completed")


async def test_hard_delete():
    """测试硬删除功能"""
    logger.info("Starting test of hard delete...")
    
    repo = get_bean_by_type(MemCellRawRepository)
    user_id = "test_user_hard_delete_001"
    
    try:
        # 清理旧数据
        await repo.hard_delete_by_user_id(user_id)
        logger.info("✅ Cleaned up existing test data")
        
        # 创建测试数据
        now = get_now_with_timezone()
        memcell = MemCell(
            user_id=user_id,
            timestamp=now,
            summary="测试硬删除的记录",
            type=DataTypeEnum.CONVERSATION,
        )
        created = await repo.append_memcell(memcell)
        event_id = str(created.event_id)
        logger.info("✅ Created test record, event_id=%s", event_id)
        
        # 执行硬删除
        deleted = await repo.hard_delete_by_event_id(event_id)
        assert deleted is True, "Hard delete should succeed"
        logger.info("✅ Hard delete succeeded")
        
        # 验证 hard_find_one 也找不到（记录被物理删除）
        hard_found = await MemCell.hard_find_one({"_id": created.id})
        assert hard_found is None, "hard_find_one should not find hard-deleted record"
        logger.info("✅ Verified: Record is completely removed after hard delete")
        
        logger.info("✅ Test completed")
        
    except Exception as e:
        logger.error("❌ Hard delete test failed: %s", e)
        import traceback
        logger.error(traceback.format_exc())
        raise
    
    logger.info("✅ Hard delete test completed")


async def test_query_with_soft_delete_filtering():
    """测试查询自动过滤软删除记录"""
    logger.info("Starting test of query filtering with soft delete...")
    
    repo = get_bean_by_type(MemCellRawRepository)
    user_id = "test_user_query_filter_001"
    
    try:
        # 清理旧数据
        await repo.hard_delete_by_user_id(user_id)
        logger.info("✅ Cleaned up existing test data")
        
        # 创建10条记录
        now = get_now_with_timezone()
        event_ids = []
        for i in range(10):
            memcell = MemCell(
                user_id=user_id,
                timestamp=now - timedelta(hours=i),
                summary=f"测试记录 {i+1}",
                type=DataTypeEnum.CONVERSATION,
            )
            created = await repo.append_memcell(memcell)
            event_ids.append(str(created.event_id))
        logger.info("✅ Created 10 test records")
        
        # 软删除前5条
        for i in range(5):
            await repo.delete_by_event_id(event_ids[i], deleted_by="filter_test")
        logger.info("✅ Soft deleted first 5 records")
        
        # 测试 find_by_user_id（应该只返回5条未删除的）
        results = await repo.find_by_user_id(user_id)
        assert len(results) == 5, f"find_by_user_id should return 5, got {len(results)}"
        logger.info("✅ Verified: find_by_user_id returns only 5 non-deleted records")
        
        # 测试 count_by_user_id（应该只计数未删除的）
        count = await repo.count_by_user_id(user_id)
        assert count == 5, f"count_by_user_id should return 5, got {count}"
        logger.info("✅ Verified: count_by_user_id counts only non-deleted records")
        
        # 使用 hard_find_many 应该能找到所有10条
        all_results = await MemCell.hard_find_many({"user_id": user_id}).to_list()
        assert len(all_results) == 10, f"hard_find_many should return 10, got {len(all_results)}"
        logger.info("✅ Verified: hard_find_many returns all 10 records including deleted")
        
        # 统计已删除和未删除的数量
        deleted_count = sum(1 for mc in all_results if mc.is_deleted())
        active_count = sum(1 for mc in all_results if not mc.is_deleted())
        assert deleted_count == 5, f"Should have 5 deleted, got {deleted_count}"
        assert active_count == 5, f"Should have 5 active, got {active_count}"
        logger.info("✅ Verified: %d deleted, %d active", deleted_count, active_count)
        
        # 清理
        await repo.hard_delete_by_user_id(user_id)
        logger.info("✅ Test completed, cleaned up test data")
        
    except Exception as e:
        logger.error("❌ Query filtering test failed: %s", e)
        import traceback
        logger.error(traceback.format_exc())
        raise
    
    logger.info("✅ Query filtering test completed")


async def test_prevent_duplicate_soft_delete():
    """测试防止重复软删除（保护审计记录）"""
    logger.info("Starting test of preventing duplicate soft delete...")
    
    repo = get_bean_by_type(MemCellRawRepository)
    user_id = "test_user_duplicate_delete_001"
    
    try:
        # 清理旧数据
        await repo.hard_delete_by_user_id(user_id)
        logger.info("✅ Cleaned up existing test data")
        
        # 创建5条测试记录
        now = get_now_with_timezone()
        event_ids = []
        for i in range(5):
            memcell = MemCell(
                user_id=user_id,
                timestamp=now - timedelta(hours=i),
                summary=f"测试记录 {i+1}",
                type=DataTypeEnum.CONVERSATION,
            )
            created = await repo.append_memcell(memcell)
            event_ids.append(str(created.event_id))
        logger.info("✅ Created 5 test records")
        
        # 第一次软删除前3条记录
        first_delete_time = get_now_with_timezone()
        for i in range(3):
            await repo.delete_by_event_id(event_ids[i], deleted_by="admin_1")
        logger.info("✅ First soft delete of 3 records by admin_1")
        
        # 获取已删除记录的审计信息
        deleted_records = []
        for i in range(3):
            mc = await MemCell.hard_find_one({"_id": ObjectId(event_ids[i])})
            assert mc is not None
            assert mc.is_deleted()
            deleted_records.append({
                "event_id": event_ids[i],
                "deleted_at": mc.deleted_at,
                "deleted_by": mc.deleted_by,
                "deleted_id": mc.deleted_id,
            })
        logger.info("✅ Captured audit info from first delete")
        
        # 等待一小段时间，确保时间戳会不同
        from asyncio import sleep
        await sleep(0.01)
        
        # 尝试再次软删除同样的记录（应该被忽略）
        result = await repo.delete_by_user_id(user_id, deleted_by="admin_2")
        # 注意：delete_by_user_id 会尝试删除所有记录，但只有未删除的会被修改
        # 前3条已删除，后2条未删除，所以应该只修改2条
        assert result == 2, f"Should only soft delete 2 un-deleted records, got {result}"
        logger.info("✅ Second delete only affected 2 un-deleted records")
        
        # 验证前3条记录的审计信息没有被修改
        for i, original in enumerate(deleted_records):
            mc = await MemCell.hard_find_one({"_id": ObjectId(original["event_id"])})
            assert mc is not None
            assert mc.is_deleted()
            assert mc.deleted_at == original["deleted_at"], \
                f"deleted_at should not change for record {i}"
            assert mc.deleted_by == original["deleted_by"], \
                f"deleted_by should not change for record {i}, expected {original['deleted_by']}, got {mc.deleted_by}"
            assert mc.deleted_id == original["deleted_id"], \
                f"deleted_id should not change for record {i}"
        logger.info("✅ Verified: First 3 records' audit info was NOT modified")
        
        # 验证后2条记录被新的删除操作标记
        for i in range(3, 5):
            mc = await MemCell.hard_find_one({"_id": ObjectId(event_ids[i])})
            assert mc is not None
            assert mc.is_deleted()
            # 这两条应该被新删除操作标记
            assert mc.deleted_at > first_delete_time, "Should have newer deletion time"
            # 注意：由于是批量删除，deleted_by 可能是 admin_2
        logger.info("✅ Verified: Last 2 records were soft deleted by second operation")
        
        # 测试单个文档的重复删除保护
        test_record = await MemCell.hard_find_one({"_id": ObjectId(event_ids[0])})
        original_deleted_at = test_record.deleted_at
        original_deleted_by = test_record.deleted_by
        original_deleted_id = test_record.deleted_id
        
        # 再次尝试删除（应该被忽略）
        await test_record.delete(deleted_by="admin_3")
        
        # 重新获取记录
        test_record_after = await MemCell.hard_find_one({"_id": ObjectId(event_ids[0])})
        assert test_record_after.deleted_at == original_deleted_at, \
            "deleted_at should not change on duplicate delete"
        assert test_record_after.deleted_by == original_deleted_by, \
            "deleted_by should not change on duplicate delete"
        assert test_record_after.deleted_id == original_deleted_id, \
            "deleted_id should not change on duplicate delete"
        logger.info("✅ Verified: Instance method delete() also prevents duplicate deletion")
        
        # 清理
        await repo.hard_delete_by_user_id(user_id)
        logger.info("✅ Test completed, cleaned up test data")
        
    except Exception as e:
        logger.error("❌ Prevent duplicate soft delete test failed: %s", e)
        import traceback
        logger.error(traceback.format_exc())
        raise
    
    logger.info("✅ Prevent duplicate soft delete test completed")


async def run_all_tests():
    """Run all tests"""
    logger.info("🚀 Starting to run all MemCellRawRepository tests...")

    try:
        await test_basic_crud_operations()
        await test_find_by_user_id()
        await test_find_by_time_range()
        await test_find_by_user_and_time_range()
        await test_find_by_group_id()
        await test_find_by_participants()
        await test_search_by_keywords()
        await test_batch_delete_operations()
        await test_statistics_and_aggregation()
        await test_get_by_event_ids()
        
        # 软删除功能测试
        logger.info("")
        logger.info("=" * 60)
        logger.info("Starting Soft Delete Feature Tests...")
        logger.info("=" * 60)
        await test_soft_delete_single()
        await test_soft_delete_batch()
        await test_hard_delete()
        await test_query_with_soft_delete_filtering()
        await test_prevent_duplicate_soft_delete()
        
        logger.info("")
        logger.info("=" * 60)
        logger.info("✅✅✅ All tests completed!")
        logger.info("=" * 60)
        logger.info("")
        logger.info("软删除功能验证总结：")
        logger.info("1. ✅ 单个软删除正常工作")
        logger.info("2. ✅ 批量软删除正常工作")
        logger.info("3. ✅ 恢复功能正常工作")
        logger.info("4. ✅ 查询自动过滤已删除记录")
        logger.info("5. ✅ hard_find 可以查询已删除记录")
        logger.info("6. ✅ 硬删除（物理删除）正常工作")
        logger.info("7. ✅ deleted_by、deleted_at、deleted_id 字段正确设置")
        logger.info("8. ✅ 防止重复软删除，保护审计记录")
    except Exception as e:
        logger.error("❌ Error occurred during testing: %s", e)
        raise


if __name__ == "__main__":
    asyncio.run(run_all_tests())
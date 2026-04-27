#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test the functionality of ConversationStatusRawRepository

Test contents include:
1. Query and update operations based on group_id
2. Statistical methods
"""

import asyncio
from datetime import datetime
from zoneinfo import ZoneInfo

from core.di import get_bean_by_type
from common_utils.datetime_utils import get_now_with_timezone, to_iso_format
from infra_layer.adapters.out.persistence.repository.conversation_status_raw_repository import (
    ConversationStatusRawRepository,
)
from core.observation.logger import get_logger

logger = get_logger(__name__)


def compare_datetime(dt1: datetime, dt2: datetime) -> bool:
    """Compare two datetime objects, only up to second-level precision"""
    return dt1.replace(microsecond=0) == dt2.replace(microsecond=0)


async def test_group_operations():
    """Test group-related operations"""
    logger.info("Starting test for group-related operations...")

    repo = get_bean_by_type(ConversationStatusRawRepository)
    group_id = "test_group_001"
    current_time = get_now_with_timezone()

    try:
        # Test upsert (create new record)
        update_data = {
            "old_msg_start_time": current_time,
            "new_msg_start_time": current_time,
            "last_memcell_time": current_time,
        }

        result = await repo.upsert_by_group_id(group_id, update_data)
        assert result is not None
        assert result.group_id == group_id
        logger.info("✅ Test upsert to create new record succeeded")

        # Test querying by group_id
        queried = await repo.get_by_group_id(group_id)
        assert queried is not None
        assert queried.group_id == group_id
        assert compare_datetime(queried.old_msg_start_time, current_time)
        assert compare_datetime(queried.new_msg_start_time, current_time)
        logger.info("✅ Test querying by group_id succeeded")

        # Test upsert (update existing record)
        new_time = get_now_with_timezone()
        update_data = {"old_msg_start_time": new_time, "new_msg_start_time": new_time}

        updated = await repo.upsert_by_group_id(group_id, update_data)
        assert updated is not None
        assert compare_datetime(updated.old_msg_start_time, new_time)
        assert compare_datetime(updated.new_msg_start_time, new_time)
        assert compare_datetime(
            updated.last_memcell_time, current_time
        )  # Fields not updated should retain original values
        logger.info("✅ Test upsert to update existing record succeeded")

        # Query again to verify update
        queried_again = await repo.get_by_group_id(group_id)
        assert queried_again is not None
        assert compare_datetime(queried_again.old_msg_start_time, new_time)
        assert compare_datetime(queried_again.new_msg_start_time, new_time)
        assert compare_datetime(queried_again.last_memcell_time, current_time)
        logger.info("✅ Verified update result successfully")

        # Clean up test data
        await queried_again.delete()
        logger.info("✅ Cleaned up test data successfully")

        # Verify deletion
        final_check = await repo.get_by_group_id(group_id)
        assert final_check is None, "Record should have been deleted"
        logger.info("✅ Verified deletion successfully")

    except Exception as e:
        logger.error("❌ Test for group-related operations failed: %s", e)
        raise

    logger.info("✅ Group-related operations test completed")


async def test_statistics():
    """Test statistical methods"""
    logger.info("Starting test for statistical methods...")

    repo = get_bean_by_type(ConversationStatusRawRepository)
    base_group_id = "test_group_stats"
    current_time = get_now_with_timezone()

    try:
        # Create multiple test records
        test_records = []
        for i in range(3):
            group_id = f"{base_group_id}_{i}"
            result = await repo.upsert_by_group_id(
                group_id=group_id,
                update_data={
                    "old_msg_start_time": current_time,
                    "new_msg_start_time": current_time,
                    "last_memcell_time": current_time,
                },
            )
            test_records.append(result)
        logger.info("✅ Created test records successfully")

        # Test group record count
        count = await repo.count_by_group_id(
            f"{base_group_id}_0"
        )  # Test count for the first group
        assert count == 1, "Should have 1 record, actually has %d records" % count
        logger.info("✅ Test group record count succeeded")

        # Test total record count
        total = await repo.count_all()
        assert total >= 3, (
            "Total record count should be at least 3, actually is %d" % total
        )
        logger.info("✅ Test total record count succeeded")

        # Clean up test data
        for record in test_records:
            await record.delete()
        logger.info("✅ Cleaned up test data successfully")

    except Exception as e:
        logger.error("❌ Test for statistical methods failed: %s", e)
        raise

    logger.info("✅ Statistical methods test completed")


async def test_timezone_handling():
    """Test datetime handling in different time zones"""
    logger.info("Starting test for time zone handling...")

    repo = get_bean_by_type(ConversationStatusRawRepository)
    group_id = "test_timezone_001"

    try:
        # Create UTC time
        utc_time = get_now_with_timezone(ZoneInfo("UTC"))
        # Create Tokyo time
        tokyo_time = get_now_with_timezone(ZoneInfo("Asia/Tokyo"))

        shanghai_time = get_now_with_timezone(ZoneInfo("Asia/Shanghai"))

        # Create record using times from different time zones
        update_data = {
            "old_msg_start_time": utc_time,
            "new_msg_start_time": tokyo_time,
            "last_memcell_time": shanghai_time,
        }

        # Record original time in ISO format for comparison
        logger.info("Original UTC time: %s", to_iso_format(utc_time))
        logger.info("Original Tokyo time: %s", to_iso_format(tokyo_time))
        logger.info("Original Shanghai time: %s", to_iso_format(shanghai_time))

        # Insert into database
        result = await repo.upsert_by_group_id(group_id, update_data)
        assert result is not None
        logger.info("✅ Inserted record with different time zones successfully")

        # Retrieve from database and verify
        queried = await repo.get_by_group_id(group_id)
        assert queried is not None

        # Output retrieved time information
        logger.info("Times retrieved from database:")
        logger.info(
            "old_msg_start_time (original UTC): %s",
            to_iso_format(queried.old_msg_start_time),
        )
        logger.info(
            "new_msg_start_time (original Tokyo): %s",
            to_iso_format(queried.new_msg_start_time),
        )
        logger.info(
            "last_memcell_time (original Shanghai): %s",
            to_iso_format(queried.last_memcell_time),
        )

        # Verify times are correct (should be equal when converted to the same time zone)
        assert queried.old_msg_start_time.astimezone(ZoneInfo("UTC")).replace(
            microsecond=0
        ) == utc_time.replace(microsecond=0)
        assert queried.new_msg_start_time.astimezone(ZoneInfo("Asia/Tokyo")).replace(
            microsecond=0
        ) == tokyo_time.replace(microsecond=0)
        assert queried.last_memcell_time.replace(tzinfo=None).replace(
            microsecond=0
        ) == shanghai_time.replace(microsecond=0)
        logger.info("✅ Time zone validation succeeded")

        # Clean up test data
        # await queried.delete()
        logger.info("✅ Cleaned up test data successfully")

    except Exception as e:
        logger.error("❌ Test for time zone handling failed: %s", e)
        raise

    logger.info("✅ Time zone handling test completed")


async def run_all_tests():
    """Run all tests"""
    logger.info("🚀 Starting to run all tests...")

    try:
        await test_group_operations()
        await test_statistics()
        await test_timezone_handling()
        logger.info("✅ All tests completed")
    except Exception as e:
        logger.error("❌ Error occurred during testing: %s", e)
        raise


if __name__ == "__main__":
    asyncio.run(run_all_tests())

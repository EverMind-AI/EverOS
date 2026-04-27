"""
Redis length-limited cache manager test script

Usage:
    python src/bootstrap.py tests/test_redis_length_cache.py

Test coverage:
1. Basic operations: append, get, clear, etc.
2. Length limit cleanup mechanism
3. Timestamp compatibility (integer and datetime objects)
4. Get data by timestamp range functionality
5. Stress testing (large-scale data processing)
6. Expiration mechanism test
7. Backward compatibility layer test
"""

import asyncio
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from common_utils.datetime_utils import get_now_with_timezone
from core.di.utils import get_bean
from core.observation.logger import get_logger

logger = get_logger(__name__)


class NonSerializableTestClass:
    """Test class that cannot be JSON-serialized, used for testing Pickle fallback"""

    def __init__(self, name, value, multiplier=2):
        self.name = name
        self.value = value
        self.multiplier = multiplier
        self.created_at = time.time()
        # Add some complex attributes to make JSON serialization fail
        self.complex_data = {
            "set_data": {1, 2, 3, 4, 5},  # set cannot be JSON-serialized
            "tuple_data": (1, 2, 3),  # tuple will become list, but we can detect
            "bytes_data": b"hello world",  # bytes cannot be JSON-serialized
        }

    def get_doubled_value(self):
        return self.value * self.multiplier

    def process_data(self, input_value):
        """Method to process data"""
        return f"{self.name}_processed_{input_value}_{self.multiplier}"

    def __eq__(self, other):
        return (
            isinstance(other, NonSerializableTestClass)
            and self.name == other.name
            and self.value == other.value
            and self.multiplier == other.multiplier
        )

    def __repr__(self):
        return f"NonSerializableTestClass(name='{self.name}', value={self.value}, multiplier={self.multiplier})"


async def test_basic_operations():
    """Test basic operations: append, get, clear, etc."""
    logger.info("Starting basic operations test...")

    # Get cache manager factory from DI container
    factory = get_bean("redis_length_cache_factory")
    cache = await factory.create_cache_manager(
        max_length=5, expire_minutes=5
    )  # Max 5 items, 5 minutes expiration

    test_key = "test_length_cache"

    # 1. Clear test queue
    logger.info("Clearing test queue...")
    await cache.clear_queue(test_key)

    # 2. Test appending data
    logger.info("Testing data append...")
    test_data = [
        "test_string",
        {"name": "test_dict", "value": 123},
        ["test_list", 1, 2, 3],
        42,
        {"complex": {"nested": "data"}},
    ]

    for i, data in enumerate(test_data):
        success = await cache.append(test_key, data)
        assert success, f"Failed to append data: {data}"
        logger.info("Successfully appended data %d", i + 1)

    # 3. Verify queue size
    size = await cache.get_queue_size(test_key)
    assert size == len(
        test_data
    ), f"Queue size mismatch: expected {len(test_data)}, actual {size}"
    logger.info("Queue size verification passed: %d", size)

    # 4. Get queue statistics
    stats = await cache.get_queue_stats(test_key)
    assert isinstance(stats, dict), "Statistics format error"
    assert stats["total_count"] == len(test_data), "Statistics count mismatch"
    assert stats["max_length"] == 5, "Max length configuration mismatch"
    assert stats["is_full"], "Queue should be full (5/5)"
    logger.info("Queue statistics: %s", stats)

    logger.info("✅ Basic operations test passed")


async def test_length_cleanup():
    """Test length limit cleanup mechanism"""
    logger.info("Starting length limit cleanup test...")

    factory = get_bean("redis_length_cache_factory")
    # Set max length to 3, cleanup probability to 0 to prevent auto cleanup, making manual verification easier
    cache = await factory.create_cache_manager(
        max_length=3, expire_minutes=10, cleanup_probability=0.0
    )

    test_key = "test_length_cleanup"

    # 1. Clear test queue
    await cache.clear_queue(test_key)

    # 2. Add data with explicit timestamps to ensure order
    logger.info("Adding data with increasing timestamps...")
    base_timestamp = int(time.time() * 1000)
    data_list = []

    for i in range(5):  # Add 5 items, exceeding max length of 3
        timestamp = base_timestamp + i * 1000  # Each data item spaced by 1 second
        data = {"index": i, "content": f"data_{i}", "timestamp": timestamp}
        data_list.append({"data": data, "timestamp": timestamp})

        success = await cache.append(test_key, data, timestamp=timestamp)
        assert success, f"Failed to append data {i+1}"
        logger.info("Added data %d: timestamp=%d, content=data_%d", i, timestamp, i)

    # 3. Verify all data has been added
    current_size = await cache.get_queue_size(test_key)
    logger.info("Queue size after adding: %d", current_size)
    assert current_size == 5, f"Expected queue size 5, actual {current_size}"

    # 4. Get all data from queue, verify order
    logger.info("Retrieving data from queue, verifying timestamp order...")
    # Note: Since there's no get_all method, we verify time range through statistics
    stats = await cache.get_queue_stats(test_key)
    logger.info(
        "Queue statistics: oldest_timestamp=%s, newest_timestamp=%s",
        stats.get("oldest_timestamp"),
        stats.get("newest_timestamp"),
    )

    # Verify oldest and newest timestamps
    expected_oldest = base_timestamp
    expected_newest = base_timestamp + 4 * 1000
    assert (
        stats["oldest_timestamp"] == expected_oldest
    ), f"Oldest timestamp mismatch: expected {expected_oldest}, actual {stats['oldest_timestamp']}"
    assert (
        stats["newest_timestamp"] == expected_newest
    ), f"Newest timestamp mismatch: expected {expected_newest}, actual {stats['newest_timestamp']}"

    # 5. Manually trigger cleanup, should delete the 2 oldest items (keep 3)
    logger.info("Manually triggering cleanup, should delete 2 oldest items...")
    cleaned_count = await cache.cleanup_excess(test_key)
    logger.info("Manual cleanup completed, cleaned count: %d", cleaned_count)
    assert (
        cleaned_count == 2
    ), f"Expected to clean 2 items, actually cleaned {cleaned_count}"

    # 6. Verify data after cleanup
    final_size = await cache.get_queue_size(test_key)
    assert final_size == 3, f"Queue size after cleanup should be 3, actual {final_size}"

    # 7. Verify remaining items are the latest 3 (index 2, 3, 4)
    stats_after = await cache.get_queue_stats(test_key)
    expected_oldest_after = base_timestamp + 2 * 1000  # data_2's timestamp
    expected_newest_after = base_timestamp + 4 * 1000  # data_4's timestamp

    logger.info(
        "Queue statistics after cleanup: oldest_timestamp=%s, newest_timestamp=%s",
        stats_after.get("oldest_timestamp"),
        stats_after.get("newest_timestamp"),
    )

    assert (
        stats_after["oldest_timestamp"] == expected_oldest_after
    ), f"After cleanup, oldest timestamp should be {expected_oldest_after}, actual {stats_after['oldest_timestamp']}"
    assert (
        stats_after["newest_timestamp"] == expected_newest_after
    ), f"After cleanup, newest timestamp should be {expected_newest_after}, actual {stats_after['newest_timestamp']}"

    logger.info(
        "✅ Verification passed: cleanup deleted oldest data (data_0 and data_1), kept latest data (data_2, data_3, data_4)"
    )

    # 8. Clean again, should delete no data (already at max length)
    additional_cleaned = await cache.cleanup_excess(test_key)
    assert (
        additional_cleaned == 0
    ), f"Second cleanup should delete 0 items, actually deleted {additional_cleaned}"

    logger.info("✅ Length limit cleanup test passed")


async def test_timestamp_compatibility():
    """Test timestamp compatibility: support for integers and datetime objects"""
    logger.info("Starting timestamp compatibility test...")

    factory = get_bean("redis_length_cache_factory")
    cache = await factory.create_cache_manager(max_length=10, expire_minutes=5)

    test_key = "test_timestamp_compat"

    # 1. Clear test queue
    await cache.clear_queue(test_key)

    # 2. Test different types of timestamps
    logger.info("Testing different types of timestamps...")

    # Use current time (default)
    success = await cache.append(test_key, "data_current_time")
    assert success, "Failed to append with current time"

    # Use integer timestamp (milliseconds)
    timestamp_ms = int(time.time() * 1000) + 1000  # 1 second later
    success = await cache.append(test_key, "data_int_timestamp", timestamp=timestamp_ms)
    assert success, "Failed to append with integer timestamp"

    # Use datetime object (naive)
    dt_naive = get_now_with_timezone()
    success = await cache.append(test_key, "data_datetime_naive", timestamp=dt_naive)
    assert success, "Failed to append with datetime object"

    # Use datetime object (with timezone)
    dt_with_tz = get_now_with_timezone(ZoneInfo("UTC"))
    success = await cache.append(test_key, "data_datetime_tz", timestamp=dt_with_tz)
    assert success, "Failed to append with timezone-aware datetime object"

    # 3. Verify all data was correctly stored
    size = await cache.get_queue_size(test_key)
    assert (
        size == 4
    ), f"Timestamp compatibility test data count mismatch: expected 4, actual {size}"

    # 4. Get statistics to verify timestamps
    stats = await cache.get_queue_stats(test_key)
    assert stats["oldest_timestamp"] is not None, "Oldest timestamp is None"
    assert stats["newest_timestamp"] is not None, "Newest timestamp is None"
    assert stats["oldest_datetime"] is not None, "Oldest datetime string is None"
    assert stats["newest_datetime"] is not None, "Newest datetime string is None"

    logger.info(
        "Timestamp range: %s to %s", stats["oldest_datetime"], stats["newest_datetime"]
    )
    logger.info("✅ Timestamp compatibility test passed")


async def test_timestamp_range_query():
    """Test get data by timestamp range functionality"""
    logger.info("Starting test for getting data by timestamp range...")

    factory = get_bean("redis_length_cache_factory")
    cache = await factory.create_cache_manager(
        max_length=20, expire_minutes=10, cleanup_probability=0.0
    )

    test_key = "test_timestamp_range"

    # 1. Clear test queue
    await cache.clear_queue(test_key)

    # 2. Add test data with explicit timestamps
    logger.info("Adding test data with explicit timestamps...")
    base_timestamp = int(time.time() * 1000)
    test_data = []

    for i in range(10):
        timestamp = base_timestamp + i * 10000  # Each data item spaced by 10 seconds
        data = {"index": i, "content": f"data_{i}", "created_at": timestamp}
        test_data.append({"data": data, "timestamp": timestamp})

        success = await cache.append(test_key, data, timestamp=timestamp)
        assert success, f"Failed to add data {i+1}"

    logger.info(
        "Added %d data items, time range: %d to %d",
        len(test_data),
        base_timestamp,
        base_timestamp + 9 * 10000,
    )

    # 3. Test getting all data (no time range limit)
    all_data = await cache.get_by_timestamp_range(test_key)
    assert (
        len(all_data) == 10
    ), f"Failed to get all data, expected 10, actual {len(all_data)}"
    logger.info("Successfully retrieved all data: %d items", len(all_data))

    # 4. Test filtering by start time (get data after the 5th item)
    start_time = base_timestamp + 4 * 10000  # data_4's timestamp
    filtered_data = await cache.get_by_timestamp_range(
        test_key, start_timestamp=start_time
    )
    assert (
        len(filtered_data) == 6
    ), f"Start time filtering failed, expected 6, actual {len(filtered_data)}"  # data_4 to data_9

    # Verify data order (newest first)
    assert (
        filtered_data[0]["data"]["index"] == 9
    ), "Data sorting error, newest should be first"
    assert (
        filtered_data[-1]["data"]["index"] == 4
    ), "Data sorting error, oldest should be last"
    logger.info("Start time filtering test passed")

    # 5. Test filtering by end time (get data before the 5th item)
    end_time = base_timestamp + 4 * 10000  # data_4's timestamp
    filtered_data = await cache.get_by_timestamp_range(test_key, end_timestamp=end_time)
    assert (
        len(filtered_data) == 5
    ), f"End time filtering failed, expected 5, actual {len(filtered_data)}"  # data_0 to data_4

    # Verify data content
    assert filtered_data[0]["data"]["index"] == 4, "End time filtering result error"
    assert filtered_data[-1]["data"]["index"] == 0, "End time filtering result error"
    logger.info("End time filtering test passed")

    # 6. Test time range filtering (get middle data)
    start_time = base_timestamp + 2 * 10000  # data_2's timestamp
    end_time = base_timestamp + 6 * 10000  # data_6's timestamp
    range_data = await cache.get_by_timestamp_range(
        test_key, start_timestamp=start_time, end_timestamp=end_time
    )
    assert (
        len(range_data) == 5
    ), f"Time range filtering failed, expected 5, actual {len(range_data)}"  # data_2 to data_6

    # Verify data range
    indexes = [item["data"]["index"] for item in range_data]
    indexes.sort()  # Sort for verification
    assert indexes == [
        2,
        3,
        4,
        5,
        6,
    ], f"Time range filtering result error, expected [2,3,4,5,6], actual {indexes}"
    logger.info("Time range filtering test passed")

    # 7. Test limiting number of results
    limited_data = await cache.get_by_timestamp_range(test_key, limit=3)
    assert (
        len(limited_data) == 3
    ), f"Limiting count failed, expected 3, actual {len(limited_data)}"
    logger.info("Limit count test passed")

    # 8. Test using datetime objects as timestamps
    dt_start = datetime.fromtimestamp((base_timestamp + 3 * 10000) / 1000)
    dt_end = datetime.fromtimestamp((base_timestamp + 7 * 10000) / 1000)
    dt_data = await cache.get_by_timestamp_range(
        test_key, start_timestamp=dt_start, end_timestamp=dt_end
    )
    assert (
        len(dt_data) == 5
    ), f"Filtering with datetime objects failed, expected 5, actual {len(dt_data)}"  # data_3 to data_7
    logger.info("Filtering with datetime objects test passed")

    # 9. Test empty result
    future_time = base_timestamp + 20 * 10000  # Future time
    empty_data = await cache.get_by_timestamp_range(
        test_key, start_timestamp=future_time
    )
    assert (
        len(empty_data) == 0
    ), f"Empty result test failed, should return 0 items, actual {len(empty_data)}"
    logger.info("Empty result test passed")

    # 10. Clean up test data
    success = await cache.clear_queue(test_key)
    assert success, "Failed to clean up test data"

    logger.info("✅ Test for getting data by timestamp range passed")


async def test_json_pickle_mixed_data():
    """Test JSON and Pickle mixed data processing"""
    logger.info("Starting test for JSON and Pickle mixed data processing...")

    factory = get_bean("redis_length_cache_factory")
    cache = await factory.create_cache_manager(
        max_length=20, expire_minutes=10, cleanup_probability=0.0
    )

    test_key = "test_mixed_serialization"

    # 1. Clear test queue
    await cache.clear_queue(test_key)

    # 2. Prepare mixed test data
    base_timestamp = int(time.time() * 1000)
    test_data = []

    # JSON serializable data
    json_data = [
        {"type": "json", "name": "dict_data", "value": 123, "items": [1, 2, 3]},
        ["json_list", "with", "strings", 456],
        "simple_json_string",
        42,
        {"nested": {"deep": {"data": "json_nested"}}},
    ]

    # Pickle serialized data (cannot be JSON serialized)
    pickle_data = [
        NonSerializableTestClass("test1", 100),
        NonSerializableTestClass("test2", 200, 3),
        {
            "complex_set": {1, 2, 3, 4, 5},
            "bytes_data": b"binary_data",
            "name": "complex_dict",
        },  # Dictionary containing set and bytes
    ]

    # 3. Add JSON data
    logger.info("Adding JSON serializable data...")
    for i, data in enumerate(json_data):
        timestamp = base_timestamp + i * 1000
        success = await cache.append(test_key, data, timestamp=timestamp)
        assert success, f"Failed to add JSON data: {data}"
        test_data.append({"type": "json", "data": data, "timestamp": timestamp})
        logger.debug("Successfully added JSON data: %s", str(data)[:50])

    # 4. Add Pickle data
    logger.info("Adding Pickle serialized data...")
    for i, data in enumerate(pickle_data):
        timestamp = base_timestamp + (len(json_data) + i) * 1000
        success = await cache.append(test_key, data, timestamp=timestamp)
        assert success, f"Failed to add Pickle data: {data}"
        test_data.append({"type": "pickle", "data": data, "timestamp": timestamp})
        logger.debug("Successfully added Pickle data: %s", str(data)[:50])

    # 5. Verify total data count
    total_count = len(json_data) + len(pickle_data)
    size = await cache.get_queue_size(test_key)
    assert (
        size == total_count
    ), f"Total data count mismatch: expected {total_count}, actual {size}"
    logger.info(
        "Data addition completed, total: %d items (JSON: %d, Pickle: %d)",
        total_count,
        len(json_data),
        len(pickle_data),
    )

    # 6. Retrieve all data and verify
    all_data = await cache.get_by_timestamp_range(test_key)
    assert (
        len(all_data) == total_count
    ), f"Retrieved data count mismatch: expected {total_count}, actual {len(all_data)}"

    # 7. Verify correctness of each data type
    json_count = 0
    pickle_count = 0

    for item in all_data:
        retrieved_data = item["data"]

        # Check JSON data
        if any(
            str(retrieved_data) == str(original["data"])
            for original in test_data
            if original["type"] == "json"
        ):
            json_count += 1
            logger.debug(
                "JSON data verification successful: %s", str(retrieved_data)[:50]
            )

        # Check Pickle data
        elif isinstance(retrieved_data, NonSerializableTestClass):
            pickle_count += 1
            # Verify Pickle object functionality
            doubled = retrieved_data.get_doubled_value()
            expected = retrieved_data.value * retrieved_data.multiplier
            assert (
                doubled == expected
            ), f"Pickle object function error: {doubled} != {expected}"

            # Verify complex data
            assert (
                "set_data" in retrieved_data.complex_data
            ), "Pickle object missing set data"
            assert (
                "bytes_data" in retrieved_data.complex_data
            ), "Pickle object missing bytes data"

            logger.debug(
                "Pickle object verification successful: %s, function test: %d * %d = %d",
                retrieved_data,
                retrieved_data.value,
                retrieved_data.multiplier,
                doubled,
            )

        elif isinstance(retrieved_data, dict) and "complex_set" in retrieved_data:
            pickle_count += 1
            # Verify dictionary containing set and bytes
            assert isinstance(retrieved_data["complex_set"], set), "Set data type error"
            assert isinstance(
                retrieved_data["bytes_data"], bytes
            ), "Bytes data type error"
            logger.debug(
                "Verification of dictionary with complex data successful: %s",
                retrieved_data["name"],
            )

        else:
            logger.warning("Unrecognized data type: %s", type(retrieved_data))

    # 8. Verify data type distribution
    assert json_count == len(
        json_data
    ), f"JSON data count mismatch: expected {len(json_data)}, actual {json_count}"
    assert pickle_count == len(
        pickle_data
    ), f"Pickle data count mismatch: expected {len(pickle_data)}, actual {pickle_count}"

    logger.info(
        "Data type verification completed: JSON data %d items, Pickle data %d items",
        json_count,
        pickle_count,
    )

    # 9. Test timestamp range query support for mixed data
    # Get first half data (mainly JSON data)
    mid_timestamp = base_timestamp + (total_count // 2) * 1000
    first_half = await cache.get_by_timestamp_range(
        test_key, end_timestamp=mid_timestamp
    )
    assert (
        len(first_half) >= len(json_data) // 2
    ), "Timestamp range query support for mixed data abnormal"

    # Get second half data (mainly Pickle data)
    second_half = await cache.get_by_timestamp_range(
        test_key, start_timestamp=mid_timestamp
    )
    assert len(second_half) >= len(
        pickle_data
    ), "Timestamp range query support for Pickle data abnormal"

    logger.info(
        "Mixed data timestamp range query test passed: first half %d items, second half %d items",
        len(first_half),
        len(second_half),
    )

    # 10. Clean up test data
    success = await cache.clear_queue(test_key)
    assert success, "Failed to clean up test data"

    logger.info("✅ JSON and Pickle mixed data processing test passed")


async def test_pickle_error_handling():
    """Test Pickle serialization error handling"""
    logger.info("Starting test for Pickle serialization error handling...")

    factory = get_bean("redis_length_cache_factory")
    cache = await factory.create_cache_manager(max_length=10, expire_minutes=5)

    test_key = "test_pickle_error_handling"

    # 1. Clear test queue
    await cache.clear_queue(test_key)

    # 2. Test normal Pickle data
    normal_pickle_data = NonSerializableTestClass("normal", 42)
    success = await cache.append(test_key, normal_pickle_data)
    assert success, "Failed to add normal Pickle data"

    # 3. Retrieve and verify normal data
    data_list = await cache.get_by_timestamp_range(test_key)
    assert len(data_list) == 1, "Failed to retrieve normal Pickle data"

    retrieved = data_list[0]["data"]
    assert isinstance(retrieved, NonSerializableTestClass), "Pickle data type error"
    assert (
        retrieved.name == "normal" and retrieved.value == 42
    ), "Pickle data content error"
    assert retrieved.get_doubled_value() == 84, "Pickle object function error"

    logger.info("Normal Pickle data processing verification passed")

    # 4. Test error recovery with mixed data
    mixed_data = [
        {"json_data": "this_is_json"},  # JSON data
        NonSerializableTestClass("pickle1", 100),  # Pickle data
        "simple_string",  # Simple string
        NonSerializableTestClass("pickle2", 200),  # Another Pickle data
    ]

    for i, data in enumerate(mixed_data):
        success = await cache.append(test_key, data)
        assert success, f"Failed to add mixed data item {i+1}"

    # 5. Verify all mixed data can be processed correctly
    all_data = await cache.get_by_timestamp_range(test_key)
    assert (
        len(all_data) == 5
    ), f"Mixed data total count error: expected 5, actual {len(all_data)}"  # 1 existing + 4 new

    # Count various data types
    json_count = sum(
        1
        for item in all_data
        if isinstance(item["data"], (dict, str))
        and not isinstance(item["data"], NonSerializableTestClass)
    )
    pickle_count = sum(
        1 for item in all_data if isinstance(item["data"], NonSerializableTestClass)
    )

    assert (
        json_count >= 2
    ), f"JSON data count abnormal: {json_count}"  # At least dict and string
    assert (
        pickle_count >= 3
    ), f"Pickle data count abnormal: {pickle_count}"  # 3 NonSerializableTestClass objects

    logger.info(
        "Mixed data error handling test passed: JSON type %d items, Pickle type %d items",
        json_count,
        pickle_count,
    )

    # 6. Clean up test data
    success = await cache.clear_queue(test_key)
    assert success, "Failed to clean up test data"

    logger.info("✅ Pickle serialization error handling test passed")


async def test_stress_operations():
    """Test stress operations: large-scale data processing and cleanup"""
    logger.info("Starting stress test...")

    factory = get_bean("redis_length_cache_factory")
    cache = await factory.create_cache_manager(
        max_length=100, expire_minutes=10, cleanup_probability=0.1
    )

    test_key = "test_length_cache_stress"

    # 1. Clear test queue
    await cache.clear_queue(test_key)

    # 2. Batch append data (exceeding max length)
    batch_size = 500
    logger.info("Appending %d items (exceeding max length 100)...", batch_size)

    start_time = time.time()
    for i in range(batch_size):
        data = {
            "index": i,
            "timestamp": time.time(),
            "data": f"stress_test_data_{i}",
            "batch": "stress_test",
        }
        # Use increasing timestamps to ensure order
        timestamp = int(time.time() * 1000) + i * 10
        success = await cache.append(test_key, data, timestamp=timestamp)
        assert success, f"Failed to batch append item {i+1}"

        # Check size every 100 items
        if (i + 1) % 100 == 0:
            current_size = await cache.get_queue_size(test_key)
            logger.info(
                "Appended %d items, current queue size: %d", i + 1, current_size
            )

    elapsed = time.time() - start_time
    logger.info("Data append completed, elapsed time: %.2f seconds", elapsed)

    # 3. Verify queue size (allow some excess due to probabilistic cleanup)
    final_size = await cache.get_queue_size(test_key)
    logger.info("Queue size after stress test: %d", final_size)

    # If queue size exceeds significantly, manually trigger cleanup to verify cleanup mechanism
    if final_size > 120:  # Allow some excess range
        logger.info(
            "Queue size exceeds significantly, manually triggering cleanup test..."
        )
        cleaned_count = await cache.cleanup_excess(test_key)
        logger.info("Manual cleanup completed, cleaned %d items", cleaned_count)

        # Verify size after cleanup
        size_after_cleanup = await cache.get_queue_size(test_key)
        assert (
            size_after_cleanup <= 100
        ), f"Queue size still exceeds limit after manual cleanup: {size_after_cleanup}"
        logger.info("Queue size after manual cleanup: %d", size_after_cleanup)

    # 4. Get final queue size (may have been manually cleaned)
    current_size = await cache.get_queue_size(test_key)

    # Verify queue statistics
    stats = await cache.get_queue_stats(test_key)
    assert stats["total_count"] == current_size, "Statistics do not match actual size"
    assert stats["is_full"] == (current_size >= 100), "Queue full status judgment error"

    # 5. Manually clean again to verify mechanism stability
    logger.info("Manually cleaning queue again to verify mechanism stability...")
    additional_cleaned = await cache.cleanup_excess(test_key)
    logger.info("Additional cleanup completed, cleaned count: %d", additional_cleaned)

    # 6. Clean up test data
    success = await cache.clear_queue(test_key)
    assert success, "Failed to clean up test data"

    logger.info("✅ Stress test passed")


async def test_expiry_mechanism():
    """Test expiration mechanism"""
    logger.info("Starting expiration mechanism test...")

    factory = get_bean("redis_length_cache_factory")
    cache = await factory.create_cache_manager(
        max_length=10, expire_minutes=1
    )  # 1 minute expiration

    test_key = "test_length_cache_expiry"

    # 1. Clear test queue
    await cache.clear_queue(test_key)

    # 2. Add test data
    test_data = {"test": "expiry", "timestamp": time.time()}
    success = await cache.append(test_key, test_data)
    assert success, "Failed to add test data"

    # 3. Verify data exists
    size = await cache.get_queue_size(test_key)
    assert size == 1, "Data not correctly added"

    stats = await cache.get_queue_stats(test_key)
    assert stats["ttl_seconds"] > 0, "TTL should be greater than 0"
    logger.info("Data TTL: %d seconds", stats["ttl_seconds"])

    # 4. Wait for expiration
    logger.info("Waiting for data to expire (70 seconds)...")
    await asyncio.sleep(70)  # Wait more than 1 minute

    # 5. Verify data has expired
    expired_size = await cache.get_queue_size(test_key)
    assert (
        expired_size == 0
    ), f"Data did not expire correctly, queue size: {expired_size}"

    expired_stats = await cache.get_queue_stats(test_key)
    assert expired_stats["total_count"] == 0, "Statistics should be 0 after expiration"

    logger.info("✅ Expiration mechanism test passed")


async def test_compatibility_layer():
    """Test backward compatibility layer"""
    logger.info("Starting backward compatibility layer test...")

    # Get default manager instance
    default_manager = get_bean("redis_length_cache_manager")
    test_key = "test_length_cache_compat"

    # 1. Clear test queue
    await default_manager.clear_queue(test_key)

    # 2. Test basic operations
    test_data = {"test": "compatibility", "type": "default_manager"}
    success = await default_manager.append(test_key, test_data)
    assert success, "Failed to append data in backward compatibility layer"

    size = await default_manager.get_queue_size(test_key)
    assert size == 1, "Queue size error in backward compatibility layer"

    stats = await default_manager.get_queue_stats(test_key)
    assert stats["total_count"] == 1, "Statistics error in backward compatibility layer"

    # 3. Test append with timestamp
    dt = get_now_with_timezone()
    success = await default_manager.append(test_key, "datetime_test", timestamp=dt)
    assert success, "Failed to append datetime in backward compatibility layer"

    final_size = await default_manager.get_queue_size(test_key)
    assert final_size == 2, "Final size error in backward compatibility layer"

    # 4. Test new timestamp range query method
    # Add some test data
    base_timestamp = int(time.time() * 1000)
    for i in range(3):
        timestamp = base_timestamp + i * 5000
        data = {"index": i, "content": f"compat_data_{i}"}
        success = await default_manager.append(test_key, data, timestamp=timestamp)
        assert success, f"Failed to add test data {i} in backward compatibility layer"

    # Test get by timestamp range
    range_data = await default_manager.get_by_timestamp_range(test_key)
    assert (
        len(range_data) == 5
    ), f"Timestamp range query in backward compatibility layer failed, expected 5, actual {len(range_data)}"  # 2 existing + 3 new

    # Test limit count
    limited_data = await default_manager.get_by_timestamp_range(test_key, limit=2)
    assert (
        len(limited_data) == 2
    ), "Limit count function in backward compatibility layer failed"

    logger.info("New method test in backward compatibility layer passed")

    # 5. Clean up test data
    success = await default_manager.clear_queue(test_key)
    assert success, "Failed to clean up data in backward compatibility layer"

    logger.info("✅ Backward compatibility layer test passed")


async def main():
    """Main test function"""
    logger.info("=" * 50)
    logger.info("Redis length-limited cache manager test started")
    logger.info("=" * 50)

    try:
        # Run all tests
        await test_json_pickle_mixed_data()
        await test_basic_operations()
        await test_length_cleanup()
        await test_timestamp_compatibility()
        await test_timestamp_range_query()
        await test_pickle_error_handling()
        await test_stress_operations()
        await test_expiry_mechanism()
        await test_compatibility_layer()

        logger.info("=" * 50)
        logger.info("✅ All tests passed")
        logger.info("=" * 50)

    except AssertionError as e:
        logger.error("❌ Test failed: %s", str(e))
        raise
    except Exception as e:
        logger.error("❌ Test error: %s", str(e))
        raise


if __name__ == "__main__":
    asyncio.run(main())

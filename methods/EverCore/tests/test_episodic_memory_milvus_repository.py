#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test the functionality of EpisodicMemoryMilvusRepository

Test contents include:
1. Basic CRUD operations (create, read, update, delete)
2. Vector search and filtering functions
3. Batch deletion function
4. Timezone handling
"""

import asyncio
from datetime import datetime, timedelta
import json
from zoneinfo import ZoneInfo
import numpy as np
from typing import List
from core.di import get_bean_by_type
from common_utils.datetime_utils import get_now_with_timezone, to_iso_format
from infra_layer.adapters.out.search.repository.episodic_memory_milvus_repository import (
    EpisodicMemoryMilvusRepository,
)
from core.observation.logger import get_logger

logger = get_logger(__name__)


def compare_datetime(dt1: datetime, dt2: datetime) -> bool:
    """Compare two datetime objects, only compare up to second-level precision"""
    return dt1.replace(microsecond=0) == dt2.replace(microsecond=0)


def generate_random_vector(dim: int = 1024) -> List[float]:
    """Generate random vectors for testing"""
    return np.random.randn(dim).astype(np.float32).tolist()


def build_episodic_memory_entity(
    event_id: str,
    user_id: str,
    timestamp: datetime,
    episode: str,
    search_content: List[str],
    vector: List[float],
    user_name: str = "",
    title: str = "",
    summary: str = "",
    group_id: str = "",
    participants: List[str] = None,
    event_type: str = "",
    keywords: List[str] = None,
    linked_entities: List[str] = None,
    created_at: datetime = None,
    updated_at: datetime = None,
) -> dict:
    """
    Build episodic memory entity for testing

    Args:
        event_id: event ID
        user_id: user ID
        timestamp: event timestamp
        episode: episode description
        search_content: list of search content
        vector: vector
        other parameters are optional

    Returns:
        dict: entity dictionary that can be directly inserted into Milvus
    """
    now = get_now_with_timezone()
    if created_at is None:
        created_at = now
    if updated_at is None:
        updated_at = now

    # Build metadata
    metadata = {}
    if user_name:
        metadata["user_name"] = user_name
    if title:
        metadata["title"] = title
    if summary:
        metadata["summary"] = summary
    if participants:
        metadata["participants"] = participants
    if keywords:
        metadata["keywords"] = keywords
    if linked_entities:
        metadata["linked_entities"] = linked_entities

    # Build entity
    entity = {
        "id": event_id,
        "user_id": user_id,
        "group_id": group_id if group_id is not None else "",
        "event_type": event_type if event_type is not None else "",
        "timestamp": int(timestamp.timestamp()),
        "episode": episode,
        "search_content": json.dumps(search_content, ensure_ascii=False),
        "metadata": json.dumps(metadata, ensure_ascii=False),
        "vector": vector,
        "created_at": int(created_at.timestamp()),
        "updated_at": int(updated_at.timestamp()),
    }

    return entity


async def test_crud_operations():
    """Test basic CRUD operations"""
    logger.info("Starting basic CRUD operations test...")

    repo = get_bean_by_type(EpisodicMemoryMilvusRepository)
    test_event_id = "test_event_crud_001"
    test_user_id = "test_user_crud_123"
    current_time = get_now_with_timezone()

    try:
        # Test Create
        entity = build_episodic_memory_entity(
            event_id=test_event_id,
            user_id=test_user_id,
            timestamp=current_time,
            episode="This is a test episodic memory",
            search_content=["test", "episode", "memory", "CRUD"],
            vector=generate_random_vector(),
            user_name="Test User",
            title="Test Title",
            summary="Test Summary",
            group_id="test_group_001",
            participants=["user1", "user2"],
            event_type="Test",
            keywords=["test", "unit test"],
            linked_entities=["entity1", "entity2"],
        )

        # Insert document
        await repo.collection.insert([entity])

        assert entity is not None
        assert entity["id"] == test_event_id
        assert entity["user_id"] == test_user_id
        assert entity["episode"] == "This is a test episodic memory"
        logger.info("✅ Create operation test successful")

        # Wait for data refresh
        await repo.flush()
        await asyncio.sleep(1)

        # Test Read
        retrieved_doc = await repo.get_by_id(test_event_id)
        assert retrieved_doc is not None
        assert retrieved_doc["id"] == test_event_id
        assert retrieved_doc["user_id"] == test_user_id
        assert retrieved_doc["episode"] == "This is a test episodic memory"
        metadata = json.loads(retrieved_doc["metadata"])
        assert metadata["title"] == "Test Title"
        assert retrieved_doc["group_id"] == "test_group_001"
        logger.info("✅ Read operation test successful")

        # Test Delete
        delete_result = await repo.delete_by_event_id(test_event_id)
        assert delete_result is True
        logger.info("✅ Delete operation test successful")

        # Verify deletion
        await repo.flush()
        deleted_check = await repo.get_by_id(test_event_id)
        assert deleted_check is None, "Document should have been deleted"
        logger.info("✅ Deletion verification successful")

    except Exception as e:
        logger.error("❌ Basic CRUD operations test failed: %s", e)
        # Clean up possible residual data
        try:
            await repo.delete_by_event_id(test_event_id)
            await repo.flush()
        except Exception:
            pass
        raise

    logger.info("✅ Basic CRUD operations test completed")


async def test_vector_search():
    """Test vector search and filtering functions"""
    logger.info("Starting vector search and filtering function test...")

    repo = get_bean_by_type(EpisodicMemoryMilvusRepository)
    test_user_id = "test_user_search_456"
    test_group_id = "test_group_search_789"
    base_time = get_now_with_timezone()
    test_event_ids = []
    base_vector = generate_random_vector()  # Base vector

    try:
        # Create multiple test memories
        test_data = [
            {
                "event_id": f"search_test_001_{int(base_time.timestamp())}",
                "episode": "Discussed the company's development strategy",
                "search_content": ["company", "development", "strategy", "discussion"],
                "vector": [
                    x + 0.1 * np.random.randn() for x in base_vector
                ],  # Similar vector
                "title": "Strategy Meeting",
                "group_id": test_group_id,
                "event_type": "Conversation",
                "keywords": ["meeting", "strategy"],
                "timestamp": base_time - timedelta(days=1),
            },
            {
                "event_id": f"search_test_002_{int(base_time.timestamp())}",
                "episode": "Learned a new technical framework",
                "search_content": [
                    "technology",
                    "framework",
                    "learning",
                    "programming",
                ],
                "vector": generate_random_vector(),  # Random vector
                "title": "Technical Learning",
                "group_id": "",
                "event_type": "Learning",
                "keywords": ["technology", "learning"],
                "timestamp": base_time - timedelta(days=2),
            },
            {
                "event_id": f"search_test_003_{int(base_time.timestamp())}",
                "episode": "Participated in team building activities",
                "search_content": ["team", "building", "activity", "participation"],
                "vector": [
                    x + 0.2 * np.random.randn() for x in base_vector
                ],  # Similar vector
                "title": "Team Activity",
                "group_id": test_group_id,
                "event_type": "Activity",
                "keywords": ["team", "activity"],
                "timestamp": base_time - timedelta(days=3),
            },
        ]

        # Batch create test data
        for data in test_data:
            entity = build_episodic_memory_entity(
                event_id=data["event_id"],
                user_id=test_user_id,
                timestamp=data["timestamp"],
                episode=data["episode"],
                search_content=data["search_content"],
                vector=data["vector"],
                title=data["title"],
                group_id=data["group_id"],
                event_type=data["event_type"],
                keywords=data["keywords"],
            )
            await repo.collection.insert([entity])
            test_event_ids.append(data["event_id"])

        # Refresh collection
        await repo.flush()
        await repo.load()  # Load into memory to improve search performance

        logger.info("✅ Created %d test memories", len(test_data))

        # Wait for data loading
        await asyncio.sleep(2)

        # Test 1: Vector similarity search
        logger.info("Test 1: Vector similarity search")
        results = await repo.vector_search(
            query_vector=base_vector, user_id=test_user_id, limit=10
        )
        assert (
            len(results) >= 2
        ), f"Should find at least 2 similar records, actually found {len(results)}"
        logger.info(
            "✅ Vector similarity search test successful, found %d results",
            len(results),
        )

        # Test 2: Vector search with user ID filter
        logger.info("Test 2: Vector search with user ID filter")
        user_results = await repo.vector_search(
            query_vector=base_vector, user_id=test_user_id, limit=10
        )
        assert (
            len(user_results) >= 2
        ), f"Should find at least 2 user records, actually found {len(user_results)}"
        logger.info(
            "✅ User ID filter test successful, found %d results", len(user_results)
        )

        # Test 3: Vector search with group ID filter
        logger.info("Test 3: Vector search with group ID filter")
        group_results = await repo.vector_search(
            query_vector=base_vector,
            user_id=test_user_id,
            group_id=test_group_id,
            limit=10,
        )
        assert (
            len(group_results) >= 1
        ), f"Should find at least 1 group record, actually found {len(group_results)}"
        logger.info(
            "✅ Group ID filter test successful, found %d results", len(group_results)
        )

        # Test 4: Vector search with event type filter
        logger.info("Test 4: Vector search with event type filter")
        type_results = await repo.vector_search(
            query_vector=base_vector,
            user_id=test_user_id,
            event_type="Conversation",
            limit=10,
        )
        assert (
            len(type_results) >= 1
        ), f"Should find at least 1 Conversation type record, actually found {len(type_results)}"
        logger.info(
            "✅ Event type filter test successful, found %d results", len(type_results)
        )

        # Test 5: Vector search with time range filter
        logger.info("Test 5: Vector search with time range filter")
        time_results = await repo.vector_search(
            query_vector=base_vector,
            user_id=test_user_id,
            start_time=base_time - timedelta(days=2),
            end_time=base_time,
            limit=10,
        )
        assert (
            len(time_results) >= 1
        ), f"Should find at least 1 record within time range, actually found {len(time_results)}"
        logger.info(
            "✅ Time range filter test successful, found %d results", len(time_results)
        )

    except Exception as e:
        logger.error("❌ Vector search and filtering function test failed: %s", e)
        raise
    finally:
        # Clean up test data
        logger.info("Cleaning up search test data...")
        try:
            cleanup_count = await repo.delete_by_filters(user_id=test_user_id)
            await repo.flush()
            logger.info("✅ Cleaned up %d search test data", cleanup_count)
        except Exception as cleanup_error:
            logger.error("Error during cleanup of search test data: %s", cleanup_error)

    logger.info("✅ Vector search and filtering function test completed")


async def test_delete_operations():
    """Test deletion functions"""
    logger.info("Starting deletion function test...")

    repo = get_bean_by_type(EpisodicMemoryMilvusRepository)
    test_user_id = "test_user_delete_789"
    test_group_id = "test_group_delete_012"
    base_time = get_now_with_timezone()
    test_event_ids = []

    try:
        # Create test data
        for i in range(6):
            event_id = f"delete_test_{i}_{int(base_time.timestamp())}"
            test_event_ids.append(event_id)

            entity = build_episodic_memory_entity(
                event_id=event_id,
                user_id=test_user_id,
                timestamp=base_time - timedelta(days=i),
                episode=f"Deletion test memory {i}",
                search_content=["deletion", "test", f"memory{i}"],
                vector=generate_random_vector(),
                title=f"Deletion test {i}",
                group_id=test_group_id if i % 2 == 0 else "",  # Some have group_id
                event_type="DeleteTest",
            )
            await repo.collection.insert([entity])

        await repo.flush()
        logger.info("✅ Created %d deletion test memories", len(test_event_ids))

        # Wait for data refresh
        await asyncio.sleep(2)

        # Test 1: Delete by event_id
        logger.info("Test 1: Delete by event_id")
        event_id_to_delete = test_event_ids[0]
        delete_result = await repo.delete_by_event_id(event_id_to_delete)
        assert delete_result is True

        # Verify deletion
        await repo.flush()
        deleted_doc = await repo.get_by_id(event_id_to_delete)
        assert deleted_doc is None, "Document should have been deleted"
        logger.info("✅ Delete by event_id test successful")

        # Test 2: Delete by filter conditions - only delete memories with group_id
        logger.info("Test 2: Delete by filter (group_id)")
        deleted_count = await repo.delete_by_filters(
            user_id=test_user_id, group_id=test_group_id
        )
        assert (
            deleted_count >= 2
        ), f"Should delete at least 2 records with group_id, actually deleted {deleted_count}"
        logger.info(
            "✅ Delete by group_id filter test successful, deleted %d records",
            deleted_count,
        )

        # Test 3: Delete by time range
        logger.info("Test 3: Delete by time range")
        deleted_count = await repo.delete_by_filters(
            user_id=test_user_id,
            start_time=base_time - timedelta(days=2),
            end_time=base_time,
        )
        logger.info(
            "✅ Delete by time range test successful, deleted %d records", deleted_count
        )

        # Test 4: Verify parameter checking
        logger.info("Test 4: Verify parameter checking")
        try:
            await repo.delete_by_filters()  # No filter conditions provided
            assert False, "Should have raised an exception but did not"
        except ValueError as e:
            logger.info("✅ Correctly caught parameter error: %s", e)

        # Final cleanup of remaining data
        remaining_count = await repo.delete_by_filters(user_id=test_user_id)
        await repo.flush()
        logger.info("✅ Final cleanup of %d remaining data", remaining_count)

    except Exception as e:
        logger.error("❌ Deletion function test failed: %s", e)
        raise
    finally:
        # Ensure all test data is cleaned up
        try:
            await repo.delete_by_filters(user_id=test_user_id)
            await repo.flush()
        except Exception:
            pass

    logger.info("✅ Deletion function test completed")


async def test_timezone_handling():
    """Test timezone handling"""
    logger.info("Starting timezone handling test...")

    repo = get_bean_by_type(EpisodicMemoryMilvusRepository)
    test_event_id = "test_timezone_001"
    test_user_id = "test_user_timezone_999"

    try:
        # Create times in different timezones
        utc_time = get_now_with_timezone(ZoneInfo("UTC"))
        tokyo_time = get_now_with_timezone(ZoneInfo("Asia/Tokyo"))
        shanghai_time = get_now_with_timezone(ZoneInfo("Asia/Shanghai"))

        logger.info("Original UTC time: %s", to_iso_format(utc_time))
        logger.info("Original Tokyo time: %s", to_iso_format(tokyo_time))
        logger.info("Original Shanghai time: %s", to_iso_format(shanghai_time))

        # Create memory using UTC time
        entity = build_episodic_memory_entity(
            event_id=test_event_id,
            user_id=test_user_id,
            timestamp=utc_time,
            episode="Timezone test memory",
            search_content=["timezone", "test"],
            vector=generate_random_vector(),
            title="Timezone Test",
            created_at=tokyo_time,
            updated_at=shanghai_time,
        )

        await repo.collection.insert([entity])

        assert entity is not None
        logger.info("✅ Created memory with timezone information successfully")

        await repo.flush()
        await asyncio.sleep(2)

        # Retrieve from database and verify
        retrieved_doc = await repo.get_by_id(test_event_id)
        assert retrieved_doc is not None

        # Parse timestamp
        retrieved_timestamp = datetime.fromtimestamp(retrieved_doc["timestamp"])
        logger.info(
            "Retrieved timestamp from database: %s", to_iso_format(retrieved_timestamp)
        )

        # Verify time conversion correctness (should be equal after converting to same timezone)
        assert compare_datetime(
            retrieved_timestamp.astimezone(ZoneInfo("UTC")),
            utc_time.astimezone(ZoneInfo("UTC")),
        )
        logger.info("✅ Timezone verification successful")

        # Test time range query
        results = await repo.vector_search(
            query_vector=generate_random_vector(),
            user_id=test_user_id,
            start_time=shanghai_time - timedelta(hours=2),
            end_time=shanghai_time + timedelta(hours=2),
            limit=10,
        )
        assert len(results) >= 1, "Should find records within time range"
        logger.info("✅ Timezone time range query test successful")

    except Exception as e:
        logger.error("❌ Timezone handling test failed: %s", e)
        raise
    finally:
        # Clean up test data
        try:
            await repo.delete_by_event_id(test_event_id)
            await repo.flush()
            logger.info("✅ Cleaned up timezone test data successfully")
        except Exception:
            pass

    logger.info("✅ Timezone handling test completed")


async def test_edge_cases():
    """Test edge cases"""
    logger.info("Starting edge cases test...")

    repo = get_bean_by_type(EpisodicMemoryMilvusRepository)
    test_user_id = "test_user_edge_111"

    try:
        # Test 1: Non-existent user
        logger.info("Test 1: Non-existent user")
        nonexistent_results = await repo.vector_search(
            query_vector=generate_random_vector(),
            user_id="nonexistent_user_999999",
            limit=10,
        )
        assert (
            len(nonexistent_results) == 0
        ), "Non-existent user should return empty results"
        logger.info("✅ Non-existent user test successful")

        # Test 2: Delete non-existent event_id
        logger.info("Test 2: Delete non-existent event_id")
        delete_result = await repo.delete_by_event_id("nonexistent_event_999999")
        assert (
            delete_result is True
        ), "Deleting non-existent document somehow returns True"
        logger.info("✅ Delete non-existent document test successful")

        # Test 3: Use invalid time range
        logger.info("Test 3: Use invalid time range")
        future_time = get_now_with_timezone(ZoneInfo("UTC")) + timedelta(days=365)
        future_results = await repo.vector_search(
            query_vector=generate_random_vector(),
            user_id=test_user_id,
            start_time=future_time,
            end_time=future_time + timedelta(days=1),
            limit=10,
        )
        assert len(future_results) == 0, "Future time range should return empty results"
        logger.info("✅ Invalid time range test successful")

        # Test 4: Vector dimension validation
        logger.info("Test 4: Vector dimension validation")
        try:
            entity = build_episodic_memory_entity(
                event_id="invalid_vector_test",
                user_id=test_user_id,
                timestamp=get_now_with_timezone(),
                episode="Invalid vector test",
                search_content=["test"],
                vector=[1.0] * 512,  # Incorrect vector dimension
            )
            await repo.collection.insert([entity])
            assert False, "Should fail due to vector dimension error"
        except Exception as e:
            assert "the length(512) of float data should divide the dim(1024)" in str(e)
            logger.info("✅ Correctly caught vector dimension error: %s", e)

    except Exception as e:
        logger.error("❌ Edge cases test failed: %s", e)
        raise

    logger.info("✅ Edge cases test completed")


async def test_performance():
    """Test performance"""
    logger.info("Starting performance test...")

    repo = get_bean_by_type(EpisodicMemoryMilvusRepository)
    test_user_id = "test_user_perf_001"
    current_time = get_now_with_timezone()
    num_docs = 1000

    try:
        # Prepare test data
        test_data = []
        base_vector = generate_random_vector()

        for i in range(num_docs):
            # Generate a vector similar to the base vector
            noise = np.random.normal(0, 0.1, len(base_vector))
            vector = [x + n for x, n in zip(base_vector, noise)]

            test_data.append(
                {
                    "event_id": f"perf_test_{i}",
                    "user_id": test_user_id,
                    "timestamp": current_time - timedelta(minutes=i),
                    "episode": f"Performance test memory {i}",
                    "search_content": ["performance", "test", f"memory{i}"],
                    "vector": vector,
                    "title": f"Performance test {i}",
                    "group_id": "perf_test_group",
                    "event_type": "PerfTest",
                }
            )

        # Test 1: Batch insertion performance
        logger.info("Test 1: Batch insertion performance (%d records)...", num_docs)
        insert_times = []
        batch_size = 100

        for i in range(0, num_docs, batch_size):
            batch = test_data[i : i + batch_size]
            start_time = get_now_with_timezone()

            for doc in batch:
                entity = build_episodic_memory_entity(**doc)
                await repo.collection.insert([entity])

            end_time = get_now_with_timezone()
            insert_time = (end_time - start_time).total_seconds()
            insert_times.append(insert_time)

            logger.info(
                "- Batch %d/%d: %.3f seconds (%.1f records/second)",
                i // batch_size + 1,
                (num_docs + batch_size - 1) // batch_size,
                insert_time,
                len(batch) / insert_time,
            )

        avg_insert_time = sum(insert_times) / len(insert_times)
        min_insert_time = min(insert_times)
        max_insert_time = max(insert_times)
        total_insert_time = sum(insert_times)

        logger.info("Insertion performance statistics:")
        logger.info("- Total time: %.3f seconds", total_insert_time)
        logger.info(
            "- Average per batch: %.3f seconds (%.1f records/second)",
            avg_insert_time,
            batch_size / avg_insert_time,
        )
        logger.info(
            "- Fastest batch: %.3f seconds (%.1f records/second)",
            min_insert_time,
            batch_size / min_insert_time,
        )
        logger.info(
            "- Slowest batch: %.3f seconds (%.1f records/second)",
            max_insert_time,
            batch_size / max_insert_time,
        )

        # Test 2: Flush performance
        logger.info("Test 2: Flush performance...")
        start_time = get_now_with_timezone()
        await repo.flush()
        flush_time = (get_now_with_timezone() - start_time).total_seconds()
        logger.info("Flush time: %.3f seconds", flush_time)

        # Wait for data loading
        await repo.load()
        await asyncio.sleep(2)

        # Test 3: Search performance
        logger.info("Test 3: Search performance...")
        search_times = []
        num_searches = 10

        for i in range(num_searches):
            # Generate a query vector similar to the base vector
            noise = np.random.normal(0, 0.1, len(base_vector))
            query_vector = [x + n for x, n in zip(base_vector, noise)]

            start_time = get_now_with_timezone()
            results = await repo.vector_search(
                query_vector=query_vector, user_id=test_user_id, limit=10
            )
            search_time = (get_now_with_timezone() - start_time).total_seconds()
            search_times.append(search_time)

            logger.info(
                "- Search %d/%d: %.3f seconds, found %d results",
                i + 1,
                num_searches,
                search_time,
                len(results),
            )

        avg_search_time = sum(search_times) / len(search_times)
        min_search_time = min(search_times)
        max_search_time = max(search_times)

        logger.info("Search performance statistics:")
        logger.info("- Average time: %.3f seconds", avg_search_time)
        logger.info("- Fastest time: %.3f seconds", min_search_time)
        logger.info("- Slowest time: %.3f seconds", max_search_time)

    except Exception as e:
        logger.error("❌ Performance test failed: %s", e)
        raise
    finally:
        # Clean up test data
        try:
            cleanup_count = await repo.delete_by_filters(user_id=test_user_id)
            await repo.flush()
            logger.info("✅ Cleaned up %d performance test data", cleanup_count)
        except Exception as cleanup_error:
            logger.error(
                "Error during cleanup of performance test data: %s", cleanup_error
            )

    logger.info("✅ Performance test completed")


async def run_all_tests():
    """Run all tests"""
    logger.info("🚀 Starting all EpisodicMemoryMilvusRepository tests...")

    try:
        await test_crud_operations()
        await test_vector_search()
        await test_delete_operations()
        await test_timezone_handling()
        await test_edge_cases()
        await test_performance()
        logger.info("✅ All tests completed")
    except Exception as e:
        logger.error("❌ Error occurred during testing: %s", e)
        raise


if __name__ == "__main__":
    asyncio.run(run_all_tests())

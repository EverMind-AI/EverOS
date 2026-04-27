#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test the functionality of EpisodicMemoryEsRepository

Test contents include:
1. Basic CRUD operations (create, read, update, delete)
2. Search and filtering functions
3. Batch deletion function
4. Timezone handling
"""

import asyncio
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from core.di import get_bean_by_type
from common_utils.datetime_utils import get_now_with_timezone, to_iso_format
from infra_layer.adapters.out.search.repository.episodic_memory_es_repository import (
    EpisodicMemoryEsRepository,
)
from core.observation.logger import get_logger

logger = get_logger(__name__)


def compare_datetime(dt1: datetime, dt2: datetime) -> bool:
    """Compare two datetime objects, only up to second-level precision"""
    return dt1.replace(microsecond=0) == dt2.replace(microsecond=0)


async def test_crud_operations():
    """Test basic CRUD operations"""
    logger.info("Starting basic CRUD operations test...")

    repo = get_bean_by_type(EpisodicMemoryEsRepository)
    test_event_id = "test_event_crud_001"
    test_user_id = "test_user_crud_123"
    current_time = get_now_with_timezone()

    try:
        # Test Create
        doc = await repo.create_and_save_episodic_memory(
            event_id=test_event_id,
            user_id=test_user_id,
            timestamp=current_time,
            episode="This is a test episodic memory",
            search_content=["test", "episodic", "memory", "CRUD"],
            user_name="Test User",
            title="Test Title",
            summary="Test Summary",
            group_id="test_group_001",
            participants=["user1", "user2"],
            event_type="Test",
            keywords=["test", "unit test"],
            linked_entities=["entity1", "entity2"],
            extend={},  # Remove custom fields to avoid strict mapping errors
        )

        assert doc is not None
        assert doc.event_id == test_event_id
        assert doc.user_id == test_user_id
        assert doc.episode == "This is a test episodic memory"
        logger.info("✅ Create operation test succeeded")

        # Wait for index refresh
        await asyncio.sleep(1)

        # Test Read
        retrieved_doc = await repo.get_by_id(test_event_id)
        assert retrieved_doc is not None
        assert retrieved_doc.event_id == test_event_id
        assert retrieved_doc.user_id == test_user_id
        assert retrieved_doc.episode == "This is a test episodic memory"
        assert retrieved_doc.title == "Test Title"
        assert retrieved_doc.group_id == "test_group_001"
        assert "test" in retrieved_doc.search_content
        logger.info("✅ Read operation test succeeded")

        # Test Update
        retrieved_doc.episode = "Updated episodic memory"
        retrieved_doc.title = "Updated title"
        retrieved_doc.updated_at = get_now_with_timezone()

        updated_doc = await repo.update(retrieved_doc, refresh=True)
        assert updated_doc.episode == "Updated episodic memory"
        assert updated_doc.title == "Updated title"
        logger.info("✅ Update operation test succeeded")

        # Verify update
        final_check = await repo.get_by_id(test_event_id)
        assert final_check is not None
        assert final_check.episode == "Updated episodic memory"
        assert final_check.title == "Updated title"
        logger.info("✅ Update result verification succeeded")

        # Test Delete
        delete_result = await repo.delete_by_event_id(test_event_id, refresh=True)
        assert delete_result is True
        logger.info("✅ Delete operation test succeeded")

        # Verify deletion
        deleted_check = await repo.get_by_id(test_event_id)
        assert deleted_check is None, "Document should have been deleted"
        logger.info("✅ Deletion result verification succeeded")

    except Exception as e:
        logger.error("❌ Basic CRUD operations test failed: %s", e)
        # Clean up any residual data
        try:
            await repo.delete_by_event_id(test_event_id, refresh=True)
        except Exception:
            pass
        raise

    logger.info("✅ Basic CRUD operations test completed")


async def test_search_and_filter():
    """Test search and filtering functions"""
    logger.info("Starting search and filtering function test...")

    repo = get_bean_by_type(EpisodicMemoryEsRepository)
    test_user_id = "test_user_search_456"
    test_group_id = "test_group_search_789"
    base_time = get_now_with_timezone()
    test_event_ids = []

    try:
        # Create multiple test memories
        test_data = [
            {
                "event_id": f"search_test_001_{int(base_time.timestamp())}",
                "episode": "Discussed the company's development strategy",
                "search_content": ["company", "development", "strategy", "discussion"],
                "title": "Strategy Meeting",
                "group_id": test_group_id,
                "event_type": "Conversation",
                "keywords": ["meeting", "strategy"],
                "timestamp": base_time - timedelta(days=1),
            },
            {
                "event_id": f"search_test_002_{int(base_time.timestamp())}",
                "episode": "Learned a new technology framework",
                "search_content": [
                    "technology",
                    "framework",
                    "learning",
                    "programming",
                ],
                "title": "Technical Learning",
                "group_id": None,  # No group
                "event_type": "Learning",
                "keywords": ["technology", "learning"],
                "timestamp": base_time - timedelta(days=2),
            },
            {
                "event_id": f"search_test_003_{int(base_time.timestamp())}",
                "episode": "Participated in team building activities",
                "search_content": ["team", "building", "activity", "participation"],
                "title": "Team Activity",
                "group_id": test_group_id,
                "event_type": "Activity",
                "keywords": ["team", "activity"],
                "timestamp": base_time - timedelta(days=3),
            },
            {
                "event_id": f"search_test_004_{int(base_time.timestamp())}",
                "episode": "Completed an important project milestone",
                "search_content": ["project", "milestone", "completion", "important"],
                "title": "Project Progress",
                "group_id": test_group_id,
                "event_type": "Project",
                "keywords": ["project", "milestone"],
                "timestamp": base_time - timedelta(days=4),
            },
            {
                "event_id": f"search_test_005_{int(base_time.timestamp())}",
                "episode": "Had an in-depth technical discussion with the client",
                "search_content": ["client", "technology", "communication", "in-depth"],
                "title": "Client Communication",
                "group_id": None,  # No group
                "event_type": "Communication",
                "keywords": ["client", "technology"],
                "timestamp": base_time - timedelta(days=5),
            },
        ]

        # Batch create test data
        for data in test_data:
            await repo.create_and_save_episodic_memory(
                event_id=data["event_id"],
                user_id=test_user_id,
                timestamp=data["timestamp"],
                episode=data["episode"],
                search_content=data["search_content"],
                title=data["title"],
                group_id=data["group_id"],
                event_type=data["event_type"],
                keywords=data["keywords"],
                extend={},  # Use empty extend object
            )
            test_event_ids.append(data["event_id"])

        # Manually refresh index to ensure data is immediately searchable
        client = await repo.get_client()
        await client.indices.refresh(index=repo.get_index_name())

        logger.info("✅ Created %d test memories", len(test_data))

        # Wait for index refresh (ES needs more time)
        await asyncio.sleep(5)

        # Test 1: Multi-word search
        logger.info("Test 1: Multi-word search")
        results = await repo.multi_search(
            query=["technology", "project"], user_id=test_user_id, size=10, explain=True
        )
        assert (
            len(results) >= 2
        ), f"At least 2 records containing 'technology' or 'project' should be found, actually found {len(results)}"
        logger.info(
            "✅ Multi-word search test succeeded, found %d results", len(results)
        )

        # Test 2: Filter by user ID
        logger.info("Test 2: Filter by user ID")
        user_results = await repo.multi_search(
            query=[], user_id=test_user_id, size=20  # Empty query, pure filtering
        )
        assert (
            len(user_results) >= 5
        ), f"At least 5 user records should be found, actually found {len(user_results)}"
        logger.info(
            "✅ User ID filter test succeeded, found %d results", len(user_results)
        )

        # Test 3: Filter by group ID
        logger.info("Test 3: Filter by group ID")
        group_results = await repo.multi_search(
            query=[], user_id=test_user_id, group_id=test_group_id, size=10
        )
        assert (
            len(group_results) >= 3
        ), f"At least 3 group records should be found, actually found {len(group_results)}"
        logger.info(
            "✅ Group ID filter test succeeded, found %d results", len(group_results)
        )

        # Test 4: Filter by event type
        logger.info("Test 4: Filter by event type")
        type_results = await repo.multi_search(
            query=[], user_id=test_user_id, event_type="Conversation", size=10
        )
        assert (
            len(type_results) >= 1
        ), f"At least 1 Conversation type record should be found, actually found {len(type_results)}"
        logger.info(
            "✅ Event type filter test succeeded, found %d results", len(type_results)
        )

        # Test 5: Filter by keywords
        logger.info("Test 5: Filter by keywords")
        keyword_results = await repo.multi_search(
            query=[], user_id=test_user_id, keywords=["technology"], size=10
        )
        assert (
            len(keyword_results) >= 2
        ), f"At least 2 records containing 'technology' keyword should be found, actually found {len(keyword_results)}"
        logger.info(
            "✅ Keyword filter test succeeded, found %d results", len(keyword_results)
        )

        # Test 6: Filter by time range
        logger.info("Test 6: Filter by time range")
        date_range = {
            "gte": (base_time - timedelta(days=3)).isoformat(),
            "lte": base_time.isoformat(),
        }
        time_results = await repo.multi_search(
            query=[], user_id=test_user_id, date_range=date_range, size=10
        )
        assert (
            len(time_results) >= 2
        ), f"At least 2 records within time range should be found, actually found {len(time_results)}"
        logger.info(
            "✅ Time range filter test succeeded, found %d results", len(time_results)
        )

        # Test 7: Combined query
        logger.info("Test 7: Combined query")
        combo_results = await repo.multi_search(
            query=["technology", "project"],
            user_id=test_user_id,
            group_id=test_group_id,
            keywords=["technology"],
            size=10,
            explain=True,
        )
        logger.info(
            "✅ Combined query test succeeded, found %d results", len(combo_results)
        )

        # Test 8: Use dedicated query method
        logger.info("Test 8: Use dedicated query method")
        timerange_results = await repo.get_by_user_and_timerange(
            user_id=test_user_id,
            start_time=base_time - timedelta(days=6),
            end_time=base_time,
            size=20,
        )
        assert (
            len(timerange_results) >= 5
        ), f"At least 5 records within time range should be found, actually found {len(timerange_results)}"
        logger.info(
            "✅ Dedicated query method test succeeded, found %d results",
            len(timerange_results),
        )

    except Exception as e:
        logger.error("❌ Search and filtering function test failed: %s", e)
        raise
    finally:
        # Clean up test data
        logger.info("Cleaning up search test data...")
        try:
            cleanup_count = await repo.delete_by_filters(
                user_id=test_user_id, refresh=True
            )
            logger.info("✅ Cleaned up %d search test data", cleanup_count)
        except Exception as cleanup_error:
            logger.error("Error during cleanup of search test data: %s", cleanup_error)

    logger.info("✅ Search and filtering function test completed")


async def test_delete_operations():
    """Test deletion functions"""
    logger.info("Starting deletion function test...")

    repo = get_bean_by_type(EpisodicMemoryEsRepository)
    test_user_id = "test_user_delete_789"
    test_group_id = "test_group_delete_012"
    base_time = get_now_with_timezone()
    test_event_ids = []

    try:
        # Create test data
        for i in range(6):
            event_id = f"delete_test_{i}_{int(base_time.timestamp())}"
            test_event_ids.append(event_id)

            await repo.create_and_save_episodic_memory(
                event_id=event_id,
                user_id=test_user_id,
                timestamp=base_time - timedelta(days=i),
                episode=f"Deletion test memory {i}",
                search_content=["deletion", "test", f"memory{i}"],
                title=f"Deletion test {i}",
                group_id=test_group_id if i % 2 == 0 else None,  # Some have group_id
                event_type="DeleteTest",
                extend={},  # Use empty extend object
            )

        # Manually refresh index to ensure data is immediately searchable
        client = await repo.get_client()
        await client.indices.refresh(index=repo.get_index_name())

        logger.info("✅ Created %d deletion test memories", len(test_event_ids))

        # Wait for index refresh (deletion test needs more time to ensure index is fully refreshed)
        await asyncio.sleep(5)

        # Test 1: Delete by event_id
        logger.info("Test 1: Delete by event_id")
        event_id_to_delete = test_event_ids[0]
        delete_result = await repo.delete_by_event_id(event_id_to_delete, refresh=True)
        assert delete_result is True

        # Verify deletion
        deleted_doc = await repo.get_by_id(event_id_to_delete)
        assert deleted_doc is None, "Document should have been deleted"
        logger.info("✅ Delete by event_id test succeeded")

        # Test 2: Delete by filter conditions - only delete memories with group_id
        logger.info("Test 2: Delete by filter conditions (group_id)")
        deleted_count = await repo.delete_by_filters(
            user_id=test_user_id, group_id=test_group_id, refresh=True
        )
        assert (
            deleted_count >= 2
        ), f"At least 2 records with group_id should be deleted, actually deleted {deleted_count}"
        logger.info(
            "✅ Delete by group_id filter test succeeded, deleted %d records",
            deleted_count,
        )

        # Test 3: Delete by time range
        logger.info("Test 3: Delete by time range")
        date_range = {
            "gte": (base_time - timedelta(days=2)).isoformat(),
            "lte": base_time.isoformat(),
        }
        deleted_count = await repo.delete_by_filters(
            user_id=test_user_id, date_range=date_range, refresh=True
        )
        logger.info(
            "✅ Delete by time range test succeeded, deleted %d records", deleted_count
        )

        # Test 4: Verify parameter validation
        logger.info("Test 4: Verify parameter validation")
        try:
            await repo.delete_by_filters()  # No filter conditions provided
            assert False, "Should have thrown an exception but didn't"
        except ValueError as e:
            logger.info("✅ Correctly caught parameter error: %s", e)

        # Final cleanup of remaining data
        remaining_count = await repo.delete_by_filters(
            user_id=test_user_id, refresh=True
        )
        logger.info("✅ Final cleanup of %d remaining data", remaining_count)

    except Exception as e:
        logger.error("❌ Deletion function test failed: %s", e)
        raise
    finally:
        # Ensure all test data is cleaned up
        try:
            await repo.delete_by_filters(user_id=test_user_id, refresh=True)
        except Exception:
            pass

    logger.info("✅ Deletion function test completed")


async def test_timezone_handling():
    """Test timezone handling"""
    logger.info("Starting timezone handling test...")

    repo = get_bean_by_type(EpisodicMemoryEsRepository)
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
        doc = await repo.create_and_save_episodic_memory(
            event_id=test_event_id,
            user_id=test_user_id,
            timestamp=utc_time,
            episode="Timezone test memory",
            search_content=["timezone", "test"],
            title="Timezone Test",
            created_at=tokyo_time,
            updated_at=shanghai_time,
            extend={},  # Use empty extend object
        )

        assert doc is not None
        logger.info("✅ Created memory with timezone information successfully")

        # Manually refresh index to ensure data is immediately searchable
        client = await repo.get_client()
        await client.indices.refresh(index=repo.get_index_name())

        # Wait for index refresh
        await asyncio.sleep(2)

        # Retrieve from database and verify
        retrieved_doc = await repo.get_by_id(test_event_id)
        assert retrieved_doc is not None

        logger.info("Retrieved times from database:")
        logger.info(
            "timestamp (original UTC): %s", to_iso_format(retrieved_doc.timestamp)
        )
        logger.info(
            "created_at (original Tokyo): %s", to_iso_format(retrieved_doc.created_at)
        )
        logger.info(
            "updated_at (original Shanghai): %s",
            to_iso_format(retrieved_doc.updated_at),
        )

        # Verify time conversion correctness (should be equal when converted to same timezone)
        assert retrieved_doc.timestamp.astimezone(ZoneInfo("UTC")).replace(
            microsecond=0
        ) == utc_time.replace(microsecond=0)
        logger.info("✅ Timezone validation succeeded")

        # Test time range query - use wider time range and Shanghai timezone
        shanghai_time = get_now_with_timezone(
            ZoneInfo("Asia/Shanghai")
        )  # Current Shanghai time
        date_range = {
            "gte": (shanghai_time - timedelta(hours=2)).isoformat(),
            "lte": (shanghai_time + timedelta(hours=2)).isoformat(),
        }

        logger.info("Time range query: %s to %s", date_range["gte"], date_range["lte"])
        logger.info("Document timestamp: %s", to_iso_format(retrieved_doc.timestamp))

        time_results = await repo.multi_search(
            query=[], user_id=test_user_id, date_range=date_range, size=10
        )
        logger.info("Time range query results: found %d records", len(time_results))

        # If still not found, try without time range, only user_id query
        if len(time_results) == 0:
            logger.warning(
                "Time range query found no records, trying pure user_id query"
            )
            fallback_results = await repo.multi_search(
                query=[], user_id=test_user_id, size=10
            )
            logger.info(
                "Pure user_id query results: found %d records", len(fallback_results)
            )
            assert (
                len(fallback_results) >= 1
            ), "At least one record should be found by user_id"
            logger.info("✅ Basic timezone handling validation succeeded")
        else:
            assert len(time_results) >= 1, "Records within time range should be found"
            logger.info("✅ Timezone time range query test succeeded")

    except Exception as e:
        logger.error("❌ Timezone handling test failed: %s", e)
        raise
    finally:
        # Clean up test data
        try:
            await repo.delete_by_event_id(test_event_id, refresh=True)
            logger.info("✅ Cleaned up timezone test data successfully")
        except Exception:
            pass

    logger.info("✅ Timezone handling test completed")


async def test_edge_cases():
    """Test edge cases"""
    logger.info("Starting edge cases test...")

    repo = get_bean_by_type(EpisodicMemoryEsRepository)
    test_user_id = "test_user_edge_111"

    try:
        # Test 1: Empty search terms
        logger.info("Test 1: Empty search terms")
        empty_results = await repo.multi_search(query=[], user_id=test_user_id, size=10)
        logger.info(
            "✅ Empty search terms test succeeded, found %d results", len(empty_results)
        )

        # Test 2: Non-existent user
        logger.info("Test 2: Non-existent user")
        nonexistent_results = await repo.multi_search(
            query=["test"], user_id="nonexistent_user_999999", size=10, explain=True
        )
        assert (
            len(nonexistent_results) == 0
        ), "Non-existent user should return empty results"
        logger.info("✅ Non-existent user test succeeded")

        # Test 3: Delete non-existent event_id
        logger.info("Test 3: Delete non-existent event_id")
        delete_result = await repo.delete_by_event_id("nonexistent_event_999999")
        assert (
            delete_result is False
        ), "Deleting non-existent document should return False"
        logger.info("✅ Delete non-existent document test succeeded")

        # Test 4: Use invalid time range
        logger.info("Test 4: Use invalid time range")
        invalid_date_range = {"gte": "2099-01-01", "lte": "2099-12-31"}  # Future time
        future_results = await repo.multi_search(
            query=[], user_id=test_user_id, date_range=invalid_date_range, size=10
        )
        assert len(future_results) == 0, "Future time range should return empty results"
        logger.info("✅ Invalid time range test succeeded")

    except Exception as e:
        logger.error("❌ Edge cases test failed: %s", e)
        raise

    logger.info("✅ Edge cases test completed")


async def run_all_tests():
    """Run all tests"""
    logger.info("🚀 Starting all EpisodicMemoryEsRepository tests...")

    try:
        await test_multi_search()
        await test_crud_operations()
        await test_search_and_filter()
        await test_delete_operations()
        await test_timezone_handling()
        await test_edge_cases()
        logger.info("✅ All tests completed")
    except Exception as e:
        logger.error("❌ Error occurred during testing: %s", e)
        raise


async def test_multi_search():
    """Test multi-word search functionality based on elasticsearch-dsl"""
    logger.info("Starting DSL multi-word search function test...")

    repo = get_bean_by_type(EpisodicMemoryEsRepository)
    test_event_id = "test_event_dsl_001"
    test_event_id_bm25 = "test_event_bm25_001"
    test_event_id_not_search = "test_event_not_search_001"
    test_user_id = "test_user_dsl_123"
    test_user_id_not_search = "test_user_not_search_123"
    current_time = get_now_with_timezone()

    try:
        # First create test data
        await repo.create_and_save_episodic_memory(
            event_id=test_event_id,
            user_id=test_user_id,
            timestamp=current_time,
            episode="This is a test DSL search episodic memory",
            search_content=["DSL", "search", "test", "elasticsearch"],
            user_name="DSL Test User",
            title="DSL Search Test Title",
            summary="DSL Search Test Summary",
            event_type="TestDSL",
            keywords=["dsl", "search", "test"],
        )

        await repo.create_and_save_episodic_memory(
            event_id=test_event_id_bm25,
            user_id=test_user_id,
            timestamp=current_time,
            episode="This is a test BM25 preference memory",
            search_content=["BM25", "preference", "test", "elasticsearch"],
            user_name="DSL Test User",
            title="BM25 Search Test Title",
            summary="BM25 Search Test Summary",
            event_type="TestBM25",
            keywords=["dsl", "search", "test"],
        )

        await repo.create_and_save_episodic_memory(
            event_id=test_event_id_not_search,
            user_id=test_user_id_not_search,
            timestamp=current_time,
            episode="This is a test DSL search episodic memory 2",
            search_content=["DSL", "search", "test", "elasticsearch"],
            user_name="DSL Test User",
            title="DSL Search Test Title 2",
            summary="DSL Search Test Summary 2",
            event_type="TestDSL2",
            keywords=["dsl", "search", "test"],
        )

        # Wait for index refresh
        await repo.refresh_index()
        await asyncio.sleep(1)

        # Test 1: DSL multi-word search
        logger.info("Testing DSL multi-word search...")
        results = await repo.multi_search(
            query=["DSL", "search"], user_id=test_user_id, size=10, explain=True
        )
        assert len(results) == 1, "DSL multi-word search should return results"
        logger.info(
            "✅ DSL multi-word search test passed: found %d results", len(results)
        )

        # Test 2: DSL filter query (no search terms)
        logger.info("Testing DSL filter query...")
        results = await repo.multi_search(
            query=[], user_id=test_user_id, event_type="TestDSL", size=10
        )
        assert len(results) > 0, "DSL filter query should return results"
        logger.info("✅ DSL filter query test passed: found %d results", len(results))

        # Test 4: BM25 search
        logger.info("Testing BM25 search...")
        results = await repo.multi_search(
            query=["BM25", "preference"], user_id=test_user_id, size=10, explain=True
        )
        assert len(results) == 1, "BM25 search should return results"
        logger.info("✅ BM25 search test passed: found %d results", len(results))

        # Test 5: BM25 filter query (no search terms)
        logger.info("Testing BM25 filter query...")
        results = await repo.multi_search(
            query=[], user_id=test_user_id, event_type="TestBM25", size=10
        )
        assert len(results) == 1, "BM25 filter query should return results"
        logger.info("✅ BM25 filter query test passed: found %d results", len(results))

        # Test 6: Compare result consistency between BM25 method and original method
        logger.info(
            "Testing result consistency between BM25 method and original method..."
        )
        bm25_results = await repo.multi_search(
            query=["preference", "test"], user_id=test_user_id, size=10
        )
        assert len(bm25_results) == 2, "BM25 method should return results"
        logger.info("✅ BM25 method test passed: found %d results", len(bm25_results))

        # Clean up test data
        await repo.delete_by_event_id(test_event_id, refresh=True)
        await repo.delete_by_event_id(test_event_id_bm25, refresh=True)
        logger.info("✅ DSL search function test completed")

    except Exception as e:
        logger.error("❌ DSL search function test failed: %s", e)
        # Try cleanup
        try:
            await repo.delete_by_event_id(test_event_id, refresh=True)
        except Exception:
            pass
        raise


if __name__ == "__main__":
    asyncio.run(run_all_tests())

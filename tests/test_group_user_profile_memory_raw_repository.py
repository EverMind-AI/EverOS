#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test version management functionality of GroupUserProfileMemoryRawRepository

Test contents include:
1. CRUD operations based on user_id+group_id (with version management support)
2. Version management related functionality tests
3. ensure_latest method test
4. only_latest functionality test for batch queries
"""

import asyncio

from core.di import get_bean_by_type
from infra_layer.adapters.out.persistence.repository.group_user_profile_memory_raw_repository import (
    GroupUserProfileMemoryRawRepository,
)
from core.observation.logger import get_logger

logger = get_logger(__name__)


async def test_basic_crud_operations():
    """Test basic CRUD operations (with version management)"""
    logger.info("Starting test of basic CRUD operations...")

    repo = get_bean_by_type(GroupUserProfileMemoryRawRepository)
    user_id = "test_user_001"
    group_id = "test_group_001"

    try:
        # First clean up any existing test data
        await repo.delete_by_user_group(user_id, group_id)
        logger.info("✅ Cleaned up existing test data")

        # Test creating a new record (version must be provided)
        profile_data = {
            "version": "v1",
            "user_name": "Zhang San",
            "hard_skills": [
                {"value": "Python", "level": "Advanced", "evidences": ["conv_001"]}
            ],
            "personality": [{"value": "Good communication", "evidences": ["conv_002"]}],
        }

        result = await repo.upsert_by_user_group(user_id, group_id, profile_data)
        assert result is not None
        assert result.user_id == user_id
        assert result.group_id == group_id
        assert result.user_name == "Zhang San"
        assert result.version == "v1"
        assert result.is_latest == True
        logger.info("✅ Successfully created new record (version=v1, is_latest=True)")

        # Test querying by user_id and group_id (should return latest version)
        queried = await repo.get_by_user_group(user_id, group_id)
        assert queried is not None
        assert queried.user_id == user_id
        assert queried.group_id == group_id
        assert queried.version == "v1"
        assert queried.is_latest == True
        logger.info("✅ Successfully queried by user_id and group_id")

        # Test updating record (without changing version)
        update_data = {
            "user_name": "Zhang San (updated)",
            "soft_skills": [
                {"value": "Leadership", "level": "Intermediate", "evidences": ["conv_003"]}
            ],
        }

        updated = await repo.update_by_user_group(user_id, group_id, update_data)
        assert updated is not None
        assert updated.user_name == "Zhang San (updated)"
        assert updated.soft_skills is not None
        assert updated.version == "v1"  # Version unchanged
        logger.info("✅ Successfully updated record (version unchanged)")

        # Test deleting specific version
        deleted = await repo.delete_by_user_group(user_id, group_id, version="v1")
        assert deleted is True
        logger.info("✅ Successfully deleted specific version")

        # Verify deletion
        final_check = await repo.get_by_user_group(user_id, group_id)
        assert final_check is None, "Record should have been deleted"
        logger.info("✅ Verified deletion success")

    except Exception as e:
        logger.error("❌ Basic CRUD operations test failed: %s", e)
        raise

    logger.info("✅ Basic CRUD operations test completed")


async def test_version_management():
    """Test version management functionality"""
    logger.info("Starting test of version management functionality...")

    repo = get_bean_by_type(GroupUserProfileMemoryRawRepository)
    user_id = "test_user_version_002"
    group_id = "test_group_version_002"

    try:
        # First clean up any existing test data
        await repo.delete_by_user_group(user_id, group_id)
        logger.info("✅ Cleaned up existing test data")

        # Create first version
        v1_data = {
            "version": "202501",
            "user_name": "Li Si v1",
            "personality": [{"value": "Introverted", "evidences": ["conv_001"]}],
        }

        v1_result = await repo.upsert_by_user_group(user_id, group_id, v1_data)
        assert v1_result is not None
        assert v1_result.version == "202501"
        assert v1_result.is_latest == True
        logger.info("✅ Created version 202501 successfully, is_latest=True")

        # Create second version
        v2_data = {
            "version": "202502",
            "user_name": "Li Si v2",
            "personality": [{"value": "Extroverted", "evidences": ["conv_002"]}],
        }

        v2_result = await repo.upsert_by_user_group(user_id, group_id, v2_data)
        assert v2_result is not None
        assert v2_result.version == "202502"
        assert v2_result.is_latest == True
        logger.info("✅ Created version 202502 successfully, is_latest=True")

        # Create third version
        v3_data = {
            "version": "202503",
            "user_name": "Li Si v3",
            "personality": [{"value": "Balanced", "evidences": ["conv_003"]}],
        }

        v3_result = await repo.upsert_by_user_group(user_id, group_id, v3_data)
        assert v3_result is not None
        assert v3_result.version == "202503"
        assert v3_result.is_latest == True
        logger.info("✅ Created version 202503 successfully, is_latest=True")

        # Test getting latest version (without specifying version_range)
        latest = await repo.get_by_user_group(user_id, group_id)
        assert latest is not None
        assert latest.version == "202503"
        assert latest.is_latest == True
        logger.info("✅ Successfully retrieved latest version: version=202503")

        # Test version range query (closed interval)
        v2_by_range = await repo.get_by_user_group(
            user_id, group_id, version_range=("202502", "202502")
        )
        assert v2_by_range is not None
        assert v2_by_range.version == "202502"
        logger.info("✅ Version range query [202502, 202502] successful, returned version=202502")

        # Test updating specific version
        update_v2 = {"user_name": "Li Si v2 (updated)"}

        updated_v2 = await repo.update_by_user_group(
            user_id, group_id, update_v2, version="202502"
        )
        assert updated_v2 is not None
        assert updated_v2.version == "202502"
        assert updated_v2.user_name == "Li Si v2 (updated)"
        logger.info("✅ Successfully updated specific version 202502")

        # Test deleting middle version
        await repo.delete_by_user_group(user_id, group_id, version="202502")
        logger.info("✅ Successfully deleted version 202502")

        # Verify latest version remains correct after deletion
        latest_after_delete = await repo.get_by_user_group(user_id, group_id)
        assert latest_after_delete is not None
        assert latest_after_delete.version == "202503"
        assert latest_after_delete.is_latest == True
        logger.info("✅ After deleting middle version, latest version is still correct")

        # Clean up all versions
        await repo.delete_by_user_group(user_id, group_id)
        logger.info("✅ Successfully cleaned up test data")

    except Exception as e:
        logger.error("❌ Version management functionality test failed: %s", e)
        raise

    logger.info("✅ Version management functionality test completed")


async def test_ensure_latest():
    """Test ensure_latest method"""
    logger.info("Starting test of ensure_latest method...")

    repo = get_bean_by_type(GroupUserProfileMemoryRawRepository)
    user_id = "test_user_ensure_003"
    group_id = "test_group_ensure_003"

    try:
        # First clean up any existing test data
        await repo.delete_by_user_group(user_id, group_id)
        logger.info("✅ Cleaned up existing test data")

        # Create multiple versions
        versions = ["202501", "202502", "202503", "202504"]
        for version in versions:
            data = {"version": version, "user_name": f"Wang Wu{version}"}
            await repo.upsert_by_user_group(user_id, group_id, data)

        logger.info("✅ Created 4 versions")

        # Manually call ensure_latest
        result = await repo.ensure_latest(user_id, group_id)
        assert result is True
        logger.info("✅ ensure_latest executed successfully")

        # Verify latest version
        latest = await repo.get_by_user_group(user_id, group_id)
        assert latest is not None
        assert latest.version == "202504"
        assert latest.is_latest == True
        logger.info("✅ Verified latest version is correct: version=202504, is_latest=True")

        # Verify old versions have is_latest=False
        for old_version in ["202501", "202502", "202503"]:
            old_doc = await repo.get_by_user_group(
                user_id, group_id, version_range=(old_version, old_version)
            )
            assert old_doc is not None
            assert old_doc.is_latest == False
            logger.info("✅ Verified old version %s has is_latest=False", old_version)

        # Test idempotency: call ensure_latest again
        result2 = await repo.ensure_latest(user_id, group_id)
        assert result2 is True
        logger.info("✅ ensure_latest idempotency verification successful")

        # Clean up test data
        await repo.delete_by_user_group(user_id, group_id)
        logger.info("✅ Successfully cleaned up test data")

    except Exception as e:
        logger.error("❌ ensure_latest method test failed: %s", e)
        raise

    logger.info("✅ ensure_latest method test completed")


async def test_batch_query_with_only_latest():
    """Test only_latest functionality in batch queries"""
    logger.info("Starting test of only_latest functionality in batch queries...")

    repo = get_bean_by_type(GroupUserProfileMemoryRawRepository)
    user_id = "test_batch_user"
    group_id = "test_batch_group"

    try:
        # Create multiple versions for multiple users in the same group
        user_ids = [f"{user_id}_{i}" for i in range(1, 4)]

        # First clean up
        for uid in user_ids:
            await repo.delete_by_user_group(uid, group_id)
        logger.info("✅ Cleaned up existing test data")

        # Create multiple versions for each user
        for uid in user_ids:
            for version in ["202501", "202502", "202503"]:
                data = {"version": version, "user_name": f"{uid}_{version}"}
                await repo.upsert_by_user_group(uid, group_id, data)

        logger.info("✅ Created 3 users with 3 versions each in the same group")

        # Test get_by_user_ids with only_latest=True (default)
        latest_results = await repo.get_by_user_ids(
            user_ids, group_id=group_id, only_latest=True
        )
        assert len(latest_results) == 3

        for result in latest_results:
            assert result.version == "202503"
            assert result.is_latest == True

        logger.info("✅ get_by_user_ids only_latest=True successful, returned 3 latest versions")

        # Test get_by_user_ids with only_latest=False (return all versions)
        all_results = await repo.get_by_user_ids(
            user_ids, group_id=group_id, only_latest=False
        )
        assert len(all_results) == 9  # 3 users * 3 versions
        logger.info("✅ get_by_user_ids only_latest=False successful, returned 9 versions")

        # Test get_by_group_id with only_latest=True
        group_latest = await repo.get_by_group_id(group_id, only_latest=True)
        assert len(group_latest) == 3  # Latest version for 3 users
        logger.info("✅ get_by_group_id only_latest=True successful, returned latest versions for 3 users")

        # Test get_by_group_id with only_latest=False
        group_all = await repo.get_by_group_id(group_id, only_latest=False)
        assert len(group_all) == 9  # All versions
        logger.info("✅ get_by_group_id only_latest=False successful, returned all 9 versions")

        # Clean up test data
        for uid in user_ids:
            await repo.delete_by_user_group(uid, group_id)
        logger.info("✅ Successfully cleaned up test data")

    except Exception as e:
        logger.error("❌ Batch query only_latest functionality test failed: %s", e)
        raise

    logger.info("✅ Batch query only_latest functionality test completed")


async def test_get_profile_method():
    """Test get_profile method"""
    logger.info("Starting test of get_profile method...")

    repo = get_bean_by_type(GroupUserProfileMemoryRawRepository)
    user_id = "test_user_profile_005"
    group_id = "test_group_profile_005"

    try:
        # First clean up
        await repo.delete_by_user_group(user_id, group_id)

        # Create record with complete profile fields
        profile_data = {
            "version": "v1",
            "user_name": "Test User",
            "hard_skills": [
                {"value": "Python", "level": "Advanced", "evidences": ["conv_001"]}
            ],
            "soft_skills": [
                {"value": "Communication", "level": "Excellent", "evidences": ["conv_002"]}
            ],
            "personality": [{"value": "Extroverted", "evidences": ["conv_003"]}],
            "interests": [{"value": "Programming", "evidences": ["conv_004"]}],
            "user_goal": [{"value": "Become a technical expert", "evidences": ["conv_005"]}],
        }

        result = await repo.upsert_by_user_group(user_id, group_id, profile_data)
        assert result is not None
        logger.info("✅ Successfully created record with complete profile fields")

        # Test get_profile method
        profile = repo.get_profile(result)
        assert profile is not None
        assert "hard_skills" in profile
        assert "soft_skills" in profile
        assert "personality" in profile
        assert "interests" in profile
        assert "user_goal" in profile
        assert "work_responsibility" in profile
        assert "working_habit_preference" in profile
        logger.info("✅ get_profile method test successful, contains all fields")

        # Clean up
        await repo.delete_by_user_group(user_id, group_id)
        logger.info("✅ Successfully cleaned up test data")

    except Exception as e:
        logger.error("❌ get_profile method test failed: %s", e)
        raise

    logger.info("✅ get_profile method test completed")


async def test_create_without_version_should_fail():
    """Test that creating without providing version should fail"""
    logger.info("Starting test that creating without version should fail...")

    repo = get_bean_by_type(GroupUserProfileMemoryRawRepository)
    user_id = "test_no_version_006"
    group_id = "test_no_version_006"

    try:
        # First clean up
        await repo.delete_by_user_group(user_id, group_id)

        # Try to create record without version
        data_without_version = {
            "user_name": "User without version",
            "personality": [{"value": "This should fail", "evidences": ["test"]}],
        }

        try:
            await repo.upsert_by_user_group(user_id, group_id, data_without_version)
            assert False, "Creating record without version should raise an exception"
        except ValueError as e:
            logger.info("✅ Correctly raised ValueError: %s", str(e))
            assert "Version field must be provided" in str(e)

        logger.info("✅ Creating without version correctly failed")

    except AssertionError:
        raise
    except Exception as e:
        logger.error("❌ Test creating without version failed: %s", e)
        raise

    logger.info("✅ Test creating without version completed")


async def test_batch_get_by_user_groups():
    """Test batch retrieval of group user profiles functionality"""
    logger.info("Starting test of batch retrieval of group user profiles functionality...")

    repo = get_bean_by_type(GroupUserProfileMemoryRawRepository)

    # Prepare test data
    test_data = [
        ("batch_user_001", "batch_group_001", "Zhao Liu"),
        ("batch_user_002", "batch_group_001", "Qian Qi"),
        ("batch_user_003", "batch_group_002", "Sun Ba"),
        ("batch_user_004", "batch_group_002", "Li Jiu"),
        ("batch_user_005", "batch_group_003", "Zhou Shi"),
    ]

    try:
        # First clean up any existing test data
        for user_id, group_id, _ in test_data:
            await repo.delete_by_user_group(user_id, group_id)
        logger.info("✅ Cleaned up existing test data")

        # Create test data, create multiple versions for each user
        for user_id, group_id, user_name in test_data:
            # Create old version
            old_data = {
                "version": "v1",
                "user_name": f"{user_name}_v1",
                "personality": [{"value": "Old personality", "evidences": ["conv_old"]}],
            }
            await repo.upsert_by_user_group(user_id, group_id, old_data)

            # Create latest version
            new_data = {
                "version": "v2",
                "user_name": f"{user_name}_v2",
                "personality": [{"value": "New personality", "evidences": ["conv_new"]}],
                "group_importance_evidence": {
                    "evidence_list": [
                        {"speak_count": 10, "refer_count": 5, "conversation_count": 20}
                    ]
                },
            }
            await repo.upsert_by_user_group(user_id, group_id, new_data)

        logger.info("✅ Created test data for 5 users (2 versions each)")

        # Test 1: Batch retrieve all user profiles (should return latest versions)
        user_group_pairs = [
            ("batch_user_001", "batch_group_001"),
            ("batch_user_002", "batch_group_001"),
            ("batch_user_003", "batch_group_002"),
            ("batch_user_004", "batch_group_002"),
            ("batch_user_005", "batch_group_003"),
        ]

        results = await repo.batch_get_by_user_groups(user_group_pairs)

        assert len(results) == 5, f"Should return 5 results, actually returned {len(results)}"
        logger.info("✅ Batch retrieval returned 5 results")

        # Verify each result is the latest version
        for (user_id, group_id), profile in results.items():
            assert (
                profile is not None
            ), f"Profile for user {user_id} in group {group_id} should not be None"
            assert (
                profile.version == "v2"
            ), f"Should return latest version v2, actually returned {profile.version}"
            assert profile.user_id == user_id
            assert profile.group_id == group_id
            assert profile.user_name.endswith("_v2"), "Should return username from latest version"
            logger.info(
                "✅ Verified user_id=%s, group_id=%s: version=%s, user_name=%s",
                user_id,
                group_id,
                profile.version,
                profile.user_name,
            )

        # Test 2: Include non-existent user-group pairs
        pairs_with_nonexist = user_group_pairs + [
            ("nonexist_user", "nonexist_group"),
            ("batch_user_001", "nonexist_group"),
        ]

        results_with_none = await repo.batch_get_by_user_groups(pairs_with_nonexist)
        assert len(results_with_none) == 7, "Should return 7 results (including non-existent ones)"
        assert results_with_none[("nonexist_user", "nonexist_group")] is None
        assert results_with_none[("batch_user_001", "nonexist_group")] is None
        logger.info("✅ Correctly handled non-existent user-group pairs, returned None")

        # Test 3: Test deduplication functionality
        duplicate_pairs = user_group_pairs + user_group_pairs[:2]  # Duplicate first two
        results_dedup = await repo.batch_get_by_user_groups(duplicate_pairs)
        assert len(results_dedup) == 5, "After deduplication should still be 5 results"
        logger.info("✅ Deduplication functionality works correctly")

        # Test 4: Empty list
        empty_results = await repo.batch_get_by_user_groups([])
        assert len(empty_results) == 0, "Empty list should return empty dictionary"
        logger.info("✅ Empty list returns empty dictionary")

        # Test 5: Verify group_importance_evidence field
        user_001_profile = results[("batch_user_001", "batch_group_001")]
        assert hasattr(user_001_profile, "group_importance_evidence")
        assert user_001_profile.group_importance_evidence is not None
        assert "evidence_list" in user_001_profile.group_importance_evidence
        logger.info("✅ group_importance_evidence field correctly retrieved")

        # Clean up test data
        for user_id, group_id, _ in test_data:
            await repo.delete_by_user_group(user_id, group_id)
        logger.info("✅ Successfully cleaned up test data")

    except Exception as e:
        logger.error("❌ Batch retrieval of group user profiles functionality test failed: %s", e)
        # Ensure cleanup
        for user_id, group_id, _ in test_data:
            try:
                await repo.delete_by_user_group(user_id, group_id)
            except Exception:
                pass
        raise

    logger.info("✅ Batch retrieval of group user profiles functionality test completed")


async def run_all_tests():
    """Run all tests"""
    logger.info("🚀 Starting to run all GroupUserProfileMemory tests...")

    try:
        await test_basic_crud_operations()
        await test_version_management()
        await test_ensure_latest()
        await test_batch_query_with_only_latest()
        await test_get_profile_method()
        await test_create_without_version_should_fail()
        await test_batch_get_by_user_groups()
        logger.info("✅ All tests completed")
    except Exception as e:
        logger.error("❌ Error occurred during testing: %s", e)
        raise


if __name__ == "__main__":
    asyncio.run(run_all_tests())
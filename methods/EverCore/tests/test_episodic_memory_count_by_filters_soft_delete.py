"""
Unit tests for count_by_filters / find_by_filters soft delete filtering

Verifies that:
1. count_by_filters and find_by_filters build correct MongoDB filter_dict
2. self.model.find() delegates to DocumentBaseWithSoftDelete.find_many(),
   which automatically appends {"deleted_at": None} for soft delete filtering
3. Various filter combinations (user_id, group_ids, time range) work correctly
4. Error handling returns safe defaults (0 for count, [] for find)

These are pure unit tests using mock — no real database required.
"""

import pytest
import sys
import os
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from infra_layer.adapters.out.persistence.repository.episodic_memory_raw_repository import (
    EpisodicMemoryRawRepository,
)
from infra_layer.adapters.out.persistence.document.memory.episodic_memory import (
    EpisodicMemory,
)
from core.oxm.constants import MAGIC_ALL


def _make_repo() -> EpisodicMemoryRawRepository:
    """Create a repository instance with mocked __init__ (skips vectorize_service)."""
    with patch.object(EpisodicMemoryRawRepository, '__init__', lambda self: None):
        repo = EpisodicMemoryRawRepository()
        repo.model = EpisodicMemory
        return repo


def _setup_find_mock(repo, return_count=0, return_list=None):
    """
    Replace repo.model.find() with a mock that returns a chainable query object.

    Supports the full chain: .find() -> .sort() -> .skip() -> .limit() -> .count() / .to_list()

    Returns:
        (mock_find, mock_query): mock_find to inspect call args,
                                 mock_query to configure return values.
    """
    mock_query = MagicMock()
    mock_query.sort.return_value = mock_query
    mock_query.skip.return_value = mock_query
    mock_query.limit.return_value = mock_query
    mock_query.count = AsyncMock(return_value=return_count)
    mock_query.to_list = AsyncMock(return_value=return_list or [])

    mock_find = MagicMock(return_value=mock_query)
    repo.model = MagicMock()
    repo.model.find = mock_find

    return mock_find, mock_query


# =============================================================================
# Test count_by_filters
# =============================================================================


class TestCountByFiltersSoftDelete:
    """Verify count_by_filters builds correct filter_dict and delegates to model.find."""

    @pytest.mark.asyncio
    async def test_count_with_user_id_filter(self):
        """When user_id is provided, filter_dict should contain {"user_id": "user_1"}."""
        repo = _make_repo()
        mock_find, _ = _setup_find_mock(repo, return_count=5)

        result = await repo.count_by_filters(user_id="user_1")

        assert result == 5
        mock_find.assert_called_once()
        filter_dict = mock_find.call_args[0][0]
        assert filter_dict == {"user_id": "user_1"}

    @pytest.mark.asyncio
    async def test_count_with_magic_all_passes_empty_filter(self):
        """When user_id is MAGIC_ALL ("__all__"), filter_dict should be empty (no user filter)."""
        repo = _make_repo()
        mock_find, _ = _setup_find_mock(repo, return_count=10)

        result = await repo.count_by_filters(user_id=MAGIC_ALL)

        assert result == 10
        filter_dict = mock_find.call_args[0][0]
        assert filter_dict == {}

    @pytest.mark.asyncio
    async def test_count_with_multiple_group_ids_uses_in_operator(self):
        """When group_ids has multiple elements, filter should use $in operator."""
        repo = _make_repo()
        mock_find, _ = _setup_find_mock(repo, return_count=3)

        result = await repo.count_by_filters(user_id="user_1", group_ids=["g1", "g2"])

        assert result == 3
        filter_dict = mock_find.call_args[0][0]
        assert filter_dict["user_id"] == "user_1"
        assert filter_dict["group_id"] == {"$in": ["g1", "g2"]}

    @pytest.mark.asyncio
    async def test_count_with_single_group_id_uses_exact_match(self):
        """When group_ids has exactly one element, filter should use exact match (not $in)."""
        repo = _make_repo()
        mock_find, _ = _setup_find_mock(repo, return_count=2)

        await repo.count_by_filters(user_id="user_1", group_ids=["g1"])

        filter_dict = mock_find.call_args[0][0]
        assert filter_dict["group_id"] == "g1"

    @pytest.mark.asyncio
    async def test_count_with_time_range_filter(self):
        """When start_time and end_time are provided, filter should contain timestamp range."""
        repo = _make_repo()
        mock_find, _ = _setup_find_mock(repo, return_count=7)

        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 2, 1, tzinfo=timezone.utc)

        result = await repo.count_by_filters(
            user_id="user_1", start_time=start, end_time=end
        )

        assert result == 7
        filter_dict = mock_find.call_args[0][0]
        assert filter_dict["timestamp"] == {"$gte": start, "$lt": end}

    @pytest.mark.asyncio
    async def test_count_with_empty_string_user_id_filters_null_or_empty(self):
        """When user_id is empty string, filter should match both None and "" values."""
        repo = _make_repo()
        mock_find, _ = _setup_find_mock(repo, return_count=1)

        await repo.count_by_filters(user_id="")

        filter_dict = mock_find.call_args[0][0]
        assert filter_dict["user_id"] == {"$in": [None, ""]}

    @pytest.mark.asyncio
    async def test_count_with_none_user_id_filters_null_or_empty(self):
        """When user_id is None, filter should match both None and "" values."""
        repo = _make_repo()
        mock_find, _ = _setup_find_mock(repo, return_count=1)

        await repo.count_by_filters(user_id=None)

        filter_dict = mock_find.call_args[0][0]
        assert filter_dict["user_id"] == {"$in": [None, ""]}

    @pytest.mark.asyncio
    async def test_count_returns_zero_on_database_exception(self):
        """When database raises an exception, count_by_filters should return 0 (not crash)."""
        repo = _make_repo()
        _, mock_query = _setup_find_mock(repo)
        mock_query.count = AsyncMock(side_effect=Exception("DB connection error"))

        result = await repo.count_by_filters(user_id="user_1")

        assert result == 0


# =============================================================================
# Test find_by_filters
# =============================================================================


class TestFindByFiltersSoftDelete:
    """Verify find_by_filters builds correct filter_dict, sorting, and pagination."""

    @pytest.mark.asyncio
    async def test_find_with_user_id_filter(self):
        """When user_id is provided, filter_dict should contain {"user_id": "user_1"}."""
        repo = _make_repo()
        mock_find, _ = _setup_find_mock(repo, return_list=["mem1", "mem2"])

        result = await repo.find_by_filters(user_id="user_1")

        assert len(result) == 2
        filter_dict = mock_find.call_args[0][0]
        assert filter_dict == {"user_id": "user_1"}

    @pytest.mark.asyncio
    async def test_find_with_multiple_group_ids_uses_in_operator(self):
        """When group_ids has multiple elements, filter should use $in operator."""
        repo = _make_repo()
        mock_find, _ = _setup_find_mock(repo, return_list=[])

        await repo.find_by_filters(user_id="user_1", group_ids=["g1", "g2"])

        filter_dict = mock_find.call_args[0][0]
        assert filter_dict["group_id"] == {"$in": ["g1", "g2"]}

    @pytest.mark.asyncio
    async def test_find_sort_descending_by_timestamp(self):
        """When sort_desc=True, query should sort by "-timestamp" (newest first)."""
        repo = _make_repo()
        _, mock_query = _setup_find_mock(repo, return_list=[])

        await repo.find_by_filters(user_id="user_1", sort_desc=True)

        mock_query.sort.assert_called_once_with("-timestamp")

    @pytest.mark.asyncio
    async def test_find_sort_ascending_by_timestamp(self):
        """When sort_desc=False, query should sort by "timestamp" (oldest first)."""
        repo = _make_repo()
        _, mock_query = _setup_find_mock(repo, return_list=[])

        await repo.find_by_filters(user_id="user_1", sort_desc=False)

        mock_query.sort.assert_called_once_with("timestamp")

    @pytest.mark.asyncio
    async def test_find_applies_skip_and_limit_for_pagination(self):
        """When skip and limit are provided, query should call .skip() and .limit()."""
        repo = _make_repo()
        _, mock_query = _setup_find_mock(repo, return_list=[])

        await repo.find_by_filters(user_id="user_1", skip=10, limit=5)

        mock_query.skip.assert_called_once_with(10)
        mock_query.limit.assert_called_once_with(5)

    @pytest.mark.asyncio
    async def test_find_returns_empty_list_on_database_exception(self):
        """When database raises an exception, find_by_filters should return [] (not crash)."""
        repo = _make_repo()
        _, mock_query = _setup_find_mock(repo)
        mock_query.to_list = AsyncMock(side_effect=Exception("DB connection error"))

        result = await repo.find_by_filters(user_id="user_1")

        assert result == []


# =============================================================================
# Test ORM-level soft delete mechanism
# =============================================================================


class TestSoftDeleteFilterMechanism:
    """
    Verify that DocumentBaseWithSoftDelete provides automatic soft delete filtering.

    These tests validate the ORM layer guarantees that make count_by_filters
    and find_by_filters safe from returning soft-deleted records.
    """

    def test_episodic_memory_inherits_soft_delete_and_find_many_injects_filter(self):
        """
        Verify the full inheritance chain:
        1. EpisodicMemory inherits from DocumentBaseWithSoftDelete
        2. DocumentBaseWithSoftDelete overrides find_many()
        3. The overridden find_many() injects {"deleted_at": None} into query args

        This is the core mechanism that ensures all queries via model.find()
        automatically exclude soft-deleted records.
        """
        from core.oxm.mongo.document_base_with_soft_delete import (
            DocumentBaseWithSoftDelete,
        )
        import inspect

        # 1. EpisodicMemory inherits from DocumentBaseWithSoftDelete
        assert issubclass(EpisodicMemory, DocumentBaseWithSoftDelete)

        # 2. find_many is overridden in DocumentBaseWithSoftDelete
        assert 'find_many' in DocumentBaseWithSoftDelete.__dict__

        # 3. The overridden find_many injects the soft delete filter
        source = inspect.getsource(DocumentBaseWithSoftDelete.find_many)
        assert (
            '{"deleted_at": None}' in source
        ), "find_many should inject deleted_at=None filter"

    def test_find_delegates_to_find_many_which_is_overridden(self):
        """
        Verify the call chain: model.find() -> FindInterface.find() -> cls.find_many()

        Since DocumentBaseWithSoftDelete overrides find_many(), calling model.find()
        in the repository will go through the soft-delete-aware version.

        This is why self.model.find() in count_by_filters / find_by_filters
        automatically filters out soft-deleted records.
        """
        import inspect

        # Beanie's FindInterface.find delegates to cls.find_many()
        for cls in EpisodicMemory.__mro__:
            if cls.__name__ == 'FindInterface' and 'find' in cls.__dict__:
                source = inspect.getsource(cls.__dict__['find'])
                assert (
                    'cls.find_many(' in source
                ), "FindInterface.find should delegate to cls.find_many"
                break
        else:
            pytest.fail("FindInterface.find not found in MRO")

        # find_many is first defined (overridden) by DocumentBaseWithSoftDelete in the MRO
        for cls in EpisodicMemory.__mro__:
            if 'find_many' in cls.__dict__:
                assert cls.__name__ == 'DocumentBaseWithSoftDelete', (
                    f"find_many should be first overridden in DocumentBaseWithSoftDelete, "
                    f"but found in {cls.__name__}"
                )
                break

    def test_hard_find_many_does_not_inject_soft_delete_filter(self):
        """
        Verify that hard_find_many() does NOT add {"deleted_at": None},
        confirming it can query soft-deleted records (used for restore, audit, etc.).
        """
        from core.oxm.mongo.document_base_with_soft_delete import (
            DocumentBaseWithSoftDelete,
        )
        import inspect

        source = inspect.getsource(DocumentBaseWithSoftDelete.hard_find_many)
        assert (
            '{"deleted_at": None}' not in source
        ), "hard_find_many should NOT inject deleted_at=None filter"

    def test_apply_soft_delete_filter_utility_method(self):
        """
        Verify the apply_soft_delete_filter() utility method behavior:
        - Default: appends {"deleted_at": None} to filter
        - include_deleted=True: does not modify filter
        - Empty input: returns {"deleted_at": None}
        - Existing deleted_at key: does not overwrite
        """
        # Default: should add deleted_at=None
        result = EpisodicMemory.apply_soft_delete_filter({"user_id": "u1"})
        assert result == {"user_id": "u1", "deleted_at": None}

        # include_deleted=True: should NOT add deleted_at
        result = EpisodicMemory.apply_soft_delete_filter(
            {"user_id": "u1"}, include_deleted=True
        )
        assert result == {"user_id": "u1"}

        # Empty input: should return only the soft delete filter
        result = EpisodicMemory.apply_soft_delete_filter()
        assert result == {"deleted_at": None}

        # Existing deleted_at in filter: should NOT be overwritten
        result = EpisodicMemory.apply_soft_delete_filter({"deleted_at": {"$ne": None}})
        assert result == {"deleted_at": {"$ne": None}}

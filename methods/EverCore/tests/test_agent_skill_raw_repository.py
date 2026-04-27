"""
AgentSkillRawRepository Unit Tests

Tests for:
- get_by_cluster_id: group_id scoping and min_confidence filtering
- _build_filter_query: query construction with various parameter combinations

Usage:
    PYTHONPATH=src pytest tests/test_agent_skill_raw_repository.py -v
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from types import SimpleNamespace

from infra_layer.adapters.out.persistence.repository.agent_skill_raw_repository import (
    AgentSkillRawRepository,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_skill_record(**kwargs):
    """Create a mock AgentSkillRecord-like object."""
    defaults = {
        "id": "skill_001",
        "cluster_id": "cluster_A",
        "group_id": "group_1",
        "name": "API Development",
        "confidence": 0.8,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _build_repo():
    """Build an AgentSkillRawRepository with mocked model."""
    with patch.object(AgentSkillRawRepository, "__init__", lambda self: None):
        repo = AgentSkillRawRepository()
        repo.model = MagicMock()
    return repo


# ---------------------------------------------------------------------------
# get_by_cluster_id
# ---------------------------------------------------------------------------

class TestGetByClusterId:
    """Tests for get_by_cluster_id group_id scoping."""

    @pytest.mark.asyncio
    async def test_query_includes_group_id_when_provided(self):
        repo = _build_repo()
        mock_to_list = AsyncMock(return_value=[])
        repo.model.find.return_value.to_list = mock_to_list

        await repo.get_by_cluster_id("cluster_A", group_id="group_1")

        called_query = repo.model.find.call_args[0][0]
        assert called_query["cluster_id"] == "cluster_A"
        assert called_query["group_id"] == "group_1"

    @pytest.mark.asyncio
    async def test_query_excludes_group_id_when_none(self):
        repo = _build_repo()
        mock_to_list = AsyncMock(return_value=[])
        repo.model.find.return_value.to_list = mock_to_list

        await repo.get_by_cluster_id("cluster_A")

        called_query = repo.model.find.call_args[0][0]
        assert called_query == {"cluster_id": "cluster_A"}
        assert "group_id" not in called_query

    @pytest.mark.asyncio
    async def test_query_includes_min_confidence(self):
        repo = _build_repo()
        mock_to_list = AsyncMock(return_value=[])
        repo.model.find.return_value.to_list = mock_to_list

        await repo.get_by_cluster_id(
            "cluster_A", group_id="group_1", min_confidence=0.5
        )

        called_query = repo.model.find.call_args[0][0]
        assert called_query["cluster_id"] == "cluster_A"
        assert called_query["group_id"] == "group_1"
        assert called_query["confidence"] == {"$gte": 0.5}

    @pytest.mark.asyncio
    async def test_query_with_only_min_confidence(self):
        repo = _build_repo()
        mock_to_list = AsyncMock(return_value=[])
        repo.model.find.return_value.to_list = mock_to_list

        await repo.get_by_cluster_id("cluster_A", min_confidence=0.3)

        called_query = repo.model.find.call_args[0][0]
        assert called_query == {
            "cluster_id": "cluster_A",
            "confidence": {"$gte": 0.3},
        }

    @pytest.mark.asyncio
    async def test_returns_empty_list_on_exception(self):
        repo = _build_repo()
        repo.model.find.side_effect = Exception("DB error")

        result = await repo.get_by_cluster_id("cluster_A", group_id="group_1")

        assert result == []


# ---------------------------------------------------------------------------
# _build_filter_query
# ---------------------------------------------------------------------------

class TestBuildFilterQuery:
    """Tests for _build_filter_query construction logic."""

    def test_empty_when_all_defaults(self):
        repo = _build_repo()
        query = repo._build_filter_query()
        assert query == {}

    def test_user_id_filter(self):
        repo = _build_repo()
        query = repo._build_filter_query(user_id="u1")
        assert query == {"user_id": "u1"}

    def test_magic_all_user_id_excluded(self):
        repo = _build_repo()
        query = repo._build_filter_query(user_id="__all__")
        assert "user_id" not in query

    def test_single_group_id(self):
        repo = _build_repo()
        query = repo._build_filter_query(group_ids=["g1"])
        assert query == {"group_id": "g1"}

    def test_multiple_group_ids(self):
        repo = _build_repo()
        query = repo._build_filter_query(group_ids=["g1", "g2"])
        assert query == {"group_id": {"$in": ["g1", "g2"]}}

    def test_cluster_id_filter(self):
        repo = _build_repo()
        query = repo._build_filter_query(cluster_id="c1")
        assert query == {"cluster_id": "c1"}

    def test_combined_filters(self):
        repo = _build_repo()
        query = repo._build_filter_query(
            user_id="u1", group_ids=["g1"], cluster_id="c1"
        )
        assert query == {
            "user_id": "u1",
            "group_id": "g1",
            "cluster_id": "c1",
        }

    def test_empty_group_ids_list_excluded(self):
        repo = _build_repo()
        query = repo._build_filter_query(group_ids=[])
        assert "group_id" not in query

    def test_none_user_id_excluded(self):
        repo = _build_repo()
        query = repo._build_filter_query(user_id=None)
        assert "user_id" not in query

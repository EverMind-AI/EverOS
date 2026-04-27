"""
GetMemoryService Unit Tests (mock)

Tests for:
- _agent_case_to_item: document-to-DTO conversion (GET variant, no score field)
- _agent_skill_to_item: document-to-DTO conversion (GET variant, no score field)
- _resolve_sort: sort field fallback for agent_skill (no timestamp)
- _get_agent_cases: pagination, sort, count, empty result
- _get_agent_skills: pagination, sort, count, empty result
- find_memories: dispatch, scope validation, unsupported type

Usage:
    PYTHONPATH=src pytest tests/test_agent_get_service.py -v
"""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any, Dict, List, Optional

from bson import ObjectId
from pymongo import ASCENDING, DESCENDING


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_case_doc(
    doc_id: str = None,
    user_id: str = "user_001",
    group_id: str = "group_001",
    session_id: str = "sess_001",
    task_intent: str = "Build REST API",
    approach: str = "1. Design\n2. Implement",
    quality_score: float = 0.8,
    key_insight: str = "Use caching for performance",
    timestamp: datetime = None,
    parent_type: str = "memcell",
    parent_id: str = "evt_001",
):
    """Create a mock AgentCaseRecord."""
    doc = MagicMock()
    doc.id = ObjectId(doc_id) if doc_id else ObjectId()
    doc.user_id = user_id
    doc.group_id = group_id
    doc.session_id = session_id
    doc.task_intent = task_intent
    doc.approach = approach
    doc.quality_score = quality_score
    doc.key_insight = key_insight
    doc.timestamp = timestamp or datetime(2025, 3, 1, 12, 0, 0)
    doc.parent_type = parent_type
    doc.parent_id = parent_id
    return doc


def _make_skill_doc(
    doc_id: str = None,
    user_id: str = "user_001",
    group_id: str = "group_001",
    cluster_id: str = "cluster_001",
    name: str = "API Development",
    description: str = "Build REST APIs",
    content: str = "## Steps\n1. Design\n2. Implement",
    confidence: float = 0.8,
    maturity_score: float = 0.75,
    source_case_ids: list = None,
):
    """Create a mock AgentSkillRecord."""
    doc = MagicMock()
    doc.id = ObjectId(doc_id) if doc_id else ObjectId()
    doc.user_id = user_id
    doc.group_id = group_id
    doc.cluster_id = cluster_id
    doc.name = name
    doc.description = description
    doc.content = content
    doc.confidence = confidence
    doc.maturity_score = maturity_score
    doc.source_case_ids = source_case_ids
    return doc


def _setup_repo_mock(mock_repo, docs, total_count):
    """Configure a repository mock so that find_by_query returns (docs, total_count)."""
    mock_repo.find_by_query = AsyncMock(return_value=(docs, total_count))
    return mock_repo


async def _coro(value):
    """Wrap a value in a coroutine for asyncio.gather compatibility."""
    return value


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def get_service():
    """Create GetMemoryService with repositories mocked."""
    mock_episodic_repo = AsyncMock()
    mock_profile_repo = AsyncMock()
    mock_case_repo = AsyncMock()
    mock_skill_repo = AsyncMock()

    with (
        patch(
            "agentic_layer.get_mem_service.get_bean_by_type",
            side_effect=lambda cls: {
                "EpisodicMemoryRawRepository": mock_episodic_repo,
                "UserProfileRawRepository": mock_profile_repo,
                "AgentCaseRawRepository": mock_case_repo,
                "AgentSkillRawRepository": mock_skill_repo,
            }.get(cls.__name__, AsyncMock()),
        ),
        patch("agentic_layer.get_mem_service.parse_mongo_filters") as mock_parse,
    ):
        from agentic_layer.get_mem_service import GetMemoryService

        svc = GetMemoryService()
        svc._mock_case_repo = mock_case_repo
        svc._mock_skill_repo = mock_skill_repo
        svc._mock_episodic_repo = mock_episodic_repo
        svc._mock_profile_repo = mock_profile_repo
        svc._mock_parse = mock_parse
        yield svc


# ===========================================================================
# _agent_case_to_item tests
# ===========================================================================


class TestAgentCaseToItem:
    """Tests for GetMemoryService._agent_case_to_item."""

    def test_basic_conversion(self, get_service):
        doc = _make_case_doc()
        item = get_service._agent_case_to_item(doc)
        assert item.id == str(doc.id)
        assert item.user_id == "user_001"
        assert item.group_id == "group_001"
        assert item.session_id == "sess_001"
        assert item.task_intent == "Build REST API"
        assert item.approach == "1. Design\n2. Implement"
        assert item.quality_score == 0.8
        assert item.parent_type == "memcell"
        assert item.parent_id == "evt_001"
        # GET DTO should NOT have score field
        assert not hasattr(item, "score") or "score" not in item.model_fields

    def test_none_task_intent_becomes_empty(self, get_service):
        doc = _make_case_doc(task_intent=None)
        item = get_service._agent_case_to_item(doc)
        assert item.task_intent == ""

    def test_none_approach_becomes_empty(self, get_service):
        doc = _make_case_doc(approach=None)
        item = get_service._agent_case_to_item(doc)
        assert item.approach == ""

    def test_none_quality_score(self, get_service):
        doc = _make_case_doc(quality_score=None)
        item = get_service._agent_case_to_item(doc)
        assert item.quality_score is None

    def test_none_optional_fields(self, get_service):
        doc = _make_case_doc(
            group_id=None, session_id=None, parent_type=None, parent_id=None
        )
        item = get_service._agent_case_to_item(doc)
        assert item.group_id is None
        assert item.session_id is None
        assert item.parent_type is None
        assert item.parent_id is None


# ===========================================================================
# _agent_skill_to_item tests
# ===========================================================================


class TestAgentSkillToItem:
    """Tests for GetMemoryService._agent_skill_to_item."""

    def test_basic_conversion(self, get_service):
        doc = _make_skill_doc(source_case_ids=["evt_001", "evt_002"])
        item = get_service._agent_skill_to_item(doc)
        assert item.id == str(doc.id)
        assert item.user_id == "user_001"
        assert item.group_id == "group_001"
        assert item.cluster_id == "cluster_001"
        assert item.name == "API Development"
        assert item.description == "Build REST APIs"
        assert item.content == "## Steps\n1. Design\n2. Implement"
        assert item.confidence == 0.8
        assert item.maturity_score == 0.75
        assert item.source_case_ids == ["evt_001", "evt_002"]
        # GET DTO should NOT have score field
        assert not hasattr(item, "score") or "score" not in item.model_fields

    def test_none_source_case_ids_becomes_empty_list(self, get_service):
        doc = _make_skill_doc(source_case_ids=None)
        item = get_service._agent_skill_to_item(doc)
        assert item.source_case_ids == []

    def test_none_optional_fields(self, get_service):
        doc = _make_skill_doc(
            group_id=None, name=None, description=None, source_case_ids=[]
        )
        item = get_service._agent_skill_to_item(doc)
        assert item.group_id is None
        assert item.name is None
        assert item.description is None


# ===========================================================================
# _resolve_sort tests
# ===========================================================================


class TestResolveSort:
    """Tests for GetMemoryService._resolve_sort."""

    def test_agent_case_default_timestamp(self, get_service):
        sort = get_service._resolve_sort("agent_case", "timestamp", "desc")
        assert sort == [("timestamp", DESCENDING)]

    def test_agent_case_asc(self, get_service):
        sort = get_service._resolve_sort("agent_case", "timestamp", "asc")
        assert sort == [("timestamp", ASCENDING)]

    def test_agent_skill_timestamp_fallback(self, get_service):
        """AgentSkillRecord has no timestamp field; must fallback to updated_at."""
        sort = get_service._resolve_sort("agent_skill", "timestamp", "desc")
        assert sort == [("updated_at", DESCENDING)]

    def test_agent_skill_timestamp_fallback_asc(self, get_service):
        sort = get_service._resolve_sort("agent_skill", "timestamp", "asc")
        assert sort == [("updated_at", ASCENDING)]

    def test_agent_skill_explicit_field_no_fallback(self, get_service):
        """When rank_by is not 'timestamp', no fallback should happen."""
        sort = get_service._resolve_sort("agent_skill", "confidence", "desc")
        assert sort == [("confidence", DESCENDING)]

    def test_profile_timestamp_fallback(self, get_service):
        sort = get_service._resolve_sort("profile", "timestamp", "desc")
        assert sort == [("updated_at", DESCENDING)]

    def test_episodic_memory_uses_timestamp(self, get_service):
        sort = get_service._resolve_sort("episodic_memory", "timestamp", "desc")
        assert sort == [("timestamp", DESCENDING)]


# ===========================================================================
# _get_episodes tests
# ===========================================================================


def _make_episode_doc(
    user_id="user_001",
    group_id="group_001",
    session_id="sess_001",
    summary="Team discussed roadmap",
    subject="Roadmap",
    episode="Alice and Bob discussed...",
    episode_type="Conversation",
):
    """Create a mock EpisodicMemoryProjection document."""
    doc = MagicMock()
    doc.id = ObjectId()
    doc.user_id = user_id
    doc.group_id = group_id
    doc.session_id = session_id
    doc.timestamp = datetime(2026, 3, 1, 10, 0)
    doc.participants = ["user_001", "user_002"]
    doc.sender_ids = ["user_001", "user_002"]
    doc.summary = summary
    doc.subject = subject
    doc.episode = episode
    doc.type = episode_type
    doc.parent_type = "memcell"
    doc.parent_id = "mc_001"
    return doc


class TestGetEpisodes:
    """Tests for GetMemoryService._get_episodes via repository."""

    @pytest.mark.asyncio
    async def test_returns_episodes_with_count(self, get_service):
        docs = [_make_episode_doc(), _make_episode_doc()]
        _setup_repo_mock(get_service._mock_episodic_repo, docs, 5)

        result = await get_service._get_episodes(
            mongo_filter={"user_id": "user_001"},
            skip=0,
            limit=20,
            sort=[("timestamp", DESCENDING)],
        )

        assert result.count == 2
        assert result.total_count == 5
        assert len(result.episodes) == 2
        assert result.episodes[0].user_id == "user_001"
        assert result.agent_cases == []
        assert result.agent_skills == []
        assert result.profiles == []

    @pytest.mark.asyncio
    async def test_empty_result(self, get_service):
        _setup_repo_mock(get_service._mock_episodic_repo, [], 0)

        result = await get_service._get_episodes(
            mongo_filter={"user_id": "nobody"},
            skip=0,
            limit=20,
            sort=[("timestamp", DESCENDING)],
        )

        assert result.count == 0
        assert result.total_count == 0
        assert result.episodes == []

    @pytest.mark.asyncio
    async def test_projection_passed(self, get_service):
        _setup_repo_mock(get_service._mock_episodic_repo, [], 0)

        await get_service._get_episodes(
            mongo_filter={"user_id": "u1"},
            skip=0,
            limit=20,
            sort=[("timestamp", DESCENDING)],
        )

        call_args = get_service._mock_episodic_repo.find_by_query.call_args
        assert call_args.kwargs.get("projection_model") is not None

    @pytest.mark.asyncio
    async def test_episode_fields_mapped(self, get_service):
        doc = _make_episode_doc()
        _setup_repo_mock(get_service._mock_episodic_repo, [doc], 1)

        result = await get_service._get_episodes(
            mongo_filter={"user_id": "user_001"},
            skip=0,
            limit=20,
            sort=[("timestamp", DESCENDING)],
        )

        ep = result.episodes[0]
        assert ep.summary == "Team discussed roadmap"
        assert ep.subject == "Roadmap"
        assert ep.episode == "Alice and Bob discussed..."
        assert ep.type == "Conversation"
        assert ep.parent_type == "memcell"


# ===========================================================================
# _get_profiles tests
# ===========================================================================


def _make_profile_doc(
    user_id="user_001", group_id="group_001", scenario="solo", memcell_count=5
):
    """Create a mock UserProfile document."""
    doc = MagicMock()
    doc.id = ObjectId()
    doc.user_id = user_id
    doc.group_id = group_id
    doc.profile_data = {"explicit_info": {"Role": "Engineer"}}
    doc.scenario = scenario
    doc.memcell_count = memcell_count
    return doc


class TestGetProfiles:
    """Tests for GetMemoryService._get_profiles via repository."""

    @pytest.mark.asyncio
    async def test_returns_profiles_with_count(self, get_service):
        docs = [_make_profile_doc(), _make_profile_doc()]
        _setup_repo_mock(get_service._mock_profile_repo, docs, 3)

        result = await get_service._get_profiles(
            mongo_filter={"user_id": "user_001"},
            skip=0,
            limit=20,
            sort=[("updated_at", DESCENDING)],
        )

        assert result.count == 2
        assert result.total_count == 3
        assert len(result.profiles) == 2
        assert result.profiles[0].user_id == "user_001"
        assert result.episodes == []
        assert result.agent_cases == []
        assert result.agent_skills == []

    @pytest.mark.asyncio
    async def test_empty_result(self, get_service):
        _setup_repo_mock(get_service._mock_profile_repo, [], 0)

        result = await get_service._get_profiles(
            mongo_filter={"user_id": "nobody"},
            skip=0,
            limit=20,
            sort=[("updated_at", DESCENDING)],
        )

        assert result.count == 0
        assert result.profiles == []

    @pytest.mark.asyncio
    async def test_no_projection(self, get_service):
        """Profile has no vector in MongoDB, so no projection needed."""
        _setup_repo_mock(get_service._mock_profile_repo, [], 0)

        await get_service._get_profiles(
            mongo_filter={"user_id": "u1"},
            skip=0,
            limit=20,
            sort=[("updated_at", DESCENDING)],
        )

        call_args = get_service._mock_profile_repo.find_by_query.call_args
        assert call_args.kwargs.get("projection_model") is None

    @pytest.mark.asyncio
    async def test_profile_fields_mapped(self, get_service):
        doc = _make_profile_doc()
        _setup_repo_mock(get_service._mock_profile_repo, [doc], 1)

        result = await get_service._get_profiles(
            mongo_filter={"user_id": "user_001"},
            skip=0,
            limit=20,
            sort=[("updated_at", DESCENDING)],
        )

        pf = result.profiles[0]
        assert pf.user_id == "user_001"
        assert pf.profile_data == {"explicit_info": {"Role": "Engineer"}}
        assert pf.scenario == "solo"
        assert pf.memcell_count == 5


# ===========================================================================
# _get_agent_cases tests
# ===========================================================================


class TestGetAgentCases:
    """Tests for GetMemoryService._get_agent_cases."""

    @pytest.mark.asyncio
    async def test_returns_cases_with_count(self, get_service):
        docs = [_make_case_doc(), _make_case_doc()]
        _setup_repo_mock(get_service._mock_case_repo, docs, 5)

        result = await get_service._get_agent_cases(
            mongo_filter={"user_id": "user_001"},
            skip=0,
            limit=20,
            sort=[("timestamp", DESCENDING)],
        )

        assert result.count == 2
        assert result.total_count == 5
        assert len(result.agent_cases) == 2
        assert result.agent_cases[0].user_id == "user_001"
        # Other lists should be empty
        assert result.episodes == []
        assert result.profiles == []
        assert result.agent_skills == []

    @pytest.mark.asyncio
    async def test_empty_result(self, get_service):
        _setup_repo_mock(get_service._mock_case_repo, [], 0)

        result = await get_service._get_agent_cases(
            mongo_filter={"user_id": "user_999"},
            skip=0,
            limit=20,
            sort=[("timestamp", DESCENDING)],
        )

        assert result.count == 0
        assert result.total_count == 0
        assert result.agent_cases == []

    @pytest.mark.asyncio
    async def test_pagination_params_passed(self, get_service):
        _setup_repo_mock(get_service._mock_case_repo, [], 0)

        await get_service._get_agent_cases(
            mongo_filter={"user_id": "u1"},
            skip=40,
            limit=10,
            sort=[("timestamp", ASCENDING)],
        )

        first_call = get_service._mock_case_repo.find_by_query.call_args
        assert first_call.args[0] == {"user_id": "u1"}
        assert first_call.kwargs["skip"] == 40
        assert first_call.kwargs["limit"] == 10
        assert first_call.kwargs["sort"] == [("timestamp", ASCENDING)]

    @pytest.mark.asyncio
    async def test_projection_passed(self, get_service):
        """AgentCase query should use AgentCaseProjection to exclude vector."""
        _setup_repo_mock(get_service._mock_case_repo, [], 0)

        await get_service._get_agent_cases(
            mongo_filter={"user_id": "u1"},
            skip=0,
            limit=20,
            sort=[("timestamp", DESCENDING)],
        )

        call_args = get_service._mock_case_repo.find_by_query.call_args
        projection = call_args.kwargs.get("projection_model")
        assert projection is not None
        assert projection.__name__ == "AgentCaseProjection"


# ===========================================================================
# _get_agent_skills tests
# ===========================================================================


class TestGetAgentSkills:
    """Tests for GetMemoryService._get_agent_skills."""

    @pytest.mark.asyncio
    async def test_returns_skills_with_count(self, get_service):
        docs = [
            _make_skill_doc(source_case_ids=["e1"]),
            _make_skill_doc(source_case_ids=["e2"]),
        ]
        _setup_repo_mock(get_service._mock_skill_repo, docs, 10)

        result = await get_service._get_agent_skills(
            mongo_filter={"user_id": "user_001"},
            skip=0,
            limit=20,
            sort=[("updated_at", DESCENDING)],
        )

        assert result.count == 2
        assert result.total_count == 10
        assert len(result.agent_skills) == 2
        assert result.agent_skills[0].cluster_id == "cluster_001"
        # Other lists should be empty
        assert result.episodes == []
        assert result.profiles == []
        assert result.agent_cases == []

    @pytest.mark.asyncio
    async def test_empty_result(self, get_service):
        _setup_repo_mock(get_service._mock_skill_repo, [], 0)

        result = await get_service._get_agent_skills(
            mongo_filter={"user_id": "nobody"},
            skip=0,
            limit=20,
            sort=[("updated_at", DESCENDING)],
        )

        assert result.count == 0
        assert result.total_count == 0
        assert result.agent_skills == []

    @pytest.mark.asyncio
    async def test_projection_passed(self, get_service):
        """AgentSkill query should use AgentSkillProjection to exclude vector."""
        _setup_repo_mock(get_service._mock_skill_repo, [], 0)

        await get_service._get_agent_skills(
            mongo_filter={"user_id": "u1"},
            skip=0,
            limit=20,
            sort=[("updated_at", DESCENDING)],
        )

        call_args = get_service._mock_skill_repo.find_by_query.call_args
        projection = call_args.kwargs.get("projection_model")
        assert projection is not None
        assert projection.__name__ == "AgentSkillProjection"


# ===========================================================================
# find_memories dispatch tests
# ===========================================================================


class TestFindMemoriesDispatch:
    """Tests for GetMemoryService.find_memories dispatching to correct handler."""

    @pytest.mark.asyncio
    async def test_dispatches_agent_case(self, get_service):
        get_service._mock_parse.return_value = ({"user_id": "u1"}, "u1", None)
        _setup_repo_mock(get_service._mock_case_repo, [_make_case_doc()], 1)

        result = await get_service.find_memories(
            filters={"user_id": "u1"},
            memory_type="agent_case",
            page=1,
            page_size=20,
            rank_by="timestamp",
            rank_order="desc",
        )

        assert len(result.agent_cases) == 1
        assert result.agent_skills == []

    @pytest.mark.asyncio
    async def test_dispatches_agent_skill(self, get_service):
        get_service._mock_parse.return_value = ({"user_id": "u1"}, "u1", None)
        _setup_repo_mock(
            get_service._mock_skill_repo, [_make_skill_doc(source_case_ids=["e1"])], 1
        )

        result = await get_service.find_memories(
            filters={"user_id": "u1"},
            memory_type="agent_skill",
            page=1,
            page_size=20,
            rank_by="timestamp",
            rank_order="desc",
        )

        assert len(result.agent_skills) == 1
        assert result.agent_cases == []

    @pytest.mark.asyncio
    async def test_agent_skill_sort_uses_updated_at(self, get_service):
        """Verify that agent_skill query uses updated_at instead of timestamp."""
        get_service._mock_parse.return_value = ({"user_id": "u1"}, "u1", None)
        _setup_repo_mock(get_service._mock_skill_repo, [], 0)

        await get_service.find_memories(
            filters={"user_id": "u1"},
            memory_type="agent_skill",
            page=1,
            page_size=20,
            rank_by="timestamp",
            rank_order="desc",
        )

        first_call = get_service._mock_skill_repo.find_by_query.call_args
        assert first_call.kwargs["sort"] == [("updated_at", DESCENDING)]

    @pytest.mark.asyncio
    async def test_missing_scope_raises(self, get_service):
        get_service._mock_parse.return_value = ({}, None, None)

        from agentic_layer.get_mem_service import InvalidScopeError

        with pytest.raises(InvalidScopeError):
            await get_service.find_memories(
                filters={},
                memory_type="agent_case",
                page=1,
                page_size=20,
                rank_by="timestamp",
                rank_order="desc",
            )

    @pytest.mark.asyncio
    async def test_unsupported_type_raises(self, get_service):
        get_service._mock_parse.return_value = ({"user_id": "u1"}, "u1", None)

        with pytest.raises(ValueError, match="Unsupported memory_type"):
            await get_service.find_memories(
                filters={"user_id": "u1"},
                memory_type="unknown_type",
                page=1,
                page_size=20,
                rank_by="timestamp",
                rank_order="desc",
            )

    @pytest.mark.asyncio
    async def test_pagination_skip_calculation(self, get_service):
        """page=3, page_size=10 should produce skip=20."""
        get_service._mock_parse.return_value = ({"user_id": "u1"}, "u1", None)
        _setup_repo_mock(get_service._mock_case_repo, [], 0)

        await get_service.find_memories(
            filters={"user_id": "u1"},
            memory_type="agent_case",
            page=3,
            page_size=10,
            rank_by="timestamp",
            rank_order="asc",
        )

        first_call = get_service._mock_case_repo.find_by_query.call_args
        assert first_call.kwargs["skip"] == 20
        assert first_call.kwargs["limit"] == 10
        assert first_call.kwargs["sort"] == [("timestamp", ASCENDING)]

    @pytest.mark.asyncio
    async def test_group_id_only_raises_for_agent_case(self, get_service):
        """agent_case with only group_id (no user_id) should raise."""
        from agentic_layer.get_mem_service import InvalidScopeError

        get_service._mock_parse.return_value = ({"group_id": "g1"}, None, ["g1"])

        with pytest.raises(InvalidScopeError, match="requires 'user_id'"):
            await get_service.find_memories(
                filters={"group_id": "g1"},
                memory_type="agent_case",
                page=1,
                page_size=20,
                rank_by="timestamp",
                rank_order="desc",
            )

    @pytest.mark.asyncio
    async def test_group_id_only_raises_for_profile(self, get_service):
        """profile with only group_id (no user_id) should raise."""
        from agentic_layer.get_mem_service import InvalidScopeError

        get_service._mock_parse.return_value = ({"group_id": "g1"}, None, ["g1"])

        with pytest.raises(InvalidScopeError, match="requires 'user_id'"):
            await get_service.find_memories(
                filters={"group_id": "g1"},
                memory_type="profile",
                page=1,
                page_size=20,
                rank_by="timestamp",
                rank_order="desc",
            )

    @pytest.mark.asyncio
    async def test_group_id_only_raises_for_agent_skill(self, get_service):
        """agent_skill with only group_id (no user_id) should raise."""
        from agentic_layer.get_mem_service import InvalidScopeError

        get_service._mock_parse.return_value = ({"group_id": "g1"}, None, ["g1"])

        with pytest.raises(InvalidScopeError, match="requires 'user_id'"):
            await get_service.find_memories(
                filters={"group_id": "g1"},
                memory_type="agent_skill",
                page=1,
                page_size=20,
                rank_by="timestamp",
                rank_order="desc",
            )

    @pytest.mark.asyncio
    async def test_episodic_group_only_adds_user_id_filter(self, get_service):
        """episodic with only group_id should add user_id not-exist condition."""
        mongo_filter = {"group_id": "g1"}
        get_service._mock_parse.return_value = (mongo_filter, None, ["g1"])
        _setup_repo_mock(get_service._mock_episodic_repo, [], 0)

        await get_service.find_memories(
            filters={"group_id": "g1"},
            memory_type="episodic_memory",
            page=1,
            page_size=20,
            rank_by="timestamp",
            rank_order="desc",
        )

        # Verify user_id not-exist filter was added
        assert mongo_filter["user_id"] == {"$in": [None, ""]}

    @pytest.mark.asyncio
    async def test_dispatches_episodic_memory(self, get_service):
        get_service._mock_parse.return_value = ({"user_id": "u1"}, "u1", None)
        _setup_repo_mock(get_service._mock_episodic_repo, [_make_episode_doc()], 1)

        result = await get_service.find_memories(
            filters={"user_id": "u1"},
            memory_type="episodic_memory",
            page=1,
            page_size=20,
            rank_by="timestamp",
            rank_order="desc",
        )

        assert len(result.episodes) == 1
        assert result.profiles == []
        assert result.agent_cases == []
        assert result.agent_skills == []

    @pytest.mark.asyncio
    async def test_dispatches_profile(self, get_service):
        get_service._mock_parse.return_value = ({"user_id": "u1"}, "u1", None)
        _setup_repo_mock(get_service._mock_profile_repo, [_make_profile_doc()], 1)

        result = await get_service.find_memories(
            filters={"user_id": "u1"},
            memory_type="profile",
            page=1,
            page_size=20,
            rank_by="timestamp",
            rank_order="desc",
        )

        assert len(result.profiles) == 1
        assert result.episodes == []
        assert result.agent_cases == []
        assert result.agent_skills == []

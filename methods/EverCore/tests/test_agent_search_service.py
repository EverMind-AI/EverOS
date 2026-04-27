"""
SearchMemoryService Agent Case/Skill Search Tests

Tests for:
- _search_agent_cases: keyword, vector, hybrid methods
- _search_agent_skills: keyword, vector, hybrid methods
- _agent_case_doc_to_item: document-to-DTO conversion
- _agent_skill_doc_to_item: document-to-DTO conversion
- _extract_hit_id: unified ID extraction from ES/Milvus hits
- search_memories: full search with agent_memory type
- _extract_filter_values: filter DSL parsing for agent filters
- Hybrid edge cases: dedup, backfill failure, empty query_words

Usage:
    PYTHONPATH=src pytest tests/test_agent_search_service.py -v
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from typing import Any, Dict, List, Optional
from bson import ObjectId
from pydantic import ValidationError

from api_specs.dtos.memory import (
    AgentMemorySearchResult,
    SearchAgentCaseItem,
    SearchAgentSkillItem,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_case_doc(
    doc_id: str = None,
    user_id: str = "user_001",
    group_id: str = "group_001",
    session_id: str = None,
    task_intent: str = "Build REST API",
    approach: str = "1. Design\n2. Implement",
    quality_score: float = 0.8,
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
    doc.timestamp = timestamp or datetime(2025, 3, 1, 12, 0, 0)
    doc.key_insight = ""
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
    doc.source_case_ids = []
    return doc


def _es_hit(doc_id: str, score: float = 1.0, **extra_source) -> Dict[str, Any]:
    """Create a mock ES hit."""
    source = {"id": doc_id, **extra_source}
    return {"_source": source, "_score": score, "_id": doc_id}


def _milvus_result(doc_id: str, score: float = 0.9, **extra) -> Dict[str, Any]:
    """Create a mock Milvus search result."""
    return {"id": doc_id, "score": score, **extra}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def search_service():
    """Create SearchMemoryService with all repositories mocked."""
    with (
        patch("agentic_layer.search_mem_service.EpisodicMemoryEsRepository"),
        patch("agentic_layer.search_mem_service.EpisodicMemoryMilvusRepository"),
        patch("agentic_layer.search_mem_service.UserProfileMilvusRepository"),
        patch(
            "agentic_layer.search_mem_service.AgentCaseEsRepository"
        ) as mock_case_es_cls,
        patch(
            "agentic_layer.search_mem_service.AgentSkillEsRepository"
        ) as mock_skill_es_cls,
        patch(
            "agentic_layer.search_mem_service.AgentCaseMilvusRepository"
        ) as mock_case_milvus_cls,
        patch(
            "agentic_layer.search_mem_service.AgentSkillMilvusRepository"
        ) as mock_skill_milvus_cls,
        patch("agentic_layer.search_mem_service.MemoryManager"),
        patch("agentic_layer.search_mem_service.RawMessageService"),
    ):

        from agentic_layer.search_mem_service import SearchMemoryService

        svc = SearchMemoryService()
        # Replace repo instances with AsyncMock
        svc.agent_case_es_repo = AsyncMock()
        svc.agent_skill_es_repo = AsyncMock()
        svc.agent_case_milvus_repo = AsyncMock()
        svc.agent_skill_milvus_repo = AsyncMock()
        yield svc


# ===========================================================================
# _agent_case_doc_to_item tests
# ===========================================================================


class TestAgentCaseDocToItem:
    """Tests for SearchMemoryService._agent_case_doc_to_item."""

    def test_basic_conversion(self, search_service):
        doc = _make_case_doc()
        item = search_service._agent_case_doc_to_item(doc, score=0.95)
        assert isinstance(item, SearchAgentCaseItem)
        assert item.id == str(doc.id)
        assert item.user_id == "user_001"
        assert item.task_intent == "Build REST API"
        assert item.score == 0.95

    def test_none_score(self, search_service):
        doc = _make_case_doc()
        item = search_service._agent_case_doc_to_item(doc)
        assert item.score is None

    def test_none_fields_handled(self, search_service):
        doc = _make_case_doc(task_intent=None, approach=None)
        item = search_service._agent_case_doc_to_item(doc)
        assert item.task_intent == ""
        assert item.approach == ""


# ===========================================================================
# _agent_skill_doc_to_item tests
# ===========================================================================


class TestAgentSkillDocToItem:
    """Tests for SearchMemoryService._agent_skill_doc_to_item."""

    def test_basic_conversion(self, search_service):
        doc = _make_skill_doc()
        item = search_service._agent_skill_doc_to_item(doc, score=0.88)
        assert isinstance(item, SearchAgentSkillItem)
        assert item.id == str(doc.id)
        assert item.name == "API Development"
        assert item.confidence == 0.8
        assert item.maturity_score == 0.75
        assert item.score == 0.88

    def test_none_score(self, search_service):
        doc = _make_skill_doc()
        item = search_service._agent_skill_doc_to_item(doc)
        assert item.score is None


# ===========================================================================
# _search_agent_cases tests
# ===========================================================================


class TestSearchAgentCases:
    """Tests for SearchMemoryService._search_agent_cases."""

    @pytest.mark.asyncio
    async def test_keyword_search(self, search_service):
        doc = _make_case_doc()
        doc_id = str(doc.id)
        search_service.agent_case_es_repo.multi_search = AsyncMock(
            return_value=[_es_hit(doc_id, score=5.0)]
        )

        with patch.object(
            search_service,
            "_fetch_agent_cases_by_ids",
            new_callable=AsyncMock,
            return_value={doc_id: doc},
        ):
            results = await search_service._search_agent_cases(
                query="REST API",
                query_words=["REST", "API"],
                query_vector=None,
                method="keyword",
                filter_values={
                    "user_id": "user_001",
                    "group_ids": None,
                    "session_id": None,
                    "start_time": None,
                    "end_time": None,
                },
                date_range={},
                top_k=10,
                radius=None,
            )
        assert len(results) == 1
        assert results[0].task_intent == "Build REST API"
        assert results[0].score == 5.0

    @pytest.mark.asyncio
    async def test_vector_search(self, search_service):
        doc = _make_case_doc()
        doc_id = str(doc.id)
        search_service.agent_case_milvus_repo.vector_search = AsyncMock(
            return_value=[_milvus_result(doc_id, score=0.92)]
        )

        with patch.object(
            search_service,
            "_fetch_agent_cases_by_ids",
            new_callable=AsyncMock,
            return_value={doc_id: doc},
        ):
            results = await search_service._search_agent_cases(
                query="",
                query_words=[],
                query_vector=[0.1, 0.2, 0.3],
                method="vector",
                filter_values={
                    "user_id": "user_001",
                    "group_ids": None,
                    "session_id": None,
                    "start_time": None,
                    "end_time": None,
                },
                date_range={},
                top_k=10,
                radius=None,
            )
        assert len(results) == 1
        assert results[0].score == 0.92

    @pytest.mark.asyncio
    async def test_vector_search_no_vector_returns_empty(self, search_service):
        """Vector search without query vector returns no results."""
        results = await search_service._search_agent_cases(
            query="",
            query_words=[],
            query_vector=None,
            method="vector",
            filter_values={
                "user_id": "u1",
                "group_ids": None,
                "session_id": None,
                "start_time": None,
                "end_time": None,
            },
            date_range={},
            top_k=10,
            radius=None,
        )
        assert results == []

    @pytest.mark.asyncio
    async def test_keyword_search_no_mongo_doc_skipped(self, search_service):
        """If ES returns an ID but MongoDB doesn't have the doc, it's skipped."""
        search_service.agent_case_es_repo.multi_search = AsyncMock(
            return_value=[_es_hit("nonexistent_id")]
        )
        with patch.object(
            search_service,
            "_fetch_agent_cases_by_ids",
            new_callable=AsyncMock,
            return_value={},
        ):
            results = await search_service._search_agent_cases(
                query="test",
                query_words=["test"],
                query_vector=None,
                method="keyword",
                filter_values={
                    "user_id": "u1",
                    "group_ids": None,
                    "session_id": None,
                    "start_time": None,
                    "end_time": None,
                },
                date_range={},
                top_k=10,
                radius=None,
            )
        assert results == []

    @pytest.mark.asyncio
    async def test_keyword_search_with_session_id(self, search_service):
        """Keyword search passes session_id to ES repo filter."""
        doc = _make_case_doc()
        doc_id = str(doc.id)
        search_service.agent_case_es_repo.multi_search = AsyncMock(
            return_value=[_es_hit(doc_id, score=3.0)]
        )

        with patch.object(
            search_service,
            "_fetch_agent_cases_by_ids",
            new_callable=AsyncMock,
            return_value={doc_id: doc},
        ):
            results = await search_service._search_agent_cases(
                query="REST API",
                query_words=["REST", "API"],
                query_vector=None,
                method="keyword",
                filter_values={
                    "user_id": "user_001",
                    "group_ids": None,
                    "session_id": "sess_abc",
                    "start_time": None,
                    "end_time": None,
                },
                date_range={},
                top_k=10,
                radius=None,
            )

        # Verify session_id was passed to ES repo
        call_kwargs = search_service.agent_case_es_repo.multi_search.call_args[1]
        assert call_kwargs["session_id"] == "sess_abc"
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_vector_search_with_session_id(self, search_service):
        """Vector search passes session_id to Milvus repo filter."""
        doc = _make_case_doc()
        doc_id = str(doc.id)
        search_service.agent_case_milvus_repo.vector_search = AsyncMock(
            return_value=[_milvus_result(doc_id, score=0.88)]
        )

        with patch.object(
            search_service,
            "_fetch_agent_cases_by_ids",
            new_callable=AsyncMock,
            return_value={doc_id: doc},
        ):
            results = await search_service._search_agent_cases(
                query="",
                query_words=[],
                query_vector=[0.1, 0.2, 0.3],
                method="vector",
                filter_values={
                    "user_id": "user_001",
                    "group_ids": None,
                    "session_id": "sess_abc",
                    "start_time": None,
                    "end_time": None,
                },
                date_range={},
                top_k=10,
                radius=None,
            )

        # Verify session_id was passed to Milvus repo
        call_kwargs = search_service.agent_case_milvus_repo.vector_search.call_args[1]
        assert call_kwargs["session_id"] == "sess_abc"
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_keyword_search_session_id_none_not_filtered(self, search_service):
        """When session_id is None, it should not be passed as a filter."""
        doc = _make_case_doc()
        doc_id = str(doc.id)
        search_service.agent_case_es_repo.multi_search = AsyncMock(
            return_value=[_es_hit(doc_id, score=3.0)]
        )

        with patch.object(
            search_service,
            "_fetch_agent_cases_by_ids",
            new_callable=AsyncMock,
            return_value={doc_id: doc},
        ):
            await search_service._search_agent_cases(
                query="test",
                query_words=["test"],
                query_vector=None,
                method="keyword",
                filter_values={
                    "user_id": "u1",
                    "group_ids": None,
                    "session_id": None,
                    "start_time": None,
                    "end_time": None,
                },
                date_range={},
                top_k=10,
                radius=None,
            )

        call_kwargs = search_service.agent_case_es_repo.multi_search.call_args[1]
        assert call_kwargs["session_id"] is None


# ===========================================================================
# _search_agent_skills tests
# ===========================================================================


class TestFetchAgentSkillsConfidenceFilter:
    """Tests for _fetch_agent_skills_by_ids confidence filtering.

    _fetch_agent_skills_by_ids adds a confidence >= retire_confidence filter
    to the MongoDB query, ensuring retired skills are excluded from results.
    """

    @pytest.mark.asyncio
    async def test_retired_skill_filtered_out(self, search_service):
        """Skills with confidence below threshold should be excluded."""
        high_conf_doc = _make_skill_doc(confidence=0.8)
        high_id = str(high_conf_doc.id)

        search_service.agent_skill_raw_repo.find_by_ids = AsyncMock(
            return_value=[high_conf_doc]
        )

        result = await search_service._fetch_agent_skills_by_ids([high_id])

        call_kwargs = search_service.agent_skill_raw_repo.find_by_ids.call_args
        assert call_kwargs[0][0] == [high_id]
        assert call_kwargs[1]["min_confidence"] == pytest.approx(0.1)
        assert high_id in result

    @pytest.mark.asyncio
    async def test_empty_ids_returns_empty(self, search_service):
        """Empty skill_ids should return empty dict without querying."""
        result = await search_service._fetch_agent_skills_by_ids([])
        assert result == {}


class TestSearchAgentSkillsConfidenceThreshold:
    """Tests that _search_agent_skills passes confidence_threshold to search repos."""

    @pytest.mark.asyncio
    async def test_keyword_passes_confidence_threshold(self, search_service):
        """Keyword search should pass confidence_threshold to ES repo."""
        search_service.agent_skill_es_repo.multi_search = AsyncMock(return_value=[])

        await search_service._search_agent_skills(
            query="test",
            query_words=["test"],
            query_vector=None,
            method="keyword",
            filter_values={"user_id": "u1", "group_ids": None, "session_id": None},
            top_k=10,
            radius=None,
        )

        call_kwargs = search_service.agent_skill_es_repo.multi_search.call_args
        assert call_kwargs.kwargs.get("confidence_threshold") == 0.1

    @pytest.mark.asyncio
    async def test_vector_passes_confidence_threshold(self, search_service):
        """Vector search should pass confidence_threshold to Milvus repo."""
        search_service.agent_skill_milvus_repo.vector_search = AsyncMock(
            return_value=[]
        )

        await search_service._search_agent_skills(
            query="",
            query_words=[],
            query_vector=[0.1, 0.2],
            method="vector",
            filter_values={"user_id": "u1", "group_ids": None, "session_id": None},
            top_k=10,
            radius=None,
        )

        call_kwargs = search_service.agent_skill_milvus_repo.vector_search.call_args
        assert call_kwargs.kwargs.get("confidence_threshold") == 0.1

    @pytest.mark.asyncio
    async def test_hybrid_passes_confidence_threshold_to_both(self, search_service):
        """Hybrid search should pass confidence_threshold to both ES and Milvus."""
        search_service.agent_skill_es_repo.multi_search = AsyncMock(return_value=[])
        search_service.agent_skill_milvus_repo.vector_search = AsyncMock(
            return_value=[]
        )

        with patch(
            "agentic_layer.search_mem_service.get_rerank_service"
        ) as mock_rerank:
            mock_rerank_svc = AsyncMock()
            mock_rerank_svc.rerank_memories = AsyncMock(return_value=[])
            mock_rerank.return_value = mock_rerank_svc

            await search_service._search_agent_skills(
                query="test",
                query_words=["test"],
                query_vector=[0.1, 0.2],
                method="hybrid",
                filter_values={"user_id": "u1", "group_ids": None, "session_id": None},
                top_k=10,
                radius=None,
            )

        es_kwargs = search_service.agent_skill_es_repo.multi_search.call_args
        assert es_kwargs.kwargs.get("confidence_threshold") == 0.1

        milvus_kwargs = search_service.agent_skill_milvus_repo.vector_search.call_args
        assert milvus_kwargs.kwargs.get("confidence_threshold") == 0.1


class TestSearchAgentSkills:
    """Tests for SearchMemoryService._search_agent_skills.

    Note: agent_skill has no date_range filtering (similar to profile).
    """

    @pytest.mark.asyncio
    async def test_keyword_search(self, search_service):
        doc = _make_skill_doc()
        doc_id = str(doc.id)
        search_service.agent_skill_es_repo.multi_search = AsyncMock(
            return_value=[_es_hit(doc_id, score=3.5)]
        )

        with patch.object(
            search_service,
            "_fetch_agent_skills_by_ids",
            new_callable=AsyncMock,
            return_value={doc_id: doc},
        ):
            results = await search_service._search_agent_skills(
                query="API development",
                query_words=["API", "development"],
                query_vector=None,
                method="keyword",
                filter_values={
                    "user_id": "user_001",
                    "group_ids": None,
                    "session_id": None,
                },
                top_k=10,
                radius=None,
            )
        assert len(results) == 1
        assert results[0].name == "API Development"
        assert results[0].score == 3.5

    @pytest.mark.asyncio
    async def test_keyword_no_date_range_passed(self, search_service):
        """agent_skill does not pass date_range to ES (no business timestamp)."""
        search_service.agent_skill_es_repo.multi_search = AsyncMock(return_value=[])

        await search_service._search_agent_skills(
            query="test",
            query_words=["test"],
            query_vector=None,
            method="keyword",
            filter_values={"user_id": "u1", "group_ids": None, "session_id": None},
            top_k=10,
            radius=None,
        )

        call_kwargs = search_service.agent_skill_es_repo.multi_search.call_args
        # date_range should NOT be in the call args
        all_kwargs = call_kwargs.kwargs if call_kwargs.kwargs else {}
        assert "date_range" not in all_kwargs

    @pytest.mark.asyncio
    async def test_vector_search(self, search_service):
        doc = _make_skill_doc()
        doc_id = str(doc.id)
        search_service.agent_skill_milvus_repo.vector_search = AsyncMock(
            return_value=[_milvus_result(doc_id, score=0.85)]
        )

        with patch.object(
            search_service,
            "_fetch_agent_skills_by_ids",
            new_callable=AsyncMock,
            return_value={doc_id: doc},
        ):
            results = await search_service._search_agent_skills(
                query="",
                query_words=[],
                query_vector=[0.1, 0.2],
                method="vector",
                filter_values={"user_id": "u1", "group_ids": None, "session_id": None},
                top_k=10,
                radius=None,
            )
        assert len(results) == 1
        assert results[0].score == 0.85

    @pytest.mark.asyncio
    async def test_keyword_no_mongo_doc_skipped(self, search_service):
        search_service.agent_skill_es_repo.multi_search = AsyncMock(
            return_value=[_es_hit("missing_id")]
        )
        with patch.object(
            search_service,
            "_fetch_agent_skills_by_ids",
            new_callable=AsyncMock,
            return_value={},
        ):
            results = await search_service._search_agent_skills(
                query="test",
                query_words=["test"],
                query_vector=None,
                method="keyword",
                filter_values={"user_id": "u1", "group_ids": None, "session_id": None},
                top_k=10,
                radius=None,
            )
        assert results == []

    @pytest.mark.asyncio
    async def test_vector_search_no_vector_returns_empty(self, search_service):
        """Vector search without query vector returns no results."""
        results = await search_service._search_agent_skills(
            query="",
            query_words=[],
            query_vector=None,
            method="vector",
            filter_values={"user_id": "u1", "group_ids": None, "session_id": None},
            top_k=10,
            radius=None,
        )
        assert results == []


# ===========================================================================
# search_memories with agent_memory type tests
# ===========================================================================


class TestSearchMemoriesAgentMemory:
    """Tests for search_memories with agent_memory memory type."""

    @pytest.mark.asyncio
    async def test_agent_memory_type_triggers_both_searches(self, search_service):
        """memory_types=['agent_memory'] triggers both case and skill searches."""
        with (
            patch.object(
                search_service,
                "_search_agent_cases",
                new_callable=AsyncMock,
                return_value=[],
            ) as mock_cases,
            patch.object(
                search_service,
                "_search_agent_skills",
                new_callable=AsyncMock,
                return_value=[],
            ) as mock_skills,
            patch.object(search_service, "_build_query_words", return_value=["test"]),
        ):

            result = await search_service.search_memories(
                query="test query",
                method="keyword",
                memory_types=["agent_memory"],
                filters={"user_id": "u1"},
                top_k=10,
                radius=None,
                include_original_data=False,
            )

            mock_cases.assert_called_once()
            mock_skills.assert_called_once()
            assert result.agent_memory is None  # No results -> None

    @pytest.mark.asyncio
    async def test_agent_memory_results_assembled(self, search_service):
        """When agent searches return results, agent_memory is populated."""
        case_item = SearchAgentCaseItem(id="c1", task_intent="Build API", score=0.9)
        skill_item = SearchAgentSkillItem(id="s1", name="API Dev", score=0.85)

        with (
            patch.object(
                search_service,
                "_search_agent_cases",
                new_callable=AsyncMock,
                return_value=[case_item],
            ),
            patch.object(
                search_service,
                "_search_agent_skills",
                new_callable=AsyncMock,
                return_value=[skill_item],
            ),
            patch.object(search_service, "_build_query_words", return_value=["api"]),
        ):

            result = await search_service.search_memories(
                query="api development",
                method="keyword",
                memory_types=["agent_memory"],
                filters={"user_id": "u1"},
                top_k=10,
                radius=None,
                include_original_data=False,
            )

            assert result.agent_memory is not None
            assert len(result.agent_memory.cases) == 1
            assert len(result.agent_memory.skills) == 1
            assert result.agent_memory.cases[0].task_intent == "Build API"
            assert result.agent_memory.skills[0].name == "API Dev"

    @pytest.mark.asyncio
    async def test_no_user_or_group_raises(self, search_service):
        """Missing both user_id and group_id raises ValueError."""
        with pytest.raises(ValueError, match="user_id.*group_id"):
            await search_service.search_memories(
                query="test",
                method="keyword",
                memory_types=["agent_memory"],
                filters={},
                top_k=10,
                radius=None,
                include_original_data=False,
            )

    @pytest.mark.asyncio
    async def test_agent_memory_with_other_types(self, search_service):
        """agent_memory can be combined with other memory types."""
        case_item = SearchAgentCaseItem(id="c1", task_intent="Task", score=0.9)

        with (
            patch.object(
                search_service,
                "_search_agent_cases",
                new_callable=AsyncMock,
                return_value=[case_item],
            ),
            patch.object(
                search_service,
                "_search_agent_skills",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch.object(
                search_service,
                "_search_episodic_memory",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch.object(search_service, "_build_query_words", return_value=["test"]),
        ):

            result = await search_service.search_memories(
                query="test",
                method="keyword",
                memory_types=["episodic_memory", "agent_memory"],
                filters={"user_id": "u1"},
                top_k=10,
                radius=None,
                include_original_data=False,
            )
            assert result.agent_memory is not None
            assert len(result.agent_memory.cases) == 1
            assert result.episodes == []

    @pytest.mark.asyncio
    async def test_top_k_applied_to_agent_results(self, search_service):
        """Top-k limit is applied to agent case and skill results."""
        cases = [
            SearchAgentCaseItem(
                id=f"c{i}", task_intent=f"Task {i}", score=1.0 - i * 0.1
            )
            for i in range(5)
        ]
        skills = [
            SearchAgentSkillItem(id=f"s{i}", name=f"Skill {i}", score=0.9 - i * 0.1)
            for i in range(5)
        ]

        with (
            patch.object(
                search_service,
                "_search_agent_cases",
                new_callable=AsyncMock,
                return_value=cases,
            ),
            patch.object(
                search_service,
                "_search_agent_skills",
                new_callable=AsyncMock,
                return_value=skills,
            ),
            patch.object(search_service, "_build_query_words", return_value=["test"]),
        ):

            result = await search_service.search_memories(
                query="test",
                method="keyword",
                memory_types=["agent_memory"],
                filters={"user_id": "u1"},
                top_k=3,
                radius=None,
                include_original_data=False,
            )
            assert len(result.agent_memory.cases) == 3
            assert len(result.agent_memory.skills) == 3


# ===========================================================================
# _extract_hit_id tests
# ===========================================================================


class TestExtractHitId:
    """Tests for SearchMemoryService._extract_hit_id."""

    def test_milvus_format(self, search_service):
        assert search_service._extract_hit_id({"id": "abc"}) == "abc"

    def test_es_format(self, search_service):
        hit = {"_source": {"id": "abc"}, "_id": "abc"}
        assert search_service._extract_hit_id(hit) == "abc"

    def test_es_fallback_to_underscore_id(self, search_service):
        hit = {"_source": {}, "_id": "abc"}
        assert search_service._extract_hit_id(hit) == "abc"

    def test_no_id_returns_none(self, search_service):
        assert search_service._extract_hit_id({}) is None

    def test_empty_string_id_returns_none(self, search_service):
        assert search_service._extract_hit_id({"id": ""}) is None

    def test_mixed_format_prefers_id(self, search_service):
        """When both 'id' and '_source.id' exist, 'id' wins."""
        hit = {"id": "milvus_id", "_source": {"id": "es_id"}, "_id": "meta_id"}
        assert search_service._extract_hit_id(hit) == "milvus_id"


# ===========================================================================
# Hybrid dedup / backfill edge cases
# ===========================================================================


class TestHybridEdgeCases:
    """Tests for hybrid search edge cases (dedup, backfill failure, empty query)."""

    @pytest.mark.asyncio
    async def test_hybrid_case_dedup_same_doc(self, search_service):
        """Same doc from ES and Milvus should be deduplicated in hybrid merge."""
        doc = _make_case_doc()
        doc_id = str(doc.id)

        search_service.agent_case_es_repo.multi_search = AsyncMock(
            return_value=[_es_hit(doc_id, score=5.0)]
        )
        search_service.agent_case_milvus_repo.vector_search = AsyncMock(
            return_value=[_milvus_result(doc_id, score=0.9)]
        )

        with patch.object(
            search_service,
            "_fetch_agent_cases_by_ids",
            new_callable=AsyncMock,
            return_value={doc_id: doc},
        ):
            results = await search_service._search_agent_cases(
                query="API",
                query_words=["API"],
                query_vector=[0.1],
                method="hybrid",
                filter_values={
                    "user_id": "u1",
                    "group_ids": None,
                    "session_id": None,
                    "start_time": None,
                    "end_time": None,
                },
                date_range={},
                top_k=10,
                radius=None,
            )

        assert len(results) == 1
        # vector_anchored_fusion: alpha*vec + (1-alpha)*sat_bm25
        # sat_bm25 = 5.0/(5.0+5.0) = 0.5; score = 0.7*0.9 + 0.3*0.5 = 0.78
        assert results[0].score == pytest.approx(0.78)

    @pytest.mark.asyncio
    async def test_hybrid_case_backfill_missing_doc(self, search_service):
        """When fused IDs are not in MongoDB, they are skipped."""
        doc = _make_case_doc()
        doc_id = str(doc.id)
        missing_id = str(ObjectId())

        search_service.agent_case_es_repo.multi_search = AsyncMock(
            return_value=[_es_hit(doc_id)]
        )
        search_service.agent_case_milvus_repo.vector_search = AsyncMock(
            return_value=[_milvus_result(missing_id)]
        )

        with patch.object(
            search_service,
            "_fetch_agent_cases_by_ids",
            new_callable=AsyncMock,
            return_value={doc_id: doc},  # missing_id not in dict
        ):
            results = await search_service._search_agent_cases(
                query="API",
                query_words=["API"],
                query_vector=[0.1],
                method="hybrid",
                filter_values={
                    "user_id": "u1",
                    "group_ids": None,
                    "session_id": None,
                    "start_time": None,
                    "end_time": None,
                },
                date_range={},
                top_k=10,
                radius=None,
            )

        assert len(results) == 1
        assert results[0].id == doc_id

    @pytest.mark.asyncio
    async def test_hybrid_skill_dedup_same_doc(self, search_service):
        """Same skill from ES and Milvus should be deduplicated."""
        doc = _make_skill_doc()
        doc_id = str(doc.id)

        search_service.agent_skill_es_repo.multi_search = AsyncMock(
            return_value=[_es_hit(doc_id)]
        )
        search_service.agent_skill_milvus_repo.vector_search = AsyncMock(
            return_value=[_milvus_result(doc_id)]
        )

        mock_rerank_svc = AsyncMock()
        mock_rerank_svc.rerank_memories = AsyncMock(
            return_value=[{"id": doc_id, "rerank_score": 0.95}]
        )

        with (
            patch.object(
                search_service,
                "_fetch_agent_skills_by_ids",
                new_callable=AsyncMock,
                return_value={doc_id: doc},
            ),
            patch(
                "agentic_layer.search_mem_service.get_rerank_service",
                return_value=mock_rerank_svc,
            ),
        ):
            results = await search_service._search_agent_skills(
                query="API",
                query_words=["API"],
                query_vector=[0.1],
                method="hybrid",
                filter_values={"user_id": "u1", "group_ids": None, "session_id": None},
                top_k=10,
                radius=None,
            )

        merged = mock_rerank_svc.rerank_memories.call_args.kwargs.get("hits")
        assert len(merged) == 1
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_hybrid_case_empty_query_words(self, search_service):
        """hybrid works when query_words is empty (only vector results)."""
        doc = _make_case_doc()
        doc_id = str(doc.id)

        search_service.agent_case_es_repo.multi_search = AsyncMock(return_value=[])
        search_service.agent_case_milvus_repo.vector_search = AsyncMock(
            return_value=[_milvus_result(doc_id, score=0.8)]
        )

        with patch.object(
            search_service,
            "_fetch_agent_cases_by_ids",
            new_callable=AsyncMock,
            return_value={doc_id: doc},
        ):
            results = await search_service._search_agent_cases(
                query="",
                query_words=[],
                query_vector=[0.1],
                method="hybrid",
                filter_values={
                    "user_id": "u1",
                    "group_ids": None,
                    "session_id": None,
                    "start_time": None,
                    "end_time": None,
                },
                date_range={},
                top_k=10,
                radius=None,
            )

        assert len(results) == 1
        # vector_anchored_fusion: only vector hit, no keyword hit
        # score = 0.7*0.8 + 0.3*0.0 = 0.56
        assert results[0].score == pytest.approx(0.56)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "top_k, expected_recall_limit",
        [
            (-1, 10),   # default: no multiplier, falls back to DEFAULT_TOP_K
            (0, 10),    # zero treated as default: no multiplier
            (3, 6),     # small explicit: multiplier applied (3*2)
            (5, 10),    # boundary: multiplier applied (5*2)
            (6, 6),     # above threshold: no multiplier
            (10, 10),   # large explicit: no multiplier
        ],
    )
    async def test_hybrid_case_recall_limit(
        self, search_service, top_k, expected_recall_limit
    ):
        """Hybrid case search applies multiplier only when 0 < top_k <= HYBRID_TOP_K_THRESHOLD."""
        doc = _make_case_doc()
        doc_id = str(doc.id)

        search_service.agent_case_es_repo.multi_search = AsyncMock(
            return_value=[_es_hit(doc_id)]
        )
        search_service.agent_case_milvus_repo.vector_search = AsyncMock(
            return_value=[_milvus_result(doc_id)]
        )

        with patch.object(
            search_service,
            "_fetch_agent_cases_by_ids",
            new_callable=AsyncMock,
            return_value={doc_id: doc},
        ):
            await search_service._search_agent_cases(
                query="API",
                query_words=["API"],
                query_vector=[0.1],
                method="hybrid",
                filter_values={
                    "user_id": "u1",
                    "group_ids": None,
                    "session_id": None,
                    "start_time": None,
                    "end_time": None,
                },
                date_range={},
                top_k=top_k,
                radius=None,
            )

        es_call_kwargs = search_service.agent_case_es_repo.multi_search.call_args.kwargs
        assert es_call_kwargs["size"] == expected_recall_limit

        milvus_call_kwargs = (
            search_service.agent_case_milvus_repo.vector_search.call_args.kwargs
        )
        assert milvus_call_kwargs["limit"] == expected_recall_limit

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "top_k, expected_recall_limit",
        [
            (-1, 10),   # default: no multiplier, falls back to DEFAULT_TOP_K
            (0, 10),    # zero treated as default: no multiplier
            (3, 6),     # small explicit: multiplier applied (3*2)
            (5, 10),    # boundary: multiplier applied (5*2)
            (6, 6),     # above threshold: no multiplier
            (10, 10),   # large explicit: no multiplier
        ],
    )
    async def test_hybrid_skill_recall_limit(
        self, search_service, top_k, expected_recall_limit
    ):
        """Hybrid skill search applies multiplier only when 0 < top_k <= HYBRID_TOP_K_THRESHOLD."""
        doc = _make_skill_doc()
        doc_id = str(doc.id)

        search_service.agent_skill_es_repo.multi_search = AsyncMock(
            return_value=[_es_hit(doc_id)]
        )
        search_service.agent_skill_milvus_repo.vector_search = AsyncMock(
            return_value=[_milvus_result(doc_id)]
        )

        mock_rerank_svc = AsyncMock()
        mock_rerank_svc.rerank_memories = AsyncMock(
            return_value=[{"id": doc_id, "rerank_score": 0.9}]
        )

        with (
            patch.object(
                search_service,
                "_fetch_agent_skills_by_ids",
                new_callable=AsyncMock,
                return_value={doc_id: doc},
            ),
            patch(
                "agentic_layer.search_mem_service.get_rerank_service",
                return_value=mock_rerank_svc,
            ),
        ):
            await search_service._search_agent_skills(
                query="API",
                query_words=["API"],
                query_vector=[0.1],
                method="hybrid",
                filter_values={
                    "user_id": "u1",
                    "group_ids": None,
                },
                top_k=top_k,
                radius=None,
            )

        es_call_kwargs = search_service.agent_skill_es_repo.multi_search.call_args.kwargs
        assert es_call_kwargs["size"] == expected_recall_limit

        milvus_call_kwargs = (
            search_service.agent_skill_milvus_repo.vector_search.call_args.kwargs
        )
        assert milvus_call_kwargs["limit"] == expected_recall_limit


# ===========================================================================
# _extract_filter_values tests
# ===========================================================================


class TestExtractFilterValues:
    """Tests for SearchMemoryService._extract_filter_values with agent-relevant filters."""

    def test_simple_user_id(self, search_service):
        result = search_service._extract_filter_values({"user_id": "agent_user_001"})
        assert result["user_id"] == "agent_user_001"

    def test_user_id_eq_format(self, search_service):
        result = search_service._extract_filter_values({"user_id": {"eq": "u1"}})
        assert result["user_id"] == "u1"

    def test_user_id_in_format(self, search_service):
        result = search_service._extract_filter_values(
            {"user_id": {"in": ["u1", "u2"]}}
        )
        assert result["user_id"] == "u1"

    def test_group_id_string(self, search_service):
        result = search_service._extract_filter_values({"group_id": "g1"})
        assert result["group_ids"] == ["g1"]

    def test_group_id_list(self, search_service):
        result = search_service._extract_filter_values({"group_id": ["g1", "g2"]})
        assert result["group_ids"] == ["g1", "g2"]

    def test_group_id_in_format(self, search_service):
        result = search_service._extract_filter_values(
            {"group_id": {"in": ["g1", "g2"]}}
        )
        assert result["group_ids"] == ["g1", "g2"]

    def test_timestamp_range(self, search_service):
        result = search_service._extract_filter_values(
            {"timestamp": {"gte": "2025-01-01T00:00:00", "lte": "2025-12-31T23:59:59"}}
        )
        assert result["start_time"] is not None
        assert result["end_time"] is not None

    def test_and_combinator(self, search_service):
        result = search_service._extract_filter_values(
            {"AND": [{"user_id": "u1"}, {"group_id": "g1"}]}
        )
        assert result["user_id"] == "u1"
        assert result["group_ids"] == ["g1"]

    def test_empty_filters(self, search_service):
        result = search_service._extract_filter_values({})
        assert result["user_id"] is None
        assert result["group_ids"] is None

    def test_session_id(self, search_service):
        result = search_service._extract_filter_values({"session_id": "sess_001"})
        assert result["session_id"] == "sess_001"

    def test_timestamp_epoch_milliseconds(self, search_service):
        result = search_service._extract_filter_values(
            {"timestamp": {"gte": 1735689600000}}  # 2025-01-01 in milliseconds
        )
        assert result["start_time"] is not None


# ===========================================================================
# SearchMemoriesRequest DTO validation tests
# ===========================================================================


class TestSearchMemoriesRequestValidation:
    """Tests for SearchMemoriesRequest DTO field validation."""

    def test_empty_query_rejected(self):
        """query='' should be rejected by Pydantic min_length=1."""
        from api_specs.dtos.memory import SearchMemoriesRequest

        with pytest.raises(ValidationError) as exc_info:
            SearchMemoriesRequest(query="", filters={"user_id": "u1"})
        assert "query" in str(exc_info.value)

    def test_valid_query_accepted(self):
        """Non-empty query should pass validation."""
        from api_specs.dtos.memory import SearchMemoriesRequest

        req = SearchMemoriesRequest(query="hello", filters={"user_id": "u1"})
        assert req.query == "hello"


# ===========================================================================
# _compute_recall_limit tests
# ===========================================================================


class TestComputeRecallLimit:
    """Tests for _compute_recall_limit helper function."""

    def test_with_multiplier_normal_top_k(self):
        from agentic_layer.search_mem_service import _compute_recall_limit

        assert _compute_recall_limit(top_k=3, apply_multiplier=True) == 6

    def test_with_multiplier_top_k_five(self):
        from agentic_layer.search_mem_service import _compute_recall_limit

        assert _compute_recall_limit(top_k=5, apply_multiplier=True) == 10

    def test_with_multiplier_top_k_zero_uses_default(self):
        from agentic_layer.search_mem_service import (
            _compute_recall_limit,
            DEFAULT_TOP_K,
            MAX_RECALL_MULTIPLIER,
        )

        assert _compute_recall_limit(top_k=0, apply_multiplier=True) == (
            DEFAULT_TOP_K * MAX_RECALL_MULTIPLIER
        )

    def test_without_multiplier_large_top_k(self):
        from agentic_layer.search_mem_service import _compute_recall_limit

        assert _compute_recall_limit(top_k=8, apply_multiplier=False) == 8

    def test_without_multiplier_top_k_ten(self):
        from agentic_layer.search_mem_service import _compute_recall_limit

        assert _compute_recall_limit(top_k=10, apply_multiplier=False) == 10

    def test_without_multiplier_top_k_zero_uses_default(self):
        from agentic_layer.search_mem_service import (
            _compute_recall_limit,
            DEFAULT_TOP_K,
        )

        assert _compute_recall_limit(top_k=0, apply_multiplier=False) == DEFAULT_TOP_K

    def test_without_multiplier_top_k_one(self):
        from agentic_layer.search_mem_service import _compute_recall_limit

        assert _compute_recall_limit(top_k=1, apply_multiplier=False) == 1

    def test_with_multiplier_top_k_one(self):
        from agentic_layer.search_mem_service import _compute_recall_limit

        assert _compute_recall_limit(top_k=1, apply_multiplier=True) == 2

    def test_negative_top_k_uses_default(self):
        from agentic_layer.search_mem_service import (
            _compute_recall_limit,
            DEFAULT_TOP_K,
            MAX_RECALL_MULTIPLIER,
        )

        assert _compute_recall_limit(top_k=-1, apply_multiplier=True) == (
            DEFAULT_TOP_K * MAX_RECALL_MULTIPLIER
        )
        assert _compute_recall_limit(top_k=-1, apply_multiplier=False) == DEFAULT_TOP_K

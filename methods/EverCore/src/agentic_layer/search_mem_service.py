"""
Memory Search service (v1)

Provides business logic for POST /api/v1/memories/search endpoint.
Handles query parsing, filter validation, multi-method retrieval (keyword/vector/hybrid/rrf/agentic),
and DTO conversion for episodic_memory and profile types.

Methods:
- keyword: BM25 keyword retrieval (ES only)
- vector: Vector semantic retrieval (Milvus only)
- hybrid: Keyword + Vector + Rerank service
- rrf: Keyword + Vector + RRF fusion
- agentic: LLM-guided multi-round retrieval
"""

import asyncio
import jieba
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from core.di import service, get_bean
from api_specs.dtos.memory import SearchAtomicFactItem
from core.nlp.stopwords_utils import filter_stopwords
from core.observation.stage_timer import timed, timed_parallel
from api_specs.dtos.memory import (
    SearchMemoriesResponseData,
    SearchEpisodeItem,
    SearchProfileItem,
    RawMessageDTO,
    AgentMemorySearchResult,
    SearchAgentCaseItem,
    SearchAgentSkillItem,
)

from infra_layer.adapters.out.search.repository.episodic_memory_es_repository import (
    EpisodicMemoryEsRepository,
)
from infra_layer.adapters.out.search.repository.episodic_memory_milvus_repository import (
    EpisodicMemoryMilvusRepository,
)
from infra_layer.adapters.out.search.repository.user_profile_milvus_repository import (
    UserProfileMilvusRepository,
)
from infra_layer.adapters.out.search.repository.agent_case_es_repository import (
    AgentCaseEsRepository,
)
from infra_layer.adapters.out.search.repository.agent_skill_es_repository import (
    AgentSkillEsRepository,
)
from infra_layer.adapters.out.search.repository.agent_case_milvus_repository import (
    AgentCaseMilvusRepository,
)
from infra_layer.adapters.out.search.repository.agent_skill_milvus_repository import (
    AgentSkillMilvusRepository,
)

# MongoDB documents for type annotations only — DB access goes through repos
from infra_layer.adapters.out.persistence.document.memory.episodic_memory import (
    EpisodicMemory,
)
from infra_layer.adapters.out.persistence.document.memory.agent_case import (
    AgentCaseRecord,
)
from infra_layer.adapters.out.persistence.document.memory.agent_skill import (
    AgentSkillRecord,
)

# Raw repositories for MongoDB persistence
from infra_layer.adapters.out.persistence.repository.episodic_memory_raw_repository import (
    EpisodicMemoryRawRepository,
)
from infra_layer.adapters.out.persistence.repository.agent_case_raw_repository import (
    AgentCaseRawRepository,
)
from infra_layer.adapters.out.persistence.repository.agent_skill_raw_repository import (
    AgentSkillRawRepository,
)

# Rerank service for hybrid method
from agentic_layer.rerank_service import get_rerank_service
from agentic_layer.retrieval_utils import vector_anchored_fusion

# MemoryManager for agentic retrieval
from agentic_layer.memory_manager import MemoryManager
from api_specs.memory_models import RetrieveMethod, MemoryType
from api_specs.dtos.memory import RetrieveMemRequest

# Retrieve metrics
from agentic_layer.metrics.retrieve_metrics import (
    record_retrieve_request,
    record_retrieve_stage,
)

# RawMessageService for pending messages
from service.raw_message_service import RawMessageService

# Memorize config (for skill_retire_confidence threshold)
from biz_layer.memorize_config import DEFAULT_MEMORIZE_CONFIG

# Constants
from core.oxm.constants import MAGIC_ALL
from biz_layer.retrieve_constants import AGENT_MEMORY_MILVUS_RADIUS

logger = logging.getLogger(__name__)

# Constants
DEFAULT_TOP_K = 10
MAX_RECALL_MULTIPLIER = 2
HYBRID_TOP_K_THRESHOLD = 5


def _compute_recall_limit(top_k: int, apply_multiplier: bool) -> int:
    """Compute the recall limit for keyword / vector searches.

    Args:
        top_k: Requested result count (0 means use DEFAULT_TOP_K).
        apply_multiplier: When True, multiply by MAX_RECALL_MULTIPLIER
            to over-recall for better rerank quality.  Set to False
            when top_k is large enough that the multiplier only adds
            rerank latency without meaningful quality gain.
    """
    effective_top_k = top_k if top_k > 0 else DEFAULT_TOP_K
    if apply_multiplier:
        return effective_top_k * MAX_RECALL_MULTIPLIER
    return effective_top_k


@service(name="search_memory_service", primary=True)
class SearchMemoryService:
    """Memory Search service for v1 API.

    Handles query parsing, filter validation, multi-method retrieval,
    and document-to-DTO conversion for search results.
    """

    def __init__(self):
        """Initialize search service with repositories."""
        # ES Repositories
        self.episodic_es_repo = EpisodicMemoryEsRepository()

        # Milvus Repositories
        self.episodic_milvus_repo = EpisodicMemoryMilvusRepository()
        self.profile_milvus_repo = UserProfileMilvusRepository()

        # Agent memory repositories
        self.agent_case_es_repo = AgentCaseEsRepository()
        self.agent_skill_es_repo = AgentSkillEsRepository()
        self.agent_case_milvus_repo = AgentCaseMilvusRepository()
        self.agent_skill_milvus_repo = AgentSkillMilvusRepository()

        # MongoDB raw repositories (for fetching full docs by id)
        self.episodic_raw_repo = EpisodicMemoryRawRepository()
        self.agent_case_raw_repo = AgentCaseRawRepository()
        self.agent_skill_raw_repo = AgentSkillRawRepository()

        # MemoryManager for agentic retrieval
        self.memory_manager = MemoryManager()

        # RawMessageService for pending messages
        self.raw_message_service = RawMessageService()

    @staticmethod
    def _make_error(code: str, message: str) -> dict:
        """Build an error response dict."""
        return {"error": {"code": code, "message": message}}

    def _build_query_words(self, query: str) -> List[str]:
        """Build query words for BM25 search."""
        if not query:
            return []
        raw_words = list(jieba.cut_for_search(query))
        return filter_stopwords(raw_words, min_length=2)

    async def _get_query_vector(
        self, query: str, retrieve_method: str = 'vector'
    ) -> List[float]:
        """Get query vector embedding."""
        stage_start = time.perf_counter()
        vectorize_service = get_bean("vectorize_service")
        embedding = await vectorize_service.get_embedding(query)
        record_retrieve_stage(
            retrieve_method=retrieve_method,
            stage='embedding',
            memory_type='shared',
            duration_seconds=time.perf_counter() - stage_start,
        )
        return embedding.tolist()

    def _parse_timestamp(self, value: Any) -> Optional[datetime]:
        """Parse timestamp from various formats."""
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, (int, float)):
            # Assume milliseconds if > 1e12
            if value > 1e12:
                return datetime.fromtimestamp(value / 1000)
            return datetime.fromtimestamp(value)
        if isinstance(value, str):
            from common_utils.datetime_utils import from_iso_format

            return from_iso_format(value)
        return None

    @staticmethod
    def _extract_hit_id(hit: Dict[str, Any]) -> Optional[str]:
        """Extract document ID from a search hit (ES or Milvus format).

        Handles both ES hits (``_source.id`` / ``_id``) and Milvus results
        (``id``).  Returns ``None`` when no usable ID is found.
        """
        return (
            hit.get("id") or hit.get("_source", {}).get("id") or hit.get("_id")
        ) or None

    async def _fetch_episodic_memories_by_ids(
        self, episode_ids: List[str]
    ) -> Dict[str, EpisodicMemory]:
        """Batch fetch episodic memories from MongoDB by IDs.

        Args:
            episode_ids: List of episode IDs to fetch

        Returns:
            Dict mapping episode ID to EpisodicMemory document
        """
        if not episode_ids:
            return {}

        episodes_data = await self.episodic_raw_repo.find_by_ids(episode_ids)
        return {str(ep.id): ep for ep in episodes_data}

    def _extract_filter_values(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Extract filter values from filters dict.

        Args:
            filters: Raw filters dict from request

        Returns:
            Dict with user_id, group_ids, session_id, start_time, end_time
        """
        result = {
            "user_id": None,
            "group_ids": None,
            "session_id": None,
            "start_time": None,
            "end_time": None,
        }

        def extract_from_node(node: Dict[str, Any]) -> None:
            """Extract values from a single filter node into result dict."""
            if "user_id" in node and result["user_id"] is None:
                user_filter = node["user_id"]
                if isinstance(user_filter, str):
                    result["user_id"] = user_filter
                elif isinstance(user_filter, dict):
                    if "in" in user_filter:
                        result["user_id"] = (
                            user_filter["in"][0] if user_filter["in"] else None
                        )
                    elif "eq" in user_filter:
                        result["user_id"] = user_filter["eq"]

            if "group_id" in node and result["group_ids"] is None:
                group_filter = node["group_id"]
                if isinstance(group_filter, str):
                    result["group_ids"] = [group_filter]
                elif isinstance(group_filter, list):
                    result["group_ids"] = group_filter
                elif isinstance(group_filter, dict):
                    if "in" in group_filter:
                        result["group_ids"] = group_filter["in"]
                    elif "eq" in group_filter:
                        result["group_ids"] = [group_filter["eq"]]

            if "session_id" in node and result["session_id"] is None:
                session_filter = node["session_id"]
                if isinstance(session_filter, str):
                    result["session_id"] = session_filter
                elif isinstance(session_filter, dict):
                    if "eq" in session_filter:
                        result["session_id"] = session_filter["eq"]
                    elif "in" in session_filter:
                        result["session_id"] = (
                            session_filter["in"][0] if session_filter["in"] else None
                        )

            if "timestamp" in node:
                ts_filter = node["timestamp"]
                if isinstance(ts_filter, dict):
                    if "gte" in ts_filter and result["start_time"] is None:
                        result["start_time"] = self._parse_timestamp(ts_filter["gte"])
                    if "gt" in ts_filter and result["start_time"] is None:
                        result["start_time"] = self._parse_timestamp(ts_filter["gt"])
                    if "lte" in ts_filter and result["end_time"] is None:
                        result["end_time"] = self._parse_timestamp(ts_filter["lte"])
                    if "lt" in ts_filter and result["end_time"] is None:
                        result["end_time"] = self._parse_timestamp(ts_filter["lt"])

        # Handle top-level fields
        extract_from_node(filters)

        # Handle AND/OR combinators
        for combinator in ["AND", "OR"]:
            if combinator in filters and isinstance(filters[combinator], list):
                for item in filters[combinator]:
                    if isinstance(item, dict):
                        extract_from_node(item)

        return result

    async def search_memories(
        self,
        query: str,
        method: str,
        memory_types: List[str],
        filters: Dict[str, Any],
        top_k: int,
        radius: Optional[float],
        include_original_data: bool,
    ) -> SearchMemoriesResponseData:
        """Search memories using v1 repositories.

        Args:
            query: Search query text
            method: Retrieval method (keyword/vector/hybrid/agentic)
            memory_types: List of memory types to search (episodic_memory/profile/raw_message)
            filters: Filter conditions using Filters DSL
            top_k: Max results
            radius: Similarity threshold
            include_original_data: Whether to include original data

        Returns:
            SearchMemoriesResponseData with search results

        Raises:
            ValueError: If validation fails
        """
        with timed("parse_filters"):
            # Validate and extract filter values
            filter_values = self._extract_filter_values(filters)

            # Validate scope: at least one of user_id or group_id is required
            if not filter_values["user_id"] and not filter_values["group_ids"]:
                raise ValueError(
                    "filters must contain at least one of 'user_id' or 'group_id'"
                )

            # Profile search requires user_id and only supports vector methods.
            # When conditions are not met, silently remove PROFILE so other types proceed.
            if MemoryType.PROFILE.value in memory_types:
                if not filter_values["user_id"] or method == "keyword":
                    memory_types = [
                        mt for mt in memory_types if mt != MemoryType.PROFILE.value
                    ]

            # Build query words for keyword search
            query_words = self._build_query_words(query)

            # Prepare results containers
            episodes: List[SearchEpisodeItem] = []
            profiles: List[SearchProfileItem] = []
            raw_messages: List[RawMessageDTO] = []
            agent_cases: List[SearchAgentCaseItem] = []
            agent_skills: List[SearchAgentSkillItem] = []

            # Build date range for ES
            date_range = {}
            if filter_values["start_time"] or filter_values["end_time"]:
                if filter_values["start_time"]:
                    date_range["gte"] = filter_values["start_time"].isoformat()
                if filter_values["end_time"]:
                    date_range["lte"] = filter_values["end_time"].isoformat()

            # Search each supported memory type.
            # atomic_fact/foresight are intentionally ignored in v1 response.
            search_tasks = []
            task_memory_types = []
            unsupported_memory_types = [
                memory_type
                for memory_type in memory_types
                if memory_type
                not in {
                    MemoryType.EPISODIC_MEMORY.value,
                    MemoryType.PROFILE.value,
                    MemoryType.RAW_MESSAGE.value,
                    MemoryType.AGENT_MEMORY.value,
                }
            ]
            if unsupported_memory_types:
                logger.warning(
                    "Unsupported memory types in search request: %s",
                    unsupported_memory_types,
                )

        # Check allowed methods (configurable via env var, comma-separated)
        allowed_methods_env = os.getenv("ALLOWED_SEARCH_METHODS")
        if allowed_methods_env:
            allowed_set = {m.strip() for m in allowed_methods_env.split(",")}
            if method not in allowed_set:
                raise ValueError(
                    f"Search method '{method}' is not supported. "
                    f"Allowed methods: {sorted(allowed_set)}"
                )

        # Extended method: delegate to DI-registered handler
        builtin_methods = {"keyword", "vector", "hybrid", "agentic"}
        if method not in builtin_methods:
            return await self._search_extended_method(
                method=method,
                query=query,
                memory_types=memory_types,
                filter_values=filter_values,
                top_k=top_k,
                radius=radius,
                include_original_data=include_original_data,
            )

        # Vector embedding (if needed)
        query_vector = None
        if method in ("vector", "hybrid"):
            # embedding_api timing is inside vectorize_service.get_embedding()
            query_vector = await self._get_query_vector(query, retrieve_method=method)

        for memory_type in memory_types:
            if memory_type == MemoryType.EPISODIC_MEMORY.value:
                if method == "hybrid":
                    # hybrid → hierarchical retrieval
                    search_tasks.append(
                        self._search_episodic_memory_hierarchical(
                            query=query,
                            query_vector=query_vector,
                            filter_values=filter_values,
                            top_k=top_k,
                            radius=radius,
                        )
                    )
                else:
                    # keyword / vector / agentic → keep original logic
                    search_tasks.append(
                        self._search_episodic_memory(
                            query=query,
                            query_words=query_words,
                            query_vector=query_vector,
                            method=method,
                            filter_values=filter_values,
                            date_range=date_range,
                            top_k=top_k,
                            radius=radius,
                        )
                    )
                task_memory_types.append(memory_type)
            elif memory_type == MemoryType.PROFILE.value:
                search_tasks.append(
                    self._search_profile(
                        query_vector=query_vector,
                        method=method,
                        filter_values=filter_values,
                        top_k=top_k,
                        radius=radius,
                    )
                )
                task_memory_types.append(memory_type)
            elif memory_type == MemoryType.RAW_MESSAGE.value:
                search_tasks.append(
                    self._search_raw_message(filter_values=filter_values, top_k=top_k)
                )
                task_memory_types.append(memory_type)
            elif memory_type == MemoryType.AGENT_MEMORY.value:
                search_tasks.append(
                    self._search_agent_cases(
                        query=query,
                        query_words=query_words,
                        query_vector=query_vector,
                        method=method,
                        filter_values=filter_values,
                        date_range=date_range,
                        top_k=top_k,
                        radius=radius,
                    )
                )
                task_memory_types.append(MemoryType.AGENT_CASE.value)
                search_tasks.append(
                    self._search_agent_skills(
                        query=query,
                        query_words=query_words,
                        query_vector=query_vector,
                        method=method,
                        filter_values=filter_values,
                        top_k=top_k,
                        radius=radius,
                    )
                )
                task_memory_types.append(MemoryType.AGENT_SKILL.value)

        # Execute searches in parallel
        search_start = time.perf_counter()
        with timed_parallel("concurrent_retrieval"):
            results = await asyncio.gather(*search_tasks, return_exceptions=True)
        search_duration = time.perf_counter() - search_start

        # Propagate critical system errors before processing results
        from common_utils.async_utils import reraise_critical_errors

        reraise_critical_errors(results)

        # Collect results from parallel searches
        has_error = False
        total_result_count = 0
        for i, result in enumerate(results):
            memory_type = task_memory_types[i]
            if isinstance(result, Exception):
                logger.error("Search failed for %s: %s", memory_type, result)
                has_error = True
                continue
            if result:
                total_result_count += len(result)
                if memory_type == MemoryType.EPISODIC_MEMORY.value:
                    episodes.extend(result)
                elif memory_type == MemoryType.PROFILE.value:
                    profiles.extend(result)
                elif memory_type == MemoryType.RAW_MESSAGE.value:
                    raw_messages.extend(result)
                elif memory_type == MemoryType.AGENT_CASE.value:
                    agent_cases.extend(result)
                elif memory_type == MemoryType.AGENT_SKILL.value:
                    agent_skills.extend(result)

        # Record request-level metrics (once per search request, not per memory_type)
        # memory_type="all" ensures 1 request = 1 count for all methods including agentic
        # Note: agentic also records per-memory_type metrics inside memory_manager,
        # but this "all" record is the authoritative one for QPS and e2e duration.
        if has_error:
            status = 'error'
        elif total_result_count > 0:
            status = 'success'
        else:
            status = 'empty_result'
        record_retrieve_request(
            memory_type='all',
            retrieve_method=method,
            status=status,
            duration_seconds=search_duration,
            results_count=total_result_count,
        )

        # Fetch full data from MongoDB if needed
        original_data = None
        if include_original_data:
            with timed("fetch_original_data"):
                original_data = await self._fetch_original_data(episodes, profiles)

        with timed("assemble_results"):

            # Apply top_k limit
            if top_k > 0:
                episodes = episodes[:top_k]
                profiles = profiles[:top_k]
                raw_messages = raw_messages[:top_k]
                agent_cases = agent_cases[:top_k]
                agent_skills = agent_skills[:top_k]

            # Build agent_memory container if any agent results exist
            agent_memory = None
            if agent_cases or agent_skills:
                agent_memory = AgentMemorySearchResult(
                    cases=agent_cases, skills=agent_skills
                )

            # Build response
            return SearchMemoriesResponseData(
                episodes=episodes,
                profiles=profiles,
                raw_messages=raw_messages,
                agent_memory=agent_memory,
                query={"text": query, "method": method, "filters_applied": filters},
                original_data=original_data,
            )

    async def _search_episodic_memory(
        self,
        query: str,
        query_words: List[str],
        query_vector: Optional[List[float]],
        method: str,
        filter_values: Dict[str, Any],
        date_range: Dict[str, str],
        top_k: int,
        radius: Optional[float],
    ) -> List[SearchEpisodeItem]:
        """Search episodic memories."""
        results: List[SearchEpisodeItem] = []
        limit = (
            top_k * MAX_RECALL_MULTIPLIER
            if top_k > 0
            else DEFAULT_TOP_K * MAX_RECALL_MULTIPLIER
        )

        memory_type = MemoryType.EPISODIC_MEMORY.value
        with timed("episode_search"):
            if method == "keyword":
                stage_start = time.perf_counter()
                with timed("keyword_search"):
                    hits = await self.episodic_es_repo.multi_search(
                        query=query_words,
                        user_id=filter_values["user_id"],
                        group_ids=filter_values["group_ids"],
                        session_id=filter_values["session_id"],
                        date_range=date_range,
                        size=limit,
                    )
                record_retrieve_stage(
                    retrieve_method=method,
                    stage='keyword_search',
                    memory_type=memory_type,
                    duration_seconds=time.perf_counter() - stage_start,
                )
                # Extract IDs for MongoDB backfill
                episode_ids = [
                    hit.get("_source", {}).get("id")
                    for hit in hits
                    if hit.get("_source", {}).get("id")
                ]

                # Batch fetch from MongoDB to get complete data
                with timed("backfill_from_store"):
                    episodes_dict = {}
                    if episode_ids:
                        episodes_dict = await self._fetch_episodic_memories_by_ids(
                            episode_ids
                        )

                for hit in hits:
                    source = hit.get("_source", {})
                    episode_id = source.get("id", "")
                    ep_doc = episodes_dict.get(episode_id)

                    results.append(
                        SearchEpisodeItem(
                            id=episode_id,
                            user_id=ep_doc.user_id if ep_doc else source.get("user_id"),
                            group_id=(
                                ep_doc.group_id if ep_doc else source.get("group_id")
                            ),
                            session_id=(
                                ep_doc.session_id
                                if ep_doc
                                else source.get("session_id")
                            ),
                            timestamp=(
                                ep_doc.timestamp
                                if ep_doc
                                else self._parse_timestamp(source.get("timestamp"))
                            ),
                            participants=(
                                ep_doc.participants
                                if ep_doc
                                else source.get("participants")
                            ),
                            summary=ep_doc.summary if ep_doc else source.get("summary"),
                            subject=ep_doc.subject if ep_doc else source.get("subject"),
                            episode=ep_doc.episode if ep_doc else source.get("episode"),
                            type=ep_doc.type if ep_doc else source.get("type"),
                            parent_type=(
                                ep_doc.parent_type
                                if ep_doc
                                else source.get("parent_type")
                            ),
                            parent_id=(
                                ep_doc.parent_id if ep_doc else source.get("parent_id")
                            ),
                            score=hit.get("_score"),
                        )
                    )

            elif method == "vector" and query_vector:
                stage_start = time.perf_counter()
                with timed("vector_search"):
                    search_results = await self.episodic_milvus_repo.vector_search(
                        query_vector=query_vector,
                        user_id=filter_values["user_id"],
                        group_ids=filter_values["group_ids"],
                        session_id=filter_values["session_id"],
                        start_time=filter_values["start_time"],
                        end_time=filter_values["end_time"],
                        limit=limit,
                        radius=radius,
                    )
                record_retrieve_stage(
                    retrieve_method=method,
                    stage='vector_search',
                    memory_type=memory_type,
                    duration_seconds=time.perf_counter() - stage_start,
                )
                # Extract IDs for MongoDB backfill
                episode_ids = [res.get("id") for res in search_results if res.get("id")]

                # Batch fetch from MongoDB to get complete data
                with timed("backfill_from_store"):
                    episodes_dict = {}
                    if episode_ids:
                        episodes_dict = await self._fetch_episodic_memories_by_ids(
                            episode_ids
                        )

                for res in search_results:
                    episode_id = res.get("id")
                    ep_doc = episodes_dict.get(episode_id)

                    ts = res.get("timestamp")
                    timestamp = (
                        datetime.fromtimestamp(ts / 1000)
                        if isinstance(ts, (int, float))
                        else None
                    )

                    # Use MongoDB data for display fields, fallback to Milvus data
                    results.append(
                        SearchEpisodeItem(
                            id=episode_id or "",
                            user_id=ep_doc.user_id if ep_doc else res.get("user_id"),
                            group_id=(
                                ep_doc.group_id if ep_doc else res.get("group_id")
                            ),
                            session_id=(
                                ep_doc.session_id if ep_doc else res.get("session_id")
                            ),
                            timestamp=timestamp,
                            participants=(
                                ep_doc.participants
                                if ep_doc
                                else res.get("participants")
                            ),
                            summary=ep_doc.summary if ep_doc else None,
                            subject=ep_doc.subject if ep_doc else None,
                            episode=(ep_doc.episode if ep_doc else res.get("episode")),
                            type=ep_doc.type if ep_doc else res.get("type"),
                            parent_type=(
                                ep_doc.parent_type if ep_doc else res.get("parent_type")
                            ),
                            parent_id=(
                                ep_doc.parent_id if ep_doc else res.get("parent_id")
                            ),
                            score=res.get("score"),
                        )
                    )

            elif method == "hybrid" and query_vector:
                # Hybrid: keyword + vector + rerank
                stage_start = time.perf_counter()
                with timed("keyword_search"):
                    keyword_hits = await self.episodic_es_repo.multi_search(
                        query=query_words,
                        user_id=filter_values["user_id"],
                        group_ids=filter_values["group_ids"],
                        session_id=filter_values["session_id"],
                        date_range=date_range,
                        size=limit,
                    )
                record_retrieve_stage(
                    retrieve_method=method,
                    stage='keyword_search',
                    memory_type=memory_type,
                    duration_seconds=time.perf_counter() - stage_start,
                )
                stage_start = time.perf_counter()
                with timed("vector_search"):
                    vector_results = await self.episodic_milvus_repo.vector_search(
                        query_vector=query_vector,
                        user_id=filter_values["user_id"],
                        group_ids=filter_values["group_ids"],
                        session_id=filter_values["session_id"],
                        start_time=filter_values["start_time"],
                        end_time=filter_values["end_time"],
                        limit=limit,
                        radius=radius,
                    )
                record_retrieve_stage(
                    retrieve_method=method,
                    stage='vector_search',
                    memory_type=memory_type,
                    duration_seconds=time.perf_counter() - stage_start,
                )
                # Tag memory_type for rerank text extraction
                for h in keyword_hits:
                    h["memory_type"] = MemoryType.EPISODIC_MEMORY.value
                for h in vector_results:
                    h["memory_type"] = MemoryType.EPISODIC_MEMORY.value
                # Merge and deduplicate
                seen_ids = {
                    h.get("_source", {}).get("id") or h.get("_id") for h in keyword_hits
                }
                merged_hits = keyword_hits + [
                    h for h in vector_results if h.get("id") not in seen_ids
                ]
                # Rerank
                stage_start = time.perf_counter()
                rerank_service = get_rerank_service()
                rerank_top_k = top_k if top_k > 0 else DEFAULT_TOP_K
                reranked_hits = await rerank_service.rerank_memories(
                    query=query, hits=merged_hits, top_k=rerank_top_k
                )
                rerank_ms = (time.perf_counter() - stage_start) * 1000
                record_retrieve_stage(
                    retrieve_method=method,
                    stage='rerank',
                    memory_type=memory_type,
                    duration_seconds=rerank_ms / 1000,
                )
                logger.info(
                    "[RERANK_DIAG] memory_type=%s in=%d out=%d top_k=%d took=%.1fms",
                    memory_type,
                    len(merged_hits),
                    len(reranked_hits),
                    rerank_top_k,
                    rerank_ms,
                )
                # Batch fetch from MongoDB for complete display fields
                episode_ids = [
                    h.get("id") or h.get("_id")
                    for h in reranked_hits
                    if h.get("id") or h.get("_id")
                ]
                with timed("backfill_from_store"):
                    episodes_dict = {}
                    if episode_ids:
                        episodes_dict = await self._fetch_episodic_memories_by_ids(
                            episode_ids
                        )

                # Convert to SearchEpisodeItem (MongoDB data preferred, hit as fallback)
                for hit in reranked_hits:
                    episode_id = hit.get("id") or hit.get("_id", "")
                    ep_doc = episodes_dict.get(episode_id)
                    results.append(
                        SearchEpisodeItem(
                            id=episode_id,
                            user_id=ep_doc.user_id if ep_doc else hit.get("user_id"),
                            group_id=ep_doc.group_id if ep_doc else hit.get("group_id"),
                            session_id=(
                                ep_doc.session_id if ep_doc else hit.get("session_id")
                            ),
                            timestamp=self._parse_timestamp(hit.get("timestamp")),
                            participants=(
                                ep_doc.participants
                                if ep_doc
                                else hit.get("participants")
                            ),
                            summary=ep_doc.summary if ep_doc else hit.get("summary"),
                            subject=ep_doc.subject if ep_doc else hit.get("subject"),
                            episode=ep_doc.episode if ep_doc else hit.get("episode"),
                            type=ep_doc.type if ep_doc else hit.get("type"),
                            parent_type=(
                                ep_doc.parent_type if ep_doc else hit.get("parent_type")
                            ),
                            parent_id=(
                                ep_doc.parent_id if ep_doc else hit.get("parent_id")
                            ),
                            score=hit.get("rerank_score", hit.get("score", 0)),
                        )
                    )

            elif method == "agentic":
                # Agentic: LLM-guided multi-round retrieval
                with timed("agentic_retrieval"):
                    results = await self._search_agentic_episodic_memory(
                        query=query, filter_values=filter_values, top_k=top_k
                    )

        return results

    async def _search_profile(
        self,
        query_vector: Optional[List[float]],
        method: str,
        filter_values: Dict[str, Any],
        top_k: int,
        radius: Optional[float],
    ) -> List[SearchProfileItem]:
        """Search user profiles."""
        results: List[SearchProfileItem] = []
        limit = (
            top_k * MAX_RECALL_MULTIPLIER
            if top_k > 0
            else DEFAULT_TOP_K * MAX_RECALL_MULTIPLIER
        )

        if method in ("vector", "hybrid", "agentic") and query_vector:
            # Profile only has Milvus (no ES), so all methods fall back to vector search.
            # Agentic multi-round pipeline adds no value for short profile items.
            stage_start = time.perf_counter()
            search_results = await self.profile_milvus_repo.vector_search(
                query_vector=query_vector,
                user_id=filter_values["user_id"],
                limit=limit,
                radius=radius,
            )
            record_retrieve_stage(
                retrieve_method=method,
                stage='vector_search',
                memory_type=MemoryType.PROFILE.value,
                duration_seconds=time.perf_counter() - stage_start,
            )
            for res in search_results:
                results.append(
                    SearchProfileItem(
                        id=res.get("id", ""),
                        user_id=res.get("user_id"),
                        group_id=res.get("group_id"),
                        scenario=res.get("scenario"),
                        memcell_count=res.get("memcell_count"),
                        profile_data={
                            "item_type": res.get("item_type", ""),
                            "embed_text": res.get("embed_text", ""),
                        },
                        score=res.get("score"),
                    )
                )

        return results

    async def _search_raw_message(
        self, filter_values: Dict[str, Any], top_k: int
    ) -> List[RawMessageDTO]:
        """Search raw unprocessed messages (pending messages).

        Retrieves pending messages from RawMessage collection based on filter criteria.
        Raw messages are unprocessed conversations that have not yet been extracted
        into memory structures (episodic_memory, profile, etc.).

        Args:
            filter_values: Filter values dict containing user_id, group_ids, etc.
            top_k: Max results to return

        Returns:
            List of RawMessageDTO with raw message data
        """
        # Build sender_id from filter_values
        # If user_id is provided, use it as sender_id filter
        # Use MAGIC_ALL when user_id is None to skip sender_id filtering
        user_id = filter_values.get("user_id")
        sender_id = user_id if user_id is not None else MAGIC_ALL

        # Get group_ids from filter_values
        group_ids = filter_values.get("group_ids")

        # Determine limit
        limit = top_k if top_k > 0 else DEFAULT_TOP_K

        try:
            # Fetch pending messages using RawMessageService
            # This returns messages with sync_status in [-1, 0] (pending/accumulating)
            pending_messages = await self.raw_message_service.get_pending_messages(
                sender_id=sender_id, group_ids=group_ids, limit=limit
            )

            logger.info(
                "[_search_raw_message] Found %d raw messages: user_id=%s, group_ids=%s",
                len(pending_messages),
                user_id,
                group_ids,
            )

            return pending_messages

        except Exception as e:
            logger.error(
                "Failed to search raw messages: sender_id=%s, group_ids=%s, error=%s",
                sender_id,
                group_ids,
                e,
            )
            return []

    async def _search_agentic_episodic_memory(
        self, query: str, filter_values: Dict[str, Any], top_k: int
    ) -> List[SearchEpisodeItem]:
        """Search episodic memories using agentic retrieval (LLM-guided multi-round).

        Delegates to MemoryManager.retrieve_mem_agentic which handles:
        - Round 1: Hybrid search
        - Rerank + LLM sufficiency check
        - Round 2 (if needed): Multi-query generation and parallel search
        - Final rerank and merge

        Args:
            query: Original search query text (passed through to embedding/rerank/LLM)
            filter_values: Filter values dict
            top_k: Max results to return

        Returns:
            List of SearchEpisodeItem with agentic retrieval results
        """
        # Build RetrieveMemRequest for MemoryManager
        retrieve_request = RetrieveMemRequest(
            query=query,
            user_id=filter_values.get("user_id"),
            group_ids=filter_values.get("group_ids"),
            memory_types=[MemoryType.EPISODIC_MEMORY],
            top_k=top_k if top_k > 0 else -1,
            retrieve_method=RetrieveMethod.AGENTIC,
            radius=None,
        )

        # Call MemoryManager's agentic retrieval
        response = await self.memory_manager.retrieve_mem_agentic(retrieve_request)

        # Collect IDs and scores from MemoryManager results
        memory_items: List[Dict[str, Any]] = []
        for memory in response.memories:
            if isinstance(memory, dict):
                memory_items.append(memory)
            else:
                memory_items.append(
                    {
                        "id": getattr(memory, "id", None),
                        "score": getattr(memory, "score", 0),
                    }
                )

        # Backfill from MongoDB to get complete display fields
        episode_ids = [m.get("id") for m in memory_items if m.get("id")]
        episodes_dict = await self._fetch_episodic_memories_by_ids(episode_ids)

        # Convert to SearchEpisodeItem with MongoDB data
        results: List[SearchEpisodeItem] = []
        for memory_item in memory_items:
            episode_id = memory_item.get("id", "")
            ep_doc = episodes_dict.get(episode_id)
            score = memory_item.get("score", 0)

            results.append(
                SearchEpisodeItem(
                    id=episode_id,
                    user_id=ep_doc.user_id if ep_doc else memory_item.get("user_id"),
                    group_id=ep_doc.group_id if ep_doc else memory_item.get("group_id"),
                    session_id=(
                        ep_doc.session_id if ep_doc else memory_item.get("session_id")
                    ),
                    timestamp=(
                        ep_doc.timestamp
                        if ep_doc
                        else self._parse_timestamp(memory_item.get("timestamp"))
                    ),
                    participants=(
                        ep_doc.participants
                        if ep_doc
                        else memory_item.get("participants")
                    ),
                    summary=ep_doc.summary if ep_doc else None,
                    subject=ep_doc.subject if ep_doc else None,
                    episode=ep_doc.episode if ep_doc else memory_item.get("episode"),
                    type=ep_doc.type if ep_doc else memory_item.get("type"),
                    parent_type=(
                        ep_doc.parent_type if ep_doc else memory_item.get("parent_type")
                    ),
                    parent_id=(
                        ep_doc.parent_id if ep_doc else memory_item.get("parent_id")
                    ),
                    score=score,
                )
            )

        return results

    async def _search_episodic_memory_hierarchical(
        self,
        query: str,
        query_vector: Optional[List[float]],
        filter_values: Dict[str, Any],
        top_k: int,
        radius: Optional[float],
    ) -> List[SearchEpisodeItem]:
        """Hierarchical retrieval for episodic_memory via DI-registered service, fallback to builtin hybrid."""
        try:
            handler = get_bean("search_method_mrag")
        except Exception:
            logger.warning(
                "Hierarchical search service not available, fallback to builtin hybrid"
            )
            query_words = self._build_query_words(query)
            date_range = {}
            if filter_values.get("start_time"):
                date_range["gte"] = filter_values["start_time"].isoformat()
            if filter_values.get("end_time"):
                date_range["lte"] = filter_values["end_time"].isoformat()
            return await self._search_episodic_memory(
                query=query,
                query_words=query_words,
                query_vector=query_vector,
                method="hybrid",
                filter_values=filter_values,
                date_range=date_range,
                top_k=top_k,
                radius=radius,
            )

        stage_start = time.perf_counter()
        response = await handler.search(
            query=query,
            query_vector=query_vector,
            memory_types=["episodic_memory"],
            filter_values=filter_values,
            top_k=top_k,
            radius=radius,
            include_original_data=False,
        )
        record_retrieve_stage(
            retrieve_method='hybrid',
            stage='hierarchical_search',
            memory_type=MemoryType.EPISODIC_MEMORY.value,
            duration_seconds=time.perf_counter() - stage_start,
        )
        return response.episodes

    async def _search_extended_method(
        self,
        method: str,
        query: str,
        memory_types: List[str],
        filter_values: Dict[str, Any],
        top_k: int,
        radius: Optional[float],
        include_original_data: bool,
    ) -> SearchMemoriesResponseData:
        """DI extension point for custom search methods.

        Looks up a bean named 'search_method_{method}' from the DI container
        and delegates the search to it.

        Args:
            method: Extended method name (e.g., 'mrag')
            query: Search query text
            memory_types: List of memory types to search
            filter_values: Extracted filter values
            top_k: Max results
            radius: Similarity threshold
            include_original_data: Whether to include original data

        Returns:
            SearchMemoriesResponseData from the extended handler

        Raises:
            ValueError: If the method handler is not found
        """
        try:
            handler = get_bean(f"search_method_{method}")
        except Exception:
            raise ValueError(f"Unknown search method: {method}")

        query_vector = await self._get_query_vector(query, retrieve_method=method)
        return await handler.search(
            query=query,
            query_vector=query_vector,
            memory_types=memory_types,
            filter_values=filter_values,
            top_k=top_k,
            radius=radius,
            include_original_data=include_original_data,
        )

    async def _fetch_original_data(
        self, episodes: List[SearchEpisodeItem], profiles: List[SearchProfileItem]
    ) -> Dict[str, Dict[str, Any]]:
        """Fetch full original data from MongoDB."""
        original_data = {"episodes": {}, "profiles": {}}

        # Collect IDs
        episode_ids = [e.id for e in episodes if e.id]

        # Fetch from MongoDB via repository
        if episode_ids:
            episodes_data = await self.episodic_raw_repo.find_by_ids(episode_ids)
            for ep in episodes_data:
                original_data["episodes"][str(ep.id)] = ep.model_dump(mode="json")

        # Profile search returns item-level results (Milvus entity IDs),
        # not MongoDB document IDs, so original_data backfill is skipped.

        return original_data

    # ------------------------------------------------------------------
    # Agent Case search
    # ------------------------------------------------------------------

    async def _fetch_agent_cases_by_ids(
        self, case_ids: List[str]
    ) -> Dict[str, AgentCaseRecord]:
        """Batch fetch AgentCaseRecords from MongoDB by IDs."""
        if not case_ids:
            return {}
        docs = await self.agent_case_raw_repo.get_by_ids(case_ids)
        return {str(d.id): d for d in docs}

    @staticmethod
    def _agent_case_doc_to_item(
        doc: AgentCaseRecord, score: Optional[float] = None
    ) -> SearchAgentCaseItem:
        """Convert AgentCaseRecord to SearchAgentCaseItem DTO."""
        return SearchAgentCaseItem(
            id=str(doc.id),
            user_id=doc.user_id,
            group_id=doc.group_id,
            session_id=doc.session_id,
            timestamp=doc.timestamp,
            task_intent=doc.task_intent or "",
            approach=doc.approach or "",
            quality_score=doc.quality_score,
            key_insight=doc.key_insight or "",
            parent_type=doc.parent_type,
            parent_id=doc.parent_id,
            score=score,
        )

    async def _search_agent_cases(
        self,
        query: str,
        query_words: List[str],
        query_vector: Optional[List[float]],
        method: str,
        filter_values: Dict[str, Any],
        date_range: Dict[str, str],
        top_k: int,
        radius: Optional[float],
    ) -> List[SearchAgentCaseItem]:
        """Search agent cases using keyword / vector / hybrid / rrf."""
        results: List[SearchAgentCaseItem] = []
        limit = _compute_recall_limit(top_k, apply_multiplier=True)

        agent_case_mt = MemoryType.AGENT_CASE.value
        with timed("agent_case_search"):
            if method == "keyword":
                stage_start = time.perf_counter()
                hits = await self.agent_case_es_repo.multi_search(
                    query=query_words,
                    user_id=filter_values["user_id"],
                    group_ids=filter_values["group_ids"],
                    session_id=filter_values["session_id"],
                    date_range=date_range,
                    size=limit,
                )
                record_retrieve_stage(
                    retrieve_method=method,
                    stage='keyword_search',
                    memory_type=agent_case_mt,
                    duration_seconds=time.perf_counter() - stage_start,
                )
                case_ids = [
                    h.get("_source", {}).get("id")
                    for h in hits
                    if h.get("_source", {}).get("id")
                ]
                cases_dict = await self._fetch_agent_cases_by_ids(case_ids)
                for hit in hits:
                    case_id = hit.get("_source", {}).get("id")
                    doc = cases_dict.get(case_id)
                    if doc:
                        results.append(
                            self._agent_case_doc_to_item(doc, score=hit.get("_score"))
                        )

            elif method == "vector" and query_vector:
                stage_start = time.perf_counter()
                search_results = await self.agent_case_milvus_repo.vector_search(
                    query_vector=query_vector,
                    user_id=filter_values["user_id"],
                    group_ids=filter_values["group_ids"],
                    session_id=filter_values["session_id"],
                    start_time=filter_values["start_time"],
                    end_time=filter_values["end_time"],
                    limit=limit,
                    radius=radius,
                )
                record_retrieve_stage(
                    retrieve_method=method,
                    stage='vector_search',
                    memory_type=agent_case_mt,
                    duration_seconds=time.perf_counter() - stage_start,
                )
                case_ids = [r.get("id") for r in search_results if r.get("id")]
                cases_dict = await self._fetch_agent_cases_by_ids(case_ids)
                for res in search_results:
                    doc = cases_dict.get(res.get("id"))
                    if doc:
                        results.append(
                            self._agent_case_doc_to_item(doc, score=res.get("score"))
                        )

            elif method == "hybrid" and query_vector:
                # Hybrid: keyword + vector, score fusion (no reranker)
                recall_limit = _compute_recall_limit(
                    top_k, apply_multiplier=0 < top_k <= HYBRID_TOP_K_THRESHOLD
                )
                stage_start = time.perf_counter()
                keyword_hits = await self.agent_case_es_repo.multi_search(
                    query=query_words,
                    user_id=filter_values["user_id"],
                    group_ids=filter_values["group_ids"],
                    session_id=filter_values["session_id"],
                    date_range=date_range,
                    size=recall_limit,
                )
                record_retrieve_stage(
                    retrieve_method=method,
                    stage='keyword_search',
                    memory_type=agent_case_mt,
                    duration_seconds=time.perf_counter() - stage_start,
                )
                stage_start = time.perf_counter()
                vector_results = await self.agent_case_milvus_repo.vector_search(
                    query_vector=query_vector,
                    user_id=filter_values["user_id"],
                    group_ids=filter_values["group_ids"],
                    session_id=filter_values["session_id"],
                    start_time=filter_values["start_time"],
                    end_time=filter_values["end_time"],
                    limit=recall_limit,
                    radius=max(AGENT_MEMORY_MILVUS_RADIUS, radius or 0.0),
                )
                record_retrieve_stage(
                    retrieve_method=method,
                    stage='vector_search',
                    memory_type=agent_case_mt,
                    duration_seconds=time.perf_counter() - stage_start,
                )
                # Fuse scores: normalize BM25 into vector score range
                vec_pairs = [
                    (self._extract_hit_id(vr), vr.get("score", 0.0))
                    for vr in vector_results
                    if self._extract_hit_id(vr)
                ]
                kw_pairs = [
                    (self._extract_hit_id(h), h.get("_score", 0.0))
                    for h in keyword_hits
                    if self._extract_hit_id(h)
                ]
                scored = vector_anchored_fusion(vec_pairs, kw_pairs)
                final_top = top_k if top_k > 0 else DEFAULT_TOP_K
                scored = scored[:final_top]
                # Backfill from MongoDB
                case_ids = [doc_id for doc_id, _ in scored]
                cases_dict = await self._fetch_agent_cases_by_ids(case_ids)
                for doc_id, score in scored:
                    doc = cases_dict.get(doc_id)
                    if doc:
                        results.append(self._agent_case_doc_to_item(doc, score=score))

            elif method == "agentic":
                results = await self._search_agentic_agent_cases(
                    query=query, filter_values=filter_values, top_k=top_k
                )

        return results

    # ------------------------------------------------------------------
    # Agent Skill search
    # ------------------------------------------------------------------

    async def _fetch_agent_skills_by_ids(
        self, skill_ids: List[str]
    ) -> Dict[str, AgentSkillRecord]:
        """Batch fetch AgentSkillRecords from MongoDB by IDs.

        Also filters out retired skills (confidence below threshold).
        """
        if not skill_ids:
            return {}
        retire_confidence = DEFAULT_MEMORIZE_CONFIG.skill_retire_confidence
        docs = await self.agent_skill_raw_repo.find_by_ids(
            skill_ids, min_confidence=retire_confidence
        )
        return {str(d.id): d for d in docs}

    @staticmethod
    def _agent_skill_doc_to_item(
        doc: AgentSkillRecord, score: Optional[float] = None
    ) -> SearchAgentSkillItem:
        """Convert AgentSkillRecord to SearchAgentSkillItem DTO."""
        return SearchAgentSkillItem(
            id=str(doc.id),
            cluster_id=doc.cluster_id,
            user_id=doc.user_id,
            group_id=doc.group_id,
            name=doc.name,
            description=doc.description,
            content=doc.content,
            confidence=doc.confidence,
            maturity_score=doc.maturity_score,
            source_case_ids=doc.source_case_ids or [],
            score=score,
        )

    async def _search_agent_skills(
        self,
        query: str,
        query_words: List[str],
        query_vector: Optional[List[float]],
        method: str,
        filter_values: Dict[str, Any],
        top_k: int,
        radius: Optional[float],
    ) -> List[SearchAgentSkillItem]:
        """Search agent skills using keyword / vector / hybrid / rrf.

        Note: Unlike episodic_memory and agent_case, agent_skill has no
        date_range filtering — skills are aggregated artefacts without
        a business timestamp, similar to profile.
        """
        results: List[SearchAgentSkillItem] = []
        limit = _compute_recall_limit(top_k, apply_multiplier=True)

        agent_skill_mt = MemoryType.AGENT_SKILL.value
        with timed("agent_skill_search"):
            retire_confidence = DEFAULT_MEMORIZE_CONFIG.skill_retire_confidence

            if method == "keyword":
                stage_start = time.perf_counter()
                hits = await self.agent_skill_es_repo.multi_search(
                    query=query_words,
                    user_id=filter_values["user_id"],
                    group_ids=filter_values["group_ids"],
                    size=limit,
                    confidence_threshold=retire_confidence,
                )
                record_retrieve_stage(
                    retrieve_method=method,
                    stage='keyword_search',
                    memory_type=agent_skill_mt,
                    duration_seconds=time.perf_counter() - stage_start,
                )
                skill_ids = [
                    h.get("_source", {}).get("id")
                    for h in hits
                    if h.get("_source", {}).get("id")
                ]
                skills_dict = await self._fetch_agent_skills_by_ids(skill_ids)
                for hit in hits:
                    skill_id = hit.get("_source", {}).get("id")
                    doc = skills_dict.get(skill_id)
                    if doc:
                        results.append(
                            self._agent_skill_doc_to_item(doc, score=hit.get("_score"))
                        )

            elif method == "vector" and query_vector:
                stage_start = time.perf_counter()
                search_results = await self.agent_skill_milvus_repo.vector_search(
                    query_vector=query_vector,
                    user_id=filter_values["user_id"],
                    group_ids=filter_values["group_ids"],
                    limit=limit,
                    radius=radius,
                    confidence_threshold=retire_confidence,
                )
                record_retrieve_stage(
                    retrieve_method=method,
                    stage='vector_search',
                    memory_type=agent_skill_mt,
                    duration_seconds=time.perf_counter() - stage_start,
                )
                skill_ids = [r.get("id") for r in search_results if r.get("id")]
                skills_dict = await self._fetch_agent_skills_by_ids(skill_ids)
                for res in search_results:
                    doc = skills_dict.get(res.get("id"))
                    if doc:
                        results.append(
                            self._agent_skill_doc_to_item(doc, score=res.get("score"))
                        )

            elif method == "hybrid" and query_vector:
                # Hybrid: keyword + vector + rerank
                recall_limit = _compute_recall_limit(
                    top_k, apply_multiplier=0 < top_k <= HYBRID_TOP_K_THRESHOLD
                )
                stage_start = time.perf_counter()
                keyword_hits = await self.agent_skill_es_repo.multi_search(
                    query=query_words,
                    user_id=filter_values["user_id"],
                    group_ids=filter_values["group_ids"],
                    size=recall_limit,
                    confidence_threshold=retire_confidence,
                )
                record_retrieve_stage(
                    retrieve_method=method,
                    stage='keyword_search',
                    memory_type=agent_skill_mt,
                    duration_seconds=time.perf_counter() - stage_start,
                )
                stage_start = time.perf_counter()
                vector_results = await self.agent_skill_milvus_repo.vector_search(
                    query_vector=query_vector,
                    user_id=filter_values["user_id"],
                    group_ids=filter_values["group_ids"],
                    limit=recall_limit,
                    radius=max(AGENT_MEMORY_MILVUS_RADIUS, radius or 0.0),
                    confidence_threshold=retire_confidence,
                )
                record_retrieve_stage(
                    retrieve_method=method,
                    stage='vector_search',
                    memory_type=agent_skill_mt,
                    duration_seconds=time.perf_counter() - stage_start,
                )
                # Tag memory_type for rerank text extraction
                for h in keyword_hits:
                    h["memory_type"] = MemoryType.AGENT_SKILL.value
                for h in vector_results:
                    h["memory_type"] = MemoryType.AGENT_SKILL.value
                # Merge and deduplicate
                seen_ids = {self._extract_hit_id(h) for h in keyword_hits} - {None}
                merged_hits = keyword_hits + [
                    h for h in vector_results if self._extract_hit_id(h) not in seen_ids
                ]
                # Rerank
                stage_start = time.perf_counter()
                rerank_service = get_rerank_service()
                rerank_top_k = top_k if top_k > 0 else DEFAULT_TOP_K
                reranked_hits = await rerank_service.rerank_memories(
                    query=query,
                    hits=merged_hits,
                    top_k=rerank_top_k,
                    instruction="Determine whether the skill's methodology and domain are applicable to the query, preferring same-domain skills with directly relevant steps."
                )
                rerank_ms = (time.perf_counter() - stage_start) * 1000
                record_retrieve_stage(
                    retrieve_method=method,
                    stage='rerank',
                    memory_type=agent_skill_mt,
                    duration_seconds=rerank_ms / 1000,
                )
                logger.info(
                    "[RERANK_DIAG] memory_type=%s in=%d out=%d top_k=%d took=%.1fms",
                    agent_skill_mt,
                    len(merged_hits),
                    len(reranked_hits),
                    rerank_top_k,
                    rerank_ms,
                )
                # Backfill from MongoDB
                skill_ids = [
                    self._extract_hit_id(h)
                    for h in reranked_hits
                    if self._extract_hit_id(h)
                ]
                skills_dict = await self._fetch_agent_skills_by_ids(skill_ids)
                for hit in reranked_hits:
                    skill_id = self._extract_hit_id(hit)
                    doc = skills_dict.get(skill_id)
                    if doc:
                        results.append(
                            self._agent_skill_doc_to_item(
                                doc, score=hit.get("rerank_score", hit.get("score", 0))
                            )
                        )

            elif method == "agentic":
                results = await self._search_agentic_agent_skills(
                    query=query, filter_values=filter_values, top_k=top_k
                )

        if results and DEFAULT_MEMORIZE_CONFIG.enable_skill_llm_verify:
            results = await self._verify_skill_relevance(query, results)

        return results

    async def _verify_skill_relevance(
        self, query: str, skills: List[SearchAgentSkillItem]
    ) -> List[SearchAgentSkillItem]:
        """Use LLM to post-verify whether retrieved skills are relevant to the query."""
        import json
        from common_utils.json_utils import parse_json_response
        from memory_layer.prompts import get_prompt_by
        from memory_layer.llm.llm_provider import build_default_provider

        if not skills or not query:
            return skills

        skills_for_prompt = [
            {
                "index": i,
                "name": skill.name or "",
                "description": skill.description or "",
                "content": skill.content or "",
            }
            for i, skill in enumerate(skills)
        ]

        prompt_template = get_prompt_by("AGENT_SKILL_RELEVANCE_VERIFY_PROMPT")
        prompt = prompt_template.format(
            query=query, skills_json=json.dumps(skills_for_prompt, ensure_ascii=False)
        )

        try:
            llm_provider = build_default_provider()
            response_text = await llm_provider.generate(
                prompt, temperature=0.0, response_format={"type": "json_object"}
            )

            result = parse_json_response(response_text)
            score_map = {
                item["index"]: item.get("score", 0.0)
                for item in result.get("results", [])
            }

            scored = []
            for i, skill in enumerate(skills):
                relevance_score = score_map.get(i, 0.0)
                if relevance_score >= 0.4:
                    skill.score = relevance_score
                    scored.append(skill)

            scored.sort(key=lambda s: s.score, reverse=True)

            logger.info(
                "Skill relevance verification: %d/%d skills passed (threshold=0.4) for query: %s",
                len(scored),
                len(skills),
                query[:60],
            )

            return scored

        except Exception as e:
            logger.warning(
                "Skill relevance verification failed, returning all results: %s", e
            )
            return skills

    # ------------------------------------------------------------------
    # Agentic retrieval for agent memory types
    # ------------------------------------------------------------------

    async def _search_agentic_agent_cases(
        self, query: str, filter_values: Dict[str, Any], top_k: int
    ) -> List[SearchAgentCaseItem]:
        """Search agent cases using agentic retrieval (LLM-guided multi-round)."""
        retrieve_request = RetrieveMemRequest(
            query=query,
            user_id=filter_values.get("user_id"),
            group_ids=filter_values.get("group_ids"),
            memory_types=[MemoryType.AGENT_CASE],
            top_k=top_k if top_k > 0 else -1,
            retrieve_method=RetrieveMethod.AGENTIC,
            radius=None,
        )

        response = await self.memory_manager.retrieve_mem_agentic(retrieve_request)

        results: List[SearchAgentCaseItem] = []
        case_ids = [
            getattr(m, "id", None) or (m.get("id") if isinstance(m, dict) else None)
            for m in response.memories
        ]
        case_ids = [cid for cid in case_ids if cid]
        cases_dict = await self._fetch_agent_cases_by_ids(case_ids)

        for memory in response.memories:
            if isinstance(memory, dict):
                case_id = memory.get("id")
                score = memory.get("score", 0)
            else:
                case_id = getattr(memory, "id", None)
                score = getattr(memory, "score", 0)

            doc = cases_dict.get(case_id)
            if doc:
                results.append(self._agent_case_doc_to_item(doc, score=score))

        return results

    async def _search_agentic_agent_skills(
        self, query: str, filter_values: Dict[str, Any], top_k: int
    ) -> List[SearchAgentSkillItem]:
        """Search agent skills using agentic retrieval (LLM-guided multi-round)."""
        retrieve_request = RetrieveMemRequest(
            query=query,
            user_id=filter_values.get("user_id"),
            group_ids=filter_values.get("group_ids"),
            memory_types=[MemoryType.AGENT_SKILL],
            top_k=top_k if top_k > 0 else -1,
            retrieve_method=RetrieveMethod.AGENTIC,
            radius=None,
        )

        response = await self.memory_manager.retrieve_mem_agentic(retrieve_request)

        results: List[SearchAgentSkillItem] = []
        skill_ids = [
            getattr(m, "id", None) or (m.get("id") if isinstance(m, dict) else None)
            for m in response.memories
        ]
        skill_ids = [sid for sid in skill_ids if sid]
        skills_dict = await self._fetch_agent_skills_by_ids(skill_ids)

        for memory in response.memories:
            if isinstance(memory, dict):
                skill_id = memory.get("id")
                score = memory.get("score", 0)
            else:
                skill_id = getattr(memory, "id", None)
                score = getattr(memory, "score", 0)

            doc = skills_dict.get(skill_id)
            if doc:
                results.append(self._agent_skill_doc_to_item(doc, score=score))

        return results

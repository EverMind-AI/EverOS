from __future__ import annotations

from typing import Any, List, Optional
import logging
import asyncio

from datetime import datetime, timedelta
import jieba
import numpy as np
import time
from typing import Dict, Any
from dataclasses import dataclass

from api_specs import memory_types
from api_specs.memory_types import (
    BaseMemory,
    EpisodeMemory,
    AtomicFact,
    Foresight,
    AgentCase,
    AgentSkill,
    RawDataType,
    ParentType,
)
from biz_layer.mem_memorize import memorize
from api_specs.dtos import MemorizeRequest
from .get_mem_service import GetMemoryService
from api_specs.dtos import (
    RawMessageDTO,
    ProfileSearchItem,
    RetrieveMemRequest,
    RetrieveMemResponse,
)
from api_specs.dtos.memory import GetMemRequest, GetMemResponse
from api_specs.memory_models import Metadata, MemoryType, QueryMetadata
from core.di import get_bean_by_type
from biz_layer.retrieve_constants import (
    DEFAULT_MILVUS_SIMILARITY_THRESHOLD,
    DEFAULT_RERANK_SCORE_THRESHOLD,
    DEFAULT_RECALL_MULTIPLIER,
    DEFAULT_TOPK_LIMIT,
)
from infra_layer.adapters.out.search.repository.episodic_memory_es_repository import (
    EpisodicMemoryEsRepository,
)
from infra_layer.adapters.out.search.repository.foresight_es_repository import (
    ForesightEsRepository,
)
from infra_layer.adapters.out.search.repository.atomic_fact_es_repository import (
    AtomicFactEsRepository,
)
from infra_layer.adapters.out.search.repository.agent_case_es_repository import (
    AgentCaseEsRepository,
)
from infra_layer.adapters.out.search.repository.agent_skill_es_repository import (
    AgentSkillEsRepository,
)
from core.observation.tracing.decorators import trace_logger
from core.observation.stage_timer import timed, timed_parallel
from core.nlp.stopwords_utils import filter_stopwords
from common_utils.datetime_utils import (
    from_iso_format,
    get_now_with_timezone,
    to_iso_format,
)
from infra_layer.adapters.out.persistence.repository.memcell_raw_repository import (
    MemCellRawRepository,
)
from service.raw_message_service import RawMessageService
from infra_layer.adapters.out.persistence.document.memory.memcell import DataTypeEnum
from infra_layer.adapters.out.persistence.document.memory.user_profile import (
    UserProfile,
)
from infra_layer.adapters.out.search.repository.episodic_memory_milvus_repository import (
    EpisodicMemoryMilvusRepository,
)
from infra_layer.adapters.out.search.repository.foresight_milvus_repository import (
    ForesightMilvusRepository,
)
from infra_layer.adapters.out.search.repository.atomic_fact_milvus_repository import (
    AtomicFactMilvusRepository,
)
from infra_layer.adapters.out.search.repository.agent_case_milvus_repository import (
    AgentCaseMilvusRepository,
)
from infra_layer.adapters.out.search.repository.agent_skill_milvus_repository import (
    AgentSkillMilvusRepository,
)
from .vectorize_service import get_vectorize_service
from .rerank_service import get_rerank_service
from .profile_search_service import (
    get_profile_search_service,
    PROFILE_RECALL_THRESHOLD,
    PROFILE_DEFAULT_TOPK,
)
from api_specs.memory_models import MemoryType, RetrieveMethod
from agentic_layer.metrics.retrieve_metrics import (
    record_retrieve_stage,
    record_retrieve_error,
)
from memory_layer.llm.llm_provider import build_default_provider
from agentic_layer.agentic_utils import (
    AgenticConfig,
    check_sufficiency,
    generate_multi_queries,
)

logger = logging.getLogger(__name__)


# MemoryType -> ES Repository mapping
ES_REPO_MAP = {
    MemoryType.FORESIGHT: ForesightEsRepository,
    MemoryType.ATOMIC_FACT: AtomicFactEsRepository,
    MemoryType.EPISODIC_MEMORY: EpisodicMemoryEsRepository,
    MemoryType.AGENT_CASE: AgentCaseEsRepository,
    MemoryType.AGENT_SKILL: AgentSkillEsRepository,
}


@dataclass
class AtomicFactCandidate:
    """Atomic Fact candidate object (used for retrieval from atomic_fact)"""

    event_id: str
    user_id: str
    group_id: str
    timestamp: datetime
    episode: str  # atomic_fact content
    summary: str
    subject: str
    extend: dict  # contains embedding


class MemoryManager:
    """Unified memory interface.

    Provides the following main functions:
    - memorize: Accept raw data and persistently store
    - get_mem: (v1) Get memories with structured filters DSL
    - retrieve_mem: Memory reading based on prompt-based retrieval methods
    """

    def __init__(self) -> None:
        # Get memory service instances
        self._get_service: GetMemoryService = get_bean_by_type(GetMemoryService)
        self._raw_message_service: RawMessageService = get_bean_by_type(
            RawMessageService
        )

        logger.info(
            "MemoryManager initialized with get_memory_service and retrieve_mem_service"
        )

    # --------- Write path (raw data -> memorize) ---------
    @trace_logger(operation_name="agentic_layer memory storage")
    async def memorize(self, memorize_request: MemorizeRequest) -> int:
        """Memorize a heterogeneous list of raw items.

        Accepts list[Any], where each item can be one of the typed raw dataclasses
        (ChatRawData / EmailRawData / MemoRawData / LincDocRawData) or any dict-like
        object. Each item is stored as a MemoryCell with a synthetic key.

        Returns:
            int: Number of memories extracted (0 if no boundary detected)
        """
        count = await memorize(memorize_request)
        return count

    # --------- Read path (v1 filters DSL -> get_mem) ---------
    @trace_logger(operation_name="agentic_layer v1 memory get")
    async def get_mem(self, request: GetMemRequest) -> GetMemResponse:
        """Get memories using v1 structured filters DSL.

        Args:
            request: GetMemRequest containing filters, memory_type, pagination, sort

        Returns:
            GetMemResponse containing query results
        """
        logger.debug(
            "get_mem called with request: memory_type=%s, page=%s, page_size=%s",
            request.memory_type,
            request.page,
            request.page_size,
        )

        response = await self._get_service.find_memories(
            filters=request.filters,
            memory_type=request.memory_type,
            page=request.page,
            page_size=request.page_size,
            rank_by=request.rank_by,
            rank_order=request.rank_order,
        )

        logger.debug(
            "get_mem returned total_count=%s, count=%s",
            response.total_count,
            response.count,
        )
        return response

    # Memory reading based on retrieve_method, including static and dynamic memory
    @trace_logger(operation_name="agentic_layer memory retrieval")
    async def retrieve_mem(
        self, retrieve_mem_request: 'RetrieveMemRequest'
    ) -> RetrieveMemResponse:
        """Retrieve memory data, dispatching to different retrieval methods based on retrieve_method

        Args:
            retrieve_mem_request: RetrieveMemRequest containing retrieval parameters

        Returns:
            RetrieveMemResponse containing retrieval results
        """
        try:
            # Validate request parameters
            if not retrieve_mem_request:
                raise ValueError("retrieve_mem_request is required for retrieve_mem")

            # Get memory types from request (defaults already applied in converter)
            memory_types = retrieve_mem_request.memory_types

            # Separate profile search from other memory types
            search_profile = MemoryType.PROFILE in memory_types
            non_profile_types = [mt for mt in memory_types if mt != MemoryType.PROFILE]

            retrieve_method = retrieve_mem_request.retrieve_method

            logger.info(
                f"retrieve_mem dispatching request: user_id={retrieve_mem_request.user_id}, "
                f"retrieve_method={retrieve_method}, query={retrieve_mem_request.query}, "
                f"search_profile={search_profile}, non_profile_types={[t.value for t in non_profile_types]}"
            )

            # Task 1: Fetch pending messages
            pending_messages_task = asyncio.create_task(
                self._get_pending_messages(
                    user_id=retrieve_mem_request.user_id,
                    group_ids=retrieve_mem_request.group_ids,
                )
            )

            # Task 2: Profile search (if needed)
            profile_task = None
            if search_profile and retrieve_mem_request.query:
                profile_task = asyncio.create_task(
                    self._search_profiles(retrieve_mem_request)
                )

            # Task 3: Non-profile memory search (if needed)
            non_profile_response = None
            if non_profile_types:
                # Create a modified request with non-profile types
                non_profile_request = RetrieveMemRequest(
                    user_id=retrieve_mem_request.user_id,
                    group_ids=retrieve_mem_request.group_ids,
                    memory_types=non_profile_types,
                    top_k=retrieve_mem_request.top_k,
                    include_metadata=retrieve_mem_request.include_metadata,
                    start_time=retrieve_mem_request.start_time,
                    end_time=retrieve_mem_request.end_time,
                    query=retrieve_mem_request.query,
                    retrieve_method=retrieve_mem_request.retrieve_method,
                    radius=retrieve_mem_request.radius,
                )

                # Dispatch based on retrieval method
                match retrieve_method:
                    case RetrieveMethod.KEYWORD:
                        non_profile_response = await self.retrieve_mem_keyword(
                            non_profile_request
                        )
                    case RetrieveMethod.VECTOR:
                        non_profile_response = await self.retrieve_mem_vector(
                            non_profile_request
                        )
                    case RetrieveMethod.HYBRID:
                        non_profile_response = await self.retrieve_mem_hybrid(
                            non_profile_request
                        )
                    case RetrieveMethod.AGENTIC:
                        non_profile_response = await self.retrieve_mem_agentic(
                            non_profile_request
                        )
                    case _:
                        raise ValueError(
                            f"Unsupported retrieval method: {retrieve_method}"
                        )

            # Await profile search results
            profile_results = []
            if profile_task:
                profile_results = await profile_task

            # Await pending messages
            pending_messages = await pending_messages_task

            # Build combined response
            response = self._build_combined_response(
                profile_results=profile_results,
                non_profile_response=non_profile_response,
                retrieve_mem_request=retrieve_mem_request,
                pending_messages=pending_messages,
            )

            return response

        except Exception as e:
            logger.error(f"Error in retrieve_mem: {e}", exc_info=True)
            return RetrieveMemResponse(
                profiles=[],
                memories=[],
                total_count=0,
                has_more=False,
                query_metadata=QueryMetadata.from_request(retrieve_mem_request),
                metadata=Metadata(
                    source="retrieve_mem_service",
                    user_id=(
                        retrieve_mem_request.user_id if retrieve_mem_request else ""
                    ),
                    memory_types=[],
                ),
                pending_messages=[],
            )

    async def _search_profiles(
        self, retrieve_mem_request: 'RetrieveMemRequest'
    ) -> List[ProfileSearchItem]:
        """
        Search user profiles using ProfileSearchService.

        Returns profile items without reranking.
        """
        try:
            profile_service = get_profile_search_service()

            # Use configured default if top_k is not positive, otherwise use top_k value
            profile_top_k = (
                PROFILE_DEFAULT_TOPK
                if retrieve_mem_request.top_k <= 0
                else retrieve_mem_request.top_k
            )

            # Use radius as score threshold if provided, otherwise use configured default
            score_threshold = PROFILE_RECALL_THRESHOLD

            result = await profile_service.search_profiles(
                query=retrieve_mem_request.query or "",
                user_id=retrieve_mem_request.user_id or "",
                group_id=(
                    retrieve_mem_request.group_ids[0]
                    if retrieve_mem_request.group_ids
                    else ""
                ),
                top_k=profile_top_k,
                score_threshold=score_threshold,
            )

            # Convert to ProfileSearchItem list
            profiles = []
            for item in result.get("profiles", []):
                profile_item = ProfileSearchItem(
                    item_type=item.get("item_type", ""),
                    category=item.get("category"),
                    trait_name=item.get("trait_name"),
                    description=item.get("description", ""),
                    score=item.get("score", 0.0),
                )
                profiles.append(profile_item)

            logger.debug(f"Profile search returned {len(profiles)} items")
            return profiles

        except Exception as e:
            logger.error(f"Error in _search_profiles: {e}", exc_info=True)
            return []

    def _build_combined_response(
        self,
        profile_results: List[ProfileSearchItem],
        non_profile_response: Optional[RetrieveMemResponse],
        retrieve_mem_request: 'RetrieveMemRequest',
        pending_messages: List[RawMessageDTO],
    ) -> RetrieveMemResponse:
        """
        Build combined response from profile and non-profile search results.
        """
        user_id = retrieve_mem_request.user_id or ""
        retrieve_method = retrieve_mem_request.retrieve_method.value

        # Get memories from non-profile response
        memories = []

        if non_profile_response:
            memories = non_profile_response.memories

        # Calculate total count
        total_count = len(profile_results) + len(memories)

        # Build memory_types list
        memory_types_searched = []
        if profile_results:
            memory_types_searched.append(MemoryType.PROFILE.value)
        if memories:
            memory_types_searched.append(MemoryType.EPISODIC_MEMORY.value)

        return RetrieveMemResponse(
            profiles=profile_results,
            memories=memories,
            total_count=total_count,
            has_more=False,
            query_metadata=QueryMetadata.from_request(retrieve_mem_request),
            metadata=Metadata(
                source=retrieve_method,
                user_id=user_id,
                memory_types=memory_types_searched,
            ),
            pending_messages=pending_messages,
        )

    async def _get_pending_messages(
        self, user_id: Optional[str] = None, group_ids: Optional[List[str]] = None
    ) -> List[RawMessageDTO]:
        """
        Get pending (unconsumed) messages from RawMessageService.

        Fetches cached memory data that hasn't been consumed yet (sync_status=-1 or 0).

        Args:
            user_id: User ID filter (from retrieve_request, mapped to sender_id)
            group_ids: List of Group IDs to filter (None means all groups)

        Returns:
            List of RawMessageDTO objects
        """
        try:
            result = await self._raw_message_service.get_pending_messages(
                sender_id=user_id, group_ids=group_ids, limit=1000
            )

            logger.debug(
                f"Retrieved {len(result)} pending messages: "
                f"user_id={user_id}, group_ids={group_ids}"
            )
            return result
        except Exception as e:
            logger.error(f"Error fetching pending messages: {e}", exc_info=True)
            return []

    # Keyword retrieval method (original retrieve_mem logic)
    @trace_logger(operation_name="agentic_layer keyword memory retrieval")
    async def retrieve_mem_keyword(
        self, retrieve_mem_request: 'RetrieveMemRequest'
    ) -> RetrieveMemResponse:
        """Keyword-based memory retrieval"""
        top_k = retrieve_mem_request.top_k
        is_unlimited_mode = top_k == -1
        memory_type = (
            retrieve_mem_request.memory_types[0].value
            if retrieve_mem_request.memory_types
            else 'unknown'
        )

        try:
            hits = await self.get_keyword_search_results(
                retrieve_mem_request, retrieve_method=RetrieveMethod.KEYWORD.value
            )

            # In normal mode (top_k > 0), truncate to top_k
            # In unlimited mode, return all results (ES doesn't apply threshold filtering)
            if not is_unlimited_mode and hits:
                hits = hits[:top_k]

            return await self._to_response(hits, retrieve_mem_request)
        except Exception as e:
            logger.error(f"Error in retrieve_mem_keyword: {e}", exc_info=True)
            return await self._to_response([], retrieve_mem_request)

    async def get_keyword_search_results(
        self,
        retrieve_mem_request: 'RetrieveMemRequest',
        retrieve_method: str = RetrieveMethod.KEYWORD.value,
    ) -> List[Dict[str, Any]]:
        """Keyword search with stage-level metrics"""
        stage_start = time.perf_counter()
        memory_type = (
            retrieve_mem_request.memory_types[0].value
            if retrieve_mem_request.memory_types
            else 'unknown'
        )

        try:
            # Get parameters from Request
            if not retrieve_mem_request:
                raise ValueError("retrieve_mem_request is required for retrieve_mem")

            top_k = retrieve_mem_request.top_k
            # Calculate effective recall limit based on mode:
            # - Unlimited mode (top_k=-1): Fixed recall of DEFAULT_TOPK_LIMIT (100)
            # - Normal mode (top_k>0): Recall top_k * RECALL_MULTIPLIER for larger candidate pool
            if top_k == -1:
                effective_limit = DEFAULT_TOPK_LIMIT
            else:
                effective_limit = top_k * DEFAULT_RECALL_MULTIPLIER
            query = retrieve_mem_request.query
            user_id = retrieve_mem_request.user_id
            group_ids = retrieve_mem_request.group_ids  # List[str] or None
            start_time = retrieve_mem_request.start_time
            end_time = retrieve_mem_request.end_time
            memory_types = retrieve_mem_request.memory_types

            # Convert query string to search word list
            # Use jieba for search mode word segmentation, then filter stopwords
            if query:
                raw_words = list(jieba.cut_for_search(query))
                query_words = filter_stopwords(raw_words, min_length=2)
            else:
                query_words = []

            logger.debug(f"query_words: {query_words}")

            # Build time range filter conditions, handle None values
            date_range = {}
            if start_time is not None:
                date_range["gte"] = start_time
            if end_time is not None:
                date_range["lte"] = end_time

            mem_type = memory_types[0]

            repo_class = ES_REPO_MAP.get(mem_type)
            if not repo_class:
                logger.warning(f"Unsupported memory_type: {mem_type}")
                return []

            es_repo = get_bean_by_type(repo_class)
            logger.debug(f"Using {repo_class.__name__} for {mem_type}")

            results = await es_repo.multi_search(
                query=query_words,
                user_id=user_id,
                group_ids=group_ids,  # Pass normalized list
                size=effective_limit,
                from_=0,
                date_range=date_range,
            )

            # Mark memory_type, search_source, and unified score
            if results:
                for r in results:
                    r['memory_type'] = mem_type.value
                    r['_search_source'] = RetrieveMethod.KEYWORD.value
                    r['id'] = r.get('_id', '')  # Unify ES '_id' to 'id'
                    r['score'] = r.get('_score', 0.0)  # Unified score field

            # Record stage metrics
            record_retrieve_stage(
                retrieve_method=retrieve_method,
                stage=RetrieveMethod.KEYWORD.value,
                memory_type=memory_type,
                duration_seconds=time.perf_counter() - stage_start,
            )

            return results or []
        except Exception as e:
            record_retrieve_stage(
                retrieve_method=retrieve_method,
                stage=RetrieveMethod.KEYWORD.value,
                memory_type=memory_type,
                duration_seconds=time.perf_counter() - stage_start,
            )
            record_retrieve_error(
                retrieve_method=retrieve_method,
                stage=RetrieveMethod.KEYWORD.value,
                error_type=self._classify_retrieve_error(e),
            )
            logger.error(f"Error in get_keyword_search_results: {e}")
            raise

    # Vector-based memory retrieval
    @trace_logger(operation_name="agentic_layer vector memory retrieval")
    async def retrieve_mem_vector(
        self, retrieve_mem_request: 'RetrieveMemRequest'
    ) -> RetrieveMemResponse:
        """Vector-based memory retrieval"""
        top_k = retrieve_mem_request.top_k
        is_unlimited_mode = top_k == -1
        memory_type = (
            retrieve_mem_request.memory_types[0].value
            if retrieve_mem_request.memory_types
            else 'unknown'
        )

        try:
            hits = await self.get_vector_search_results(
                retrieve_mem_request, retrieve_method=RetrieveMethod.VECTOR.value
            )

            # In normal mode (top_k > 0), truncate to top_k
            # In unlimited mode, results are already filtered by Milvus threshold
            if not is_unlimited_mode and hits:
                hits = hits[:top_k]

            return await self._to_response(hits, retrieve_mem_request)
        except Exception as e:
            logger.error(f"Error in retrieve_mem_vector: {e}")
            return await self._to_response([], retrieve_mem_request)

    async def get_vector_search_results(
        self,
        retrieve_mem_request: 'RetrieveMemRequest',
        retrieve_method: str = RetrieveMethod.VECTOR.value,
    ) -> List[Dict[str, Any]]:
        """Vector search with stage-level metrics (embedding + milvus_search)"""
        memory_type = (
            retrieve_mem_request.memory_types[0].value
            if retrieve_mem_request.memory_types
            else 'unknown'
        )

        try:
            # Get parameters from Request
            logger.debug(
                f"get_vector_search_results called with retrieve_mem_request: {retrieve_mem_request}"
            )
            if not retrieve_mem_request:
                raise ValueError(
                    "retrieve_mem_request is required for get_vector_search_results"
                )
            query = retrieve_mem_request.query
            if not query:
                raise ValueError("query is required for retrieve_mem_vector")

            user_id = retrieve_mem_request.user_id
            group_ids = retrieve_mem_request.group_ids  # List[str] or None
            top_k = retrieve_mem_request.top_k
            # Calculate effective recall limit based on mode:
            # - Unlimited mode (top_k=-1): Fixed recall of DEFAULT_TOPK_LIMIT (100)
            # - Normal mode (top_k>0): Recall top_k * RECALL_MULTIPLIER for larger candidate pool
            if top_k == -1:
                effective_limit = DEFAULT_TOPK_LIMIT
            else:
                effective_limit = top_k * DEFAULT_RECALL_MULTIPLIER
            # Milvus similarity threshold (only applied in unlimited mode or when user specifies radius)
            effective_radius = None
            start_time = retrieve_mem_request.start_time
            end_time = retrieve_mem_request.end_time
            mem_type = retrieve_mem_request.memory_types[0]

            logger.debug(
                f"retrieve_mem_vector called with query: {query}, user_id: {user_id}, group_ids: {group_ids}, top_k: {top_k}"
            )

            # Get vectorization service
            vectorize_service = get_vectorize_service()

            # Convert query text to vector (embedding stage)
            logger.debug(f"Starting to vectorize query text: {query}")
            embedding_start = time.perf_counter()
            query_vector = await vectorize_service.get_embedding(query)
            query_vector_list = query_vector.tolist()  # Convert to list format
            record_retrieve_stage(
                retrieve_method=retrieve_method,
                stage='embedding',
                memory_type=memory_type,
                duration_seconds=time.perf_counter() - embedding_start,
            )
            logger.debug(
                f"Query text vectorization completed, vector dimension: {len(query_vector_list)}"
            )

            # Select Milvus repository based on memory type
            match mem_type:
                case MemoryType.FORESIGHT:
                    milvus_repo = get_bean_by_type(ForesightMilvusRepository)
                case MemoryType.ATOMIC_FACT:
                    milvus_repo = get_bean_by_type(AtomicFactMilvusRepository)
                case MemoryType.EPISODIC_MEMORY:
                    milvus_repo = get_bean_by_type(EpisodicMemoryMilvusRepository)
                case MemoryType.AGENT_CASE:
                    milvus_repo = get_bean_by_type(AgentCaseMilvusRepository)
                case MemoryType.AGENT_SKILL:
                    milvus_repo = get_bean_by_type(AgentSkillMilvusRepository)
                case _:
                    raise ValueError(f"Unsupported memory type: {mem_type}")

            # Handle time range filter conditions
            start_time_dt = None
            end_time_dt = None

            if start_time is not None:
                start_time_dt = (
                    from_iso_format(start_time)
                    if isinstance(start_time, str)
                    else start_time
                )

            if end_time is not None:
                if isinstance(end_time, str):
                    end_time_dt = from_iso_format(end_time)
                    # If date only format, set to end of day
                    if len(end_time) == 10:
                        end_time_dt = end_time_dt.replace(hour=23, minute=59, second=59)
                else:
                    end_time_dt = end_time

            # Handle foresight time range (only valid for foresight)
            if mem_type == MemoryType.FORESIGHT:
                if retrieve_mem_request.start_time:
                    start_time_dt = from_iso_format(retrieve_mem_request.start_time)
                if retrieve_mem_request.end_time:
                    end_time_dt = from_iso_format(retrieve_mem_request.end_time)

            # Call Milvus vector search (pass different parameters based on memory type)
            # Threshold logic:
            # - User specified radius: always use it
            # - Unlimited mode (top_k=-1): apply DEFAULT_MILVUS_SIMILARITY_THRESHOLD (0.6)
            # - Normal mode (top_k>0): no threshold filtering (rely on top_k limit)
            if retrieve_mem_request.radius is not None:
                # User specified radius, use it
                effective_radius = retrieve_mem_request.radius
            elif top_k == -1:
                # Unlimited mode: apply default Milvus threshold for quality filtering
                effective_radius = DEFAULT_MILVUS_SIMILARITY_THRESHOLD
            # else: keep None (no threshold filtering for normal top_k mode)

            milvus_start = time.perf_counter()
            if mem_type == MemoryType.FORESIGHT:
                # Foresight: supports time range and validity filtering, supports radius parameter
                search_results = await milvus_repo.vector_search(
                    query_vector=query_vector_list,
                    user_id=user_id,
                    group_ids=group_ids,  # Pass normalized list
                    start_time=start_time_dt,
                    end_time=end_time_dt,
                    limit=effective_limit,
                    score_threshold=0.0,
                    radius=effective_radius,
                )
            elif mem_type == MemoryType.AGENT_SKILL:
                # Agent skill: no timestamp filtering
                search_results = await milvus_repo.vector_search(
                    query_vector=query_vector_list,
                    user_id=user_id,
                    group_ids=group_ids,
                    limit=effective_limit,
                    score_threshold=0.0,
                    radius=effective_radius,
                )
            else:
                # Episodic memory, atomic fact, agent case: use timestamp filtering
                search_results = await milvus_repo.vector_search(
                    query_vector=query_vector_list,
                    user_id=user_id,
                    group_ids=group_ids,  # Pass normalized list
                    start_time=start_time_dt,
                    end_time=end_time_dt,
                    limit=effective_limit,
                    score_threshold=0.0,
                    radius=effective_radius,
                )
            record_retrieve_stage(
                retrieve_method=retrieve_method,
                stage='milvus_search',
                memory_type=memory_type,
                duration_seconds=time.perf_counter() - milvus_start,
            )

            for r in search_results:
                r['memory_type'] = mem_type.value
                r['_search_source'] = RetrieveMethod.VECTOR.value
                # Milvus already uses 'score', no need to rename

            return search_results
        except Exception as e:
            record_retrieve_stage(
                retrieve_method=retrieve_method,
                stage=RetrieveMethod.VECTOR.value,
                memory_type=memory_type,
                duration_seconds=time.perf_counter() - milvus_start,
            )
            record_retrieve_error(
                retrieve_method=retrieve_method,
                stage=RetrieveMethod.VECTOR.value,
                error_type=self._classify_retrieve_error(e),
            )
            logger.error(f"Error in get_vector_search_results: {e}")
            raise

    # Hybrid memory retrieval
    @trace_logger(operation_name="agentic_layer hybrid memory retrieval")
    async def retrieve_mem_hybrid(
        self, retrieve_mem_request: 'RetrieveMemRequest'
    ) -> RetrieveMemResponse:
        """Hybrid memory retrieval: keyword + vector + rerank"""
        memory_type = (
            retrieve_mem_request.memory_types[0].value
            if retrieve_mem_request.memory_types
            else 'unknown'
        )

        try:
            hits = await self._search_hybrid(
                retrieve_mem_request, retrieve_method=RetrieveMethod.HYBRID.value
            )
            return await self._to_response(hits, retrieve_mem_request)
        except Exception as e:
            logger.error(f"Error in retrieve_mem_hybrid: {e}")
            return await self._to_response([], retrieve_mem_request)

    # ================== Core Internal Methods ==================

    async def _rerank(
        self,
        query: str,
        hits: List[Dict],
        top_k: int,
        memory_type: str = 'unknown',
        retrieve_method: str = RetrieveMethod.HYBRID.value,
        instruction: str = None,
        apply_threshold: bool = False,
    ) -> List[Dict]:
        """Rerank hits using rerank service with stage metrics

        Args:
            query: Query text for reranking
            hits: List of candidate documents to rerank
            top_k: Maximum number of results to return after rerank
            memory_type: Memory type for metrics
            retrieve_method: Retrieval method for metrics
            instruction: Optional instruction for reranker
            apply_threshold: If True, filter results by DEFAULT_RERANK_SCORE_THRESHOLD
                            (used in unlimited mode to ensure quality)

        Returns:
            List of reranked documents, optionally filtered by threshold
        """
        if not hits:
            return []

        stage_start = time.perf_counter()
        try:
            result = await get_rerank_service().rerank_memories(
                query, hits, top_k, instruction=instruction
            )

            # Apply rerank threshold filtering in unlimited mode
            if apply_threshold and result:
                original_count = len(result)
                result = [
                    doc
                    for doc in result
                    if doc.get('score', 0.0) >= DEFAULT_RERANK_SCORE_THRESHOLD
                ]
                filtered_count = original_count - len(result)
                if filtered_count > 0:
                    logger.debug(
                        f"Rerank threshold filtering: {filtered_count} docs filtered "
                        f"(threshold={DEFAULT_RERANK_SCORE_THRESHOLD})"
                    )

            record_retrieve_stage(
                retrieve_method=retrieve_method,
                stage='rerank',
                memory_type=memory_type,
                duration_seconds=time.perf_counter() - stage_start,
            )
            return result
        except Exception as e:
            record_retrieve_error(
                retrieve_method=retrieve_method,
                stage='rerank',
                error_type=self._classify_retrieve_error(e),
            )
            raise

    async def _search_hybrid(
        self,
        request: 'RetrieveMemRequest',
        retrieve_method: str = RetrieveMethod.HYBRID.value,
    ) -> List[Dict]:
        """Core hybrid search: keyword + vector + rerank, returns flat list"""
        memory_type = (
            request.memory_types[0].value if request.memory_types else 'unknown'
        )
        top_k = request.top_k
        is_unlimited_mode = top_k == -1

        # Run keyword and vector search concurrently
        kw_results, vec_results = await asyncio.gather(
            self.get_keyword_search_results(request, retrieve_method=retrieve_method),
            self.get_vector_search_results(request, retrieve_method=retrieve_method),
        )
        # Deduplicate by id
        seen_ids = {h.get('id') for h in kw_results}
        merged_results = kw_results + [
            h for h in vec_results if h.get('id') not in seen_ids
        ]
        # When top_k is -1, use DEFAULT_TOPK_LIMIT for rerank
        rerank_limit = DEFAULT_TOPK_LIMIT if is_unlimited_mode else top_k

        # Apply rerank threshold filtering in unlimited mode
        reranked = await self._rerank(
            request.query,
            merged_results,
            rerank_limit,
            memory_type,
            retrieve_method,
            apply_threshold=is_unlimited_mode,
        )

        # In normal mode, truncate to top_k; in unlimited mode, return all that passed threshold
        if not is_unlimited_mode:
            return reranked[:top_k]
        return reranked

    def _classify_retrieve_error(self, error: Exception) -> str:
        """Classify error type for metrics"""
        error_str = str(error).lower()
        if 'timeout' in error_str or 'timed out' in error_str:
            return 'timeout'
        elif 'connection' in error_str or 'connect' in error_str:
            return 'connection_error'
        elif 'not found' in error_str or 'notfound' in error_str:
            return 'not_found'
        elif 'validation' in error_str or 'invalid' in error_str:
            return 'validation_error'
        else:
            return 'unknown'

    async def _to_response(
        self, hits: List[Dict], req: 'RetrieveMemRequest'
    ) -> RetrieveMemResponse:
        """Convert flat hits list to grouped RetrieveMemResponse"""
        user_id = req.user_id if req else ""
        source_type = req.retrieve_method.value
        memory_types = req.memory_types
        # Convert MemoryType enums to string values for Metadata
        memory_types_str = [mt.value for mt in memory_types] if memory_types else []

        if not hits:
            return RetrieveMemResponse(
                profiles=[],
                memories=[],
                total_count=0,
                has_more=False,
                query_metadata=QueryMetadata.from_request(req),
                metadata=Metadata(
                    source=source_type,
                    user_id=user_id or "",
                    memory_types=memory_types_str,
                ),
            )
        memories, total_count = await self.group_by_groupid_stratagy(
            hits, source_type=source_type
        )
        return RetrieveMemResponse(
            profiles=[],
            memories=memories,
            total_count=total_count,
            has_more=False,
            query_metadata=QueryMetadata.from_request(req),
            metadata=Metadata(
                source=source_type, user_id=user_id or "", memory_types=memory_types_str
            ),
        )

    # --------- Agentic retrieval (LLM-guided multi-round) ---------
    @trace_logger(operation_name="agentic_layer Agentic memory retrieval")
    async def retrieve_mem_agentic(
        self, retrieve_mem_request: 'RetrieveMemRequest'
    ) -> RetrieveMemResponse:
        """Agentic retrieval: LLM-guided multi-round intelligent retrieval

        Process: Round 1 (Hybrid) → Rerank → LLM sufficiency check → Round 2 (multi-query) → Merge → Final Rerank

        Behavior:
        - When top_k > 0: Returns exactly top_k results (or fewer if insufficient data)
        - When top_k == -1 (unlimited): Returns up to AgenticConfig limits
        - LLM sufficiency check always uses AgenticConfig.round1_rerank_top_n (default: 10)

        Design Note:
        Rerank quantity is max(config_value, top_k) to ensure:
        1. Enough results for LLM sufficiency check (config_value)
        2. Enough results to satisfy user request (top_k)
        This maintains LLM judgment quality while meeting user expectations.
        """
        req = retrieve_mem_request  # alias
        top_k = req.top_k
        is_unlimited_mode = top_k == -1
        config = AgenticConfig()
        memory_type = req.memory_types[0].value if req.memory_types else 'unknown'

        try:
            llm_provider = build_default_provider()

            logger.info(f"Agentic Retrieval: {req.query[:60]}...")

            # ========== Round 1: Hybrid search ==========
            req1 = RetrieveMemRequest(
                query=req.query,
                user_id=req.user_id,
                group_ids=req.group_ids,
                top_k=config.round1_top_n,
                memory_types=req.memory_types,
            )
            round1 = await self._search_hybrid(req1, retrieve_method='agentic')
            logger.info(f"Round 1: {len(round1)} memories")

            if not round1:
                return await self._to_response([], req)

            # ========== Rerank for LLM sufficiency check ==========
            # Calculate rerank quantity: satisfy both LLM check (10) and user request (top_k)
            if is_unlimited_mode:
                rerank_n = config.round1_rerank_top_n
            else:
                rerank_n = max(config.round1_rerank_top_n, top_k)

            reranked = await self._rerank(
                req.query,
                round1,
                rerank_n,
                memory_type,
                'agentic',
                instruction=config.reranker_instruction,
            )
            # LLM always uses fixed number for sufficiency check
            topn_for_llm = reranked[: config.round1_rerank_top_n]
            topn_pairs = [(m, m.get("score", 0)) for m in topn_for_llm]

            # ========== LLM sufficiency check ==========
            with timed("check_sufficiency"):
                is_sufficient, reasoning, missing_info = await check_sufficiency(
                    query=req.query,
                    results=topn_pairs,
                    llm_provider=llm_provider,
                    max_docs=config.round1_rerank_top_n,
                )
            logger.info(
                f"LLM: {'Sufficient' if is_sufficient else 'Insufficient'} - {reasoning}"
            )

            if is_sufficient:
                # Return results respecting user's top_k request
                final_results = reranked[:top_k] if not is_unlimited_mode else reranked
                return await self._to_response(final_results, req)

            # ========== Round 2: Multi-query ==========
            with timed("expand_queries"):
                refined_queries, _ = await generate_multi_queries(
                    original_query=req.query,
                    results=topn_pairs,
                    missing_info=missing_info,
                    llm_provider=llm_provider,
                    max_docs=config.round1_rerank_top_n,
                    num_queries=config.num_queries,
                )
            logger.info(f"Generated {len(refined_queries)} queries")

            # Parallel hybrid search
            async def do_search(q: str) -> List[Dict]:
                return await self._search_hybrid(
                    RetrieveMemRequest(
                        query=q,
                        user_id=req.user_id,
                        group_ids=req.group_ids,
                        top_k=config.round2_per_query_top_n,
                        memory_types=req.memory_types,
                    ),
                    retrieve_method='agentic',
                )

            with timed_parallel("multi_query_retrieval"):
                round2_results = await asyncio.gather(
                    *[do_search(q) for q in refined_queries], return_exceptions=True
                )
            from common_utils.async_utils import reraise_critical_errors

            reraise_critical_errors(round2_results)
            all_round2 = [
                h for r in round2_results if not isinstance(r, Exception) for h in r
            ]

            # Deduplicate and merge
            seen_ids = {m.get("id") for m in round1}
            round2_unique = [m for m in all_round2 if m.get("id") not in seen_ids]
            combined = round1 + round2_unique[: config.combined_total - len(round1)]
            logger.info(f"Combined: {len(combined)} memories")

            # ========== Final Rerank ==========
            # Calculate final rerank quantity: satisfy both config (40) and user request (top_k)
            if is_unlimited_mode:
                final_rerank_n = config.combined_total
            else:
                final_rerank_n = max(config.combined_total, top_k)

            final = await self._rerank(
                req.query,
                combined,
                final_rerank_n,
                memory_type,
                'agentic',
                instruction=config.reranker_instruction,
            )

            # Return results respecting user's top_k request
            final_results = final[:top_k] if not is_unlimited_mode else final
            return await self._to_response(final_results, req)

        except Exception as e:
            logger.error(f"Error in retrieve_mem_agentic: {e}", exc_info=True)
            return await self._to_response([], req)

    async def _batch_get_memcells(
        self, event_ids: List[str], batch_size: int = 100
    ) -> Dict[str, Any]:
        """Batch get MemCells, supports batch queries to control single query size

        Args:
            event_ids: List of event_id to get
            batch_size: Number of items per batch, default 100

        Returns:
            Dict[event_id, MemCell]: Mapping dictionary from event_id to MemCell
        """
        if not event_ids:
            return {}

        # Deduplicate event_ids
        unique_event_ids = list(set(event_ids))
        logger.debug(
            f"Batch get MemCells: Total {len(unique_event_ids)} (before deduplication: {len(event_ids)})"
        )

        memcell_repo = get_bean_by_type(MemCellRawRepository)
        all_memcells = {}

        # Batch get
        for i in range(0, len(unique_event_ids), batch_size):
            batch_event_ids = unique_event_ids[i : i + batch_size]
            logger.debug(
                f"Getting batch {i // batch_size + 1} MemCells: {len(batch_event_ids)} items"
            )

            batch_memcells = await memcell_repo.get_by_event_ids(batch_event_ids)
            all_memcells.update(batch_memcells)

        logger.debug(
            f"Batch get MemCells completed: Successfully retrieved {len(all_memcells)} items"
        )
        return all_memcells

    def _get_type_str(self, val) -> str:
        """Extract string value of type field"""
        if isinstance(val, RawDataType):
            return val.value
        return str(val) if val else ''

    def _extract_hit_fields_from_es(self, hit: Dict[str, Any]) -> Dict[str, Any]:
        """Extract fields from ES search result"""
        source = hit.get('_source', {})
        return {
            'hit_id': source.get('event_id', '')
            or source.get('id', '')
            or hit.get('_id', ''),
            'user_id': source.get('user_id', ''),
            'group_id': source.get('group_id', ''),
            'timestamp_raw': source.get('timestamp', ''),
            'episode': source.get('episode', ''),
            'parent_type': source.get('parent_type', ''),
            'parent_id': source.get('parent_id', ''),
            'subject': source.get('subject', ''),
            'summary': source.get('summary', ''),
            'participants': source.get('participants', []),
            'sender_ids': source.get('sender_ids', []),
            'event_type': source.get('type', ''),
            'atomic_fact': source.get('atomic_fact', ''),
            'foresight': source.get('foresight', ''),
            'evidence': source.get('evidence', ''),
            'extend_data': source.get('extend', {}) or {},
            'search_source': 'keyword',
            # Agent-specific fields
            'task_intent': source.get('task_intent', ''),
            'approach': source.get('approach', ''),
            'quality_score': source.get('quality_score'),
            'name': source.get('name', ''),
            'description': source.get('description', ''),
            'content': source.get('content', ''),
            'confidence': source.get('confidence'),
            'maturity_score': source.get('maturity_score'),
            'cluster_id': source.get('cluster_id', ''),
        }

    def _extract_hit_fields_from_milvus(self, hit: Dict[str, Any]) -> Dict[str, Any]:
        """Extract fields from Milvus search result.

        Note: Milvus collections no longer store a 'metadata' JSON field.
        Fields like subject/summary are not stored in Milvus and will be empty
        here — callers should backfill from MongoDB when display fields are needed.
        """
        timestamp_val = hit.get('timestamp') or hit.get('start_time')
        return {
            'hit_id': hit.get('id', ''),
            'user_id': hit.get('user_id', ''),
            'group_id': hit.get('group_id', ''),
            'timestamp_raw': timestamp_val,
            'episode': hit.get('episode', ''),
            'parent_type': hit.get('parent_type', ''),
            'parent_id': hit.get('parent_id', ''),
            'subject': '',
            'summary': '',
            'participants': hit.get('participants', []),
            'sender_ids': hit.get('sender_ids', []),
            'event_type': self._get_type_str(hit.get('type') or hit.get('event_type')),
            'atomic_fact': hit.get('atomic_fact', ''),
            'foresight': hit.get(
                'content', ''
            ),  # Milvus foresight uses 'content' field
            'evidence': hit.get('evidence', ''),
            'extend_data': {},
            'search_source': 'vector',
            # Agent-specific fields
            'task_intent': hit.get('task_intent', ''),
            'approach': hit.get('approach', ''),
            'quality_score': hit.get('quality_score'),
            'name': hit.get('name', ''),
            'description': hit.get('description', ''),
            'content': hit.get('content', ''),
            'confidence': hit.get('confidence'),
            'maturity_score': hit.get('maturity_score'),
            'cluster_id': hit.get('cluster_id', ''),
        }

    def _extract_hit_fields(self, hit: Dict[str, Any]) -> Dict[str, Any]:
        """Extract fields from search result based on _search_source"""
        search_source = hit.get('_search_source')
        match search_source:
            case RetrieveMethod.KEYWORD.value:
                return self._extract_hit_fields_from_es(hit)
            case RetrieveMethod.VECTOR.value:
                return self._extract_hit_fields_from_milvus(hit)
            case _:
                raise ValueError(f"Unknown _search_source: {search_source}")

    async def group_by_groupid_stratagy(
        self,
        search_results: List[Dict[str, Any]],
        source_type: str = RetrieveMethod.VECTOR.value,
    ) -> tuple:
        """Generic search result grouping processing strategy

        Args:
            search_results: List of search results
            source_type: Retrieval method (keyword/vector/hybrid)

        Returns:
            tuple: (memories, scores, original_data, total_count)
        """
        # Step 1: Collect all data needed for queries
        all_memcell_event_ids = []
        all_user_group_pairs = []

        for hit in search_results:
            fields = self._extract_hit_fields(hit)
            parent_type = fields['parent_type']
            parent_id = fields['parent_id']
            user_id = fields['user_id']
            group_id = fields['group_id']

            if parent_type == ParentType.MEMCELL.value and parent_id:
                all_memcell_event_ids.append(parent_id)

            # Collect user_id and group_id pairs
            if user_id and group_id:
                all_user_group_pairs.append((user_id, group_id))

        # Step 2: Execute two batch query tasks concurrently
        memcells_task = asyncio.create_task(
            self._batch_get_memcells(all_memcell_event_ids)
        )

        # Wait for all tasks to complete
        memcells_cache = await memcells_task

        # Step 3: Process search results
        memories_by_group = {}  # {group_id: [Memory]}

        for hit in search_results:
            # Extract fields
            fields = self._extract_hit_fields(hit)
            # Get score (each retrieval method uses its own score field)
            score = hit.get('score', 0.0)

            hit_id = fields['hit_id']
            user_id = fields['user_id']
            group_id = fields['group_id']
            timestamp_raw = fields['timestamp_raw']
            parent_type = fields['parent_type']
            parent_id = fields['parent_id']
            episode = fields['episode']
            subject = fields['subject']
            summary = fields['summary']
            participants = fields['participants']
            sender_ids = fields['sender_ids']
            event_type = fields['event_type']
            atomic_fact = fields['atomic_fact']
            foresight = fields['foresight']
            evidence = fields['evidence']
            extend_data = fields['extend_data']
            search_source = fields['search_source']
            # Process timestamp
            timestamp = from_iso_format(timestamp_raw)

            # Get memcell data from cache (foresight doesn't need this)
            memory_type_value = hit.get('memory_type', MemoryType.EPISODIC_MEMORY.value)
            original_data = None
            if parent_type == ParentType.MEMCELL.value and parent_id:
                memcell = memcells_cache.get(parent_id)
                if memcell and memcell.original_data:
                    original_data = memcell.original_data
                else:
                    logger.debug(f"Memcell not found: event_id={parent_id}")

            # Create object based on memory type
            base_kwargs = dict(
                id=hit_id,
                memory_type=memory_type_value,
                user_id=user_id,
                timestamp=timestamp,
                group_id=group_id,
                participants=participants,
                sender_ids=sender_ids,
                parent_type=parent_type,
                parent_id=parent_id,
                type=RawDataType.from_string(event_type),
                score=score,
                original_data=original_data,
                extend={'_search_source': search_source},
            )

            match memory_type_value:
                case MemoryType.ATOMIC_FACT.value:
                    memory = AtomicFact(**base_kwargs, atomic_fact=atomic_fact)
                case MemoryType.FORESIGHT.value:
                    memory = Foresight(
                        **base_kwargs, foresight=foresight, evidence=evidence
                    )
                case MemoryType.EPISODIC_MEMORY.value:
                    # EpisodeMemory has additional fields: subject, summary, episode
                    memory = EpisodeMemory(
                        **base_kwargs, subject=subject, summary=summary, episode=episode
                    )
                case MemoryType.AGENT_CASE.value:
                    memory = AgentCase(
                        **base_kwargs,
                        task_intent=fields.get('task_intent', ''),
                        approach=fields.get('approach', ''),
                        quality_score=fields.get('quality_score'),
                        key_insight=fields.get('key_insight', '')
                    )
                case MemoryType.AGENT_SKILL.value:
                    # AgentSkill doesn't have parent_type/parent_id fields
                    skill_kwargs = {
                        k: v
                        for k, v in base_kwargs.items()
                        if k not in ('parent_type', 'parent_id')
                    }
                    memory = AgentSkill(
                        **skill_kwargs,
                        name=fields.get('name', ''),
                        description=fields.get('description', ''),
                        content=fields.get('content', ''),
                        confidence=fields.get('confidence', 0.0) or 0.0,
                        cluster_id=fields.get('cluster_id', ''),
                        maturity_score=fields.get('maturity_score', 0.0) or 0.0,
                    )
                case _:
                    raise ValueError(f"Unsupported memory type: {memory_type_value}")

            # Group by group_id
            if group_id not in memories_by_group:
                memories_by_group[group_id] = []

            memories_by_group[group_id].append(memory)

        # Collect all memories and sort by score descending
        memories = []
        for group_memories in memories_by_group.values():
            memories.extend(group_memories)

        # Sort by score descending (higher relevance first)
        memories.sort(key=lambda m: m.score or 0.0, reverse=True)

        total_count = len(memories)
        return memories, total_count

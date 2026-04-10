from __future__ import annotations

from typing import Any, List, Optional, Tuple
import logging
import asyncio

from datetime import datetime, timedelta
import jieba
import numpy as np
import time
from typing import Dict, Any
from dataclasses import dataclass

from api_specs.memory_types import (
    BaseMemory,
    EpisodeMemory,
    EventLog,
    Foresight,
    RawDataType,
)
from biz_layer.mem_memorize import memorize
from api_specs.dtos import MemorizeRequest
from .fetch_mem_service import get_fetch_memory_service
from api_specs.dtos import (
    FetchMemRequest,
    FetchMemResponse,
    PendingMessage,
    RetrieveMemRequest,
    RetrieveMemResponse,
)
from api_specs.memory_models import Metadata
from core.di import get_bean_by_type
from core.oxm.constants import MAGIC_ALL
from infra_layer.adapters.out.search.repository.episodic_memory_es_repository import (
    EpisodicMemoryEsRepository,
)
from infra_layer.adapters.out.search.repository.foresight_es_repository import (
    ForesightEsRepository,
)
from infra_layer.adapters.out.search.repository.event_log_es_repository import (
    EventLogEsRepository,
)
from core.observation.tracing.decorators import trace_logger
from core.nlp.stopwords_utils import filter_stopwords
from common_utils.datetime_utils import (
    from_iso_format,
    get_now_with_timezone,
    to_iso_format,
)
from infra_layer.adapters.out.persistence.repository.memcell_raw_repository import (
    MemCellRawRepository,
)
from service.memory_request_log_service import MemoryRequestLogService
from infra_layer.adapters.out.persistence.repository.group_user_profile_memory_raw_repository import (
    GroupUserProfileMemoryRawRepository,
)
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
from infra_layer.adapters.out.search.repository.event_log_milvus_repository import (
    EventLogMilvusRepository,
)
from .vectorize_service import get_vectorize_service
from .rerank_service import get_rerank_service
from api_specs.memory_models import MemoryType, RetrieveMethod
from agentic_layer.metrics.retrieve_metrics import (
    record_retrieve_request,
    record_retrieve_stage,
    record_retrieve_error,
)
import os
from memory_layer.llm.llm_provider import LLMProvider
from agentic_layer.agentic_utils import (
    AgenticConfig,
    check_sufficiency,
    generate_multi_queries,
)
from agentic_layer.retrieval_utils import reciprocal_rank_fusion

logger = logging.getLogger(__name__)


# MemoryType -> ES Repository mapping
ES_REPO_MAP = {
    MemoryType.FORESIGHT: ForesightEsRepository,
    MemoryType.EVENT_LOG: EventLogEsRepository,
    MemoryType.EPISODIC_MEMORY: EpisodicMemoryEsRepository,
}


@dataclass
class EventLogCandidate:
    """Event Log candidate object (used for retrieval from atomic_fact)"""

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
    - fetch_mem: Retrieve memory fields by key, supports multiple memory types
    - retrieve_mem: Memory reading based on prompt-based retrieval methods
    """

    def __init__(self) -> None:
        # Get memory service instance
        self._fetch_service = get_fetch_memory_service()
        self._request_log_service: MemoryRequestLogService = get_bean_by_type(
            MemoryRequestLogService
        )

        logger.info(
            "MemoryManager initialized with fetch_mem_service and retrieve_mem_service"
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

    # --------- Read path (query -> fetch_mem) ---------
    # Memory reading based on key-value, including static and dynamic memory
    @trace_logger(operation_name="agentic_layer memory reading")
    async def fetch_mem(self, request: FetchMemRequest) -> FetchMemResponse:
        """Retrieve memory data, supports multiple memory types

        Args:
            request: FetchMemRequest containing query parameters

        Returns:
            FetchMemResponse containing query results
        """
        logger.debug(
            f"fetch_mem called with request: user_id={request.user_id}, group_id={request.group_id}, "
            f"memory_type={request.memory_type}, time_range=[{request.start_time}, {request.end_time}]"
        )

        # repository supports MemoryType.EPISODIC_MEMORY type, default is episodic memory
        response = await self._fetch_service.find_memories(
            user_id=request.user_id,
            memory_type=request.memory_type,
            group_id=request.group_id,
            start_time=request.start_time,
            end_time=request.end_time,
            version_range=request.version_range,
            limit=request.limit,
        )

        # Note: response.metadata already contains complete employee information
        # including source, user_id, memory_type, limit, email, phone, full_name
        # No need to update again here, as fetch_mem_service already provides correct information

        logger.debug(
            f"fetch_mem returned {len(response.memories)} memories for user {request.user_id}"
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

            # Dispatch based on retrieve_method
            retrieve_method = retrieve_mem_request.retrieve_method

            logger.info(
                f"retrieve_mem dispatching request: user_id={retrieve_mem_request.user_id}, "
                f"retrieve_method={retrieve_method}, query={retrieve_mem_request.query}"
            )

            # Create task to fetch pending messages concurrently
            pending_messages_task = asyncio.create_task(
                self._get_pending_messages(
                    user_id=retrieve_mem_request.user_id,
                    group_id=retrieve_mem_request.group_id,
                )
            )

            # Dispatch based on retrieval method
            match retrieve_method:
                case RetrieveMethod.KEYWORD:
                    response = await self.retrieve_mem_keyword(retrieve_mem_request)
                case RetrieveMethod.VECTOR:
                    response = await self.retrieve_mem_vector(retrieve_mem_request)
                case RetrieveMethod.HYBRID:
                    response = await self.retrieve_mem_hybrid(retrieve_mem_request)
                case RetrieveMethod.RRF:
                    response = await self.retrieve_mem_rrf(retrieve_mem_request)
                case RetrieveMethod.AGENTIC:
                    response = await self.retrieve_mem_agentic(retrieve_mem_request)
                case _:
                    raise ValueError(f"Unsupported retrieval method: {retrieve_method}")

            # Await pending messages and attach to response
            pending_messages = await pending_messages_task
            response.pending_messages = pending_messages

            return response

        except Exception as e:
            logger.error(f"Error in retrieve_mem: {e}", exc_info=True)
            return RetrieveMemResponse(
                memories=[],
                original_data=[],
                scores=[],
                importance_scores=[],
                total_count=0,
                has_more=False,
                query_metadata=Metadata(
                    source="retrieve_mem_service",
                    user_id=(
                        retrieve_mem_request.user_id if retrieve_mem_request else ""
                    ),
                    memory_type="retrieve",
                ),
                metadata=Metadata(
                    source="retrieve_mem_service",
                    user_id=(
                        retrieve_mem_request.user_id if retrieve_mem_request else ""
                    ),
                    memory_type="retrieve",
                ),
                pending_messages=[],
            )

    async def _get_pending_messages(
        self, user_id: Optional[str] = None, group_id: Optional[str] = None
    ) -> List[PendingMessage]:
        """
        Get pending (unconsumed) messages from MemoryRequestLogService.

        Fetches cached memory data that hasn't been consumed yet (sync_status=-1 or 0).

        Args:
            user_id: User ID filter (from retrieve_request)
            group_id: Group ID filter (from retrieve_request)

        Returns:
            List of PendingMessage objects
        """
        try:
            result = await self._request_log_service.get_pending_messages(
                user_id=user_id, group_id=group_id, limit=1000
            )

            logger.debug(
                f"Retrieved {len(result)} pending messages: "
                f"user_id={user_id}, group_id={group_id}"
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
        start_time = time.perf_counter()
        memory_type_label = (
            ','.join([mt.value for mt in retrieve_mem_request.memory_types])
            if retrieve_mem_request.memory_types
            else 'unknown'
        )

        try:
            hits = await self.get_keyword_search_results(
                retrieve_mem_request, retrieve_method=RetrieveMethod.KEYWORD.value
            )
            duration = time.perf_counter() - start_time
            status = 'success' if hits else 'empty_result'

            record_retrieve_request(
                memory_type=memory_type_label,
                retrieve_method=RetrieveMethod.KEYWORD.value,
                status=status,
                duration_seconds=duration,
                results_count=len(hits),
            )

            return await self._to_response(hits, retrieve_mem_request)
        except Exception as e:
            duration = time.perf_counter() - start_time
            record_retrieve_request(
                memory_type=memory_type_label,
                retrieve_method=RetrieveMethod.KEYWORD.value,
                status='error',
                duration_seconds=duration,
                results_count=0,
            )
            logger.error(f"Error in retrieve_mem_keyword: {e}", exc_info=True)
            return await self._to_response([], retrieve_mem_request)

    async def get_keyword_search_results(
        self,
        retrieve_mem_request: 'RetrieveMemRequest',
        retrieve_method: str = RetrieveMethod.KEYWORD.value,
    ) -> List[Dict[str, Any]]:
        """Keyword search with stage-level metrics, supports multiple memory_types"""
        stage_start = time.perf_counter()
        memory_types = retrieve_mem_request.memory_types or []
        memory_type_label = (
            ','.join([mt.value for mt in memory_types])
            if memory_types
            else 'unknown'
        )

        try:
            # Get parameters from Request
            if not retrieve_mem_request:
                raise ValueError("retrieve_mem_request is required for retrieve_mem")

            top_k = retrieve_mem_request.top_k
            query = retrieve_mem_request.query
            user_id = retrieve_mem_request.user_id
            group_id = retrieve_mem_request.group_id
            start_time = retrieve_mem_request.start_time
            end_time = retrieve_mem_request.end_time

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

            # Search across ALL memory_types, not just the first one
            all_results = []
            for mem_type in memory_types:
                repo_class = ES_REPO_MAP.get(mem_type)
                if not repo_class:
                    logger.warning(f"Unsupported memory_type: {mem_type}, skipping")
                    continue

                es_repo = get_bean_by_type(repo_class)
                logger.debug(f"Using {repo_class.__name__} for {mem_type}")

                results = await es_repo.multi_search(
                    query=query_words,
                    user_id=user_id,
                    group_id=group_id,
                    size=top_k,
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
                    all_results.extend(results)

            # Record stage metrics
            record_retrieve_stage(
                retrieve_method=retrieve_method,
                stage=RetrieveMethod.KEYWORD.value,
                memory_type=memory_type_label,
                duration_seconds=time.perf_counter() - stage_start,
            )

            return all_results
        except Exception as e:
            record_retrieve_stage(
                retrieve_method=retrieve_method,
                stage=RetrieveMethod.KEYWORD.value,
                memory_type=memory_type_label,
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
        start_time = time.perf_counter()
        memory_type_label = (
            ','.join([mt.value for mt in retrieve_mem_request.memory_types])
            if retrieve_mem_request.memory_types
            else 'unknown'
        )

        try:
            hits = await self.get_vector_search_results(
                retrieve_mem_request, retrieve_method=RetrieveMethod.VECTOR.value
            )
            duration = time.perf_counter() - start_time
            status = 'success' if hits else 'empty_result'

            record_retrieve_request(
                memory_type=memory_type_label,
                retrieve_method=RetrieveMethod.VECTOR.value,
                status=status,
                duration_seconds=duration,
                results_count=len(hits),
            )

            return await self._to_response(hits, retrieve_mem_request)
        except Exception as e:
            duration = time.perf_counter() - start_time
            record_retrieve_request(
                memory_type=memory_type_label,
                retrieve_method=RetrieveMethod.VECTOR.value,
                status='error',
                duration_seconds=duration,
                results_count=0,
            )
            logger.error(f"Error in retrieve_mem_vector: {e}")
            return await self._to_response([], retrieve_mem_request)

    async def get_vector_search_results(
        self,
        retrieve_mem_request: 'RetrieveMemRequest',
        retrieve_method: str = RetrieveMethod.VECTOR.value,
    ) -> List[Dict[str, Any]]:
        """Vector search with stage-level metrics (embedding + milvus_search), supports multiple memory_types"""
        memory_types = retrieve_mem_request.memory_types or []
        memory_type_label = (
            ','.join([mt.value for mt in memory_types])
            if memory_types
            else 'unknown'
        )
        stage_start = time.perf_counter()

        milvus_start = time.perf_counter()  # init early so except handler never fails
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
            group_id = retrieve_mem_request.group_id
            top_k = retrieve_mem_request.top_k
            start_time = retrieve_mem_request.start_time
            end_time = retrieve_mem_request.end_time

            logger.debug(
                f"retrieve_mem_vector called with query: {query}, user_id: {user_id}, group_id: {group_id}, top_k: {top_k}"
            )

            # Get vectorization service
            vectorize_service = get_vectorize_service()

            # Convert query text to vector (embedding stage) - only once for all types
            logger.debug(f"Starting to vectorize query text: {query}")
            embedding_start = time.perf_counter()
            query_vector = await vectorize_service.get_embedding(query)
            query_vector_list = query_vector.tolist()  # Convert to list format
            record_retrieve_stage(
                retrieve_method=retrieve_method,
                stage='embedding',
                memory_type=memory_type_label,
                duration_seconds=time.perf_counter() - embedding_start,
            )
            logger.debug(
                f"Query text vectorization completed, vector dimension: {len(query_vector_list)}"
            )

            # Map memory type to Milvus repository
            MILVUS_REPO_MAP = {
                MemoryType.FORESIGHT: ForesightMilvusRepository,
                MemoryType.EVENT_LOG: EventLogMilvusRepository,
                MemoryType.EPISODIC_MEMORY: EpisodicMemoryMilvusRepository,
            }

            # Search across ALL memory_types, not just the first one
            all_search_results = []
            for mem_type in memory_types:
                milvus_repo_class = MILVUS_REPO_MAP.get(mem_type)
                if not milvus_repo_class:
                    logger.warning(f"Unsupported memory type for vector search: {mem_type}, skipping")
                    continue

                milvus_repo = get_bean_by_type(milvus_repo_class)

                # Handle time range filter conditions
                start_time_dt = None
                end_time_dt = None
                current_time_dt = None

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
                    if retrieve_mem_request.current_time:
                        current_time_dt = from_iso_format(retrieve_mem_request.current_time)

                # Call Milvus vector search (pass different parameters based on memory type)
                milvus_start = time.perf_counter()
                if mem_type == MemoryType.FORESIGHT:
                    # Foresight: supports time range and validity filtering, supports radius parameter
                    search_results = await milvus_repo.vector_search(
                        query_vector=query_vector_list,
                        user_id=user_id,
                        group_id=group_id,
                        start_time=start_time_dt,
                        end_time=end_time_dt,
                        current_time=current_time_dt,
                        limit=top_k,
                        score_threshold=0.0,
                        radius=retrieve_mem_request.radius,
                    )
                else:
                    # Episodic memory and event log: use timestamp filtering, supports radius parameter
                    search_results = await milvus_repo.vector_search(
                        query_vector=query_vector_list,
                        user_id=user_id,
                        group_id=group_id,
                        start_time=start_time_dt,
                        end_time=end_time_dt,
                        limit=top_k,
                        score_threshold=0.0,
                        radius=retrieve_mem_request.radius,
                    )
                record_retrieve_stage(
                    retrieve_method=retrieve_method,
                    stage='milvus_search',
                    memory_type=mem_type.value,
                    duration_seconds=time.perf_counter() - milvus_start,
                )

                for r in search_results:
                    r['memory_type'] = mem_type.value
                    r['_search_source'] = RetrieveMethod.VECTOR.value
                    # Milvus already uses 'score', no need to rename

                all_search_results.extend(search_results)

            return all_search_results
        except Exception as e:
            record_retrieve_stage(
                retrieve_method=retrieve_method,
                stage=RetrieveMethod.VECTOR.value,
                memory_type=memory_type_label,
                duration_seconds=time.perf_counter() - stage_start,
            )
            record_retrieve_error(
                retrieve_method=retrieve_method,
                stage=RetrieveMethod.VECTOR.value,
                error_type=self._classify_retrieve_error(e),
            )
            logger.error(f"Error in get_vector_search_results: {e}")
            return []
            raise

    # Hybrid memory retrieval
    @trace_logger(operation_name="agentic_layer hybrid memory retrieval")
    async def retrieve_mem_hybrid(
        self, retrieve_mem_request: 'RetrieveMemRequest'
    ) -> RetrieveMemResponse:
        """Hybrid memory retrieval: keyword + vector + rerank"""
        start_time = time.perf_counter()
        memory_type_label = (
            ','.join([mt.value for mt in retrieve_mem_request.memory_types])
            if retrieve_mem_request.memory_types
            else 'unknown'
        )

        try:
            hits = await self._search_hybrid(
                retrieve_mem_request, retrieve_method=RetrieveMethod.HYBRID.value
            )
            duration = time.perf_counter() - start_time
            status = 'success' if hits else 'empty_result'

            record_retrieve_request(
                memory_type=memory_type_label,
                retrieve_method=RetrieveMethod.HYBRID.value,
                status=status,
                duration_seconds=duration,
                results_count=len(hits),
            )

            return await self._to_response(hits, retrieve_mem_request)
        except Exception as e:
            duration = time.perf_counter() - start_time
            record_retrieve_request(
                memory_type=memory_type_label,
                retrieve_method=RetrieveMethod.HYBRID.value,
                status='error',
                duration_seconds=duration,
                results_count=0,
            )
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
    ) -> List[Dict]:
        """Rerank hits using rerank service with stage metrics"""
        if not hits:
            return []

        stage_start = time.perf_counter()
        try:
            result = await get_rerank_service().rerank_memories(
                query, hits, top_k, instruction=instruction
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
        memory_type_label = (
            ','.join([mt.value for mt in request.memory_types])
            if request.memory_types
            else 'unknown'
        )
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
        return await self._rerank(
            request.query, merged_results, request.top_k, memory_type_label, retrieve_method
        )

    async def _search_rrf(
        self,
        request: 'RetrieveMemRequest',
        retrieve_method: str = RetrieveMethod.RRF.value,
    ) -> List[Dict]:
        """Core RRF search: keyword + vector + RRF fusion, returns flat list"""
        memory_type_label = (
            ','.join([mt.value for mt in request.memory_types])
            if request.memory_types
            else 'unknown'
        )

        # Run keyword and vector search concurrently
        kw, vec = await asyncio.gather(
            self.get_keyword_search_results(request, retrieve_method=retrieve_method),
            self.get_vector_search_results(request, retrieve_method=retrieve_method),
        )

        # RRF fusion with stage metrics
        rrf_start = time.perf_counter()
        kw_tuples = [(h, h.get('score', 0)) for h in kw]
        vec_tuples = [(h, h.get('score', 0)) for h in vec]
        fused = reciprocal_rank_fusion(kw_tuples, vec_tuples, k=60)
        record_retrieve_stage(
            retrieve_method=retrieve_method,
            stage='rrf_fusion',
            memory_type=memory_type_label,
            duration_seconds=time.perf_counter() - rrf_start,
        )

        return [dict(doc, score=score) for doc, score in fused[: request.top_k]]

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
        memory_type_label = (
            ','.join([mt.value for mt in req.memory_types])
            if req.memory_types
            else 'unknown'
        )

        if not hits:
            return RetrieveMemResponse(
                memories=[],
                original_data=[],
                scores=[],
                importance_scores=[],
                total_count=0,
                has_more=False,
                query_metadata=Metadata(
                    source=source_type, user_id=user_id or "", memory_type=memory_type_label
                ),
                metadata=Metadata(
                    source=source_type, user_id=user_id or "", memory_type=memory_type_label
                ),
            )
        memories, scores, importance_scores, original_data, total_count = (
            await self.group_by_groupid_stratagy(hits, source_type=source_type)
        )
        return RetrieveMemResponse(
            memories=memories,
            scores=scores,
            importance_scores=importance_scores,
            original_data=original_data,
            total_count=total_count,
            has_more=False,
            query_metadata=Metadata(
                source=source_type, user_id=user_id or "", memory_type=memory_type_label
            ),
            metadata=Metadata(
                source=source_type, user_id=user_id or "", memory_type=memory_type_label
            ),
        )

    # --------- RRF retrieval (keyword + vector + RRF fusion, no rerank) ---------
    @trace_logger(operation_name="agentic_layer RRF memory retrieval")
    async def retrieve_mem_rrf(
        self, retrieve_mem_request: 'RetrieveMemRequest'
    ) -> RetrieveMemResponse:
        """RRF-based memory retrieval: keyword + vector + RRF fusion"""
        start_time = time.perf_counter()
        memory_type_label = (
            ','.join([mt.value for mt in retrieve_mem_request.memory_types])
            if retrieve_mem_request.memory_types
            else 'unknown'
        )

        try:
            hits = await self._search_rrf(
                retrieve_mem_request, retrieve_method=RetrieveMethod.RRF.value
            )
            duration = time.perf_counter() - start_time
            status = 'success' if hits else 'empty_result'

            record_retrieve_request(
                memory_type=memory_type_label,
                retrieve_method=RetrieveMethod.RRF.value,
                status=status,
                duration_seconds=duration,
                results_count=len(hits),
            )

            return await self._to_response(hits, retrieve_mem_request)
        except Exception as e:
            duration = time.perf_counter() - start_time
            record_retrieve_request(
                memory_type=memory_type_label,
                retrieve_method=RetrieveMethod.RRF.value,
                status='error',
                duration_seconds=duration,
                results_count=0,
            )
            logger.error(f"Error in retrieve_mem_rrf: {e}", exc_info=True)
            return await self._to_response([], retrieve_mem_request)

    # --------- Agentic retrieval (LLM-guided multi-round) ---------
    @trace_logger(operation_name="agentic_layer Agentic memory retrieval")
    async def retrieve_mem_agentic(
        self, retrieve_mem_request: 'RetrieveMemRequest'
    ) -> RetrieveMemResponse:
        """Agentic retrieval: LLM-guided multi-round intelligent retrieval

        Process: Round 1 (Hybrid) → Rerank → LLM sufficiency check → Round 2 (multi-query) → Merge → Final Rerank
        """
        start_time = time.perf_counter()
        req = retrieve_mem_request  # alias
        top_k = req.top_k
        config = AgenticConfig()
        memory_type_label = (
            ','.join([mt.value for mt in req.memory_types])
            if req.memory_types
            else 'unknown'
        )

        try:
            llm_provider = LLMProvider(
                provider_type=os.getenv("LLM_PROVIDER", "openai"),
                model=os.getenv("LLM_MODEL", "Qwen3-235B"),
                base_url=os.getenv("LLM_BASE_URL"),
                api_key=os.getenv("LLM_API_KEY"),
                temperature=float(os.getenv("LLM_TEMPERATURE", "0.3")),
                max_tokens=int(os.getenv("LLM_MAX_TOKENS", "16384")),
            )

            logger.info(f"Agentic Retrieval: {req.query[:60]}...")

            # ========== Round 1: Hybrid search ==========
            req1 = RetrieveMemRequest(
                query=req.query,
                user_id=req.user_id,
                group_id=req.group_id,
                top_k=config.round1_top_n,
                memory_types=req.memory_types,
            )
            round1 = await self._search_hybrid(req1, retrieve_method='agentic')
            logger.info(f"Round 1: {len(round1)} memories")

            if not round1:
                duration = time.perf_counter() - start_time
                record_retrieve_request(
                    memory_type=memory_type_label,
                    retrieve_method=RetrieveMethod.AGENTIC.value,
                    status='empty_result',
                    duration_seconds=duration,
                    results_count=0,
                )
                return await self._to_response([], req)

            # ========== Rerank → max(5, top_k) for LLM & return ==========
            rerank_n = max(config.round1_rerank_top_n, top_k)
            reranked = await self._rerank(
                req.query, round1, rerank_n, memory_type_label, 'agentic',
                instruction=config.reranker_instruction,
            )
            # Use top 5 for sufficiency check
            topn_for_llm = reranked[:config.round1_rerank_top_n]
            topn_pairs = [(m, m.get("score", 0)) for m in topn_for_llm]

            # ========== LLM sufficiency check ==========
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
                # Return reranked results (already done above, no extra rerank)
                final_results = reranked[:top_k]
                duration = time.perf_counter() - start_time
                record_retrieve_request(
                    memory_type=memory_type_label,
                    retrieve_method=RetrieveMethod.AGENTIC.value,
                    status='success',
                    duration_seconds=duration,
                    results_count=len(final_results),
                )
                return await self._to_response(final_results, req)

            # ========== Round 2: Multi-query ==========
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
                        group_id=req.group_id,
                        top_k=config.round2_per_query_top_n,
                        memory_types=req.memory_types,
                    ),
                    retrieve_method='agentic',
                )

            round2_results = await asyncio.gather(
                *[do_search(q) for q in refined_queries], return_exceptions=True
            )
            all_round2 = [
                h for r in round2_results if not isinstance(r, Exception) for h in r
            ]

            # Deduplicate and merge
            seen_ids = {m.get("id") for m in round1}
            round2_unique = [m for m in all_round2 if m.get("id") not in seen_ids]
            combined = round1 + round2_unique[: config.combined_total - len(round1)]
            logger.info(f"Combined: {len(combined)} memories")

            # ========== Final Rerank ==========
            final = await self._rerank(
                req.query, combined, top_k, memory_type_label, 'agentic',
                instruction=config.reranker_instruction,
            )

            duration = time.perf_counter() - start_time
            record_retrieve_request(
                memory_type=memory_type_label,
                retrieve_method=RetrieveMethod.AGENTIC.value,
                status='success',
                duration_seconds=duration,
                results_count=len(final[:top_k]),
            )

            return await self._to_response(final[:top_k], req)

        except Exception as e:
            duration = time.perf_counter() - start_time
            record_retrieve_request(
                memory_type=memory_type_label,
                retrieve_method=RetrieveMethod.AGENTIC.value,
                status='error',
                duration_seconds=duration,
                results_count=0,
            )
            logger.error(f"Error in retrieve_mem_agentic: {e}", exc_info=True)
            return await self._to_response([], req)

    def _calculate_importance_score(
        self, importance_evidence: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate group importance score

        Calculate score based on group importance evidence, mainly considering:
        - speak_count: User's speaking count in this group
        - refer_count: Number of times user was mentioned
        - conversation_count: Total conversation count in this group

        Importance score = (total speaking count + total mention count) / total conversation count

        Args:
            importance_evidence: Group importance evidence dictionary

        Returns:
            float: Importance score, range [0, +∞), larger value means more important group
        """
        if not importance_evidence or not isinstance(importance_evidence, dict):
            return 0.0

        evidence_list = importance_evidence.get('evidence_list', [])
        if not evidence_list:
            return 0.0

        total_speak_count = 0
        total_refer_count = 0
        total_conversation_count = 0

        # Accumulate statistics from all evidence
        for evidence in evidence_list:
            if isinstance(evidence, dict):
                total_speak_count += evidence.get('speak_count', 0)
                total_refer_count += evidence.get('refer_count', 0)
                total_conversation_count += evidence.get('conversation_count', 0)

        # Avoid division by zero
        if total_conversation_count == 0:
            return 0.0

        # Calculate importance score
        return (total_speak_count + total_refer_count) / total_conversation_count

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

    async def _batch_get_group_profiles(
        self, user_group_pairs: List[Tuple[str, str]]
    ) -> Dict[Tuple[str, str], Any]:
        """Batch get group user profiles, supports efficient querying

        Args:
            user_group_pairs: List of (user_id, group_id) tuples

        Returns:
            Dict[(user_id, group_id), GroupUserProfileMemory]: Mapping dictionary
        """
        if not user_group_pairs:
            return {}

        # Deduplicate
        unique_pairs = list(set(user_group_pairs))
        logger.debug(
            f"Batch get group user profiles: Total {len(unique_pairs)} (before deduplication: {len(user_group_pairs)})"
        )

        group_user_profile_repo = get_bean_by_type(GroupUserProfileMemoryRawRepository)
        profiles = await group_user_profile_repo.batch_get_by_user_groups(unique_pairs)

        logger.debug(
            f"Batch get group user profiles completed: Successfully retrieved {len([v for v in profiles.values() if v is not None])} items"
        )
        return profiles

    def _get_type_str(self, val) -> str:
        """Extract string value of type field"""
        if isinstance(val, RawDataType):
            return val.value
        return str(val) if val else ''

    def _extract_hit_fields_from_es(self, hit: Dict[str, Any]) -> Dict[str, Any]:
        """Extract fields from ES search result"""
        source = hit.get('_source', {})
        return {
            'hit_id': source.get('event_id', ''),
            'user_id': source.get('user_id', ''),
            'group_id': source.get('group_id', ''),
            'timestamp_raw': source.get('timestamp', ''),
            'episode': source.get('episode', ''),
            'memcell_event_id_list': source.get('memcell_event_id_list', []),
            'subject': source.get('subject', ''),
            'summary': source.get('summary', ''),
            'participants': source.get('participants', []),
            'event_type': source.get('type', ''),
            'atomic_fact': source.get('atomic_fact', ''),
            'foresight': source.get('foresight', ''),
            'evidence': source.get('evidence', ''),
            'extend_data': source.get('extend', {}) or {},
            'search_source': 'keyword',
        }

    def _extract_hit_fields_from_milvus(self, hit: Dict[str, Any]) -> Dict[str, Any]:
        """Extract fields from Milvus search result"""
        metadata = hit.get('metadata', {})
        timestamp_val = hit.get('timestamp') or hit.get('start_time')
        return {
            'hit_id': hit.get('id', ''),
            'user_id': hit.get('user_id', ''),
            'group_id': hit.get('group_id', ''),
            'timestamp_raw': timestamp_val,
            'episode': hit.get('episode', ''),
            'memcell_event_id_list': metadata.get('memcell_event_id_list', []),
            'subject': metadata.get('subject', ''),
            'summary': metadata.get('summary', ''),
            'participants': metadata.get('participants', []),
            'event_type': self._get_type_str(hit.get('type') or hit.get('event_type')),
            'atomic_fact': hit.get('atomic_fact', ''),
            'foresight': hit.get(
                'content', ''
            ),  # Milvus foresight uses 'content' field
            'evidence': hit.get('evidence', ''),
            'extend_data': metadata.get('extend', {}) or {},
            'search_source': 'vector',
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
            tuple: (memories, scores, importance_scores, original_data, total_count)
        """
        # Step 1: Collect all data needed for queries
        all_memcell_event_ids = []
        all_user_group_pairs = []

        for hit in search_results:
            fields = self._extract_hit_fields(hit)
            memcell_event_id_list = fields['memcell_event_id_list']
            user_id = fields['user_id']
            group_id = fields['group_id']

            if memcell_event_id_list:
                all_memcell_event_ids.extend(memcell_event_id_list)

            # Collect user_id and group_id pairs
            if user_id and group_id:
                all_user_group_pairs.append((user_id, group_id))

        # Step 2: Execute two batch query tasks concurrently
        memcells_task = asyncio.create_task(
            self._batch_get_memcells(all_memcell_event_ids)
        )
        profiles_task = asyncio.create_task(
            self._batch_get_group_profiles(all_user_group_pairs)
        )

        # Wait for all tasks to complete
        memcells_cache, profiles_cache = await asyncio.gather(
            memcells_task, profiles_task
        )

        # Step 3: Process search results
        memories_by_group = (
            {}
        )  # {group_id: {'memories': [Memory], 'scores': [float], 'importance_evidence': dict}}
        original_data_by_group = {}

        for hit in search_results:
            # Extract fields
            fields = self._extract_hit_fields(hit)
            # Get score (each retrieval method uses its own score field)
            score = hit.get('score', 0.0)

            hit_id = fields['hit_id']
            user_id = fields['user_id']
            group_id = fields['group_id']
            timestamp_raw = fields['timestamp_raw']
            memcell_event_id_list = fields['memcell_event_id_list']
            episode = fields['episode']
            subject = fields['subject']
            summary = fields['summary']
            participants = fields['participants']
            event_type = fields['event_type']
            atomic_fact = fields['atomic_fact']
            foresight = fields['foresight']
            evidence = fields['evidence']
            extend_data = fields['extend_data']
            search_source = fields['search_source']
            # Process timestamp
            timestamp = from_iso_format(timestamp_raw)

            # Get memcell data from cache (foresight doesn't need this)
            memory_type_value = hit.get('memory_type', 'episodic_memory')
            memcells = []
            if memcell_event_id_list:
                # Get memcells from cache in original order
                for event_id in memcell_event_id_list:
                    memcell = memcells_cache.get(event_id)
                    if memcell:
                        memcells.append(memcell)
                    else:
                        logger.debug(f"Memcell not found: event_id={event_id}")
                        continue

            # Add raw data for each memcell
            for memcell in memcells:
                if group_id not in original_data_by_group:
                    original_data_by_group[group_id] = []
                # Use extend instead of append to flatten the list structure
                # memcell.original_data is a List[Dict], not a single Dict
                if memcell.original_data:
                    original_data_by_group[group_id].extend(memcell.original_data)

            # Create object based on memory type
            base_kwargs = dict(
                id=hit_id,
                memory_type=memory_type_value,
                user_id=user_id,
                timestamp=timestamp,
                ori_event_id_list=[hit_id],
                group_id=group_id,
                participants=participants,
                memcell_event_id_list=memcell_event_id_list,
                type=RawDataType.from_string(event_type),
                extend={
                    '_search_source': search_source,
                    'parent_type': extend_data.get('parent_type'),
                    'parent_id': extend_data.get('parent_id'),
                },
            )

            match memory_type_value:
                case MemoryType.EVENT_LOG.value:
                    memory = EventLog(**base_kwargs, atomic_fact=atomic_fact)
                case MemoryType.FORESIGHT.value:
                    memory = Foresight(
                        **base_kwargs, foresight=foresight, evidence=evidence
                    )
                case MemoryType.EPISODIC_MEMORY.value:
                    # EpisodeMemory has additional fields: subject, summary, episode
                    memory = EpisodeMemory(
                        **base_kwargs, subject=subject, summary=summary, episode=episode
                    )
                case _:
                    raise ValueError(f"Unsupported memory type: {memory_type_value}")

            # Get group_importance_evidence from cache
            group_importance_evidence = None
            if user_id and group_id:
                group_user_profile = profiles_cache.get((user_id, group_id))
                if (
                    group_user_profile
                    and hasattr(group_user_profile, 'group_importance_evidence')
                    and group_user_profile.group_importance_evidence
                ):
                    group_importance_evidence = (
                        group_user_profile.group_importance_evidence
                    )
                    # Add group_importance_evidence to memory's extend field
                    if not hasattr(memory, 'extend') or memory.extend is None:
                        memory.extend = {}
                    memory.extend['group_importance_evidence'] = (
                        group_importance_evidence
                    )
                    logger.debug(
                        f"Added group_importance_evidence to memory: user_id={user_id}, group_id={group_id}"
                    )

            # Group by group_id
            if group_id not in memories_by_group:
                memories_by_group[group_id] = {
                    'memories': [],
                    'scores': [],
                    'importance_evidence': group_importance_evidence,
                }

            memories_by_group[group_id]['memories'].append(memory)
            memories_by_group[group_id]['scores'].append(score)  # Save original score

            # Update group_importance_evidence (if current memory has updated evidence)
            if group_importance_evidence:
                memories_by_group[group_id][
                    'importance_evidence'
                ] = group_importance_evidence

        # Sort memories within each group by timestamp, and calculate importance score
        group_scores = []
        for group_id, group_data in memories_by_group.items():
            # Sort memories by timestamp
            group_data['memories'].sort(
                key=lambda m: m.timestamp if m.timestamp else ''
            )

            # Calculate importance score
            importance_score = self._calculate_importance_score(
                group_data['importance_evidence']
            )
            group_scores.append((group_id, importance_score))

        # Sort groups by importance score
        group_scores.sort(key=lambda x: x[1], reverse=True)

        # Build final results
        memories = []
        scores = []
        importance_scores = []
        original_data = []
        for group_id, importance_score in group_scores:
            group_data = memories_by_group[group_id]
            group_memories = group_data['memories']
            group_scores_list = group_data['scores']
            group_original_data = original_data_by_group.get(group_id, [])
            memories.append({group_id: group_memories})
            # scores structure consistent with memories: List[Dict[str, List[float]]]
            scores.append({group_id: group_scores_list})
            # original_data structure consistent with memories: List[Dict[str, List[Dict[str, Any]]]]
            original_data.append({group_id: group_original_data})
            importance_scores.append(importance_score)

        total_count = sum(
            len(group_data['memories']) for group_data in memories_by_group.values()
        )
        return memories, scores, importance_scores, original_data, total_count

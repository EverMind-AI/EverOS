"""
Agent case raw data repository.

Provides CRUD operations for agent case records in MongoDB.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pymongo.asynchronous.client_session import AsyncClientSession
from bson import ObjectId
from core.observation.logger import get_logger
from core.di.decorators import repository
from core.oxm.constants import MAGIC_ALL
from core.oxm.mongo.base_repository import BaseRepository
from core.oxm.mongo.mongo_utils import build_id_filter as _build_id_filter
from infra_layer.adapters.out.persistence.document.memory.agent_case import (
    AgentCaseRecord,
    AgentCaseProjection,
)
from agentic_layer.vectorize_service import get_vectorize_service

logger = get_logger(__name__)


@repository("agent_case_raw_repository", primary=True)
class AgentCaseRawRepository(BaseRepository[AgentCaseRecord]):
    """
    Agent case raw data repository.

    Provides CRUD operations and query functions for agent case records.
    """

    def __init__(self):
        super().__init__(AgentCaseRecord)

    async def append_experience(
        self, record: AgentCaseRecord, session: Optional[AsyncClientSession] = None
    ) -> Optional[AgentCaseRecord]:
        """
        Insert a new agent case record.

        Auto-vectorizes if vector is missing but experiences have task_intent.

        Args:
            record: AgentCaseRecord to insert
            session: Optional MongoDB session for transaction support

        Returns:
            Inserted AgentCaseRecord or None on failure
        """
        # Auto-vectorize if vector is missing
        if not record.vector and record.task_intent:
            try:
                vs = get_vectorize_service()
                vec = await vs.get_embedding(record.task_intent)
                record.vector = vec.tolist() if hasattr(vec, "tolist") else list(vec)
                record.vector_model = vs.get_model_name()
            except Exception as e:
                logger.error(f"[AgentCaseRepo] Auto-vectorize failed: {e}")

        if not record.vector:
            logger.warning(
                "[AgentCaseRepo] Saving AgentCase without vector — "
                "record will not be retrievable via semantic search"
            )

        try:
            result = await record.insert(session=session)
            logger.debug(
                f"[AgentCaseRepo] Inserted experience: id={result.id}, "
                f"intent='{(result.task_intent or '')[:80]}'"
            )
            return result
        except Exception as e:
            logger.error(f"[AgentCaseRepo] Failed to insert experience: {e}")
            return None

    async def get_by_event_id(
        self, event_id: str, session: Optional[AsyncClientSession] = None
    ) -> Optional[AgentCaseRecord]:
        """Retrieve agent case by its own ID."""
        try:
            object_id = ObjectId(event_id)
            return await self.model.find_one({"_id": object_id}, session=session)
        except Exception as e:
            logger.error(f"[AgentCaseRepo] Failed to get by event_id: {e}")
            return None

    async def get_by_ids(
        self, case_ids: List[str], session: Optional[AsyncClientSession] = None
    ) -> List[AgentCaseRecord]:
        """Batch retrieve agent cases by their own IDs.

        Accepts both ObjectId-like strings and raw string IDs.
        """
        query_filter = _build_id_filter(case_ids)
        if query_filter is None:
            return []
        try:
            return await self.model.find(query_filter, session=session).to_list()
        except Exception as e:
            logger.error(f"[AgentCaseRepo] Failed to get by ids: {e}")
            return []

    async def get_by_parent_id(
        self, parent_id: str, session: Optional[AsyncClientSession] = None
    ) -> Optional[AgentCaseRecord]:
        """Retrieve agent case linked to a specific MemCell."""
        try:
            return await self.model.find_one({"parent_id": parent_id}, session=session)
        except Exception as e:
            logger.error(f"[AgentCaseRepo] Failed to get by parent_id: {e}")
            return None

    async def get_by_parent_ids(
        self, parent_ids: List[str], session: Optional[AsyncClientSession] = None
    ) -> List[AgentCaseRecord]:
        """Batch retrieve agent cases by parent MemCell IDs.

        Useful for cluster-level knowledge extraction.
        """
        try:
            results = await self.model.find(
                {"parent_id": {"$in": parent_ids}}, session=session
            ).to_list()
            return results
        except Exception as e:
            logger.error(f"[AgentCaseRepo] Failed to get by parent_ids: {e}")
            return []

    async def find_by_filters(
        self,
        user_id: Optional[str] = MAGIC_ALL,
        group_ids: Optional[List[str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 10,
        skip: int = 0,
        sort_desc: bool = True,
        session: Optional[AsyncClientSession] = None,
    ) -> List[AgentCaseRecord]:
        """
        Find agent cases with flexible filters.

        Args:
            user_id: Filter by user ID.
                - MAGIC_ALL ("__all__"): Don't filter by user_id
                - None or "": Filter for null/empty values
                - Other values: Exact match
            group_ids: List of Group IDs.
                - None: Skip group filtering
                - []: Empty list, skip filtering
                - ["g1"]: Single element, exact match
                - ["g1", "g2"]: Multiple elements, use $in operator
            start_time: Filter by timestamp >= start_time
            end_time: Filter by timestamp <= end_time
            limit: Maximum number of results
            skip: Number of results to skip
            sort_desc: Sort by timestamp descending (newest first)
            session: Optional MongoDB session

        Returns:
            List of matching AgentCaseRecord
        """
        try:
            query: Dict[str, Any] = {}

            # Handle user_id filter (consistent with EpisodicMemoryRawRepository)
            if user_id != MAGIC_ALL:
                if user_id == "" or user_id is None:
                    query["user_id"] = {"$in": [None, ""]}
                else:
                    query["user_id"] = user_id

            # Handle group_ids filter (consistent with EpisodicMemoryRawRepository)
            if group_ids is not None and len(group_ids) > 0:
                if len(group_ids) == 1:
                    query["group_id"] = group_ids[0]
                else:
                    query["group_id"] = {"$in": group_ids}

            if start_time is not None or end_time is not None:
                ts_filter = {}
                if start_time is not None:
                    ts_filter["$gte"] = start_time
                if end_time is not None:
                    ts_filter["$lte"] = end_time
                query["timestamp"] = ts_filter

            sort_field = "-timestamp" if sort_desc else "timestamp"
            q = self.model.find(query, session=session).sort(sort_field)

            if skip:
                q = q.skip(skip)
            if limit:
                q = q.limit(limit)

            results = await q.to_list()

            logger.debug(
                "[AgentCaseRepo] find_by_filters: user_id=%s, group_ids=%s, "
                "time_range=[%s, %s], found %d records",
                user_id,
                group_ids,
                start_time,
                end_time,
                len(results),
            )
            return results
        except Exception as e:
            logger.error(f"[AgentCaseRepo] Failed to find by filters: {e}")
            return []

    async def count_by_filters(
        self,
        user_id: Optional[str] = MAGIC_ALL,
        group_ids: Optional[List[str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        session: Optional[AsyncClientSession] = None,
    ) -> int:
        """
        Count agent cases by filters (without pagination).

        Args:
            user_id: User ID filter (same semantics as find_by_filters)
            group_ids: Group IDs filter (same semantics as find_by_filters)
            start_time: Optional start time (inclusive)
            end_time: Optional end time (inclusive)
            session: Optional MongoDB session

        Returns:
            Total count of matching records
        """
        try:
            query: Dict[str, Any] = {}

            if user_id != MAGIC_ALL:
                if user_id == "" or user_id is None:
                    query["user_id"] = {"$in": [None, ""]}
                else:
                    query["user_id"] = user_id

            if group_ids is not None and len(group_ids) > 0:
                if len(group_ids) == 1:
                    query["group_id"] = group_ids[0]
                else:
                    query["group_id"] = {"$in": group_ids}

            if start_time is not None or end_time is not None:
                ts_filter = {}
                if start_time is not None:
                    ts_filter["$gte"] = start_time
                if end_time is not None:
                    ts_filter["$lte"] = end_time
                query["timestamp"] = ts_filter

            count = await self.model.find(query, session=session).count()
            logger.debug(
                "[AgentCaseRepo] count_by_filters: user_id=%s, group_ids=%s, count=%d",
                user_id,
                group_ids,
                count,
            )
            return count
        except Exception as e:
            logger.error(f"[AgentCaseRepo] Failed to count by filters: {e}")
            return 0

    async def delete_by_user_id(
        self, user_id: str, session: Optional[AsyncClientSession] = None
    ) -> int:
        """
        Soft-delete all agent cases by user ID.

        Args:
            user_id: User ID
            session: Optional MongoDB session

        Returns:
            Number of soft-deleted records
        """
        try:
            result = await self.model.delete_many({"user_id": user_id}, session=session)
            count = result.modified_count if result else 0
            logger.info(
                "[AgentCaseRepo] Soft-deleted experiences by user_id=%s, count=%d",
                user_id,
                count,
            )
            return count
        except Exception as e:
            logger.error(f"[AgentCaseRepo] Failed to delete by user_id: {e}")
            return 0

    async def delete_by_filters(
        self,
        user_id: Optional[str] = MAGIC_ALL,
        group_id: Optional[str] = MAGIC_ALL,
        parent_id: Optional[str] = None,
        session: Optional[AsyncClientSession] = None,
    ) -> int:
        """
        Soft delete agent cases by filter conditions.

        Args:
            user_id: User ID filter (MAGIC_ALL to skip)
            group_id: Group ID filter (MAGIC_ALL to skip)
            parent_id: Parent ID filter (for cascade delete)
            session: Optional MongoDB session

        Returns:
            Number of deleted records
        """
        try:
            filter_dict: Dict[str, Any] = {}

            if user_id != MAGIC_ALL:
                if user_id == "" or user_id is None:
                    filter_dict["user_id"] = {"$in": [None, ""]}
                else:
                    filter_dict["user_id"] = user_id

            if group_id != MAGIC_ALL:
                if group_id == "" or group_id is None:
                    filter_dict["group_id"] = {"$in": [None, ""]}
                else:
                    filter_dict["group_id"] = group_id

            if parent_id:
                filter_dict["parent_id"] = parent_id

            if not filter_dict:
                logger.warning(
                    "[AgentCaseRepo] No filter conditions for delete_by_filters"
                )
                return 0

            result = await self.model.delete_many(filter_dict, session=session)
            count = result.modified_count if result else 0
            logger.info(
                "[AgentCaseRepo] Deleted experiences by filters: filter=%s, count=%d",
                filter_dict,
                count,
            )
            return count
        except Exception as e:
            logger.error(f"[AgentCaseRepo] Failed to delete by filters: {e}")
            return 0

    async def fetch_task_intents_by_event_ids(
        self, event_ids: List[str]
    ) -> Dict[str, str]:
        """Fetch task_intent texts from AgentCase DB by parent event IDs.

        Used as context_fetcher callback for ClusterManager in LLM mode.

        Args:
            event_ids: List of memcell event IDs (used as parent_id in agent cases)

        Returns:
            Dict mapping event_id -> task_intent text
        """
        if not event_ids:
            return {}

        try:
            cases = (
                await self.model.find({"parent_id": {"$in": event_ids}})
                .project(AgentCaseProjection)
                .to_list()
            )

            result: Dict[str, str] = {}
            for case in cases:
                if case.parent_id and case.task_intent:
                    result[case.parent_id] = case.task_intent
            return result
        except Exception as e:
            logger.error(f"[AgentCaseRepo] Failed to fetch task intents: {e}")
            return {}

    async def find_by_filter_paginated(
        self,
        query_filter: Optional[Dict[str, Any]] = None,
        skip: int = 0,
        limit: int = 100,
        sort_field: str = "created_at",
        sort_desc: bool = False,
    ) -> List[AgentCaseRecord]:
        """
        Paginated query of AgentCaseRecord, used for data synchronization.

        Args:
            query_filter: Query filter conditions, query all if None
            skip: Number of results to skip
            limit: Limit number of returned results
            sort_field: Sort field, default is created_at
            sort_desc: Whether to sort in descending order

        Returns:
            List of AgentCaseRecord
        """
        try:
            filter_dict = query_filter if query_filter else {}
            q = self.model.find(filter_dict)

            if sort_desc:
                q = q.sort(f"-{sort_field}")
            else:
                q = q.sort(sort_field)

            q = q.skip(skip).limit(limit)
            results = await q.to_list()
            logger.debug(
                "[AgentCaseRepo] find_by_filter_paginated: filter=%s, skip=%d, limit=%d, found %d",
                filter_dict,
                skip,
                limit,
                len(results),
            )
            return results
        except Exception as e:
            logger.error(f"[AgentCaseRepo] Failed paginated query: {e}")
            return []

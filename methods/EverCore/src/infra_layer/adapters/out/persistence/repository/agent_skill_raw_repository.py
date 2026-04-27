"""
Agent skill raw data repository.

Provides CRUD operations for agent skill records in MongoDB.
Skills are cluster-scoped: one repository manages all skill items per MemScene.
"""

from typing import List, Optional, Dict, Any
from pymongo.asynchronous.client_session import AsyncClientSession
from bson import ObjectId
from core.observation.logger import get_logger
from core.di.decorators import repository
from core.oxm.mongo.base_repository import BaseRepository
from core.oxm.mongo.mongo_utils import build_id_filter as _build_id_filter
from core.oxm.constants import MAGIC_ALL
from infra_layer.adapters.out.persistence.document.memory.agent_skill import (
    AgentSkillRecord,
)

logger = get_logger(__name__)


@repository("agent_skill_raw_repository", primary=True)
class AgentSkillRawRepository(BaseRepository[AgentSkillRecord]):
    """
    Agent skill raw data repository.

    Manages skill items extracted from MemScene clusters (AgentCase clusters).
    Supports incremental operations: add, update, and soft-delete individual skills.
    """

    def __init__(self):
        super().__init__(AgentSkillRecord)

    async def find_by_ids(
        self,
        ids: List[str],
        min_confidence: Optional[float] = None,
        session: Optional[AsyncClientSession] = None,
    ) -> List[AgentSkillRecord]:
        """Batch fetch agent skills by _id list, optionally filtering retired skills.

        Accepts both ObjectId-like strings and raw string IDs.

        Args:
            ids: List of document _id strings
            min_confidence: Exclude skills with confidence below this threshold.
                None to include all skills regardless of confidence.
            session: Optional MongoDB session

        Returns:
            List of AgentSkillRecord
        """
        id_filter = _build_id_filter(ids)
        if id_filter is None:
            return []
        try:
            if min_confidence is None:
                query: Dict[str, Any] = id_filter
            elif "$or" in id_filter:
                # id_filter is {"$or": [...]}; combine with $and
                query = {"$and": [id_filter, {"confidence": {"$gte": min_confidence}}]}
            else:
                # id_filter is {"_id": {"$in": [...]}}; merge directly
                query = {**id_filter, "confidence": {"$gte": min_confidence}}
            return await self.model.find(query, session=session).to_list()
        except Exception as e:
            logger.error(f"[AgentSkillRepo] Failed to find by ids: {e}")
            return []

    async def save_skill(
        self, record: AgentSkillRecord, session: Optional[AsyncClientSession] = None
    ) -> Optional[AgentSkillRecord]:
        """Insert a new agent skill record."""
        try:
            result = await record.insert(session=session)
            logger.debug(
                f"[AgentSkillRepo] Inserted skill: id={result.id}, "
                f"cluster={result.cluster_id}, name='{result.name}'"
            )
            return result
        except Exception as e:
            logger.error(f"[AgentSkillRepo] Failed to insert skill: {e}")
            return None

    async def get_by_cluster_id(
        self,
        cluster_id: str,
        group_id: Optional[str] = None,
        min_confidence: Optional[float] = None,
        session: Optional[AsyncClientSession] = None,
    ) -> List[AgentSkillRecord]:
        """Retrieve skill records for a cluster (MemScene).

        Args:
            cluster_id: MemScene cluster ID
            group_id: Group ID to scope the query; prevents cross-group reads.
            min_confidence: Exclude skills with confidence below this threshold.
                None to include all skills regardless of confidence.
            session: Optional MongoDB session
        """
        try:
            query: Dict[str, Any] = {"cluster_id": cluster_id}
            if group_id is not None:
                query["group_id"] = group_id
            if min_confidence is not None:
                query["confidence"] = {"$gte": min_confidence}
            results = await self.model.find(query, session=session).to_list()
            return results
        except Exception as e:
            logger.error(f"[AgentSkillRepo] Failed to get by cluster_id: {e}")
            return []

    async def update_skill_by_id(
        self,
        record_id: Any,
        updates: Dict[str, Any],
        session: Optional[AsyncClientSession] = None,
    ) -> bool:
        """Update specific fields of a skill record by its ID.

        Args:
            record_id: The document _id
            updates: Dict of field_name -> new_value to set
            session: Optional MongoDB session

        Returns:
            True if the update was applied, False otherwise.
        """
        try:
            from common_utils.datetime_utils import get_now_with_timezone

            updates["updated_at"] = get_now_with_timezone()
            result = await AgentSkillRecord.get_pymongo_collection().update_one(
                {"_id": record_id, "deleted_at": None},
                {"$set": updates},
                session=session,
            )
            if result.modified_count > 0:
                logger.debug(
                    f"[AgentSkillRepo] Updated skill id={record_id}, "
                    f"fields={list(updates.keys())}"
                )
                return True
            logger.warning(
                f"[AgentSkillRepo] No document matched for update id={record_id}"
            )
            return False
        except Exception as e:
            logger.error(f"[AgentSkillRepo] Failed to update skill id={record_id}: {e}")
            return False

    async def soft_delete_by_id(
        self, record_id: Any, session: Optional[AsyncClientSession] = None
    ) -> bool:
        """Soft-delete a single skill record by its ID.

        Args:
            record_id: The document _id
            session: Optional MongoDB session

        Returns:
            True if deleted, False otherwise.
        """
        try:
            from common_utils.datetime_utils import get_now_with_timezone

            now = get_now_with_timezone()
            result = await AgentSkillRecord.get_pymongo_collection().update_one(
                {"_id": record_id, "deleted_at": None},
                {"$set": {"deleted_at": now, "deleted_id": abs(hash(str(record_id)))}},
                session=session,
            )
            if result.modified_count > 0:
                logger.debug(f"[AgentSkillRepo] Soft-deleted skill id={record_id}")
                return True
            logger.warning(
                f"[AgentSkillRepo] No document matched for soft-delete id={record_id}"
            )
            return False
        except Exception as e:
            logger.error(
                f"[AgentSkillRepo] Failed to soft-delete skill id={record_id}: {e}"
            )
            return False

    def _build_filter_query(
        self,
        user_id: Optional[str] = MAGIC_ALL,
        group_ids: Optional[List[str]] = None,
        cluster_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build a filter query dict from common parameters.

        Args:
            user_id: User ID filter
                - MAGIC_ALL ("__all__"): Don't filter by user_id
                - Other values: Exact match
        """
        query: Dict[str, Any] = {}
        if user_id is not None and user_id != MAGIC_ALL:
            query["user_id"] = user_id
        if group_ids is not None and len(group_ids) > 0:
            if len(group_ids) == 1:
                query["group_id"] = group_ids[0]
            else:
                query["group_id"] = {"$in": group_ids}
        if cluster_id is not None:
            query["cluster_id"] = cluster_id
        return query

    async def find_by_filters(
        self,
        user_id: Optional[str] = MAGIC_ALL,
        group_ids: Optional[List[str]] = None,
        cluster_id: Optional[str] = None,
        limit: int = 50,
        skip: int = 0,
        session: Optional[AsyncClientSession] = None,
    ) -> List[AgentSkillRecord]:
        """Find skill records with flexible filters."""
        try:
            query = self._build_filter_query(
                user_id=user_id, group_ids=group_ids, cluster_id=cluster_id
            )

            results = (
                await self.model.find(query, session=session)
                .skip(skip)
                .limit(limit)
                .to_list()
            )
            return results
        except Exception as e:
            logger.error(f"[AgentSkillRepo] Failed to find by filters: {e}")
            return []

    async def count_by_filters(
        self,
        user_id: Optional[str] = MAGIC_ALL,
        group_ids: Optional[List[str]] = None,
        cluster_id: Optional[str] = None,
        session: Optional[AsyncClientSession] = None,
    ) -> int:
        """Count skill records by filters (without pagination).

        Args:
            user_id: User ID filter
                - MAGIC_ALL ("__all__"): Don't filter by user_id
                - Other values: Exact match
            group_ids: Group IDs filter (list, supports $in for multiple)
            cluster_id: Cluster ID filter
            session: Optional MongoDB session

        Returns:
            Total count of matching records
        """
        try:
            query = self._build_filter_query(
                user_id=user_id, group_ids=group_ids, cluster_id=cluster_id
            )
            count = await self.model.find(query, session=session).count()
            logger.debug(
                "[AgentSkillRepo] count_by_filters: user_id=%s, group_ids=%s, cluster_id=%s, count=%d",
                user_id,
                group_ids,
                cluster_id,
                count,
            )
            return count
        except Exception as e:
            logger.error(f"[AgentSkillRepo] Failed to count by filters: {e}")
            return 0

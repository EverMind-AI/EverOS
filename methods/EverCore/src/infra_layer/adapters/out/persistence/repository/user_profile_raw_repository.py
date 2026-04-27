"""
UserProfile native CRUD repository

User profile data access layer based on Beanie ODM.
Provides ProfileStorage compatible interface (duck typing).

Supports automatic Milvus indexing on profile create/update.
"""

from typing import Optional, Dict, Any, List
from beanie.operators import Or, Eq, In
from core.observation.logger import get_logger
from core.di.decorators import repository
from core.oxm.mongo.base_repository import BaseRepository
from core.oxm.constants import MAGIC_ALL

from api_specs.memory_types import ScenarioType
from infra_layer.adapters.out.persistence.document.memory.user_profile import (
    UserProfile,
)
from api_specs.memory_types import ProfileMemory
from memory_layer.profile_indexer import index_user_profile


logger = get_logger(__name__)


@repository("user_profile_raw_repository", primary=True)
class UserProfileRawRepository(BaseRepository[UserProfile]):
    """
    UserProfile native CRUD repository

    Provides ProfileStorage compatible interfaces:
    - save_profile(user_id, profile, metadata) -> bool
    - get_profile(user_id) -> Optional[Any]
    - get_all_profiles() -> Dict[str, Any]
    - get_profile_history(user_id, limit) -> List[Dict]
    - clear() -> bool
    """

    def __init__(self):
        super().__init__(UserProfile)

    # ==================== ProfileStorage interface implementation ====================

    async def save_profile(
        self, user_id: str, profile: Any, metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        metadata = metadata or {}
        group_id = metadata.get("group_id", "default")

        profile_data = profile.to_dict() if hasattr(profile, 'to_dict') else profile
        result = await self.upsert(user_id, group_id, profile_data, metadata)
        return result is not None

    async def get_profile(
        self, user_id: str, group_id: str = "default"
    ) -> Optional[Any]:
        user_profile = await self.get_by_user_and_group(user_id, group_id)
        if user_profile is None:
            return None
        return user_profile.profile_data

    async def get_all_profiles(self, group_id: str = "default") -> Dict[str, Any]:
        user_profiles = await self.get_all_by_group(group_id)
        return {up.user_id: up.profile_data for up in user_profiles}

    async def get_profile_history(
        self, user_id: str, group_id: str = "default", limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        user_profile = await self.get_by_user_and_group(user_id, group_id)
        if user_profile is None:
            return []

        history = [
            {
                "update_count": user_profile.update_count,
                "profile": user_profile.profile_data,
                "confidence": user_profile.confidence,
                "updated_at": user_profile.updated_at,
                "last_updated_ts": user_profile.last_updated_ts,
                "memcell_count": user_profile.memcell_count,
            }
        ]
        return history[:limit] if limit else history

    async def clear(self, group_id: Optional[str] = None) -> bool:
        if group_id is None:
            await self.delete_all()
        else:
            await self.delete_by_group(group_id)
        return True

    # ==================== Native CRUD methods ====================

    async def get_by_user_and_group(
        self, user_id: str, group_id: str
    ) -> Optional[UserProfile]:
        try:
            return await self.model.find_one(
                UserProfile.user_id == user_id, UserProfile.group_id == group_id
            )
        except Exception as e:
            logger.error(
                f"Failed to retrieve user profile: user_id={user_id}, group_id={group_id}, error={e}"
            )
            return None

    async def get_all_by_group(self, group_id: str) -> List[UserProfile]:
        try:
            return await self.model.find(UserProfile.group_id == group_id).to_list()
        except Exception as e:
            logger.error(
                f"Failed to retrieve group user profiles: group_id={group_id}, error={e}"
            )
            return []

    async def get_all_by_user(self, user_id: str, limit: int = 40) -> List[UserProfile]:
        try:
            return (
                await self.model.find(UserProfile.user_id == user_id)
                .sort([("update_count", -1)])
                .limit(limit)
                .to_list()
            )
        except Exception as e:
            logger.error(f"Failed to get user profile: user_id={user_id}, error={e}")
            return []

    async def find_by_filters(
        self,
        user_id: Optional[str] = MAGIC_ALL,
        group_ids: Optional[List[str]] = None,
        limit: Optional[int] = None,
        skip: Optional[int] = None,
    ) -> List[UserProfile]:
        """
        Retrieve list of user profiles by filters (user_id and/or group_ids)

        Args:
            user_id: User ID
                - Not provided or MAGIC_ALL ("__all__"): Don't filter by user_id
                - None or "": Filter for null/empty values (records with user_id as None or "")
                - Other values: Exact match
            group_ids: List of Group IDs
                - None: Skip group filtering
                - []: Empty array, skip filtering
                - ["g1"]: Single element array, exact match
                - ["g1", "g2"]: Multiple elements, use In operator
            limit: Limit number of returned results
            skip: Number of results to skip (pagination offset)

        Returns:
            List of UserProfile
        """
        try:
            # Build query conditions
            conditions = []

            # Handle user_id filter
            if user_id != MAGIC_ALL:
                if user_id == "" or user_id is None:
                    # Explicitly filter for null or empty string
                    conditions.append(
                        Or(Eq(UserProfile.user_id, None), Eq(UserProfile.user_id, ""))
                    )
                else:
                    conditions.append(UserProfile.user_id == user_id)

            # Handle group_ids filter (array, no MAGIC_ALL)
            if group_ids is not None and len(group_ids) > 0:
                if len(group_ids) == 1:
                    # Single element: exact match
                    conditions.append(UserProfile.group_id == group_ids[0])
                else:
                    # Multiple elements: use In operator
                    conditions.append(In(UserProfile.group_id, group_ids))
            # group_ids is None or empty: skip group filtering

            # Build query
            if conditions:
                # Combine conditions with AND
                query = self.model.find(*conditions)
            else:
                # No conditions - find all
                query = self.model.find()

            # Sort by update_count descending
            query = query.sort([("update_count", -1)])

            # Apply skip (offset)
            if skip:
                query = query.skip(skip)

            # Apply limit
            if limit:
                query = query.limit(limit)

            logger.debug(
                "🔍 UserProfile.find_by_filters query: %s, sort=[('update_count', -1)], skip=%s, limit=%s",
                query.get_filter_query(),
                skip,
                limit,
            )

            results = await query.to_list()
            logger.debug(
                "✅ Retrieved user profiles successfully: user_id=%s, group_ids=%s, found %d records",
                user_id,
                group_ids,
                len(results),
            )
            return results
        except Exception as e:
            logger.error("❌ Failed to retrieve user profiles: %s", e)
            return []

    async def count_by_filters(
        self, user_id: Optional[str] = MAGIC_ALL, group_ids: Optional[List[str]] = None
    ) -> int:
        """
        Count user profiles by filters (without pagination)

        Args:
            user_id: User ID filter (same semantics as find_by_filters)
            group_ids: Group IDs filter (same semantics as find_by_filters)

        Returns:
            Total count of matching records
        """
        try:
            # Build query conditions (same as find_by_filters)
            conditions = []

            # Handle user_id filter
            if user_id != MAGIC_ALL:
                if user_id == "" or user_id is None:
                    conditions.append(
                        Or(Eq(UserProfile.user_id, None), Eq(UserProfile.user_id, ""))
                    )
                else:
                    conditions.append(UserProfile.user_id == user_id)

            # Handle group_ids filter
            if group_ids is not None and len(group_ids) > 0:
                if len(group_ids) == 1:
                    conditions.append(UserProfile.group_id == group_ids[0])
                else:
                    conditions.append(In(UserProfile.group_id, group_ids))

            # Build query
            if conditions:
                query = self.model.find(*conditions)
            else:
                query = self.model.find()

            count = await query.count()
            logger.debug(
                "✅ Counted user profiles: user_id=%s, group_ids=%s, count=%d",
                user_id,
                group_ids,
                count,
            )
            return count
        except Exception as e:
            logger.error("❌ Failed to count user profiles: %s", e)
            return 0

    async def upsert(
        self,
        user_id: str,
        group_id: str,
        profile_data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        trigger_index: bool = True,
    ) -> Optional[UserProfile]:
        """
        Create or update user profile with optional Milvus indexing

        Args:
            user_id: User ID
            group_id: Group ID
            profile_data: Profile data dict (containing explicit_info, implicit_traits, etc.)
            metadata: Additional metadata (confidence, cluster_id, memcell_count, etc.)
            trigger_index: Whether to trigger Milvus indexing after save (default True)

        Returns:
            Saved UserProfile object or None on error
        """
        try:
            metadata = metadata or {}
            existing = await self.get_by_user_and_group(user_id, group_id)

            if existing:
                existing.profile_data = profile_data
                existing.update_count += 1
                existing.confidence = metadata.get("confidence", existing.confidence)

                if "memcell_count" in metadata:
                    existing.memcell_count = metadata["memcell_count"]

                if "last_updated_ts" in metadata:
                    existing.last_updated_ts = metadata["last_updated_ts"]

                await existing.save()
                logger.debug(
                    f"Updated user profile: user_id={user_id}, group_id={group_id}, update_count={existing.update_count}"
                )
                saved_profile = existing
            else:
                user_profile = UserProfile(
                    user_id=user_id,
                    group_id=group_id,
                    profile_data=profile_data,
                    scenario=metadata.get("scenario", ScenarioType.TEAM.value),
                    confidence=metadata.get("confidence", 0.0),
                    update_count=1,
                    memcell_count=metadata.get("memcell_count", 0),
                    last_updated_ts=metadata.get("last_updated_ts"),
                )
                await user_profile.insert()
                logger.info(
                    f"Created user profile: user_id={user_id}, group_id={group_id}"
                )
                saved_profile = user_profile

            # Trigger Milvus indexing (runs in clustering background task, not on hot path)
            if trigger_index:
                await self._trigger_milvus_indexing(user_id, group_id, profile_data, doc_id=str(saved_profile.id))

            return saved_profile

        except Exception as e:
            logger.error(
                f"Failed to save user profile: user_id={user_id}, group_id={group_id}, error={e}"
            )
            return None

    async def _trigger_milvus_indexing(
        self, user_id: str, group_id: str, profile_data: Dict[str, Any], doc_id: str = ""
    ) -> None:
        """
        Trigger Milvus indexing for profile (delete-then-insert strategy)

        This runs asynchronously to avoid blocking the main save operation.

        Args:
            user_id: User ID
            group_id: Group ID
            profile_data: Profile data dict
            doc_id: MongoDB document ID for generating unique Milvus entity IDs
        """
        try:
            # Convert profile_data dict to ProfileMemory object
            profile = ProfileMemory.from_dict(
                profile_data, user_id=user_id, group_id=group_id
            )

            # Trigger indexing (delete existing + insert new)
            stats = await index_user_profile(user_id, group_id, profile, doc_id=doc_id)

            logger.info(
                f"✅ Profile Milvus indexing completed: user_id={user_id}, group_id={group_id}, "
                f"deleted={stats.get('deleted_count', 0)}, indexed={stats.get('total_count', 0)}"
            )

        except Exception as e:
            # Log error but don't fail the main operation
            logger.error(
                f"❌ Failed to trigger Milvus indexing: user_id={user_id}, group_id={group_id}, error={e}",
                exc_info=True,
            )

    async def delete_by_group(self, group_id: str) -> int:
        try:
            result = await self.model.find(UserProfile.group_id == group_id).delete()
            count = result.deleted_count if result else 0
            logger.info(
                f"Deleted group user profiles: group_id={group_id}, count={count}"
            )
            return count
        except Exception as e:
            logger.error(
                f"Failed to delete group user profiles: group_id={group_id}, error={e}"
            )
            return 0

    async def delete_all(self) -> int:
        try:
            result = await self.model.delete_all()
            count = result.deleted_count if result else 0
            logger.info(f"Deleted all user profiles: {count} items")
            return count
        except Exception as e:
            logger.error(f"Failed to delete all user profiles: {e}")
            return 0

"""
Memory retrieval service

This module provides a service layer interface for accessing memory data, interfacing with repository classes that access the database.
Provides ID-based query functionality, supporting retrieval of various memory types.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

from core.di import get_bean_by_type, get_bean, service
from core.oxm.constants import MAGIC_ALL
from common_utils.datetime_utils import from_iso_format
from infra_layer.adapters.out.persistence.document.memory.foresight_record import (
    ForesightRecord,
    ForesightRecordProjection,
)
from infra_layer.adapters.out.persistence.repository.episodic_memory_raw_repository import (
    EpisodicMemoryRawRepository,
)
from infra_layer.adapters.out.persistence.repository.core_memory_raw_repository import (
    CoreMemoryRawRepository,
)
from infra_layer.adapters.out.persistence.repository.behavior_history_raw_repository import (
    BehaviorHistoryRawRepository,
)
from infra_layer.adapters.out.persistence.repository.conversation_meta_raw_repository import (
    ConversationMetaRawRepository,
)
from infra_layer.adapters.out.persistence.repository.event_log_record_raw_repository import (
    EventLogRecordRawRepository,
)
from infra_layer.adapters.out.persistence.document.memory.event_log_record import (
    EventLogRecord,
    EventLogRecordProjection,
)
from infra_layer.adapters.out.persistence.repository.foresight_record_repository import (
    ForesightRecordRawRepository,
)
from infra_layer.adapters.out.persistence.document.memory.foresight_record import (
    ForesightRecordProjection,
)
from infra_layer.adapters.out.persistence.repository.user_profile_raw_repository import (
    UserProfileRawRepository,
)
from infra_layer.adapters.out.persistence.repository.global_user_profile_raw_repository import (
    GlobalUserProfileRawRepository,
)
from api_specs.dtos import FetchMemResponse
from api_specs.memory_models import (
    MemoryType,
    BaseMemoryModel,
    ProfileModel,
    GlobalUserProfileModel,
    CombinedProfileModel,
    PreferenceModel,
    EpisodicMemoryModel,
    BehaviorHistoryModel,
    CoreMemoryModel,
    EventLogModel,
    ForesightModel,
    Metadata,
)

logger = logging.getLogger(__name__)


class FetchMemoryServiceInterface(ABC):
    """Memory retrieval service interface"""

    @abstractmethod
    async def find_memories(
        self,
        user_id: str,
        memory_type: MemoryType,
        group_id: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        version_range: Optional[Tuple[Optional[str], Optional[str]]] = None,
        limit: int = 10,
        full: bool = False,
    ) -> FetchMemResponse:
        """
        Find memories by user ID and optional filters

        Args:
            user_id: User ID
            memory_type: Memory type
            group_id: Group ID for group memory retrieval (optional)
            start_time: Start time for time range filtering (optional)
            end_time: End time for time range filtering (optional)
            version_range: Version range (start, end), closed interval [start, end]
            limit: Limit on number of returned items
            full: When True, returns full episode content for episodic memory

        Returns:
            Memory query response
        """
        pass


@service(name="fetch_memory_service", primary=True)
class FetchMemoryServiceImpl(FetchMemoryServiceInterface):
    """Real implementation of memory retrieval service

    Uses repository instances injected by DI framework for database access.
    """

    def __init__(self):
        """Initialize service"""
        self._episodic_repo = None
        self._core_repo = None
        self._behavior_repo = None
        self._conversation_meta_repo = None
        self._event_log_repo = None
        self._foresight_record_repo = None
        self._user_profile_repo = None
        self._global_user_profile_repo = None
        logger.info("FetchMemoryServiceImpl initialized")

    def _get_repositories(self):
        """Get repository instances"""
        if self._episodic_repo is None:
            self._episodic_repo = get_bean_by_type(EpisodicMemoryRawRepository)
        if self._core_repo is None:
            self._core_repo = get_bean_by_type(CoreMemoryRawRepository)
        if self._behavior_repo is None:
            self._behavior_repo = get_bean_by_type(BehaviorHistoryRawRepository)
        if self._conversation_meta_repo is None:
            self._conversation_meta_repo = get_bean_by_type(
                ConversationMetaRawRepository
            )
        if self._event_log_repo is None:
            self._event_log_repo = get_bean_by_type(EventLogRecordRawRepository)
        if self._foresight_record_repo is None:
            self._foresight_record_repo = get_bean_by_type(ForesightRecordRawRepository)
        if self._user_profile_repo is None:
            self._user_profile_repo = get_bean_by_type(UserProfileRawRepository)
        if self._global_user_profile_repo is None:
            self._global_user_profile_repo = get_bean_by_type(
                GlobalUserProfileRawRepository
            )

    async def _get_user_details_cache(self, group_id: str) -> dict:
        """
        Get user details cache from conversation-meta for batch processing

        Args:
            group_id: Group ID

        Returns:
            Dictionary mapping user_id to user details (full_name, email, phone)
        """
        try:
            if not group_id or group_id == MAGIC_ALL:
                return {}

            # Ensure repository is initialized
            if self._conversation_meta_repo is None:
                self._get_repositories()

            # Query conversation metadata
            conversation_meta = await self._conversation_meta_repo.get_by_group_id(
                group_id
            )

            if not conversation_meta or not conversation_meta.user_details:
                return {}

            # Build user details cache
            user_cache = {}
            for uid, user_detail in conversation_meta.user_details.items():
                user_cache[uid] = {
                    'full_name': user_detail.full_name,
                    'email': (
                        user_detail.extra.get('email') if user_detail.extra else None
                    ),
                    'phone': (
                        user_detail.extra.get('phone') if user_detail.extra else None
                    ),
                }

            return user_cache

        except Exception as e:
            logger.warning(f"Failed to get user details cache: {e}")
            return {}

    def _convert_base_memory(self, core_memory) -> BaseMemoryModel:
        """Convert core memory to base memory model

        Args:
            core_memory: Core memory document

        Returns:
            BaseMemoryModel with basic user information
        """
        base_info = self._core_repo.get_base(core_memory)

        return BaseMemoryModel(
            id=str(core_memory.id),
            user_id=core_memory.user_id,
            content=f"User: {base_info.get('user_name', 'Unknown')} | Position: {base_info.get('position', 'Unknown')} | Department: {base_info.get('department', 'Unknown')}",
            created_at=core_memory.created_at,
            updated_at=core_memory.updated_at,
            metadata={
                "user_name": base_info.get('user_name', ''),
                "position": base_info.get('position', ''),
                "department": base_info.get('department', ''),
                "company": base_info.get('company', ''),
                "location": base_info.get('location', ''),
                "contact": base_info.get('contact', {}),
            },
        )

    def _convert_user_profile(self, user_profile) -> ProfileModel:
        """Convert user profile document to ProfileModel

        Args:
            user_profile: User profile document

        Returns:
            ProfileModel instance
        """
        return ProfileModel(
            id=str(user_profile.id),
            user_id=user_profile.user_id,
            group_id=user_profile.group_id,
            profile_data=user_profile.profile_data,
            scenario=user_profile.scenario,
            confidence=user_profile.confidence,
            version=user_profile.version,
            cluster_ids=user_profile.cluster_ids,
            memcell_count=user_profile.memcell_count,
            last_updated_cluster=user_profile.last_updated_cluster,
            created_at=user_profile.created_at,
            updated_at=user_profile.updated_at,
        )

    def _convert_global_user_profile(
        self, global_user_profile
    ) -> GlobalUserProfileModel:
        """Convert global user profile document to GlobalUserProfileModel

        Args:
            global_user_profile: Global user profile document

        Returns:
            GlobalUserProfileModel instance
        """
        return GlobalUserProfileModel(
            id=str(global_user_profile.id),
            user_id=global_user_profile.user_id,
            profile_data=global_user_profile.profile_data,
            custom_profile_data=global_user_profile.custom_profile_data,
            confidence=global_user_profile.confidence,
            memcell_count=global_user_profile.memcell_count,
            created_at=global_user_profile.created_at,
            updated_at=global_user_profile.updated_at,
        )

    def _convert_preferences_from_core_memory(
        self, core_memory
    ) -> list[PreferenceModel]:
        """Convert core memory to preference models

        Args:
            core_memory: Core memory document

        Returns:
            List of PreferenceModel instances
        """
        preference_info = self._core_repo.get_preference(core_memory)
        memories = []

        for key, value in preference_info.items():
            memories.append(
                PreferenceModel(
                    id=f"{core_memory.id}_{key}",
                    user_id=core_memory.user_id,
                    category="Personal preference",
                    preference_key=key,
                    preference_value=str(value),
                    confidence_score=1.0,
                    created_at=core_memory.created_at,
                    updated_at=core_memory.updated_at,
                    metadata={"source": "core_memory", "original_key": key},
                )
            )

        return memories

    def _convert_core_memory(
        self, core_memory, metadata: Metadata = None
    ) -> CoreMemoryModel:
        """Convert core memory document to model"""
        # If no metadata provided, create a simple one
        if metadata is None:
            metadata = Metadata(
                source=MemoryType.CORE.value,
                user_id=core_memory.user_id,
                memory_type=MemoryType.CORE.value,
            )

        return CoreMemoryModel(
            id=str(core_memory.id),
            user_id=core_memory.user_id,
            version=core_memory.version,
            is_latest=core_memory.is_latest,
            # BaseMemory fields
            user_name=core_memory.user_name,
            gender=core_memory.gender,
            position=core_memory.position,
            supervisor_user_id=core_memory.supervisor_user_id,
            team_members=core_memory.team_members,
            okr=core_memory.okr,
            base_location=core_memory.base_location,
            hiredate=core_memory.hiredate,
            age=core_memory.age,
            department=core_memory.department,
            # Profile fields
            hard_skills=core_memory.hard_skills,
            soft_skills=core_memory.soft_skills,
            output_reasoning=core_memory.output_reasoning,
            motivation_system=core_memory.motivation_system,
            fear_system=core_memory.fear_system,
            value_system=core_memory.value_system,
            humor_use=core_memory.humor_use,
            colloquialism=core_memory.colloquialism,
            personality=core_memory.personality,
            way_of_decision_making=core_memory.way_of_decision_making,
            projects_participated=core_memory.projects_participated,
            user_goal=core_memory.user_goal,
            work_responsibility=core_memory.work_responsibility,
            working_habit_preference=core_memory.working_habit_preference,
            interests=core_memory.interests,
            tendency=core_memory.tendency,
            # Common fields
            extend=core_memory.extend,
            created_at=core_memory.created_at,
            updated_at=core_memory.updated_at,
            metadata=metadata,
        )

    def _convert_episodic_memory(
        self, episodic_memory, user_details_cache: dict = None, full: bool = False
    ) -> EpisodicMemoryModel:
        """Convert episodic memory document to model

        Args:
            episodic_memory: Episodic memory document
            user_details_cache: User details cache for batch metadata creation
            full: When True, includes the full episode content
        """
        # Create metadata with user details from cache
        user_info = (
            user_details_cache.get(episodic_memory.user_id, {})
            if user_details_cache
            else {}
        )
        metadata = Metadata(
            source=MemoryType.EPISODIC_MEMORY.value,
            user_id=episodic_memory.user_id,
            group_id=episodic_memory.group_id,
            memory_type=MemoryType.EPISODIC_MEMORY.value,
            full_name=user_info.get('full_name'),
            email=user_info.get('email'),
            phone=user_info.get('phone'),
        )

        # Build the model - include episode only when full=True
        model_kwargs = {
            "id": str(episodic_memory.id),
            "user_id": episodic_memory.user_id,
            "episode_id": str(episodic_memory.event_id),
            "title": episodic_memory.subject,
            "summary": episodic_memory.summary,
            "participants": episodic_memory.participants or [],
            "location": (
                episodic_memory.extend.get("location", "")
                if episodic_memory.extend
                else ""
            ),
            "key_events": episodic_memory.keywords or [],
            "group_id": episodic_memory.group_id,
            "group_name": episodic_memory.group_name,
            "created_at": episodic_memory.created_at,
            "updated_at": episodic_memory.updated_at,
            "metadata": metadata,
        }

        # Add episode field when full=True (backward compatible)
        if full:
            model_kwargs["episode"] = getattr(episodic_memory, "episode", None)

        return EpisodicMemoryModel(**model_kwargs)

    def _convert_behavior_history(self, behavior) -> BehaviorHistoryModel:
        """Convert behavior history document to model"""
        return BehaviorHistoryModel(
            id=str(behavior.id),
            user_id=behavior.user_id,
            action_type=(
                behavior.behavior_type[0]
                if behavior.behavior_type
                else "Unknown behavior"
            ),
            action_description=f"Behavior type: {behavior.behavior_type}",
            context=behavior.meta or {},
            result="Success",
            session_id=behavior.event_id,
            created_at=behavior.created_at,
            updated_at=behavior.updated_at,
            metadata=Metadata(
                source=MemoryType.BEHAVIOR_HISTORY.value,
                user_id=behavior.user_id,
                memory_type=MemoryType.BEHAVIOR_HISTORY.value,
            ),
        )

    def _convert_event_log(
        self,
        event_log: Union[EventLogRecord, EventLogRecordProjection],
        user_details_cache: dict = None,
    ) -> EventLogModel:
        """Convert event log document to model

        Supports both EventLogRecord and EventLogRecordShort types.
        EventLogRecordShort does not contain the vector field.

        Args:
            event_log: Event log document
            user_details_cache: User details cache for batch metadata creation
        """
        # Create metadata with user details from cache
        user_info = (
            user_details_cache.get(event_log.user_id, {}) if user_details_cache else {}
        )
        metadata = Metadata(
            source=MemoryType.EVENT_LOG.value,
            user_id=event_log.user_id,
            group_id=event_log.group_id,
            memory_type=MemoryType.EVENT_LOG.value,
            full_name=user_info.get('full_name'),
            email=user_info.get('email'),
            phone=user_info.get('phone'),
        )

        return EventLogModel(
            id=str(event_log.id),
            user_id=event_log.user_id,
            atomic_fact=event_log.atomic_fact,
            parent_type=event_log.parent_type,
            parent_id=event_log.parent_id,
            timestamp=event_log.timestamp,
            user_name=event_log.user_name,
            group_id=event_log.group_id,
            group_name=event_log.group_name,
            participants=event_log.participants,
            vector=getattr(
                event_log, 'vector', None
            ),  # EventLogRecordShort does not have vector field
            vector_model=event_log.vector_model,
            event_type=event_log.event_type,
            extend=event_log.extend,
            created_at=event_log.created_at,
            updated_at=event_log.updated_at,
            metadata=metadata,
        )

    def _convert_foresight_record(
        self,
        foresight_record: Union[ForesightRecord, ForesightRecordProjection],
        user_details_cache: dict = None,
    ) -> ForesightModel:
        """Convert foresight record document to model

        Supports both ForesightRecord and ForesightRecordProjection types.
        ForesightRecordProjection does not contain the vector field.

        Args:
            foresight_record: Foresight record document
            user_details_cache: User details cache for batch metadata creation
        """
        # Create metadata with user details from cache
        uid = foresight_record.user_id or ""
        user_info = user_details_cache.get(uid, {}) if user_details_cache else {}
        metadata = Metadata(
            source=MemoryType.FORESIGHT.value,
            user_id=uid,
            group_id=foresight_record.group_id,
            memory_type=MemoryType.FORESIGHT.value,
            full_name=user_info.get('full_name'),
            email=user_info.get('email'),
            phone=user_info.get('phone'),
        )

        return ForesightModel(
            id=str(foresight_record.id),
            content=foresight_record.content,
            parent_type=foresight_record.parent_type,
            parent_id=foresight_record.parent_id,
            user_id=foresight_record.user_id,
            user_name=foresight_record.user_name,
            group_id=foresight_record.group_id,
            group_name=foresight_record.group_name,
            start_time=foresight_record.start_time,
            end_time=foresight_record.end_time,
            duration_days=foresight_record.duration_days,
            participants=foresight_record.participants,
            vector=getattr(
                foresight_record, 'vector', None
            ),  # ForesightRecordProjection does not have vector field
            vector_model=foresight_record.vector_model,
            evidence=foresight_record.evidence,
            extend=foresight_record.extend,
            created_at=foresight_record.created_at,
            updated_at=foresight_record.updated_at,
            metadata=metadata,
        )

    async def find_memories(
        self,
        user_id: str,
        memory_type: MemoryType,
        group_id: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        version_range: Optional[Tuple[Optional[str], Optional[str]]] = None,
        limit: int = 10,
        full: bool = False,
    ) -> FetchMemResponse:
        """
        Find memories by user ID and optional filters

        Args:
            user_id: User ID (MAGIC_ALL to skip user filtering)
            memory_type: Memory type
            group_id: Group ID for group memory retrieval (MAGIC_ALL to skip group filtering)
            start_time: Start time for time range filtering (ISO format string)
            end_time: End time for time range filtering (ISO format string)
            version_range: Version range (start, end), closed interval [start, end].
                          If not provided or None, get the latest version (ordered by version descending)
            limit: Limit on number of returned items
            full: When True, returns full episode content for episodic memory

        Returns:
            Memory query response

        Time Field Mapping by Memory Type:
        ----------------------------------
        The start_time and end_time parameters map to different fields based on memory type:

        - EPISODIC_MEMORY: Filters by `timestamp` field (event occurrence time)
        - EVENT_LOG: Filters by `timestamp` field (log record time)
        - FORESIGHT: Filters by validity period overlap (`start_time`, `end_time` fields)
                    Uses overlap logic: foresight active if [foresight.start, foresight.end] overlaps [query.start, query.end]
        - PROFILE: No time filtering supported (only has `created_at`, `updated_at` audit fields)
        - BASE_MEMORY: No time filtering supported (core memory snapshot)
        - PREFERENCE: No time filtering supported (extracted from core memory)
        - ENTITY: No time filtering supported in current implementation
        - RELATION: No time filtering supported in current implementation
        - BEHAVIOR_HISTORY: No time filtering supported in current implementation
        """
        logger.debug(
            f"Fetching {memory_type} memories for user_id={user_id}, group_id={group_id}, "
            f"time_range=[{start_time}, {end_time}], limit={limit}"
        )

        try:
            self._get_repositories()
            memories = []

            # Parse time range if provided
            start_dt = from_iso_format(start_time) if start_time else None
            end_dt = from_iso_format(end_time) if end_time else None

            # Fetch user details cache once for batch metadata creation
            # This optimizes performance by querying conversation_meta only once
            user_details_cache = await self._get_user_details_cache(group_id)
            logger.debug(
                f"Fetched user details cache with {len(user_details_cache)} users"
            )

            match memory_type:
                case MemoryType.FORESIGHT:
                    # Foresight: supports group_id filtering and time range overlap queries
                    # Time filtering is based on foresight validity period (start_time, end_time fields)
                    foresight_records = (
                        await self._foresight_record_repo.find_by_filters(
                            user_id=user_id,
                            group_id=group_id,
                            start_time=start_dt,
                            end_time=end_dt,
                            limit=limit,
                            model=ForesightRecordProjection,
                        )
                    )

                    memories = [
                        self._convert_foresight_record(
                            record, user_details_cache=user_details_cache
                        )
                        for record in foresight_records
                    ]

                case MemoryType.EPISODIC_MEMORY:
                    # Episodic memory: fully supports group_id and timestamp filtering at DB level
                    episodic_memories = await self._episodic_repo.find_by_filters(
                        user_id=user_id,
                        group_id=group_id,
                        start_time=start_dt,
                        end_time=end_dt,
                        limit=limit,
                        sort_desc=True,
                    )

                    memories = [
                        self._convert_episodic_memory(
                            mem, user_details_cache=user_details_cache, full=full
                        )
                        for mem in episodic_memories
                    ]
                case MemoryType.EVENT_LOG:
                    # Event log: fully supports group_id and timestamp filtering at DB level
                    event_logs = await self._event_log_repo.find_by_filters(
                        user_id=user_id,
                        group_id=group_id,
                        start_time=start_dt,
                        end_time=end_dt,
                        limit=limit,
                        sort_desc=True,
                        model=EventLogRecordProjection,
                    )

                    memories = [
                        self._convert_event_log(
                            event_log, user_details_cache=user_details_cache
                        )
                        for event_log in event_logs
                    ]

                case MemoryType.PROFILE:
                    # Profile: supports user_id and group_id filtering, no time filtering
                    # Uses created_at/updated_at fields (not time range filterable)
                    # Also fetches global_user_profile and returns CombinedProfileModel

                    # Fetch user_profiles and global_user_profile concurrently
                    user_profiles_task = self._user_profile_repo.find_by_filters(
                        user_id=user_id, group_id=group_id, limit=limit
                    )

                    global_profile_task = None
                    if user_id and user_id != MAGIC_ALL:
                        global_profile_task = (
                            self._global_user_profile_repo.get_by_user_id(
                                user_id=user_id
                            )
                        )

                    # Execute concurrently
                    if global_profile_task:
                        user_profiles, global_user_profile = await asyncio.gather(
                            user_profiles_task, global_profile_task
                        )
                    else:
                        user_profiles = await user_profiles_task
                        global_user_profile = None

                    profile_models = [
                        self._convert_user_profile(up) for up in user_profiles[:limit]
                    ]

                    global_profile_model = None
                    if global_user_profile:
                        global_profile_model = self._convert_global_user_profile(
                            global_user_profile
                        )

                    # Return CombinedProfileModel containing both profiles
                    combined_profile = CombinedProfileModel(
                        user_id=user_id,
                        group_id=group_id,
                        profiles=profile_models,
                        global_profile=global_profile_model,
                    )
                    memories = [combined_profile]

                case MemoryType.BASE_MEMORY:
                    # Base memory: extract basic information from core memory
                    # Does NOT support group_id or time filtering (single record per user)
                    if user_id and user_id != MAGIC_ALL:
                        core_memory = await self._core_repo.get_by_user_id(user_id)
                        if core_memory:
                            memories = [self._convert_base_memory(core_memory)]
                        else:
                            memories = []
                    else:
                        logger.warning("BASE_MEMORY requires a specific user_id")
                        memories = []

                case MemoryType.PREFERENCE:
                    # Preferences: extract preference settings from core memory
                    # Does NOT support group_id or time filtering (single record per user)
                    if user_id and user_id != MAGIC_ALL:
                        core_memory = await self._core_repo.get_by_user_id(user_id)
                        if core_memory:
                            memories = self._convert_preferences_from_core_memory(
                                core_memory
                            )
                        else:
                            memories = []
                    else:
                        logger.warning("PREFERENCE requires a specific user_id")
                        memories = []

                case MemoryType.BEHAVIOR_HISTORY:
                    # Behavior history: user behaviors sorted by time
                    # Does NOT support group_id or time filtering in current implementation
                    # TODO: BehaviorHistory repository needs enhancement for filtering
                    if user_id and user_id != MAGIC_ALL:
                        behaviors = await self._behavior_repo.get_by_user_id(
                            user_id, limit=limit
                        )
                        memories = [
                            self._convert_behavior_history(behavior)
                            for behavior in behaviors
                        ]
                    else:
                        logger.warning("BEHAVIOR_HISTORY requires a specific user_id")
                        memories = []
            # Create response-level metadata (for the query itself)
            # This is query-level metadata, not user-specific
            response_metadata = Metadata(
                source=memory_type.value,
                user_id=user_id,
                group_id=group_id,
                memory_type=memory_type.value,
                limit=limit,
            )

            return FetchMemResponse(
                memories=memories,
                total_count=len(memories),
                has_more=len(memories) == limit,
                metadata=response_metadata,
            )

        except Exception as e:
            import traceback

            traceback.print_exc()
            logger.error(
                f"Error fetching memories for user_id={user_id}, group_id={group_id}: {e}",
                exc_info=True,
            )
            # Return error response with basic metadata
            error_metadata = Metadata(
                source=memory_type.value,
                user_id=user_id,
                group_id=group_id,
                memory_type=memory_type.value,
                limit=limit,
            )

            return FetchMemResponse(
                memories=[], total_count=0, has_more=False, metadata=error_metadata
            )


def get_fetch_memory_service() -> FetchMemoryServiceInterface:
    """Get memory retrieval service instance

    Retrieve service instance via dependency injection framework, supporting singleton pattern.
    """
    return get_bean("fetch_memory_service")

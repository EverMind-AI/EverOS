"""
Memory GET service (v1)

Provides business logic for POST /api/v1/memories/get endpoint.
Handles filter parsing, scope validation, pagination, querying, and DTO conversion.
"""

import logging
from typing import Any, Dict, List

from pymongo import ASCENDING, DESCENDING

from api_specs.memory_models import MemoryType

from core.di import service, get_bean_by_type
from core.observation.stage_timer import timed
from agentic_layer.filter_parser import parse_mongo_filters
from api_specs.dtos.memory import (
    GetMemResponse,
    EpisodeItem,
    ProfileItem,
    AgentCaseItem,
    AgentSkillItem,
)
from infra_layer.adapters.out.persistence.document.memory.episodic_memory import (
    EpisodicMemoryProjection,
)
from infra_layer.adapters.out.persistence.document.memory.agent_case import (
    AgentCaseProjection,
)
from infra_layer.adapters.out.persistence.document.memory.agent_skill import (
    AgentSkillProjection,
)
from infra_layer.adapters.out.persistence.repository.episodic_memory_raw_repository import (
    EpisodicMemoryRawRepository,
)
from infra_layer.adapters.out.persistence.repository.user_profile_raw_repository import (
    UserProfileRawRepository,
)
from infra_layer.adapters.out.persistence.repository.agent_case_raw_repository import (
    AgentCaseRawRepository,
)
from infra_layer.adapters.out.persistence.repository.agent_skill_raw_repository import (
    AgentSkillRawRepository,
)
from biz_layer.memorize_config import DEFAULT_MEMORIZE_CONFIG

logger = logging.getLogger(__name__)


class InvalidScopeError(Exception):
    """Raised when filters lack required user_id or group_id scope."""

    pass


@service(name="get_memory_service", primary=True)
class GetMemoryService:
    """Memory GET service for v1 API.

    Handles filter parsing, scope validation, pagination, querying,
    and document-to-DTO conversion. All DB access goes through repository layer.
    """

    def __init__(self):
        self._episodic_repo = get_bean_by_type(EpisodicMemoryRawRepository)
        self._profile_repo = get_bean_by_type(UserProfileRawRepository)
        self._agent_case_repo = get_bean_by_type(AgentCaseRawRepository)
        self._agent_skill_repo = get_bean_by_type(AgentSkillRawRepository)

    @staticmethod
    def _episode_to_item(doc) -> EpisodeItem:
        """Convert EpisodicMemoryProjection document to EpisodeItem."""
        return EpisodeItem(
            id=str(doc.id),
            user_id=doc.user_id,
            group_id=doc.group_id,
            session_id=doc.session_id,
            timestamp=doc.timestamp,
            participants=doc.participants,
            sender_ids=doc.sender_ids,
            summary=doc.summary,
            subject=doc.subject,
            episode=doc.episode,
            type=doc.type,
            parent_type=doc.parent_type,
            parent_id=doc.parent_id,
        )

    @staticmethod
    def _profile_to_item(doc) -> ProfileItem:
        """Convert UserProfile document to ProfileItem."""
        return ProfileItem(
            id=str(doc.id),
            user_id=doc.user_id,
            group_id=doc.group_id,
            profile_data=doc.profile_data,
            scenario=doc.scenario,
            memcell_count=doc.memcell_count,
        )

    # Memory types that lack a 'timestamp' field; fall back to updated_at.
    _NO_TIMESTAMP_TYPES = {MemoryType.PROFILE.value, MemoryType.AGENT_SKILL.value}

    # Memory types that require user_id in filters (only group_id is not allowed).
    _USER_REQUIRED_TYPES = {
        MemoryType.PROFILE.value,
        MemoryType.AGENT_CASE.value,
        MemoryType.AGENT_SKILL.value,
    }

    @staticmethod
    def _resolve_sort(memory_type: str, rank_by: str, rank_order: str) -> List[tuple]:
        """Resolve sort field and direction.

        Profile and AgentSkill have no timestamp field, fallback to updated_at.
        """
        sort_field = rank_by
        if (
            rank_by == "timestamp"
            and memory_type in GetMemoryService._NO_TIMESTAMP_TYPES
        ):
            sort_field = "updated_at"
        sort_direction = DESCENDING if rank_order == "desc" else ASCENDING
        return [(sort_field, sort_direction)]

    async def find_memories(
        self,
        filters: Dict[str, Any],
        memory_type: str,
        page: int,
        page_size: int,
        rank_by: str,
        rank_order: str,
    ) -> GetMemResponse:
        """Find memories by filters DSL.

        Args:
            filters: Filter conditions dict from request body
            memory_type: Memory type (episodic_memory, profile, agent_case, agent_skill)
            page: Page number, starts from 1
            page_size: Items per page
            rank_by: Sort field
            rank_order: Sort order (asc or desc)

        Returns:
            GetMemResponse with matching memories

        Raises:
            InvalidScopeError: If filters lack user_id or group_id scope
        """
        # 1. Parse filters DSL into MongoDB query
        with timed("parse_filters"):
            mongo_filter, user_id, group_ids = parse_mongo_filters(filters)

            if not user_id and not group_ids:
                raise InvalidScopeError(
                    "filters must contain at least one of 'user_id' or 'group_id'"
                )

            # 2. Scope semantics by memory_type
            if not user_id and group_ids:
                if memory_type in self._USER_REQUIRED_TYPES:
                    raise InvalidScopeError(
                        f"memory_type '{memory_type}' requires 'user_id' in filters"
                    )
                if memory_type == MemoryType.EPISODIC_MEMORY.value:
                    # Group-only episodic query: exclude personal episodes
                    mongo_filter["user_id"] = {"$in": [None, ""]}

            # 3. Pagination and sort
            skip = (page - 1) * page_size
            limit = page_size
            sort = self._resolve_sort(memory_type, rank_by, rank_order)

        # 4. Dispatch by memory_type
        match memory_type:
            case MemoryType.EPISODIC_MEMORY.value:
                return await self._get_episodes(mongo_filter, skip, limit, sort)
            case MemoryType.PROFILE.value:
                return await self._get_profiles(mongo_filter, skip, limit, sort)
            case MemoryType.AGENT_CASE.value:
                return await self._get_agent_cases(mongo_filter, skip, limit, sort)
            case MemoryType.AGENT_SKILL.value:
                return await self._get_agent_skills(mongo_filter, skip, limit, sort)
            case _:
                raise ValueError(f"Unsupported memory_type: {memory_type}")

    async def _get_episodes(
        self, mongo_filter: dict, skip: int, limit: int, sort: list
    ) -> GetMemResponse:
        """Query v1_episodic_memories via repository and return GetMemResponse."""
        with timed("query_memories"):
            docs, total_count = await self._episodic_repo.find_by_query(
                mongo_filter,
                skip=skip,
                limit=limit,
                sort=sort,
                projection_model=EpisodicMemoryProjection,
            )

        with timed("assemble_results"):
            episodes = [self._episode_to_item(doc) for doc in docs]
            return GetMemResponse(
                episodes=episodes, total_count=total_count, count=len(episodes)
            )

    async def _get_profiles(
        self, mongo_filter: dict, skip: int, limit: int, sort: list
    ) -> GetMemResponse:
        """Query v1_user_profiles via repository and return GetMemResponse."""
        with timed("query_memories"):
            docs, total_count = await self._profile_repo.find_by_query(
                mongo_filter, skip=skip, limit=limit, sort=sort
            )

        with timed("assemble_results"):
            profiles = [self._profile_to_item(doc) for doc in docs]
            return GetMemResponse(
                profiles=profiles, total_count=total_count, count=len(profiles)
            )

    async def _get_agent_cases(
        self, mongo_filter: dict, skip: int, limit: int, sort: list
    ) -> GetMemResponse:
        """Query v1_agent_cases via repository and return GetMemResponse."""
        with timed("query_memories"):
            docs, total_count = await self._agent_case_repo.find_by_query(
                mongo_filter,
                skip=skip,
                limit=limit,
                sort=sort,
                projection_model=AgentCaseProjection,
            )

        with timed("assemble_results"):
            agent_cases = [self._agent_case_to_item(doc) for doc in docs]
            return GetMemResponse(
                agent_cases=agent_cases, total_count=total_count, count=len(agent_cases)
            )

    async def _get_agent_skills(
        self, mongo_filter: dict, skip: int, limit: int, sort: list
    ) -> GetMemResponse:
        """Query v1_agent_skills via repository and return GetMemResponse."""
        retire_confidence = DEFAULT_MEMORIZE_CONFIG.skill_retire_confidence
        mongo_filter.setdefault("confidence", {"$gte": retire_confidence})
        with timed("query_memories"):
            docs, total_count = await self._agent_skill_repo.find_by_query(
                mongo_filter,
                skip=skip,
                limit=limit,
                sort=sort,
                projection_model=AgentSkillProjection,
            )

        with timed("assemble_results"):
            agent_skills = [self._agent_skill_to_item(doc) for doc in docs]
            return GetMemResponse(
                agent_skills=agent_skills,
                total_count=total_count,
                count=len(agent_skills),
            )

    @staticmethod
    def _agent_case_to_item(doc) -> AgentCaseItem:
        """Convert AgentCaseRecord/Projection document to AgentCaseItem."""
        return AgentCaseItem(
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
        )

    @staticmethod
    def _agent_skill_to_item(doc) -> AgentSkillItem:
        """Convert AgentSkillRecord/Projection document to AgentSkillItem."""
        return AgentSkillItem(
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
        )

"""GetManager — top-level orchestrator for ``POST /api/v2/memory/get``.

Hard partition by ``(owner_type, memory_type)`` (validated by
:class:`GetRequest`):

* ``user`` + ``episode``       → ``data.episodes``
* ``user`` + ``profile``       → ``data.profiles`` (one-row KV fetch
  from the ``user_profile`` table; at most one item)
* ``agent`` + ``agent_case``   → ``data.agent_cases``
* ``agent`` + ``agent_skill``  → ``data.agent_skills``

Reads only — never writes. Filters are compiled through
:func:`compile_filters_for_get` so the column allow-list stays
shared with :mod:`memory.search`. Pagination + sort run in the index
behind the injected ports.
"""

from __future__ import annotations

from everos.component.utils.datetime import from_timestamp_ms, to_display_tz
from everos.core.context import resolve_request_id
from everos.core.observability.logging import get_logger
from everos.memory.models import AgentCase, AgentSkill, Episode
from everos.memory.ports import (
    AgentCaseIndex,
    AgentSkillIndex,
    EpisodeIndex,
    Page,
    ProfileIndex,
    Scope,
    Stored,
)

from .dto import (
    GetAgentCaseItem,
    GetAgentSkillItem,
    GetData,
    GetEpisodeItem,
    GetMemoryType,
    GetProfileItem,
    GetRequest,
    GetResponse,
)
from .filters_adapter import compile_filters_for_get

logger = get_logger(__name__)


class GetManager:
    """Dispatch ``GetRequest`` to the matching index and shape the stored
    domain objects into the public DTO.

    Args:
        episode_index: Episode reads.
        agent_case_index: Agent-case reads.
        agent_skill_index: Agent-skill reads.
        profile_index: User-profile reads.
    """

    def __init__(
        self,
        *,
        episode_index: EpisodeIndex,
        agent_case_index: AgentCaseIndex,
        agent_skill_index: AgentSkillIndex,
        profile_index: ProfileIndex,
    ) -> None:
        self._ep = episode_index
        self._case = agent_case_index
        self._skill = agent_skill_index
        self._profile = profile_index

    # ── Public entry ─────────────────────────────────────────────────

    async def get(self, req: GetRequest) -> GetResponse:
        request_id = resolve_request_id()
        descending = req.sort_order == "desc"
        where = compile_filters_for_get(
            req.filters,
            owner_id=req.owner_id,
            owner_type=req.owner_type,
            app_id=req.app_id,
            project_id=req.project_id,
        )
        page = Page(
            sort_by=req.sort_by,
            number=req.page,
            size=req.page_size,
            descending=descending,
        )

        match req.memory_type:
            case GetMemoryType.EPISODE:
                episodes = await self._ep.paginate(where, page)
                items = [self._shape_episode(e) for e in episodes.items]
                data = GetData(
                    episodes=items,
                    total_count=episodes.total,
                    count=len(items),
                )
            case GetMemoryType.PROFILE:
                profiles = await self._fetch_profile(req)
                data = GetData(
                    profiles=profiles,
                    total_count=len(profiles),
                    count=len(profiles),
                )
            case GetMemoryType.AGENT_CASE:
                cases = await self._case.paginate(where, page)
                items = [self._shape_agent_case(c) for c in cases.items]
                data = GetData(
                    agent_cases=items,
                    total_count=cases.total,
                    count=len(items),
                )
            case GetMemoryType.AGENT_SKILL:
                # AgentSkill has no ``timestamp`` column. Silently
                # downgrade ``sort_by`` to ``updated_at`` (the index's
                # row-maintenance column) so the caller cannot
                # accidentally trigger a schema error.
                skills = await self._skill.paginate(
                    where,
                    Page(
                        sort_by="updated_at",
                        number=req.page,
                        size=req.page_size,
                        descending=descending,
                    ),
                )
                items = [self._shape_agent_skill(s) for s in skills.items]
                data = GetData(
                    agent_skills=items,
                    total_count=skills.total,
                    count=len(items),
                )

        return GetResponse(request_id=request_id, data=data)

    # ── Shapers ──────────────────────────────────────────────────────

    @staticmethod
    def _shape_episode(stored: Stored[Episode]) -> GetEpisodeItem:
        episode = stored.item
        return GetEpisodeItem(
            id=stored.ref.id,
            user_id=episode.owner_id,
            app_id=stored.scope.app_id,
            project_id=stored.scope.project_id,
            session_id=episode.session_id,
            timestamp=to_display_tz(from_timestamp_ms(episode.timestamp)),
            sender_ids=episode.sender_ids,
            summary=episode.summary or "",
            subject=episode.subject or "",
            episode=episode.episode,
            type="Conversation",
        )

    @staticmethod
    def _shape_agent_case(stored: Stored[AgentCase]) -> GetAgentCaseItem:
        case = stored.item
        return GetAgentCaseItem(
            id=stored.ref.id,
            agent_id=case.owner_id,
            app_id=stored.scope.app_id,
            project_id=stored.scope.project_id,
            session_id=case.session_id,
            task_intent=case.task_intent,
            approach=case.approach,
            quality_score=case.quality_score,
            key_insight=case.key_insight,
            timestamp=to_display_tz(from_timestamp_ms(case.timestamp)),
        )

    @staticmethod
    def _shape_agent_skill(stored: Stored[AgentSkill]) -> GetAgentSkillItem:
        skill = stored.item
        return GetAgentSkillItem(
            id=stored.ref.id,
            agent_id=stored.scope.owner_id,
            app_id=stored.scope.app_id,
            project_id=stored.scope.project_id,
            name=skill.name,
            description=skill.description,
            content=skill.content,
            confidence=skill.confidence,
            maturity_score=skill.maturity_score,
            source_case_ids=skill.source_case_ids,
        )

    # ── Profile ──────────────────────────────────────────────────────

    async def _fetch_profile(self, req: GetRequest) -> list[GetProfileItem]:
        """Fetch the owner's single profile from the profile index (kept in
        sync with ``users/<id>/user.md`` by cascade).

        Profile is one-row-per-owner KV — there is no pagination / sort /
        filter surface, so at most one item is returned. Mirrors the
        ``/search`` ``ProfileRecaller`` minus the (unused) ``score`` field.
        Empty list (not 404) keeps the response valid during the cold-start
        window before a profile has been synthesised.
        """
        if not req.owner_id:
            return []
        stored = await self._profile.get(
            Scope(
                app_id=req.app_id,
                project_id=req.project_id,
                owner_id=req.owner_id,
                owner_type="user",
            )
        )
        if stored is None:
            logger.debug("get_profile_miss", owner_id=req.owner_id)
            return []
        profile = stored.item
        profile_data: dict[str, object] = {
            "summary": profile.summary,
            "explicit_info": profile.explicit_info,
            "implicit_traits": profile.implicit_traits,
            "profile_timestamp_ms": profile.timestamp,
        }
        return [
            GetProfileItem(
                id=stored.ref.id,
                user_id=profile.owner_id,
                app_id=stored.scope.app_id,
                project_id=stored.scope.project_id,
                profile_data=profile_data,
            )
        ]

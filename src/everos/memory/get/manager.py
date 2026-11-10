"""GetManager — top-level orchestrator for ``POST /api/v2/memory/get``.

Hard partition by ``(owner_type, memory_type)`` (validated by
:class:`GetRequest`):

* ``user`` + ``episode``       → ``data.episodes``
* ``user`` + ``profile``       → ``data.profiles`` (one-row KV fetch
  from the ``user_profile`` table; at most one item)
* ``agent`` + ``agent_case``   → ``data.agent_cases``
* ``agent`` + ``agent_skill``  → ``data.agent_skills``
* ``group`` + ``episode``      → ``data.episodes`` inside the asker's windows

Reads only — never writes. The owner part of every ``where`` is the access
policy's read scope (:mod:`everos.memory.access`); filters are compiled onto
it through :func:`compile_scoped_for_get` so the column allow-list stays
shared with :mod:`memory.search`. Pagination + sort run in the index
behind the injected ports.
"""

from __future__ import annotations

from everos.core.context import resolve_request_id
from everos.core.observability.logging import get_logger
from everos.core.utils.datetime import from_timestamp_ms, to_display_tz
from everos.memory.access import AccessPolicy, ScopeRequest, resolve_identity
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
from .filters_adapter import compile_scoped_for_get

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
        access: AccessPolicy,
    ) -> None:
        self._access = access
        self._ep = episode_index
        self._case = agent_case_index
        self._skill = agent_skill_index
        self._profile = profile_index

    # ── Public entry ─────────────────────────────────────────────────

    async def get(
        self, req: GetRequest, *, asserted_asker: str | None = None
    ) -> GetResponse:
        """One page of the requested kind, inside the asker's read scope.

        Raises:
            ForbiddenError: The asker may not read the scope (403).
            InvalidInputError: A group scope without an asker (422).
        """
        request_id = resolve_request_id()
        descending = req.sort_order == "desc"
        scope = await self._access.read_scope(
            resolve_identity(header=asserted_asker, body_asker=req.asker_id),
            ScopeRequest(
                app_id=req.app_id,
                project_id=req.project_id,
                user_id=req.user_id,
                agent_id=req.agent_id,
                group_ids=req.scope_group_ids,
                audience=req.audience,
                context_group_id=req.context_group_id,
            ),
        )
        where = compile_scoped_for_get(req.filters, scope.predicate)
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
                cases = await self._case.paginate(scope.cases_where(where), page)
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
        group = stored.scope.owner_type == "group"
        return GetEpisodeItem(
            id=stored.ref.id,
            user_id=None if group else episode.owner_id,
            group_id=episode.owner_id if group else None,
            app_id=stored.scope.app_id,
            project_id=stored.scope.project_id,
            session_id=episode.session_id,
            timestamp=to_display_tz(from_timestamp_ms(episode.timestamp)),
            sender_ids=episode.sender_ids,
            subject_ids=list(episode.subject_ids),
            origin=episode.origin,
            summary=episode.summary,
            subject=episode.subject,
            episode=episode.episode,
            type="Conversation",
        )

    @staticmethod
    def _shape_agent_case(stored: Stored[AgentCase]) -> GetAgentCaseItem:
        case = stored.item
        return GetAgentCaseItem(
            id=stored.ref.id,
            entry_id=stored.ref.entry_id,
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
            "explicit_info": [i.model_dump() for i in profile.explicit_info],
            "implicit_traits": [i.model_dump() for i in profile.implicit_traits],
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

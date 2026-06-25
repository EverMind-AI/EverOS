"""GetManager — top-level orchestrator for ``POST /api/v1/memory/get``.

Hard partition by ``(owner_type, memory_type)`` (validated by
:class:`GetRequest`):

* ``user`` + ``episode``       → ``data.episodes``
* ``user`` + ``profile``       → ``data.profiles`` (one-row KV fetch
  from the ``user_profile`` table; at most one item)
* ``agent`` + ``agent_case``   → ``data.agent_cases``
* ``agent`` + ``agent_skill``  → ``data.agent_skills``

Reads only — never writes. Filters are compiled through
:func:`compile_filters_for_get` so the column allow-list stays
shared with :mod:`memory.search`. Pagination + in-memory sort
runs through :meth:`LanceRepoBase.find_where_paginated`.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from everos.component.utils.datetime import to_display_tz
from everos.core.observability.logging import get_logger
from everos.core.observability.tracing import gen_request_id

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

if TYPE_CHECKING:
    from everos.core.persistence.lancedb import LanceRepoBase
    from everos.infra.persistence.lancedb import (
        AgentCase,
        AgentSkill,
        Episode,
        UserProfile,
    )

logger = get_logger(__name__)


class GetManager:
    """Dispatch ``GetRequest`` to the matching LanceDB-backed repo and
    shape rows into the public DTO."""

    def __init__(
        self,
        *,
        episode_repo: LanceRepoBase[Episode],
        agent_case_repo: LanceRepoBase[AgentCase],
        agent_skill_repo: LanceRepoBase[AgentSkill],
        user_profile_repo: LanceRepoBase[UserProfile],
    ) -> None:
        self._ep = episode_repo
        self._case = agent_case_repo
        self._skill = agent_skill_repo
        self._profile = user_profile_repo

    # ── Public entry ─────────────────────────────────────────────────

    async def get(self, req: GetRequest) -> GetResponse:
        request_id = gen_request_id()
        descending = req.sort_order == "desc"
        where = compile_filters_for_get(
            req.filters,
            owner_id=req.owner_id,
            owner_type=req.owner_type,
            app_id=req.app_id,
            project_id=req.project_id,
            exclude_deprecated=req.memory_type == GetMemoryType.EPISODE,
        )

        match req.memory_type:
            case GetMemoryType.EPISODE:
                rows, total = await self._ep.find_where_paginated(
                    where,
                    sort_by=req.sort_by,
                    descending=descending,
                    page=req.page,
                    page_size=req.page_size,
                )
                items = [self._shape_episode(r) for r in rows]
                data = GetData(
                    episodes=items,
                    total_count=total,
                    count=len(items),
                )
            case GetMemoryType.PROFILE:
                profiles = await self._fetch_profile(req.owner_id)
                data = GetData(
                    profiles=profiles,
                    total_count=len(profiles),
                    count=len(profiles),
                )
            case GetMemoryType.AGENT_CASE:
                rows, total = await self._case.find_where_paginated(
                    where,
                    sort_by=req.sort_by,
                    descending=descending,
                    page=req.page,
                    page_size=req.page_size,
                )
                items = [self._shape_agent_case(r) for r in rows]
                data = GetData(
                    agent_cases=items,
                    total_count=total,
                    count=len(items),
                )
            case GetMemoryType.AGENT_SKILL:
                # AgentSkill has no ``timestamp`` column. Silently
                # downgrade ``sort_by`` to ``updated_at`` (from
                # :class:`BaseLanceTable`) so the caller cannot
                # accidentally trigger a schema error.
                rows, total = await self._skill.find_where_paginated(
                    where,
                    sort_by="updated_at",
                    descending=descending,
                    page=req.page,
                    page_size=req.page_size,
                )
                items = [self._shape_agent_skill(r) for r in rows]
                data = GetData(
                    agent_skills=items,
                    total_count=total,
                    count=len(items),
                )

        return GetResponse(request_id=request_id, data=data)

    # ── Shapers ──────────────────────────────────────────────────────

    @staticmethod
    def _shape_episode(row: Episode) -> GetEpisodeItem:
        return GetEpisodeItem(
            id=row.id,
            user_id=row.owner_id,
            app_id=row.app_id,
            project_id=row.project_id,
            session_id=row.session_id,
            timestamp=to_display_tz(row.timestamp),
            sender_ids=row.sender_ids,
            summary=row.summary or "",
            subject=row.subject or "",
            episode=row.episode,
            type="Conversation",
        )

    @staticmethod
    def _shape_agent_case(row: AgentCase) -> GetAgentCaseItem:
        return GetAgentCaseItem(
            id=row.id,
            agent_id=row.owner_id,
            app_id=row.app_id,
            project_id=row.project_id,
            session_id=row.session_id,
            task_intent=row.task_intent,
            approach=row.approach,
            quality_score=row.quality_score,
            key_insight=row.key_insight,
            timestamp=to_display_tz(row.timestamp),
        )

    @staticmethod
    def _shape_agent_skill(row: AgentSkill) -> GetAgentSkillItem:
        return GetAgentSkillItem(
            id=row.id,
            agent_id=row.owner_id,
            app_id=row.app_id,
            project_id=row.project_id,
            name=row.name,
            description=row.description,
            content=row.content,
            confidence=row.confidence,
            maturity_score=row.maturity_score,
            source_case_ids=row.source_case_ids,
        )

    # ── Profile ──────────────────────────────────────────────────────

    async def _fetch_profile(self, owner_id: str) -> list[GetProfileItem]:
        """Fetch the owner's single profile row from the ``user_profile``
        LanceDB table (kept in sync with ``users/<id>/user.md`` by cascade).

        Profile is one-row-per-owner KV — there is no pagination / sort /
        filter surface, so at most one item is returned. Mirrors the
        ``/search`` ``ProfileRecaller`` minus the (unused) ``score`` field.
        Empty list (not 404) keeps the response valid during the cold-start
        window before a profile has been synthesised.
        """
        if not owner_id:
            return []
        row = await self._profile.get_by_id(owner_id)
        if row is None:
            logger.debug("get_profile_miss", owner_id=owner_id)
            return []
        profile_data: dict[str, object] = {
            "summary": row.summary,
            "explicit_info": _load_json(row.explicit_info_json),
            "implicit_traits": _load_json(row.implicit_traits_json),
            "profile_timestamp_ms": row.profile_timestamp_ms,
        }
        return [
            GetProfileItem(
                id=row.id,
                user_id=row.owner_id,
                app_id=row.app_id,
                project_id=row.project_id,
                profile_data=profile_data,
            )
        ]


def _load_json(text: str) -> Any:
    """Decode a json-encoded profile frontmatter bucket.

    Returns ``[]`` on empty / malformed input so a row with a stale
    encoding doesn't break the get response (mirrors the search recaller).
    """
    if not text:
        return []
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.debug("get_profile_json_decode_failed", payload_head=text[:80])
        return []

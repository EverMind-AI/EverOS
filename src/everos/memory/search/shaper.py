"""Ranked candidates -> public DTO items: the outbound mapper of the search path.

The rank algorithms speak everalgo ``Candidate`` / ``ScoredItem``; the
response speaks :mod:`everos.memory.search.dto`. This module is the only
place the two meet. A ranked item contributes its id and score; every
other DTO field comes from the stored domain record the recaller returned
next to it (``stored_by_id``, keyed by ``Candidate.id``), so no column
name of any index is read here.

* :func:`shape_episodes`, :func:`shape_agent_cases`,
  :func:`shape_agent_skills` (and their ``*_ranked`` twins for an everalgo
  ``RankOutput``) emit one DTO per candidate.

A candidate whose record is missing or belongs to the wrong owner type is
dropped with a warning rather than failing the request.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from everalgo.types import Candidate, RankOutput, ScoredItem

from everos.core.observability.logging import get_logger
from everos.core.utils.datetime import from_timestamp_ms, to_display_tz
from everos.memory.models import AgentCase, AgentSkill, Episode
from everos.memory.ports import Stored

from .dto import (
    SearchAgentCaseItem,
    SearchAgentSkillItem,
    SearchEpisodeItem,
)

logger = get_logger(__name__)

type _Ranked = Candidate | ScoredItem


# ── Episodes ────────────────────────────────────────────────────────────


def shape_episodes(
    ranked: Iterable[_Ranked], stored_by_id: Mapping[str, Stored[Episode]]
) -> list[SearchEpisodeItem]:
    """One episode DTO per ranked candidate, in order; unknown ids dropped."""
    return [
        item
        for item in (_episode(r.id, r.score, stored_by_id) for r in ranked)
        if item is not None
    ]


def _episode(
    candidate_id: str,
    score: float,
    stored_by_id: Mapping[str, Stored[Episode]],
) -> SearchEpisodeItem | None:
    stored = _lookup(candidate_id, stored_by_id, kind="episode", owner_type="user")
    if stored is None:
        return None
    episode = stored.item
    return SearchEpisodeItem(
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
        score=float(score),
    )


def shape_episodes_ranked(
    output: RankOutput, stored_by_id: Mapping[str, Stored[Episode]]
) -> list[SearchEpisodeItem]:
    """:func:`shape_episodes` over an everalgo ``arank`` result."""
    return shape_episodes(output.items, stored_by_id)


# ── Agent cases / skills ─────────────────────────────────────────────────


def shape_agent_cases(
    ranked: Iterable[_Ranked], stored_by_id: Mapping[str, Stored[AgentCase]]
) -> list[SearchAgentCaseItem]:
    """One case DTO per ranked candidate, in order; unknown ids dropped."""
    items: list[SearchAgentCaseItem] = []
    for r in ranked:
        stored = _lookup(r.id, stored_by_id, kind="case", owner_type="agent")
        if stored is None:
            continue
        case = stored.item
        items.append(
            SearchAgentCaseItem(
                id=stored.ref.id,
                entry_id=stored.ref.entry_id,
                agent_id=case.owner_id,
                app_id=stored.scope.app_id,
                project_id=stored.scope.project_id,
                session_id=case.session_id,
                task_intent=case.task_intent,
                approach=case.approach,
                quality_score=float(case.quality_score),
                key_insight=case.key_insight,
                timestamp=to_display_tz(from_timestamp_ms(case.timestamp)),
                score=float(r.score),
            )
        )
    return items


def shape_agent_skills(
    ranked: Iterable[_Ranked], stored_by_id: Mapping[str, Stored[AgentSkill]]
) -> list[SearchAgentSkillItem]:
    """One skill DTO per ranked candidate, in order; unknown ids dropped."""
    items: list[SearchAgentSkillItem] = []
    for r in ranked:
        stored = _lookup(r.id, stored_by_id, kind="skill", owner_type="agent")
        if stored is None:
            continue
        skill = stored.item
        items.append(
            SearchAgentSkillItem(
                id=stored.ref.id,
                agent_id=stored.scope.owner_id,
                app_id=stored.scope.app_id,
                project_id=stored.scope.project_id,
                name=skill.name,
                description=skill.description,
                content=skill.content,
                confidence=float(skill.confidence),
                maturity_score=float(skill.maturity_score),
                source_case_ids=skill.source_case_ids,
                score=float(r.score),
            )
        )
    return items


def shape_agent_cases_ranked(
    output: RankOutput, stored_by_id: Mapping[str, Stored[AgentCase]]
) -> list[SearchAgentCaseItem]:
    """:func:`shape_agent_cases` over an everalgo ``arank`` result."""
    return shape_agent_cases(output.items, stored_by_id)


def shape_agent_skills_ranked(
    output: RankOutput, stored_by_id: Mapping[str, Stored[AgentSkill]]
) -> list[SearchAgentSkillItem]:
    """:func:`shape_agent_skills` over an everalgo ``arank`` result."""
    return shape_agent_skills(output.items, stored_by_id)


# ── Lookup ──────────────────────────────────────────────────────────────


def _lookup[T](
    candidate_id: str,
    stored_by_id: Mapping[str, Stored[T]],
    *,
    kind: str,
    owner_type: Any,
) -> Stored[T] | None:
    stored = stored_by_id.get(candidate_id)
    if stored is None:
        logger.warning("shape_missing_record", kind=kind, id=candidate_id)
        return None
    if stored.scope.owner_type != owner_type:
        logger.warning(
            "shape_unexpected_owner_type",
            kind=kind,
            id=candidate_id,
            owner_type=stored.scope.owner_type,
        )
        return None
    return stored

"""Ranked candidates -> public DTO items: the outbound mapper of the search path.

The rank algorithms speak everalgo ``Candidate`` / ``ScoredItem``; the
response speaks :mod:`everos.memory.search.dto`. This module is the only
place the two meet. A ranked item contributes its id and score; every
other DTO field comes from the stored domain record the recaller returned
next to it (``stored_by_id``, keyed by ``Candidate.id``), so no column
name of any index is read here.

* Simple kinds -- :func:`shape_episodes`, :func:`shape_agent_cases`,
  :func:`shape_agent_skills` (and their ``*_ranked`` twins for an everalgo
  ``RankOutput``) emit one DTO per candidate.
* Hybrid output -- :func:`reshape_hybrid_output` collapses the mixed
  ``episode`` + ``atomic_fact`` ``ScoredItem`` list into episodes with
  their facts nested.

A candidate whose record is missing or belongs to the wrong owner type is
dropped with a warning rather than failing the request.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from everalgo.types import Candidate, RankOutput, ScoredItem

from everos.core.observability.logging import get_logger
from everos.core.utils.datetime import from_timestamp_ms, to_display_tz
from everos.memory.models import AgentCase, AgentSkill, AtomicFact, Episode
from everos.memory.ports import Stored

from .dto import (
    SearchAgentCaseItem,
    SearchAgentSkillItem,
    SearchAtomicFactItem,
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


def reshape_hybrid_output(
    scored: Iterable[ScoredItem],
    *,
    episodes_by_id: Mapping[str, Stored[Episode]],
    facts_by_id: Mapping[str, Stored[AtomicFact]],
) -> list[SearchEpisodeItem]:
    """Collapse the mixed episode + atomic_fact output into nested episodes.

    Fact eviction can swap a parent episode for its top atomic fact. Facts
    are re-attached to their parent so the response only surfaces episodes,
    with their facts nested. A fact whose parent was evicted brings the
    parent back from ``episodes_by_id`` (the whole recall pool), scored at
    the fact's score so the eviction relevance signal survives; a fact whose
    parent is not in the pool at all is dropped with a warning.
    """
    scored_items = list(scored)
    episode_scores = {s.id: s.score for s in scored_items if s.item_type == "episode"}
    facts_by_parent: dict[str, list[ScoredItem]] = defaultdict(list)
    for s in scored_items:
        if s.item_type == "atomic_fact" and s.parent_episode_id:
            facts_by_parent[s.parent_episode_id].append(s)

    out: list[SearchEpisodeItem] = []
    for ep_id, score in episode_scores.items():
        facts = _fact_items(facts_by_parent.get(ep_id, []), facts_by_id)
        item = _episode(ep_id, score, episodes_by_id, atomic_facts=facts)
        if item is not None:
            out.append(item)

    for parent_id, fact_scoreds in facts_by_parent.items():
        if parent_id in episode_scores:
            continue
        if parent_id not in episodes_by_id:
            logger.warning(
                "orphan_fact_parent_missing",
                parent_id=parent_id,
                fact_count=len(fact_scoreds),
            )
            continue
        top_score = max((f.score for f in fact_scoreds), default=0.0)
        facts = _fact_items(fact_scoreds, facts_by_id)
        item = _episode(parent_id, top_score, episodes_by_id, atomic_facts=facts)
        if item is not None:
            out.append(item)

    out.sort(key=lambda e: e.score, reverse=True)
    return out


def _episode(
    candidate_id: str,
    score: float,
    stored_by_id: Mapping[str, Stored[Episode]],
    *,
    atomic_facts: list[SearchAtomicFactItem] | None = None,
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
        atomic_facts=atomic_facts or [],
    )


def _fact_items(
    scoreds: list[ScoredItem], facts_by_id: Mapping[str, Stored[AtomicFact]]
) -> list[SearchAtomicFactItem]:
    items: list[SearchAtomicFactItem] = []
    for s in sorted(scoreds, key=lambda s: s.score, reverse=True):
        stored = facts_by_id.get(s.id)
        if stored is None:
            logger.warning("shape_fact_missing_record", id=s.id)
            continue
        items.append(
            SearchAtomicFactItem(
                id=stored.ref.id, content=stored.item.content, score=float(s.score)
            )
        )
    return items


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

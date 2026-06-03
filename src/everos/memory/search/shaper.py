"""Translate everalgo ``Candidate`` / ``ScoredItem`` → public DTO items.

This module handles two distinct flows:

* Simple kinds — :func:`shape_episode_from_candidate`,
  :func:`shape_agent_case_from_candidate`,
  :func:`shape_agent_skill_from_candidate`. Each consumes one
  :class:`Candidate` and emits the matching ``SearchXxxItem``.
* Hybrid output — :func:`reshape_hybrid_output` collapses the mixed
  ``episode`` + ``atomic_fact`` ScoredItem list into
  :class:`SearchEpisodeItem` instances with nested ``atomic_facts``.

The shaper is pure — no LanceDB calls. When fact eviction replaces a
parent episode with its facts, the orphan fact's parent is looked up
in an ``episode_pool`` dict that the manager assembles from the
pre-fusion ``sparse + dense`` candidates.
"""

from __future__ import annotations

import datetime as _dt
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from everalgo.types import Candidate, ScoredItem

from everos.component.utils.datetime import to_display_tz
from everos.core.observability.logging import get_logger

from .dto import (
    SearchAgentCaseItem,
    SearchAgentSkillItem,
    SearchAtomicFactItem,
    SearchEpisodeItem,
)

logger = get_logger(__name__)


# ── Episode shaping ─────────────────────────────────────────────────────


def shape_episode_from_candidate(
    candidate: Candidate,
    *,
    atomic_facts: list[SearchAtomicFactItem] | None = None,
) -> SearchEpisodeItem | None:
    """Build a :class:`SearchEpisodeItem` from a recall ``Candidate``.

    Returns ``None`` if the row is malformed (e.g. owner_type is not
    ``"user"``, mandatory fields missing). Callers drop ``None`` results.
    """
    md = candidate.metadata
    owner_type = md.get("owner_type")
    if owner_type != "user":
        logger.warning(
            "shape_episode_unexpected_owner_type",
            id=candidate.id,
            owner_type=owner_type,
        )
        return None
    timestamp = _coerce_datetime(md.get("timestamp"))
    if timestamp is None:
        logger.warning("shape_episode_missing_timestamp", id=candidate.id)
        return None
    session_id = md.get("session_id")
    episode = md.get("episode")
    if not isinstance(session_id, str) or not isinstance(episode, str):
        logger.warning("shape_episode_missing_required_field", id=candidate.id)
        return None
    return SearchEpisodeItem(
        id=candidate.id,
        user_id=_as_optional_str(md.get("owner_id")),
        app_id=_as_str(md.get("app_id")) or "default",
        project_id=_as_str(md.get("project_id")) or "default",
        session_id=session_id,
        timestamp=timestamp,
        sender_ids=_as_str_list(md.get("sender_ids")),
        summary=_as_str(md.get("summary")),
        subject=_as_str(md.get("subject")),
        episode=episode,
        type="Conversation",
        score=float(candidate.score),
        atomic_facts=atomic_facts or [],
    )


# ── Atomic fact shaping ─────────────────────────────────────────────────


def shape_atomic_fact_from_candidate(candidate: Candidate) -> SearchAtomicFactItem:
    """Cast a LanceDB ``atomic_fact`` candidate row into the nested fact DTO."""
    content = _as_str(candidate.metadata.get("fact"))
    return SearchAtomicFactItem(
        id=candidate.id,
        content=content,
        score=float(candidate.score),
    )


def shape_atomic_fact_from_scored(scored: ScoredItem) -> SearchAtomicFactItem:
    """Cast an everalgo ``ScoredItem(item_type='atomic_fact')`` into a fact DTO."""
    content = _as_str(scored.metadata.get("fact"))
    return SearchAtomicFactItem(
        id=scored.id,
        content=content,
        score=float(scored.score),
    )


# ── AgentCase shaping ───────────────────────────────────────────────────


def shape_agent_case_from_candidate(candidate: Candidate) -> SearchAgentCaseItem | None:
    md = candidate.metadata
    if md.get("owner_type") != "agent":
        logger.warning(
            "shape_case_unexpected_owner_type",
            id=candidate.id,
            owner_type=md.get("owner_type"),
        )
        return None
    timestamp = _coerce_datetime(md.get("timestamp"))
    if timestamp is None:
        logger.warning("shape_case_missing_timestamp", id=candidate.id)
        return None
    owner_id = md.get("owner_id")
    session_id = md.get("session_id")
    task_intent = md.get("task_intent")
    approach = md.get("approach")
    quality = md.get("quality_score")
    if not (
        isinstance(owner_id, str)
        and isinstance(session_id, str)
        and isinstance(task_intent, str)
        and isinstance(approach, str)
        and isinstance(quality, (int, float))
    ):
        logger.warning("shape_case_missing_required_field", id=candidate.id)
        return None
    return SearchAgentCaseItem(
        id=candidate.id,
        agent_id=owner_id,
        app_id=_as_str(md.get("app_id")) or "default",
        project_id=_as_str(md.get("project_id")) or "default",
        session_id=session_id,
        task_intent=task_intent,
        approach=approach,
        quality_score=float(quality),
        key_insight=_as_optional_str(md.get("key_insight")),
        timestamp=timestamp,
        score=float(candidate.score),
    )


# ── AgentSkill shaping ──────────────────────────────────────────────────


def shape_agent_skill_from_candidate(
    candidate: Candidate,
) -> SearchAgentSkillItem | None:
    md = candidate.metadata
    if md.get("owner_type") != "agent":
        logger.warning(
            "shape_skill_unexpected_owner_type",
            id=candidate.id,
            owner_type=md.get("owner_type"),
        )
        return None
    owner_id = md.get("owner_id")
    name = md.get("name")
    description = md.get("description")
    content = md.get("content")
    confidence = md.get("confidence")
    maturity = md.get("maturity_score")
    if not (
        isinstance(owner_id, str)
        and isinstance(name, str)
        and isinstance(description, str)
        and isinstance(content, str)
        and isinstance(confidence, (int, float))
        and isinstance(maturity, (int, float))
    ):
        logger.warning("shape_skill_missing_required_field", id=candidate.id)
        return None
    return SearchAgentSkillItem(
        id=candidate.id,
        agent_id=owner_id,
        app_id=_as_str(md.get("app_id")) or "default",
        project_id=_as_str(md.get("project_id")) or "default",
        name=name,
        description=description,
        content=content,
        confidence=float(confidence),
        maturity_score=float(maturity),
        source_case_ids=_as_str_list(md.get("source_case_ids")),
        score=float(candidate.score),
    )


# ── Hybrid mixed output reshape ─────────────────────────────────────────


def reshape_hybrid_output(
    scored: Iterable[ScoredItem],
    *,
    episode_pool: dict[str, Candidate],
) -> list[SearchEpisodeItem]:
    """Collapse the mixed episode + atomic_fact output into nested SearchEpisodeItems.

    Fact eviction can swap a parent episode for its top atomic fact.
    We re-attach facts to their parent so the API response only ever
    surfaces episodes (with their facts nested).

    ``episode_pool`` is the union of pre-fusion sparse + dense episode
    candidates, keyed by id. When a fact's parent is missing from the
    final scored list, we fall back to this pool. Facts whose parent
    is in neither place are dropped with a warning (very rare —
    requires a parent absent from the recall pool altogether).
    """
    scored_items = list(scored)
    episodes: dict[str, ScoredItem] = {
        s.id: s for s in scored_items if s.item_type == "episode"
    }
    facts_by_parent: dict[str, list[ScoredItem]] = defaultdict(list)
    for s in scored_items:
        if s.item_type == "atomic_fact" and s.parent_episode_id:
            facts_by_parent[s.parent_episode_id].append(s)

    out: list[SearchEpisodeItem] = []
    seen: set[str] = set()

    # 1. Episodes still in top-N — attach any facts grouped under them.
    for ep_id, ep_scored in episodes.items():
        seen.add(ep_id)
        facts = _build_fact_items(facts_by_parent.get(ep_id, []))
        item = _shape_episode_from_scored(ep_scored, atomic_facts=facts)
        if item is not None:
            out.append(item)

    # 2. Orphan facts — parent evicted but available in the pool.
    for parent_id, fact_scoreds in facts_by_parent.items():
        if parent_id in seen:
            continue
        parent = episode_pool.get(parent_id)
        if parent is None:
            logger.warning(
                "orphan_fact_parent_missing",
                parent_id=parent_id,
                fact_count=len(fact_scoreds),
            )
            continue
        seen.add(parent_id)
        facts = _build_fact_items(fact_scoreds)
        # Score the synthetic episode entry at the top fact's score so the
        # response retains the eviction relevance signal.
        top_score = max((f.score for f in fact_scoreds), default=0.0)
        parent_with_score = parent.model_copy(update={"score": top_score})
        item = shape_episode_from_candidate(parent_with_score, atomic_facts=facts)
        if item is not None:
            out.append(item)

    out.sort(key=lambda e: e.score, reverse=True)
    return out


def _build_fact_items(scoreds: list[ScoredItem]) -> list[SearchAtomicFactItem]:
    return [
        shape_atomic_fact_from_scored(s)
        for s in sorted(scoreds, key=lambda s: s.score, reverse=True)
    ]


def _shape_episode_from_scored(
    scored: ScoredItem,
    *,
    atomic_facts: list[SearchAtomicFactItem],
) -> SearchEpisodeItem | None:
    """Adapt a ScoredItem(item_type='episode') with the same checks as Candidate."""
    pseudo = Candidate(
        id=scored.id,
        score=scored.score,
        source="other",
        metadata=dict(scored.metadata),
    )
    return shape_episode_from_candidate(pseudo, atomic_facts=atomic_facts)


# ── Coercion helpers ────────────────────────────────────────────────────


def _coerce_datetime(value: Any) -> _dt.datetime | None:
    """Coerce a storage-side datetime to the display timezone, or ``None``.

    LanceDB's Arrow schema declares timestamp columns with ``tz=UTC``
    (see :attr:`BaseLanceTable.UTC_DATETIME_FIELDS`), so PyArrow returns
    aware UTC datetimes. ``to_display_tz`` is a pure ``astimezone(...)``
    in that case. Aware non-UTC and naive inputs (test fixtures) flow
    through ``to_display_tz`` safely as well — naive is treated as
    already display-tz local.

    Non-datetime input returns ``None`` so callers can treat it as
    "missing field" without raising.
    """
    if not isinstance(value, _dt.datetime):
        return None
    return to_display_tz(value)


def _as_str(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _as_optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _as_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [v for v in value if isinstance(v, str)]

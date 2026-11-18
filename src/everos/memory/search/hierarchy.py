"""Episode HYBRID hierarchy — the ranking layers over three recall lanes.

The manager recalls the lanes: BM25 over episodes, dense recall over episode
vectors, and the atomic-fact lane (episodes scored by their best-matching
fact, see ``recall/maxsim.py``). This module ranks them:

    L1  ``rrf(sparse, dense)``
    L3  ``rrf(L1, fact lane)``, cut to ``top_k`` episodes
    L4  each kept episode scored ``cosine_to_lr_score(cosine, bm25)``, the
        cosine being the better of the episode's own vector and its best
        atomic fact; when the fact is better, it is attached to the episode

``top_k`` counts episodes, and an episode carries at most the one fact its
score came from. The score is LR-calibrated to ``[0, 1]``, which is what
``min_score`` filters on.

Pure synchronous module — zero I/O, zero async.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from everalgo.rank.fusion import cosine_to_lr_score, rrf
from everalgo.types import Candidate, ScoredItem

from everos.memory.search._candidates import FactCandidate


def build_ep_to_fact_parents(
    episodes: Sequence[Candidate],
) -> dict[str, list[str]]:
    """Map episode candidate id to possible fact parent_id values.

    Args:
        episodes: Episode candidate list.

    Returns:
        Dict mapping episode id to parent_ids (entry_id and/or memcell_id).
    """
    result: dict[str, list[str]] = {}
    for ep in episodes:
        parents: list[str] = []
        entry_id = ep.metadata.get("entry_id")
        if isinstance(entry_id, str) and entry_id:
            parents.append(entry_id)
        parent_id = ep.metadata.get("parent_id")
        if isinstance(parent_id, str) and parent_id and parent_id != entry_id:
            parents.append(parent_id)
        if parents:
            result[ep.id] = parents
    return result


def fuse_lanes(
    sparse: Sequence[Candidate],
    dense: Sequence[Candidate],
    fact_lane: Sequence[Candidate],
    *,
    top_k: int,
) -> list[Candidate]:
    """L1 + L3: the episodes kept, in fused order.

    Args:
        sparse: BM25 episode candidates.
        dense: Episode-vector candidates.
        fact_lane: Episodes scored by their best atomic fact.
        top_k: Episodes to keep.

    Returns:
        At most ``top_k`` episode candidates.
    """
    return rrf(rrf(list(sparse), list(dense)), list(fact_lane))[:top_k]


def score_episodes(
    kept: Sequence[Candidate],
    episode_to_facts: Mapping[str, Sequence[FactCandidate]],
    *,
    sparse: Sequence[Candidate],
    dense: Sequence[Candidate],
    fact_lane: Sequence[Candidate],
) -> list[ScoredItem]:
    """L4: score each kept episode and attach the fact it ranked on.

    Args:
        kept: The episodes :func:`fuse_lanes` kept.
        episode_to_facts: Each episode's facts, closest to the query first.
        sparse: BM25 episode candidates, for the BM25 term.
        dense: Episode-vector candidates, for the episode's own cosine.
        fact_lane: Atomic-fact lane candidates, for the best fact cosine.

    Returns:
        One episode item per kept episode, each followed by its best fact
        when that fact out-scored the episode's own vector. The fact carries
        the episode's score.
    """
    own_cosine = _max_score_by_id(dense)
    fact_cosine = _max_score_by_id(fact_lane)
    for episode_id, facts in episode_to_facts.items():
        if facts:
            fact_cosine[episode_id] = max(
                fact_cosine.get(episode_id, 0.0), facts[0].score
            )
    bm25 = {c.id: c.score for c in sparse if c.id}
    scored: list[ScoredItem] = []
    for episode in kept:
        own = own_cosine.get(episode.id, 0.0)
        via_fact = fact_cosine.get(episode.id, 0.0)
        score = cosine_to_lr_score(max(own, via_fact), bm25.get(episode.id, 0.0))
        scored.append(ScoredItem(id=episode.id, score=score, item_type="episode"))
        facts = episode_to_facts.get(episode.id)
        if facts and via_fact > own:
            scored.append(
                ScoredItem(
                    id=facts[0].id,
                    score=score,
                    item_type="atomic_fact",
                    parent_episode_id=episode.id,
                )
            )
    return scored


def _max_score_by_id(candidates: Sequence[Candidate]) -> dict[str, float]:
    """Best score per id: a lane can surface the same episode more than once."""
    best: dict[str, float] = {}
    for candidate in candidates:
        if candidate.id:
            best[candidate.id] = max(best.get(candidate.id, 0.0), candidate.score)
    return best

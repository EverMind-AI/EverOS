"""MaxSim episode recall: atomic-fact ANN, max-pooled onto parent episodes.

Atomic facts are ~28x denser than episodes, so a long episode whose single
mean-pooled vector dilutes a specific topic is recovered through the
matching fact's own embedding. When no fact in scope carries a parent --
extraction disabled, not caught up, or failing -- the recall falls back to
ANN over the episode's own vector instead of returning nothing.
"""

from __future__ import annotations

from collections.abc import Sequence

from everos.core.observability.logging import get_logger
from everos.memory.models import Episode
from everos.memory.ports import Filter

from .atomic_fact import AtomicFactRecaller
from .base import RecallResult
from .episode import EpisodeRecaller

logger = get_logger(__name__)


async def maxsim_episode_recall(
    vector: Sequence[float],
    where: Filter,
    *,
    facts: AtomicFactRecaller,
    episodes: EpisodeRecaller,
    top_k: int,
    fact_limit: int,
    fallback_limit: int,
    owner_id: str,
) -> RecallResult[Episode]:
    """Top ``top_k`` episodes by their best-matching atomic fact.

    Args:
        vector: Query embedding; empty returns nothing.
        where: Filter scoping both recalls.
        facts: Atomic-fact recall.
        episodes: Episode recall (parent fetch and fallback).
        top_k: Episodes to keep after max-pooling.
        fact_limit: Fact pool size for the ANN.
        fallback_limit: Episode pool size when the fact pool has no parent.
        owner_id: Owner searched, for the fallback log line.

    Returns:
        Episodes keyed by global id, sorted by score descending.
    """
    if not vector:
        return RecallResult.empty()
    fact_pool = await facts.dense_recall(vector, where, limit=fact_limit)
    best: dict[str, float] = {}
    for fact in fact_pool.candidates:
        entry_id = fact_pool.stored_by_id[fact.id].item.parent_id
        if entry_id and fact.score > best.get(entry_id, -1.0):
            best[entry_id] = fact.score
    if not best:
        fallback = await episodes.dense_recall(vector, where, limit=fallback_limit)
        logger.info(
            "vector_recall_fact_pool_empty",
            owner_id=owner_id,
            hits=len(fallback.candidates),
        )
        return fallback
    ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    score_by_entry = dict(ranked)
    parents = await episodes.fetch_by_entry_ids([eid for eid, _ in ranked], where)
    rescored = [
        c.model_copy(
            update={
                "score": score_by_entry.get(
                    parents.stored_by_id[c.id].ref.entry_id, 0.0
                )
            }
        )
        for c in parents.candidates
    ]
    rescored.sort(key=lambda c: c.score, reverse=True)
    return RecallResult(candidates=rescored, stored_by_id=parents.stored_by_id)

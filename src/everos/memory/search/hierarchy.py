"""Hierarchical episode retrieval — two-path recall fused with per-fact eviction.

Episode HYBRID search path: combines episode-level hybrid recall (Layer 1)
with fact-driven MaxSim re-scoring (Layer 2), merges via RRF (Layer 3), then
runs a single-pass eviction where a fact that outscores its parent episode
enters top-N in place of the episode (Layer 4).

Uses everalgo operators as pure algorithm primitives; all I/O is injected
via recaller callbacks.  No changes to the everalgo library are required.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from everalgo.rank import amaxsim_retrieve
from everalgo.rank.fusion import rrf
from everalgo.types import Candidate, FactCandidate, ScoredItem

from everos.core.observability.logging import get_logger

from .dto import SearchEpisodeItem
from .shaper import reshape_hybrid_output

if TYPE_CHECKING:
    from collections.abc import Sequence

    from everos.memory.search.recall.atomic_fact import AtomicFactRecaller
    from everos.memory.search.recall.episode import EpisodeRecaller

logger = get_logger(__name__)


async def hierarchy_retrieve_episodes(
    query: str,
    *,
    sparse: list[Candidate],
    dense: list[Candidate],
    query_vector: list[float],
    fact_recaller: AtomicFactRecaller,
    episode_recaller: EpisodeRecaller,
    where: str,
    top_k: int,
    fact_child_candidates: int = 200,
) -> list[SearchEpisodeItem]:
    """Run the four-layer hierarchical episode retrieval pipeline.

    Layer 1: RRF fusion over pre-recalled sparse + dense episode candidates.
    Layer 2: MaxSim re-score via atomic-fact child retrieval (fact cosine ANN
             → group by parent memcell → episode re-score by best fact).
    Layer 3: RRF merge of Layer-1 and Layer-2 results, sliced to top_k.
    Layer 4: Pre-fetch facts for merged episodes, then single-pass eviction
             (fact outscoring its parent episode enters top-N instead).

    Args:
        query: Raw query string passed to amaxsim_retrieve.
        sparse: BM25 episode candidates from the caller's recall phase.
        dense: Vector ANN episode candidates from the caller's recall phase.
        query_vector: Pre-computed query embedding; reused for fact ANN recall
            and per-fact scoring in facts_for_episodes.
        fact_recaller: AtomicFactRecaller instance for child retrieval and
            facts_for_episodes.
        episode_recaller: EpisodeRecaller instance for MaxSim parent fetch.
        where: LanceDB filter clause (owner scope, tenant, etc.).
        top_k: Maximum number of items in the final merged slice before eviction.
        fact_child_candidates: How many atomic-fact ANN candidates to pull in
            Layer 2. Default 200.

    Returns:
        Shaped SearchEpisodeItem list (episodes with nested atomic_facts),
        sorted by score descending.
    """
    # Layer 1 — episode RRF fusion
    layer1_episodes = rrf(sparse, dense)

    # Layer 2 — MaxSim re-score via atomic-fact child retrieval
    layer2_episodes = await _maxsim_episode_rescore(
        query=query,
        query_vector=query_vector,
        fact_recaller=fact_recaller,
        episode_recaller=episode_recaller,
        where=where,
        child_candidates=fact_child_candidates,
    )

    # Layer 3 — RRF merge of episode-level results, slice to top_k
    merged = rrf(layer1_episodes, layer2_episodes)[:top_k]

    if not merged:
        logger.info("hierarchy_retrieve_empty_merge", top_k=top_k)
        return []

    # Layer 4a — pre-fetch facts for merged episodes
    ep_to_memcell = _build_ep_to_memcell(merged)
    episode_to_facts = await fact_recaller.facts_for_episodes(
        ep_to_memcell,
        where,
        per_episode=max(top_k * 2, 20),
        query_vector=query_vector,
    )

    # Layer 4b — single-pass eviction
    scored_items = _hierarchy_eviction_pass(merged, episode_to_facts)

    # Build episode pool for orphan fact parent lookup.
    # Include layer2_episodes so episodes surfaced only via MaxSim path
    # (not in the original sparse/dense recall) can still serve as parent.
    episode_pool = {c.id: c for c in (*sparse, *dense, *layer2_episodes)}

    return reshape_hybrid_output(scored_items, episode_pool=episode_pool)


def _hierarchy_eviction_pass(
    merged: list[Candidate],
    episode_to_facts: dict[str, list[FactCandidate]],
) -> list[ScoredItem]:
    """Single-pass eviction: fact outscoring its parent episode enters top-N.

    For each episode in merged order: if its best matching atomic fact scores
    higher than the episode itself, emit the fact as a ScoredItem
    (item_type='atomic_fact') and mark the episode as an orphan parent.
    Otherwise emit the episode directly as item_type='episode'.

    Args:
        merged: RRF-merged episode candidates, ordered by descending score.
        episode_to_facts: Map from episode_id to its pre-fetched FactCandidates,
            sorted by cosine similarity descending.

    Returns:
        Mixed list of ScoredItem instances (episodes and atomic_facts) ready
        for reshape_hybrid_output.
    """
    out: list[ScoredItem] = []

    for episode in merged:
        facts = episode_to_facts.get(episode.id, [])
        best_fact = facts[0] if facts else None

        if best_fact is not None and best_fact.score > episode.score:
            # Fact wins: emit fact; episode becomes orphan parent
            out.append(
                ScoredItem(
                    id=best_fact.id,
                    score=best_fact.score,
                    item_type="atomic_fact",
                    metadata=best_fact.metadata,
                    parent_episode_id=episode.id,
                )
            )
            logger.debug(
                "hierarchy_eviction_fact_wins",
                episode_id=episode.id,
                fact_id=best_fact.id,
                fact_score=best_fact.score,
                episode_score=episode.score,
            )
        else:
            # Episode wins: emit episode with its metadata intact
            out.append(
                ScoredItem(
                    id=episode.id,
                    score=episode.score,
                    item_type="episode",
                    metadata=dict(episode.metadata),
                    parent_episode_id=None,
                )
            )

    return out


# ── Internal helpers ─────────────────────────────────────────────────────


async def _maxsim_episode_rescore(
    *,
    query: str,
    query_vector: list[float],
    fact_recaller: AtomicFactRecaller,
    episode_recaller: EpisodeRecaller,
    where: str,
    child_candidates: int,
) -> list[Candidate]:
    """Run amaxsim_retrieve to produce MaxSim-rescored episode candidates.

    Atomic facts serve as child documents (their metadata["parent_id"] is
    the memcell_id). Episodes are fetched as parents via
    episode_recaller.fetch_by_parent_ids.

    ``amaxsim_retrieve`` calls ``child_retrieve`` exactly once with the
    original query string. We reuse the pre-computed ``query_vector`` to
    avoid a redundant embed call.

    Args:
        query: Raw query string (passed verbatim to amaxsim_retrieve).
        query_vector: Pre-computed query embedding; used directly for child
            ANN recall, bypassing a second embed call.
        fact_recaller: Provides the child ANN retrieval function.
        episode_recaller: Provides the parent fetch function.
        where: LanceDB filter clause.
        child_candidates: Number of atomic-fact candidates to pull per call.

    Returns:
        Episode candidates re-scored by their best matching atomic fact.
    """

    async def child_retrieve(_q: str, n: int) -> Sequence[Candidate]:
        # amaxsim_retrieve calls this exactly once with the original query string.
        # Reuse the pre-computed query_vector instead of re-embedding.
        return await fact_recaller.dense_recall(query_vector, where, limit=n)

    async def parent_fetch(memcell_ids: list[str]) -> list[Candidate]:
        return await episode_recaller.fetch_by_parent_ids(memcell_ids, where)

    return await amaxsim_retrieve(
        query,
        child_retrieve=child_retrieve,
        parent_fetch=parent_fetch,
        top_n=50,
        child_candidates=child_candidates,
    )


def _build_ep_to_memcell(episodes: list[Candidate]) -> dict[str, str]:
    """Extract episode_id → memcell_id mapping from episode candidates.

    Episodes store their source memcell id in metadata["parent_id"].
    Entries missing or having a non-string parent_id are silently skipped
    (they will receive no facts during Layer 4).

    Args:
        episodes: Merged episode candidate list.

    Returns:
        Dict mapping episode LanceDB id to memcell id.
    """
    result: dict[str, str] = {}
    for ep in episodes:
        mc_id = ep.metadata.get("parent_id")
        if isinstance(mc_id, str) and mc_id:
            result[ep.id] = mc_id
    return result

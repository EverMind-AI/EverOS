"""Hierarchical episode retrieval — two-path recall fused with per-fact eviction.

Episode HYBRID search path: combines episode-level hybrid recall (Layer 1)
with fact-driven MaxSim re-scoring (Layer 2), merges via RRF (Layer 3), then
runs a hierarchical fact eviction where parent episode and its facts compete on a
single LR-calibrated scale and the best fact replaces the episode when it
wins (Layer 4).

Uses everalgo operators as pure algorithm primitives; all I/O is injected
via recaller callbacks.  No changes to the everalgo library are required.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from everalgo.rank import amaxsim_retrieve
from everalgo.rank.fusion import cosine_to_lr_score, rrf
from everalgo.types import Candidate, FactCandidate, ScoredItem

from everos.core.observability.logging import get_logger

from .dto import SearchEpisodeItem
from .shaper import reshape_hybrid_output

if TYPE_CHECKING:
    from collections.abc import Sequence

    from everos.memory.search.recall.atomic_fact import AtomicFactRecaller
    from everos.memory.search.recall.episode import EpisodeRecaller

logger = get_logger(__name__)

_HIERARCHY_ALPHA = 1.0
_HIERARCHY_FACTS_PER_EPISODE = 3


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
    alpha: float = _HIERARCHY_ALPHA,
    min_score: float | None = None,
) -> list[SearchEpisodeItem]:
    """Run the four-layer hierarchical episode retrieval pipeline.

    Layer 1: RRF fusion over pre-recalled sparse + dense episode candidates.
    Layer 2: MaxSim re-score via atomic-fact child retrieval (fact cosine ANN
             → group by parent memcell → episode re-score by best fact).
    Layer 3: RRF merge of Layer-1 and Layer-2 results, sliced to top_k.
    Layer 4: Pre-fetch facts for merged episodes, then hierarchical eviction —
             parent and facts compete on one LR-calibrated scale; the best
             fact replaces its episode when it wins.

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
        alpha: Child (fact) weight in the Layer-4 LR-scale blend. Default
            ``_HIERARCHY_ALPHA``.
        min_score: Optional post-Layer-4 relevance floor on the LR-calibrated
            score in ``[0, 1]``; items below it are dropped. ``None`` disables.

    Returns:
        Shaped SearchEpisodeItem list (episodes with nested atomic_facts),
        sorted by score descending, each carrying an LR-calibrated score.
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
    ep_to_parents = _build_ep_to_fact_parents(merged)
    episode_to_facts = await fact_recaller.facts_for_episodes(
        ep_to_parents,
        where,
        per_episode=max(top_k * 2, 20),
        query_vector=query_vector,
    )

    ep_cosine: dict[str, float] = {}
    for c in (*dense, *layer2_episodes):
        if c.id:
            ep_cosine[c.id] = max(ep_cosine.get(c.id, 0.0), c.score)
    ep_bm25 = {c.id: c.score for c in sparse if c.id}

    scored_items = _hierarchy_eviction_pass(
        merged,
        episode_to_facts,
        ep_cosine=ep_cosine,
        ep_bm25=ep_bm25,
        alpha=alpha,
    )

    # Build episode pool for orphan fact parent lookup.
    # Include layer2_episodes so episodes surfaced only via MaxSim path
    # (not in the original sparse/dense recall) can still serve as parent.
    episode_pool = {c.id: c for c in (*sparse, *dense, *layer2_episodes)}

    shaped = reshape_hybrid_output(scored_items, episode_pool=episode_pool)

    # Post-Layer-4 relevance floor on the LR-calibrated score.
    if min_score is not None:
        shaped = [item for item in shaped if item.score >= min_score]
    return shaped


def _hierarchy_eviction_pass(
    merged: list[Candidate],
    episode_to_facts: dict[str, list[FactCandidate]],
    *,
    ep_cosine: dict[str, float],
    ep_bm25: dict[str, float],
    alpha: float = _HIERARCHY_ALPHA,
    facts_per_episode: int = _HIERARCHY_FACTS_PER_EPISODE,
) -> list[ScoredItem]:
    """Hierarchical fact eviction: parent and facts compete on one LR-calibrated scale.

    For each merged episode the parent and its candidate facts are calibrated
    to an LR probability via ``cosine_to_lr_score`` so a raw fact cosine and an
    episode's recall relevance become directly comparable (replacing the prior
    cosine-vs-RRF comparison, which mixed scales). Each fact's blended score is
    ``alpha * child_lr + (1 - alpha) * parent_lr``; the single best-scoring
    fact replaces the episode (eviction) when it beats the parent's own LR
    score, otherwise the episode is emitted at ``parent_lr``.

    Args:
        merged: RRF-merged episode candidates, ordered by descending score.
            Their ``.score`` (RRF) is used only for ordering, not for scoring.
        episode_to_facts: Map from episode id to its pre-fetched FactCandidates,
            sorted by cosine similarity descending.
        ep_cosine: Per-episode best cosine relevance (dense / MaxSim routes).
        ep_bm25: Per-episode BM25 score (sparse route); ``0.0`` when absent.
        alpha: Child (fact) weight in the blend; ``1.0`` lets the fact's own
            calibrated relevance fully drive the blended score.
        facts_per_episode: Max facts per episode entered into the competition.

    Returns:
        Mixed list of ScoredItem instances (episodes and atomic_facts), each
        carrying an LR-calibrated ``score`` in ``[0, 1]``, ready for
        reshape_hybrid_output.
    """
    out: list[ScoredItem] = []

    for episode in merged:
        parent_bm25 = ep_bm25.get(episode.id, 0.0)
        parent_cosine = ep_cosine.get(episode.id, 0.0)
        parent_lr = cosine_to_lr_score(parent_cosine, parent_bm25)

        # A fact must strictly beat the parent's LR score to evict it.
        best_fact: FactCandidate | None = None
        best_blended = parent_lr
        for fact in episode_to_facts.get(episode.id, [])[:facts_per_episode]:
            child_lr = cosine_to_lr_score(fact.score, parent_bm25)
            blended = alpha * child_lr + (1.0 - alpha) * parent_lr
            if blended > best_blended:
                best_blended = blended
                best_fact = fact

        if best_fact is not None:
            # Fact wins: emit fact at its blended score; episode becomes orphan parent.
            out.append(
                ScoredItem(
                    id=best_fact.id,
                    score=best_blended,
                    item_type="atomic_fact",
                    metadata=best_fact.metadata,
                    parent_episode_id=episode.id,
                )
            )
            logger.debug(
                "hierarchy_eviction_fact_wins",
                episode_id=episode.id,
                fact_id=best_fact.id,
                fact_score=best_blended,
                episode_score=parent_lr,
            )
        else:
            # Episode wins: emit episode at its LR-calibrated parent score.
            out.append(
                ScoredItem(
                    id=episode.id,
                    score=parent_lr,
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
    the episode entry_id). Episodes are fetched as parents via
    episode_recaller.fetch_by_entry_ids.

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

    async def parent_fetch(entry_ids: list[str]) -> list[Candidate]:
        return await episode_recaller.fetch_by_entry_ids(entry_ids, where)

    return await amaxsim_retrieve(
        query,
        child_retrieve=child_retrieve,
        parent_fetch=parent_fetch,
        top_n=50,
        child_candidates=child_candidates,
    )


def _build_ep_to_fact_parents(episodes: list[Candidate]) -> dict[str, list[str]]:
    """Map episode candidate id to all possible fact parent_id values.

    New facts (post-1.5): parent_id = episode entry_id.
    Old facts (pre-1.5): parent_id = memcell_id (episode.parent_id).
    Both are collected so the IN query covers both eras without backfill.

    Invariant: entry_id (ep_*) and memcell_id (mc_*) namespaces never
    overlap, so mixing them in one IN clause is safe.

    Args:
        episodes: Merged episode candidate list.

    Returns:
        Dict mapping episode LanceDB id to a list of candidate parent_ids
        (entry_id and/or memcell_id).
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

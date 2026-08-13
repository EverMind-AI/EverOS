"""Heap-expand hierarchy — heap-driven lazy expansion with global top-N competition.

Replaces the four-layer serial pipeline (hierarchy.py) with a heap-based
approach where RRF orders the expansion priority and LR-calibrated scores
drive the global competition between episodes and their atomic facts.

Pure synchronous module — zero I/O, zero async. Designed for zero-change
migration to everalgo.
"""

from __future__ import annotations

import heapq
from typing import TYPE_CHECKING

from everalgo.rank.fusion import cosine_to_lr_score, lr, rrf
from everalgo.types import Candidate, FactCandidate, ScoredItem

if TYPE_CHECKING:
    from everalgo.rank.weight import LRCoefs


def build_ep_to_fact_parents(
    episodes: list[Candidate],
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


def heap_expand(
    *,
    sparse: list[Candidate],
    dense: list[Candidate],
    episode_to_facts: dict[str, list[FactCandidate]],
    top_k: int = 10,
    alpha: float = 1.0,
    rrf_k: int = 60,
    max_convergence_rounds: int = 10,
    facts_per_episode: int = 3,
    lr_coefs: LRCoefs | None = None,
) -> list[ScoredItem]:
    """Heap-expand hierarchy: RRF orders expansion, LR scores drive competition.

    Args:
        sparse: BM25 episode candidates (descending by BM25 score).
        dense: Vector ANN episode candidates (descending by cosine).
        episode_to_facts: Pre-fetched facts per episode, each list sorted
            by cosine descending.
        top_k: Maximum items in the final output.
        alpha: Fact weight in the blend (1.0 = fact score only).
        rrf_k: RRF constant (default 60).
        max_convergence_rounds: Stop after this many consecutive rounds
            with no top-N change.
        facts_per_episode: Max facts per episode to enter competition.
        lr_coefs: Override LR coefficients; None uses production defaults.

    Returns:
        Mixed list of ScoredItem (episodes + atomic_facts), sorted by
        score descending. Ready for ``reshape_hybrid_output``.
    """
    if not sparse and not dense:
        return []

    # Phase 1 — dual fusion
    bm25_scores = {c.id: c.score for c in sparse}
    lr_results = lr(dense, sparse, coefs=lr_coefs)
    episode_scores = {c.id: c.score for c in lr_results}
    rrf_results = rrf(sparse, dense, k=rrf_k)

    if not rrf_results:
        return []

    # Phase 2 — heap + top-N init
    heap: list[tuple[float, str]] = []
    for doc in rrf_results:
        heapq.heappush(heap, (-doc.score, doc.id))

    # topn: {id: (Candidate, lr_score, item_type, source_episode_id)}
    topn: dict[str, tuple[Candidate | FactCandidate, float, str, str]] = {}
    for doc in rrf_results[:top_k]:
        topn[doc.id] = (doc, episode_scores.get(doc.id, 0.0), "episode", doc.id)

    # Phase 3 — heap convergence loop
    convergence_count = 0
    while heap and convergence_count < max_convergence_rounds:
        _, episode_id = heapq.heappop(heap)
        prev_keys = set(topn.keys())

        _expand_one_episode(
            episode_id,
            topn=topn,
            episode_to_facts=episode_to_facts,
            bm25_scores=bm25_scores,
            episode_scores=episode_scores,
            alpha=alpha,
            facts_per_episode=facts_per_episode,
            top_k=top_k,
            lr_coefs=lr_coefs,
        )

        if set(topn.keys()) == prev_keys:
            convergence_count += 1
        else:
            convergence_count = 0

    # Phase 4 — output
    sorted_entries = sorted(topn.values(), key=lambda v: v[1], reverse=True)
    result: list[ScoredItem] = []
    for item, score, item_type, source_ep_id in sorted_entries:
        if item_type == "episode":
            result.append(
                ScoredItem(
                    id=item.id,
                    score=score,
                    item_type="episode",
                    metadata=dict(item.metadata),
                )
            )
        else:
            result.append(
                ScoredItem(
                    id=item.id,
                    score=score,
                    item_type="atomic_fact",
                    metadata=dict(item.metadata),
                    parent_episode_id=source_ep_id,
                )
            )
    return result


def _expand_one_episode(
    episode_id: str,
    *,
    topn: dict[str, tuple[Candidate | FactCandidate, float, str, str]],
    episode_to_facts: dict[str, list[FactCandidate]],
    bm25_scores: dict[str, float],
    episode_scores: dict[str, float],
    alpha: float,
    facts_per_episode: int,
    top_k: int,
    lr_coefs: LRCoefs | None,
) -> None:
    """Expand one episode's facts and compete with top-N in place."""
    # Pre-fetch 2× candidates so LR scoring can filter to the best N.
    facts = episode_to_facts.get(episode_id, [])[: facts_per_episode * 2]
    if not facts:
        return

    parent_bm25 = bm25_scores.get(episode_id, 0.0)
    parent_lr = episode_scores.get(episode_id, 0.0)

    scored: list[tuple[float, FactCandidate]] = []
    for fact in facts:
        child_lr = cosine_to_lr_score(fact.score, parent_bm25, coefs=lr_coefs)
        blended = alpha * child_lr + (1.0 - alpha) * parent_lr
        scored.append((blended, fact))

    scored.sort(key=lambda t: t[0], reverse=True)
    top_facts = scored[:facts_per_episode]

    min_topn_score = min((v[1] for v in topn.values()), default=0.0) if topn else 0.0
    any_entered = False

    for fact_score, fact in top_facts:
        if fact_score <= 0.0:
            continue
        if len(topn) < top_k or fact_score > min_topn_score:
            topn[fact.id] = (fact, fact_score, "atomic_fact", episode_id)
            any_entered = True
            if len(topn) > top_k:
                worst_id = min(topn, key=lambda k: topn[k][1])
                del topn[worst_id]
            min_topn_score = min((v[1] for v in topn.values()), default=0.0)

    if any_entered and episode_id in topn:
        del topn[episode_id]

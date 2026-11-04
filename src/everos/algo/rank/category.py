"""Category-aware retrieval — soft, search-after, rolled-up-from-hits boost.

The category dimension is treated as a *precision* signal layered on top of a
two-stage ``base_retrieve`` → ``rerank_fn`` pipeline. The design is deliberately
modest:

1. After recall, estimate a category mass distribution ``p(c)`` from the top
   recall hits, weighted by their recall ``Candidate.score`` (not by count, so
   a few strong hits dominate a long tail of weak ones).
2. Compute ``conf = p.top1 - p.top2`` — a confidence proxy. When the query is
   ambiguous or spans multiple categories, ``p`` is flat and ``conf`` collapses
   toward zero, automatically muting the boost.
3. Rerank the recall pool with the caller's ``rerank_fn`` (e.g. a cross-encoder).
4. Combine ``final = norm_rel + λ·conf·p(category_of(t))`` and resort.

Category never gates recall: misclassified or unclassified hits stay in the
pool. The boost is a tiebreaker, not a filter — this is the price of avoiding
"router selected the wrong shard" failure modes at the cost of a few percent
NDCG gain over pure cross-encoder rerank.

**Timing invariant.** ``rollup_category_mass`` MUST be called *before*
``rerank_fn``, because ``rerank_fn`` overwrites ``Candidate.score`` with the
relevance score and the mass needs the recall score. ``apply_category_boost``
runs on the rerank output. The ``acategory_retrieve`` facade enforces this
ordering.

The primitives ``rollup_category_mass`` and ``apply_category_boost`` are
pure-compute and exposed so callers that want a learning-to-rank reformulation
(treating category affinity as one feature among many) can compose them
without going through the facade.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from everos.algo.rank.protocols import RerankFn, RetrieveFn
    from everos.algo.types import Candidate

__all__ = [
    "acategory_retrieve",
    "apply_category_boost",
    "rollup_category_mass",
]


def rollup_category_mass(
    recall_hits: Sequence[Candidate],
    *,
    category_key: str = "category_id",
    top_m: int = 50,
) -> tuple[dict[str, float], float]:
    """Score-weighted category distribution rolled up from recall hits.

    Reads ``Candidate.metadata[category_key]`` from up to the first ``top_m`` hits and
    accumulates ``mass[cat] += hit.score``. Empty/missing categories are skipped so
    unclassified hits do not form a phantom bucket that competes with real ones.

    Must be called **before** ``rerank_fn`` rewrites ``Candidate.score``.

    Args:
        recall_hits: Output of ``base_retrieve`` — score is the recall score (e.g. ANN
            cosine, RRF, BM25). The function does not care which; it treats whatever
            ``Candidate.score`` carries as the relative-ordering signal.
        category_key: ``metadata`` key to read the per-candidate category id from.
        top_m: Window over the recall hits used to compute the distribution. The
            recall pool is typically much larger than the rerank pool; sampling the
            top of it keeps low-quality hits from skewing ``p(c)``.

    Returns:
        ``(p, conf)``:
            - ``p`` — probability distribution over categories that received any mass.
              Empty dict when no hit carried a non-empty category.
            - ``conf = p.top1 - p.top2``; degenerates to ``0.0`` when ``p`` is empty
              and ``1.0`` when ``p`` has a single entry.
    """
    mass: dict[str, float] = {}
    for hit in recall_hits[:top_m]:
        cat = hit.metadata.get(category_key, "")
        if not isinstance(cat, str) or cat == "":
            continue
        mass[cat] = mass.get(cat, 0.0) + float(hit.score)

    total = sum(mass.values())
    if total <= 0.0:
        return {}, 0.0

    p = {cat: m / total for cat, m in mass.items()}
    sorted_probs = sorted(p.values(), reverse=True)
    if len(sorted_probs) == 1:
        return p, 1.0
    conf = sorted_probs[0] - sorted_probs[1]
    return p, conf


def apply_category_boost(
    reranked: Sequence[Candidate],
    p: dict[str, float],
    conf: float,
    *,
    lam: float = 0.1,
    category_key: str = "category_id",
) -> list[Candidate]:
    """Blend a category mass distribution into post-rerank scores.

    ``final[i] = norm_rel[i] + lam * clamp(conf, 0, 1) * p.get(cat_i, 0.0)``

    The min-max normalization brings ``rel`` and ``p`` to comparable scales so a
    small ``lam`` (default ``0.1``) keeps the boost in tiebreaker territory.
    ``conf`` shrinks the effective lambda for ambiguous queries automatically.

    Args:
        reranked: Output of ``rerank_fn`` — ``Candidate.score`` is the relevance score.
        p: Category distribution from ``rollup_category_mass``.
        conf: Confidence (top1 - top2 of ``p``) from ``rollup_category_mass``.
        lam: Boost strength. Small by design — the boost is a tiebreaker, not a
            primary ranking signal.
        category_key: ``metadata`` key holding each candidate's category id.

    Returns:
        New ``Candidate`` list (input untouched) with ``score`` rewritten to ``final``
        and sorted descending. Empty list when ``reranked`` is empty.
    """
    if not reranked:
        return []

    lam_eff = lam * max(0.0, min(1.0, conf))

    rels = [c.score for c in reranked]
    lo, hi = min(rels), max(rels)
    # All-equal rels collapse to a flat 0.5 — the category boost is then the only
    # differentiator, which is the entire point of this layer when rerank can't
    # separate the pool.
    norm_rels = [(r - lo) / (hi - lo) for r in rels] if hi > lo else [0.5] * len(rels)

    boosted: list[Candidate] = []
    for cand, norm_rel in zip(reranked, norm_rels, strict=True):
        cat = cand.metadata.get(category_key, "")
        p_cat = p.get(cat, 0.0) if isinstance(cat, str) else 0.0
        final = norm_rel + lam_eff * p_cat
        boosted.append(cand.model_copy(update={"score": final}))

    boosted.sort(key=lambda c: c.score, reverse=True)
    return boosted


async def acategory_retrieve(
    query: str,
    *,
    base_retrieve: RetrieveFn,
    rerank_fn: RerankFn,
    recall_n: int = 200,
    rerank_n: int | None = None,
    mass_top_m: int = 50,
    lam: float = 0.1,
    category_key: str = "category_id",
    top_n: int = 20,
) -> list[Candidate]:
    """End-to-end retrieve → rollup → rerank → boost facade.

    Pure orchestration over caller-injected I/O callables. The single non-trivial
    invariant is the ordering between mass computation and rerank: the mass uses
    the recall score, and ``rerank_fn`` overwrites that score. This function does
    the rollup **before** dispatching to ``rerank_fn``.

    Args:
        query: The search query.
        base_retrieve: Caller's recall function (e.g. an ``ahybrid_retrieve`` closure
            over dense + sparse routes). Must return ``Candidate`` objects whose
            ``metadata[category_key]`` is set to the document's category id where one
            is known.
        rerank_fn: Caller's rerank function (e.g. a cross-encoder closure). Returns
            ``Candidate`` objects with ``score`` overwritten to relevance.
        recall_n: How many candidates to fetch from ``base_retrieve``. Should be wide
            enough that the right answer is in the pool on relevance alone — the
            category boost is precision-only, never compensates for missed recall.
        rerank_n: Slice of the recall pool passed to ``rerank_fn``. ``None`` reranks
            the entire pool.
        mass_top_m: Window of recall hits used to estimate ``p(c)``. See
            ``rollup_category_mass``.
        lam: Boost strength (small by design). See ``apply_category_boost``.
        category_key: ``metadata`` key holding each candidate's category id.
        top_n: Final truncation after boost.

    Returns:
        Top-``top_n`` candidates by ``final`` score (relevance + category boost).
        Empty list when ``base_retrieve`` returns nothing.

    Resource contract:
        - ``base_retrieve`` is called exactly once with ``recall_n``.
        - ``rerank_fn`` is called exactly once with ``min(recall_n, rerank_n)``.
        - No LLM calls owned by this facade — both I/O calls are caller-supplied.
    """
    hits = await base_retrieve(query, recall_n)
    if not hits:
        return []

    p, conf = rollup_category_mass(hits, category_key=category_key, top_m=mass_top_m)

    rerank_pool = hits[:rerank_n] if rerank_n is not None else hits
    reranked = await rerank_fn(query, rerank_pool)

    boosted = apply_category_boost(
        reranked, p, conf, lam=lam, category_key=category_key
    )
    return boosted[:top_n]

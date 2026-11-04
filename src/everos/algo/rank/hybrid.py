"""Hybrid retrieval — dual-route RRF / LR fusion convenience wrapper.

Caller injects ``dense_retrieve`` and ``sparse_retrieve`` (any ``RetrieveFn`` —
typically vector + BM25 in production); this operator runs them in parallel, fuses with
either Reciprocal Rank Fusion (``everos.algo.rank.fusion.rrf``) or Logistic Regression fusion
(``everos.algo.rank.fusion.lr``), and returns the top-N. Pure algorithm orchestration:
no LLM, no I/O, no metadata.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Literal

from everos.algo.rank.fusion import lr, rrf

if TYPE_CHECKING:
    from everos.algo.rank.protocols import RetrieveFn
    from everos.algo.rank.weight import LRCoefs
    from everos.algo.types import Candidate

__all__ = ["ahybrid_retrieve"]


async def ahybrid_retrieve(
    query: str,
    *,
    dense_retrieve: RetrieveFn,
    sparse_retrieve: RetrieveFn,
    top_n: int | None = None,
    dense_candidates: int = 50,
    sparse_candidates: int = 50,
    fusion: Literal["rrf", "lr"] = "rrf",
    rrf_k: int = 60,
    lr_coefs: LRCoefs | None = None,
    min_score: float | None = None,
) -> list[Candidate]:
    """Dual-route RRF / LR fusion retrieval.

    Fires both retrieval routes in parallel via ``asyncio.gather``, fuses the ranked lists
    with either Reciprocal Rank Fusion or Logistic Regression fusion, slices to ``top_n``,
    and optionally filters by minimum score.

    Args:
        query: The search query string.
        dense_retrieve: Async retrieval function for the dense (vector) route.
        sparse_retrieve: Async retrieval function for the sparse (keyword) route.
        top_n: Maximum number of candidates to return. ``None`` means no truncation.
        dense_candidates: How many candidates to request from ``dense_retrieve``.
        sparse_candidates: How many candidates to request from ``sparse_retrieve``.
        fusion: Fusion algorithm — ``"rrf"`` (Reciprocal Rank Fusion, default) or ``"lr"`` (Logistic Regression).
        rrf_k: Reciprocal Rank Fusion smoothing constant (higher = less aggressive re-ranking). Ignored when ``fusion="lr"``.
        lr_coefs: LR weighting coefficients for ``fusion="lr"``. ``None`` defers to ``weight.default_lr_coefs()``.
            Ignored when ``fusion="rrf"``.
        min_score: When set, candidates whose score is strictly below this threshold are filtered from the result.
            Applied after ``top_n`` truncation (same order as ``amaxsim_retrieve``).

    Returns:
        Fused and ranked list of at most ``top_n`` candidates, optionally filtered by ``min_score``.

    Algorithm:
        1. Fire ``dense_retrieve`` and ``sparse_retrieve`` in parallel.
        2. If both routes return results, fuse with the selected algorithm (RRF or LR).
        3. Truncate to ``top_n``.
        4. Apply ``min_score`` filter if set (``score >= min_score``).

    Resource contract:
        - ``dense_retrieve`` is called exactly once with ``dense_candidates``.
        - ``sparse_retrieve`` is called exactly once with ``sparse_candidates``.
        - No LLM calls are made.
    """
    dense, sparse = await asyncio.gather(
        dense_retrieve(query, dense_candidates),
        sparse_retrieve(query, sparse_candidates),
    )
    if not dense and not sparse:
        return []
    if not dense:
        fused = sparse
    elif not sparse:
        fused = dense
    elif fusion == "lr":
        fused = lr(dense, sparse, coefs=lr_coefs)
    else:  # rrf
        fused = rrf(dense, sparse, k=rrf_k)
    if top_n is not None:
        fused = fused[:top_n]
    if min_score is not None:
        fused = [c for c in fused if c.score >= min_score]
    return fused

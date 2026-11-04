"""Fusion algorithms — RRF / LR / cosine→LR / propagation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from everos.algo.rank import weight as _weight
from everos.algo.types import Candidate

if TYPE_CHECKING:
    from collections.abc import Sequence

    from everos.algo.rank.weight import LRCoefs

__all__ = [
    "cosine_to_lr_score",
    "lr",
    "rrf",
    "score_propagation",
    "vector_anchored",
]

logger = logging.getLogger(__name__)


def rrf(*sources: Sequence[Candidate], k: int = 60) -> list[Candidate]:
    """Reciprocal Rank Fusion over N ranked lists; ``score = Σ 1/(k+rank_i)``.

    Returns candidates sorted descending by accumulated RRF score; empty input → empty list.
    """
    doc_rrf_scores: dict[str, float] = {}
    doc_map: dict[str, Candidate] = {}

    for ranked_list in sources:
        for rank, doc in enumerate(ranked_list, start=1):
            if not doc.id:
                continue
            doc_map.setdefault(doc.id, doc)
            doc_rrf_scores[doc.id] = doc_rrf_scores.get(doc.id, 0.0) + 1.0 / (k + rank)

    items: list[tuple[str, float]] = sorted(
        doc_rrf_scores.items(), key=lambda kv: kv[1], reverse=True
    )
    return [
        doc_map[doc_id].model_copy(update={"score": rrf_score})
        for doc_id, rrf_score in items
    ]


def lr(
    emb_results: Sequence[Candidate],
    bm25_results: Sequence[Candidate],
    *,
    coefs: LRCoefs | None = None,
) -> list[Candidate]:
    """Logistic Regression fusion of embedding + BM25 results.

    ``logit = emb*emb_coef + bm25*bm25_coef + intercept; prob = sigmoid(logit)``.
    Returns candidates sorted descending by probability; ``coefs=None`` defers to ``weight.default_lr_coefs()``.
    """
    return _weight.multi_field_weighting(
        {"emb": list(emb_results), "bm25": list(bm25_results)},
        coefs=coefs,
    )


def vector_anchored(
    dense: Sequence[Candidate],
    sparse: Sequence[Candidate],
    *,
    saturation_k: float = 5.0,
    alpha: float = 0.7,
) -> list[Candidate]:
    """Vector-anchored fusion of dense (cosine) + sparse (BM25) candidates."""
    vec_score_map: dict[str, float] = {c.id: c.score for c in dense if c.id}
    kw_sat_map: dict[str, float] = {
        c.id: (c.score / (c.score + saturation_k) if c.score > 0 else 0.0)
        for c in sparse
        if c.id
    }

    vec_floor = min(vec_score_map.values()) if vec_score_map else 0.0
    kw_floor = min(kw_sat_map.values()) if kw_sat_map else 0.0

    doc_map: dict[str, Candidate] = {c.id: c for c in dense if c.id}
    for c in sparse:
        if c.id and c.id not in doc_map:
            doc_map[c.id] = c

    out: list[Candidate] = []
    for doc_id, doc in doc_map.items():
        vs = vec_score_map.get(doc_id, vec_floor)
        ks = kw_sat_map.get(doc_id, kw_floor)
        out.append(doc.model_copy(update={"score": alpha * vs + (1.0 - alpha) * ks}))

    out.sort(key=lambda c: c.score, reverse=True)
    return out


def cosine_to_lr_score(
    sim: float,
    parent_bm25: float = 0.0,
    *,
    coefs: LRCoefs | None = None,
) -> float:
    """Calibrate a raw cosine similarity to an LR probability in ``[0, 1]``."""
    out = _weight.multi_field_weighting(
        {
            "emb": [Candidate(id="_scalar", score=sim)],
            "bm25": [Candidate(id="_scalar", score=parent_bm25)],
        },
        coefs=coefs,
    )
    return out[0].score


def score_propagation(
    parents: Sequence[Candidate],
    children: Sequence[Candidate],
    *,
    alpha: float = 1.0,
) -> list[Candidate]:
    """Blend child + parent scores: ``final = alpha*child + (1-alpha)*parent``.

    Children whose parent cannot be resolved still appear with parent contribution treated as ``0``.
    Caller decides whether to sort the result.
    """
    parent_score_lookup = {p.id: p.score for p in parents}

    out: list[Candidate] = []
    for child in children:
        parent_id = child.metadata.get("parent_id", "")
        parent_score = (
            parent_score_lookup.get(parent_id, 0.0)
            if isinstance(parent_id, str)
            else 0.0
        )
        final = alpha * child.score + (1.0 - alpha) * parent_score
        out.append(child.model_copy(update={"score": final}))
    return out

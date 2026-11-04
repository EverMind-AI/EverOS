"""Weight tools — single-list / multi-list LR + LR coefficient supply."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Sequence

    from everos.algo.types import Candidate

__all__ = [
    "LRCoefs",
    "default_lr_coefs",
    "multi_field_weighting",
    "weighted_score",
]


def weighted_score(
    items: Sequence[Candidate],
    *,
    fields: dict[str, float],
    intercept: float = 0.0,
) -> list[Candidate]:
    """Single-list LR weighting — replace each item's score with ``sigmoid(Σ metadata[k]*coef + intercept)``.

    Missing metadata keys contribute 0. Result is **not** sorted; caller decides.
    """
    out: list[Candidate] = []
    for item in items:
        logit = (
            sum(_get_scalar(item.metadata, key) * coef for key, coef in fields.items())
            + intercept
        )
        prob = 1.0 / (1.0 + math.exp(-logit))
        out.append(item.model_copy(update={"score": prob}))
    return out


def multi_field_weighting(
    sources: dict[str, Sequence[Candidate]],
    *,
    weights: dict[str, float] | None = None,
    intercept: float = 0.0,
    coefs: LRCoefs | None = None,
) -> list[Candidate]:
    """Multi-source LR fusion — N ranked lists → single LR probability per doc, sorted descending.

    ``weights=None`` activates LR-trained mode using ``coefs`` (defaults to ``default_lr_coefs()``);
    ``weights=dict(...)`` activates generic mode with caller-supplied coefficients.
    """
    if weights is None:
        weights, intercept = _lrcoefs_to_weights(coefs)

    by_id: dict[str, dict[str, float]] = {}
    doc_map: dict[str, Candidate] = {}

    for name, ranked in sources.items():
        for item in ranked:
            if not item.id:
                continue
            doc_map.setdefault(item.id, item)
            by_id.setdefault(item.id, {})[name] = item.score

    probs: list[tuple[str, float]] = []
    for doc_id, score_map in by_id.items():
        logit = intercept + sum(
            score_map.get(name, 0.0) * coef for name, coef in weights.items()
        )
        prob = 1.0 / (1.0 + math.exp(-logit))
        probs.append((doc_id, prob))

    probs.sort(key=lambda kv: kv[1], reverse=True)
    return [
        doc_map[doc_id].model_copy(update={"score": prob}) for doc_id, prob in probs
    ]


class LRCoefs(NamedTuple):
    """Trained LR coefficients for the 2-source (emb + bm25) fusion in ``fusion.lr`` / ``cosine_to_lr_score``."""

    emb_coef: float = 6.27473151675093
    bm25_coef: float = 0.09395183408310023
    intercept: float = -4.858095765012703


def default_lr_coefs() -> LRCoefs:
    """Return the current default LR coefficients (monkey-patchable; ``fusion.lr`` defers here when ``coefs=None``)."""
    return LRCoefs()


# ─── Internal helpers ───────────────────────────────────────────────────────


def _lrcoefs_to_weights(coefs: LRCoefs | None) -> tuple[dict[str, float], float]:
    """Resolve ``LRCoefs | None`` to a ``(weights, intercept)`` pair; ``None`` uses ``default_lr_coefs()``."""
    resolved = coefs or default_lr_coefs()
    return {"emb": resolved.emb_coef, "bm25": resolved.bm25_coef}, resolved.intercept


def _get_scalar(metadata: dict[str, Any], key: str) -> float:
    """Extract a scalar metadata value; treat missing / non-numeric as 0."""
    value = metadata.get(key, 0)
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0

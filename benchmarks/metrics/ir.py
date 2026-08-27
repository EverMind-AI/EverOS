"""Ranked-retrieval metrics over the injected episode list.

These grade the list the ANSWER model actually sees -- assembly order, ``top_k``
included -- which is the right target for "did the context contain the evidence". For
grading the decider's own choice use :mod:`benchmarks.metrics.core` instead; the two
answer different questions and were conflated once already.

Relevance is session-level: an episode counts as relevant when its session id is in
gold. Duplicate sessions in the list are credited once for recall and each time for
precision, matching how a reader would experience the context.
"""

from __future__ import annotations

import math
import statistics as st
from collections.abc import Sequence
from typing import Any

KS = (1, 3, 5, 10, 20)


def _dcg(rels: Sequence[int]) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


def score(
    ranked: Sequence[Sequence[str]], gold: Sequence[set[str]], ks: Sequence[int] = KS
) -> dict[str, Any]:
    """``ranked[i]`` is question i's session ids in injection order; ``gold[i]`` its
    gold.
    """
    per_k: dict[int, dict[str, list[float]]] = {
        k: {"recall": [], "precision": [], "ndcg": []} for k in ks
    }
    rr: list[float] = []
    ap: list[float] = []
    full: list[float] = []
    n = 0
    for sessions, g in zip(ranked, gold, strict=False):
        if not g:
            continue
        n += 1
        rels = [1 if s in g else 0 for s in sessions]
        first = next((i + 1 for i, r in enumerate(rels) if r), None)
        rr.append(1.0 / first if first else 0.0)
        seen: set[str] = set()
        hits = 0
        precs: list[float] = []
        for i, s in enumerate(sessions):
            if s in g and s not in seen:
                seen.add(s)
                hits += 1
                precs.append(hits / (i + 1))
        ap.append(sum(precs) / len(g) if precs else 0.0)
        full.append(1.0 if len({s for s in sessions if s in g}) == len(g) else 0.0)
        for k in ks:
            top, rl = list(sessions[:k]), rels[:k]
            per_k[k]["recall"].append(len({s for s in top if s in g}) / len(g))
            per_k[k]["precision"].append(sum(rl) / k)
            ideal = _dcg(sorted(rels, reverse=True)[:k])
            per_k[k]["ndcg"].append(_dcg(rl) / ideal if ideal > 0 else 0.0)
    if not n:
        return {"n": 0}
    out: dict[str, Any] = {
        "n": n,
        "mrr": st.mean(rr),
        "map": st.mean(ap),
        "full_recall_rate": st.mean(full),
    }
    for k in ks:
        out[f"recall@{k}"] = st.mean(per_k[k]["recall"])
        out[f"precision@{k}"] = st.mean(per_k[k]["precision"])
        out[f"ndcg@{k}"] = st.mean(per_k[k]["ndcg"])
    return out

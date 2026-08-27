"""Core-selection metrics for the multi-round decider (V3).

WHY THIS CANNOT READ THE RETURNED EPISODE LIST. Assembly is core-first, then one
guaranteed non-core per sub-query, then a max-RRF fill, all capped at ``top_k`` -- so a
``top_k=20`` request comes back padded with episodes the decider never picked. Scoring
that list and calling it core-only silently measures the top-20 arm instead: it produced
core sizes of 17.9-20.0 and a precision of ~0.10 where the real core averages 1.4-8.7
items.

Core membership survives in exactly one place: the trace's ``injected`` records, which
flag ``is_core`` per episode id. Hence ``EVEROS_LLMMR_TRACE_DUMP`` is mandatory for
these metrics, not optional.

Reference values on LongMemEval's held-out 100 (session-level gold, 27B decider):
core P 0.959 / R 0.938 / F1 0.933, 1.84 items, 89% full recall.
"""

from __future__ import annotations

import json
import os
import statistics as st
from typing import Any


def core_sessions_from_trace(trace_path: str) -> dict[str, set[str]]:
    """``owner_id -> {session_id}`` for the episodes the decider actually pinned.

    Later records for the same owner overwrite earlier ones: the final injection is the
    state that was scored.
    """
    out: dict[str, set[str]] = {}
    if not trace_path or not os.path.exists(trace_path):
        return out
    with open(trace_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if "injected" not in rec:
                continue
            owner = str(rec.get("owner_id") or "")
            if not owner:
                continue
            out[owner] = {
                str(x.get("session_id"))
                for x in rec["injected"]
                if x.get("is_core") and x.get("session_id")
            }
    return out


def decider_reliability(trace_path: str) -> dict[str, Any]:
    """Rounds, fallbacks and per-round core sizes.

    A fallback means the decider's reply could not be parsed and the loop fell back to
    score order. Reporting it beside the metrics is load-bearing: a 62.8% fallback rate
    once looked like a mediocre policy rather than a truncated ``max_tokens``.
    """
    rounds = fallbacks = 0
    sizes: list[int] = []
    if not trace_path or not os.path.exists(trace_path):
        return {"rounds": 0, "fallbacks": 0, "fallback_rate": 0.0, "core_per_round": {}}
    with open(trace_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if "core_indices" not in rec:
                continue
            rounds += 1
            fallbacks += bool(rec.get("decider_failed"))
            sizes.append(len(rec.get("core_indices") or []))
    return {
        "rounds": rounds,
        "fallbacks": fallbacks,
        "fallback_rate": (fallbacks / rounds) if rounds else 0.0,
        "core_per_round": {
            "mean": st.mean(sizes) if sizes else 0.0,
            "median": st.median(sizes) if sizes else 0.0,
            "max": max(sizes) if sizes else 0,
        },
    }


def score(core: dict[str, set[str]], gold: dict[str, set[str]]) -> dict[str, Any]:
    """Core precision / recall / F1 against session-level gold.

    Owners with empty gold are skipped rather than scored 0 -- a benchmark question with
    no resolvable evidence says nothing about the policy. The count is reported so a bad
    gold mapping cannot hide as a low score.
    """
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    sizes: list[int] = []
    skipped = 0
    for owner, pool in core.items():
        g = gold.get(owner) or set()
        if not g:
            skipped += 1
            continue
        tp = len(pool & g)
        p = tp / len(pool) if pool else 0.0
        r = tp / len(g)
        precisions.append(p)
        recalls.append(r)
        f1s.append(2 * p * r / (p + r) if (p + r) else 0.0)
        sizes.append(len(pool))
    n = len(precisions)
    if not n:
        return {"n": 0, "skipped_no_gold": skipped}
    return {
        "n": n,
        "skipped_no_gold": skipped,
        "precision": st.mean(precisions),
        "recall": st.mean(recalls),
        "f1": st.mean(f1s),
        "core_size_mean": st.mean(sizes),
        "core_size_median": st.median(sizes),
        "full_recall_rate": sum(1 for x in recalls if x == 1.0) / n,
        "clean_rate": sum(1 for x in precisions if x == 1.0) / n,
    }

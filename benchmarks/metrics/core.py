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
from collections.abc import Hashable, Mapping
from typing import Any

QuestionKey = tuple[str, str]
"""``(owner_id, question)`` -- the grain a core-selection metric is defined on."""


def core_sessions_from_trace(trace_path: str) -> dict[QuestionKey, set[str]]:
    """``(owner_id, question) -> {session_id}`` for the episodes the decider pinned.

    Keyed per question, not per owner. Owner alone holds only for a dataset that asks
    one question per owner: LongMemEval's owners are ``longmemeval_<question id>``, so
    the two grains coincide there and the dict-overwrite that used to key this looked
    right. LoCoMo puts every question of a conversation under one owner, where the
    same overwrite silently kept the last question's core and dropped the rest -- and
    which one survived depended on the order the trace happened to be written in.

    ``question_id`` is the natural key but the search layer cannot fill it: a
    ``/search`` request carries a query and an owner, so the trace writes ``None``
    there and the question text is the stable identifier available. Later records for
    one question still overwrite earlier ones, which is intended: the final injection
    is the state that was scored.
    """
    out: dict[QuestionKey, set[str]] = {}
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
            question = str(rec.get("question_id") or rec.get("question") or "")
            out[owner, question] = {
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


def score(
    core: Mapping[Hashable, set[str]], gold: Mapping[Hashable, set[str]]
) -> dict[str, Any]:
    """Core precision / recall / F1 against session-level gold.

    Both mappings must be keyed the same way -- ``core_sessions_from_trace`` returns
    ``(owner_id, question)``, so ``gold`` has to as well. A mismatch shows up as every
    entry counted in ``skipped_no_gold`` with ``n == 0``, which is loud; the shape it
    replaced was silent.

    Entries with empty gold are skipped rather than scored 0 -- a benchmark question
    with no resolvable evidence says nothing about the policy. The count is reported so
    a bad gold mapping cannot hide as a low score.
    """
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    sizes: list[int] = []
    skipped = 0
    for key, pool in core.items():
        g = gold.get(key) or set()
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

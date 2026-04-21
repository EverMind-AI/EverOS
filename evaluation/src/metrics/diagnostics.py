"""
Cross-stage diagnostics aggregator.

Reads per-question retrieval_metadata (from search_results) and per-question
metadata (from answer_results), plus optional per-conversation add_summary
files, and emits averages / distributions.

Every aggregation ignores None values so adapters that don't emit the field
(e.g. mem0/memos on latency) don't poison the mean.

SCOPE CAVEAT (pre-Phase-1): the ``*_latency_ms`` fields currently come
from adapter-reported metadata, so their measurement boundary differs
between adapters (EverMemOS includes on-disk index load, OpenClaw only
covers the backend RPC, etc.). They should be read as *within-adapter*
trend signals, not cross-adapter comparisons. Phase 1 of the
latency-alignment plan (docs/latency-alignment.md) replaces this with
harness-owned measurement for a canonical, apples-to-apples scope.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Iterable, Optional


def _safe_mean(values: Iterable[float | None]) -> Optional[float]:
    valid = [v for v in values if isinstance(v, (int, float))]
    if not valid:
        return None
    return float(mean(valid))


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Linear-interp percentile over a pre-sorted list. pct in [0, 100]."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    k = (len(sorted_values) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = k - lo
    return float(sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac)


def _stats(values: Iterable[float | None]) -> Optional[dict]:
    """Return mean/p50/p95/max/n over numeric values, or None if empty.

    bool is a subclass of int in Python; exclude it explicitly so a
    field accidentally set to True/False cannot poison the distribution
    with 1.0/0.0 datapoints.
    """
    valid = [
        float(v)
        for v in values
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    ]
    if not valid:
        return None
    valid.sort()
    return {
        "n": len(valid),
        "mean": float(mean(valid)),
        "p50": _percentile(valid, 50),
        "p95": _percentile(valid, 95),
        "max": float(valid[-1]),
    }


def _distribution(values: Iterable[str | None]) -> dict[str, int]:
    return dict(Counter(v for v in values if isinstance(v, str) and v))


def _iter_add_summary_paths(index: Optional[dict]) -> list[Path]:
    """Find per-conversation add_summary.json files for supported adapters."""
    if not index:
        return []

    kind = index.get("type")
    paths: list[Path] = []

    if kind == "openclaw_sandboxes":
        for sandbox in (index.get("conversations") or {}).values():
            metrics_dir = sandbox.get("metrics_dir")
            if metrics_dir:
                paths.append(Path(metrics_dir) / "add_summary.json")
        return paths

    if kind == "lazy_load":
        # EverMemOS: metrics/<conv_index>/add_summary.json under output dir
        metrics_root = index.get("metrics_dir")
        conv_ids = index.get("conversation_ids") or []
        if not metrics_root:
            return []
        root = Path(metrics_root)
        for conv_id in conv_ids:
            # Strip speaker-prefix / non-numeric suffix the same way adapters do
            key = str(conv_id).rsplit("_", 1)[-1] if "_" in str(conv_id) else str(conv_id)
            paths.append(root / key / "add_summary.json")
        return paths

    return []


def aggregate_diagnostics(
    search_results, answer_results_metadata, index: Optional[dict] = None
) -> dict:
    """Pull time-series across stages into a single summary dict."""
    retrieval_latencies = [
        sr.retrieval_metadata.get("retrieval_latency_ms") for sr in search_results
    ]
    scheduler_waits = [
        sr.retrieval_metadata.get("scheduler_wait_ms") for sr in search_results
    ]
    routes = [sr.retrieval_metadata.get("retrieval_route") for sr in search_results]
    backends = [sr.retrieval_metadata.get("backend_mode") for sr in search_results]

    empty_hits = sum(1 for sr in search_results if not sr.results)
    empty_rate = empty_hits / len(search_results) if search_results else 0.0

    answer_latencies = [m.get("answer_latency_ms") for m in answer_results_metadata]
    context_tokens = [m.get("final_context_tokens") for m in answer_results_metadata]
    context_chars = [m.get("final_context_chars") for m in answer_results_metadata]

    # Per-conversation add telemetry (optional; adapters opt-in by writing
    # add_summary.json under their metrics_dir).
    add_latencies: list[float] = []
    flush_count = 0
    index_settle_total = 0.0
    retry_events = 0
    fallback_events = 0
    failed_adds = 0
    add_n = 0
    for summary_path in _iter_add_summary_paths(index):
        if not summary_path.exists():
            continue
        try:
            summary = json.loads(summary_path.read_text())
        except Exception:
            continue
        add_n += 1
        lat = summary.get("add_latency_ms")
        if isinstance(lat, (int, float)):
            add_latencies.append(lat)
        flush_count += int(summary.get("flush_triggered_count", 0))
        settle = summary.get("index_settle_latency_ms") or 0
        if isinstance(settle, (int, float)):
            index_settle_total += settle
        if summary.get("flush_retry_count", 0) and int(summary["flush_retry_count"]) > 0:
            retry_events += 1
        if summary.get("flush_fallback", False):
            fallback_events += 1
        if summary.get("failed", False):
            failed_adds += 1

    retry_rate = (retry_events / add_n) if add_n else None
    fallback_rate = (fallback_events / add_n) if add_n else None
    failed_rate = (failed_adds / add_n) if add_n else None

    return {
        # Legacy scalar fields kept for backward-compat with existing
        # benchmark_summary.py / report.txt consumers.
        "add_latency_ms_mean": _safe_mean(add_latencies),
        "retrieval_latency_ms_mean": _safe_mean(retrieval_latencies),
        "scheduler_wait_ms_mean": _safe_mean(scheduler_waits),
        "answer_latency_ms_mean": _safe_mean(answer_latencies),
        "empty_retrieval_rate": empty_rate,
        "final_context_tokens_mean": _safe_mean(context_tokens),
        "final_context_chars_mean": _safe_mean(context_chars),
        "retrieval_route_distribution": _distribution(routes),
        "backend_mode_distribution": _distribution(backends),
        "flush_triggered_count_total": flush_count,
        "index_settle_latency_ms_total": index_settle_total,
        # New distribution stats. Consumers that want p50/p95 should read
        # these; consumers that only need mean can keep using *_mean above.
        "add_latency_ms_stats": _stats(add_latencies),
        "retrieval_latency_ms_stats": _stats(retrieval_latencies),
        "answer_latency_ms_stats": _stats(answer_latencies),
        "final_context_tokens_stats": _stats(context_tokens),
        # Reliability signals (None when no add telemetry was captured).
        "add_retry_rate": retry_rate,
        "add_fallback_rate": fallback_rate,
        "add_failed_rate": failed_rate,
        "add_samples": add_n,
    }

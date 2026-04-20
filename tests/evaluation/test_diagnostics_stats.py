"""Distribution stats (p50/p95/max) + EverMemOS add_summary aggregation."""
from evaluation.src.core.data_models import SearchResult
from evaluation.src.metrics.diagnostics import aggregate_diagnostics


def test_aggregate_diagnostics_emits_latency_stats():
    """aggregate_diagnostics exposes p50/p95/max distributions alongside
    the legacy *_mean fields, so downstream consumers can pick either."""
    search_results = [
        SearchResult("q", "c0", [{}], {"retrieval_latency_ms": 10.0}),
        SearchResult("q", "c0", [{}], {"retrieval_latency_ms": 30.0}),
        SearchResult("q", "c0", [{}], {"retrieval_latency_ms": 50.0}),
    ]
    answer_meta = [{"answer_latency_ms": 100.0}] * 3
    diag = aggregate_diagnostics(search_results, answer_meta)

    ret = diag["retrieval_latency_ms_stats"]
    assert ret["n"] == 3
    assert ret["mean"] == 30.0
    assert ret["p50"] == 30.0
    assert ret["max"] == 50.0

    # No add samples captured when no index is provided.
    assert diag["add_latency_ms_stats"] is None
    assert diag["add_retry_rate"] is None
    assert diag["add_samples"] == 0


def test_aggregate_diagnostics_reads_evermemos_add_summary(tmp_path):
    """Diagnostics picks up per-conv add_summary.json under a
    ``lazy_load`` index (EverMemOS adapter layout)."""
    import json

    metrics_root = tmp_path / "metrics"
    for key, payload in [
        ("0", {"add_latency_ms": 100.0}),
        (
            "5",
            {
                "add_latency_ms": 300.0,
                "flush_retry_count": 2,
                "flush_fallback": True,
            },
        ),
    ]:
        (metrics_root / key).mkdir(parents=True)
        (metrics_root / key / "add_summary.json").write_text(json.dumps(payload))

    index = {
        "type": "lazy_load",
        "metrics_dir": str(metrics_root),
        "conversation_ids": ["locomo_0", "locomo_5"],
    }
    search_results = [
        SearchResult("q", "c0", [{}], {"retrieval_latency_ms": 10.0}),
    ]
    diag = aggregate_diagnostics(
        search_results, [{"answer_latency_ms": 1.0}], index=index
    )

    assert diag["add_samples"] == 2
    assert diag["add_latency_ms_mean"] == 200.0
    stats = diag["add_latency_ms_stats"]
    assert stats["n"] == 2
    assert stats["max"] == 300.0
    assert diag["add_retry_rate"] == 0.5
    assert diag["add_fallback_rate"] == 0.5
    assert diag["add_failed_rate"] == 0.0

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


def test_aggregate_diagnostics_counts_failed_add_samples(tmp_path):
    """If an adapter writes add_summary.json with failed=true the
    diagnostic must count it in add_failed_rate and still include the
    (partial) latency sample. This guards the failure path of the
    return_exceptions=True branch in evermemos_adapter.add()."""
    import json

    metrics_root = tmp_path / "metrics"
    for key, payload in [
        ("0", {"add_latency_ms": 50.0, "failed": False}),
        ("1", {"add_latency_ms": 120.0, "failed": True, "error": "boom"}),
        ("2", {"add_latency_ms": 60.0, "failed": False}),
    ]:
        (metrics_root / key).mkdir(parents=True)
        (metrics_root / key / "add_summary.json").write_text(json.dumps(payload))

    index = {
        "type": "lazy_load",
        "metrics_dir": str(metrics_root),
        "conversation_ids": ["locomo_0", "locomo_1", "locomo_2"],
    }
    diag = aggregate_diagnostics([], [], index=index)

    # All three samples count, including the failed one.
    assert diag["add_samples"] == 3
    assert diag["add_latency_ms_stats"]["n"] == 3
    # Exactly 1/3 of the samples failed.
    assert abs(diag["add_failed_rate"] - 1 / 3) < 1e-9


def test_stats_ignores_bool_samples():
    """_stats must treat bool as non-numeric even though bool<:int in
    Python; otherwise a mis-typed telemetry field set to True/False
    would silently contribute 1.0/0.0 datapoints to a latency
    distribution."""
    from evaluation.src.metrics.diagnostics import _stats

    assert _stats([True, False, True]) is None
    mixed = _stats([10.0, True, 20.0, False, 30.0])
    assert mixed["n"] == 3
    assert mixed["mean"] == 20.0

"""
Task 6: session-level retrieval metrics + diagnostics + summary.
"""
from evaluation.src.core.data_models import QAPair, SearchResult
from evaluation.src.metrics.retrieval_metrics import (
    evaluate_retrieval_for_question,
    evaluate_retrieval_metrics,
    normalize_gold_sessions,
)
from evaluation.src.metrics.diagnostics import aggregate_diagnostics
from evaluation.src.metrics.benchmark_summary import build_benchmark_summary


def test_normalize_gold_sessions_accepts_both_formats():
    assert normalize_gold_sessions(["D3:11", "D3:12"]) == ["S3"]
    assert normalize_gold_sessions(["S1", "S3"]) == ["S1", "S3"]
    assert normalize_gold_sessions(["D1:0", "S3", "D3:5"]) == ["S1", "S3"]


def test_normalize_gold_sessions_fails_on_unknown_format():
    import pytest

    with pytest.raises(ValueError):
        normalize_gold_sessions(["session_1"])


def test_normalize_gold_sessions_splits_compound_evidence():
    """LoCoMo ships a small number of evidence strings with multiple
    message ids glued together by ``; ``, ``,``, or spaces
    (e.g. 'D8:6; D9:17', 'D21:18 D21:22 D11:15'). Projection must split
    first, not fail-closed on the whole run."""
    assert normalize_gold_sessions(["D8:6; D9:17"]) == ["S8", "S9"]
    assert normalize_gold_sessions(["D1:1,D2:2"]) == ["S1", "S2"]
    assert normalize_gold_sessions(["D21:18 D21:22 D11:15 D11:19"]) == [
        "S11",
        "S21",
    ]
    # mixed with well-formed entries, still deduped and sorted
    assert normalize_gold_sessions(["S3", "D3:1; D3:2", "D1:0"]) == [
        "S1",
        "S3",
    ]


def test_normalize_gold_sessions_recovers_locomo_typos():
    """Two concrete LoCoMo annotator typos exist in locomo10.json:
       * a bare ``'D'`` token mixed in with valid entries (conv3/qa88),
       * ``'D:11:26'`` with a leading colon (conv4/qa18) meaning ``'D11:26'``.
    Both are recoverable without guessing: skip the bare ``D`` and
    tolerate one extra ``:`` right after ``D``. Any other malformed
    token still fails closed."""
    # bare 'D' is dropped; the rest resolve normally
    assert normalize_gold_sessions(["D1:18", "D", "D1:20"]) == ["S1"]
    # 'D:11:26' is interpreted as 'D11:26'
    assert normalize_gold_sessions(["D:11:26"]) == ["S11"]
    # mixed: typo + clean tokens
    assert normalize_gold_sessions(
        ["D1:14", "D:11:26", "D20:21"]
    ) == ["S1", "S11", "S20"]


def test_batch_quarantines_questions_with_corrupt_evidence():
    """When evidence contains a genuinely unrecognised token (not a bare
    'D' typo, not ``D:N:M`` - those are recovered), the question is
    quarantined into ``unresolved_question_ids`` so the rest of the
    run still reports metrics."""
    qas = [
        QAPair(question_id="q_good", question="", answer="",
               evidence=["D0:0"], metadata={"conversation_id": "c0"}),
        QAPair(question_id="q_bad", question="", answer="",
               evidence=["bogus_xyz"], metadata={"conversation_id": "c0"}),
    ]
    srs = [
        SearchResult("", "c0", [{"metadata": {"source_sessions": ["S0"]}}],
                     {"question_id": "q_good"}),
        SearchResult("", "c0", [{"metadata": {"source_sessions": ["S9"]}}],
                     {"question_id": "q_bad"}),
    ]
    metrics = evaluate_retrieval_metrics(qas, srs, k=1)
    assert "q_bad" in metrics["unresolved_question_ids"]
    assert [p["question_id"] for p in metrics["per_question"]] == ["q_good"]
    # mean should be taken over the 1 successful question, not diluted
    assert metrics["evidence_hit_at_k_mean"] == 1.0


def test_retrieval_metrics_use_session_level_projection():
    qa = QAPair(
        question_id="q1",
        question="",
        answer="",
        evidence=["D3:11", "D3:12"],
        metadata={"conversation_id": "c0"},
    )
    sr = SearchResult(
        query="",
        conversation_id="c0",
        results=[
            {"metadata": {"source_sessions": ["S1"]}},
            {"metadata": {"source_sessions": ["S3"]}},
        ],
        retrieval_metadata={},
    )
    metrics = evaluate_retrieval_for_question(qa, sr, k=2)
    assert metrics["evidence_hit_at_k"] == 1.0
    assert metrics["mrr"] == 0.5
    assert metrics["evidence_recall_at_k"] == 1.0  # 1/1 gold session covered


def test_ndcg_at_k_nontrivial_when_not_all_gold_covered():
    """P2-1: ndcg must penalize late discovery + missing gold.

    Gold = {S0, S1, S2}; at k=3 we retrieve S0 (rank 1, new), S3 (rank 2,
    unrelated), S1 (rank 3, new). Ideal DCG covers 3 gold at ranks 1-3.

    Actual DCG = 1/log2(2) + 0 + 1/log2(4) = 1.0 + 0.5 = 1.5
    Ideal  DCG = 1/log2(2) + 1/log2(3) + 1/log2(4)
               = 1.0 + 0.6309 + 0.5 = 2.1309
    NDCG@3 = 1.5 / 2.1309 ~= 0.7039
    """
    import math
    qa = QAPair(
        question_id="q_ndcg",
        question="",
        answer="",
        evidence=["D0:0", "D1:0", "D2:0"],
        metadata={"conversation_id": "c0"},
    )
    sr = SearchResult(
        query="",
        conversation_id="c0",
        results=[
            {"metadata": {"source_sessions": ["S0"]}},
            {"metadata": {"source_sessions": ["S3"]}},
            {"metadata": {"source_sessions": ["S1"]}},
        ],
        retrieval_metadata={},
    )
    metrics = evaluate_retrieval_for_question(qa, sr, k=3)

    actual_dcg = 1.0 / math.log2(2) + 0.0 + 1.0 / math.log2(4)
    ideal_dcg = sum(1.0 / math.log2(i + 1) for i in range(1, 4))
    expected = actual_dcg / ideal_dcg

    assert abs(metrics["ndcg_at_k"] - expected) < 1e-9
    assert 0.6 < metrics["ndcg_at_k"] < 0.75  # also a loose sanity window


def test_retrieval_metrics_zero_when_no_match():
    qa = QAPair(
        question_id="q2",
        question="",
        answer="",
        evidence=["D3:11"],
        metadata={"conversation_id": "c0"},
    )
    sr = SearchResult(
        query="",
        conversation_id="c0",
        results=[
            {"metadata": {"source_sessions": ["S1"]}},
            {"metadata": {"source_sessions": ["S2"]}},
        ],
        retrieval_metadata={},
    )
    metrics = evaluate_retrieval_for_question(qa, sr, k=2)
    assert metrics["evidence_hit_at_k"] == 0.0
    assert metrics["mrr"] == 0.0
    assert metrics["evidence_recall_at_k"] == 0.0


def test_batch_pairs_by_question_id_not_position():
    """P1-1: batch wrapper must pair by retrieval_metadata.question_id so
    upstream re-ordering (checkpoint resume, subset filter) can't silently
    misalign which search_result scores which qa."""
    qas = [
        QAPair(question_id="qa-A", question="", answer="",
               evidence=["D0:0"], metadata={"conversation_id": "c0"}),
        QAPair(question_id="qa-B", question="", answer="",
               evidence=["D1:0"], metadata={"conversation_id": "c0"}),
    ]
    # search_results are in REVERSED order; ids carried in retrieval_metadata
    # still let us pair correctly.
    srs = [
        SearchResult("", "c0", [{"metadata": {"source_sessions": ["S1"]}}],
                     {"question_id": "qa-B"}),
        SearchResult("", "c0", [{"metadata": {"source_sessions": ["S0"]}}],
                     {"question_id": "qa-A"}),
    ]
    metrics = evaluate_retrieval_metrics(qas, srs, k=1)
    # qa-A should score hit=1 (S0 matches gold D0:0 -> S0)
    # qa-B should score hit=1 (S1 matches gold D1:0 -> S1)
    assert metrics["per_question"][0]["question_id"] == "qa-A"
    assert metrics["per_question"][0]["evidence_hit_at_k"] == 1.0
    assert metrics["per_question"][1]["question_id"] == "qa-B"
    assert metrics["per_question"][1]["evidence_hit_at_k"] == 1.0


def test_batch_reports_unresolved_when_id_missing():
    qas = [
        QAPair(question_id="qa-A", question="", answer="",
               evidence=["D0:0"], metadata={"conversation_id": "c0"}),
    ]
    srs = [
        # search result for a DIFFERENT question - must not be paired.
        SearchResult("", "c0", [{"metadata": {"source_sessions": ["S9"]}}],
                     {"question_id": "qa-OTHER"}),
    ]
    metrics = evaluate_retrieval_metrics(qas, srs, k=1)
    assert metrics["unresolved_question_ids"] == ["qa-A"]
    assert metrics["per_question"] == []


def test_evaluate_retrieval_metrics_batch_aggregates_mean():
    qas = [
        QAPair(question_id="q1", question="", answer="", evidence=["D0:0"],
               metadata={"conversation_id": "c0"}),
        QAPair(question_id="q2", question="", answer="", evidence=["D1:0"],
               metadata={"conversation_id": "c0"}),
    ]
    srs = [
        SearchResult("", "c0", [{"metadata": {"source_sessions": ["S0"]}}], {}),
        SearchResult("", "c0", [{"metadata": {"source_sessions": ["S9"]}}], {}),
    ]
    metrics = evaluate_retrieval_metrics(qas, srs, k=1)
    assert metrics["per_question"][0]["evidence_hit_at_k"] == 1.0
    assert metrics["per_question"][1]["evidence_hit_at_k"] == 0.0
    assert metrics["evidence_hit_at_k_mean"] == 0.5
    assert metrics["evidence_recall_at_k_mean"] == 0.5


def test_aggregate_diagnostics_tolerates_missing_fields():
    # Emulate a non-openclaw adapter that doesn't populate latency metadata.
    search_results = [
        SearchResult("q", "c0", [{}], {"retrieval_latency_ms": 10.0,
                                         "retrieval_route": "search_only",
                                         "backend_mode": "fts_only"}),
        SearchResult("q", "c0", [], {}),  # empty retrieval
    ]
    answer_results_meta = [
        {"answer_latency_ms": 50.0, "final_context_tokens": 100},
        {"answer_latency_ms": None, "final_context_tokens": 0},
    ]
    diag = aggregate_diagnostics(search_results, answer_results_meta)
    assert diag["retrieval_latency_ms_mean"] == 10.0  # only one valid
    assert diag["answer_latency_ms_mean"] == 50.0
    assert diag["empty_retrieval_rate"] == 0.5
    assert diag["final_context_tokens_mean"] == 50.0
    assert diag["retrieval_route_distribution"] == {"search_only": 1}
    assert diag["backend_mode_distribution"] == {"fts_only": 1}


def test_build_benchmark_summary_shape():
    eval_result = {"accuracy": 0.6}
    retrieval_metrics = {
        "evidence_hit_at_k_mean": 0.5,
        "evidence_recall_at_k_mean": 0.5,
        "mrr_mean": 0.5,
        "ndcg_at_k_mean": 0.5,
        "per_question": [],
    }
    answer_aux_metrics = {
        "f1_mean": 0.4,
        "bleu1_mean": 0.2,
    }
    diagnostics = {
        "add_latency_ms_mean": 1000.0,
        "retrieval_latency_ms_mean": 30.0,
        "answer_latency_ms_mean": 120.0,
        "final_context_tokens_mean": 100.0,
    }
    summary = build_benchmark_summary(
        system="openclaw", dataset="locomo",
        eval_result=eval_result,
        retrieval_metrics=retrieval_metrics,
        answer_aux_metrics=answer_aux_metrics,
        diagnostics=diagnostics,
        k=5,
    )
    assert summary["system"] == "openclaw"
    assert summary["dataset"] == "locomo"
    assert summary["answer_level"]["accuracy"] == 0.6
    assert summary["answer_level"]["f1_mean"] == 0.4
    assert summary["answer_level"]["bleu1_mean"] == 0.2
    # Session-level metrics moved under adapter_specific_retrieval in
    # Phase 6 (content_overlap@k became the canonical cross-adapter
    # metric). retrieval_level now only carries content_overlap fields;
    # evidence_hit / mrr etc. live in the adapter-specific section.
    assert summary["adapter_specific_retrieval"]["evidence_hit_at_5"] == 0.5
    assert summary["adapter_specific_retrieval"]["mrr"] == 0.5
    assert summary["retrieval_level"]["content_overlap_at_5"] is None  # no content_overlap passed
    assert summary["diagnostics"]["retrieval_latency_ms_mean"] == 30.0
    # Summary no longer aliases time_to_visible - field absent entirely.
    assert "time_to_visible_ms_mean" not in summary["diagnostics"]


def test_build_benchmark_summary_preserves_none_for_missing_sources():
    """P2-2: missing upstream data must stay None so reports can
    distinguish 'system scored 0' from 'source didn't run'."""
    summary = build_benchmark_summary(
        system="stub",
        dataset="stub",
        eval_result=None,
        retrieval_metrics=None,
        answer_aux_metrics=None,
        diagnostics=None,
        k=5,
    )
    assert summary["answer_level"]["accuracy"] is None
    assert summary["answer_level"]["f1_mean"] is None
    assert summary["retrieval_level"]["content_overlap_at_5"] is None
    assert summary["adapter_specific_retrieval"]["evidence_hit_at_5"] is None
    assert summary["adapter_specific_retrieval"]["mrr"] is None
    assert summary["diagnostics"]["retrieval_latency_ms_mean"] is None

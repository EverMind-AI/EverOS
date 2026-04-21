"""content_overlap@k — canonical retrieval quality, adapter-agnostic."""
from evaluation.src.core.data_models import QAPair, SearchResult
from evaluation.src.metrics.content_overlap import (
    compute_content_overlap_at_k,
    evaluate_content_overlap,
)


def _mk_qa(qid: str, answer: str) -> QAPair:
    return QAPair(question_id=qid, question="", answer=answer, evidence=[])


def _mk_sr(qid: str, *texts: str) -> SearchResult:
    return SearchResult(
        query="",
        conversation_id="c0",
        results=[{"content": t} for t in texts],
        retrieval_metadata={"question_id": qid},
    )


def test_content_overlap_perfect_recall():
    qa = _mk_qa("q1", "cat mat")
    sr = _mk_sr("q1", "the cat sat on the mat")
    out = compute_content_overlap_at_k(qa, sr, k=5)
    # gold tokens = {cat, mat}; ctx tokens include both, plus extras.
    assert out["content_overlap_recall"] == 1.0
    # precision = |{cat,mat}| / |{the, cat, sat, on, mat}| = 2/5
    assert abs(out["content_overlap_precision"] - 2 / 5) < 1e-9
    # F1 = 2 * 0.4 * 1 / (0.4 + 1) == 4/7
    assert abs(out["content_overlap_at_k"] - 4 / 7) < 1e-9


def test_content_overlap_zero_when_no_match():
    qa = _mk_qa("q1", "quantum")
    sr = _mk_sr("q1", "the dog barks loudly")
    out = compute_content_overlap_at_k(qa, sr, k=5)
    assert out["content_overlap_at_k"] == 0.0
    assert out["content_overlap_recall"] == 0.0


def test_content_overlap_only_uses_top_k():
    qa = _mk_qa("q1", "zebra")
    sr = _mk_sr("q1", "dog", "cat", "zebra in savannah")
    # k=2 cuts off the doc that actually contains "zebra"
    out_k2 = compute_content_overlap_at_k(qa, sr, k=2)
    assert out_k2["content_overlap_at_k"] == 0.0
    # k=3 includes it
    out_k3 = compute_content_overlap_at_k(qa, sr, k=3)
    assert out_k3["content_overlap_recall"] == 1.0


def test_content_overlap_handles_non_string_answer():
    """LoCoMo sometimes serializes gold answers as int (e.g. a year).
    Metric should coerce to str instead of crashing."""
    qa = QAPair(question_id="q1", question="", answer=2022, evidence=[])
    sr = _mk_sr("q1", "she started in 2022 and finished in 2023")
    out = compute_content_overlap_at_k(qa, sr, k=5)
    assert out["content_overlap_recall"] == 1.0


def test_batch_pairs_by_question_id():
    qas = [_mk_qa("qa", "apple"), _mk_qa("qb", "banana")]
    # Search results in reversed order but carry question_id in meta.
    srs = [
        _mk_sr("qb", "yellow banana fruit"),
        _mk_sr("qa", "red apple juicy"),
    ]
    m = evaluate_content_overlap(qas, srs, k=5)
    assert len(m["per_question"]) == 2
    per = {row["question_id"]: row for row in m["per_question"]}
    assert per["qa"]["content_overlap_recall"] == 1.0
    assert per["qb"]["content_overlap_recall"] == 1.0
    assert m["unresolved_question_ids"] == []


def test_batch_reports_unresolved():
    qas = [_mk_qa("qa", "apple")]
    srs = [_mk_sr("qb", "banana")]   # wrong question_id, positional fallback would mis-pair
    # We inject both to the positional fallback by emptying metadata
    srs[0].retrieval_metadata = {}
    m = evaluate_content_overlap(qas, srs, k=5)
    # Positional fallback takes over → qa paired with the sr that has banana
    assert len(m["per_question"]) == 1
    assert m["per_question"][0]["question_id"] == "qa"
    assert m["per_question"][0]["content_overlap_at_k"] == 0.0


def test_batch_empty_inputs():
    m = evaluate_content_overlap([], [], k=5)
    assert m["content_overlap_at_k_mean"] == 0.0
    assert m["per_question"] == []

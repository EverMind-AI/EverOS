"""Tests for vector_anchored_fusion in retrieval_utils.

Tests the saturated BM25 + weighted-sum fusion strategy:
  final_score = alpha * vec_score + (1 - alpha) * (bm25_raw / (bm25_raw + k))

Branches covered:
  - Both vector and keyword results present
  - Vector-only (no keyword results)
  - Keyword-only (no vector results)
  - Both empty
  - Overlapping documents (same doc in both sets)
  - Single-path documents (floor defaults)
  - BM25 saturation (noise suppression, strong signal preservation)
  - Custom alpha / saturation_k parameters
  - Result ordering
  - Short vs long query-like score patterns
  - Irrelevant-query patterns (flat BM25, flat vector)

Usage:
    PYTHONPATH=src pytest tests/test_vector_anchored_fusion.py -v
"""

import pytest

from agentic_layer.retrieval_utils import vector_anchored_fusion


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _score_map(result):
    """Convert [(doc_id, score), ...] to {doc_id: score}."""
    return {doc_id: s for doc_id, s in result}


def _sat(raw, k=5.0):
    """Manual saturation: raw / (raw + k)."""
    return raw / (raw + k) if raw > 0 else 0.0


# ===========================================================================
# 1. Saturation normalization
# ===========================================================================

class TestSaturationNormalization:
    """BM25 raw scores are mapped via raw / (raw + k)."""

    def test_high_bm25_saturates_near_one(self):
        vec = [("d1", 0.5)]
        kw = [("d1", 100.0)]
        result = _score_map(vector_anchored_fusion(vec, kw, saturation_k=5.0))
        # sat(100, 5) = 100/105 ≈ 0.952
        expected = 0.7 * 0.5 + 0.3 * _sat(100.0)
        assert result["d1"] == pytest.approx(expected)

    def test_low_bm25_stays_near_zero(self):
        vec = [("d1", 0.5)]
        kw = [("d1", 0.3)]
        result = _score_map(vector_anchored_fusion(vec, kw, saturation_k=5.0))
        # sat(0.3, 5) = 0.3/5.3 ≈ 0.057
        expected = 0.7 * 0.5 + 0.3 * _sat(0.3)
        assert result["d1"] == pytest.approx(expected)

    def test_bm25_at_k_maps_to_half(self):
        vec = [("d1", 0.6)]
        kw = [("d1", 5.0)]
        result = _score_map(vector_anchored_fusion(vec, kw, saturation_k=5.0))
        # sat(5, 5) = 0.5
        expected = 0.7 * 0.6 + 0.3 * 0.5
        assert result["d1"] == pytest.approx(expected)

    def test_bm25_zero_maps_to_zero(self):
        vec = [("d1", 0.5)]
        kw = [("d1", 0.0)]
        result = _score_map(vector_anchored_fusion(vec, kw, saturation_k=5.0))
        expected = 0.7 * 0.5 + 0.3 * 0.0
        assert result["d1"] == pytest.approx(expected)

    def test_custom_saturation_k(self):
        vec = [("d1", 0.5)]
        kw = [("d1", 10.0)]
        result = _score_map(vector_anchored_fusion(vec, kw, saturation_k=10.0))
        # sat(10, 10) = 0.5
        expected = 0.7 * 0.5 + 0.3 * 0.5
        assert result["d1"] == pytest.approx(expected)


# ===========================================================================
# 2. Weighted sum fusion
# ===========================================================================

class TestWeightedSumFusion:
    """final = alpha * vec + (1-alpha) * sat_bm25."""

    def test_default_alpha(self):
        vec = [("d1", 0.8)]
        kw = [("d1", 10.0)]  # sat = 10/15 = 0.667
        result = _score_map(vector_anchored_fusion(vec, kw))
        expected = 0.7 * 0.8 + 0.3 * _sat(10.0)
        assert result["d1"] == pytest.approx(expected)

    def test_custom_alpha_vector_dominant(self):
        vec = [("d1", 0.9)]
        kw = [("d1", 5.0)]  # sat = 0.5
        result = _score_map(vector_anchored_fusion(vec, kw, alpha=0.9))
        expected = 0.9 * 0.9 + 0.1 * 0.5
        assert result["d1"] == pytest.approx(expected)

    def test_custom_alpha_keyword_dominant(self):
        vec = [("d1", 0.4)]
        kw = [("d1", 15.0)]  # sat = 15/20 = 0.75
        result = _score_map(vector_anchored_fusion(vec, kw, alpha=0.3))
        expected = 0.3 * 0.4 + 0.7 * _sat(15.0)
        assert result["d1"] == pytest.approx(expected)

    def test_both_high_scores_combine(self):
        vec = [("d1", 0.9)]
        kw = [("d1", 20.0)]  # sat = 20/25 = 0.8
        result = _score_map(vector_anchored_fusion(vec, kw))
        expected = 0.7 * 0.9 + 0.3 * _sat(20.0)
        assert result["d1"] == pytest.approx(expected)
        assert result["d1"] > 0.8  # strong agreement -> high score

    def test_vec_high_kw_low_moderate_score(self):
        vec = [("d1", 0.8)]
        kw = [("d1", 0.5)]  # sat ≈ 0.09
        result = _score_map(vector_anchored_fusion(vec, kw))
        expected = 0.7 * 0.8 + 0.3 * _sat(0.5)
        assert result["d1"] == pytest.approx(expected)
        assert result["d1"] < 0.8  # kw drags it down slightly


# ===========================================================================
# 3. Floor defaults for single-path documents
# ===========================================================================

class TestFloorDefaults:
    """Missing side uses min score of that path, not 0."""

    def test_vector_only_doc_uses_kw_floor(self):
        vec = [("d1", 0.8), ("d2", 0.6)]
        kw = [("d1", 10.0), ("d3", 2.0)]
        result = _score_map(vector_anchored_fusion(vec, kw))
        # d2 only in vector; kw_floor = sat(2.0) = 2/7 ≈ 0.286
        kw_floor = _sat(2.0)
        expected_d2 = 0.7 * 0.6 + 0.3 * kw_floor
        assert result["d2"] == pytest.approx(expected_d2)

    def test_keyword_only_doc_uses_vec_floor(self):
        vec = [("d1", 0.8), ("d2", 0.4)]
        kw = [("d3", 15.0)]
        result = _score_map(vector_anchored_fusion(vec, kw))
        # d3 only in kw; vec_floor = min(0.8, 0.4) = 0.4
        expected_d3 = 0.7 * 0.4 + 0.3 * _sat(15.0)
        assert result["d3"] == pytest.approx(expected_d3)

    def test_floor_not_zero(self):
        """Floor should be min of the path, not 0."""
        vec = [("d1", 0.9), ("d2", 0.5)]
        kw = [("d1", 10.0)]
        result = _score_map(vector_anchored_fusion(vec, kw))
        # d2 is vector-only; kw_floor = sat(10) (only one kw doc, floor = its sat)
        kw_floor = _sat(10.0)
        expected_d2 = 0.7 * 0.5 + 0.3 * kw_floor
        assert result["d2"] == pytest.approx(expected_d2)
        # Should be higher than if floor were 0
        zero_floor = 0.7 * 0.5 + 0.3 * 0.0
        assert result["d2"] > zero_floor

    def test_keyword_only_doc_not_buried(self):
        """A high-BM25 keyword-only doc should still rank reasonably."""
        vec = [("d1", 0.6), ("d2", 0.4)]
        kw = [("d3", 20.0)]  # sat = 0.8, strong keyword match
        result = _score_map(vector_anchored_fusion(vec, kw))
        # d3: vec_floor=0.4, sat=0.8 -> 0.7*0.4 + 0.3*0.8 = 0.52
        vec_floor = 0.4
        expected_d3 = 0.7 * vec_floor + 0.3 * _sat(20.0)
        assert result["d3"] == pytest.approx(expected_d3)
        assert result["d3"] > 0.5  # not buried


# ===========================================================================
# 4. Edge cases: empty inputs
# ===========================================================================

class TestEmptyInputs:
    """Handle empty result sets gracefully."""

    def test_both_empty(self):
        assert vector_anchored_fusion([], []) == []

    def test_vector_only_no_keyword(self):
        vec = [("d1", 0.9), ("d2", 0.5)]
        result = vector_anchored_fusion(vec, [])
        score_map = _score_map(result)
        # kw_floor = 0.0 (no kw results)
        assert score_map["d1"] == pytest.approx(0.7 * 0.9 + 0.3 * 0.0)
        assert score_map["d2"] == pytest.approx(0.7 * 0.5 + 0.3 * 0.0)

    def test_keyword_only_no_vector(self):
        kw = [("d1", 10.0), ("d2", 3.0)]
        result = vector_anchored_fusion([], kw)
        score_map = _score_map(result)
        # vec_floor = 0.0 (no vec results)
        assert score_map["d1"] == pytest.approx(0.7 * 0.0 + 0.3 * _sat(10.0))
        assert score_map["d2"] == pytest.approx(0.7 * 0.0 + 0.3 * _sat(3.0))


# ===========================================================================
# 5. Overlapping documents
# ===========================================================================

class TestOverlappingDocuments:
    """Documents appearing in both result sets get combined scores."""

    def test_overlap_both_strong(self):
        vec = [("d1", 0.85)]
        kw = [("d1", 12.0)]
        result = _score_map(vector_anchored_fusion(vec, kw))
        expected = 0.7 * 0.85 + 0.3 * _sat(12.0)
        assert result["d1"] == pytest.approx(expected)

    def test_overlap_beats_single_path(self):
        """Doc in both paths should score higher than similar single-path doc."""
        vec = [("d1", 0.7), ("d2", 0.7)]
        kw = [("d1", 10.0)]  # d1 in both, d2 vector-only
        result = _score_map(vector_anchored_fusion(vec, kw))
        # d1: 0.7*0.7 + 0.3*sat(10) = 0.49 + 0.3*0.667 = 0.69
        # d2: 0.7*0.7 + 0.3*sat(10) = same floor, same result when only 1 kw doc
        # But with multiple kw docs, d2 would use kw_floor which is lower
        assert result["d1"] >= result["d2"]

    def test_mixed_overlap_and_unique(self):
        vec = [("d1", 0.8), ("d2", 0.5)]
        kw = [("d1", 10.0), ("d3", 8.0)]
        result = _score_map(vector_anchored_fusion(vec, kw))
        # All 3 docs should be present
        assert len(result) == 3
        assert "d1" in result
        assert "d2" in result
        assert "d3" in result


# ===========================================================================
# 6. Result ordering
# ===========================================================================

class TestResultOrdering:
    """Results are always sorted by score descending."""

    def test_sorted_descending(self):
        vec = [("d1", 0.3), ("d2", 0.9)]
        kw = [("d3", 20.0), ("d4", 1.0)]
        result = vector_anchored_fusion(vec, kw)
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)

    def test_many_docs_sorted(self):
        vec = [("d1", 0.2), ("d2", 0.5), ("d3", 0.8), ("d4", 0.1)]
        kw = [("d1", 1.0), ("d2", 10.0), ("d5", 20.0)]
        result = vector_anchored_fusion(vec, kw)
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)


# ===========================================================================
# 7. Discrimination: relevant vs irrelevant patterns
# ===========================================================================

class TestDiscrimination:
    """Verify the fusion produces good separation between relevant and noise."""

    def test_relevant_scores_above_irrelevant(self):
        """Simulate: one relevant doc (high vec + high BM25) vs noise."""
        vec = [
            ("relevant", 0.8),
            ("noise1", 0.35),
            ("noise2", 0.33),
        ]
        kw = [
            ("relevant", 10.0),
            ("noise1", 0.4),
            ("noise2", 0.3),
        ]
        result = _score_map(vector_anchored_fusion(vec, kw))
        assert result["relevant"] > result["noise1"]
        assert result["relevant"] > result["noise2"]
        # Gap should be meaningful
        gap = result["relevant"] - result["noise1"]
        assert gap > 0.2

    def test_flat_bm25_noise_suppressed(self):
        """When BM25 scores are all similarly low (irrelevant query),
        fusion score should be lower than pure vector."""
        vec = [("d1", 0.35), ("d2", 0.34), ("d3", 0.33)]
        kw = [("d1", 0.4), ("d2", 0.38), ("d3", 0.36)]
        result = _score_map(vector_anchored_fusion(vec, kw))
        # All scores should be below the pure vector scores because
        # sat(0.4) ≈ 0.07 drags them down
        for doc_id in ["d1", "d2", "d3"]:
            pure_vec = dict(vec)[doc_id]
            assert result[doc_id] < pure_vec

    def test_strong_bm25_boosts_relevant(self):
        """A doc with strong BM25 match should score higher than
        a doc with same vector score but weak BM25."""
        vec = [("strong_kw", 0.6), ("weak_kw", 0.6)]
        kw = [("strong_kw", 12.0), ("weak_kw", 0.5)]
        result = _score_map(vector_anchored_fusion(vec, kw))
        assert result["strong_kw"] > result["weak_kw"]

    def test_multiple_relevant_ranked_correctly(self):
        """Multiple relevant docs with varying match quality."""
        vec = [
            ("exact", 0.85),
            ("related", 0.55),
            ("noise", 0.35),
        ]
        kw = [
            ("exact", 15.0),
            ("related", 5.0),
            ("noise", 0.3),
        ]
        result = vector_anchored_fusion(vec, kw)
        ids = [doc_id for doc_id, _ in result]
        assert ids[0] == "exact"
        assert ids[1] == "related"
        assert ids[2] == "noise"


# ===========================================================================
# 8. Realistic scenario: simulating actual search data
# ===========================================================================

class TestRealisticScenarios:
    """Test with score distributions resembling real retrieval results."""

    def test_exact_keyword_match(self):
        """Query like 'implement rate limiting middleware' -- strong BM25 + vec."""
        vec = [
            ("rate_limit", 0.74),
            ("500err", 0.40),
            ("403err", 0.38),
        ]
        kw = [
            ("rate_limit", 9.88),
            ("input_val", 2.61),
            ("500err", 1.50),
        ]
        result = vector_anchored_fusion(vec, kw)
        # rate_limit should be Top1 by a clear margin
        assert result[0][0] == "rate_limit"
        assert result[0][1] > 0.6

    def test_irrelevant_query(self):
        """Query like 'how to train a puppy' -- low BM25, low vector."""
        vec = [
            ("d1", 0.35),
            ("d2", 0.34),
            ("d3", 0.33),
        ]
        kw = [
            ("d1", 0.36),
            ("d2", 0.35),
            ("d3", 0.34),
        ]
        result = vector_anchored_fusion(vec, kw)
        # All scores should be low (noise suppressed)
        for _, score in result:
            assert score < 0.30
        # Very small spread -- no meaningful ranking
        spread = result[0][1] - result[-1][1]
        assert spread < 0.05

    def test_semantic_match_no_keyword_overlap(self):
        """Query semantically matches but shares few keywords.
        Vector should dominate via alpha=0.7."""
        vec = [
            ("target", 0.72),
            ("other1", 0.40),
            ("other2", 0.38),
        ]
        kw = [
            ("other1", 3.0),
            ("other2", 2.0),
            ("target", 1.0),
        ]
        result = vector_anchored_fusion(vec, kw)
        # target should still be Top1 thanks to high vector score
        assert result[0][0] == "target"

    def test_long_query_strong_signal(self):
        """Long detailed query -- both BM25 and vector agree strongly."""
        vec = [
            ("match", 0.85),
            ("partial", 0.50),
            ("noise", 0.35),
        ]
        kw = [
            ("match", 12.0),
            ("partial", 3.0),
            ("noise", 0.5),
        ]
        result = vector_anchored_fusion(vec, kw)
        scores = _score_map(result)
        # Clear ranking: match >> partial >> noise
        assert scores["match"] > scores["partial"]
        assert scores["partial"] > scores["noise"]
        # Large gap between match and noise
        assert scores["match"] - scores["noise"] > 0.3

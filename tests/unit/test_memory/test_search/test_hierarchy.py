"""Unit tests for ``memory.search.hierarchy``.

White-box surfaces accessed:
    - ``_hierarchy_eviction_pass`` (internal, tested directly for unit coverage)
    - ``hierarchy_retrieve_episodes`` (public function, tested with stubbed I/O)

Layer 4 uses hierarchical fact eviction: parent episode and its facts are
calibrated to an LR probability via ``cosine_to_lr_score`` and compete on that
single scale, so the expected scores below are computed with the same helper
rather than hard-coded — the assertions track the calibration, not magic
numbers.

All I/O (fact_recaller, episode_recaller) is injected via AsyncMock stubs.
No LanceDB or network calls are made.
"""

from __future__ import annotations

import datetime as _dt
from unittest.mock import AsyncMock, MagicMock

import pytest
from everalgo.rank.fusion import cosine_to_lr_score
from everalgo.types import Candidate, FactCandidate

from everos.memory.search.hierarchy import (
    _build_ep_to_fact_parents,
    _hierarchy_eviction_pass,
    hierarchy_retrieve_episodes,
)

# ── Fixtures / helpers ───────────────────────────────────────────────────


def _ts() -> _dt.datetime:
    return _dt.datetime(2026, 1, 1, tzinfo=_dt.UTC)


def _episode_candidate(
    *,
    ep_id: str = "ep-1",
    score: float = 0.7,
    memcell_id: str = "mc-1",
    entry_id: str | None = None,
) -> Candidate:
    metadata = {
        "parent_id": memcell_id,
        "owner_id": "u1",
        "owner_type": "user",
        "session_id": "sess-1",
        "timestamp": _ts(),
        "episode": "Some episode text.",
        "sender_ids": ["u1"],
        "subject": "Test subject",
        "summary": "Test summary",
    }
    if entry_id is not None:
        metadata["entry_id"] = entry_id
    return Candidate(
        id=ep_id,
        score=score,
        source="vector",
        metadata=metadata,
    )


def _fact_candidate(
    *,
    fact_id: str = "fact-1",
    parent_episode_id: str = "ep-1",
    score: float = 0.9,
) -> FactCandidate:
    return FactCandidate(
        id=fact_id,
        parent_episode_id=parent_episode_id,
        score=score,
        metadata={"fact": "Some fact text."},
    )


def _make_recallers(
    *,
    dense_facts: list[Candidate] | None = None,
    fetched_episodes: list[Candidate] | None = None,
    facts_for_episodes: dict[str, list[FactCandidate]] | None = None,
) -> tuple[MagicMock, MagicMock]:
    """Build stubbed fact_recaller and episode_recaller."""
    fact_recaller = MagicMock()
    fact_recaller.dense_recall = AsyncMock(return_value=dense_facts or [])
    fact_recaller.facts_for_episodes = AsyncMock(return_value=facts_for_episodes or {})

    episode_recaller = MagicMock()
    episode_recaller.fetch_by_entry_ids = AsyncMock(return_value=fetched_episodes or [])

    return fact_recaller, episode_recaller


# ── _hierarchy_eviction_pass unit tests ─────────────────────────────────


class TestHierarchyEvictionPass:
    def test_fact_wins_emits_atomic_fact_at_lr_score(self) -> None:
        # Fact cosine (0.9) > parent cosine (0.5) → fact wins; alpha=1.0 so the
        # emitted score is the fact's own LR-calibrated value.
        episode = _episode_candidate(ep_id="ep-1")
        fact = _fact_candidate(fact_id="fact-1", parent_episode_id="ep-1", score=0.9)

        result = _hierarchy_eviction_pass(
            [episode],
            {"ep-1": [fact]},
            ep_cosine={"ep-1": 0.5},
            ep_bm25={},
        )

        assert len(result) == 1
        item = result[0]
        assert item.item_type == "atomic_fact"
        assert item.id == "fact-1"
        assert item.score == pytest.approx(cosine_to_lr_score(0.9, 0.0))

    def test_episode_wins_emits_episode_at_parent_lr_score(self) -> None:
        # Fact cosine (0.6) < parent cosine (0.8) → episode wins at parent_lr.
        episode = _episode_candidate(ep_id="ep-1")
        fact = _fact_candidate(fact_id="fact-1", parent_episode_id="ep-1", score=0.6)

        result = _hierarchy_eviction_pass(
            [episode],
            {"ep-1": [fact]},
            ep_cosine={"ep-1": 0.8},
            ep_bm25={},
        )

        assert len(result) == 1
        item = result[0]
        assert item.item_type == "episode"
        assert item.id == "ep-1"
        assert item.score == pytest.approx(cosine_to_lr_score(0.8, 0.0))

    def test_no_facts_emits_episode_at_parent_lr(self) -> None:
        episode = _episode_candidate(ep_id="ep-1")

        result = _hierarchy_eviction_pass(
            [episode],
            {},
            ep_cosine={"ep-1": 0.7},
            ep_bm25={},
        )

        assert len(result) == 1
        assert result[0].item_type == "episode"
        assert result[0].id == "ep-1"
        assert result[0].score == pytest.approx(cosine_to_lr_score(0.7, 0.0))

    def test_ordering_preserved_matches_input_order(self) -> None:
        ep_a = _episode_candidate(ep_id="ep-a", memcell_id="mc-a")
        ep_b = _episode_candidate(ep_id="ep-b", memcell_id="mc-b")
        ep_c = _episode_candidate(ep_id="ep-c", memcell_id="mc-c")
        merged = [ep_a, ep_b, ep_c]

        result = _hierarchy_eviction_pass(
            merged,
            {},
            ep_cosine={"ep-a": 0.9, "ep-b": 0.8, "ep-c": 0.7},
            ep_bm25={},
        )

        assert [r.id for r in result] == ["ep-a", "ep-b", "ep-c"]

    def test_parent_episode_id_set_on_evicted_fact(self) -> None:
        episode = _episode_candidate(ep_id="ep-1")
        fact = _fact_candidate(fact_id="fact-1", parent_episode_id="ep-1", score=0.8)

        result = _hierarchy_eviction_pass(
            [episode],
            {"ep-1": [fact]},
            ep_cosine={"ep-1": 0.4},
            ep_bm25={},
        )

        assert result[0].parent_episode_id == "ep-1"

    def test_episode_wins_parent_episode_id_is_none(self) -> None:
        episode = _episode_candidate(ep_id="ep-1")
        fact = _fact_candidate(fact_id="fact-1", parent_episode_id="ep-1", score=0.5)

        result = _hierarchy_eviction_pass(
            [episode],
            {"ep-1": [fact]},
            ep_cosine={"ep-1": 0.9},
            ep_bm25={},
        )

        assert result[0].parent_episode_id is None

    def test_multiple_episodes_mixed_eviction(self) -> None:
        ep1 = _episode_candidate(ep_id="ep-1", memcell_id="mc-1")
        ep2 = _episode_candidate(ep_id="ep-2", memcell_id="mc-2")
        ep3 = _episode_candidate(ep_id="ep-3", memcell_id="mc-3")
        fact1 = _fact_candidate(fact_id="fact-1", parent_episode_id="ep-1", score=0.9)
        fact2 = _fact_candidate(fact_id="fact-2", parent_episode_id="ep-2", score=0.4)

        result = _hierarchy_eviction_pass(
            [ep1, ep2, ep3],
            {"ep-1": [fact1], "ep-2": [fact2]},
            ep_cosine={"ep-1": 0.5, "ep-2": 0.8, "ep-3": 0.6},
            ep_bm25={},
        )

        assert len(result) == 3
        assert result[0].item_type == "atomic_fact"  # 0.9 > 0.5
        assert result[0].id == "fact-1"
        assert result[1].item_type == "episode"  # 0.4 < 0.8
        assert result[1].id == "ep-2"
        assert result[2].item_type == "episode"  # no fact
        assert result[2].id == "ep-3"

    def test_best_fact_across_window_used_for_comparison(self) -> None:
        episode = _episode_candidate(ep_id="ep-1")
        best_fact = _fact_candidate(
            fact_id="fact-best", parent_episode_id="ep-1", score=0.85
        )
        weak_fact = _fact_candidate(
            fact_id="fact-weak", parent_episode_id="ep-1", score=0.3
        )

        result = _hierarchy_eviction_pass(
            [episode],
            {"ep-1": [best_fact, weak_fact]},
            ep_cosine={"ep-1": 0.7},
            ep_bm25={},
        )

        assert result[0].item_type == "atomic_fact"
        assert result[0].id == "fact-best"

    def test_fact_equal_to_parent_does_not_evict(self) -> None:
        # Blend must strictly beat parent_lr; equal scores keep the episode.
        episode = _episode_candidate(ep_id="ep-1")
        fact = _fact_candidate(fact_id="fact-1", parent_episode_id="ep-1", score=0.7)

        result = _hierarchy_eviction_pass(
            [episode],
            {"ep-1": [fact]},
            ep_cosine={"ep-1": 0.7},
            ep_bm25={},
        )

        assert result[0].item_type == "episode"

    def test_alpha_blends_parent_and_child(self) -> None:
        # alpha=0.5 → score = 0.5*child_lr + 0.5*parent_lr.
        episode = _episode_candidate(ep_id="ep-1")
        fact = _fact_candidate(fact_id="fact-1", parent_episode_id="ep-1", score=0.9)

        result = _hierarchy_eviction_pass(
            [episode],
            {"ep-1": [fact]},
            ep_cosine={"ep-1": 0.5},
            ep_bm25={},
            alpha=0.5,
        )

        parent_lr = cosine_to_lr_score(0.5, 0.0)
        child_lr = cosine_to_lr_score(0.9, 0.0)
        expected = 0.5 * child_lr + 0.5 * parent_lr
        assert result[0].item_type == "atomic_fact"
        assert result[0].score == pytest.approx(expected)

    def test_bm25_raises_calibrated_score(self) -> None:
        # BM25 is folded into both parent and child calibration (children
        # inherit parent_bm25), so it does not change the intra-episode
        # parent-vs-fact outcome at alpha=1 — it lifts the absolute LR score.
        episode = _episode_candidate(ep_id="ep-1")

        without_bm25 = _hierarchy_eviction_pass(
            [episode], {}, ep_cosine={"ep-1": 0.5}, ep_bm25={}
        )
        with_bm25 = _hierarchy_eviction_pass(
            [episode], {}, ep_cosine={"ep-1": 0.5}, ep_bm25={"ep-1": 50.0}
        )

        assert with_bm25[0].score > without_bm25[0].score
        assert with_bm25[0].score == pytest.approx(cosine_to_lr_score(0.5, 50.0))

    def test_facts_per_episode_window_caps_competition(self) -> None:
        # A high-scoring fact beyond the window must not win.
        episode = _episode_candidate(ep_id="ep-1")
        in_window = _fact_candidate(fact_id="in", parent_episode_id="ep-1", score=0.55)
        out_window = _fact_candidate(
            fact_id="out", parent_episode_id="ep-1", score=0.99
        )

        result = _hierarchy_eviction_pass(
            [episode],
            {"ep-1": [in_window, out_window]},
            ep_cosine={"ep-1": 0.6},
            ep_bm25={},
            facts_per_episode=1,
        )

        # Only ``in`` (0.55) is in the 1-fact window and it loses to 0.6 →
        # episode wins; the 0.99 ``out`` fact is never considered.
        assert result[0].item_type == "episode"


# ── _build_ep_to_fact_parents unit tests ────────────────────────────────


class TestBuildEpToFactParents:
    """Unit tests for the dual parent_id mapping builder."""

    def test_entry_id_and_parent_id_both_collected(self) -> None:
        """Post-1.5 episode with entry_id and parent_id both present."""
        ep = _episode_candidate(ep_id="ep-1", memcell_id="mc-1", entry_id="ep_entry_1")

        result = _build_ep_to_fact_parents([ep])

        assert result == {"ep-1": ["ep_entry_1", "mc-1"]}

    def test_memcell_id_only_no_entry_id(self) -> None:
        """Pre-1.5 episode: only parent_id (memcell_id), no entry_id."""
        ep = _episode_candidate(ep_id="ep-1", memcell_id="mc-1")

        result = _build_ep_to_fact_parents([ep])

        assert result == {"ep-1": ["mc-1"]}

    def test_entry_id_equals_parent_id_no_duplicate(self) -> None:
        """When entry_id == parent_id, only one value in the list."""
        ep = _episode_candidate(ep_id="ep-1", memcell_id="same_id", entry_id="same_id")

        result = _build_ep_to_fact_parents([ep])

        assert result == {"ep-1": ["same_id"]}

    def test_missing_parent_id_skipped(self) -> None:
        """Episode with no parent_id and no entry_id is excluded."""
        ep = Candidate(
            id="ep-orphan",
            score=0.5,
            source="vector",
            metadata={"owner_id": "u1"},
        )

        result = _build_ep_to_fact_parents([ep])

        assert result == {}

    def test_empty_string_parent_id_skipped(self) -> None:
        """Empty string parent_id is filtered out."""
        ep = _episode_candidate(ep_id="ep-1", memcell_id="")

        result = _build_ep_to_fact_parents([ep])

        assert result == {}

    def test_empty_string_entry_id_skipped(self) -> None:
        """Empty string entry_id is filtered; parent_id still collected."""
        ep = _episode_candidate(ep_id="ep-1", memcell_id="mc-1", entry_id="")

        result = _build_ep_to_fact_parents([ep])

        assert result == {"ep-1": ["mc-1"]}

    def test_multiple_episodes_independent(self) -> None:
        ep_a = _episode_candidate(
            ep_id="ep-a", memcell_id="mc-a", entry_id="ep_entry_a"
        )
        ep_b = _episode_candidate(ep_id="ep-b", memcell_id="mc-b")

        result = _build_ep_to_fact_parents([ep_a, ep_b])

        assert result == {
            "ep-a": ["ep_entry_a", "mc-a"],
            "ep-b": ["mc-b"],
        }

    def test_empty_list_returns_empty_dict(self) -> None:
        result = _build_ep_to_fact_parents([])
        assert result == {}


# ── hierarchy_retrieve_episodes integration-style unit tests ─────────────


class TestHierarchyRetrieveEpisodes:
    """Integration-style unit tests with fully stubbed I/O.

    amaxsim_retrieve and rrf are exercised with real implementations but
    all LanceDB / network calls are replaced by AsyncMock.
    """

    async def test_empty_sparse_dense_returns_empty_list(self) -> None:
        fact_recaller, episode_recaller = _make_recallers()

        result = await hierarchy_retrieve_episodes(
            query="test query",
            sparse=[],
            dense=[],
            query_vector=[0.1, 0.2, 0.3],
            fact_recaller=fact_recaller,
            episode_recaller=episode_recaller,
            where="owner_id = 'u1'",
            top_k=10,
        )

        assert result == []

    async def test_happy_path_episode_wins_no_nested_facts(self) -> None:
        ep = _episode_candidate(ep_id="ep-1", score=0.8, memcell_id="mc-1")

        fact_recaller, episode_recaller = _make_recallers(
            dense_facts=[],
            fetched_episodes=[],
            facts_for_episodes={},
        )

        result = await hierarchy_retrieve_episodes(
            query="test query",
            sparse=[ep],
            dense=[ep],
            query_vector=[0.1, 0.2, 0.3],
            fact_recaller=fact_recaller,
            episode_recaller=episode_recaller,
            where="owner_id = 'u1'",
            top_k=10,
        )

        assert len(result) == 1
        episode_item = result[0]
        assert episode_item.id == "ep-1"
        assert episode_item.atomic_facts == []
        # Episode-win score is the LR-calibrated parent score (cosine 0.8, bm25 0.8).
        assert episode_item.score == pytest.approx(cosine_to_lr_score(0.8, 0.8))

    async def test_happy_path_fact_evicts_episode_nested_in_result(self) -> None:
        ep = _episode_candidate(ep_id="ep-2", score=0.5, memcell_id="mc-2")
        fact = _fact_candidate(fact_id="fact-2", parent_episode_id="ep-2", score=0.95)

        # No Layer-2 boost (dense_facts empty) → ep_cosine comes from the dense
        # episode recall (0.5); sparse empty → parent_bm25 = 0.0. The Layer-4
        # fact (0.95) calibrates above the parent and evicts it.
        fact_recaller, episode_recaller = _make_recallers(
            dense_facts=[],
            fetched_episodes=[],
            facts_for_episodes={"ep-2": [fact]},
        )

        result = await hierarchy_retrieve_episodes(
            query="test query",
            sparse=[],
            dense=[ep],
            query_vector=[0.1, 0.2, 0.3],
            fact_recaller=fact_recaller,
            episode_recaller=episode_recaller,
            where="owner_id = 'u1'",
            top_k=10,
        )

        assert len(result) == 1
        episode_item = result[0]
        assert episode_item.atomic_facts != []
        nested_fact = episode_item.atomic_facts[0]
        assert nested_fact.id == "fact-2"
        # Evicted-fact score is its own LR-calibrated value (alpha default 1.0).
        assert episode_item.score == pytest.approx(cosine_to_lr_score(0.95, 0.0))

    async def test_min_score_filters_below_threshold(self) -> None:
        ep = _episode_candidate(ep_id="ep-1", score=0.5, memcell_id="mc-1")
        fact_recaller, episode_recaller = _make_recallers()

        produced = cosine_to_lr_score(0.5, 0.0)  # episode-win score (sparse empty)

        result = await hierarchy_retrieve_episodes(
            query="test query",
            sparse=[],
            dense=[ep],
            query_vector=[0.1, 0.2, 0.3],
            fact_recaller=fact_recaller,
            episode_recaller=episode_recaller,
            where="owner_id = 'u1'",
            top_k=10,
            min_score=produced + 0.05,
        )

        assert result == []

    async def test_min_score_none_keeps_all(self) -> None:
        ep = _episode_candidate(ep_id="ep-1", score=0.5, memcell_id="mc-1")
        fact_recaller, episode_recaller = _make_recallers()

        result = await hierarchy_retrieve_episodes(
            query="test query",
            sparse=[],
            dense=[ep],
            query_vector=[0.1, 0.2, 0.3],
            fact_recaller=fact_recaller,
            episode_recaller=episode_recaller,
            where="owner_id = 'u1'",
            top_k=10,
            min_score=None,
        )

        assert len(result) == 1
        assert result[0].id == "ep-1"

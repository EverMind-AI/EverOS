"""Unit tests for ``memory.search.hierarchy``.

White-box surfaces accessed:
    - ``_hierarchy_eviction_pass`` (internal, tested directly for unit coverage)
    - ``hierarchy_retrieve_episodes`` (public function, tested with stubbed I/O)

All I/O (fact_recaller, episode_recaller) is injected via AsyncMock stubs.
No LanceDB or network calls are made.
"""

from __future__ import annotations

import datetime as _dt
from unittest.mock import AsyncMock, MagicMock

import pytest
from everalgo.types import Candidate, FactCandidate

from everos.memory.search.hierarchy import (
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
) -> Candidate:
    return Candidate(
        id=ep_id,
        score=score,
        source="vector",
        metadata={
            "parent_id": memcell_id,
            "owner_id": "u1",
            "owner_type": "user",
            "session_id": "sess-1",
            "timestamp": _ts(),
            "episode": "Some episode text.",
            "sender_ids": ["u1"],
            "subject": "Test subject",
            "summary": "Test summary",
        },
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
    episode_recaller.fetch_by_parent_ids = AsyncMock(
        return_value=fetched_episodes or []
    )

    return fact_recaller, episode_recaller


# ── _hierarchy_eviction_pass unit tests ─────────────────────────────────


class TestHierarchyEvictionPass:
    def test_fact_wins_emits_atomic_fact_scored_item(self) -> None:
        episode = _episode_candidate(ep_id="ep-1", score=0.5)
        fact = _fact_candidate(fact_id="fact-1", parent_episode_id="ep-1", score=0.9)

        result = _hierarchy_eviction_pass([episode], {"ep-1": [fact]})

        assert len(result) == 1
        item = result[0]
        assert item.item_type == "atomic_fact"
        assert item.id == "fact-1"
        assert item.score == pytest.approx(0.9)

    def test_episode_wins_emits_episode_scored_item(self) -> None:
        episode = _episode_candidate(ep_id="ep-1", score=0.8)
        fact = _fact_candidate(fact_id="fact-1", parent_episode_id="ep-1", score=0.6)

        result = _hierarchy_eviction_pass([episode], {"ep-1": [fact]})

        assert len(result) == 1
        item = result[0]
        assert item.item_type == "episode"
        assert item.id == "ep-1"
        assert item.score == pytest.approx(0.8)

    def test_no_facts_emits_episode(self) -> None:
        episode = _episode_candidate(ep_id="ep-1", score=0.7)

        result = _hierarchy_eviction_pass([episode], {})

        assert len(result) == 1
        assert result[0].item_type == "episode"
        assert result[0].id == "ep-1"

    def test_ordering_preserved_matches_input_order(self) -> None:
        ep_a = _episode_candidate(ep_id="ep-a", score=0.9, memcell_id="mc-a")
        ep_b = _episode_candidate(ep_id="ep-b", score=0.8, memcell_id="mc-b")
        ep_c = _episode_candidate(ep_id="ep-c", score=0.7, memcell_id="mc-c")
        merged = [ep_a, ep_b, ep_c]

        result = _hierarchy_eviction_pass(merged, {})

        assert [r.id for r in result] == ["ep-a", "ep-b", "ep-c"]

    def test_parent_episode_id_set_on_evicted_fact(self) -> None:
        episode = _episode_candidate(ep_id="ep-1", score=0.4)
        fact = _fact_candidate(fact_id="fact-1", parent_episode_id="ep-1", score=0.8)

        result = _hierarchy_eviction_pass([episode], {"ep-1": [fact]})

        assert result[0].parent_episode_id == "ep-1"

    def test_episode_wins_parent_episode_id_is_none(self) -> None:
        episode = _episode_candidate(ep_id="ep-1", score=0.9)
        fact = _fact_candidate(fact_id="fact-1", parent_episode_id="ep-1", score=0.5)

        result = _hierarchy_eviction_pass([episode], {"ep-1": [fact]})

        assert result[0].parent_episode_id is None

    def test_multiple_episodes_mixed_eviction(self) -> None:
        ep1 = _episode_candidate(ep_id="ep-1", score=0.5, memcell_id="mc-1")
        ep2 = _episode_candidate(ep_id="ep-2", score=0.8, memcell_id="mc-2")
        ep3 = _episode_candidate(ep_id="ep-3", score=0.6, memcell_id="mc-3")
        fact1 = _fact_candidate(fact_id="fact-1", parent_episode_id="ep-1", score=0.9)
        fact2 = _fact_candidate(fact_id="fact-2", parent_episode_id="ep-2", score=0.4)

        result = _hierarchy_eviction_pass(
            [ep1, ep2, ep3],
            {"ep-1": [fact1], "ep-2": [fact2]},
        )

        assert len(result) == 3
        assert result[0].item_type == "atomic_fact"
        assert result[0].id == "fact-1"
        assert result[1].item_type == "episode"
        assert result[1].id == "ep-2"
        assert result[2].item_type == "episode"
        assert result[2].id == "ep-3"

    def test_best_fact_is_first_element_used_for_comparison(self) -> None:
        episode = _episode_candidate(ep_id="ep-1", score=0.7)
        best_fact = _fact_candidate(
            fact_id="fact-best", parent_episode_id="ep-1", score=0.8
        )
        second_fact = _fact_candidate(
            fact_id="fact-second", parent_episode_id="ep-1", score=0.3
        )

        result = _hierarchy_eviction_pass([episode], {"ep-1": [best_fact, second_fact]})

        assert result[0].item_type == "atomic_fact"
        assert result[0].id == "fact-best"

    def test_fact_score_equal_to_episode_score_episode_wins(self) -> None:
        episode = _episode_candidate(ep_id="ep-1", score=0.7)
        fact = _fact_candidate(fact_id="fact-1", parent_episode_id="ep-1", score=0.7)

        result = _hierarchy_eviction_pass([episode], {"ep-1": [fact]})

        assert result[0].item_type == "episode"


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

    async def test_happy_path_fact_evicts_episode_nested_in_result(self) -> None:
        ep = _episode_candidate(ep_id="ep-2", score=0.6, memcell_id="mc-2")
        fact = _fact_candidate(fact_id="fact-2", parent_episode_id="ep-2", score=0.95)

        fact_recaller, episode_recaller = _make_recallers(
            dense_facts=[
                Candidate(
                    id="fact-2",
                    score=0.95,
                    source="vector",
                    metadata={"parent_id": "mc-2"},
                )
            ],
            fetched_episodes=[ep],
            facts_for_episodes={"ep-2": [fact]},
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
        assert episode_item.atomic_facts != []
        nested_fact = episode_item.atomic_facts[0]
        assert nested_fact.id == "fact-2"
        assert nested_fact.score == pytest.approx(0.95)

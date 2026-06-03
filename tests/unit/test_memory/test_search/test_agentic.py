"""Unit tests for ``memory.search.agentic.search_episodes_agentic``.

White-box: patches ``aagentic_retrieve`` to assert benchmark hyperparameters
are wired correctly, plus a shaping test to verify id remapping.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Sequence
from typing import Any, ClassVar
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
from everalgo.clustering import Cluster
from everalgo.rank.protocols import AgenticDecision
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import Candidate

from everos.component.utils.datetime import from_timestamp
from everos.memory.search.agentic import (
    _restore_shaper_metadata,
    _to_everalgo_doc_metadata,
    search_episodes_agentic,
)
from everos.memory.search.dto import SearchEpisodeItem

# ── Stubs ────────────────────────────────────────────────────────────────


def _ts() -> _dt.datetime:
    return _dt.datetime(2026, 1, 1, tzinfo=_dt.UTC)


def _mc_candidate(mc_id: str, ep_id: str, score: float = 0.8) -> Candidate:
    """Candidate keyed by memcell_id (as returned by amaxsim/fetch_all_for_owner)."""
    return Candidate(
        id=mc_id,
        score=score,
        source="vector",
        metadata={
            "episode_id": ep_id,
            "owner_id": "alice",
            "owner_type": "user",
            "session_id": "sess_a",
            "timestamp": _ts(),
            "sender_ids": ["alice"],
            "subject": "Alice eats oat milk",
            "summary": "Alice food preferences",
            "episode": "Alice prefers oat milk in her coffee",
            "parent_id": mc_id,
        },
    )


class _StubEpisodeRecaller:
    kind: ClassVar[str] = "episode"
    everalgo_memory_type: ClassVar[str] = "episodic"
    text_field: ClassVar[str] = "episode"

    def __init__(
        self, all_docs: list[Candidate], by_parent: dict[str, Candidate]
    ) -> None:
        self._all_docs = all_docs
        self._by_parent = by_parent

    async def sparse_recall(self, *_: Any, **__: Any) -> list[Candidate]:
        return []

    async def dense_recall(self, *_: Any, **__: Any) -> list[Candidate]:
        return list(self._all_docs)

    async def fetch_by_parent_ids(
        self, parent_ids: Sequence[str], where: str
    ) -> list[Candidate]:
        """Returns Candidate with id=episode_id (real LanceDB id)."""
        return [self._by_parent[p] for p in parent_ids if p in self._by_parent]

    async def fetch_all_for_owner(self, where: str) -> list[Candidate]:
        """Returns Candidate with id=memcell_id and metadata['episode_id']."""
        return list(self._all_docs)


class _StubFactRecaller:
    kind: ClassVar[str] = "atomic_fact"
    everalgo_memory_type: ClassVar[str] = "episodic"
    text_field: ClassVar[str] = "fact"

    def __init__(self, facts: list[Candidate]) -> None:
        self._facts = facts

    async def sparse_recall(self, *_: Any, **__: Any) -> list[Candidate]:
        return list(self._facts)

    async def dense_recall(self, *_: Any, **__: Any) -> list[Candidate]:
        return list(self._facts)


class _StubReranker:
    async def rerank(
        self, query: str, passages: list[str], *, instruction: str | None = None
    ) -> list[Any]:
        class _R:
            def __init__(self, idx: int) -> None:
                self.index = idx
                self.score = 1.0 - idx * 0.1

        return [_R(i) for i in range(len(passages))]


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture()
def mc_cand() -> Candidate:
    return _mc_candidate("mc_1", "ep_1")


@pytest.fixture()
def ep_recaller(mc_cand: Candidate) -> _StubEpisodeRecaller:
    ep_raw = Candidate(
        id="ep_1",
        score=0.0,
        source="vector",
        metadata=mc_cand.metadata,
    )
    return _StubEpisodeRecaller(
        all_docs=[mc_cand],
        by_parent={"mc_1": ep_raw},
    )


@pytest.fixture()
def fact_cand() -> Candidate:
    return Candidate(
        id="f_1",
        score=0.9,
        source="vector",
        metadata={"parent_id": "mc_1", "fact": "Alice prefers oat milk"},
    )


@pytest.fixture()
def fact_recaller(fact_cand: Candidate) -> _StubFactRecaller:
    return _StubFactRecaller([fact_cand])


@pytest.fixture()
def clusters() -> list[Cluster]:
    # ``cluster_repo.list_for_owner`` is mocked in every test, so cluster
    # contents are never exercised by everalgo; we only need a valid instance
    # that satisfies the everalgo ``Cluster`` schema (ndarray centroid + last_ts).
    return [
        Cluster(
            id="cl_1",
            members=["mc_1"],
            centroid=np.zeros(4, dtype=np.float32),
            last_ts=0,
        )
    ]


# ── Tests ─────────────────────────────────────────────────────────────────


async def test_agentic_search_wires_benchmark_hyperparams(
    ep_recaller: _StubEpisodeRecaller,
    fact_recaller: _StubFactRecaller,
    clusters: list[Cluster],
) -> None:
    """aagentic_retrieve must be called with the exact benchmark hyperparams."""
    captured: dict[str, Any] = {}

    async def fake_aagentic(
        query: str,
        *,
        base_retrieve: Any,
        llm: Any,
        rerank_fn: Any,
        round2_retrieve: Any,
        round2_cap: int,
        top_n: int,
        round1_top_n: int,
        round1_rerank_top_n: int,
        refinement_strategy: str,
        multi_query_count: int,
        rrf_k: int,
    ) -> tuple[list[Candidate], AgenticDecision]:
        captured.update(
            top_n=top_n,
            round1_top_n=round1_top_n,
            round1_rerank_top_n=round1_rerank_top_n,
            round2_cap=round2_cap,
            multi_query_count=multi_query_count,
            rrf_k=rrf_k,
            refinement_strategy=refinement_strategy,
            has_round2=round2_retrieve is not None,
        )
        return [], AgenticDecision(is_multi_round=False)

    async def fake_embed(q: str) -> list[float]:
        return [0.1, 0.2, 0.3, 0.4]

    with (
        patch("everos.memory.search.agentic.aagentic_retrieve", fake_aagentic),
        patch(
            "everos.memory.search.agentic.cluster_repo.list_for_owner",
            AsyncMock(return_value=clusters),
        ),
    ):
        await search_episodes_agentic(
            "What did Alice eat?",
            owner_id="alice",
            where="owner_id = 'alice' AND owner_type = 'user'",
            episode_recaller=ep_recaller,
            atomic_fact_recaller=fact_recaller,
            embed_query_fn=fake_embed,
            reranker=_StubReranker(),
            llm=FakeLLMClient(responses=[]),
            top_k=10,
        )

    assert captured["top_n"] == 10
    assert captured["round1_top_n"] == 50
    assert captured["round1_rerank_top_n"] == 10
    assert captured["round2_cap"] == 40
    assert captured["multi_query_count"] == 3
    assert captured["rrf_k"] == 40
    assert captured["refinement_strategy"] == "multi_query"
    assert captured["has_round2"] is True


async def test_agentic_search_loads_user_memory_clusters(
    ep_recaller: _StubEpisodeRecaller,
    fact_recaller: _StubFactRecaller,
) -> None:
    """cluster_repo.list_for_owner must be called with kind='user_memory'."""
    mock_list = AsyncMock(return_value=[])

    async def fake_embed(q: str) -> list[float]:
        return [0.1] * 4

    with (
        patch(
            "everos.memory.search.agentic.aagentic_retrieve",
            AsyncMock(return_value=([], AgenticDecision(is_multi_round=False))),
        ),
        patch("everos.memory.search.agentic.cluster_repo.list_for_owner", mock_list),
    ):
        await search_episodes_agentic(
            "q",
            owner_id="alice",
            where="owner_id = 'alice' AND owner_type = 'user'",
            episode_recaller=ep_recaller,
            atomic_fact_recaller=fact_recaller,
            embed_query_fn=fake_embed,
            reranker=_StubReranker(),
            llm=FakeLLMClient(responses=[]),
            top_k=10,
        )

    mock_list.assert_called_once_with("alice", "user_memory")


async def test_agentic_search_shapes_candidates_with_episode_id(
    ep_recaller: _StubEpisodeRecaller,
    fact_recaller: _StubFactRecaller,
    clusters: list[Cluster],
    mc_cand: Candidate,
) -> None:
    """SearchEpisodeItem.id must be episode_id (not memcell_id) after retrieve."""

    async def fake_aagentic(
        *_: Any, **__: Any
    ) -> tuple[list[Candidate], AgenticDecision]:
        return [mc_cand], AgenticDecision(is_multi_round=False)

    async def fake_embed(q: str) -> list[float]:
        return [0.1] * 4

    with (
        patch("everos.memory.search.agentic.aagentic_retrieve", fake_aagentic),
        patch(
            "everos.memory.search.agentic.cluster_repo.list_for_owner",
            AsyncMock(return_value=clusters),
        ),
    ):
        result = await search_episodes_agentic(
            "What did Alice eat?",
            owner_id="alice",
            where="owner_id = 'alice' AND owner_type = 'user'",
            episode_recaller=ep_recaller,
            atomic_fact_recaller=fact_recaller,
            embed_query_fn=fake_embed,
            reranker=_StubReranker(),
            llm=FakeLLMClient(responses=[]),
            top_k=10,
        )

    assert len(result) == 1
    assert isinstance(result[0], SearchEpisodeItem)
    assert result[0].id == "ep_1", (
        f"Expected episode_id='ep_1' but got {result[0].id!r}. "
        "Shaper must remap from memcell_id via metadata['episode_id']."
    )


# ── Metadata bridge to the everalgo _format_docs contract ──────────────────


def test_to_everalgo_doc_metadata_injects_text_and_ms_timestamp() -> None:
    """Bridge adds `text` (episode body) + ms-epoch `timestamp` for _format_docs.

    Without this the sufficiency / multi-query LLM prompt falls back to the
    memcell id as the doc body and renders the date as "N/A". ``episode`` is
    left untouched so the reranker / shaper (both expecting a str) keep working.
    """
    original = _ts()
    md = {
        "episode": "Alice prefers oat milk",
        "timestamp": original,
        "subject": "Alice eats oat milk",
    }
    out = _to_everalgo_doc_metadata(md)
    assert out["text"] == "Alice prefers oat milk"
    assert out["episode"] == "Alice prefers oat milk"  # untouched for rerank/shaper
    assert isinstance(out["timestamp"], int)
    assert from_timestamp(out["timestamp"]) == original


def test_restore_shaper_metadata_reverts_ms_timestamp_to_datetime() -> None:
    """The ms-epoch timestamp is reverted to the datetime the shaper requires."""
    original = _ts()
    bridged = _to_everalgo_doc_metadata({"episode": "x", "timestamp": original})
    restored = _restore_shaper_metadata(bridged)
    assert isinstance(restored["timestamp"], _dt.datetime)
    assert restored["timestamp"] == original

"""Unit tests for ``memory.search.agentic.search_episodes_agentic``.

White-box: patches ``aagentic_retrieve`` to assert benchmark hyperparameters
are wired correctly, plus a shaping test to verify id remapping.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from everalgo.clustering import Cluster
from everalgo.rank.protocols import AgenticDecision
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import Candidate

from everos.component.tokenizer import Tokenizer
from everos.memory.mappers import cluster as cluster_mapper
from everos.memory.models import Cluster as DomainCluster
from everos.memory.ports import Scope
from everos.memory.search.agentic import _with_doc_metadata, search_episodes_agentic
from everos.memory.search.dto import SearchEpisodeItem
from everos.memory.search.recall import AtomicFactRecaller, EpisodeRecaller
from everos.memory.search.recall.base import RecallerDeps

from ._records import ScriptedIndex, hit, stored_episode, stored_fact

# ── Stubs ────────────────────────────────────────────────────────────────

_EPISODE = stored_episode("ep_1", parent_id="mc_1")
_FACT = stored_fact("af_1", parent_id="ep_1")


def _deps() -> RecallerDeps:
    tok = MagicMock(spec=Tokenizer)
    tok.tokenize.return_value = ["alice"]
    return RecallerDeps(tokenizer=tok)


def _entry_keyed(cid: str = "ep_1", score: float = 0.8) -> Candidate:
    """A pipeline candidate: episodes are keyed by entry id inside agentic."""
    return Candidate(id=cid, score=score, source="vector", metadata={})


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
def ep_recaller() -> EpisodeRecaller:
    index = ScriptedIndex(dense=[hit(_EPISODE, 0.7)], found=[hit(_EPISODE)])
    return EpisodeRecaller(_deps(), index)  # type: ignore[arg-type]


@pytest.fixture()
def fact_recaller() -> AtomicFactRecaller:
    index = ScriptedIndex(sparse=[hit(_FACT, 2.0)], dense=[hit(_FACT, 0.9)])
    return AtomicFactRecaller(_deps(), index)  # type: ignore[arg-type]


class _StubClusterStore:
    """Serves a fixed cluster set for any owner."""

    def __init__(self, clusters: list[Cluster]) -> None:
        self._clusters = [cluster_mapper.from_algo(c) for c in clusters]

    async def list_for_owner(self, scope: Scope, kind: str) -> list[DomainCluster]:
        return list(self._clusters)


@pytest.fixture()
def clusters() -> list[Cluster]:
    # The cluster store is stubbed in every test, so cluster
    # contents are never exercised by everalgo; we only need a valid instance
    # that satisfies the everalgo ``Cluster`` schema (ndarray centroid + last_ts).
    return [
        Cluster(
            id="cl_1",
            members=["ep_1"],
            centroid=np.zeros(4, dtype=np.float32),
            last_ts=0,
        )
    ]


# ── Tests ─────────────────────────────────────────────────────────────────


async def test_agentic_search_wires_benchmark_hyperparams(
    ep_recaller: EpisodeRecaller,
    fact_recaller: AtomicFactRecaller,
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
    ):
        await search_episodes_agentic(
            "What did Alice eat?",
            owner_id="alice",
            where="owner_id = 'alice' AND owner_type = 'user'",
            app_id="test_app",
            project_id="test_proj",
            episode_recaller=ep_recaller,
            atomic_fact_recaller=fact_recaller,
            cluster_store=_StubClusterStore(clusters),
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
    ep_recaller: EpisodeRecaller,
    fact_recaller: AtomicFactRecaller,
) -> None:
    """The cluster store is asked for the owner's user-memory clusters."""
    mock_list = AsyncMock(return_value=[])

    async def fake_embed(q: str) -> list[float]:
        return [0.1] * 4

    with (
        patch(
            "everos.memory.search.agentic.aagentic_retrieve",
            AsyncMock(return_value=([], AgenticDecision(is_multi_round=False))),
        ),
    ):
        await search_episodes_agentic(
            "q",
            owner_id="alice",
            where="owner_id = 'alice' AND owner_type = 'user'",
            app_id="bench_app",
            project_id="bench_proj",
            episode_recaller=ep_recaller,
            atomic_fact_recaller=fact_recaller,
            cluster_store=MagicMock(list_for_owner=mock_list),
            embed_query_fn=fake_embed,
            reranker=_StubReranker(),
            llm=FakeLLMClient(responses=[]),
            top_k=10,
        )

    mock_list.assert_called_once_with(
        Scope(
            app_id="bench_app",
            project_id="bench_proj",
            owner_id="alice",
            owner_type="user",
        ),
        "user_memory",
    )


async def test_agentic_search_shapes_candidates_with_episode_id(
    ep_recaller: EpisodeRecaller,
    fact_recaller: AtomicFactRecaller,
    clusters: list[Cluster],
) -> None:
    """The DTO id is the episode's global id, not the entry id it was keyed by."""

    async def fake_aagentic(
        query: str, *, base_retrieve: Any, **__: Any
    ) -> tuple[list[Candidate], AgenticDecision]:
        return await base_retrieve(query, 10), AgenticDecision(is_multi_round=False)

    async def fake_embed(q: str) -> list[float]:
        return [0.1] * 4

    with (
        patch("everos.memory.search.agentic.aagentic_retrieve", fake_aagentic),
    ):
        result = await search_episodes_agentic(
            "What did Alice eat?",
            owner_id="alice",
            where="owner_id = 'alice' AND owner_type = 'user'",
            app_id="test_app",
            project_id="test_proj",
            episode_recaller=ep_recaller,
            atomic_fact_recaller=fact_recaller,
            cluster_store=_StubClusterStore(clusters),
            embed_query_fn=fake_embed,
            reranker=_StubReranker(),
            llm=FakeLLMClient(responses=[]),
            top_k=10,
        )

    assert len(result) == 1
    assert isinstance(result[0], SearchEpisodeItem)
    assert result[0].id == _EPISODE.ref.id
    assert result[0].episode == "body ep_1"


# ── Metadata bridge to the everalgo _format_docs contract ──────────────────


def test_doc_metadata_bridge_wraps_the_episode_for_format_docs() -> None:
    """``_format_docs`` expects ``metadata["episode"] = {"subject", "content"}``
    and a ms-epoch ``timestamp``; the flat body is kept as ``text``."""
    cand = Candidate(
        id="ep_1",
        score=0.5,
        metadata={
            "episode": "Alice prefers oat milk",
            "subject": "Oat milk",
            "timestamp": 5,
        },
    )

    out = _with_doc_metadata(cand).metadata

    assert out["text"] == "Alice prefers oat milk"
    assert out["episode"] == {
        "subject": "Oat milk",
        "content": "Alice prefers oat milk",
    }
    assert out["timestamp"] == 5


async def test_agentic_emits_recall_and_rank_spans(
    ep_recaller: EpisodeRecaller,
    fact_recaller: AtomicFactRecaller,
    clusters: list[Cluster],
) -> None:
    """The agentic recall closures (base_retrieve) and the cross-encoder
    rerank_fn emit everos.search.recall / everos.search.rank spans. Driven
    by a fake aagentic_retrieve that actually invokes both callbacks — the
    same way the real everalgo driver does — so the assertion is
    deterministic and independent of everalgo's loop internals."""
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    from everos.config.settings import ObservabilitySettings
    from everos.core.observability.tracing import (
        force_flush,
        init_tracing,
        shutdown_tracing,
    )

    async def exercising_driver(
        query: str, *, base_retrieve: Any, rerank_fn: Any, **_: Any
    ) -> tuple[list[Candidate], AgenticDecision]:
        cands = await base_retrieve(query, 10)  # -> cluster_scoped -> hybrid_full
        # Real driver reranks the recalled hits; feed a non-empty list so the
        # cross-encoder rerank actually runs (empty input short-circuits).
        await rerank_fn(query, cands or [_entry_keyed("ep_r")])
        return [], AgenticDecision(is_multi_round=False)

    async def fake_embed(q: str) -> list[float]:
        return [0.1, 0.2, 0.3, 0.4]

    exporter = InMemorySpanExporter()
    shutdown_tracing()
    init_tracing(
        ObservabilitySettings(enabled=True, endpoint="http://collector.invalid"),
        span_processor=SimpleSpanProcessor(exporter),
    )
    try:
        with (
            patch("everos.memory.search.agentic.aagentic_retrieve", exercising_driver),
        ):
            await search_episodes_agentic(
                "What did Alice eat?",
                owner_id="alice",
                where="owner_id = 'alice'",
                app_id="test_app",
                project_id="test_proj",
                episode_recaller=ep_recaller,
                atomic_fact_recaller=fact_recaller,
                cluster_store=_StubClusterStore(clusters),
                embed_query_fn=fake_embed,
                reranker=_StubReranker(),
                llm=FakeLLMClient(responses=[]),
                top_k=10,
            )
        force_flush()
    finally:
        shutdown_tracing()

    names = {s.name for s in exporter.get_finished_spans()}
    assert "everos.search.recall" in names
    assert "everos.search.rank" in names


async def test_cross_encoder_scores_the_episode_text(
    ep_recaller: EpisodeRecaller,
    fact_recaller: AtomicFactRecaller,
    clusters: list[Cluster],
) -> None:
    """The reranker sees the episode body, not the ``_format_docs`` dict."""
    passages: list[str] = []

    class _SpyReranker:
        async def rerank(
            self, query: str, texts: list[str], *, instruction: str | None = None
        ) -> list[Any]:
            passages.extend(texts)
            return []

    async def driver(
        query: str, *, base_retrieve: Any, rerank_fn: Any, **_: Any
    ) -> tuple[list[Candidate], AgenticDecision]:
        await rerank_fn(query, await base_retrieve(query, 10))
        return [], AgenticDecision(is_multi_round=False)

    async def fake_embed(q: str) -> list[float]:
        return [0.1] * 4

    with patch("everos.memory.search.agentic.aagentic_retrieve", driver):
        await search_episodes_agentic(
            "q",
            owner_id="alice",
            where="owner_id = 'alice'",
            episode_recaller=ep_recaller,
            atomic_fact_recaller=fact_recaller,
            cluster_store=_StubClusterStore(clusters),
            embed_query_fn=fake_embed,
            reranker=_SpyReranker(),
            llm=FakeLLMClient(responses=[]),
            top_k=10,
        )

    assert passages == ["body ep_1"]

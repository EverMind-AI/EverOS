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

    async def sparse_recall_as_child(self, *_: Any, **__: Any) -> list[Candidate]:
        return []

    async def dense_recall_subject_as_child(
        self, *_: Any, **__: Any
    ) -> list[Candidate]:
        return []

    async def fetch_by_parent_ids(
        self, parent_ids: Sequence[str], where: str
    ) -> list[Candidate]:
        """Returns Candidate with id=episode_id (real LanceDB id)."""
        return [self._by_parent[p] for p in parent_ids if p in self._by_parent]

    async def fetch_by_entry_ids(
        self, entry_ids: Sequence[str], where: str
    ) -> list[Candidate]:
        return [self._by_parent[e] for e in entry_ids if e in self._by_parent]

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
            app_id="test_app",
            project_id="test_proj",
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
            app_id="bench_app",
            project_id="bench_proj",
            episode_recaller=ep_recaller,
            atomic_fact_recaller=fact_recaller,
            embed_query_fn=fake_embed,
            reranker=_StubReranker(),
            llm=FakeLLMClient(responses=[]),
            top_k=10,
        )

    mock_list.assert_called_once_with(
        "alice",
        "user_memory",
        app_id="bench_app",
        project_id="bench_proj",
    )


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
            app_id="test_app",
            project_id="test_proj",
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


def test_to_everalgo_doc_metadata_bridges_episode_and_timestamp() -> None:
    """Bridge restructures episode to dict and converts timestamp to ms-epoch.

    ``_format_docs`` expects ``metadata["episode"] = {"subject": ..., "content": ...}``
    and a ms-epoch ``timestamp``. The flat ``episode`` string is also kept as
    ``text`` for the reranker.
    """
    original = _ts()
    md = {
        "episode": "Alice prefers oat milk",
        "timestamp": original,
        "subject": "Alice eats oat milk",
    }
    out = _to_everalgo_doc_metadata(md)
    assert out["text"] == "Alice prefers oat milk"
    assert out["episode"] == {
        "subject": "Alice eats oat milk",
        "content": "Alice prefers oat milk",
    }
    assert isinstance(out["timestamp"], int)
    assert from_timestamp(out["timestamp"]) == original


def test_restore_shaper_metadata_reverts_bridged_fields() -> None:
    """Restore reverts both ms-epoch timestamp and dict episode to shaper format."""
    original = _ts()
    bridged = _to_everalgo_doc_metadata(
        {"episode": "x", "timestamp": original, "subject": "s"}
    )
    restored = _restore_shaper_metadata(bridged)
    assert isinstance(restored["timestamp"], _dt.datetime)
    assert restored["timestamp"] == original
    assert restored["episode"] == "x"


async def test_agentic_emits_recall_and_rank_spans(
    ep_recaller: _StubEpisodeRecaller,
    fact_recaller: _StubFactRecaller,
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
        await rerank_fn(query, cands or [_mc_candidate("mc_r", "ep_r")])
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
            patch(
                "everos.memory.search.agentic.cluster_repo.list_for_owner",
                AsyncMock(return_value=clusters),
            ),
        ):
            await search_episodes_agentic(
                "What did Alice eat?",
                owner_id="alice",
                where="owner_id = 'alice'",
                app_id="test_app",
                project_id="test_proj",
                episode_recaller=ep_recaller,
                atomic_fact_recaller=fact_recaller,
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


@pytest.mark.parametrize("radius", [None, 0.0, 0.5, 1.0])
async def test_radius_filters_fact_and_subject_children_before_maxsim(
    radius: float | None,
) -> None:
    children = [
        Candidate(id=name, score=score, metadata={"parent_id": name})
        for name, score in [
            ("fact_high", 0.9),
            ("fact_low", 0.2),
            ("subject_edge", 0.5),
            ("subject_low", 0.1),
            ("subject_zero", 0.0),
        ]
    ]
    keyword = Candidate(id="keyword", score=0.1, metadata={"parent_id": "keyword"})
    parents = {
        c.id: _mc_candidate(c.id, c.id).model_copy(
            update={
                "metadata": {**_mc_candidate(c.id, c.id).metadata, "entry_id": c.id}
            }
        )
        for c in [*children, keyword]
    }
    ep = _StubEpisodeRecaller([], parents)
    ep.dense_recall_subject_as_child = AsyncMock(return_value=children[2:])
    fact = _StubFactRecaller(children[:2])
    fact.sparse_recall = AsyncMock(return_value=[keyword])
    expected = {c.id for c in children if radius is None or c.score >= radius} | {
        "keyword"
    }

    async def cluster_passthrough(query: str, **kwargs: Any) -> list[Candidate]:
        return await kwargs["base_retrieve"](query, 20)

    async def drive_rounds(
        query: str, **kwargs: Any
    ) -> tuple[list[Candidate], AgenticDecision]:
        for retrieve, q in [
            (kwargs["base_retrieve"], query),
            (kwargs["round2_retrieve"], "refined query"),
        ]:
            hits = await retrieve(q, 20)
            assert {c.id for c in hits} == expected
            if radius == 0.5:
                assert all(c.score < radius for c in hits)
        return [], AgenticDecision(is_multi_round=True)

    with (
        patch(
            "everos.memory.search.agentic.cluster_repo.list_for_owner",
            AsyncMock(return_value=[]),
        ),
        patch("everos.memory.search.agentic.acluster_retrieve", cluster_passthrough),
        patch("everos.memory.search.agentic.aagentic_retrieve", drive_rounds),
    ):
        await search_episodes_agentic(
            "query",
            owner_id="alice",
            where=None,
            episode_recaller=ep,
            atomic_fact_recaller=fact,
            embed_query_fn=AsyncMock(return_value=[0.1, 0.2]),
            reranker=_StubReranker(),
            llm=FakeLLMClient(responses=[]),
            top_k=10,
            radius=radius,
        )
    assert ep.dense_recall_subject_as_child.await_count == 2


async def test_radius_checks_query_specific_children_at_maxsim_entry() -> None:
    from everalgo.rank import amaxsim_retrieve

    queries = ["original", "refined", "facet"]
    vectors = {q: [float(i), 1.0] for i, q in enumerate(queries)}
    query_for_vector = {tuple(v): q for q, v in vectors.items()}

    def child(cid: str, score: float, parent: str) -> Candidate:
        return Candidate(
            id=cid, score=score, source="vector", metadata={"parent_id": parent}
        )

    facts = {
        q: [child("crossing", score, "shared"), child(f"{q}_fact", 0.7, q)]
        for q, score in zip(queries, [0.9, 0.2, 0.8], strict=True)
    }
    subjects = {
        q: [
            child(f"{q}_subject", 0.5, q),
            # Shares a parent with a stronger fact: checking only the pooled
            # parent cannot prove this child was removed before MaxSim.
            child("weak_subject", 0.1, "shared"),
        ]
        for q in queries
    }
    keyword = Candidate(
        id="keyword", score=0.1, source="keyword", metadata={"parent_id": "keyword"}
    )
    parents = {
        pid: _mc_candidate(pid, pid).model_copy(
            update={"metadata": {**_mc_candidate(pid, pid).metadata, "entry_id": pid}}
        )
        for pid in [*queries, "shared", "keyword"]
    }
    ep = _StubEpisodeRecaller([], parents)
    fact = _StubFactRecaller([])

    async def recall_facts(vector: Sequence[float], *_: Any, **__: Any):
        return facts[query_for_vector[tuple(vector)]]

    async def recall_subjects(vector: Sequence[float], *_: Any, **__: Any):
        return subjects[query_for_vector[tuple(vector)]]

    fact.dense_recall = AsyncMock(side_effect=recall_facts)
    fact.sparse_recall = AsyncMock(return_value=[keyword])
    ep.dense_recall_subject_as_child = AsyncMock(side_effect=recall_subjects)
    embed = AsyncMock(side_effect=lambda q: vectors[q])
    observed: dict[str, list[dict[str, float]]] = {q: [] for q in queries}

    async def inspect_maxsim(q: str, *, child_retrieve: Any, **kwargs: Any):
        async def inspect_children(query: str, k: int) -> list[Candidate]:
            children = await child_retrieve(query, k)
            observed[query].append({c.id: c.score for c in children})
            return children

        return await amaxsim_retrieve(q, child_retrieve=inspect_children, **kwargs)

    async def cluster_passthrough(query: str, **kwargs: Any):
        return await kwargs["base_retrieve"](query, 20)

    async def drive_rounds(query: str, **kwargs: Any):
        for i, q in enumerate(queries):
            retrieve = kwargs["base_retrieve" if i == 0 else "round2_retrieve"]
            hits = await retrieve(q, 20)
            expected = {q, "keyword"} | ({"shared"} if i != 1 else set())
            assert {c.id for c in hits} == expected
            assert all(c.score < 0.5 for c in hits)  # real RRF scores survive
        return [], AgenticDecision(is_multi_round=True)

    with (
        patch("everos.memory.search.agentic.amaxsim_retrieve", inspect_maxsim),
        patch("everos.memory.search.agentic.acluster_retrieve", cluster_passthrough),
        patch("everos.memory.search.agentic.aagentic_retrieve", drive_rounds),
        patch(
            "everos.memory.search.agentic.cluster_repo.list_for_owner",
            AsyncMock(return_value=[]),
        ),
    ):
        await search_episodes_agentic(
            queries[0],
            owner_id="alice",
            where=None,
            episode_recaller=ep,
            atomic_fact_recaller=fact,
            embed_query_fn=embed,
            reranker=_StubReranker(),
            llm=FakeLLMClient(responses=[]),
            top_k=10,
            radius=0.5,
        )

    for i, q in enumerate(queries):
        dense = {f"{q}_fact": 0.7, f"{q}_subject": 0.5}
        if i != 1:
            dense["crossing"] = [0.9, 0.2, 0.8][i]
        assert len(observed[q]) == 2
        assert dense in observed[q]
        assert {"keyword": 0.1} in observed[q]
    assert [call.args[0] for call in embed.await_args_list] == queries
    for recaller in (fact.dense_recall, ep.dense_recall_subject_as_child):
        assert [call.args[0] for call in recaller.await_args_list] == list(
            vectors.values()
        )

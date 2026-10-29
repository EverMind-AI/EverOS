"""Unit tests for knowledge search service.

The real recaller runs over a scripted knowledge-topic index; the everalgo
facade (``acategory_retrieve``) is replaced by a driver that does what it
does -- recall through ``base_retrieve``, rerank through ``rerank_fn`` --
so the service's wiring of both callbacks is exercised. The catalog runs
over mocked repositories (topic content, document context).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from everos.component.rerank import RerankResult
from everos.component.tokenizer import Tokenizer
from everos.component.utils.datetime import get_utc_now
from everos.core.errors import ProviderNotConfiguredError
from everos.infra.persistence.sqlite.tables.knowledge import (
    KnowledgeDocumentRow,
    KnowledgeTopicRow,
)
from everos.memory.search.recall import KnowledgeTopicRecaller, RecallerDeps
from everos.service.knowledge import (
    DocumentContext,
    KnowledgeService,
    SearchKnowledgeResult,
    compile_knowledge_where,
)
from tests.fakes import make_knowledge_service
from tests.unit.test_memory.test_search._records import (
    ScriptedIndex,
    topic,
    topic_hit,
)

_FACADE = "everos.service.knowledge.acategory_retrieve"

# ── Helpers ──────────────────────────────────────────────────────────────────


def _hit(
    node_id: str = "n_001",
    *,
    topic_name: str = "Neural Networks",
    topic_path: str = "Technology / Neural Networks",
    summary: str = "Overview of neural networks.",
) -> Any:
    t = topic(node_id, category_id="Technology").model_copy(
        update={
            "doc_id": "d_testdoc00001",
            "topic_name": topic_name,
            "topic_path": topic_path,
            "summary": summary,
        }
    )
    return topic_hit(t, score=1.0)


def _topic_row(node_id: str, content: str) -> KnowledgeTopicRow:
    now = get_utc_now()
    return KnowledgeTopicRow(
        node_id=node_id,
        doc_id="d_testdoc00001",
        category_id="Technology",
        topic_index=1,
        topic_name="t",
        topic_path="t",
        depth=1,
        summary="s",
        content=content,
        md_path="knowledge/x/index.md",
        created_at=now,
        updated_at=now,
    )


def _doc_row(
    doc_id: str = "d_testdoc00001",
    title: str = "AI Handbook",
    summary: str = "A handbook on AI.",
) -> KnowledgeDocumentRow:
    now = get_utc_now()
    return KnowledgeDocumentRow(
        doc_id=doc_id,
        app_id="default",
        project_id="default",
        category_id="Technology",
        title=title,
        summary=summary,
        source_name="ai.pdf",
        source_type="file",
        md_path="/tmp/knowledge/Technology/d_testdoc00001",
        created_at=now,
        updated_at=now,
    )


def _mock_settings() -> MagicMock:
    """Build a mock Settings with knowledge.search defaults."""
    s = MagicMock()
    s.knowledge.search.recall_n = 200
    s.knowledge.search.rerank_n = 50
    s.knowledge.search.mass_top_m = 50
    s.knowledge.search.lam = 0.1
    s.knowledge.search.top_k_cap = 100
    return s


def _driver() -> AsyncMock:
    """Stands in for ``acategory_retrieve``: recall, rerank, truncate."""

    async def _run(
        query: str,
        *,
        base_retrieve: Any,
        rerank_fn: Any,
        recall_n: int,
        top_n: int,
        **_: Any,
    ) -> list[Any]:
        recalled = await base_retrieve(query, recall_n)
        return (await rerank_fn(query, recalled))[:top_n]

    return AsyncMock(side_effect=_run)


def _patch_stack(
    hits: Sequence[Any] = (),
    *,
    scores: Sequence[float] = (),
    contents: dict[str, str] | None = None,
    doc_rows: list[KnowledgeDocumentRow] | None = None,
) -> dict[str, Any]:
    """Mocks for one search; the reranker scores the pool with ``scores``."""
    doc_repo = AsyncMock()
    doc_repo.get_documents_by_ids = AsyncMock(return_value=doc_rows or [])
    topic_repo = AsyncMock()
    topic_repo.get_topics_by_ids = AsyncMock(
        return_value=[_topic_row(tid, text) for tid, text in (contents or {}).items()]
    )

    embedder = AsyncMock()
    embedder.embed = AsyncMock(return_value=[0.1] * 4)

    reranker = AsyncMock()
    reranker.rerank = AsyncMock(
        side_effect=lambda q, passages, **kw: [
            RerankResult(index=i, score=scores[i] if i < len(scores) else 0.5)
            for i in range(len(passages))
        ]
    )

    tok = MagicMock(spec=Tokenizer)
    tok.tokenize.return_value = ["q"]
    recaller = KnowledgeTopicRecaller(
        RecallerDeps(tokenizer=tok),
        ScriptedIndex(sparse=hits, dense=hits),  # type: ignore[arg-type]
    )
    return {
        "doc_repo": doc_repo,
        "topic_repo": topic_repo,
        "embedder": embedder,
        "reranker": reranker,
        "recaller": recaller,
        "settings": _mock_settings(),
    }


def _service(mocks: dict[str, Any], **overrides: object) -> KnowledgeService:
    """A service over the mocks of :func:`_patch_stack`; overrides replace one."""
    fields = {
        "embedding": mocks["embedder"],
        "reranker": mocks["reranker"],
        "recaller": mocks["recaller"],
        "search_config": mocks["settings"].knowledge.search,
        "documents": mocks["doc_repo"],
        "topics": mocks["topic_repo"],
    }
    return make_knowledge_service(root=Path("/unused"), **{**fields, **overrides})


# ── compile_knowledge_where ──────────────────────────────────────────────────


class TestCompileKnowledgeWhere:
    def test_basic_clause(self) -> None:
        result = compile_knowledge_where("myapp", "myproj")
        from everos.infra.persistence.index import all_of, eq

        assert result == all_of(eq("app_id", "myapp"), eq("project_id", "myproj"))

    def test_defaults(self) -> None:
        result = compile_knowledge_where("default", "default")
        from everos.infra.persistence.lancedb import render_predicate

        rendered = render_predicate(result)
        assert "app_id = 'default'" in rendered
        assert "project_id = 'default'" in rendered

    def test_rejects_invalid_app_id_with_sql_injection(self) -> None:
        with pytest.raises(ValueError, match="app_id"):
            compile_knowledge_where("app'; DROP TABLE --", "proj")

    def test_rejects_invalid_project_id_with_sql_injection(self) -> None:
        with pytest.raises(ValueError, match="project_id"):
            compile_knowledge_where("app", "proj'); DELETE FROM--")

    def test_rejects_empty_app_id(self) -> None:
        with pytest.raises(ValueError, match="app_id"):
            compile_knowledge_where("", "proj")

    def test_rejects_empty_project_id(self) -> None:
        with pytest.raises(ValueError, match="project_id"):
            compile_knowledge_where("app", "")

    def test_accepts_valid_ids_with_special_chars(self) -> None:
        result = compile_knowledge_where("my_app.v2", "project-1")
        from everos.infra.persistence.lancedb import render_predicate

        rendered = render_predicate(result)
        assert "my_app.v2" in rendered
        assert "project-1" in rendered

    def test_accepts_valid_ids_with_at_plus(self) -> None:
        result = compile_knowledge_where("app@org+v1", "proj_1")
        from everos.infra.persistence.lancedb import render_predicate

        rendered = render_predicate(result)
        assert "app@org+v1" in rendered
        assert "proj_1" in rendered


# ── search_knowledge ─────────────────────────────────────────────────────────


class TestSearchKnowledgeFacadeWiring:
    """search_knowledge delegates to acategory_retrieve with the config."""

    async def test_calls_facade_with_config_params(self) -> None:
        mocks = _patch_stack([_hit()], doc_rows=[_doc_row()])

        with patch(_FACADE, _driver()) as facade:
            await _service(mocks).search_knowledge(
                query="neural nets", method="keyword"
            )

        facade.assert_awaited_once()
        args, kwargs = facade.call_args
        assert args[0] == "neural nets"
        assert kwargs["recall_n"] == 200
        assert kwargs["rerank_n"] == 50
        assert kwargs["mass_top_m"] == 50
        assert kwargs["lam"] == pytest.approx(0.1)
        assert kwargs["top_n"] == 10

    async def test_top_n_capped_by_top_k_cap(self) -> None:
        mocks = _patch_stack()

        with patch(_FACADE, _driver()) as facade:
            await _service(mocks).search_knowledge(
                query="test", method="keyword", top_k=200
            )

        assert facade.call_args.kwargs["top_n"] == 100

    async def test_reranks_on_the_catalog_content(self) -> None:
        """Passages are the topics' full content, read from the catalog."""
        mocks = _patch_stack(
            [_hit("n_001"), _hit("n_002")],
            contents={"n_001": "Content A.", "n_002": "Content B."},
        )

        with patch(_FACADE, _driver()):
            await _service(mocks).search_knowledge(query="q", method="keyword")

        passages = mocks["reranker"].rerank.await_args.args[1]
        assert passages == ["Content A.", "Content B."]


class TestSearchKnowledgeResults:
    async def test_returns_hits_with_scores(self) -> None:
        mocks = _patch_stack(
            [_hit("n_001"), _hit("n_002", topic_name="CNNs")],
            scores=[0.9, 0.7],
            doc_rows=[_doc_row()],
        )

        with patch(_FACADE, _driver()):
            result = await _service(mocks).search_knowledge(
                query="neural networks", method="keyword"
            )

        assert isinstance(result, SearchKnowledgeResult)
        assert [(h.topic_id, h.score) for h in result.hits] == [
            ("n_001", 0.9),
            ("n_002", 0.7),
        ]
        assert result.total == 2

    async def test_empty_results(self) -> None:
        with patch(_FACADE, _driver()):
            result = await _service(_patch_stack()).search_knowledge(
                query="nothing", method="keyword"
            )

        assert result.hits == []
        assert result.total == 0


class TestSearchKnowledgeIncludeContent:
    async def test_content_populated_when_true(self) -> None:
        text = "Full content of neural networks topic."
        mocks = _patch_stack([_hit("n_001")], contents={"n_001": text})

        with patch(_FACADE, _driver()):
            result = await _service(mocks).search_knowledge(
                query="neural nets", method="keyword", include_content=True
            )

        assert result.hits[0].content == text

    async def test_content_none_when_false(self) -> None:
        mocks = _patch_stack([_hit("n_001")], contents={"n_001": "Some content."})

        with patch(_FACADE, _driver()):
            result = await _service(mocks).search_knowledge(
                query="neural nets", method="keyword", include_content=False
            )

        assert result.hits[0].content is None


class TestSearchKnowledgeScoreThreshold:
    async def test_filters_low_score_candidates(self) -> None:
        mocks = _patch_stack(
            [_hit("n_001"), _hit("n_002", topic_name="Low")], scores=[0.9, 0.1]
        )

        with patch(_FACADE, _driver()):
            result = await _service(mocks).search_knowledge(
                query="neural nets", method="keyword", score_threshold=0.5
            )

        assert [h.topic_id for h in result.hits] == ["n_001"]

    async def test_no_filtering_when_threshold_none(self) -> None:
        mocks = _patch_stack(
            [_hit("n_001"), _hit("n_002", topic_name="Low")], scores=[0.9, 0.1]
        )

        with patch(_FACADE, _driver()):
            result = await _service(mocks).search_knowledge(
                query="test", method="keyword"
            )

        assert len(result.hits) == 2


class TestSearchKnowledgeDocumentContext:
    async def test_hits_carry_document_context(self) -> None:
        doc = _doc_row(title="AI Handbook", summary="A handbook on AI.")
        mocks = _patch_stack([_hit("n_001")], doc_rows=[doc])

        with patch(_FACADE, _driver()):
            result = await _service(mocks).search_knowledge(
                query="AI", method="keyword"
            )

        hit = result.hits[0]
        assert isinstance(hit.document, DocumentContext)
        assert hit.document.doc_id == "d_testdoc00001"
        assert hit.document.title == "AI Handbook"
        assert hit.document.summary == "A handbook on AI."


class TestSearchKnowledgeTookMs:
    async def test_took_ms_positive(self) -> None:
        with patch(_FACADE, _driver()):
            result = await _service(_patch_stack()).search_knowledge(
                query="test", method="keyword"
            )

        assert result.took_ms >= 0


class TestSearchKnowledgeReturnFields:
    """Verify all SearchHit fields are correctly populated."""

    async def test_all_hit_fields_populated(self) -> None:
        mocks = _patch_stack([_hit("n_001")], doc_rows=[_doc_row()])

        with patch(_FACADE, _driver()):
            result = await _service(mocks).search_knowledge(
                query="neural", method="keyword"
            )

        hit = result.hits[0]
        assert hit.topic_id == "n_001"
        assert hit.category_id == "Technology"
        assert hit.topic_name == "Neural Networks"
        assert hit.topic_path == "Technology / Neural Networks"
        assert hit.depth == 1
        assert hit.summary == "Overview of neural networks."
        assert hit.retrieval_method == "keyword"
        assert hit.source == "keyword"

    async def test_hybrid_fuses_both_routes(self) -> None:
        """Hybrid recalls both routes and fuses them before the rerank."""
        mocks = _patch_stack([_hit("n_001")])

        with patch(_FACADE, _driver()):
            result = await _service(mocks).search_knowledge(query="q", method="hybrid")

        assert [h.topic_id for h in result.hits] == ["n_001"]
        assert result.hits[0].retrieval_method == "hybrid"


class TestSearchKnowledgeProviderRequired:
    """Missing providers raise ProviderNotConfiguredError (HTTP 422)."""

    async def test_raises_without_embedding(self) -> None:
        with pytest.raises(ProviderNotConfiguredError, match="embedding"):
            await _service(_patch_stack(), embedding=None).search_knowledge(
                query="test", method="keyword"
            )

    async def test_raises_without_reranker(self) -> None:
        with pytest.raises(ProviderNotConfiguredError, match="rerank"):
            await _service(_patch_stack(), reranker=None).search_knowledge(
                query="test", method="keyword"
            )

"""Unit tests for knowledge search service.

White-box surfaces mocked:
    ``acategory_retrieve`` (the everalgo facade)
    ``knowledge_document_repo`` (get_documents_by_ids)
    ``_get_embedding`` (embedding provider)
    ``_get_reranker`` (rerank provider)
    ``load_settings`` (knowledge search settings)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from everalgo.types import Candidate

from everos.component.rerank.protocol import RerankResult
from everos.component.utils.datetime import get_utc_now
from everos.core.errors import ProviderNotConfiguredError
from everos.infra.persistence.sqlite.tables.knowledge import (
    KnowledgeDocumentRow,
)
from everos.service.knowledge import (
    DocumentContext,
    SearchKnowledgeResult,
    _enrich_with_content,
    _run_category_pipeline,
    compile_knowledge_where,
    search_knowledge,
)

_MOD = "everos.service.knowledge"
_CONFIG_MOD = "everos.config"

# ── Helpers ──────────────────────────────────────────────────────────────────


def _candidate(
    node_id: str = "n_001",
    score: float = 0.85,
    source: str = "keyword",
    doc_id: str = "d_testdoc00001",
    category_id: str = "Technology",
    topic_name: str = "Neural Networks",
    topic_path: str = "Technology / Neural Networks",
    depth: int = 1,
    summary: str = "Overview of neural networks.",
    content: str = "",
) -> Candidate:
    return Candidate(
        id=node_id,
        score=score,
        source=source,
        metadata={
            "doc_id": doc_id,
            "category_id": category_id,
            "topic_name": topic_name,
            "topic_path": topic_path,
            "depth": depth,
            "summary": summary,
            "content": content,
        },
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
    s.embedding.model = ""
    s.embedding.api_key = None
    return s


def _patch_stack(
    facade_return: list[Candidate] | None = None,
    doc_rows: list[KnowledgeDocumentRow] | None = None,
    embed_vector: list[float] | None = None,
):
    """Return a dict of patches for common mocks.

    ``acategory_retrieve`` is mocked at the module level — its internal
    behavior (recall -> rollup -> rerank -> boost) is tested in everalgo.
    """
    acategory = AsyncMock(return_value=facade_return or [])

    doc_repo = AsyncMock()
    doc_repo.get_documents_by_ids = AsyncMock(return_value=doc_rows or [])

    embedder = AsyncMock()
    embedder.embed = AsyncMock(return_value=embed_vector or [0.1] * 1024)

    reranker = AsyncMock()

    recaller = AsyncMock()

    settings = _mock_settings()

    return {
        "acategory": acategory,
        "doc_repo": doc_repo,
        "embedder": embedder,
        "reranker": reranker,
        "recaller": recaller,
        "settings": settings,
    }


# ── compile_knowledge_where ──────────────────────────────────────────────────


class TestCompileKnowledgeWhere:
    def test_basic_clause(self) -> None:
        result = compile_knowledge_where("myapp", "myproj")
        assert result == "app_id = 'myapp' AND project_id = 'myproj'"

    def test_defaults(self) -> None:
        result = compile_knowledge_where("default", "default")
        assert "app_id = 'default'" in result
        assert "project_id = 'default'" in result

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
        assert "my_app.v2" in result
        assert "project-1" in result

    def test_accepts_valid_ids_with_at_plus(self) -> None:
        result = compile_knowledge_where("app@org+v1", "proj_1")
        assert "app@org+v1" in result
        assert "proj_1" in result


class TestKnowledgeRerankContent:
    async def test_empty_sqlite_content_uses_summary_only_for_reranking(self) -> None:
        candidate = _candidate(
            node_id="n_empty_content",
            summary="Non-empty topic summary for reranking.",
            content="",
        )
        topic = MagicMock(node_id="n_empty_content", content="")

        with patch(
            f"{_MOD}.knowledge_topic_sqlite_repo.get_topics_by_ids",
            new=AsyncMock(return_value=[topic]),
        ):
            enriched = await _enrich_with_content([candidate])

        assert enriched[0].metadata["content"] == ""
        assert enriched[0].metadata["rerank_content"] == (
            "Non-empty topic summary for reranking."
        )

    async def test_blank_content_and_summary_fall_through_to_topic_name(self) -> None:
        candidate = _candidate(
            node_id="n_blank_summary",
            summary="  \t ",
            topic_name="Useful Topic Name",
        )
        topic = MagicMock(node_id="n_blank_summary", content=" \n ")

        with patch(
            f"{_MOD}.knowledge_topic_sqlite_repo.get_topics_by_ids",
            new=AsyncMock(return_value=[topic]),
        ):
            enriched = await _enrich_with_content([candidate])

        assert enriched[0].metadata["content"] == " \n "
        assert enriched[0].metadata["rerank_content"] == "Useful Topic Name"

    async def test_missing_row_and_blank_metadata_fall_back_to_candidate_id(
        self,
    ) -> None:
        candidate = _candidate(
            node_id="n_stable_fallback",
            summary=" ",
            topic_name="\n",
        )

        with patch(
            f"{_MOD}.knowledge_topic_sqlite_repo.get_topics_by_ids",
            new=AsyncMock(return_value=[]),
        ):
            enriched = await _enrich_with_content([candidate])

        assert enriched[0].metadata["content"] == ""
        assert enriched[0].metadata["rerank_content"] == "n_stable_fallback"

    async def test_all_blank_fields_use_nonempty_terminal_fallback(self) -> None:
        candidate = _candidate(node_id="", summary=" ", topic_name="\n")

        with patch(
            f"{_MOD}.knowledge_topic_sqlite_repo.get_topics_by_ids",
            new=AsyncMock(return_value=[]),
        ):
            enriched = await _enrich_with_content([candidate])

        assert enriched[0].metadata["rerank_content"] == "Knowledge topic"

    async def test_nonempty_sqlite_content_remains_authoritative(self) -> None:
        candidate = _candidate(node_id="n_full_content", summary="Fallback summary")
        topic = MagicMock(
            node_id="n_full_content",
            content="\n  Authoritative body.  \n",
        )

        with patch(
            f"{_MOD}.knowledge_topic_sqlite_repo.get_topics_by_ids",
            new=AsyncMock(return_value=[topic]),
        ):
            enriched = await _enrich_with_content([candidate])

        assert enriched[0].metadata["content"] == "\n  Authoritative body.  \n"
        assert enriched[0].metadata["rerank_content"] == "Authoritative body."

    async def test_category_pipeline_sends_nonempty_rerank_passage(self) -> None:
        candidate = _candidate(
            node_id="n_provider_seam",
            summary="Provider-safe topic summary.",
            content="",
        )
        topic = MagicMock(node_id="n_provider_seam", content="")
        recaller = AsyncMock()
        recaller.sparse_recall = AsyncMock(return_value=[candidate])
        reranker = AsyncMock()
        reranker.rerank = AsyncMock(return_value=[RerankResult(0, 0.9)])
        config = _mock_settings().knowledge.search

        async def _run_facade(query, *, base_retrieve, rerank_fn, **_kwargs):
            recalled = await base_retrieve(query, 10)
            return await rerank_fn(query, recalled)

        with (
            patch(f"{_MOD}._build_recaller", return_value=recaller),
            patch(
                f"{_MOD}.knowledge_topic_sqlite_repo.get_topics_by_ids",
                new=AsyncMock(return_value=[topic]),
            ),
            patch("everalgo.rank.acategory_retrieve", new=_run_facade),
        ):
            await _run_category_pipeline(
                "query",
                "app_id = 'default' AND project_id = 'default'",
                method="keyword",
                vector=[],
                reranker=reranker,
                config=config,
                top_k=5,
            )

        documents = reranker.rerank.await_args.args[1]
        assert documents == ["Provider-safe topic summary."]
        assert all(document.strip() for document in documents)


# ── search_knowledge ─────────────────────────────────────────────────────────


class TestSearchKnowledgeFacadeWiring:
    """Verify search_knowledge delegates to acategory_retrieve correctly."""

    async def test_calls_facade_with_config_params(self) -> None:
        c = _candidate(node_id="n_001", score=0.9, content="Neural net content.")
        mocks = _patch_stack(facade_return=[c], doc_rows=[_doc_row()])

        with (
            patch(f"{_MOD}._build_recaller", return_value=mocks["recaller"]),
            patch(f"{_MOD}.knowledge_document_repo", mocks["doc_repo"]),
            patch(f"{_MOD}._get_embedding", return_value=mocks["embedder"]),
            patch(f"{_MOD}._get_reranker", return_value=mocks["reranker"]),
            patch(f"{_CONFIG_MOD}.load_settings", return_value=mocks["settings"]),
            patch(
                "everalgo.rank.acategory_retrieve", mocks["acategory"]
            ) as mock_facade,
        ):
            await search_knowledge(query="neural nets", method="keyword")

        mock_facade.assert_awaited_once()
        call_kwargs = mock_facade.call_args
        assert call_kwargs[0][0] == "neural nets"
        assert call_kwargs[1]["recall_n"] == 200
        assert call_kwargs[1]["rerank_n"] == 50
        assert call_kwargs[1]["mass_top_m"] == 50
        assert call_kwargs[1]["lam"] == pytest.approx(0.1)
        assert call_kwargs[1]["top_n"] == 10

    async def test_top_n_capped_by_top_k_cap(self) -> None:
        mocks = _patch_stack(doc_rows=[_doc_row()])

        with (
            patch(f"{_MOD}._build_recaller", return_value=mocks["recaller"]),
            patch(f"{_MOD}.knowledge_document_repo", mocks["doc_repo"]),
            patch(f"{_MOD}._get_embedding", return_value=mocks["embedder"]),
            patch(f"{_MOD}._get_reranker", return_value=mocks["reranker"]),
            patch(f"{_CONFIG_MOD}.load_settings", return_value=mocks["settings"]),
            patch(
                "everalgo.rank.acategory_retrieve", mocks["acategory"]
            ) as mock_facade,
        ):
            # top_k=200 but top_k_cap=100 → effective_k=100
            await search_knowledge(query="test", method="keyword", top_k=200)

        assert mock_facade.call_args[1]["top_n"] == 100


class TestSearchKnowledgeResults:
    """Verify result assembly from facade output."""

    async def test_returns_hits_with_scores(self) -> None:
        c1 = _candidate(node_id="n_001", score=0.9, content="Content A.")
        c2 = _candidate(
            node_id="n_002", score=0.7, topic_name="CNNs", content="Content B."
        )
        mocks = _patch_stack(facade_return=[c1, c2], doc_rows=[_doc_row()])

        with (
            patch(f"{_MOD}._build_recaller", return_value=mocks["recaller"]),
            patch(f"{_MOD}.knowledge_document_repo", mocks["doc_repo"]),
            patch(f"{_MOD}._get_embedding", return_value=mocks["embedder"]),
            patch(f"{_MOD}._get_reranker", return_value=mocks["reranker"]),
            patch(f"{_CONFIG_MOD}.load_settings", return_value=mocks["settings"]),
            patch("everalgo.rank.acategory_retrieve", mocks["acategory"]),
        ):
            result = await search_knowledge(query="neural networks", method="keyword")

        assert isinstance(result, SearchKnowledgeResult)
        assert len(result.hits) == 2
        assert result.total == 2

    async def test_empty_results(self) -> None:
        mocks = _patch_stack()

        with (
            patch(f"{_MOD}._build_recaller", return_value=mocks["recaller"]),
            patch(f"{_MOD}.knowledge_document_repo", mocks["doc_repo"]),
            patch(f"{_MOD}._get_embedding", return_value=mocks["embedder"]),
            patch(f"{_MOD}._get_reranker", return_value=mocks["reranker"]),
            patch(f"{_CONFIG_MOD}.load_settings", return_value=mocks["settings"]),
            patch("everalgo.rank.acategory_retrieve", mocks["acategory"]),
        ):
            result = await search_knowledge(query="nothing", method="keyword")

        assert result.hits == []
        assert result.total == 0


class TestSearchKnowledgeIncludeContent:
    async def test_content_populated_when_true(self) -> None:
        content_text = "Full content of neural networks topic."
        c = _candidate(node_id="n_001", score=0.9, content=content_text)
        mocks = _patch_stack(facade_return=[c], doc_rows=[_doc_row()])

        with (
            patch(f"{_MOD}._build_recaller", return_value=mocks["recaller"]),
            patch(f"{_MOD}.knowledge_document_repo", mocks["doc_repo"]),
            patch(f"{_MOD}._get_embedding", return_value=mocks["embedder"]),
            patch(f"{_MOD}._get_reranker", return_value=mocks["reranker"]),
            patch(f"{_CONFIG_MOD}.load_settings", return_value=mocks["settings"]),
            patch("everalgo.rank.acategory_retrieve", mocks["acategory"]),
        ):
            result = await search_knowledge(
                query="neural nets", method="keyword", include_content=True
            )

        assert result.hits[0].content == content_text

    async def test_content_none_when_false(self) -> None:
        c = _candidate(node_id="n_001", score=0.9, content="Some content.")
        mocks = _patch_stack(facade_return=[c], doc_rows=[_doc_row()])

        with (
            patch(f"{_MOD}._build_recaller", return_value=mocks["recaller"]),
            patch(f"{_MOD}.knowledge_document_repo", mocks["doc_repo"]),
            patch(f"{_MOD}._get_embedding", return_value=mocks["embedder"]),
            patch(f"{_MOD}._get_reranker", return_value=mocks["reranker"]),
            patch(f"{_CONFIG_MOD}.load_settings", return_value=mocks["settings"]),
            patch("everalgo.rank.acategory_retrieve", mocks["acategory"]),
        ):
            result = await search_knowledge(
                query="neural nets", method="keyword", include_content=False
            )

        assert result.hits[0].content is None


class TestSearchKnowledgeScoreThreshold:
    async def test_filters_low_score_candidates(self) -> None:
        high = _candidate(node_id="n_001", score=0.9)
        low = _candidate(node_id="n_002", score=0.1, topic_name="Low")
        mocks = _patch_stack(facade_return=[high, low], doc_rows=[_doc_row()])

        with (
            patch(f"{_MOD}._build_recaller", return_value=mocks["recaller"]),
            patch(f"{_MOD}.knowledge_document_repo", mocks["doc_repo"]),
            patch(f"{_MOD}._get_embedding", return_value=mocks["embedder"]),
            patch(f"{_MOD}._get_reranker", return_value=mocks["reranker"]),
            patch(f"{_CONFIG_MOD}.load_settings", return_value=mocks["settings"]),
            patch("everalgo.rank.acategory_retrieve", mocks["acategory"]),
        ):
            result = await search_knowledge(
                query="neural nets", method="keyword", score_threshold=0.5
            )

        assert len(result.hits) == 1
        assert result.hits[0].topic_id == "n_001"

    async def test_no_filtering_when_threshold_none(self) -> None:
        c1 = _candidate(node_id="n_001", score=0.9)
        c2 = _candidate(node_id="n_002", score=0.1, topic_name="Low")
        mocks = _patch_stack(facade_return=[c1, c2], doc_rows=[_doc_row()])

        with (
            patch(f"{_MOD}._build_recaller", return_value=mocks["recaller"]),
            patch(f"{_MOD}.knowledge_document_repo", mocks["doc_repo"]),
            patch(f"{_MOD}._get_embedding", return_value=mocks["embedder"]),
            patch(f"{_MOD}._get_reranker", return_value=mocks["reranker"]),
            patch(f"{_CONFIG_MOD}.load_settings", return_value=mocks["settings"]),
            patch("everalgo.rank.acategory_retrieve", mocks["acategory"]),
        ):
            result = await search_knowledge(query="test", method="keyword")

        assert len(result.hits) == 2


class TestSearchKnowledgeDocumentContext:
    async def test_hits_carry_document_context(self) -> None:
        c = _candidate(node_id="n_001", score=0.9)
        doc = _doc_row(title="AI Handbook", summary="A handbook on AI.")
        mocks = _patch_stack(facade_return=[c], doc_rows=[doc])

        with (
            patch(f"{_MOD}._build_recaller", return_value=mocks["recaller"]),
            patch(f"{_MOD}.knowledge_document_repo", mocks["doc_repo"]),
            patch(f"{_MOD}._get_embedding", return_value=mocks["embedder"]),
            patch(f"{_MOD}._get_reranker", return_value=mocks["reranker"]),
            patch(f"{_CONFIG_MOD}.load_settings", return_value=mocks["settings"]),
            patch("everalgo.rank.acategory_retrieve", mocks["acategory"]),
        ):
            result = await search_knowledge(query="AI", method="keyword")

        hit = result.hits[0]
        assert isinstance(hit.document, DocumentContext)
        assert hit.document.doc_id == "d_testdoc00001"
        assert hit.document.title == "AI Handbook"
        assert hit.document.summary == "A handbook on AI."


class TestSearchKnowledgeTookMs:
    async def test_took_ms_positive(self) -> None:
        mocks = _patch_stack()

        with (
            patch(f"{_MOD}._build_recaller", return_value=mocks["recaller"]),
            patch(f"{_MOD}.knowledge_document_repo", mocks["doc_repo"]),
            patch(f"{_MOD}._get_embedding", return_value=mocks["embedder"]),
            patch(f"{_MOD}._get_reranker", return_value=mocks["reranker"]),
            patch(f"{_CONFIG_MOD}.load_settings", return_value=mocks["settings"]),
            patch("everalgo.rank.acategory_retrieve", mocks["acategory"]),
        ):
            result = await search_knowledge(query="test", method="keyword")

        assert result.took_ms >= 0


class TestSearchKnowledgeReturnFields:
    """Verify all SearchHit fields are correctly populated."""

    async def test_all_hit_fields_populated(self) -> None:
        c = _candidate(
            node_id="n_001",
            score=0.85,
            source="keyword",
            doc_id="d_testdoc00001",
            category_id="Technology",
            topic_name="Neural Networks",
            topic_path="Technology / Neural Networks",
            depth=1,
            summary="Overview of neural networks.",
        )
        mocks = _patch_stack(facade_return=[c], doc_rows=[_doc_row()])

        with (
            patch(f"{_MOD}._build_recaller", return_value=mocks["recaller"]),
            patch(f"{_MOD}.knowledge_document_repo", mocks["doc_repo"]),
            patch(f"{_MOD}._get_embedding", return_value=mocks["embedder"]),
            patch(f"{_MOD}._get_reranker", return_value=mocks["reranker"]),
            patch(f"{_CONFIG_MOD}.load_settings", return_value=mocks["settings"]),
            patch("everalgo.rank.acategory_retrieve", mocks["acategory"]),
        ):
            result = await search_knowledge(query="neural", method="keyword")

        hit = result.hits[0]
        assert hit.topic_id == "n_001"
        assert hit.category_id == "Technology"
        assert hit.topic_name == "Neural Networks"
        assert hit.topic_path == "Technology / Neural Networks"
        assert hit.depth == 1
        assert hit.summary == "Overview of neural networks."
        assert hit.retrieval_method == "keyword"
        assert hit.source == "keyword"


class TestSearchKnowledgeProviderRequired:
    """Missing providers raise ProviderNotConfiguredError (HTTP 422)."""

    async def test_raises_without_embedding(self) -> None:
        mocks = _patch_stack()

        with (
            patch(f"{_MOD}._get_embedding", return_value=None),
            patch(f"{_MOD}._get_reranker", return_value=mocks["reranker"]),
            patch(f"{_CONFIG_MOD}.load_settings", return_value=mocks["settings"]),
            pytest.raises(ProviderNotConfiguredError, match="embedding"),
        ):
            await search_knowledge(query="test", method="keyword")

    async def test_raises_without_reranker(self) -> None:
        mocks = _patch_stack()

        with (
            patch(f"{_MOD}._get_embedding", return_value=mocks["embedder"]),
            patch(f"{_MOD}._get_reranker", return_value=None),
            patch(f"{_CONFIG_MOD}.load_settings", return_value=mocks["settings"]),
            pytest.raises(ProviderNotConfiguredError, match="rerank"),
        ):
            await search_knowledge(query="test", method="keyword")

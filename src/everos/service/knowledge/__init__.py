"""KnowledgeService -- knowledge document CRUD and search behind one facade.

The facade only delegates: uploads and writes live in ``_documents``, reads
in ``_queries``, category-aware retrieval in ``_search``, result types in
``_models``. Storage is two ports: ``KnowledgeStore`` (the markdown truth)
and ``KnowledgeCatalog`` (the records the cascade keeps); when the catalog
has not caught up with a document yet, delete and patch fall back to md.
"""

from __future__ import annotations

__all__ = [
    "CategoryOverview",
    "CreateDocumentResult",
    "DeleteResult",
    "DocumentContext",
    "DocumentDetail",
    "DocumentListResult",
    "DocumentOverviewItem",
    "KnowledgeExtractor",
    "KnowledgeService",
    "PatchResult",
    "SearchHit",
    "SearchKnowledgeResult",
    "TopicDetail",
    "TopicOverview",
    "compile_knowledge_where",
]

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from everalgo.types import ParsedContent, RawFile

from everos.memory.ports import KnowledgeCatalog, KnowledgeStore

from ._documents import DocumentManager
from ._models import CategoryOverview as CategoryOverview
from ._models import CreateDocumentResult as CreateDocumentResult
from ._models import DeleteResult as DeleteResult
from ._models import DocumentContext as DocumentContext
from ._models import DocumentDetail as DocumentDetail
from ._models import DocumentListResult as DocumentListResult
from ._models import DocumentOverviewItem as DocumentOverviewItem
from ._models import KnowledgeExtractor as KnowledgeExtractor
from ._models import PatchResult as PatchResult
from ._models import SearchHit as SearchHit
from ._models import SearchKnowledgeResult as SearchKnowledgeResult
from ._models import TopicDetail as TopicDetail
from ._models import TopicOverview as TopicOverview
from ._models import compile_knowledge_where as compile_knowledge_where
from ._queries import KnowledgeQueries
from ._search import KnowledgeSearcher

if TYPE_CHECKING:
    from everos.component.embedding import EmbeddingProvider
    from everos.component.rerank import RerankProvider
    from everos.config.settings import KnowledgeSearchSettings
    from everos.memory.search.recall import KnowledgeMethod, KnowledgeTopicRecaller

    from ._upload import KnowledgeUpload


class KnowledgeService:
    """Knowledge document CRUD and search, over the ports it is built with.

    Args:
        store: The markdown truth of documents, topics and the taxonomy.
        catalog: The document / topic records the cascade keeps.
        extractor: Topic extractor; ``None`` (no LLM) makes create / replace raise.
        embedding: Query embedder for search; ``None`` when not configured.
        reranker: Reranker for search; ``None`` when not configured.
        recaller: Recall over the knowledge-topic index.
        search_config: The ``[knowledge.search]`` section.
        parse_file: Parses an uploaded non-text file with the multimodal LLM.
        parser_available: Whether the parser is installed (else UTF-8 only).
        max_upload_bytes: The largest upload accepted.
    """

    def __init__(
        self,
        *,
        store: KnowledgeStore,
        catalog: KnowledgeCatalog,
        extractor: KnowledgeExtractor | None,
        embedding: EmbeddingProvider | None,
        reranker: RerankProvider | None,
        recaller: KnowledgeTopicRecaller,
        search_config: KnowledgeSearchSettings,
        parse_file: Callable[[RawFile], Awaitable[ParsedContent]],
        parser_available: bool = False,
        max_upload_bytes: int = 52_428_800,
    ) -> None:
        self._documents = DocumentManager(
            store=store,
            catalog=catalog,
            extractor=extractor,
            parse_file=parse_file,
            parser_available=parser_available,
            max_upload_bytes=max_upload_bytes,
        )
        self._queries = KnowledgeQueries(store=store, catalog=catalog)
        self._searcher = KnowledgeSearcher(
            catalog=catalog,
            embedding=embedding,
            reranker=reranker,
            recaller=recaller,
            config=search_config,
        )

    def require_search_providers(self) -> None:
        """Refuse what cannot work end to end without embedding and rerank.

        Uploads and search need both; reads, deletes and patches need neither.

        Raises:
            ProviderNotConfiguredError: Embedding (checked first) or rerank.
        """
        self._searcher.require_providers()

    async def create_from_upload(
        self,
        upload: KnowledgeUpload,
        *,
        title: str,
        app_id: str = "default",
        project_id: str = "default",
        source_type: str | None = None,
        category_id: str | None = None,
    ) -> CreateDocumentResult:
        """Create a document from an uploaded file (size check, parse, extract).

        Raises:
            InvalidInputError: The upload is too large or has no content.
            UnsupportedModalityError: The file cannot be parsed.
            LLMNotConfiguredError: No LLM is configured to extract topics.
            ExtractionEmptyError: When the extractor produces no memories.
        """
        parsed, raw = await self._documents.read_upload(upload)
        return await self.create_document(
            parsed=parsed,
            title=title,
            app_id=app_id,
            project_id=project_id,
            source_name=upload.filename,
            source_type=source_type,
            category_id=category_id,
            file_content=raw,
        )

    async def replace_from_upload(
        self,
        doc_id: str,
        upload: KnowledgeUpload,
        *,
        title: str,
        app_id: str = "default",
        project_id: str = "default",
        source_type: str | None = None,
        category_id: str | None = None,
    ) -> CreateDocumentResult:
        """Replace a document with an uploaded file.

        Raises:
            As :meth:`create_from_upload`, plus ``DocumentNotFoundError``.
        """
        parsed, raw = await self._documents.read_upload(upload)
        return await self.replace_document(
            doc_id=doc_id,
            parsed=parsed,
            title=title,
            app_id=app_id,
            project_id=project_id,
            source_name=upload.filename,
            source_type=source_type,
            category_id=category_id,
            file_content=raw,
        )

    async def create_document(
        self,
        *,
        parsed: ParsedContent,
        title: str,
        app_id: str = "default",
        project_id: str = "default",
        source_name: str | None = None,
        source_type: str | None = None,
        doc_id: str | None = None,
        category_id: str | None = None,
        file_content: bytes | None = None,
    ) -> CreateDocumentResult:
        """Create a knowledge document from parsed content.

        Raises:
            LLMNotConfiguredError: No LLM is configured to extract topics.
            DuplicateDocumentError: When ``doc_id`` already exists.
            ExtractionEmptyError: When the extractor produces no memories.
        """
        return await self._documents.create(
            doc_id=doc_id,
            parsed=parsed,
            title=title,
            app_id=app_id,
            project_id=project_id,
            source_name=source_name,
            source_type=source_type,
            category_id=category_id,
            file_content=file_content,
        )

    async def replace_document(
        self,
        *,
        doc_id: str,
        parsed: ParsedContent,
        title: str,
        app_id: str = "default",
        project_id: str = "default",
        source_name: str | None = None,
        source_type: str | None = None,
        category_id: str | None = None,
        file_content: bytes | None = None,
    ) -> CreateDocumentResult:
        """Replace a document atomically -- the old files come back on failure.

        Raises:
            LLMNotConfiguredError: No LLM is configured to extract topics.
            DocumentNotFoundError: When ``doc_id`` does not exist.
            ExtractionEmptyError: When extraction produces no memories.
        """
        return await self._documents.replace(
            doc_id=doc_id,
            parsed=parsed,
            title=title,
            app_id=app_id,
            project_id=project_id,
            source_name=source_name,
            source_type=source_type,
            category_id=category_id,
            file_content=file_content,
        )

    async def delete_document(
        self, doc_id: str, app_id: str, project_id: str
    ) -> DeleteResult:
        """Remove the document's markdown; idempotent (``deleted_topics=0``)."""
        return await self._documents.delete(doc_id, app_id, project_id)

    async def patch_document(
        self,
        doc_id: str,
        app_id: str,
        project_id: str,
        title: str | None = None,
        category_id: str | None = None,
    ) -> PatchResult:
        """Update mutable document metadata in md (truth) and the catalog (now).

        Raises:
            DocumentNotFoundError: Neither the catalog nor the markdown has it.
        """
        return await self._documents.patch(
            doc_id, app_id, project_id, title=title, category_id=category_id
        )

    async def get_document(
        self, doc_id: str, app_id: str, project_id: str
    ) -> DocumentDetail:
        """Fetch a document with its topic list.

        Raises:
            DocumentNotFoundError: When no record exists for ``doc_id``.
        """
        return await self._queries.get_document(doc_id)

    async def list_documents(
        self,
        app_id: str,
        project_id: str,
        category_id: str | None = None,
        *,
        page: int = 1,
        page_size: int = 20,
        sort_by: str = "created_at",
        sort_order: str = "desc",
    ) -> DocumentListResult:
        """Paginated document list with per-document topic counts."""
        return await self._queries.list_documents(
            app_id,
            project_id,
            category_id,
            page=page,
            page_size=page_size,
            sort_by=sort_by,
            sort_order=sort_order,
        )

    async def get_topic(
        self, topic_id: str, app_id: str, project_id: str
    ) -> TopicDetail:
        """Fetch a single topic with full content.

        Raises:
            TopicNotFoundError: When no record exists for ``topic_id``.
        """
        return await self._queries.get_topic(topic_id)

    async def list_categories(
        self, app_id: str, project_id: str
    ) -> list[CategoryOverview]:
        """Taxonomy categories (created on first use) with document counts."""
        return await self._queries.list_categories(app_id, project_id)

    async def search_knowledge(
        self,
        *,
        query: str,
        method: KnowledgeMethod = "hybrid",
        top_k: int = 10,
        score_threshold: float | None = None,
        include_content: bool = False,
        app_id: str = "default",
        project_id: str = "default",
    ) -> SearchKnowledgeResult:
        """Search knowledge topics by keyword, vector, or hybrid retrieval.

        Raises:
            ProviderNotConfiguredError: Embedding or rerank is not configured.
        """
        return await self._searcher.search(
            query=query,
            method=method,
            top_k=top_k,
            score_threshold=score_threshold,
            include_content=include_content,
            app_id=app_id,
            project_id=project_id,
        )

"""Knowledge document CRUD + search use cases, behind :class:`KnowledgeService`.

Methods (one per use case):
    create_document   — full upload-document pipeline.
    get_document      — fetch document detail with topic list.
    list_documents    — paginated document listing.
    get_topic         — fetch a single topic with content.
    delete_document   — remove the document's markdown (cascade clears state).
    replace_document  — atomic replace (the store restores on failure).
    patch_document    — update mutable document metadata fields.
    list_categories   — taxonomy categories with document counts.
    search_knowledge  — knowledge retrieval pipeline (keyword / vector / hybrid).

Storage is reached through two ports: :class:`KnowledgeStore` (the markdown
truth: document trees, taxonomy, uploaded originals) and
:class:`KnowledgeCatalog` (the document / topic records the cascade keeps).
Markdown is the truth: when the catalog has not caught up with a document
yet, reads that must still work (delete, patch) fall back to the store.
"""

from __future__ import annotations

import asyncio
import dataclasses
import re
import time
from collections.abc import Awaitable, Callable, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Protocol
from uuid import uuid4

from everalgo.rank import acategory_retrieve
from everalgo.types import (
    CategorySpec,
    KnowledgeMemory,
    ParsedContent,
    RawFile,
)

from everos.component.utils.datetime import get_utc_now
from everos.core.errors import (
    DocumentNotFoundError,
    DuplicateDocumentError,
    ExtractionEmptyError,
    LLMNotConfiguredError,
    ProviderNotConfiguredError,
    TopicNotFoundError,
)
from everos.core.observability.logging import get_logger
from everos.memory.mappers import knowledge as knowledge_mapper
from everos.memory.models import KnowledgeDocument, KnowledgeTopic
from everos.memory.models import KnowledgeMemory as DomainKnowledgeMemory
from everos.memory.ports import (
    Filter,
    KnowledgeCatalog,
    KnowledgeStore,
    Page,
    WrittenDocument,
    all_of,
    eq,
)
from everos.memory.search.callbacks import build_fetched_rerank_fn

if TYPE_CHECKING:
    from everos.component.embedding import EmbeddingProvider
    from everos.component.rerank import RerankProvider
    from everos.config.settings import KnowledgeSearchSettings
    from everos.memory.search.recall import KnowledgeMethod, KnowledgeTopicRecaller

logger = get_logger(__name__)

_FALLBACK_CATEGORY = "Others"
_DOC_ID_PREFIX = "d_"
_DOC_ID_HEX_LEN = 12
_MAX_MINT_RETRIES = 5
_SCOPE_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_.\-@+]+$")
_KNOWLEDGE_FEATURE = "knowledge"


class KnowledgeExtractor(Protocol):
    """Structural type for the algo extractor.

    Matches ``everalgo.knowledge.KnowledgeExtractor.aextract`` so
    callers can pass any compatible implementation (including test
    doubles).
    """

    async def aextract(
        self,
        parsed: ParsedContent,
        *,
        doc_id: str,
        title: str,
        categories: list[CategorySpec] | None = None,
        category_id: str = "",
    ) -> list[KnowledgeMemory]: ...


@dataclasses.dataclass(frozen=True)
class CreateDocumentResult:
    """Returned by :func:`create_document` on success."""

    doc_id: str
    category_id: str
    topic_count: int
    source_name: str | None
    md_path: str
    original_file_path: str | None = None


@dataclasses.dataclass(frozen=True)
class TopicOverview:
    """Minimal topic summary embedded in :class:`DocumentDetail`."""

    topic_id: str
    topic_name: str
    topic_path: str
    depth: int
    summary: str


@dataclasses.dataclass(frozen=True)
class DocumentDetail:
    """Full document record returned by :func:`get_document`."""

    doc_id: str
    category_id: str
    title: str
    summary: str
    source_name: str | None
    source_type: str | None
    original_file_path: str | None
    topics: list[TopicOverview]
    created_at: datetime
    updated_at: datetime


@dataclasses.dataclass(frozen=True)
class DocumentOverviewItem:
    """One row in a paginated document list."""

    doc_id: str
    category_id: str
    title: str
    topic_count: int
    created_at: datetime


@dataclasses.dataclass(frozen=True)
class DocumentListResult:
    """Paginated document list returned by :func:`list_documents`."""

    documents: list[DocumentOverviewItem]
    total: int
    page: int
    page_size: int


@dataclasses.dataclass(frozen=True)
class TopicDetail:
    """Full topic record returned by :func:`get_topic`."""

    topic_id: str
    doc_id: str
    category_id: str
    topic_name: str
    topic_path: str
    depth: int
    summary: str
    content: str
    content_labels: list[str]
    parent_topic_id: str | None
    children_topic_ids: list[str]
    created_at: datetime
    updated_at: datetime


@dataclasses.dataclass(frozen=True)
class CategoryOverview:
    """One taxonomy category with its document count."""

    category_id: str
    description: str
    document_count: int


@dataclasses.dataclass(frozen=True)
class DeleteResult:
    """Result of :func:`delete_document`."""

    doc_id: str
    deleted_topics: int


@dataclasses.dataclass(frozen=True)
class PatchResult:
    """Result of :func:`patch_document`."""

    doc_id: str
    updated_fields: list[str]
    updated_at: datetime


@dataclasses.dataclass(frozen=True)
class DocumentContext:
    """L1 document metadata attached to every :class:`SearchHit`."""

    doc_id: str
    title: str
    summary: str


@dataclasses.dataclass(frozen=True)
class SearchHit:
    """One ranked result from :func:`search_knowledge`."""

    topic_id: str
    category_id: str
    topic_name: str
    topic_path: str
    depth: int
    summary: str
    content: str | None
    score: float
    retrieval_method: str
    source: str | None
    document: DocumentContext


@dataclasses.dataclass(frozen=True)
class SearchKnowledgeResult:
    """Envelope returned by :func:`search_knowledge`."""

    hits: list[SearchHit]
    total: int
    took_ms: float


# ── Search filter ────────────────────────────────────────────────────────────


def _validate_scope_id(value: str, name: str) -> None:
    """Reject scope ids with characters that could break an index filter.

    Raises:
        ValueError: If the value is empty or contains invalid characters.
    """
    if not value or not _SCOPE_ID_PATTERN.match(value):
        raise ValueError(f"{name} contains invalid characters: {value!r}")


def compile_knowledge_where(app_id: str, project_id: str) -> Filter:
    """Build a backend-neutral filter scoped to the given tenant.

    Raises:
        ValueError: If either id contains invalid characters.
    """
    _validate_scope_id(app_id, "app_id")
    _validate_scope_id(project_id, "project_id")
    return all_of(eq("app_id", app_id), eq("project_id", project_id))


# ── Facade ───────────────────────────────────────────────────────────────────


class KnowledgeService:
    """Knowledge document CRUD and search, over the ports it is built with.

    Args:
        store: The markdown truth of documents, topics and the taxonomy.
        catalog: The document / topic records the cascade keeps.
        extractor: Topic extractor for uploads; ``None`` when no LLM is
            configured, in which case create / replace raise.
        embedding: Query embedder for search; ``None`` when not configured.
        reranker: Reranker for search; ``None`` when not configured.
        recaller: Recall over the knowledge-topic index.
        search_config: The ``[knowledge.search]`` section.
        parse_file: Parses an uploaded non-text file with the multimodal
            LLM the application was built with.
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
    ) -> None:
        self._store = store
        self._catalog = catalog
        self._extractor = extractor
        self._embedding = embedding
        self._reranker = reranker
        self._recaller = recaller
        self._search_config = search_config
        self._parse_file = parse_file

    async def parse_file(self, raw_file: RawFile) -> ParsedContent:
        """Parse an uploaded file with the application's multimodal parser.

        Raises:
            LLMNotConfiguredError: ``[multimodal]`` is not configured.
            UnsupportedModalityError: The parser does not support the format.
        """
        return await self._parse_file(raw_file)

    # ── Upload ───────────────────────────────────────────────────────────

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
        extractor = self._require_extractor()
        doc_id = doc_id or await self._mint_doc_id()
        if await self._catalog.document_exists(doc_id):
            raise DuplicateDocumentError(
                f"Document {doc_id!r} already exists; use PUT to replace"
            )
        memories = await self._extract(
            extractor, parsed, doc_id, title, app_id, project_id, category_id
        )
        written = await self._store.write_document(
            app_id,
            project_id,
            memories,
            source_name=source_name,
            source_type=source_type,
            original=file_content,
        )
        return _created(doc_id, memories, source_name, written)

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
        """Replace a document atomically — old data preserved on failure.

        Extraction runs before anything on disk changes; the store then
        swaps the document's files and brings the old ones back if the
        write fails. Catalog rows are upserted by the cascade, so nothing is
        deleted from it here.

        Raises:
            LLMNotConfiguredError: No LLM is configured to extract topics.
            DocumentNotFoundError: When ``doc_id`` does not exist.
            ExtractionEmptyError: When extraction produces no memories.
        """
        extractor = self._require_extractor()
        previous = await self._catalog.get_document(doc_id)
        if previous is None:
            raise DocumentNotFoundError(f"Document {doc_id!r} not found")
        memories = await self._extract(
            extractor, parsed, doc_id, title, app_id, project_id, category_id
        )
        written = await self._store.replace_document(
            previous,
            memories,
            source_name=source_name,
            source_type=source_type,
            original=file_content,
        )
        return _created(doc_id, memories, source_name, written)

    # ── Reads ────────────────────────────────────────────────────────────

    async def get_document(
        self, doc_id: str, app_id: str, project_id: str
    ) -> DocumentDetail:
        """Fetch a document with its topic list.

        Raises:
            DocumentNotFoundError: When no record exists for ``doc_id``.
        """
        document = await self._catalog.get_document(doc_id)
        if document is None:
            raise DocumentNotFoundError(f"Document {doc_id!r} not found")
        topics = await self._catalog.topics_of(doc_id)
        original = await self._store.original_file(document)
        return DocumentDetail(
            doc_id=document.doc_id,
            category_id=document.category_id,
            title=document.title,
            summary=document.summary,
            source_name=document.source_name,
            source_type=document.source_type,
            original_file_path=original,
            topics=[_topic_overview(t) for t in topics],
            created_at=_recorded(document.created_at),
            updated_at=_recorded(document.updated_at),
        )

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
        listed = await self._catalog.list_documents(
            app_id,
            project_id,
            category_id=category_id,
            page=Page(
                sort_by=sort_by,
                number=page,
                size=page_size,
                descending=sort_order.lower() != "asc",
            ),
        )
        counts = await asyncio.gather(
            *(self._catalog.count_topics(d.doc_id) for d in listed.items)
        )
        return DocumentListResult(
            documents=[
                DocumentOverviewItem(
                    doc_id=d.doc_id,
                    category_id=d.category_id,
                    title=d.title,
                    topic_count=count,
                    created_at=_recorded(d.created_at),
                )
                for d, count in zip(listed.items, counts, strict=True)
            ],
            total=listed.total,
            page=page,
            page_size=page_size,
        )

    async def get_topic(
        self, topic_id: str, app_id: str, project_id: str
    ) -> TopicDetail:
        """Fetch a single topic with full content.

        Raises:
            TopicNotFoundError: When no record exists for ``topic_id``.
        """
        topics = await self._catalog.get_topics([topic_id])
        if not topics:
            raise TopicNotFoundError(f"Topic {topic_id!r} not found")
        t = topics[0]
        return TopicDetail(
            topic_id=t.topic_id,
            doc_id=t.doc_id,
            category_id=t.category_id,
            topic_name=t.topic_name,
            topic_path=t.topic_path,
            depth=t.depth,
            summary=t.summary,
            content=t.content,
            content_labels=t.content_labels,
            parent_topic_id=t.parent_topic_id,
            children_topic_ids=t.children_topic_ids,
            created_at=t.created_at,
            updated_at=t.updated_at,
        )

    async def list_categories(
        self, app_id: str, project_id: str
    ) -> list[CategoryOverview]:
        """Taxonomy categories with per-category document counts.

        Creates the taxonomy with its defaults if it does not exist yet.
        """
        categories = await self._store.categories(app_id, project_id)
        counts = await self._catalog.count_by_category(app_id, project_id)
        return [
            CategoryOverview(
                category_id=c.id,
                description=c.description,
                document_count=counts.get(c.id, 0),
            )
            for c in categories
        ]

    # ── Writes ───────────────────────────────────────────────────────────

    async def delete_document(
        self, doc_id: str, app_id: str, project_id: str
    ) -> DeleteResult:
        """Remove the document's markdown; the cascade clears its state.

        Idempotent: ``deleted_topics=0`` when neither the catalog nor the
        disk has the document. The catalog trails the markdown by seconds,
        so a document created a moment ago is deleted from disk all the
        same -- otherwise the cascade would index it right back in.
        """
        document = await self._catalog.get_document(doc_id)
        if document is None:
            removed = await self._store.delete_document(
                app_id, project_id, doc_id, md_path=None
            )
            logger.info(
                "document deleted", doc_id=doc_id, topic_count=removed, indexed=False
            )
            return DeleteResult(doc_id=doc_id, deleted_topics=removed)

        topic_count = await self._catalog.count_topics(doc_id)
        await self._store.delete_document(
            document.app_id, document.project_id, doc_id, md_path=document.md_path
        )
        logger.info("document deleted", doc_id=doc_id, topic_count=topic_count)
        return DeleteResult(doc_id=doc_id, deleted_topics=topic_count)

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
            DocumentNotFoundError: When neither the catalog nor the markdown
                has ``doc_id``.
        """
        current = await self._catalog.get_document(
            doc_id
        ) or await self._store.read_document(app_id, project_id, doc_id)
        if current is None:
            raise DocumentNotFoundError(f"Document {doc_id!r} not found")

        updated = current.model_copy(
            update={
                "title": current.title if title is None else title,
                "category_id": current.category_id
                if category_id is None
                else category_id,
            }
        )
        changed = [
            name
            for name in ("title", "category_id")
            if getattr(updated, name) != getattr(current, name)
        ]
        if changed:
            recorded = await self._store.update_document(
                updated, previous_category_id=current.category_id
            )
            await self._catalog.record_document(recorded)
        return PatchResult(
            doc_id=doc_id, updated_fields=changed, updated_at=get_utc_now()
        )

    # ── Search ───────────────────────────────────────────────────────────

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

        Recall feeds everalgo's category-aware retrieval: the recall pool's
        category mass is rolled up, the pool is reranked on the topics' full
        content (read from the catalog), and a small category boost breaks
        ties.

        Raises:
            ProviderNotConfiguredError: Embedding or rerank is not configured.
        """
        t0 = time.monotonic()
        where = compile_knowledge_where(app_id, project_id)
        embedder, reranker = self._require_search_providers()
        vector = await embedder.embed(query)

        retriever = self._recaller.retriever(where, method=method, vector=vector)
        contents: dict[str, str] = {}

        async def passages(topic_ids: Sequence[str]) -> list[str]:
            topics = await self._catalog.get_topics(topic_ids)
            contents.update({t.topic_id: t.content for t in topics})
            return [contents.get(tid, "") for tid in topic_ids]

        config = self._search_config
        ranked = await acategory_retrieve(
            query,
            base_retrieve=retriever,
            rerank_fn=build_fetched_rerank_fn(reranker, passages=passages),
            recall_n=config.recall_n,
            rerank_n=config.rerank_n,
            mass_top_m=config.mass_top_m,
            lam=config.lam,
            top_n=min(top_k, config.top_k_cap),
        )
        if score_threshold is not None:
            ranked = [c for c in ranked if c.score >= score_threshold]

        documents = {
            d.doc_id: d
            for d in await self._catalog.get_documents(
                {retriever.topics_by_id[c.id].doc_id for c in ranked}
            )
        }
        hits = [
            _search_hit(
                retriever.topics_by_id[c.id],
                score=c.score,
                source=None if c.source == "other" else c.source,
                method=method,
                document=documents.get(retriever.topics_by_id[c.id].doc_id),
                content=contents.get(c.id, "") if include_content else None,
            )
            for c in ranked
        ]

        took_ms = (time.monotonic() - t0) * 1000
        logger.info(
            "knowledge_search_complete",
            method=method,
            query_len=len(query),
            hits=len(hits),
            took_ms=round(took_ms, 1),
        )
        return SearchKnowledgeResult(hits=hits, total=len(hits), took_ms=took_ms)

    # ── Helpers ──────────────────────────────────────────────────────────

    async def _extract(
        self,
        extractor: KnowledgeExtractor,
        parsed: ParsedContent,
        doc_id: str,
        title: str,
        app_id: str,
        project_id: str,
        category_id: str | None,
    ) -> list[DomainKnowledgeMemory]:
        """Extract the topic tree, classifying into the project's taxonomy.

        Raises:
            ExtractionEmptyError: When the extractor produces no memories.
        """
        categories = await self._store.categories(app_id, project_id)
        extracted = await extractor.aextract(
            parsed,
            doc_id=doc_id,
            title=title,
            categories=[knowledge_mapper.category_to_algo(c) for c in categories],
            category_id=category_id or "",
        )
        if not extracted:
            raise ExtractionEmptyError(
                f"Extractor returned no memories for doc_id={doc_id!r}"
            )
        return [_with_category(knowledge_mapper.memory_from_algo(m)) for m in extracted]

    async def _mint_doc_id(self) -> str:
        """Generate a unique ``d_<hex12>`` document id.

        Retries up to ``_MAX_MINT_RETRIES`` times if the id already exists
        (astronomically unlikely with 48-bit hex, but defensive).
        """
        for _ in range(_MAX_MINT_RETRIES):
            candidate = f"{_DOC_ID_PREFIX}{uuid4().hex[:_DOC_ID_HEX_LEN]}"
            if not await self._catalog.document_exists(candidate):
                return candidate
        return f"{_DOC_ID_PREFIX}{uuid4().hex}"

    def _require_extractor(self) -> KnowledgeExtractor:
        if self._extractor is None:
            raise LLMNotConfiguredError(
                "knowledge uploads need an LLM; set [llm] api_key and base_url"
            )
        return self._extractor

    def _require_search_providers(self) -> tuple[EmbeddingProvider, RerankProvider]:
        """The embedding + rerank providers, which search cannot run without.

        The HTTP router already gates both capabilities; this guards any
        caller that bypasses it.

        Raises:
            ProviderNotConfiguredError: When either provider is missing.
        """
        if self._embedding is None:
            raise ProviderNotConfiguredError(
                provider="embedding", feature=_KNOWLEDGE_FEATURE
            )
        if self._reranker is None:
            raise ProviderNotConfiguredError(
                provider="rerank", feature=_KNOWLEDGE_FEATURE
            )
        return self._embedding, self._reranker


# ── Mapping to results ───────────────────────────────────────────────────────


def _with_category(memory: DomainKnowledgeMemory) -> DomainKnowledgeMemory:
    """An unclassified node falls back to the ``Others`` category."""
    if memory.category_id:
        return memory
    return memory.model_copy(update={"category_id": _FALLBACK_CATEGORY})


def _created(
    doc_id: str,
    memories: Sequence[DomainKnowledgeMemory],
    source_name: str | None,
    written: WrittenDocument,
) -> CreateDocumentResult:
    category_id = memories[0].category_id
    topic_count = sum(1 for m in memories if m.topic_index != 0)
    logger.info(
        "document created",
        doc_id=doc_id,
        category_id=category_id,
        topic_count=topic_count,
    )
    return CreateDocumentResult(
        doc_id=doc_id,
        category_id=category_id,
        topic_count=topic_count,
        source_name=source_name,
        md_path=written.location,
        original_file_path=written.original_file_path,
    )


def _topic_overview(topic: KnowledgeTopic) -> TopicOverview:
    return TopicOverview(
        topic_id=topic.topic_id,
        topic_name=topic.topic_name,
        topic_path=topic.topic_path,
        depth=topic.depth,
        summary=topic.summary,
    )


def _search_hit(
    topic: KnowledgeTopic,
    *,
    score: float,
    source: str | None,
    method: str,
    document: KnowledgeDocument | None,
    content: str | None,
) -> SearchHit:
    return SearchHit(
        topic_id=topic.topic_id,
        category_id=topic.category_id,
        topic_name=topic.topic_name,
        topic_path=topic.topic_path,
        depth=topic.depth,
        summary=topic.summary,
        content=content,
        score=score,
        retrieval_method=method,
        source=source,
        document=DocumentContext(
            doc_id=document.doc_id if document else "",
            title=document.title if document else "",
            summary=document.summary if document else "",
        ),
    )


def _recorded(stamp: datetime | None) -> datetime:
    """A catalog timestamp; the catalog always sets one."""
    if stamp is None:
        raise ValueError("catalog document without a timestamp")
    return stamp

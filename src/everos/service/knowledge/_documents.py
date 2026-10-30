"""The knowledge write use cases: create, replace, delete and patch a document.

An upload is size-checked before it is read, decoded as UTF-8 when it is
plain text (or no parser is installed) and parsed otherwise; extraction runs
before anything on disk changes. Markdown is the truth: when the catalog has
not caught up with a document yet, delete and patch fall back to the store.
"""

from __future__ import annotations

__all__ = ["DocumentManager"]

from collections.abc import Awaitable, Callable, Sequence
from uuid import uuid4

from everalgo.types import ParsedContent, RawFile

from everos.core.errors import (
    DocumentNotFoundError,
    DuplicateDocumentError,
    ExtractionEmptyError,
    InvalidInputError,
)
from everos.core.observability.logging import get_logger
from everos.core.utils.datetime import get_utc_now
from everos.memory.mappers import knowledge as knowledge_mapper
from everos.memory.models import KnowledgeMemory
from everos.memory.ports import KnowledgeCatalog, KnowledgeStore, WrittenDocument

from ._models import (
    CreateDocumentResult,
    DeleteResult,
    KnowledgeExtractor,
    PatchResult,
)
from ._upload import (
    KnowledgeUpload,
    decode_utf8,
    extension_of,
    looks_like_utf8_text,
    require_content,
)

logger = get_logger("everos.service.knowledge")

_FALLBACK_CATEGORY = "Others"
_DOC_ID_PREFIX = "d_"
_DOC_ID_HEX_LEN = 12
_MAX_MINT_RETRIES = 5


class DocumentManager:
    """Runs the document write use cases over the knowledge ports.

    Args:
        store: The markdown truth of documents, topics and the taxonomy.
        catalog: The document / topic records the cascade keeps.
        extractor: Topic extractor.
        parse_file: Parses an uploaded non-text file.
        parser_available: Whether the optional parser is installed.
        max_upload_bytes: The largest upload accepted.
    """

    def __init__(
        self,
        *,
        store: KnowledgeStore,
        catalog: KnowledgeCatalog,
        extractor: KnowledgeExtractor,
        parse_file: Callable[[RawFile], Awaitable[ParsedContent]],
        parser_available: bool,
        max_upload_bytes: int,
    ) -> None:
        self._store = store
        self._catalog = catalog
        self._extractor = extractor
        self._parse_file = parse_file
        self._parser_available = parser_available
        self._max_upload_bytes = max_upload_bytes

    async def read_upload(self, upload: KnowledgeUpload) -> tuple[ParsedContent, bytes]:
        """Refuse an oversized upload before reading it, then parse it.

        Plain text skips the parser even when it is installed.

        Raises:
            InvalidInputError: The upload is too large or has no content.
            UnsupportedModalityError: The file cannot be parsed.
        """
        self._reject_oversized(upload.size)
        raw = await upload.read()
        if looks_like_utf8_text(upload) or not self._parser_available:
            return decode_utf8(raw), raw
        parsed = await self._parse_file(
            RawFile(
                content=raw,
                mime=upload.content_type or "",
                extension=extension_of(upload.filename),
            )
        )
        require_content(parsed)
        return parsed, raw

    async def create(
        self,
        *,
        doc_id: str | None,
        parsed: ParsedContent,
        title: str,
        app_id: str,
        project_id: str,
        source_name: str | None,
        source_type: str | None,
        category_id: str | None,
        file_content: bytes | None,
    ) -> CreateDocumentResult:
        """Extract ``parsed`` into a new document and write it."""
        doc_id = doc_id or await self._mint_doc_id()
        if await self._catalog.document_exists(doc_id):
            raise DuplicateDocumentError(
                f"Document {doc_id!r} already exists; use PUT to replace"
            )
        memories = await self._extract(
            parsed, doc_id, title, app_id, project_id, category_id
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

    async def replace(
        self,
        *,
        doc_id: str,
        parsed: ParsedContent,
        title: str,
        app_id: str,
        project_id: str,
        source_name: str | None,
        source_type: str | None,
        category_id: str | None,
        file_content: bytes | None,
    ) -> CreateDocumentResult:
        """Extract first, then let the store swap the document's files.

        Catalog rows are upserted by the cascade, so nothing is deleted from
        it here.
        """
        previous = await self._catalog.get_document(doc_id)
        if previous is None:
            raise DocumentNotFoundError(f"Document {doc_id!r} not found")
        memories = await self._extract(
            parsed, doc_id, title, app_id, project_id, category_id
        )
        written = await self._store.replace_document(
            previous,
            memories,
            source_name=source_name,
            source_type=source_type,
            original=file_content,
        )
        return _created(doc_id, memories, source_name, written)

    async def delete(self, doc_id: str, app_id: str, project_id: str) -> DeleteResult:
        """Remove the document's markdown; the cascade clears its state.

        The catalog trails the markdown by seconds, so a document created a
        moment ago is deleted from disk all the same -- otherwise the cascade
        would index it right back in.
        """
        document = await self._catalog.get_document(doc_id)
        if document is None:
            removed = await self._store.delete_document(app_id, project_id, doc_id)
            logger.info(
                "document deleted", doc_id=doc_id, topic_count=removed, indexed=False
            )
            return DeleteResult(doc_id=doc_id, deleted_topics=removed)

        topic_count = await self._catalog.count_topics(doc_id)
        await self._store.delete_document(document.app_id, document.project_id, doc_id)
        logger.info("document deleted", doc_id=doc_id, topic_count=topic_count)
        return DeleteResult(doc_id=doc_id, deleted_topics=topic_count)

    async def patch(
        self,
        doc_id: str,
        app_id: str,
        project_id: str,
        *,
        title: str | None,
        category_id: str | None,
    ) -> PatchResult:
        """Write the changed metadata to md (truth) and the catalog (now)."""
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
            written = await self._store.update_document(
                updated, previous_category_id=current.category_id
            )
            await self._catalog.record_document(updated, md_path=written.md_path)
        return PatchResult(
            doc_id=doc_id, updated_fields=changed, updated_at=get_utc_now()
        )

    def _reject_oversized(self, size: int | None) -> None:
        if size is not None and size > self._max_upload_bytes:
            limit_mib = self._max_upload_bytes / (1024 * 1024)
            raise InvalidInputError(
                f"Uploaded file exceeds the {limit_mib:.1f} MiB limit."
            )

    async def _extract(
        self,
        parsed: ParsedContent,
        doc_id: str,
        title: str,
        app_id: str,
        project_id: str,
        category_id: str | None,
    ) -> list[KnowledgeMemory]:
        """Extract the topic tree, classifying into the project's taxonomy.

        Raises:
            ExtractionEmptyError: When the extractor produces no memories.
        """
        categories = await self._store.categories(app_id, project_id)
        extracted = await self._extractor.aextract(
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


def _with_category(memory: KnowledgeMemory) -> KnowledgeMemory:
    """An unclassified node falls back to the ``Others`` category."""
    if memory.category_id:
        return memory
    return memory.model_copy(update={"category_id": _FALLBACK_CATEGORY})


def _created(
    doc_id: str,
    memories: Sequence[KnowledgeMemory],
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

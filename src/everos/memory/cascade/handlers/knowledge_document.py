"""KnowledgeDocument cascade handler — md → SQLite only.

Handles ``index.md`` files inside knowledge document directories
(``knowledge/{category}/{doc_title}/index.md``).  Unlike
:class:`KnowledgeTopicHandler`, this handler writes to **SQLite only**
— there is no LanceDB write, no embedding, and no tokenization.

The document-level index carries title, category, and a short summary;
the summary is the body of the file (plain text, no entries).  Writes
are always upserted — the SQLite upsert is cheap and document metadata
changes infrequently.

md contract:

- ``knowledge/{category}/{doc_title}/index.md`` frontmatter:
  ``type: knowledge_document``, ``id`` (== ``doc_id``),
  ``category_id``, ``title``, ``source_name`` (optional),
  ``source_type`` (optional).
- Body: document summary (plain text).
"""

from __future__ import annotations

from typing import Any

from everos.core.persistence import MarkdownReader, ParsedMarkdown
from everos.infra.persistence.sqlite import (
    DocumentUpsertPayload,
    knowledge_document_repo,
)

from ..types import HandlerOutcome
from ._common import resolve_scope
from .base import Handler


class KnowledgeDocumentHandler(Handler):
    """Cascade handler for ``knowledge/{category}/{doc_title}/index.md``."""

    kind = "knowledge_document"

    async def handle_added_or_modified(self, md_path: str) -> HandlerOutcome:
        absolute = self._deps.memory_root.root / md_path
        parsed = await MarkdownReader.read(absolute)

        if parsed.frontmatter.get("type") != "knowledge_document":
            return HandlerOutcome(
                md_path=md_path,
                kind=self.kind,
                upserted=0,
                deleted=0,
                skipped=1,
            )

        payload = self._build_payload(parsed, md_path)
        await knowledge_document_repo.upsert_from_handler(payload)

        return HandlerOutcome(
            md_path=md_path,
            kind=self.kind,
            upserted=1,
            deleted=0,
            skipped=0,
        )

    def _build_payload(
        self, parsed: ParsedMarkdown, md_path: str
    ) -> DocumentUpsertPayload:
        """Construct the SQLite upsert payload from parsed frontmatter and body."""
        fm: dict[str, Any] = parsed.frontmatter
        app_id, project_id = resolve_scope(md_path)
        source_name = fm.get("source_name")
        source_type = fm.get("source_type")
        return DocumentUpsertPayload(
            doc_id=str(fm["id"]),
            app_id=app_id,
            project_id=project_id,
            category_id=str(fm.get("category_id", "")),
            title=str(fm.get("title", "")),
            summary=parsed.body.strip(),
            source_name=source_name if isinstance(source_name, str) else None,
            source_type=source_type if isinstance(source_type, str) else None,
            md_path=md_path,
        )

    async def handle_deleted(self, md_path: str) -> HandlerOutcome:
        deleted = await knowledge_document_repo.delete_by_md_path(md_path)
        return HandlerOutcome(
            md_path=md_path,
            kind=self.kind,
            upserted=0,
            deleted=deleted,
            skipped=0,
        )

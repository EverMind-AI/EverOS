"""Frontmatter schema for ``knowledge/{category}/{doc_title}/index.md``."""

from __future__ import annotations

from typing import Literal

from everos.core.persistence.markdown import (
    BaseFrontmatter,
    KnowledgeDocumentPathMixin,
    KnowledgeScopedMixin,
)


class KnowledgeDocumentFrontmatter(
    KnowledgeDocumentPathMixin,
    KnowledgeScopedMixin,
    BaseFrontmatter,
):
    """L1 document-level frontmatter (index.md). Body = doc summary."""

    type: Literal["knowledge_document"] = "knowledge_document"
    doc_id: str
    category_id: str
    title: str
    source_name: str | None = None
    source_type: str | None = None

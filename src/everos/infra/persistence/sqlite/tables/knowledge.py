"""``knowledge_documents`` + ``knowledge_topics`` — L1/L2 knowledge metadata.

``KnowledgeDocumentRow`` holds per-document metadata (category, title,
summary, source, md path). ``KnowledgeTopicRow`` holds per-topic content
and tree structure (parent / children, depth, path) derived from the parsed
document outline.

Both tables use ``BaseTable`` so they inherit ``created_at`` / ``updated_at``
with automatic UTC enforcement and ``onupdate`` refresh.
"""

from __future__ import annotations

from sqlalchemy import Index

from everos.core.persistence.sqlite import BaseTable, Field


class KnowledgeDocumentRow(BaseTable, table=True):
    """One row per knowledge document. PK ``doc_id``."""

    __tablename__ = "knowledge_documents"  # type: ignore[assignment]  -- SQLModel tablename typing limitation
    __table_args__ = (
        Index(
            "ix_knowledge_documents_category",
            "app_id",
            "project_id",
            "category_id",
        ),
        Index("ix_knowledge_documents_md_path", "md_path"),
    )

    doc_id: str = Field(primary_key=True)
    app_id: str = Field(default="default")
    project_id: str = Field(default="default")
    category_id: str
    title: str
    summary: str
    source_name: str | None = Field(default=None)
    source_type: str | None = Field(default=None)
    md_path: str


class KnowledgeTopicRow(BaseTable, table=True):
    """One row per topic node within a knowledge document. PK ``node_id``."""

    __tablename__ = "knowledge_topics"  # type: ignore[assignment]  -- SQLModel tablename typing limitation
    __table_args__ = (
        Index("ix_knowledge_topics_doc", "doc_id"),
        Index("ix_knowledge_topics_app_proj", "app_id", "project_id"),
    )

    node_id: str = Field(primary_key=True)
    doc_id: str
    app_id: str = Field(default="default")
    project_id: str = Field(default="default")
    category_id: str
    topic_index: int
    topic_name: str
    topic_path: str
    depth: int
    parent_node_id: str | None = Field(default=None)
    children_node_ids: str | None = Field(default=None)
    """JSON-encoded list[str] of child node ids."""
    summary: str
    content: str
    content_labels: str | None = Field(default=None)
    """JSON-encoded list[str] of content labels / tags."""
    md_path: str

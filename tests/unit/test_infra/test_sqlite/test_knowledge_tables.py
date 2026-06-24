"""SQLite table model validation for knowledge_documents + knowledge_topics."""

from __future__ import annotations

from everos.infra.persistence.sqlite import (
    KnowledgeDocumentRow,
    KnowledgeTopicRow,
)


class TestKnowledgeDocumentRow:
    def test_tablename(self) -> None:
        assert KnowledgeDocumentRow.__tablename__ == "knowledge_documents"

    def test_primary_key(self) -> None:
        pk_cols = [c.name for c in KnowledgeDocumentRow.__table__.primary_key.columns]
        assert pk_cols == ["doc_id"]


class TestKnowledgeTopicRow:
    def test_tablename(self) -> None:
        assert KnowledgeTopicRow.__tablename__ == "knowledge_topics"

    def test_primary_key(self) -> None:
        pk_cols = [c.name for c in KnowledgeTopicRow.__table__.primary_key.columns]
        assert pk_cols == ["node_id"]

    def test_has_content_column(self) -> None:
        cols = {c.name for c in KnowledgeTopicRow.__table__.columns}
        assert "content" in cols
        assert "summary" in cols

    def test_topic_doc_id_has_no_fk(self) -> None:
        """doc_id has no FK — cascade handler ordering is not guaranteed."""
        from sqlmodel import SQLModel

        table = SQLModel.metadata.tables["knowledge_topics"]
        fks = list(table.foreign_key_constraints)
        assert len(fks) == 0

"""Frontmatter parse/dump round-trip for knowledge document + topic."""

from __future__ import annotations

from everos.infra.persistence.markdown import (
    KnowledgeDocumentFrontmatter,
    KnowledgeTopicFrontmatter,
)


class TestKnowledgeDocumentFrontmatter:
    def test_type_literal(self) -> None:
        fm = KnowledgeDocumentFrontmatter(
            type="knowledge_document",
            id="d_abc123000000",
            doc_id="d_abc123000000",
            category_id="Sports",
            title="Olympics Plan",
            schema_version=1,
        )
        assert fm.type == "knowledge_document"
        assert fm.id == fm.doc_id

    def test_path_glob(self) -> None:
        assert KnowledgeDocumentFrontmatter.path_glob() == "*/*/knowledge/*/*/index.md"

    def test_scope_dir(self) -> None:
        assert KnowledgeDocumentFrontmatter.SCOPE_DIR == "knowledge"

    def test_optional_fields_default_none(self) -> None:
        fm = KnowledgeDocumentFrontmatter(
            type="knowledge_document",
            id="d_abc",
            doc_id="d_abc",
            category_id="Sports",
            title="X",
            schema_version=1,
        )
        assert fm.source_name is None
        assert fm.source_type is None


class TestKnowledgeTopicFrontmatter:
    def test_type_literal(self) -> None:
        fm = KnowledgeTopicFrontmatter(
            type="knowledge_topic",
            id="d_abc_1",
            node_id="d_abc_1",
            doc_id="d_abc",
            category_id="Sports",
            topic_index=1,
            topic_name="Budget",
            topic_path="Olympics > Budget",
            summary="Budget overview.",
            depth=1,
            schema_version=1,
        )
        assert fm.type == "knowledge_topic"
        assert fm.id == fm.node_id

    def test_path_glob(self) -> None:
        assert KnowledgeTopicFrontmatter.path_glob() == "*/*/knowledge/*/*/[0-9]*.md"

    def test_defaults(self) -> None:
        fm = KnowledgeTopicFrontmatter(
            type="knowledge_topic",
            id="d_abc_1",
            node_id="d_abc_1",
            doc_id="d_abc",
            category_id="Sports",
            topic_index=1,
            topic_name="Budget",
            topic_path="Olympics > Budget",
            summary="Summary.",
            depth=1,
            schema_version=1,
        )
        assert fm.parent_node_id is None
        assert fm.children_node_ids == []
        assert fm.content_labels == []

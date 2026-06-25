"""Tests for knowledge-scoped frontmatter mixins."""

from __future__ import annotations

from everos.core.persistence.markdown.frontmatter import (
    BaseFrontmatter,
    KnowledgeDocumentPathMixin,
    KnowledgeScopedMixin,
    KnowledgeTopicPathMixin,
)


class _DocFm(KnowledgeDocumentPathMixin, KnowledgeScopedMixin, BaseFrontmatter):
    type: str = "knowledge_document"
    id: str = "d_abc"
    schema_version: int = 1


class _TopicFm(KnowledgeTopicPathMixin, KnowledgeScopedMixin, BaseFrontmatter):
    type: str = "knowledge_topic"
    id: str = "d_abc_1"
    schema_version: int = 1


class TestKnowledgeScopedMixin:
    def test_scope_dir_is_knowledge(self) -> None:
        assert _DocFm.SCOPE_DIR == "knowledge"

    def test_doc_path_glob_matches_index_md(self) -> None:
        glob = _DocFm.path_glob()
        assert glob == "*/*/knowledge/*/*/index.md"

    def test_topic_path_glob_matches_numbered_md(self) -> None:
        glob = _TopicFm.path_glob()
        assert glob == "*/*/knowledge/*/*/[0-9]*.md"

    def test_doc_glob_has_app_project_prefix(self) -> None:
        from pathlib import PurePosixPath

        glob = _DocFm.path_glob()
        path = "default_app/default_project/knowledge/Sports/Olympics/index.md"
        assert PurePosixPath(path).match(glob)

    def test_topic_glob_matches_numbered_topic(self) -> None:
        from pathlib import PurePosixPath

        glob = _TopicFm.path_glob()
        path = "default_app/default_project/knowledge/Sports/Olympics/1_Budget.md"
        assert PurePosixPath(path).match(glob)

    def test_topic_glob_does_not_match_index(self) -> None:
        from pathlib import PurePosixPath

        glob = _TopicFm.path_glob()
        path = "default_app/default_project/knowledge/Sports/Olympics/index.md"
        assert not PurePosixPath(path).match(glob)

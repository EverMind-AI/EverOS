"""Knowledge kinds registered in CASCADE registry."""

from __future__ import annotations

from everos.memory.cascade.registry import KIND_REGISTRY, match_kind


class TestKnowledgeKindRegistration:
    def test_knowledge_document_registered(self) -> None:
        names = [k.name for k in KIND_REGISTRY]
        assert "knowledge_document" in names

    def test_knowledge_topic_registered(self) -> None:
        names = [k.name for k in KIND_REGISTRY]
        assert "knowledge_topic" in names

    def test_match_index_md(self) -> None:
        spec = match_kind(
            "default_app/default_project/knowledge/Sports/Olympics/index.md"
        )
        assert spec is not None
        assert spec.name == "knowledge_document"

    def test_match_topic_md(self) -> None:
        spec = match_kind(
            "default_app/default_project/knowledge/Sports/Olympics/1_Budget.md"
        )
        assert spec is not None
        assert spec.name == "knowledge_topic"

    def test_knowledge_document_has_no_index_schema(self) -> None:
        spec = next(k for k in KIND_REGISTRY if k.name == "knowledge_document")
        assert spec.index_schema is None
        assert spec.index_repo is None

    def test_knowledge_topic_has_index_schema(self) -> None:
        spec = next(k for k in KIND_REGISTRY if k.name == "knowledge_topic")
        assert spec.index_schema is not None
        assert spec.index_repo is not None

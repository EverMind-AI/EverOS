"""KnowledgeTopic LanceDB table schema validation."""

from __future__ import annotations

from everos.infra.persistence.lancedb import KnowledgeTopic


class TestKnowledgeTopicSchema:
    def test_table_name(self) -> None:
        assert KnowledgeTopic.TABLE_NAME == "knowledge_topic"

    def test_bm25_fields_dual_column(self) -> None:
        assert KnowledgeTopic.BM25_FIELDS == ["summary_tokens", "content_tokens"]

    def test_has_required_fields(self) -> None:
        fields = set(KnowledgeTopic.model_fields.keys())
        required = {
            "id",
            "doc_id",
            "category_id",
            "app_id",
            "project_id",
            "topic_name",
            "topic_path",
            "depth",
            "parent_node_id",
            "summary",
            "summary_tokens",
            "content_tokens",
            "content_labels",
            "md_path",
            "content_sha256",
            "vector",
        }
        assert required.issubset(fields), f"Missing: {required - fields}"

    def test_arrow_schema_has_utc_timestamps(self) -> None:
        schema = KnowledgeTopic.to_arrow_schema()
        assert schema is not None

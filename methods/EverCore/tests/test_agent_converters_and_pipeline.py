"""
Supplementary tests for Agent Case/Skill pipeline gaps.

Covers:
- ES Converters: AgentCaseConverter, AgentSkillConverter
- Milvus Converters: AgentCaseMilvusConverter, AgentSkillMilvusConverter
- _trigger_agent_skill_extraction: full pipeline with mocked dependencies
- AgentSkillExtractor edge cases: confidence boundary, mixed operations, embedding failure
- AgentCaseExtractor: pre-compress with large content, _compress_tool_chunk

Usage:
    PYTHONPATH=src pytest tests/test_agent_converters_and_pipeline.py -v
"""

import json
import pytest
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any, Dict, List

from bson import ObjectId


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_case_record(**overrides):
    """Create a mock AgentCaseRecord for converter tests."""
    defaults = dict(
        id=ObjectId(),
        user_id="user_001",
        group_id="group_001",
        session_id="sess_001",
        timestamp=datetime(2025, 3, 1, 12, 0, 0),
        task_intent="Build a REST API for user management",
        approach="1. Design the schema\n2. Implement CRUD endpoints\n3. Add validation",
        quality_score=0.85,
        parent_type="memcell",
        parent_id="evt_001",
        vector=[0.1, 0.2, 0.3],
        vector_model="text-embedding-3-small",
        key_insight="",
        created_at=datetime(2025, 3, 1, 12, 0, 0),
        updated_at=datetime(2025, 3, 1, 12, 5, 0),
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _mock_skill_record(**overrides):
    """Create a mock AgentSkillRecord for converter tests."""
    defaults = dict(
        id=ObjectId(),
        user_id="user_001",
        group_id="group_001",
        cluster_id="cluster_001",
        name="API Development",
        description="Build REST APIs with proper error handling and validation",
        content="## Steps\n1. Design schema\n2. Implement endpoints\n3. Add validation\n4. Write tests\n5. Deploy",
        confidence=0.8,
        maturity_score=0.75,
        vector=[0.1, 0.2, 0.3],
        vector_model="text-embedding-3-small",
        source_case_ids=["evt_001"],
        created_at=datetime(2025, 3, 1, 12, 0, 0),
        updated_at=datetime(2025, 3, 1, 12, 5, 0),
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


# ===========================================================================
# AgentCaseConverter tests
# ===========================================================================


class TestAgentCaseConverter:
    """Tests for AgentCaseConverter.from_mongo and _build_search_content."""

    def test_from_mongo_basic(self):
        from infra_layer.adapters.out.search.elasticsearch.converter.agent_case_converter import (
            AgentCaseConverter,
        )

        doc = _mock_case_record()
        es_doc = AgentCaseConverter.from_mongo(doc)
        assert es_doc.task_intent == "Build a REST API for user management"
        assert (
            es_doc.approach
            == "1. Design the schema\n2. Implement CRUD endpoints\n3. Add validation"
        )
        assert es_doc.user_id == "user_001"
        assert es_doc.parent_type == "memcell"

    def test_from_mongo_none_raises(self):
        from infra_layer.adapters.out.search.elasticsearch.converter.agent_case_converter import (
            AgentCaseConverter,
        )

        with pytest.raises(ValueError, match="cannot be empty"):
            AgentCaseConverter.from_mongo(None)

    def test_from_mongo_none_fields_fallback(self):
        from infra_layer.adapters.out.search.elasticsearch.converter.agent_case_converter import (
            AgentCaseConverter,
        )

        doc = _mock_case_record(task_intent=None, approach=None)
        es_doc = AgentCaseConverter.from_mongo(doc)
        assert es_doc.task_intent == ""
        assert es_doc.approach == ""

    def test_build_search_content_deduplicates(self):
        from infra_layer.adapters.out.search.elasticsearch.converter.agent_case_converter import (
            AgentCaseConverter,
        )

        doc = _mock_case_record(
            task_intent="API design patterns",
            approach="API design patterns for REST services",
        )
        content = AgentCaseConverter._build_search_content(doc)
        # Words should be deduplicated
        assert len(content) == len(set(content))

    def test_build_search_content_empty_fallback(self):
        from infra_layer.adapters.out.search.elasticsearch.converter.agent_case_converter import (
            AgentCaseConverter,
        )

        doc = _mock_case_record(task_intent="", approach="")
        content = AgentCaseConverter._build_search_content(doc)
        assert content == [""]

    def test_build_search_content_only_task_intent_fallback(self):
        """When filtering removes all words, fallback to raw task_intent."""
        from infra_layer.adapters.out.search.elasticsearch.converter.agent_case_converter import (
            AgentCaseConverter,
        )

        doc = _mock_case_record(task_intent="a", approach="")
        content = AgentCaseConverter._build_search_content(doc)
        # Single char filtered by min_length=2, fallback to raw
        assert content == ["a"]


# ===========================================================================
# AgentSkillConverter tests
# ===========================================================================


class TestAgentSkillConverter:
    """Tests for AgentSkillConverter.from_mongo and _build_search_content."""

    def test_from_mongo_basic(self):
        from infra_layer.adapters.out.search.elasticsearch.converter.agent_skill_converter import (
            AgentSkillConverter,
        )

        doc = _mock_skill_record()
        es_doc = AgentSkillConverter.from_mongo(doc)
        assert es_doc.name == "API Development"
        assert es_doc.cluster_id == "cluster_001"
        assert es_doc.confidence == 0.8
        assert es_doc.maturity_score == 0.75

    def test_from_mongo_none_raises(self):
        from infra_layer.adapters.out.search.elasticsearch.converter.agent_skill_converter import (
            AgentSkillConverter,
        )

        with pytest.raises(ValueError):
            AgentSkillConverter.from_mongo(None)

    def test_from_mongo_none_fields(self):
        from infra_layer.adapters.out.search.elasticsearch.converter.agent_skill_converter import (
            AgentSkillConverter,
        )

        doc = _mock_skill_record(name=None, description=None, content=None)
        es_doc = AgentSkillConverter.from_mongo(doc)
        assert es_doc.name == ""
        assert es_doc.description == ""
        assert es_doc.content == ""

    def test_build_search_content_combines_all_fields(self):
        from infra_layer.adapters.out.search.elasticsearch.converter.agent_skill_converter import (
            AgentSkillConverter,
        )

        doc = _mock_skill_record()
        content = AgentSkillConverter._build_search_content(doc)
        assert len(content) > 0
        assert len(content) == len(set(content))  # deduplicated

    def test_build_search_content_fallback_to_description(self):
        """When filtering removes all words, fallback to description."""
        from infra_layer.adapters.out.search.elasticsearch.converter.agent_skill_converter import (
            AgentSkillConverter,
        )

        doc = _mock_skill_record(name="", description="x", content="")
        content = AgentSkillConverter._build_search_content(doc)
        assert content == ["x"]

    def test_build_search_content_all_empty(self):
        from infra_layer.adapters.out.search.elasticsearch.converter.agent_skill_converter import (
            AgentSkillConverter,
        )

        doc = _mock_skill_record(name="", description="", content="")
        content = AgentSkillConverter._build_search_content(doc)
        assert content == [""]


# ===========================================================================
# AgentCaseMilvusConverter tests
# ===========================================================================


class TestAgentCaseMilvusConverter:
    """Tests for AgentCaseMilvusConverter.from_mongo."""

    def test_basic_conversion(self):
        from infra_layer.adapters.out.search.milvus.converter.agent_case_milvus_converter import (
            AgentCaseMilvusConverter,
        )

        doc = _mock_case_record()
        entity = AgentCaseMilvusConverter.from_mongo(doc)
        assert entity["id"] == str(doc.id)
        assert entity["vector"] == [0.1, 0.2, 0.3]
        assert entity["user_id"] == "user_001"
        assert entity["task_intent"] == "Build a REST API for user management"
        assert entity["parent_type"] == "memcell"

    def test_none_raises(self):
        from infra_layer.adapters.out.search.milvus.converter.agent_case_milvus_converter import (
            AgentCaseMilvusConverter,
        )

        with pytest.raises(ValueError):
            AgentCaseMilvusConverter.from_mongo(None)

    def test_none_timestamp_defaults_to_zero(self):
        from infra_layer.adapters.out.search.milvus.converter.agent_case_milvus_converter import (
            AgentCaseMilvusConverter,
        )

        doc = _mock_case_record(timestamp=None)
        entity = AgentCaseMilvusConverter.from_mongo(doc)
        assert entity["timestamp"] == 0

    def test_none_vector_defaults_to_empty_list(self):
        from infra_layer.adapters.out.search.milvus.converter.agent_case_milvus_converter import (
            AgentCaseMilvusConverter,
        )

        doc = _mock_case_record(vector=None)
        entity = AgentCaseMilvusConverter.from_mongo(doc)
        assert entity["vector"] == []

    def test_long_text_truncated(self):
        from infra_layer.adapters.out.search.milvus.converter.agent_case_milvus_converter import (
            AgentCaseMilvusConverter,
        )

        doc = _mock_case_record(task_intent="x" * 10000)
        entity = AgentCaseMilvusConverter.from_mongo(doc)
        assert len(entity["task_intent"]) == 5000
        assert "search_content" not in entity

    def test_none_fields_default_to_empty_string(self):
        from infra_layer.adapters.out.search.milvus.converter.agent_case_milvus_converter import (
            AgentCaseMilvusConverter,
        )

        doc = _mock_case_record(
            user_id=None,
            group_id=None,
            task_intent=None,
            approach=None,
            parent_type=None,
            parent_id=None,
            vector_model=None,
        )
        entity = AgentCaseMilvusConverter.from_mongo(doc)
        assert entity["user_id"] == ""
        assert entity["group_id"] == ""
        assert entity["task_intent"] == ""
        assert entity["parent_type"] == ""

    def test_no_metadata_field(self):
        from infra_layer.adapters.out.search.milvus.converter.agent_case_milvus_converter import (
            AgentCaseMilvusConverter,
        )

        doc = _mock_case_record()
        entity = AgentCaseMilvusConverter.from_mongo(doc)
        assert "metadata" not in entity


# ===========================================================================
# AgentSkillMilvusConverter tests
# ===========================================================================


class TestAgentSkillMilvusConverter:
    """Tests for AgentSkillMilvusConverter.from_mongo."""

    def test_basic_conversion(self):
        from infra_layer.adapters.out.search.milvus.converter.agent_skill_milvus_converter import (
            AgentSkillMilvusConverter,
        )

        doc = _mock_skill_record()
        entity = AgentSkillMilvusConverter.from_mongo(doc)
        assert entity["id"] == str(doc.id)
        assert entity["vector"] == [0.1, 0.2, 0.3]
        assert entity["cluster_id"] == "cluster_001"
        assert entity["maturity_score"] == 0.75
        assert entity["confidence"] == 0.8
        assert "API Development" in entity["content"]

    def test_none_raises(self):
        from infra_layer.adapters.out.search.milvus.converter.agent_skill_milvus_converter import (
            AgentSkillMilvusConverter,
        )

        with pytest.raises(ValueError):
            AgentSkillMilvusConverter.from_mongo(None)

    def test_none_vector_defaults_to_empty(self):
        from infra_layer.adapters.out.search.milvus.converter.agent_skill_milvus_converter import (
            AgentSkillMilvusConverter,
        )

        doc = _mock_skill_record(vector=None)
        entity = AgentSkillMilvusConverter.from_mongo(doc)
        assert entity["vector"] == []

    def test_no_timestamp_fields(self):
        from infra_layer.adapters.out.search.milvus.converter.agent_skill_milvus_converter import (
            AgentSkillMilvusConverter,
        )

        doc = _mock_skill_record()
        entity = AgentSkillMilvusConverter.from_mongo(doc)
        assert "created_at" not in entity
        assert "updated_at" not in entity

    def test_content_field_is_name_plus_description(self):
        from infra_layer.adapters.out.search.milvus.converter.agent_skill_milvus_converter import (
            AgentSkillMilvusConverter,
        )

        doc = _mock_skill_record(name="Skill Name", description="Skill Description")
        entity = AgentSkillMilvusConverter.from_mongo(doc)
        assert "Skill Name" in entity["content"]
        assert "Skill Description" in entity["content"]

    def test_search_content_removed(self):
        from infra_layer.adapters.out.search.milvus.converter.agent_skill_milvus_converter import (
            AgentSkillMilvusConverter,
        )

        doc = _mock_skill_record()
        entity = AgentSkillMilvusConverter.from_mongo(doc)
        assert "search_content" not in entity

    def test_no_metadata_field(self):
        from infra_layer.adapters.out.search.milvus.converter.agent_skill_milvus_converter import (
            AgentSkillMilvusConverter,
        )

        doc = _mock_skill_record()
        entity = AgentSkillMilvusConverter.from_mongo(doc)
        assert "metadata" not in entity

    def test_none_fields_default_to_empty(self):
        from infra_layer.adapters.out.search.milvus.converter.agent_skill_milvus_converter import (
            AgentSkillMilvusConverter,
        )

        doc = _mock_skill_record(
            name=None,
            description=None,
            content=None,
            user_id=None,
            group_id=None,
            cluster_id=None,
            vector_model=None,
        )
        entity = AgentSkillMilvusConverter.from_mongo(doc)
        assert entity["user_id"] == ""
        assert entity["cluster_id"] == ""

    def test_confidence_is_top_level_field(self):
        """confidence must be a top-level entity field for Milvus filtering."""
        from infra_layer.adapters.out.search.milvus.converter.agent_skill_milvus_converter import (
            AgentSkillMilvusConverter,
        )

        doc = _mock_skill_record(confidence=0.65)
        entity = AgentSkillMilvusConverter.from_mongo(doc)
        assert "confidence" in entity
        assert entity["confidence"] == 0.65

    def test_confidence_zero(self):
        """confidence=0.0 should be preserved, not dropped."""
        from infra_layer.adapters.out.search.milvus.converter.agent_skill_milvus_converter import (
            AgentSkillMilvusConverter,
        )

        doc = _mock_skill_record(confidence=0.0)
        entity = AgentSkillMilvusConverter.from_mongo(doc)
        assert entity["confidence"] == 0.0

    def test_confidence_boundary_values(self):
        """confidence boundary values 0.0 and 1.0 should be accepted."""
        from infra_layer.adapters.out.search.milvus.converter.agent_skill_milvus_converter import (
            AgentSkillMilvusConverter,
        )

        for val in (0.0, 0.1, 0.5, 0.95, 1.0):
            doc = _mock_skill_record(confidence=val)
            entity = AgentSkillMilvusConverter.from_mongo(doc)
            assert entity["confidence"] == val


class TestAgentSkillCollectionSchema:
    """Tests for AgentSkillCollection Milvus schema definition."""

    def test_schema_has_confidence_field(self):
        """Schema must include a confidence FLOAT field for filter expressions."""
        from infra_layer.adapters.out.search.milvus.memory.agent_skill_collection import (
            AgentSkillCollection,
        )
        from pymilvus import DataType

        schema = AgentSkillCollection._SCHEMA
        field_map = {f.name: f for f in schema.fields}

        assert "confidence" in field_map, "confidence field missing from Milvus schema"
        assert field_map["confidence"].dtype == DataType.FLOAT

    def test_confidence_field_has_index(self):
        """confidence field should have an AUTOINDEX for efficient filtering."""
        from infra_layer.adapters.out.search.milvus.memory.agent_skill_collection import (
            AgentSkillCollection,
        )

        indexed_fields = [cfg.field_name for cfg in AgentSkillCollection._INDEX_CONFIGS]
        assert "confidence" in indexed_fields

    def test_schema_field_parity_with_converter(self):
        """All top-level keys produced by the converter must exist in the schema."""
        from infra_layer.adapters.out.search.milvus.converter.agent_skill_milvus_converter import (
            AgentSkillMilvusConverter,
        )
        from infra_layer.adapters.out.search.milvus.memory.agent_skill_collection import (
            AgentSkillCollection,
        )

        doc = _mock_skill_record()
        entity = AgentSkillMilvusConverter.from_mongo(doc)
        schema_fields = {f.name for f in AgentSkillCollection._SCHEMA.fields}

        for key in entity:
            assert (
                key in schema_fields
            ), f"Converter produces key '{key}' not in Collection schema"


# ===========================================================================
# AgentSkillExtractor additional edge cases
# ===========================================================================


class TestAgentSkillExtractorEdgeCases:
    """Additional edge case tests for AgentSkillExtractor."""

    @pytest.mark.asyncio
    async def test_apply_update_confidence_exactly_0_1_auto_deletes(self):
        """Confidence == 0.1 should NOT auto-delete (threshold is < 0.1)."""
        from memory_layer.memory_extractor.agent_skill_extractor import (
            AgentSkillExtractor,
            SkillExtractionResult,
        )

        extractor = AgentSkillExtractor(
            llm_provider=MagicMock(),
            success_extract_prompt="",
            failure_extract_prompt="",
        )
        repo = AsyncMock()
        repo.update_skill_by_id = AsyncMock(return_value=True)
        existing = [
            SimpleNamespace(
                id="s1",
                name="Skill",
                description="Desc",
                content="Content",
                confidence=0.5,
                vector=[0.1],
                vector_model="m",
                source_case_ids=[],
                maturity_score=0.7,
                updated_at=None,
            )
        ]
        result = SkillExtractionResult()

        with (
            patch.object(
                extractor,
                "_compute_embedding",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch.object(
                extractor,
                "_evaluate_maturity",
                new_callable=AsyncMock,
                return_value=0.7,
            ),
        ):
            op = {"action": "update", "index": 0, "data": {"confidence": 0.1}}
            success = await extractor._apply_update(op, existing, repo, result)

        # 0.1 is NOT < 0.1, so should update, not delete
        assert success is True
        assert len(result.deleted_ids) == 0
        assert len(result.updated_records) == 1

    @pytest.mark.asyncio
    async def test_apply_update_confidence_0_09_retires(self):
        """Confidence 0.09 < 0.1, skill is retired via update (not soft-deleted)."""
        from memory_layer.memory_extractor.agent_skill_extractor import (
            AgentSkillExtractor,
            SkillExtractionResult,
        )

        extractor = AgentSkillExtractor(
            llm_provider=MagicMock(),
            success_extract_prompt="",
            failure_extract_prompt="",
        )
        repo = AsyncMock()
        repo.update_skill_by_id = AsyncMock(return_value=True)
        existing = [
            SimpleNamespace(
                id="s1",
                name="Skill",
                description="Desc",
                confidence=0.5,
                source_case_ids=[],
                maturity_score=0.7,
                updated_at=None,
                content="steps",
                cluster_id="c1",
            )
        ]
        result = SkillExtractionResult()
        op = {"action": "update", "index": 0, "data": {"confidence": 0.09}}
        success = await extractor._apply_update(op, existing, repo, result)
        assert success is True
        assert "s1" in result.deleted_ids
        # Verify it updated confidence, not soft-deleted
        repo.update_skill_by_id.assert_called_once()
        call_args = repo.update_skill_by_id.call_args[0]
        assert call_args[1] == {"confidence": 0.09}

    @pytest.mark.asyncio
    async def test_extract_and_save_mixed_operations(self):
        """Test add + update + none in a single extraction."""
        from memory_layer.memory_extractor.agent_skill_extractor import (
            AgentSkillExtractor,
        )

        response = json.dumps(
            {
                "operations": [
                    {"action": "update", "index": 0, "data": {"confidence": 0.9}},
                    {
                        "action": "add",
                        "data": {
                            "name": "New Skill",
                            "description": "Description",
                            "content": "## Steps for the new task at hand\n1. Step one here\n2. Step two here\n3. Step three here\n4. Step four here\n5. Step five here",
                            "confidence": 0.6,
                        },
                    },
                    {"action": "none"},
                ],
                "update_note": "Updated existing skill and added new one",
            }
        )

        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(return_value=response)
        extractor = AgentSkillExtractor(
            llm_provider=mock_llm,
            success_extract_prompt="{new_case_json}{existing_skills_json}",
            failure_extract_prompt="{new_case_json}{existing_skills_json}",
        )

        existing_skill = SimpleNamespace(
            id="s1",
            name="Old Skill",
            description="Old Desc",
            content="Old Content",
            confidence=0.5,
            vector=[0.1],
            vector_model="m",
            source_case_ids=[],
            maturity_score=0.7,
            updated_at=None,
        )

        repo = AsyncMock()
        repo.update_skill_by_id = AsyncMock(return_value=True)
        repo.save_skill = AsyncMock(side_effect=lambda rec: rec)

        case = SimpleNamespace(
            task_intent="Build something",
            approach="Steps here",
            quality_score=0.8,
            timestamp=datetime(2025, 1, 1),
        )

        with (
            patch.object(
                extractor,
                "_compute_embedding",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch.object(
                extractor,
                "_evaluate_maturity",
                new_callable=AsyncMock,
                return_value=0.75,
            ),
        ):

            import infra_layer.adapters.out.persistence.document.memory.agent_skill as skill_mod

            with patch.object(
                skill_mod, "AgentSkillRecord", return_value=MagicMock(id="new_s")
            ):
                result = await extractor.extract_and_save(
                    cluster_id="c1",
                    group_id="g1",
                    new_case_records=[case],
                    existing_skill_records=[existing_skill],
                    skill_repo=repo,
                    user_id="u1",
                )

        # 1 update + 1 add + 1 none
        assert len(result.added_records) == 1
        assert len(result.updated_records) == 1
        assert len(result.deleted_ids) == 0

    @pytest.mark.asyncio
    async def test_extract_and_save_empty_operations_list(self):
        """LLM returns valid JSON but empty operations list."""
        from memory_layer.memory_extractor.agent_skill_extractor import (
            AgentSkillExtractor,
        )

        response = json.dumps({"operations": [], "update_note": "nothing to do"})
        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(return_value=response)
        extractor = AgentSkillExtractor(
            llm_provider=mock_llm,
            success_extract_prompt="{new_case_json}{existing_skills_json}",
            failure_extract_prompt="{new_case_json}{existing_skills_json}",
        )
        case = SimpleNamespace(
            task_intent="Task",
            approach="Steps",
            quality_score=0.8,
            timestamp=datetime(2025, 1, 1),
        )
        repo = AsyncMock()
        result = await extractor.extract_and_save(
            cluster_id="c1",
            group_id="g1",
            new_case_records=[case],
            existing_skill_records=[],
            skill_repo=repo,
        )
        assert result.added_records == []
        assert result.updated_records == []


# ===========================================================================
# AgentCaseExtractor additional edge cases
# ===========================================================================


class TestAgentCaseExtractorEdgeCases:
    """Additional edge case tests for AgentCaseExtractor."""

    @pytest.fixture(autouse=True)
    def _mock_tokenizer(self):
        import tiktoken

        encoding = tiktoken.get_encoding("o200k_base")
        mock_factory = MagicMock()
        mock_factory.get_tokenizer_from_tiktoken.return_value = encoding
        with patch(
            "memory_layer.memory_extractor.agent_case_extractor.get_bean_by_type",
            return_value=mock_factory,
        ):
            yield

    @pytest.mark.asyncio
    async def test_compress_tool_chunk_valid_response(self):
        from memory_layer.memory_extractor.agent_case_extractor import (
            AgentCaseExtractor,
        )

        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "c1", "function": {"name": "search", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "content": "result", "tool_call_id": "c1"},
        ]
        response = json.dumps({"compressed_messages": messages})
        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(return_value=response)

        extractor = AgentCaseExtractor(
            llm_provider=mock_llm, tool_pre_compress_prompt="{messages_json}{new_count}"
        )
        result = await extractor._compress_tool_chunk(messages)
        assert result is not None
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_compress_tool_chunk_invalid_count_retries(self):
        """When compressed count doesn't match, retries and returns None."""
        from memory_layer.memory_extractor.agent_case_extractor import (
            AgentCaseExtractor,
        )

        messages = [{"role": "tool", "content": "r", "tool_call_id": "c1"}]
        # Return wrong count
        response = json.dumps(
            {"compressed_messages": [{"role": "tool"}, {"role": "tool"}]}
        )
        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(return_value=response)

        extractor = AgentCaseExtractor(
            llm_provider=mock_llm, tool_pre_compress_prompt="{messages_json}{new_count}"
        )
        result = await extractor._compress_tool_chunk(messages)
        assert result is None
        assert mock_llm.generate.call_count == 2

    @pytest.mark.asyncio
    async def test_compress_tool_chunk_llm_exception(self):
        from memory_layer.memory_extractor.agent_case_extractor import (
            AgentCaseExtractor,
        )

        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(side_effect=Exception("fail"))
        extractor = AgentCaseExtractor(
            llm_provider=mock_llm, tool_pre_compress_prompt="{messages_json}{new_count}"
        )
        result = await extractor._compress_tool_chunk(
            [{"role": "tool", "content": "r"}]
        )
        assert result is None

    def test_calc_tool_content_size_user_message(self):
        from memory_layer.memory_extractor.agent_case_extractor import (
            AgentCaseExtractor,
        )

        msg = {"role": "user", "content": "hello world"}
        assert AgentCaseExtractor._calc_tool_content_size(msg) == 0

    def test_calc_tool_content_size_tool_message(self):
        from memory_layer.memory_extractor.agent_case_extractor import (
            AgentCaseExtractor,
        )

        msg = {"role": "tool", "content": "some tool output here"}
        size = AgentCaseExtractor._calc_tool_content_size(msg)
        assert size > 0

    def test_calc_tool_content_size_assistant_with_tool_calls(self):
        from memory_layer.memory_extractor.agent_case_extractor import (
            AgentCaseExtractor,
        )

        msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "c1", "function": {"name": "f", "arguments": '{"key": "value"}'}}
            ],
        }
        size = AgentCaseExtractor._calc_tool_content_size(msg)
        assert size > 0

    def test_calc_tool_content_size_assistant_no_tools(self):
        from memory_layer.memory_extractor.agent_case_extractor import (
            AgentCaseExtractor,
        )

        msg = {"role": "assistant", "content": "Final answer here"}
        assert AgentCaseExtractor._calc_tool_content_size(msg) == 0

    def test_json_default_datetime(self):
        from memory_layer.memory_extractor.agent_case_extractor import (
            AgentCaseExtractor,
        )

        dt = datetime(2025, 6, 15, 10, 30, 0)
        result = AgentCaseExtractor._json_default(dt)
        assert result == "2025-06-15T10:30:00"

    def test_json_default_other(self):
        from memory_layer.memory_extractor.agent_case_extractor import (
            AgentCaseExtractor,
        )

        result = AgentCaseExtractor._json_default(42)
        assert result == "42"


# ===========================================================================
# _trigger_agent_skill_extraction pipeline test
# ===========================================================================


class TestTriggerAgentSkillExtraction:
    """Tests for _trigger_agent_skill_extraction from mem_memorize.py.

    _trigger_agent_skill_extraction uses local imports, so we need to patch
    at the source module level where the imports actually resolve.
    """

    @pytest.mark.asyncio
    async def test_full_pipeline_add_and_sync(self):
        """Test complete pipeline: extract skill -> sync to Milvus + ES."""
        from memory_layer.memory_extractor.agent_skill_extractor import (
            SkillExtractionResult,
        )
        from api_specs.memory_types import MemCell, RawDataType, AgentCase
        from api_specs.memory_models import MemoryType

        memcell = MemCell(
            user_id_list=["u1"],
            original_data=[
                {"message": {"role": "user", "content": "hi", "sender_id": "u1"}}
            ],
            timestamp=datetime(2025, 3, 1),
            event_id="evt_001",
            group_id="g1",
            type=RawDataType.AGENTCONVERSATION,
        )
        agent_case = AgentCase(
            memory_type=MemoryType.AGENT_CASE,
            user_id="u1",
            timestamp=datetime(2025, 3, 1),
            task_intent="Build API",
            approach="Steps here",
            quality_score=0.8,
        )

        mock_added_record = MagicMock(id=ObjectId(), vector=[0.1, 0.2])
        extraction_result = SkillExtractionResult(
            added_records=[mock_added_record], updated_records=[], deleted_ids=[]
        )

        mock_skill_repo = AsyncMock()
        mock_skill_repo.get_by_cluster_id = AsyncMock(return_value=[])

        mock_milvus_repo = AsyncMock()
        mock_es_repo = AsyncMock()

        mock_milvus_converter_result = {
            "vector": [0.1, 0.2],
            "id": str(mock_added_record.id),
        }

        # Patch at the source modules where local imports resolve
        with (
            patch("core.lock.redis_distributed_lock.distributed_lock") as mock_lock,
            patch("core.di.get_bean_by_type") as mock_get_bean,
            patch(
                "memory_layer.llm.llm_provider.build_default_provider"
            ) as mock_provider,
            patch(
                "memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor"
            ) as mock_extractor_cls,
            patch(
                "infra_layer.adapters.out.search.milvus.converter.agent_skill_milvus_converter.AgentSkillMilvusConverter"
            ) as mock_milvus_conv,
            patch(
                "infra_layer.adapters.out.search.elasticsearch.converter.agent_skill_converter.AgentSkillConverter"
            ) as mock_es_conv,
        ):

            # Setup distributed lock as async context manager
            mock_lock_ctx = AsyncMock()
            mock_lock_ctx.__aenter__ = AsyncMock(return_value=True)
            mock_lock_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_lock.return_value = mock_lock_ctx

            # Setup bean resolution
            def get_bean_side_effect(cls):
                name = cls.__name__ if hasattr(cls, '__name__') else str(cls)
                if "SkillRaw" in name:
                    return mock_skill_repo
                if "SkillMilvus" in name and "Repository" in name:
                    return mock_milvus_repo
                if "SkillEs" in name and "Repository" in name:
                    return mock_es_repo
                return MagicMock()

            mock_get_bean.side_effect = get_bean_side_effect

            mock_provider.return_value = MagicMock()

            mock_extractor = AsyncMock()
            mock_extractor.extract_and_save = AsyncMock(return_value=extraction_result)
            mock_extractor_cls.return_value = mock_extractor

            mock_milvus_conv.from_mongo.return_value = mock_milvus_converter_result
            mock_es_conv.from_mongo.return_value = MagicMock()

            # Force re-import to pick up patches on local imports
            import importlib
            import biz_layer.mem_memorize as memorize_mod

            importlib.reload(memorize_mod)

            await memorize_mod._trigger_agent_skill_extraction(
                group_id="g1",
                cluster_id="cluster_001",
                memcell=memcell,
                agent_case=agent_case,
            )

            # Verify extractor was called
            mock_extractor.extract_and_save.assert_called_once()

    @pytest.mark.asyncio
    async def test_extraction_exception_handled(self):
        """Exceptions in extraction are caught and logged, not propagated."""
        from api_specs.memory_types import MemCell, RawDataType, AgentCase
        from api_specs.memory_models import MemoryType

        memcell = MemCell(
            user_id_list=["u1"],
            original_data=[
                {"message": {"role": "user", "content": "hi", "sender_id": "u1"}}
            ],
            timestamp=datetime(2025, 3, 1),
            event_id="evt_001",
            group_id="g1",
            type=RawDataType.AGENTCONVERSATION,
        )
        agent_case = AgentCase(
            memory_type=MemoryType.AGENT_CASE,
            user_id="u1",
            timestamp=datetime(2025, 3, 1),
            task_intent="t",
            approach="a",
            quality_score=0.5,
        )

        # Patch top-level exception: the outer try-except should catch this
        with patch(
            "core.lock.redis_distributed_lock.distributed_lock",
            side_effect=Exception("Lock init failed"),
        ):

            import importlib
            import biz_layer.mem_memorize as memorize_mod

            importlib.reload(memorize_mod)

            # Should not raise - exception is caught in the outer try-except
            await memorize_mod._trigger_agent_skill_extraction(
                group_id="g1", cluster_id="c1", memcell=memcell, agent_case=agent_case
            )


# ===========================================================================
# _extract_user_id_from_memcell additional edge cases
# ===========================================================================


class TestExtractUserIdEdgeCases:
    """Additional edge cases for _extract_user_id_from_memcell."""

    def test_multiple_users_returns_first(self):
        from api_specs.memory_types import MemCell, RawDataType
        from biz_layer.mem_db_operations import _extract_user_id_from_memcell

        memcell = MemCell(
            user_id_list=["u1", "u2"],
            original_data=[
                {
                    "message": {
                        "role": "user",
                        "content": "first",
                        "sender_id": "first_user",
                    }
                },
                {"message": {"role": "assistant", "content": "resp"}},
                {
                    "message": {
                        "role": "user",
                        "content": "second",
                        "sender_id": "second_user",
                    }
                },
            ],
            timestamp=datetime(2025, 1, 1),
            type=RawDataType.AGENTCONVERSATION,
        )
        assert _extract_user_id_from_memcell(memcell) == "first_user"

    def test_sender_id_empty_string_returns_none(self):
        from api_specs.memory_types import MemCell, RawDataType
        from biz_layer.mem_db_operations import _extract_user_id_from_memcell

        memcell = MemCell(
            user_id_list=["u1"],
            original_data=[
                {"message": {"role": "user", "content": "hi", "sender_id": ""}}
            ],
            timestamp=datetime(2025, 1, 1),
            type=RawDataType.AGENTCONVERSATION,
        )
        # Empty string is falsy, should return None
        assert _extract_user_id_from_memcell(memcell) is None

    def test_mixed_wrapped_and_bare(self):
        """Some items have 'message' key, some don't."""
        from api_specs.memory_types import MemCell
        from biz_layer.mem_db_operations import _extract_user_id_from_memcell

        memcell = MemCell(
            user_id_list=["u1"],
            original_data=[
                {"role": "assistant", "content": "hi"},  # bare, no sender_id
                {
                    "message": {
                        "role": "user",
                        "content": "hello",
                        "sender_id": "u_bare",
                    }
                },
            ],
            timestamp=datetime(2025, 1, 1),
        )
        assert _extract_user_id_from_memcell(memcell) == "u_bare"


# ---------------------------------------------------------------------------
# _should_skip_atomic_fact_for_agent tests
# ---------------------------------------------------------------------------


class TestShouldSkipAtomicFactForAgent:
    """Tests for _should_skip_atomic_fact_for_agent heuristic.

    Skip when: has tool calls AND cumulative assistant response >= 1000 chars.
    """

    def _make_memcell(self, messages):
        from api_specs.memory_types import MemCell, RawDataType

        return MemCell(
            user_id_list=["u1"],
            original_data=messages,
            timestamp=datetime(2025, 1, 1),
            type=RawDataType.AGENTCONVERSATION,
        )

    def test_no_tool_calls_returns_false(self):
        """Without tool calls, never skip regardless of response length."""
        from biz_layer.mem_memorize import _should_skip_atomic_fact_for_agent

        memcell = self._make_memcell(
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "x" * 2000},
            ]
        )
        assert _should_skip_atomic_fact_for_agent(memcell) is False

    def test_tool_calls_with_short_response_returns_false(self):
        """With tool calls but short assistant response, do not skip."""
        from biz_layer.mem_memorize import _should_skip_atomic_fact_for_agent

        memcell = self._make_memcell(
            [
                {"role": "user", "content": "do something"},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]},
                {"role": "tool", "content": "result"},
                {"role": "assistant", "content": "Done."},
            ]
        )
        assert _should_skip_atomic_fact_for_agent(memcell) is False

    def test_tool_calls_with_long_response_returns_true(self):
        """With tool calls and long cumulative assistant response, skip."""
        from biz_layer.mem_memorize import _should_skip_atomic_fact_for_agent

        memcell = self._make_memcell(
            [
                {"role": "user", "content": "do something"},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]},
                {"role": "tool", "content": "result"},
                {"role": "assistant", "content": "x" * 1000},
            ]
        )
        assert _should_skip_atomic_fact_for_agent(memcell) is True

    def test_cumulative_across_multiple_responses(self):
        """Multiple assistant responses should be summed."""
        from biz_layer.mem_memorize import _should_skip_atomic_fact_for_agent

        memcell = self._make_memcell(
            [
                {"role": "user", "content": "step 1"},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]},
                {"role": "tool", "content": "result 1"},
                {"role": "assistant", "content": "x" * 500},
                {"role": "user", "content": "step 2"},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "2"}]},
                {"role": "tool", "content": "result 2"},
                {"role": "assistant", "content": "y" * 501},
            ]
        )
        # 500 + 501 = 1001 >= 1000
        assert _should_skip_atomic_fact_for_agent(memcell) is True

    def test_exactly_999_chars_returns_false(self):
        """Boundary: 999 chars should not skip."""
        from biz_layer.mem_memorize import _should_skip_atomic_fact_for_agent

        memcell = self._make_memcell(
            [
                {"role": "user", "content": "do something"},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]},
                {"role": "tool", "content": "result"},
                {"role": "assistant", "content": "x" * 999},
            ]
        )
        assert _should_skip_atomic_fact_for_agent(memcell) is False

    def test_wrapped_message_format(self):
        """Works with wrapped {"message": {...}} format."""
        from biz_layer.mem_memorize import _should_skip_atomic_fact_for_agent

        memcell = self._make_memcell(
            [
                {"message": {"role": "user", "content": "do something"}},
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{"id": "1"}],
                    }
                },
                {"message": {"role": "tool", "content": "result"}},
                {"message": {"role": "assistant", "content": "x" * 1200}},
            ]
        )
        assert _should_skip_atomic_fact_for_agent(memcell) is True

    def test_tool_role_detected_as_tool_call(self):
        """A 'tool' role message alone counts as having tool calls."""
        from biz_layer.mem_memorize import _should_skip_atomic_fact_for_agent

        memcell = self._make_memcell(
            [
                {"role": "user", "content": "do something"},
                {"role": "tool", "content": "tool output"},
                {"role": "assistant", "content": "x" * 1500},
            ]
        )
        assert _should_skip_atomic_fact_for_agent(memcell) is True

    def test_content_items_list_format(self):
        """Content as v1 API content items list should be handled correctly."""
        from biz_layer.mem_memorize import _should_skip_atomic_fact_for_agent

        memcell = self._make_memcell(
            [
                {"role": "user", "content": "do something"},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]},
                {"role": "tool", "content": "result"},
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "x" * 1000}],
                },
            ]
        )
        assert _should_skip_atomic_fact_for_agent(memcell) is True

    def test_content_items_list_short(self):
        """Short content items list should not skip."""
        from biz_layer.mem_memorize import _should_skip_atomic_fact_for_agent

        memcell = self._make_memcell(
            [
                {"role": "user", "content": "do something"},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]},
                {"role": "tool", "content": "result"},
                {"role": "assistant", "content": [{"type": "text", "text": "Done."}]},
            ]
        )
        assert _should_skip_atomic_fact_for_agent(memcell) is False

    def test_empty_original_data(self):
        """Empty original_data should not skip."""
        from biz_layer.mem_memorize import _should_skip_atomic_fact_for_agent
        from api_specs.memory_types import MemCell, RawDataType

        memcell = MemCell(
            user_id_list=["u1"],
            original_data=[{"role": "user", "content": "hi"}],
            timestamp=datetime(2025, 1, 1),
            type=RawDataType.AGENTCONVERSATION,
        )
        assert _should_skip_atomic_fact_for_agent(memcell) is False


# ===========================================================================
# save_memory_docs - AgentCase branch
# ===========================================================================


def _build_case_get_bean(mock_case_repo, mock_es_repo, mock_milvus_repo):
    """Build a get_bean_by_type side effect that resolves repos by class name."""

    def side_effect(cls):
        name = cls.__name__ if hasattr(cls, "__name__") else str(cls)
        if "CaseRaw" in name:
            return mock_case_repo
        if "CaseEs" in name:
            return mock_es_repo
        if "CaseMilvus" in name and "Repository" in name:
            return mock_milvus_repo
        return MagicMock()

    return side_effect


class TestSaveMemoryDocsAgentCase:
    """Tests for the AgentCase branch in save_memory_docs."""

    @pytest.mark.asyncio
    async def test_agent_case_saved_to_mongo_es_milvus(self):
        """AgentCase docs are saved to MongoDB, synced to ES and Milvus."""
        from api_specs.memory_models import MemoryType
        from biz_layer.mem_memorize import MemoryDocPayload, save_memory_docs

        mock_doc = MagicMock()
        mock_saved_doc = MagicMock()
        mock_saved_doc.event_id = "evt_1"
        mock_saved_doc.id = ObjectId()

        mock_case_repo = AsyncMock()
        mock_case_repo.append_experience = AsyncMock(return_value=mock_saved_doc)
        mock_es_repo = AsyncMock()
        mock_milvus_repo = AsyncMock()

        mock_es_doc = MagicMock()
        mock_milvus_entity = {"vector": [0.1, 0.2], "id": "test"}

        with (
            patch(
                "biz_layer.mem_memorize.get_bean_by_type",
                side_effect=_build_case_get_bean(
                    mock_case_repo, mock_es_repo, mock_milvus_repo
                ),
            ),
            patch(
                "infra_layer.adapters.out.search.elasticsearch.converter.agent_case_converter.AgentCaseConverter.from_mongo",
                return_value=mock_es_doc,
            ),
            patch(
                "infra_layer.adapters.out.search.milvus.converter.agent_case_milvus_converter.AgentCaseMilvusConverter.from_mongo",
                return_value=mock_milvus_entity,
            ),
        ):

            payload = MemoryDocPayload(MemoryType.AGENT_CASE, mock_doc)
            result = await save_memory_docs([payload])

            assert MemoryType.AGENT_CASE in result
            assert len(result[MemoryType.AGENT_CASE]) == 1
            mock_case_repo.append_experience.assert_called_once_with(mock_doc)
            mock_es_repo.create.assert_called_once()
            mock_milvus_repo.insert.assert_called_once()

    @pytest.mark.asyncio
    async def test_agent_case_es_failure_does_not_block_milvus(self):
        """ES sync failure should not prevent Milvus sync."""
        from api_specs.memory_models import MemoryType
        from biz_layer.mem_memorize import MemoryDocPayload, save_memory_docs

        mock_doc = MagicMock()
        mock_saved_doc = MagicMock()
        mock_saved_doc.event_id = "evt_1"
        mock_saved_doc.id = ObjectId()

        mock_case_repo = AsyncMock()
        mock_case_repo.append_experience = AsyncMock(return_value=mock_saved_doc)
        mock_es_repo = AsyncMock()
        mock_milvus_repo = AsyncMock()
        mock_milvus_entity = {"vector": [0.1], "id": "test"}

        with (
            patch(
                "biz_layer.mem_memorize.get_bean_by_type",
                side_effect=_build_case_get_bean(
                    mock_case_repo, mock_es_repo, mock_milvus_repo
                ),
            ),
            patch(
                "infra_layer.adapters.out.search.elasticsearch.converter.agent_case_converter.AgentCaseConverter.from_mongo",
                side_effect=Exception("ES convert error"),
            ),
            patch(
                "infra_layer.adapters.out.search.milvus.converter.agent_case_milvus_converter.AgentCaseMilvusConverter.from_mongo",
                return_value=mock_milvus_entity,
            ),
        ):

            payload = MemoryDocPayload(MemoryType.AGENT_CASE, mock_doc)
            result = await save_memory_docs([payload])

            assert MemoryType.AGENT_CASE in result
            mock_milvus_repo.insert.assert_called_once()

    @pytest.mark.asyncio
    async def test_agent_case_no_vector_skips_milvus(self):
        """AgentCase with no vector should skip Milvus write."""
        from api_specs.memory_models import MemoryType
        from biz_layer.mem_memorize import MemoryDocPayload, save_memory_docs

        mock_doc = MagicMock()
        mock_saved_doc = MagicMock()
        mock_saved_doc.event_id = "evt_1"
        mock_saved_doc.id = ObjectId()

        mock_case_repo = AsyncMock()
        mock_case_repo.append_experience = AsyncMock(return_value=mock_saved_doc)
        mock_es_repo = AsyncMock()
        mock_milvus_repo = AsyncMock()
        mock_es_doc = MagicMock()
        mock_milvus_entity = {"vector": [], "id": "test"}

        with (
            patch(
                "biz_layer.mem_memorize.get_bean_by_type",
                side_effect=_build_case_get_bean(
                    mock_case_repo, mock_es_repo, mock_milvus_repo
                ),
            ),
            patch(
                "infra_layer.adapters.out.search.elasticsearch.converter.agent_case_converter.AgentCaseConverter.from_mongo",
                return_value=mock_es_doc,
            ),
            patch(
                "infra_layer.adapters.out.search.milvus.converter.agent_case_milvus_converter.AgentCaseMilvusConverter.from_mongo",
                return_value=mock_milvus_entity,
            ),
        ):

            payload = MemoryDocPayload(MemoryType.AGENT_CASE, mock_doc)
            result = await save_memory_docs([payload])

            assert MemoryType.AGENT_CASE in result
            mock_milvus_repo.insert.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_payloads_returns_empty(self):
        """No payloads returns empty dict."""
        from biz_layer.mem_memorize import save_memory_docs

        result = await save_memory_docs([])
        assert result == {}

    @pytest.mark.asyncio
    async def test_none_doc_in_payload_skipped(self):
        """Payload with None doc should be skipped."""
        from api_specs.memory_models import MemoryType
        from biz_layer.mem_memorize import MemoryDocPayload, save_memory_docs

        payload = MemoryDocPayload(MemoryType.AGENT_CASE, None)
        result = await save_memory_docs([payload])
        assert MemoryType.AGENT_CASE not in result


# ===========================================================================
# _trigger_agent_skill_extraction - additional coverage
# ===========================================================================


def _make_agent_memcell_for_trigger(**overrides):
    """Create a minimal agent conversation MemCell."""
    from api_specs.memory_types import MemCell, RawDataType

    defaults = dict(
        user_id_list=["u1"],
        original_data=[
            {
                "message": {
                    "role": "user",
                    "content": "Deploy the app",
                    "sender_id": "u1",
                }
            },
            {"message": {"role": "assistant", "content": "Done deploying."}},
        ],
        timestamp=datetime(2025, 6, 1, 10, 0, 0),
        event_id="evt_100",
        group_id="g1",
        type=RawDataType.AGENTCONVERSATION,
    )
    defaults.update(overrides)
    return MemCell(**defaults)


def _make_agent_case_for_trigger(**overrides):
    """Create a minimal AgentCase BO."""
    from api_specs.memory_types import AgentCase
    from api_specs.memory_models import MemoryType

    defaults = dict(
        memory_type=MemoryType.AGENT_CASE,
        user_id="u1",
        timestamp=datetime(2025, 6, 1, 10, 0, 0),
        task_intent="Deploy the application to production",
        approach="1. Build docker image\n2. Push to registry\n3. Deploy to k8s",
        quality_score=0.85,
        vector=[0.1, 0.2, 0.3],
        vector_model="text-embedding-3-small",
    )
    defaults.update(overrides)
    return AgentCase(**defaults)


def _build_skill_trigger_patches(mock_skill_repo, mock_milvus_repo, mock_es_repo):
    """Build a get_bean_by_type side effect for skill extraction tests."""

    def side_effect(cls):
        name = cls.__name__ if hasattr(cls, "__name__") else str(cls)
        if "SkillRaw" in name:
            return mock_skill_repo
        if "SkillMilvus" in name and "Repository" in name:
            return mock_milvus_repo
        if "SkillEs" in name and "Repository" in name:
            return mock_es_repo
        return MagicMock()

    return side_effect


class TestTriggerAgentSkillExtractionGaps:
    """Additional tests for _trigger_agent_skill_extraction."""

    @pytest.mark.asyncio
    async def test_lock_not_acquired_skips_extraction(self):
        """When distributed lock is not acquired, extraction is skipped."""
        memcell = _make_agent_memcell_for_trigger()
        agent_case = _make_agent_case_for_trigger()

        with (
            patch("core.lock.redis_distributed_lock.distributed_lock") as mock_lock,
            patch("core.di.get_bean_by_type") as mock_get_bean,
        ):

            mock_lock_ctx = AsyncMock()
            mock_lock_ctx.__aenter__ = AsyncMock(return_value=False)
            mock_lock_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_lock.return_value = mock_lock_ctx

            import importlib
            import biz_layer.mem_memorize as memorize_mod

            importlib.reload(memorize_mod)

            await memorize_mod._trigger_agent_skill_extraction(
                group_id="g1", cluster_id="c1", memcell=memcell, agent_case=agent_case
            )

    @pytest.mark.asyncio
    async def test_updated_records_delete_old_then_insert_new(self):
        """Updated records should delete old entries and insert new ones in search engines."""
        from memory_layer.memory_extractor.agent_skill_extractor import (
            SkillExtractionResult,
        )

        memcell = _make_agent_memcell_for_trigger()
        agent_case = _make_agent_case_for_trigger()

        updated_record = MagicMock(id=ObjectId(), vector=[0.1, 0.2])
        extraction_result = SkillExtractionResult(
            added_records=[], updated_records=[updated_record], deleted_ids=[]
        )

        mock_skill_repo = AsyncMock()
        mock_skill_repo.get_by_cluster_id = AsyncMock(return_value=[])
        mock_milvus_repo = AsyncMock()
        mock_es_repo = AsyncMock()

        with (
            patch("core.lock.redis_distributed_lock.distributed_lock") as mock_lock,
            patch(
                "core.di.get_bean_by_type",
                side_effect=_build_skill_trigger_patches(
                    mock_skill_repo, mock_milvus_repo, mock_es_repo
                ),
            ),
            patch(
                "memory_layer.llm.llm_provider.build_default_provider",
                return_value=MagicMock(),
            ),
            patch(
                "memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor"
            ) as mock_ext_cls,
            patch(
                "infra_layer.adapters.out.search.milvus.converter.agent_skill_milvus_converter.AgentSkillMilvusConverter"
            ) as mock_milvus_conv,
            patch(
                "infra_layer.adapters.out.search.elasticsearch.converter.agent_skill_converter.AgentSkillConverter"
            ) as mock_es_conv,
        ):

            mock_lock_ctx = AsyncMock()
            mock_lock_ctx.__aenter__ = AsyncMock(return_value=True)
            mock_lock_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_lock.return_value = mock_lock_ctx

            mock_extractor = AsyncMock()
            mock_extractor.extract_and_save = AsyncMock(return_value=extraction_result)
            mock_ext_cls.return_value = mock_extractor

            mock_milvus_conv.from_mongo.return_value = {"vector": [0.1, 0.2], "id": "x"}
            mock_es_conv.from_mongo.return_value = MagicMock()

            import importlib
            import biz_layer.mem_memorize as memorize_mod

            importlib.reload(memorize_mod)

            await memorize_mod._trigger_agent_skill_extraction(
                group_id="g1", cluster_id="c1", memcell=memcell, agent_case=agent_case
            )

            updated_id = str(updated_record.id)
            mock_milvus_repo.delete_by_id.assert_called_with(updated_id)
            mock_milvus_repo.insert.assert_called_once()
            mock_es_repo.delete_by_id.assert_called_with(updated_id)
            mock_es_repo.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_record_without_vector_skipped_in_milvus(self):
        """Records without vector should be skipped in Milvus but still synced to ES."""
        from memory_layer.memory_extractor.agent_skill_extractor import (
            SkillExtractionResult,
        )

        memcell = _make_agent_memcell_for_trigger()
        agent_case = _make_agent_case_for_trigger()

        added_record = MagicMock(id=ObjectId(), vector=None)
        extraction_result = SkillExtractionResult(
            added_records=[added_record], updated_records=[], deleted_ids=[]
        )

        mock_skill_repo = AsyncMock()
        mock_skill_repo.get_by_cluster_id = AsyncMock(return_value=[])
        mock_milvus_repo = AsyncMock()
        mock_es_repo = AsyncMock()

        with (
            patch("core.lock.redis_distributed_lock.distributed_lock") as mock_lock,
            patch(
                "core.di.get_bean_by_type",
                side_effect=_build_skill_trigger_patches(
                    mock_skill_repo, mock_milvus_repo, mock_es_repo
                ),
            ),
            patch(
                "memory_layer.llm.llm_provider.build_default_provider",
                return_value=MagicMock(),
            ),
            patch(
                "memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor"
            ) as mock_ext_cls,
            patch(
                "infra_layer.adapters.out.search.milvus.converter.agent_skill_milvus_converter.AgentSkillMilvusConverter"
            ) as mock_milvus_conv,
            patch(
                "infra_layer.adapters.out.search.elasticsearch.converter.agent_skill_converter.AgentSkillConverter"
            ) as mock_es_conv,
        ):

            mock_lock_ctx = AsyncMock()
            mock_lock_ctx.__aenter__ = AsyncMock(return_value=True)
            mock_lock_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_lock.return_value = mock_lock_ctx

            mock_extractor = AsyncMock()
            mock_extractor.extract_and_save = AsyncMock(return_value=extraction_result)
            mock_ext_cls.return_value = mock_extractor

            mock_milvus_conv.from_mongo.return_value = {"vector": None, "id": "x"}
            mock_es_conv.from_mongo.return_value = MagicMock()

            import importlib
            import biz_layer.mem_memorize as memorize_mod

            importlib.reload(memorize_mod)

            await memorize_mod._trigger_agent_skill_extraction(
                group_id="g1", cluster_id="c1", memcell=memcell, agent_case=agent_case
            )

            mock_milvus_repo.insert.assert_not_called()
            mock_es_repo.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_milvus_failure_does_not_block_es(self):
        """Milvus sync failure should not prevent ES sync."""
        from memory_layer.memory_extractor.agent_skill_extractor import (
            SkillExtractionResult,
        )

        memcell = _make_agent_memcell_for_trigger()
        agent_case = _make_agent_case_for_trigger()

        added_record = MagicMock(id=ObjectId(), vector=[0.1])
        extraction_result = SkillExtractionResult(
            added_records=[added_record], updated_records=[], deleted_ids=[]
        )

        mock_skill_repo = AsyncMock()
        mock_skill_repo.get_by_cluster_id = AsyncMock(return_value=[])
        mock_milvus_repo = AsyncMock()
        mock_milvus_repo.insert = AsyncMock(side_effect=Exception("Milvus down"))
        mock_es_repo = AsyncMock()

        with (
            patch("core.lock.redis_distributed_lock.distributed_lock") as mock_lock,
            patch(
                "core.di.get_bean_by_type",
                side_effect=_build_skill_trigger_patches(
                    mock_skill_repo, mock_milvus_repo, mock_es_repo
                ),
            ),
            patch(
                "memory_layer.llm.llm_provider.build_default_provider",
                return_value=MagicMock(),
            ),
            patch(
                "memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor"
            ) as mock_ext_cls,
            patch(
                "infra_layer.adapters.out.search.milvus.converter.agent_skill_milvus_converter.AgentSkillMilvusConverter"
            ) as mock_milvus_conv,
            patch(
                "infra_layer.adapters.out.search.elasticsearch.converter.agent_skill_converter.AgentSkillConverter"
            ) as mock_es_conv,
        ):

            mock_lock_ctx = AsyncMock()
            mock_lock_ctx.__aenter__ = AsyncMock(return_value=True)
            mock_lock_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_lock.return_value = mock_lock_ctx

            mock_extractor = AsyncMock()
            mock_extractor.extract_and_save = AsyncMock(return_value=extraction_result)
            mock_ext_cls.return_value = mock_extractor

            mock_milvus_conv.from_mongo.return_value = {"vector": [0.1], "id": "x"}
            mock_es_conv.from_mongo.return_value = MagicMock()

            import importlib
            import biz_layer.mem_memorize as memorize_mod

            importlib.reload(memorize_mod)

            await memorize_mod._trigger_agent_skill_extraction(
                group_id="g1", cluster_id="c1", memcell=memcell, agent_case=agent_case
            )

            mock_es_repo.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_es_failure_does_not_raise(self):
        """ES sync failure should be caught and not propagate."""
        from memory_layer.memory_extractor.agent_skill_extractor import (
            SkillExtractionResult,
        )

        memcell = _make_agent_memcell_for_trigger()
        agent_case = _make_agent_case_for_trigger()

        added_record = MagicMock(id=ObjectId(), vector=[0.1])
        extraction_result = SkillExtractionResult(
            added_records=[added_record], updated_records=[], deleted_ids=[]
        )

        mock_skill_repo = AsyncMock()
        mock_skill_repo.get_by_cluster_id = AsyncMock(return_value=[])
        mock_milvus_repo = AsyncMock()
        mock_es_repo = AsyncMock()
        mock_es_repo.create = AsyncMock(side_effect=Exception("ES down"))

        with (
            patch("core.lock.redis_distributed_lock.distributed_lock") as mock_lock,
            patch(
                "core.di.get_bean_by_type",
                side_effect=_build_skill_trigger_patches(
                    mock_skill_repo, mock_milvus_repo, mock_es_repo
                ),
            ),
            patch(
                "memory_layer.llm.llm_provider.build_default_provider",
                return_value=MagicMock(),
            ),
            patch(
                "memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor"
            ) as mock_ext_cls,
            patch(
                "infra_layer.adapters.out.search.milvus.converter.agent_skill_milvus_converter.AgentSkillMilvusConverter"
            ) as mock_milvus_conv,
            patch(
                "infra_layer.adapters.out.search.elasticsearch.converter.agent_skill_converter.AgentSkillConverter"
            ) as mock_es_conv,
        ):

            mock_lock_ctx = AsyncMock()
            mock_lock_ctx.__aenter__ = AsyncMock(return_value=True)
            mock_lock_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_lock.return_value = mock_lock_ctx

            mock_extractor = AsyncMock()
            mock_extractor.extract_and_save = AsyncMock(return_value=extraction_result)
            mock_ext_cls.return_value = mock_extractor

            mock_milvus_conv.from_mongo.return_value = {"vector": [0.1], "id": "x"}
            mock_es_conv.from_mongo.return_value = MagicMock()

            import importlib
            import biz_layer.mem_memorize as memorize_mod

            importlib.reload(memorize_mod)

            await memorize_mod._trigger_agent_skill_extraction(
                group_id="g1", cluster_id="c1", memcell=memcell, agent_case=agent_case
            )

    @pytest.mark.asyncio
    async def test_deleted_ids_removed_from_search_engines(self):
        """Deleted skill IDs should be removed from both Milvus and ES."""
        from memory_layer.memory_extractor.agent_skill_extractor import (
            SkillExtractionResult,
        )

        memcell = _make_agent_memcell_for_trigger()
        agent_case = _make_agent_case_for_trigger()

        extraction_result = SkillExtractionResult(
            added_records=[],
            updated_records=[],
            deleted_ids=["dead_skill_1", "dead_skill_2"],
        )

        mock_skill_repo = AsyncMock()
        mock_skill_repo.get_by_cluster_id = AsyncMock(return_value=[])
        mock_milvus_repo = AsyncMock()
        mock_es_repo = AsyncMock()

        with (
            patch("core.lock.redis_distributed_lock.distributed_lock") as mock_lock,
            patch(
                "core.di.get_bean_by_type",
                side_effect=_build_skill_trigger_patches(
                    mock_skill_repo, mock_milvus_repo, mock_es_repo
                ),
            ),
            patch(
                "memory_layer.llm.llm_provider.build_default_provider",
                return_value=MagicMock(),
            ),
            patch(
                "memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor"
            ) as mock_ext_cls,
            patch(
                "infra_layer.adapters.out.search.milvus.converter.agent_skill_milvus_converter.AgentSkillMilvusConverter"
            ) as mock_milvus_conv,
            patch(
                "infra_layer.adapters.out.search.elasticsearch.converter.agent_skill_converter.AgentSkillConverter"
            ) as mock_es_conv,
        ):

            mock_lock_ctx = AsyncMock()
            mock_lock_ctx.__aenter__ = AsyncMock(return_value=True)
            mock_lock_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_lock.return_value = mock_lock_ctx

            mock_extractor = AsyncMock()
            mock_extractor.extract_and_save = AsyncMock(return_value=extraction_result)
            mock_ext_cls.return_value = mock_extractor

            import importlib
            import biz_layer.mem_memorize as memorize_mod

            importlib.reload(memorize_mod)

            await memorize_mod._trigger_agent_skill_extraction(
                group_id="g1", cluster_id="c1", memcell=memcell, agent_case=agent_case
            )

            assert mock_milvus_repo.delete_by_id.call_count == 2
            mock_milvus_repo.delete_by_id.assert_any_call("dead_skill_1")
            mock_milvus_repo.delete_by_id.assert_any_call("dead_skill_2")

            assert mock_es_repo.delete_by_id.call_count == 2
            mock_es_repo.delete_by_id.assert_any_call("dead_skill_1")
            mock_es_repo.delete_by_id.assert_any_call("dead_skill_2")

    @pytest.mark.asyncio
    async def test_no_changes_skips_sync(self):
        """When extraction produces no changes, search engine sync is skipped."""
        from memory_layer.memory_extractor.agent_skill_extractor import (
            SkillExtractionResult,
        )

        memcell = _make_agent_memcell_for_trigger()
        agent_case = _make_agent_case_for_trigger()

        extraction_result = SkillExtractionResult()

        mock_skill_repo = AsyncMock()
        mock_skill_repo.get_by_cluster_id = AsyncMock(return_value=[])
        mock_milvus_repo = AsyncMock()
        mock_es_repo = AsyncMock()

        with (
            patch("core.lock.redis_distributed_lock.distributed_lock") as mock_lock,
            patch(
                "core.di.get_bean_by_type",
                side_effect=_build_skill_trigger_patches(
                    mock_skill_repo, mock_milvus_repo, mock_es_repo
                ),
            ),
            patch(
                "memory_layer.llm.llm_provider.build_default_provider",
                return_value=MagicMock(),
            ),
            patch(
                "memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor"
            ) as mock_ext_cls,
            patch(
                "infra_layer.adapters.out.search.milvus.converter.agent_skill_milvus_converter.AgentSkillMilvusConverter"
            ) as mock_milvus_conv,
            patch(
                "infra_layer.adapters.out.search.elasticsearch.converter.agent_skill_converter.AgentSkillConverter"
            ) as mock_es_conv,
        ):

            mock_lock_ctx = AsyncMock()
            mock_lock_ctx.__aenter__ = AsyncMock(return_value=True)
            mock_lock_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_lock.return_value = mock_lock_ctx

            mock_extractor = AsyncMock()
            mock_extractor.extract_and_save = AsyncMock(return_value=extraction_result)
            mock_ext_cls.return_value = mock_extractor

            import importlib
            import biz_layer.mem_memorize as memorize_mod

            importlib.reload(memorize_mod)

            await memorize_mod._trigger_agent_skill_extraction(
                group_id="g1", cluster_id="c1", memcell=memcell, agent_case=agent_case
            )

            mock_milvus_repo.insert.assert_not_called()
            mock_milvus_repo.delete_by_id.assert_not_called()
            mock_milvus_repo.flush.assert_not_called()
            mock_es_repo.create.assert_not_called()
            mock_es_repo.delete_by_id.assert_not_called()


# ===========================================================================
# _update_memcell_and_cluster - agent case integration
# ===========================================================================


class TestUpdateMemcellAndClusterAgentCase:
    """Tests for _update_memcell_and_cluster passing agent_case to clustering."""

    @pytest.mark.asyncio
    async def test_agent_case_passed_to_trigger_clustering(self):
        """Agent case from state should be passed to _trigger_clustering."""
        import biz_layer.mem_memorize as mod

        agent_case = _make_agent_case_for_trigger()
        memcell = _make_agent_memcell_for_trigger()
        group_episode = SimpleNamespace(episode="Agent deployed the app successfully.")

        state = SimpleNamespace(
            request=SimpleNamespace(group_id="g1"),
            memcell=memcell,
            scene="solo",
            group_episode=group_episode,
            agent_case=agent_case,
        )

        original_trigger = mod._trigger_clustering
        try:
            mock_cluster = AsyncMock()
            mod._trigger_clustering = mock_cluster

            await mod._update_memcell_and_cluster(state)

            mock_cluster.assert_called_once()
            call_kwargs = mock_cluster.call_args.kwargs
            assert call_kwargs.get("agent_case") is agent_case
        finally:
            mod._trigger_clustering = original_trigger

    @pytest.mark.asyncio
    async def test_no_group_id_skips_clustering(self):
        """No group_id should skip clustering entirely."""
        import biz_layer.mem_memorize as mod

        state = SimpleNamespace(
            request=SimpleNamespace(group_id=None),
            memcell=_make_agent_memcell_for_trigger(),
            scene="solo",
            group_episode=SimpleNamespace(episode="text"),
            agent_case=_make_agent_case_for_trigger(),
        )

        original_trigger = mod._trigger_clustering
        try:
            mock_cluster = AsyncMock()
            mod._trigger_clustering = mock_cluster

            await mod._update_memcell_and_cluster(state)
            mock_cluster.assert_not_called()
        finally:
            mod._trigger_clustering = original_trigger

    @pytest.mark.asyncio
    async def test_no_group_episode_skips_clustering(self):
        """No group_episode should skip clustering entirely."""
        import biz_layer.mem_memorize as mod

        state = SimpleNamespace(
            request=SimpleNamespace(group_id="g1"),
            memcell=_make_agent_memcell_for_trigger(),
            scene="solo",
            group_episode=None,
            agent_case=_make_agent_case_for_trigger(),
        )

        original_trigger = mod._trigger_clustering
        try:
            mock_cluster = AsyncMock()
            mod._trigger_clustering = mock_cluster

            await mod._update_memcell_and_cluster(state)
            mock_cluster.assert_not_called()
        finally:
            mod._trigger_clustering = original_trigger

    @pytest.mark.asyncio
    async def test_agent_conversation_uses_agent_config(self):
        """Agent conversations should use DEFAULT_MEMORIZE_CONFIG."""
        from biz_layer.memorize_config import DEFAULT_MEMORIZE_CONFIG
        import biz_layer.mem_memorize as mod

        memcell = _make_agent_memcell_for_trigger()
        group_episode = SimpleNamespace(episode="text")

        state = SimpleNamespace(
            request=SimpleNamespace(group_id="g1"),
            memcell=memcell,
            scene="solo",
            group_episode=group_episode,
            agent_case=_make_agent_case_for_trigger(),
        )

        original_trigger = mod._trigger_clustering
        try:
            mock_cluster = AsyncMock()
            mod._trigger_clustering = mock_cluster

            await mod._update_memcell_and_cluster(state)

            call_kwargs = mock_cluster.call_args.kwargs
            assert call_kwargs["config"] is DEFAULT_MEMORIZE_CONFIG
        finally:
            mod._trigger_clustering = original_trigger

    @pytest.mark.asyncio
    async def test_clustering_exception_handled(self):
        """Clustering exceptions should be caught, not propagated."""
        import biz_layer.mem_memorize as mod

        state = SimpleNamespace(
            request=SimpleNamespace(group_id="g1"),
            memcell=_make_agent_memcell_for_trigger(),
            scene="solo",
            group_episode=SimpleNamespace(episode="text"),
            agent_case=_make_agent_case_for_trigger(),
        )

        original_trigger = mod._trigger_clustering
        try:
            mod._trigger_clustering = AsyncMock(side_effect=Exception("boom"))

            # Should not raise
            await mod._update_memcell_and_cluster(state)
        finally:
            mod._trigger_clustering = original_trigger


# ---------------------------------------------------------------------------
# tool_calls / tool_call_id round-trip tests (RawMessage, Mapper, Repository)
# ---------------------------------------------------------------------------


class TestRawMessageToolCallsFields:
    """Tests for tool_calls/tool_call_id fields on RawMessage document model."""

    def _construct(self, **kwargs):
        """Build a RawMessage without Beanie collection init."""
        from infra_layer.adapters.out.persistence.document.request.raw_message import (
            RawMessage,
        )

        defaults = dict(group_id="g1", request_id="r1")
        defaults.update(kwargs)
        return RawMessage.model_construct(**defaults)

    def test_raw_message_accepts_tool_calls(self):
        """RawMessage should store tool_calls list."""
        tool_calls = [
            {
                "id": "call_001",
                "type": "function",
                "function": {"name": "web_search", "arguments": "{}"},
            }
        ]
        msg = self._construct(role="assistant", tool_calls=tool_calls)
        assert msg.tool_calls == tool_calls

    def test_raw_message_accepts_tool_call_id(self):
        """RawMessage should store tool_call_id string."""
        msg = self._construct(role="tool", tool_call_id="call_001")
        assert msg.tool_call_id == "call_001"

    def test_raw_message_defaults_none(self):
        """tool_calls and tool_call_id default to None."""
        msg = self._construct(role="user")
        assert msg.tool_calls is None
        assert msg.tool_call_id is None


class TestRawMessageMapperToolCalls:
    """Tests for RawMessageMapper preserving tool_calls/tool_call_id."""

    def _make_raw_message(self, **overrides):
        from infra_layer.adapters.out.persistence.document.request.raw_message import (
            RawMessage,
        )

        defaults = dict(
            group_id="g1",
            request_id="r1",
            message_id="msg_001",
            sender_id="assistant",
            sender_name="assistant",
            role="assistant",
            content_items=[{"type": "text", "text": "hi"}],
            timestamp="2025-01-15T10:00:00+00:00",
            tool_calls=None,
            tool_call_id=None,
        )
        defaults.update(overrides)
        return RawMessage.model_construct(**defaults)

    def test_mapper_preserves_tool_calls(self):
        """to_raw_data should include tool_calls in content dict."""
        from infra_layer.adapters.out.persistence.mapper.raw_message_mapper import (
            RawMessageMapper,
        )

        tool_calls = [
            {
                "id": "call_001",
                "type": "function",
                "function": {"name": "search", "arguments": "{}"},
            }
        ]
        msg = self._make_raw_message(tool_calls=tool_calls)
        raw_data = RawMessageMapper.to_raw_data(msg)

        assert raw_data is not None
        assert raw_data.content.get("tool_calls") == tool_calls

    def test_mapper_preserves_tool_call_id(self):
        """to_raw_data should include tool_call_id in content dict."""
        from infra_layer.adapters.out.persistence.mapper.raw_message_mapper import (
            RawMessageMapper,
        )

        msg = self._make_raw_message(
            role="tool", tool_call_id="call_001", tool_calls=None
        )
        raw_data = RawMessageMapper.to_raw_data(msg)

        assert raw_data is not None
        assert raw_data.content.get("tool_call_id") == "call_001"

    def test_mapper_omits_none_tool_fields(self):
        """to_raw_data should not include tool_calls/tool_call_id when None."""
        from infra_layer.adapters.out.persistence.mapper.raw_message_mapper import (
            RawMessageMapper,
        )

        msg = self._make_raw_message()
        raw_data = RawMessageMapper.to_raw_data(msg)

        assert raw_data is not None
        assert "tool_calls" not in raw_data.content
        assert "tool_call_id" not in raw_data.content

    def test_mapper_round_trip_full_conversation(self):
        """A full agent conversation should round-trip tool_calls correctly."""
        from infra_layer.adapters.out.persistence.mapper.raw_message_mapper import (
            RawMessageMapper,
        )

        messages = [
            self._make_raw_message(
                message_id="msg_u1",
                role="user",
                sender_id="u1",
                content_items=[{"type": "text", "text": "do something"}],
            ),
            self._make_raw_message(
                message_id="msg_a1",
                role="assistant",
                content_items=[{"type": "text", "text": "searching..."}],
                tool_calls=[
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "search", "arguments": "{}"},
                    }
                ],
            ),
            self._make_raw_message(
                message_id="msg_t1",
                role="tool",
                sender_id="tool_1",
                content_items=[{"type": "text", "text": "result"}],
                tool_call_id="c1",
            ),
            self._make_raw_message(
                message_id="msg_a2",
                role="assistant",
                content_items=[{"type": "text", "text": "Done."}],
            ),
        ]

        raw_data_list = RawMessageMapper.to_raw_data_list(messages)
        assert len(raw_data_list) == 4

        # assistant with tool_calls
        assert raw_data_list[1].content.get("tool_calls") == messages[1].tool_calls
        # tool with tool_call_id
        assert raw_data_list[2].content.get("tool_call_id") == "c1"
        # user and final assistant have no tool fields
        assert "tool_calls" not in raw_data_list[0].content
        assert "tool_calls" not in raw_data_list[3].content


class TestRepositorySaveToolCalls:
    """Tests for RawMessageRepository.save_from_raw_data preserving tool_calls.

    Uses patch to intercept RawMessage construction (avoids Beanie init).
    """

    @pytest.mark.asyncio
    async def test_save_extracts_tool_calls(self):
        """save_from_raw_data should pass tool_calls to RawMessage."""
        from infra_layer.adapters.out.persistence.repository.raw_message_repository import (
            RawMessageRepository,
        )

        repo = RawMessageRepository.__new__(RawMessageRepository)
        repo.save = AsyncMock()

        tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "search", "arguments": "{}"},
            }
        ]
        content_dict = {
            "sender_id": "assistant",
            "sender_name": "assistant",
            "role": "assistant",
            "content": [{"type": "text", "text": "searching..."}],
            "tool_calls": tool_calls,
            "timestamp": "2025-01-15T10:00:00+00:00",
        }

        captured = {}

        def capture_init(**kwargs):
            captured.update(kwargs)
            return MagicMock()

        with patch(
            "infra_layer.adapters.out.persistence.repository.raw_message_repository.RawMessage",
            side_effect=capture_init,
        ):
            await repo.save_from_raw_data(
                raw_data_content=content_dict,
                data_id="msg_001",
                group_id="g1",
                request_id="r1",
            )

        assert captured["tool_calls"] == tool_calls
        assert captured["tool_call_id"] is None

    @pytest.mark.asyncio
    async def test_save_extracts_tool_call_id(self):
        """save_from_raw_data should pass tool_call_id to RawMessage."""
        from infra_layer.adapters.out.persistence.repository.raw_message_repository import (
            RawMessageRepository,
        )

        repo = RawMessageRepository.__new__(RawMessageRepository)
        repo.save = AsyncMock()

        content_dict = {
            "sender_id": "tool_1",
            "role": "tool",
            "content": [{"type": "text", "text": "result"}],
            "tool_call_id": "call_1",
            "timestamp": "2025-01-15T10:00:00+00:00",
        }

        captured = {}

        def capture_init(**kwargs):
            captured.update(kwargs)
            return MagicMock()

        with patch(
            "infra_layer.adapters.out.persistence.repository.raw_message_repository.RawMessage",
            side_effect=capture_init,
        ):
            await repo.save_from_raw_data(
                raw_data_content=content_dict,
                data_id="msg_002",
                group_id="g1",
                request_id="r1",
            )

        assert captured["tool_call_id"] == "call_1"
        assert captured.get("tool_calls") is None

    @pytest.mark.asyncio
    async def test_save_without_tool_fields(self):
        """save_from_raw_data for a normal user message should pass None tool fields."""
        from infra_layer.adapters.out.persistence.repository.raw_message_repository import (
            RawMessageRepository,
        )

        repo = RawMessageRepository.__new__(RawMessageRepository)
        repo.save = AsyncMock()

        content_dict = {
            "sender_id": "user_1",
            "role": "user",
            "content": [{"type": "text", "text": "hello"}],
            "timestamp": "2025-01-15T10:00:00+00:00",
        }

        captured = {}

        def capture_init(**kwargs):
            captured.update(kwargs)
            return MagicMock()

        with patch(
            "infra_layer.adapters.out.persistence.repository.raw_message_repository.RawMessage",
            side_effect=capture_init,
        ):
            await repo.save_from_raw_data(
                raw_data_content=content_dict,
                data_id="msg_003",
                group_id="g1",
                request_id="r1",
            )

        assert captured.get("tool_calls") is None
        assert captured.get("tool_call_id") is None

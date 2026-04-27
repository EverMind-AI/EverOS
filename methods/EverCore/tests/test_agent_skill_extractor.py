"""
AgentSkillExtractor Unit Tests

Tests for:
- _format_cases: case record formatting for LLM prompt
- _format_existing_skills: existing skills formatting with indices
- _cosine_similarity: vector cosine similarity
- _select_prompt: quality-based prompt selection
- _is_skill_content_sufficient: content validation
- _apply_add: new skill creation logic
- _apply_update: existing skill update logic
- extract_and_save: full incremental extraction flow

Usage:
    PYTHONPATH=src pytest tests/test_agent_skill_extractor.py -v
"""

import json
import pytest
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from memory_layer.memory_extractor.agent_skill_extractor import (
    AgentSkillExtractor,
    SkillExtractionResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_case_record(
    task_intent: str = "Build a REST API",
    approach: str = "1. Design schema\n2. Implement endpoints",
    quality_score: float = 0.8,
    timestamp: datetime = None,
    key_insight: str = None,
    record_id: str = None,
):
    """Create a mock AgentCaseRecord-like object."""
    return SimpleNamespace(
        id=record_id,
        task_intent=task_intent,
        approach=approach,
        quality_score=quality_score,
        key_insight=key_insight,
        timestamp=timestamp or datetime(2025, 1, 15, 10, 0, 0),
    )


def _make_skill_record(
    name: str = "API Development",
    description: str = "Build REST APIs with proper error handling",
    content: str = "## Steps\n1. Design schema\n2. Implement endpoints\n3. Add validation\n4. Write tests\n5. Deploy",
    confidence: float = 0.7,
    vector: list = None,
    source_case_ids: list = None,
    record_id: str = "skill_001",
):
    """Create a mock AgentSkillRecord-like object."""
    rec = SimpleNamespace(
        id=record_id,
        name=name,
        description=description,
        content=content,
        confidence=confidence,
        vector=vector or [0.1, 0.2, 0.3],
        vector_model="test-model",
        source_case_ids=source_case_ids or [],
        maturity_score=0.7,
        updated_at=None,
    )
    return rec


def _build_extractor(
    llm_response: str = None,
    maturity_threshold: float = 0.6,
) -> AgentSkillExtractor:
    """Build an extractor with mocked LLM provider."""
    mock_llm = MagicMock()
    if llm_response:
        mock_llm.generate = AsyncMock(return_value=llm_response)
    else:
        mock_llm.generate = AsyncMock(return_value='{"operations": [{"action": "none"}]}')

    return AgentSkillExtractor(
        llm_provider=mock_llm,
        success_extract_prompt="{new_case_json}{existing_skills_json}",
        failure_extract_prompt="{new_case_json}{existing_skills_json}",
        maturity_threshold=maturity_threshold,
    )


def _mock_skill_repo():
    """Create a mock skill repository."""
    repo = AsyncMock()
    repo.save_skill = AsyncMock(side_effect=lambda rec: rec)
    repo.update_skill_by_id = AsyncMock(return_value=True)
    repo.soft_delete_by_id = AsyncMock(return_value=True)
    return repo


# ===========================================================================
# _format_cases tests
# ===========================================================================


class TestFormatCases:
    """Tests for AgentSkillExtractor._format_cases."""

    def test_single_case(self):
        extractor = _build_extractor()
        cases = [_make_case_record()]
        result = extractor._format_cases(cases)
        parsed = json.loads(result)
        assert len(parsed) == 1
        assert parsed[0]["task_intent"] == "Build a REST API"
        assert parsed[0]["quality_score"] == 0.8

    def test_multiple_cases(self):
        extractor = _build_extractor()
        cases = [
            _make_case_record(task_intent="Task A"),
            _make_case_record(task_intent="Task B", quality_score=0.3),
        ]
        result = extractor._format_cases(cases)
        parsed = json.loads(result)
        assert len(parsed) == 2
        assert parsed[1]["task_intent"] == "Task B"
        assert parsed[1]["quality_score"] == 0.3

    def test_empty_cases(self):
        extractor = _build_extractor()
        result = extractor._format_cases([])
        assert json.loads(result) == []

    def test_none_timestamp(self):
        extractor = _build_extractor()
        case = _make_case_record()
        case.timestamp = None
        result = extractor._format_cases([case])
        parsed = json.loads(result)
        assert parsed[0]["timestamp"] is None


# ===========================================================================
# _format_existing_skills tests
# ===========================================================================


class TestFormatExistingSkills:
    """Tests for AgentSkillExtractor._format_existing_skills."""

    def test_empty_returns_placeholder(self):
        extractor = _build_extractor()
        result = extractor._format_existing_skills([])
        assert "empty" in result.lower()

    def test_single_skill_has_index(self):
        extractor = _build_extractor()
        result = extractor._format_existing_skills([_make_skill_record()])
        parsed = json.loads(result)
        assert parsed[0]["index"] == 0
        assert parsed[0]["name"] == "API Development"

    def test_multiple_skills_indexed_sequentially(self):
        extractor = _build_extractor()
        skills = [
            _make_skill_record(name="Skill A"),
            _make_skill_record(name="Skill B"),
            _make_skill_record(name="Skill C"),
        ]
        result = extractor._format_existing_skills(skills)
        parsed = json.loads(result)
        assert [s["index"] for s in parsed] == [0, 1, 2]


# ===========================================================================
# _cosine_similarity tests
# ===========================================================================


class TestCosineSimilarity:
    """Tests for AgentSkillExtractor._cosine_similarity."""

    def test_identical_vectors(self):
        sim = AgentSkillExtractor._cosine_similarity([1, 0, 0], [1, 0, 0])
        assert abs(sim - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        sim = AgentSkillExtractor._cosine_similarity([1, 0, 0], [0, 1, 0])
        assert abs(sim) < 1e-6

    def test_opposite_vectors(self):
        sim = AgentSkillExtractor._cosine_similarity([1, 0], [-1, 0])
        assert abs(sim - (-1.0)) < 1e-6

    def test_zero_vector(self):
        sim = AgentSkillExtractor._cosine_similarity([0, 0], [1, 1])
        assert sim == 0.0

    def test_both_zero(self):
        sim = AgentSkillExtractor._cosine_similarity([0, 0], [0, 0])
        assert sim == 0.0


# ===========================================================================
# _select_prompt tests
# ===========================================================================


class TestSelectPrompt:
    """Tests for AgentSkillExtractor._select_prompt."""

    def test_high_quality_uses_success_prompt(self):
        extractor = _build_extractor()
        extractor.success_extract_prompt = "SUCCESS"
        extractor.failure_extract_prompt = "FAILURE"
        cases = [_make_case_record(quality_score=0.8)]
        assert extractor._select_prompt(cases) == "SUCCESS"

    def test_low_quality_uses_failure_prompt(self):
        extractor = _build_extractor()
        extractor.success_extract_prompt = "SUCCESS"
        extractor.failure_extract_prompt = "FAILURE"
        cases = [_make_case_record(quality_score=0.3)]
        assert extractor._select_prompt(cases) == "FAILURE"

    def test_boundary_quality_uses_success(self):
        """quality_score == 0.5 (threshold) uses success prompt."""
        extractor = _build_extractor()
        extractor.success_extract_prompt = "SUCCESS"
        extractor.failure_extract_prompt = "FAILURE"
        cases = [_make_case_record(quality_score=0.5)]
        assert extractor._select_prompt(cases) == "SUCCESS"

    def test_just_below_threshold_uses_failure(self):
        extractor = _build_extractor()
        extractor.success_extract_prompt = "SUCCESS"
        extractor.failure_extract_prompt = "FAILURE"
        cases = [_make_case_record(quality_score=0.49)]
        assert extractor._select_prompt(cases) == "FAILURE"

    def test_multiple_cases_uses_max_quality(self):
        """When one case has high quality, uses success prompt."""
        extractor = _build_extractor()
        extractor.success_extract_prompt = "SUCCESS"
        extractor.failure_extract_prompt = "FAILURE"
        cases = [
            _make_case_record(quality_score=0.2),
            _make_case_record(quality_score=0.8),
        ]
        assert extractor._select_prompt(cases) == "SUCCESS"

    def test_none_quality_treated_as_default(self):
        """None quality_score defaults to 0.5 (boundary) -> success prompt."""
        extractor = _build_extractor()
        extractor.success_extract_prompt = "SUCCESS"
        extractor.failure_extract_prompt = "FAILURE"
        case = _make_case_record()
        case.quality_score = None
        assert extractor._select_prompt([case]) == "SUCCESS"


# ===========================================================================
# _is_skill_content_sufficient tests
# ===========================================================================


class TestIsSkillContentSufficient:
    """Tests for AgentSkillExtractor._is_skill_content_sufficient."""

    def test_empty_content(self):
        assert AgentSkillExtractor._is_skill_content_sufficient("") is False

    def test_none_content(self):
        assert AgentSkillExtractor._is_skill_content_sufficient(None) is False

    def test_too_short(self):
        assert AgentSkillExtractor._is_skill_content_sufficient("short") is False

    def test_too_few_lines(self):
        # Long enough characters but only 2 lines
        text = "A" * 100 + "\n" + "B" * 100
        assert AgentSkillExtractor._is_skill_content_sufficient(text) is False

    def test_sufficient_content(self):
        text = "## Steps\n1. First step\n2. Second step\n3. Third step\n4. Fourth step\n5. Fifth step"
        assert AgentSkillExtractor._is_skill_content_sufficient(text) is True

    def test_whitespace_only_lines_ignored(self):
        text = "line1\n\n\nline2\n\n\nline3\n   \n"
        assert AgentSkillExtractor._is_skill_content_sufficient(text) is False

    def test_custom_thresholds(self):
        text = "ab\ncd\nef"
        assert AgentSkillExtractor._is_skill_content_sufficient(text, min_lines=3, min_length=5) is True
        assert AgentSkillExtractor._is_skill_content_sufficient(text, min_lines=3, min_length=100) is False


# ===========================================================================
# _call_llm tests
# ===========================================================================


class TestCallLLM:
    """Tests for AgentSkillExtractor._call_llm."""

    @pytest.mark.asyncio
    async def test_valid_response(self):
        response = json.dumps({"operations": [{"action": "none"}]})
        extractor = _build_extractor(llm_response=response)
        result = await extractor._call_llm("cases", "skills", "{new_case_json}{existing_skills_json}")
        assert result is not None
        assert "operations" in result

    @pytest.mark.asyncio
    async def test_invalid_json_retries(self):
        extractor = _build_extractor()
        extractor.llm_provider.generate = AsyncMock(return_value="not valid json")
        result = await extractor._call_llm("cases", "skills", "{new_case_json}{existing_skills_json}")
        assert result is None
        assert extractor.llm_provider.generate.call_count == 3

    @pytest.mark.asyncio
    async def test_missing_operations_key_retries(self):
        extractor = _build_extractor()
        extractor.llm_provider.generate = AsyncMock(
            return_value=json.dumps({"result": "no operations key"})
        )
        result = await extractor._call_llm("cases", "skills", "{new_case_json}{existing_skills_json}")
        assert result is None

    @pytest.mark.asyncio
    async def test_exception_retries(self):
        extractor = _build_extractor()
        extractor.llm_provider.generate = AsyncMock(side_effect=Exception("LLM error"))
        result = await extractor._call_llm("cases", "skills", "{new_case_json}{existing_skills_json}")
        assert result is None
        assert extractor.llm_provider.generate.call_count == 3


# ===========================================================================
# _evaluate_maturity tests
# ===========================================================================


class TestEvaluateMaturity:
    """Tests for AgentSkillExtractor._evaluate_maturity."""

    @pytest.mark.asyncio
    async def test_valid_scores(self):
        """Test 4-dimension scoring: (completeness + executability + evidence + clarity) / 20."""
        response = json.dumps({
            "completeness": 4,
            "executability": 4,
            "evidence": 4,
            "clarity": 5,
            "reason": "Well documented skill",
        })
        extractor = _build_extractor()
        extractor.maturity_prompt = "{name}{description}{content}{confidence}"
        extractor.llm_provider.generate = AsyncMock(return_value=response)
        score = await extractor._evaluate_maturity("Test", "Desc", "Content", 0.8)
        assert score is not None
        expected = (4 + 4 + 4 + 5) / 20.0
        assert abs(score - expected) < 1e-6

    @pytest.mark.asyncio
    async def test_max_score_clamped(self):
        response = json.dumps({
            "completeness": 5,
            "executability": 5,
            "evidence": 5,
            "clarity": 5,
        })
        extractor = _build_extractor()
        extractor.maturity_prompt = "{name}{description}{content}{confidence}"
        extractor.llm_provider.generate = AsyncMock(return_value=response)
        score = await extractor._evaluate_maturity("Test", "Desc", "Content", 0.8)
        assert score == 1.0

    @pytest.mark.asyncio
    async def test_invalid_response_returns_none(self):
        extractor = _build_extractor()
        extractor.maturity_prompt = "{name}{description}{content}{confidence}"
        extractor.llm_provider.generate = AsyncMock(return_value="not json")
        score = await extractor._evaluate_maturity("Test", "Desc", "Content", 0.8)
        assert score is None

    @pytest.mark.asyncio
    async def test_missing_dimension_returns_none(self):
        response = json.dumps({
            "completeness": 4,
            "executability": 4,
            # missing evidence, clarity
        })
        extractor = _build_extractor()
        extractor.maturity_prompt = "{name}{description}{content}{confidence}"
        extractor.llm_provider.generate = AsyncMock(return_value=response)
        score = await extractor._evaluate_maturity("Test", "Desc", "Content", 0.8)
        assert score is None

    @pytest.mark.asyncio
    async def test_llm_exception_returns_none(self):
        extractor = _build_extractor()
        extractor.maturity_prompt = "{name}{description}{content}{confidence}"
        extractor.llm_provider.generate = AsyncMock(side_effect=Exception("fail"))
        score = await extractor._evaluate_maturity("Test", "Desc", "Content", 0.8)
        assert score is None


# ===========================================================================
# _apply_add tests
# ===========================================================================


class TestApplyAdd:
    """Tests for AgentSkillExtractor._apply_add."""

    @pytest.mark.asyncio
    @patch("memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor._compute_embedding")
    @patch("memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor._evaluate_maturity")
    async def test_successful_add(self, mock_maturity, mock_embed):
        mock_embed.return_value = {"embedding": [0.1, 0.2], "vector_model": "test"}
        mock_maturity.return_value = 0.8
        repo = _mock_skill_repo()

        extractor = _build_extractor()
        op = {
            "action": "add",
            "data": {
                "name": "New Skill",
                "description": "Does things",
                "content": "## Steps\n1. Do this\n2. Do that\n3. Check results\n4. Validate\n5. Deploy",
                "confidence": 0.6,
            },
        }
        # AgentSkillRecord is imported locally inside _apply_add.
        # Pre-import the module then patch the class on it.
        import infra_layer.adapters.out.persistence.document.memory.agent_skill as skill_mod
        with patch.object(skill_mod, "AgentSkillRecord", return_value=MagicMock(id="new_001")):
            result = await extractor._apply_add(
                op, "cluster_001", "group_001", "user_001", repo, source_case_ids=["evt_001"]
            )
        assert result is not None
        repo.save_skill.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_content_skipped(self):
        extractor = _build_extractor()
        repo = _mock_skill_repo()
        op = {"action": "add", "data": {"name": "Skill", "description": "Desc", "content": ""}}
        result = await extractor._apply_add(op, "c", "g", "u", repo)
        assert result is None
        repo.save_skill.assert_not_called()

    @pytest.mark.asyncio
    async def test_insufficient_content_skipped(self):
        extractor = _build_extractor()
        repo = _mock_skill_repo()
        op = {"action": "add", "data": {"name": "Skill", "description": "Desc", "content": "too short"}}
        result = await extractor._apply_add(op, "c", "g", "u", repo)
        assert result is None

    @pytest.mark.asyncio
    async def test_no_name_no_description_skipped(self):
        extractor = _build_extractor()
        repo = _mock_skill_repo()
        op = {
            "action": "add",
            "data": {
                "name": "",
                "description": "",
                "content": "## Steps for completing the task properly\n1. Analyze requirements carefully\n2. Build the implementation\n3. Check edge cases\n4. Deploy to staging\n5. Verify in production",
            },
        }
        result = await extractor._apply_add(op, "c", "g", "u", repo)
        assert result is None

    @pytest.mark.asyncio
    @patch("memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor._compute_embedding")
    @patch("memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor._evaluate_maturity")
    async def test_invalid_confidence_defaults_to_half(self, mock_maturity, mock_embed):
        mock_embed.return_value = {"embedding": [0.1], "vector_model": "test"}
        mock_maturity.return_value = 0.7
        repo = _mock_skill_repo()

        extractor = _build_extractor()
        op = {
            "action": "add",
            "data": {
                "name": "Skill",
                "description": "Desc",
                "content": "## Steps for completing the task properly\n1. Analyze the requirements carefully\n2. Build the implementation with tests\n3. Check all edge cases thoroughly\n4. Deploy to staging environment\n5. Verify in production setup",
                "confidence": "invalid",
            },
        }
        import infra_layer.adapters.out.persistence.document.memory.agent_skill as skill_mod
        with patch.object(
            skill_mod, "AgentSkillRecord", return_value=MagicMock(id="new_002")
        ) as mock_record_cls:
            result = await extractor._apply_add(op, "c", "g", "u", repo)
        assert result is not None
        # Check the record was created with confidence=0.5 (default)
        call_args = mock_record_cls.call_args
        assert call_args.kwargs["confidence"] == 0.5


# ===========================================================================
# _apply_update tests
# ===========================================================================


class TestApplyUpdate:
    """Tests for AgentSkillExtractor._apply_update."""

    @pytest.mark.asyncio
    @patch("memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor._compute_embedding")
    @patch("memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor._evaluate_maturity")
    async def test_update_content(self, mock_maturity, mock_embed):
        mock_embed.return_value = None
        mock_maturity.return_value = 0.75
        repo = _mock_skill_repo()
        existing = [_make_skill_record()]
        result_obj = SkillExtractionResult()

        extractor = _build_extractor()
        op = {
            "action": "update",
            "index": 0,
            "data": {
                "content": "## Updated\n1. New step 1\n2. New step 2\n3. Step 3\n4. Step 4\n5. Step 5",
            },
        }
        success = await extractor._apply_update(
            op, existing, repo, result_obj, source_case_ids=["evt_002"]
        )
        assert success is True
        assert len(result_obj.updated_records) == 1
        repo.update_skill_by_id.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_out_of_range_index(self):
        repo = _mock_skill_repo()
        existing = [_make_skill_record()]
        result_obj = SkillExtractionResult()

        extractor = _build_extractor()
        op = {"action": "update", "index": 5, "data": {"content": "new"}}
        success = await extractor._apply_update(op, existing, repo, result_obj)
        assert success is False
        repo.update_skill_by_id.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_negative_index(self):
        repo = _mock_skill_repo()
        result_obj = SkillExtractionResult()
        extractor = _build_extractor()
        op = {"action": "update", "index": -1, "data": {"name": "X"}}
        success = await extractor._apply_update(op, [_make_skill_record()], repo, result_obj)
        assert success is False

    @pytest.mark.asyncio
    async def test_update_invalid_index_type(self):
        repo = _mock_skill_repo()
        result_obj = SkillExtractionResult()
        extractor = _build_extractor()
        op = {"action": "update", "index": "abc", "data": {"name": "X"}}
        success = await extractor._apply_update(op, [_make_skill_record()], repo, result_obj)
        assert success is False

    @pytest.mark.asyncio
    async def test_update_no_fields_skipped(self):
        repo = _mock_skill_repo()
        existing = [_make_skill_record()]
        result_obj = SkillExtractionResult()
        extractor = _build_extractor()
        op = {"action": "update", "index": 0, "data": {}}
        success = await extractor._apply_update(op, existing, repo, result_obj)
        assert success is False

    @pytest.mark.asyncio
    async def test_auto_delete_low_confidence(self):
        """When confidence drops below threshold, skill is retired via update (not soft-deleted)."""
        repo = _mock_skill_repo()
        existing = [_make_skill_record()]
        result_obj = SkillExtractionResult()
        extractor = _build_extractor()
        op = {"action": "update", "index": 0, "data": {"confidence": 0.05}}
        success = await extractor._apply_update(op, existing, repo, result_obj)
        assert success is True
        assert len(result_obj.deleted_ids) == 1
        assert str(existing[0].id) in result_obj.deleted_ids
        # Should update confidence in-place, not soft-delete
        repo.update_skill_by_id.assert_called_once()
        call_args = repo.update_skill_by_id.call_args[0]
        assert call_args[1] == {"confidence": 0.05}

    @pytest.mark.asyncio
    @patch("memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor._compute_embedding")
    @patch("memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor._evaluate_maturity")
    async def test_update_name_triggers_re_embed(self, mock_maturity, mock_embed):
        mock_embed.return_value = {"embedding": [0.5, 0.6], "vector_model": "new-model"}
        mock_maturity.return_value = 0.8
        repo = _mock_skill_repo()
        existing = [_make_skill_record(name="Old Name")]
        result_obj = SkillExtractionResult()
        extractor = _build_extractor()
        op = {"action": "update", "index": 0, "data": {"name": "New Name"}}
        await extractor._apply_update(op, existing, repo, result_obj)
        mock_embed.assert_called_once()
        # Verify vector was included in updates
        update_call = repo.update_skill_by_id.call_args
        updates = update_call[0][1]
        assert "vector" in updates

    @pytest.mark.asyncio
    @patch("memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor._compute_embedding")
    @patch("memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor._evaluate_maturity")
    async def test_source_case_id_appended(self, mock_maturity, mock_embed):
        mock_embed.return_value = None
        mock_maturity.return_value = 0.7
        repo = _mock_skill_repo()
        existing = [_make_skill_record(source_case_ids=["evt_001"])]
        result_obj = SkillExtractionResult()
        extractor = _build_extractor()
        op = {"action": "update", "index": 0, "data": {"confidence": 0.9}}
        await extractor._apply_update(
            op, existing, repo, result_obj, source_case_ids=["evt_002"]
        )
        update_call = repo.update_skill_by_id.call_args
        updates = update_call[0][1]
        assert "evt_002" in updates["source_case_ids"]
        assert "evt_001" in updates["source_case_ids"]

    @pytest.mark.asyncio
    @patch("memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor._compute_embedding")
    @patch("memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor._evaluate_maturity")
    async def test_duplicate_source_case_id_not_added(self, mock_maturity, mock_embed):
        mock_embed.return_value = None
        mock_maturity.return_value = 0.7
        repo = _mock_skill_repo()
        # Use a fresh list so the test owns the mutable state
        existing = [_make_skill_record(source_case_ids=["evt_001"])]
        result_obj = SkillExtractionResult()
        extractor = _build_extractor()
        op = {"action": "update", "index": 0, "data": {"confidence": 0.9}}
        await extractor._apply_update(
            op, existing, repo, result_obj, source_case_ids=["evt_001"]
        )
        update_call = repo.update_skill_by_id.call_args
        updates = update_call[0][1]
        # source_case_ids should NOT be in updates since "evt_001" is already present
        assert "source_case_ids" not in updates or updates["source_case_ids"].count("evt_001") == 1

    @pytest.mark.asyncio
    async def test_update_insufficient_content_skipped(self):
        repo = _mock_skill_repo()
        existing = [_make_skill_record()]
        result_obj = SkillExtractionResult()
        extractor = _build_extractor()
        op = {"action": "update", "index": 0, "data": {"content": "too short"}}
        success = await extractor._apply_update(op, existing, repo, result_obj)
        assert success is False


# ===========================================================================
# extract_and_save full flow tests
# ===========================================================================


class TestExtractAndSave:
    """Tests for AgentSkillExtractor.extract_and_save full flow."""

    @pytest.mark.asyncio
    async def test_no_new_cases_returns_empty(self):
        extractor = _build_extractor()
        repo = _mock_skill_repo()
        result = await extractor.extract_and_save(
            cluster_id="c1",
            group_id="g1",
            new_case_records=[],
            existing_skill_records=[],
            skill_repo=repo,
        )
        assert result.added_records == []
        assert result.updated_records == []
        assert result.deleted_ids == []

    @pytest.mark.asyncio
    async def test_llm_failure_returns_empty(self):
        extractor = _build_extractor()
        extractor.llm_provider.generate = AsyncMock(return_value="bad json")
        repo = _mock_skill_repo()
        result = await extractor.extract_and_save(
            cluster_id="c1",
            group_id="g1",
            new_case_records=[_make_case_record()],
            existing_skill_records=[],
            skill_repo=repo,
        )
        assert result.added_records == []

    @pytest.mark.asyncio
    async def test_none_action_is_noop(self):
        response = json.dumps({"operations": [{"action": "none"}]})
        extractor = _build_extractor(llm_response=response)
        repo = _mock_skill_repo()
        result = await extractor.extract_and_save(
            cluster_id="c1",
            group_id="g1",
            new_case_records=[_make_case_record()],
            existing_skill_records=[],
            skill_repo=repo,
        )
        assert result.added_records == []
        assert result.updated_records == []

    @pytest.mark.asyncio
    @patch("memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor._apply_add")
    async def test_add_operation_dispatched(self, mock_apply_add):
        mock_apply_add.return_value = MagicMock(id="new_skill")
        response = json.dumps({
            "operations": [
                {
                    "action": "add",
                    "data": {"name": "Skill", "description": "Desc", "content": "Content"},
                }
            ]
        })
        extractor = _build_extractor(llm_response=response)
        repo = _mock_skill_repo()
        result = await extractor.extract_and_save(
            cluster_id="c1",
            group_id="g1",
            new_case_records=[_make_case_record()],
            existing_skill_records=[],
            skill_repo=repo,
        )
        assert len(result.added_records) == 1
        mock_apply_add.assert_called_once()

    @pytest.mark.asyncio
    @patch("memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor._apply_update")
    async def test_update_operation_dispatched(self, mock_apply_update):
        mock_apply_update.return_value = True
        response = json.dumps({
            "operations": [
                {"action": "update", "index": 0, "data": {"confidence": 0.9}}
            ]
        })
        extractor = _build_extractor(llm_response=response)
        repo = _mock_skill_repo()
        result = await extractor.extract_and_save(
            cluster_id="c1",
            group_id="g1",
            new_case_records=[_make_case_record()],
            existing_skill_records=[_make_skill_record()],
            skill_repo=repo,
        )
        mock_apply_update.assert_called_once()

    @pytest.mark.asyncio
    async def test_unknown_action_skipped(self):
        response = json.dumps({
            "operations": [{"action": "delete_all"}]
        })
        extractor = _build_extractor(llm_response=response)
        repo = _mock_skill_repo()
        result = await extractor.extract_and_save(
            cluster_id="c1",
            group_id="g1",
            new_case_records=[_make_case_record()],
            existing_skill_records=[],
            skill_repo=repo,
        )
        assert result.added_records == []

    @pytest.mark.asyncio
    async def test_duplicate_update_index_skipped(self):
        """Second update on same index is skipped."""
        response = json.dumps({
            "operations": [
                {"action": "update", "index": 0, "data": {"confidence": 0.9}},
                {"action": "update", "index": 0, "data": {"confidence": 0.8}},
            ]
        })
        extractor = _build_extractor(llm_response=response)
        repo = _mock_skill_repo()

        # Mock _apply_update to succeed
        with patch.object(extractor, "_apply_update", new_callable=AsyncMock) as mock_update:
            mock_update.return_value = True
            await extractor.extract_and_save(
                cluster_id="c1",
                group_id="g1",
                new_case_records=[_make_case_record()],
                existing_skill_records=[_make_skill_record()],
                skill_repo=repo,
            )
            # Only first update should be called
            assert mock_update.call_count == 1

    @pytest.mark.asyncio
    async def test_top_k_selection_when_too_many_skills(self):
        """When existing skills exceed max_skills_in_prompt, top-k selection is used."""
        response = json.dumps({"operations": [{"action": "none"}]})
        extractor = _build_extractor(llm_response=response)
        repo = _mock_skill_repo()

        # Create more skills than max_skills_in_prompt
        many_skills = [_make_skill_record(name=f"Skill_{i}", record_id=f"s_{i}") for i in range(15)]

        with patch.object(
            extractor, "_select_top_k_skills", new_callable=AsyncMock
        ) as mock_top_k:
            mock_top_k.return_value = many_skills[:5]
            await extractor.extract_and_save(
                cluster_id="c1",
                group_id="g1",
                new_case_records=[_make_case_record()],
                existing_skill_records=many_skills,
                skill_repo=repo,
                max_skills_in_prompt=10,
            )
            mock_top_k.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_invalid_index_string_skipped(self):
        """Update with non-integer index is skipped without crashing."""
        response = json.dumps({
            "operations": [
                {"action": "update", "index": "first", "data": {"confidence": 0.9}}
            ]
        })
        extractor = _build_extractor(llm_response=response)
        repo = _mock_skill_repo()
        result = await extractor.extract_and_save(
            cluster_id="c1",
            group_id="g1",
            new_case_records=[_make_case_record()],
            existing_skill_records=[_make_skill_record()],
            skill_repo=repo,
        )
        # Should not crash and result should be empty
        assert result.updated_records == []


# ===========================================================================
# _select_top_k_skills tests
# ===========================================================================


class TestSelectTopKSkills:
    """Tests for AgentSkillExtractor._select_top_k_skills."""

    @pytest.mark.asyncio
    async def test_fewer_than_k_returns_all(self):
        extractor = _build_extractor()
        skills = [_make_skill_record(name=f"S{i}") for i in range(3)]
        with patch.object(extractor, "_compute_embedding", new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = {"embedding": [0.1, 0.2, 0.3]}
            result = await extractor._select_top_k_skills(skills, [_make_case_record()], top_k=10)
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_empty_query_text_returns_first_k(self):
        extractor = _build_extractor()
        skills = [_make_skill_record(name=f"S{i}") for i in range(5)]
        case = _make_case_record(task_intent="")
        result = await extractor._select_top_k_skills(skills, [case], top_k=3)
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_embedding_failure_returns_first_k(self):
        extractor = _build_extractor()
        skills = [_make_skill_record(name=f"S{i}") for i in range(5)]
        with patch.object(extractor, "_compute_embedding", new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = None
            result = await extractor._select_top_k_skills(skills, [_make_case_record()], top_k=3)
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_skills_without_vectors_included_as_fallback(self):
        extractor = _build_extractor()
        skill_with_vec = _make_skill_record(name="With Vec", vector=[0.1, 0.2, 0.3])
        skill_no_vec = _make_skill_record(name="No Vec", vector=None)
        skills = [skill_with_vec, skill_no_vec]

        with patch.object(extractor, "_compute_embedding", new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = {"embedding": [0.1, 0.2, 0.3]}
            result = await extractor._select_top_k_skills(skills, [_make_case_record()], top_k=10)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_single_case_with_vector_reuses_embedding(self):
        """When single new case already has a vector, skip _compute_embedding."""
        extractor = _build_extractor()
        skills = [_make_skill_record(name="S0", vector=[0.1, 0.2, 0.3])]
        case = _make_case_record()
        case.vector = [0.1, 0.2, 0.3]  # case already has vector

        with patch.object(extractor, "_compute_embedding", new_callable=AsyncMock) as mock_embed:
            result = await extractor._select_top_k_skills(skills, [case], top_k=10)
        # _compute_embedding should NOT have been called
        mock_embed.assert_not_called()
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_single_case_without_vector_computes_embedding(self):
        """When single new case has no vector, _compute_embedding is called."""
        extractor = _build_extractor()
        skills = [_make_skill_record(name="S0", vector=[0.1, 0.2, 0.3])]
        case = _make_case_record()
        case.vector = None

        with patch.object(extractor, "_compute_embedding", new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = {"embedding": [0.1, 0.2, 0.3]}
            result = await extractor._select_top_k_skills(skills, [case], top_k=10)
        mock_embed.assert_called_once()

    @pytest.mark.asyncio
    async def test_multiple_cases_always_computes_embedding(self):
        """When multiple new cases, always compute embedding even if they have vectors."""
        extractor = _build_extractor()
        skills = [_make_skill_record(name="S0", vector=[0.1, 0.2, 0.3])]
        case_a = _make_case_record(task_intent="Task A")
        case_a.vector = [0.1, 0.2, 0.3]
        case_b = _make_case_record(task_intent="Task B")
        case_b.vector = [0.4, 0.5, 0.6]

        with patch.object(extractor, "_compute_embedding", new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = {"embedding": [0.2, 0.3, 0.4]}
            result = await extractor._select_top_k_skills(skills, [case_a, case_b], top_k=10)
        mock_embed.assert_called_once()


# ===========================================================================
# Maturity skip optimization tests
# ===========================================================================


class TestMaturitySkipOptimization:
    """Tests for the maturity evaluation skip logic in _apply_update."""

    @pytest.mark.asyncio
    @patch("memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor._compute_embedding")
    @patch("memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor._evaluate_maturity")
    async def test_skip_maturity_when_mature_and_stable(self, mock_maturity, mock_embed):
        """Maturity eval is skipped when skill is mature + confidence stable + change < 30%."""
        mock_embed.return_value = None
        mock_maturity.return_value = 0.9  # should not be called
        repo = _mock_skill_repo()

        # Existing skill: maturity=0.8 (above threshold 0.6), confidence=0.7
        existing = [_make_skill_record(confidence=0.7)]
        existing[0].maturity_score = 0.8
        result_obj = SkillExtractionResult()
        extractor = _build_extractor(maturity_threshold=0.6)

        op = {
            "action": "update",
            "index": 0,
            "data": {
                # Minor content change (< 30% change_ratio) to ensure skip
                "content": "## Steps\n1. Design schema\n2. Implement endpoints\n3. Add validation\n4. Write tests\n5. Deploy to prod",
                "confidence": 0.8,  # not dropping
            },
        }
        success = await extractor._apply_update(
            op, existing, repo, result_obj,
            source_case_ids=["evt_003"],
        )
        assert success is True
        # _evaluate_maturity should NOT have been called
        mock_maturity.assert_not_called()
        # maturity_score should NOT be in the updates
        update_call = repo.update_skill_by_id.call_args
        updates = update_call[0][1]
        assert "maturity_score" not in updates

    @pytest.mark.asyncio
    @patch("memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor._compute_embedding")
    @patch("memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor._evaluate_maturity")
    async def test_skip_when_mature_and_confidence_stable(self, mock_maturity, mock_embed):
        """Moderate change on mature skill with stable confidence: skip maturity re-eval."""
        mock_embed.return_value = None
        mock_maturity.return_value = 0.75
        repo = _mock_skill_repo()

        existing = [_make_skill_record(confidence=0.7)]
        existing[0].maturity_score = 0.8  # above threshold
        result_obj = SkillExtractionResult()
        extractor = _build_extractor(maturity_threshold=0.6)

        op = {
            "action": "update",
            "index": 0,
            "data": {
                "content": "## Steps\n1. Design DB schema first\n2. Build REST endpoints\n3. Add request validation\n4. Write integration tests\n5. Deploy and monitor",
                "confidence": 0.8,  # not dropping
            },
        }
        success = await extractor._apply_update(
            op, existing, repo, result_obj,
            source_case_ids=["evt_004"],
        )
        assert success is True
        mock_maturity.assert_not_called()  # mature + stable confidence: skip
        update_call = repo.update_skill_by_id.call_args
        updates = update_call[0][1]
        assert "maturity_score" not in updates

    @pytest.mark.asyncio
    @patch("memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor._compute_embedding")
    @patch("memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor._evaluate_maturity")
    async def test_llm_rescore_when_immature_with_decent_case(self, mock_maturity, mock_embed):
        """Moderate change on immature skill with decent source quality: LLM rescore."""
        mock_embed.return_value = None
        mock_maturity.return_value = 0.7
        repo = _mock_skill_repo()

        existing = [_make_skill_record(confidence=0.7)]
        existing[0].maturity_score = 0.4  # below threshold of 0.6
        result_obj = SkillExtractionResult()
        extractor = _build_extractor(maturity_threshold=0.6)

        op = {
            "action": "update",
            "index": 0,
            "data": {
                "content": "## Steps\n1. Design DB schema first\n2. Build REST endpoints\n3. Add request validation\n4. Write integration tests\n5. Deploy and monitor",
                "confidence": 0.8,
            },
        }
        success = await extractor._apply_update(
            op, existing, repo, result_obj,
            source_case_ids=["evt_005"],
            source_quality=0.7,  # decent quality
        )
        assert success is True
        mock_maturity.assert_called_once()  # immature + decent case: LLM rescore
        update_call = repo.update_skill_by_id.call_args
        updates = update_call[0][1]
        assert "maturity_score" in updates
        assert updates["maturity_score"] == 0.7

    @pytest.mark.asyncio
    @patch("memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor._compute_embedding")
    @patch("memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor._evaluate_maturity")
    async def test_llm_rescore_when_mature_but_confidence_dropping(self, mock_maturity, mock_embed):
        """Moderate change on mature skill with dropping confidence: LLM rescore."""
        mock_embed.return_value = None
        mock_maturity.return_value = 0.65
        repo = _mock_skill_repo()

        existing = [_make_skill_record(confidence=0.7)]
        existing[0].maturity_score = 0.8  # above threshold
        result_obj = SkillExtractionResult()
        extractor = _build_extractor(maturity_threshold=0.6)

        op = {
            "action": "update",
            "index": 0,
            "data": {
                "content": "## Steps\n1. Design DB schema first\n2. Build REST endpoints\n3. Add request validation\n4. Write integration tests\n5. Deploy and monitor",
                "confidence": 0.4,  # dropping from 0.7 to below 0.5
            },
        }
        success = await extractor._apply_update(
            op, existing, repo, result_obj,
            source_case_ids=["evt_006"],
        )
        assert success is True
        mock_maturity.assert_called_once()  # mature but confidence dropping below 0.5: LLM rescore
        update_call = repo.update_skill_by_id.call_args
        updates = update_call[0][1]
        assert "maturity_score" in updates
        assert updates["maturity_score"] == 0.65

    @pytest.mark.asyncio
    @patch("memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor._compute_embedding")
    @patch("memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor._evaluate_maturity")
    async def test_skip_when_only_confidence_update_no_content_change(self, mock_maturity, mock_embed):
        """Confidence-only update (no content change) should NOT trigger maturity eval at all."""
        mock_embed.return_value = None
        mock_maturity.return_value = 0.9
        repo = _mock_skill_repo()

        existing = [_make_skill_record(confidence=0.7)]
        existing[0].maturity_score = 0.8
        result_obj = SkillExtractionResult()
        extractor = _build_extractor(maturity_threshold=0.6)

        op = {
            "action": "update",
            "index": 0,
            "data": {"confidence": 0.85},
        }
        success = await extractor._apply_update(
            op, existing, repo, result_obj,
        )
        assert success is True
        # No content/name/description changed, maturity eval should not trigger
        mock_maturity.assert_not_called()

    @pytest.mark.asyncio
    @patch("memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor._compute_embedding")
    @patch("memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor._evaluate_maturity")
    async def test_llm_maturity_when_major_content_change(self, mock_maturity, mock_embed):
        """Major content change (ratio >= 0.5) uses LLM maturity evaluation."""
        mock_embed.return_value = {"embedding": [0.5, 0.6], "vector_model": "m"}
        mock_maturity.return_value = 0.8
        repo = _mock_skill_repo()

        existing = [_make_skill_record(name="Old Name", confidence=0.7)]
        existing[0].maturity_score = 0.8
        result_obj = SkillExtractionResult()
        extractor = _build_extractor(maturity_threshold=0.6)

        op = {
            "action": "update",
            "index": 0,
            # Completely different content triggers change_ratio >= 0.5
            "data": {"content": "## Revised\n1. Totally new approach A\n2. Totally new approach B\n3. New step C\n4. New step D\n5. New step E"},
        }
        success = await extractor._apply_update(
            op, existing, repo, result_obj,
        )
        assert success is True
        mock_maturity.assert_called_once()  # LLM evaluation for major change
        update_call = repo.update_skill_by_id.call_args
        updates = update_call[0][1]
        assert "maturity_score" in updates
        assert updates["maturity_score"] == 0.8

    @pytest.mark.asyncio
    @patch("memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor._compute_embedding")
    @patch("memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor._evaluate_maturity")
    async def test_llm_maturity_on_hypothesis_promotion(self, mock_maturity, mock_embed):
        """Hypothesis promotion (Potential Steps -> Steps) uses LLM maturity evaluation."""
        mock_embed.return_value = None
        mock_maturity.return_value = 0.75
        repo = _mock_skill_repo()

        # Existing skill has Potential Steps (hypothesis from a failed case)
        existing = [_make_skill_record(confidence=0.5)]
        existing[0].content = (
            "## Potential Steps\n"
            "> Extracted from a failed case.\n"
            "1. Try approach X\n"
            "   - How: run command X\n"
            "   - Check: verify output\n"
            "\n## Pitfalls\n- Avoid Y"
        )
        existing[0].maturity_score = 0.4
        result_obj = SkillExtractionResult()
        extractor = _build_extractor(maturity_threshold=0.6)

        # New content promotes to verified Steps
        op = {
            "action": "update",
            "index": 0,
            "data": {
                "content": (
                    "## Steps\n"
                    "1. Execute approach X\n"
                    "   - How: run verified command X\n"
                    "   - Check: verify output\n"
                    "2. Follow up with Z\n"
                    "   - How: run command Z\n"
                    "   - Check: confirm result\n"
                    "\n## Pitfalls\n- Avoid Y"
                ),
                "confidence": 0.6,
            },
        }
        success = await extractor._apply_update(
            op, existing, repo, result_obj,
            source_case_ids=["evt_promotion"],
        )
        assert success is True
        mock_maturity.assert_called_once()  # LLM used for hypothesis promotion
        update_call = repo.update_skill_by_id.call_args
        updates = update_call[0][1]
        assert updates["maturity_score"] == 0.75


# ===========================================================================
# _format_cases None quality_score tests
# ===========================================================================


class TestFormatCasesNoneQuality:
    """Tests for quality_score=None handling in _format_cases."""

    def test_none_quality_defaults_to_0_5(self):
        extractor = _build_extractor()
        case = _make_case_record()
        case.quality_score = None
        result = extractor._format_cases([case])
        parsed = json.loads(result)
        assert parsed[0]["quality_score"] == 0.5


# ===========================================================================
# content_changed fix tests
# ===========================================================================


class TestContentChangedFix:
    """Tests verifying content_changed only triggers on actual changes."""

    @pytest.mark.asyncio
    @patch("memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor._compute_embedding")
    @patch("memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor._evaluate_maturity")
    async def test_same_content_no_maturity_eval(self, mock_maturity, mock_embed):
        """When update has same content as existing, maturity eval is NOT triggered."""
        mock_embed.return_value = None
        mock_maturity.return_value = 0.9
        repo = _mock_skill_repo()

        original_content = "## Steps\n1. Design schema\n2. Implement endpoints\n3. Add validation\n4. Write tests\n5. Deploy"
        existing = [_make_skill_record(content=original_content, confidence=0.7)]
        existing[0].maturity_score = 0.8
        result_obj = SkillExtractionResult()
        extractor = _build_extractor(maturity_threshold=0.6)

        op = {
            "action": "update",
            "index": 0,
            "data": {
                "content": original_content,  # same content
                "confidence": 0.8,
            },
        }
        success = await extractor._apply_update(
            op, existing, repo, result_obj,
        )
        assert success is True
        mock_maturity.assert_not_called()

    @pytest.mark.asyncio
    @patch("memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor._compute_embedding")
    @patch("memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor._evaluate_maturity")
    async def test_same_name_no_re_embed(self, mock_maturity, mock_embed):
        """When update has same name as existing, re-embed is NOT triggered."""
        mock_embed.return_value = {"embedding": [0.5], "vector_model": "m"}
        mock_maturity.return_value = 0.8
        repo = _mock_skill_repo()

        existing = [_make_skill_record(name="API Development", confidence=0.7)]
        result_obj = SkillExtractionResult()
        extractor = _build_extractor()

        op = {
            "action": "update",
            "index": 0,
            "data": {"name": "API Development"},  # same name
        }
        success = await extractor._apply_update(
            op, existing, repo, result_obj,
        )
        assert success is True
        mock_embed.assert_not_called()


# ===========================================================================
# _evaluate_maturity edge-case tests
# ===========================================================================


class TestEvaluateMaturityEdgeCases:
    """Additional edge-case tests for _evaluate_maturity."""

    @pytest.mark.asyncio
    async def test_zero_scores_returns_zero(self):
        """All dimension scores of 0 produce a 0.0 maturity score."""
        response = json.dumps({
            "completeness": 0, "executability": 0, "evidence": 0, "clarity": 0,
        })
        extractor = _build_extractor()
        extractor.maturity_prompt = "{name}{description}{content}{confidence}"
        extractor.llm_provider.generate = AsyncMock(return_value=response)
        score = await extractor._evaluate_maturity("N", "D", "C", 0.5)
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_negative_scores_clamped_to_zero(self):
        """Negative dimension totals are clamped to 0.0."""
        response = json.dumps({
            "completeness": -5, "executability": -5, "evidence": -5, "clarity": -5,
        })
        extractor = _build_extractor()
        extractor.maturity_prompt = "{name}{description}{content}{confidence}"
        extractor.llm_provider.generate = AsyncMock(return_value=response)
        score = await extractor._evaluate_maturity("N", "D", "C", 0.5)
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_overflow_scores_clamped_to_one(self):
        """Scores that exceed 20 total are clamped to 1.0."""
        response = json.dumps({
            "completeness": 10, "executability": 10, "evidence": 10, "clarity": 10,
        })
        extractor = _build_extractor()
        extractor.maturity_prompt = "{name}{description}{content}{confidence}"
        extractor.llm_provider.generate = AsyncMock(return_value=response)
        score = await extractor._evaluate_maturity("N", "D", "C", 0.5)
        assert score == 1.0

    @pytest.mark.asyncio
    async def test_non_numeric_dimension_raises_returns_none(self):
        """Non-numeric dimension value causes ValueError, caught and returns None."""
        response = json.dumps({
            "completeness": "high", "executability": 4, "evidence": 4, "clarity": 4,
        })
        extractor = _build_extractor()
        extractor.maturity_prompt = "{name}{description}{content}{confidence}"
        extractor.llm_provider.generate = AsyncMock(return_value=response)
        score = await extractor._evaluate_maturity("N", "D", "C", 0.5)
        assert score is None

    @pytest.mark.asyncio
    async def test_none_name_description_content_in_prompt(self):
        """None values for name/description/content are formatted as empty strings."""
        response = json.dumps({
            "completeness": 3, "executability": 3, "evidence": 3, "clarity": 3,
        })
        extractor = _build_extractor()
        extractor.maturity_prompt = "name={name}|desc={description}|content={content}|conf={confidence}"

        captured_prompt = None

        async def _capture(prompt, **kwargs):
            nonlocal captured_prompt
            captured_prompt = prompt
            return response

        extractor.llm_provider.generate = _capture
        score = await extractor._evaluate_maturity(None, None, None, 0.7)
        assert score is not None
        assert captured_prompt == "name=|desc=|content=|conf=0.7"


# ===========================================================================
# _content_change_ratio tests
# ===========================================================================


class TestContentChangeRatio:
    """Tests for AgentSkillExtractor._content_change_ratio."""

    def test_both_empty(self):
        assert AgentSkillExtractor._content_change_ratio("", "") == 0.0

    def test_both_none(self):
        assert AgentSkillExtractor._content_change_ratio(None, None) == 0.0

    def test_old_empty_new_filled(self):
        assert AgentSkillExtractor._content_change_ratio("", "new content") == 1.0

    def test_old_none_new_filled(self):
        assert AgentSkillExtractor._content_change_ratio(None, "new content") == 1.0

    def test_old_filled_new_empty(self):
        assert AgentSkillExtractor._content_change_ratio("old content", "") == 1.0

    def test_identical_strings(self):
        text = "## Steps\n1. Do this\n2. Do that"
        assert AgentSkillExtractor._content_change_ratio(text, text) == 0.0

    def test_minor_change(self):
        old = "## Steps\n1. Design schema\n2. Implement endpoints\n3. Add validation\n4. Write tests\n5. Deploy"
        new = "## Steps\n1. Design schema\n2. Implement endpoints\n3. Add validation\n4. Write tests\n5. Deploy to prod"
        ratio = AgentSkillExtractor._content_change_ratio(old, new)
        assert 0.0 < ratio < 0.15  # small change

    def test_major_rewrite(self):
        old = "## Steps\n1. Design schema\n2. Implement endpoints"
        new = "## Totally New\n1. Different approach A\n2. Different approach B"
        ratio = AgentSkillExtractor._content_change_ratio(old, new)
        assert ratio >= 0.3  # significant change

    def test_result_range_0_to_1(self):
        """Result is always in [0.0, 1.0]."""
        ratio = AgentSkillExtractor._content_change_ratio("aaa", "zzz")
        assert 0.0 <= ratio <= 1.0


# ===========================================================================
# Maturity re-evaluation trigger logic edge cases in _apply_update
# ===========================================================================


class TestMaturityReevalEdgeCases:
    """Edge cases for maturity re-evaluation within _apply_update."""

    @pytest.mark.asyncio
    @patch("memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor._compute_embedding")
    @patch("memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor._evaluate_maturity")
    async def test_trivial_change_skips_even_if_immature(self, mock_maturity, mock_embed):
        """Change ratio < 10% skips maturity eval even when skill is immature."""
        mock_embed.return_value = None
        mock_maturity.return_value = 0.9
        repo = _mock_skill_repo()

        original = "## Steps\n1. Design schema\n2. Implement endpoints\n3. Add validation\n4. Write tests\n5. Deploy"
        # Tiny tweak: "Deploy" -> "Deploy now"
        tweaked = "## Steps\n1. Design schema\n2. Implement endpoints\n3. Add validation\n4. Write tests\n5. Deploy now"

        existing = [_make_skill_record(content=original, confidence=0.7)]
        existing[0].maturity_score = 0.3  # immature
        result_obj = SkillExtractionResult()
        extractor = _build_extractor(maturity_threshold=0.6)

        op = {"action": "update", "index": 0, "data": {"content": tweaked, "confidence": 0.8}}
        success = await extractor._apply_update(op, existing, repo, result_obj)
        assert success is True
        mock_maturity.assert_not_called()

    @pytest.mark.asyncio
    @patch("memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor._compute_embedding")
    @patch("memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor._evaluate_maturity")
    async def test_llm_maturity_on_major_content_change_returns_none(self, mock_maturity, mock_embed):
        """Major content change (ratio >= 0.5) triggers LLM; if LLM returns None, maturity_score is not updated."""
        mock_embed.return_value = None
        mock_maturity.return_value = None  # LLM fails
        repo = _mock_skill_repo()

        existing = [_make_skill_record(confidence=0.7)]
        existing[0].maturity_score = 0.4
        result_obj = SkillExtractionResult()
        extractor = _build_extractor(maturity_threshold=0.6)

        op = {
            "action": "update", "index": 0,
            "data": {"content": "## Revised\n1. Totally new A\n2. Totally new B\n3. New C\n4. New D\n5. New E"},
        }
        success = await extractor._apply_update(op, existing, repo, result_obj)
        assert success is True
        mock_maturity.assert_called_once()  # LLM called for major change
        updates = repo.update_skill_by_id.call_args[0][1]
        assert "maturity_score" not in updates  # LLM returned None, not updated

    @pytest.mark.asyncio
    @patch("memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor._compute_embedding")
    @patch("memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor._evaluate_maturity")
    async def test_name_change_only_skips_maturity_due_to_trivial_content_ratio(self, mock_maturity, mock_embed):
        """Name-only change has content change_ratio=0.0, which is < trivial threshold — maturity eval skipped."""
        mock_embed.return_value = {"embedding": [0.5], "vector_model": "m"}
        mock_maturity.return_value = 0.75
        repo = _mock_skill_repo()

        existing = [_make_skill_record(name="Old Name", confidence=0.7)]
        existing[0].maturity_score = 0.4  # immature
        result_obj = SkillExtractionResult()
        extractor = _build_extractor(maturity_threshold=0.6)

        op = {"action": "update", "index": 0, "data": {"name": "Completely New Name"}}
        success = await extractor._apply_update(op, existing, repo, result_obj)
        assert success is True
        # content_change_ratio=0.0 < MATURITY_TRIVIAL_CHANGE_RATIO, so maturity eval skipped
        mock_maturity.assert_not_called()
        updates = repo.update_skill_by_id.call_args[0][1]
        assert "maturity_score" not in updates

    @pytest.mark.asyncio
    @patch("memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor._compute_embedding")
    @patch("memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor._evaluate_maturity")
    async def test_description_change_only_skips_maturity_due_to_trivial_content_ratio(self, mock_maturity, mock_embed):
        """Description-only change has content change_ratio=0.0, which is < trivial threshold — maturity eval skipped."""
        mock_embed.return_value = {"embedding": [0.5], "vector_model": "m"}
        mock_maturity.return_value = 0.8
        repo = _mock_skill_repo()

        existing = [_make_skill_record(description="Old desc", confidence=0.7)]
        existing[0].maturity_score = 0.4  # immature
        result_obj = SkillExtractionResult()
        extractor = _build_extractor(maturity_threshold=0.6)

        op = {"action": "update", "index": 0, "data": {"description": "Completely different description"}}
        success = await extractor._apply_update(op, existing, repo, result_obj)
        assert success is True
        # content_change_ratio=0.0 < MATURITY_TRIVIAL_CHANGE_RATIO, so maturity eval skipped
        mock_maturity.assert_not_called()
        updates = repo.update_skill_by_id.call_args[0][1]
        assert "maturity_score" not in updates

    @pytest.mark.asyncio
    @patch("memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor._compute_embedding")
    @patch("memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor._evaluate_maturity")
    async def test_name_plus_major_content_change_uses_llm(self, mock_maturity, mock_embed):
        """Name change combined with major content change (ratio >= 0.5) uses LLM maturity."""
        mock_embed.return_value = {"embedding": [0.5], "vector_model": "m"}
        mock_maturity.return_value = 0.75
        repo = _mock_skill_repo()

        existing = [_make_skill_record(name="Old Name", confidence=0.7)]
        existing[0].maturity_score = 0.4  # immature
        result_obj = SkillExtractionResult()
        extractor = _build_extractor(maturity_threshold=0.6)

        op = {
            "action": "update", "index": 0,
            "data": {
                "name": "Completely New Name",
                "content": "## Revised\n1. Totally new A\n2. Totally new B\n3. New C\n4. New D\n5. New E",
            },
        }
        success = await extractor._apply_update(op, existing, repo, result_obj)
        assert success is True
        mock_maturity.assert_called_once()  # LLM for major change
        updates = repo.update_skill_by_id.call_args[0][1]
        assert "maturity_score" in updates
        assert updates["maturity_score"] == 0.75

    @pytest.mark.asyncio
    @patch("memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor._compute_embedding")
    @patch("memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor._evaluate_maturity")
    async def test_llm_maturity_score_written_on_major_change(self, mock_maturity, mock_embed):
        """Major content change (ratio >= 0.5) triggers LLM and writes its score."""
        mock_embed.return_value = None
        mock_maturity.return_value = 0.85
        repo = _mock_skill_repo()

        existing = [_make_skill_record(confidence=0.7)]
        existing[0].maturity_score = 0.4
        result_obj = SkillExtractionResult()
        extractor = _build_extractor(maturity_threshold=0.6)

        op = {
            "action": "update", "index": 0,
            "data": {"content": "## Revised\n1. Totally new A\n2. Totally new B\n3. New C\n4. New D\n5. New E"},
        }
        success = await extractor._apply_update(op, existing, repo, result_obj)
        assert success is True
        mock_maturity.assert_called_once()  # LLM used for major change
        updates = repo.update_skill_by_id.call_args[0][1]
        assert "maturity_score" in updates
        assert updates["maturity_score"] == 0.85


# ===========================================================================
# _evaluate_maturity_heuristic unit tests
# ===========================================================================




# ===========================================================================
# _is_hypothesis_promotion unit tests
# ===========================================================================


class TestIsHypothesisPromotion:
    """Direct unit tests for hypothesis promotion detection."""

    def test_classic_promotion(self):
        """Potential Steps -> Steps is a promotion."""
        old = "## Potential Steps\n1. Try X\n\n## Pitfalls\n- Y failed"
        new = "## Steps\n1. Do X\n2. Do Y\n\n## Pitfalls\n- Y failed"
        assert AgentSkillExtractor._is_hypothesis_promotion(old, new) is True

    def test_not_promotion_both_steps(self):
        """Steps -> Steps is NOT a promotion."""
        old = "## Steps\n1. Do X"
        new = "## Steps\n1. Do X\n2. Do Y"
        assert AgentSkillExtractor._is_hypothesis_promotion(old, new) is False

    def test_not_promotion_both_potential(self):
        """Potential Steps -> Potential Steps is NOT a promotion."""
        old = "## Potential Steps\n1. Try X"
        new = "## Potential Steps\n1. Try X\n2. Try Y"
        assert AgentSkillExtractor._is_hypothesis_promotion(old, new) is False

    def test_not_promotion_steps_to_potential(self):
        """Steps -> Potential Steps is NOT a promotion (regression)."""
        old = "## Steps\n1. Do X"
        new = "## Potential Steps\n1. Try X"
        assert AgentSkillExtractor._is_hypothesis_promotion(old, new) is False

    def test_not_promotion_new_still_has_potential(self):
        """If new content has BOTH Steps and Potential Steps, not a clean promotion."""
        old = "## Potential Steps\n1. Try X"
        new = "## Steps\n1. Do X\n\n## Potential Steps\n1. Try Y"
        assert AgentSkillExtractor._is_hypothesis_promotion(old, new) is False

    def test_empty_old_content(self):
        old = ""
        new = "## Steps\n1. Do X"
        assert AgentSkillExtractor._is_hypothesis_promotion(old, new) is False

    def test_empty_new_content(self):
        old = "## Potential Steps\n1. Try X"
        new = ""
        assert AgentSkillExtractor._is_hypothesis_promotion(old, new) is False

    def test_both_empty(self):
        assert AgentSkillExtractor._is_hypothesis_promotion("", "") is False

    def test_none_inputs(self):
        assert AgentSkillExtractor._is_hypothesis_promotion(None, None) is False
        assert AgentSkillExtractor._is_hypothesis_promotion(None, "## Steps\n1. X") is False
        assert AgentSkillExtractor._is_hypothesis_promotion("## Potential Steps\n1. X", None) is False

    def test_case_sensitivity(self):
        """Heading detection is case-sensitive — '## steps' is not '## Steps'."""
        old = "## Potential Steps\n1. Try X"
        new = "## steps\n1. Do X"
        assert AgentSkillExtractor._is_hypothesis_promotion(old, new) is False

    def test_heading_with_extra_spaces(self):
        """Extra spaces after ## should still match."""
        old = "##  Potential Steps\n1. Try X"
        new = "##  Steps\n1. Do X"
        # regex is r"^##\s+Steps" and r"^##\s+Potential Steps" so \s+ matches multiple spaces
        assert AgentSkillExtractor._is_hypothesis_promotion(old, new) is True

    def test_heading_not_at_line_start(self):
        """Heading must be at line start (indented headings should not match)."""
        old = "  ## Potential Steps\n1. Try X"
        new = "## Steps\n1. Do X"
        # "  ##" does not match "^##"
        assert AgentSkillExtractor._is_hypothesis_promotion(old, new) is False


# ===========================================================================
# _apply_update integration: heuristic vs LLM maturity path tests
# ===========================================================================


class TestApplyUpdateMaturityPath:
    """Integration tests verifying the correct maturity evaluation path is chosen."""

    @pytest.mark.asyncio
    @patch("memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor._compute_embedding")
    @patch("memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor._evaluate_maturity")
    async def test_promotion_with_llm_failure_no_maturity_update(self, mock_maturity, mock_embed):
        """When hypothesis promotion triggers LLM but LLM returns None, maturity_score is NOT written."""
        mock_embed.return_value = None
        mock_maturity.return_value = None  # LLM failed
        repo = _mock_skill_repo()

        existing = [_make_skill_record(confidence=0.5)]
        existing[0].content = "## Potential Steps\n1. Try X\n   - How: cmd\n   - Check: ok\n\n## Pitfalls\n- Y"
        existing[0].maturity_score = 0.3
        result_obj = SkillExtractionResult()
        extractor = _build_extractor(maturity_threshold=0.6)

        op = {
            "action": "update", "index": 0,
            "data": {
                "content": "## Steps\n1. Do X\n   - How: verified cmd\n   - Check: ok\n2. Do Z\n   - How: cmd Z\n   - Check: ok\n\n## Pitfalls\n- Y",
                "confidence": 0.6,
            },
        }
        success = await extractor._apply_update(op, existing, repo, result_obj, source_case_ids=["evt_p"])
        assert success is True
        mock_maturity.assert_called_once()  # LLM was attempted
        updates = repo.update_skill_by_id.call_args[0][1]
        assert "maturity_score" not in updates  # LLM returned None, no update

    @pytest.mark.asyncio
    @patch("memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor._compute_embedding")
    @patch("memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor._evaluate_maturity")
    async def test_major_change_always_calls_llm(self, mock_maturity, mock_embed):
        """Large content change (>= 40%) always triggers LLM maturity, regardless of other factors."""
        mock_embed.return_value = None
        mock_maturity.return_value = 0.9
        repo = _mock_skill_repo()

        existing = [_make_skill_record(confidence=0.6)]
        existing[0].content = "## Steps\n1. Old step A\n2. Old step B\n3. Old C\n4. Old D\n5. Old E"
        existing[0].maturity_score = 0.3  # very immature
        result_obj = SkillExtractionResult()
        extractor = _build_extractor(maturity_threshold=0.6)

        op = {
            "action": "update", "index": 0,
            "data": {
                "content": "## Steps\n1. Completely new X\n2. Completely new Y\n3. New Z\n4. New W\n5. New V",
                "confidence": 0.3,  # confidence dropping
            },
        }
        success = await extractor._apply_update(op, existing, repo, result_obj)
        assert success is True
        mock_maturity.assert_called_once()  # major change: always LLM
        updates = repo.update_skill_by_id.call_args[0][1]
        assert updates["maturity_score"] == 0.9

    @pytest.mark.asyncio
    @patch("memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor._compute_embedding")
    @patch("memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor._evaluate_maturity")
    async def test_llm_score_used_for_major_content_changes(self, mock_maturity, mock_embed):
        """Major content changes (ratio >= 0.5) trigger LLM maturity evaluation."""
        mock_embed.return_value = None
        repo = _mock_skill_repo()
        extractor = _build_extractor()

        # First call: poor content -> LLM returns low score
        mock_maturity.return_value = 0.4
        existing_poor = [_make_skill_record(confidence=0.5)]
        existing_poor[0].content = "## Steps\n1. Old\n2. Old\n3. Old\n4. Old\n5. Old"
        result_poor = SkillExtractionResult()
        op_poor = {
            "action": "update", "index": 0,
            "data": {"content": "## Steps\n1. Do step X here\n2. Do step Y here\n3. Do step Z\n4. Do step W\n5. Do step V\n"},
        }
        await extractor._apply_update(op_poor, existing_poor, repo, result_poor)
        score_poor = repo.update_skill_by_id.call_args[0][1]["maturity_score"]

        # Second call: rich content -> LLM returns high score
        mock_maturity.return_value = 0.9
        repo.reset_mock()
        existing_rich = [_make_skill_record(confidence=0.8)]
        existing_rich[0].content = "## Steps\n1. Old\n2. Old\n3. Old\n4. Old\n5. Old"
        result_rich = SkillExtractionResult()
        rich_content = (
            "## Steps\n"
            "1. Diagnose\n"
            "   - How: Check logs with `grep ERROR`\n"
            "   - e.g., `grep -i error /var/log/app.log`\n"
            "   - Check: Error found\n"
            "2. Fix config\n"
            "   - How: Edit `config.yaml`\n"
            "   - e.g., `vim config.yaml`\n"
            "   - Check: Validated\n"
            "3. Restart\n"
            "   - How: `systemctl restart app`\n"
            "   - Check: Active\n"
            "4. Verify\n"
            "   - How: `curl localhost/health`\n"
            "   - Check: 200 OK\n"
            "\n## Pitfalls\n- Don't skip config validation\n"
        )
        op_rich = {
            "action": "update", "index": 0,
            "data": {"content": rich_content},
        }
        await extractor._apply_update(op_rich, existing_rich, repo, result_rich)
        score_rich = repo.update_skill_by_id.call_args[0][1]["maturity_score"]

        assert mock_maturity.call_count == 2  # LLM called for both major changes
        assert score_rich > score_poor

    @pytest.mark.asyncio
    @patch("memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor._compute_embedding")
    @patch("memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor._evaluate_maturity")
    async def test_trivial_change_skips_both_heuristic_and_llm(self, mock_maturity, mock_embed):
        """Content change < 10% skips all maturity evaluation (neither heuristic nor LLM)."""
        mock_embed.return_value = None
        repo = _mock_skill_repo()

        original_content = "## Steps\n1. Step A details here\n2. Step B details here\n3. Step C\n4. Step D\n5. Step E"
        # Very minor tweak (< 10% change)
        tweaked_content = "## Steps\n1. Step A details here\n2. Step B details here\n3. Step C\n4. Step D\n5. Step F"

        existing = [_make_skill_record(confidence=0.7)]
        existing[0].content = original_content
        existing[0].maturity_score = 0.5
        result_obj = SkillExtractionResult()
        extractor = _build_extractor()

        op = {
            "action": "update", "index": 0,
            "data": {"content": tweaked_content},
        }
        success = await extractor._apply_update(op, existing, repo, result_obj)
        assert success is True
        mock_maturity.assert_not_called()
        updates = repo.update_skill_by_id.call_args[0][1]
        # maturity_score should NOT be in updates for trivial changes
        assert "maturity_score" not in updates


# ===========================================================================
# key_insight field tests
# ===========================================================================


class TestKeyInsightInFormatCases:
    """Tests for key_insight inclusion in _format_cases."""

    def test_key_insight_included_when_present(self):
        extractor = _build_extractor()
        cases = [_make_case_record(key_insight="Switched from REST to GraphQL")]
        result = extractor._format_cases(cases)
        parsed = json.loads(result)
        assert parsed[0]["key_insight"] == "Switched from REST to GraphQL"

    def test_key_insight_excluded_when_none(self):
        extractor = _build_extractor()
        cases = [_make_case_record(key_insight=None)]
        result = extractor._format_cases(cases)
        parsed = json.loads(result)
        assert "key_insight" not in parsed[0]

    def test_key_insight_excluded_when_empty(self):
        extractor = _build_extractor()
        cases = [_make_case_record(key_insight="")]
        result = extractor._format_cases(cases)
        parsed = json.loads(result)
        assert "key_insight" not in parsed[0]

    def test_mixed_cases_with_and_without_key_insight(self):
        extractor = _build_extractor()
        cases = [
            _make_case_record(task_intent="A", key_insight="Pivoted approach"),
            _make_case_record(task_intent="B", key_insight=None),
        ]
        result = extractor._format_cases(cases)
        parsed = json.loads(result)
        assert "key_insight" in parsed[0]
        assert "key_insight" not in parsed[1]


# ===========================================================================
# _summarize_case_for_prompt tests
# ===========================================================================


class TestSummarizeCaseForPrompt:
    """Tests for AgentSkillExtractor._summarize_case_for_prompt."""

    def test_basic_summary(self):
        extractor = _build_extractor()
        case = _make_case_record(record_id="case_001")
        result = extractor._summarize_case_for_prompt(case)
        assert result["task_intent"] == "Build a REST API"
        assert result["quality_score"] == 0.8
        assert "source_case_id" not in result

    def test_key_insight_included(self):
        extractor = _build_extractor()
        case = _make_case_record(key_insight="Retry with exponential backoff")
        result = extractor._summarize_case_for_prompt(case)
        assert result["key_insight"] == "Retry with exponential backoff"

    def test_key_insight_excluded_when_none(self):
        extractor = _build_extractor()
        case = _make_case_record(key_insight=None)
        result = extractor._summarize_case_for_prompt(case)
        assert "key_insight" not in result

    @patch.object(AgentSkillExtractor, "_get_tokenizer")
    def test_approach_truncated(self, mock_tok):
        from tiktoken import get_encoding
        mock_tok.return_value = get_encoding("o200k_base")
        extractor = _build_extractor()
        long_approach = "x " * 500
        case = _make_case_record(approach=long_approach)
        result = extractor._summarize_case_for_prompt(case, max_approach_tokens=10)
        assert result["approach"].endswith("... [omitted]")
        assert len(result["approach"]) < len(long_approach)

    def test_no_approach_omitted(self):
        extractor = _build_extractor()
        case = _make_case_record(approach=None)
        result = extractor._summarize_case_for_prompt(case)
        assert "approach" not in result


# ===========================================================================
# _truncate_text tests
# ===========================================================================


@patch.object(AgentSkillExtractor, "_get_tokenizer")
class TestTruncateText:
    """Tests for AgentSkillExtractor._truncate_text (token-based)."""

    @staticmethod
    def _real_tokenizer():
        from tiktoken import get_encoding
        return get_encoding("o200k_base")

    def test_short_text_unchanged(self, mock_tok):
        mock_tok.return_value = self._real_tokenizer()
        assert AgentSkillExtractor._truncate_text("hello", max_tokens=10) == "hello"

    def test_exact_tokens_unchanged(self, mock_tok):
        mock_tok.return_value = self._real_tokenizer()
        # "hello" is 1 token in o200k_base
        assert AgentSkillExtractor._truncate_text("hello", max_tokens=1) == "hello"

    def test_long_text_truncated_with_default_suffix(self, mock_tok):
        mock_tok.return_value = self._real_tokenizer()
        long_text = "word " * 200
        result = AgentSkillExtractor._truncate_text(long_text, max_tokens=10)
        assert result.endswith("... [omitted]")
        assert len(result) < len(long_text)

    def test_none_returns_none(self, mock_tok):
        mock_tok.return_value = self._real_tokenizer()
        assert AgentSkillExtractor._truncate_text(None, max_tokens=10) is None

    def test_empty_returns_empty(self, mock_tok):
        mock_tok.return_value = self._real_tokenizer()
        assert AgentSkillExtractor._truncate_text("", max_tokens=10) == ""

    def test_whitespace_stripped(self, mock_tok):
        mock_tok.return_value = self._real_tokenizer()
        assert AgentSkillExtractor._truncate_text("  hello  ", max_tokens=100) == "hello"

    def test_custom_suffix(self, mock_tok):
        mock_tok.return_value = self._real_tokenizer()
        long_text = "word " * 200
        result = AgentSkillExtractor._truncate_text(long_text, max_tokens=10, suffix="...")
        assert result.endswith("...")
        assert not result.endswith("... [omitted]")
        assert len(result) < len(long_text)

    def test_default_suffix(self, mock_tok):
        mock_tok.return_value = self._real_tokenizer()
        long_text = "word " * 200
        result = AgentSkillExtractor._truncate_text(long_text, max_tokens=10)
        assert result.endswith("... [omitted]")


# ===========================================================================
# Description truncation in _apply_add / _apply_update
# ===========================================================================


class TestDescriptionTruncation:
    """Tests for description truncation via MAX_DESCRIPTION_TOKENS."""

    @pytest.mark.asyncio
    @patch.object(AgentSkillExtractor, "_get_tokenizer")
    @patch("memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor._compute_embedding")
    @patch("memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor._evaluate_maturity")
    async def test_add_truncates_long_description(self, mock_maturity, mock_embed, mock_tok):
        from tiktoken import get_encoding
        mock_tok.return_value = get_encoding("o200k_base")
        mock_embed.return_value = {"embedding": [0.1], "vector_model": "test"}
        mock_maturity.return_value = 0.8
        repo = _mock_skill_repo()
        extractor = _build_extractor()

        long_desc = "word " * 2000
        op = {
            "action": "add",
            "data": {
                "name": "Skill",
                "description": long_desc,
                "content": "## Steps\n1. Analyze the requirements carefully\n2. Build the implementation with tests\n3. Check all edge cases thoroughly\n4. Deploy to staging environment\n5. Verify in production setup",
                "confidence": 0.7,
            },
        }
        import infra_layer.adapters.out.persistence.document.memory.agent_skill as skill_mod
        with patch.object(
            skill_mod, "AgentSkillRecord",
            side_effect=lambda **kwargs: SimpleNamespace(**kwargs, id="new_001"),
        ) as mock_cls:
            await extractor._apply_add(op, "c", "g", "u", repo, source_case_ids=["e1"])
            saved_desc = mock_cls.call_args[1]["description"]
        assert len(saved_desc) < len(long_desc)
        assert saved_desc.endswith("...")

    @pytest.mark.asyncio
    @patch.object(AgentSkillExtractor, "_get_tokenizer")
    @patch("memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor._compute_embedding")
    @patch("memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor._evaluate_maturity")
    async def test_add_short_description_not_truncated(self, mock_maturity, mock_embed, mock_tok):
        from tiktoken import get_encoding
        mock_tok.return_value = get_encoding("o200k_base")
        mock_embed.return_value = {"embedding": [0.1], "vector_model": "test"}
        mock_maturity.return_value = 0.8
        repo = _mock_skill_repo()
        extractor = _build_extractor()

        op = {
            "action": "add",
            "data": {
                "name": "Skill",
                "description": "Short desc",
                "content": "## Steps\n1. Analyze the requirements carefully\n2. Build the implementation with tests\n3. Check all edge cases thoroughly\n4. Deploy to staging environment\n5. Verify in production setup",
                "confidence": 0.7,
            },
        }
        import infra_layer.adapters.out.persistence.document.memory.agent_skill as skill_mod
        with patch.object(
            skill_mod, "AgentSkillRecord",
            side_effect=lambda **kwargs: SimpleNamespace(**kwargs, id="new_001"),
        ) as mock_cls:
            await extractor._apply_add(op, "c", "g", "u", repo, source_case_ids=["e1"])
            saved_desc = mock_cls.call_args[1]["description"]
        assert saved_desc == "Short desc"
        assert not saved_desc.endswith("...")

    @pytest.mark.asyncio
    @patch.object(AgentSkillExtractor, "_get_tokenizer")
    @patch("memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor._compute_embedding")
    @patch("memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor._evaluate_maturity")
    async def test_update_truncates_long_description(self, mock_maturity, mock_embed, mock_tok):
        from tiktoken import get_encoding
        mock_tok.return_value = get_encoding("o200k_base")
        mock_embed.return_value = {"embedding": [0.1], "vector_model": "test"}
        mock_maturity.return_value = 0.8
        repo = _mock_skill_repo()
        extractor = _build_extractor()

        long_desc = "word " * 2000
        record = _make_skill_record()
        op = {
            "action": "update",
            "index": 0,
            "data": {"description": long_desc},
        }
        result = SkillExtractionResult()
        await extractor._apply_update(
            op, [record], repo, result, source_case_ids=["e1"]
        )
        update_dict = repo.update_skill_by_id.call_args[0][1]
        saved_desc = update_dict["description"]
        assert len(saved_desc) < len(long_desc)
        assert saved_desc.endswith("...")

    @pytest.mark.asyncio
    @patch.object(AgentSkillExtractor, "_get_tokenizer")
    @patch("memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor._compute_embedding")
    @patch("memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor._evaluate_maturity")
    async def test_update_short_description_not_truncated(self, mock_maturity, mock_embed, mock_tok):
        from tiktoken import get_encoding
        mock_tok.return_value = get_encoding("o200k_base")
        mock_embed.return_value = {"embedding": [0.1], "vector_model": "test"}
        mock_maturity.return_value = 0.8
        repo = _mock_skill_repo()
        extractor = _build_extractor()

        record = _make_skill_record()
        op = {
            "action": "update",
            "index": 0,
            "data": {"description": "New short desc"},
        }
        result = SkillExtractionResult()
        await extractor._apply_update(
            op, [record], repo, result, source_case_ids=["e1"]
        )
        update_dict = repo.update_skill_by_id.call_args[0][1]
        assert update_dict["description"] == "New short desc"

    @pytest.mark.asyncio
    async def test_update_empty_description_not_added_to_updates(self):
        repo = _mock_skill_repo()
        extractor = _build_extractor()

        record = _make_skill_record()
        op = {
            "action": "update",
            "index": 0,
            "data": {"description": "", "confidence": 0.9},
        }
        result = SkillExtractionResult()
        await extractor._apply_update(op, [record], repo, result)
        update_dict = repo.update_skill_by_id.call_args[0][1]
        assert "description" not in update_dict


# ===========================================================================
# _format_existing_skills with case_history tests
# ===========================================================================


class TestFormatExistingSkillsWithCaseHistory:
    """Tests for _format_existing_skills when case_history is provided."""

    def test_supporting_cases_attached(self):
        extractor = _build_extractor()
        case = _make_case_record(
            task_intent="Deploy service", key_insight="Blue-green deploy", record_id="c1"
        )
        skill = _make_skill_record(source_case_ids=["c1"])
        result = extractor._format_existing_skills([skill], case_history=[case])
        parsed = json.loads(result)
        assert parsed[0]["supporting_case_count"] == 1
        assert len(parsed[0]["supporting_cases"]) == 1
        assert parsed[0]["supporting_cases"][0]["task_intent"] == "Deploy service"
        assert parsed[0]["supporting_cases"][0]["key_insight"] == "Blue-green deploy"

    def test_no_case_history_no_supporting_cases(self):
        extractor = _build_extractor()
        skill = _make_skill_record(source_case_ids=["c1"])
        result = extractor._format_existing_skills([skill], case_history=None)
        parsed = json.loads(result)
        assert "supporting_cases" not in parsed[0]

    def test_unmatched_case_ids_no_supporting_cases(self):
        extractor = _build_extractor()
        case = _make_case_record(record_id="c99")
        skill = _make_skill_record(source_case_ids=["c1"])
        result = extractor._format_existing_skills([skill], case_history=[case])
        parsed = json.loads(result)
        assert "supporting_cases" not in parsed[0]

    def test_max_support_cases_respected(self):
        extractor = _build_extractor()
        cases = [
            _make_case_record(task_intent=f"Task {i}", record_id=f"c{i}")
            for i in range(5)
        ]
        skill = _make_skill_record(source_case_ids=[f"c{i}" for i in range(5)])
        result = extractor._format_existing_skills(
            [skill], case_history=cases, max_support_cases=2
        )
        parsed = json.loads(result)
        assert parsed[0]["supporting_case_count"] == 5
        assert len(parsed[0]["supporting_cases"]) == 2

    def test_takes_most_recent_cases(self):
        """max_support_cases takes the last N (most recent) matched IDs."""
        extractor = _build_extractor()
        cases = [
            _make_case_record(task_intent=f"Task {i}", record_id=f"c{i}")
            for i in range(4)
        ]
        skill = _make_skill_record(source_case_ids=["c0", "c1", "c2", "c3"])
        result = extractor._format_existing_skills(
            [skill], case_history=cases, max_support_cases=2
        )
        parsed = json.loads(result)
        intents = [sc["task_intent"] for sc in parsed[0]["supporting_cases"]]
        assert intents == ["Task 2", "Task 3"]

    def test_empty_source_case_ids_no_supporting(self):
        extractor = _build_extractor()
        case = _make_case_record(record_id="c1")
        skill = _make_skill_record(source_case_ids=[])
        result = extractor._format_existing_skills([skill], case_history=[case])
        parsed = json.loads(result)
        assert "supporting_cases" not in parsed[0]


# ===========================================================================
# _load_case_history tests
# ===========================================================================


class TestLoadCaseHistory:
    """Tests for AgentSkillExtractor._load_case_history."""

    @pytest.mark.asyncio
    async def test_empty_skills_returns_empty(self):
        extractor = _build_extractor()
        result = await extractor._load_case_history([])
        assert result == []

    @pytest.mark.asyncio
    async def test_no_source_case_ids_returns_empty(self):
        extractor = _build_extractor()
        skill = _make_skill_record(source_case_ids=[])
        result = await extractor._load_case_history([skill])
        assert result == []

    @pytest.mark.asyncio
    @patch("core.di.utils.get_bean_by_type")
    async def test_loads_and_sorts_by_quality_desc(self, mock_get_bean):
        extractor = _build_extractor()
        mock_repo = AsyncMock()
        mock_get_bean.return_value = mock_repo

        records = [
            _make_case_record(task_intent="Low", quality_score=0.3, record_id="c1"),
            _make_case_record(task_intent="High", quality_score=0.9, record_id="c2"),
            _make_case_record(task_intent="Mid", quality_score=0.6, record_id="c3"),
        ]
        mock_repo.get_by_ids = AsyncMock(return_value=records)

        skill = _make_skill_record(source_case_ids=["c1", "c2", "c3"])
        result = await extractor._load_case_history([skill])

        assert len(result) == 3
        assert result[0].task_intent == "High"
        assert result[1].task_intent == "Mid"
        assert result[2].task_intent == "Low"

    @pytest.mark.asyncio
    @patch("core.di.utils.get_bean_by_type")
    async def test_max_cases_limit(self, mock_get_bean):
        extractor = _build_extractor()
        mock_repo = AsyncMock()
        mock_get_bean.return_value = mock_repo

        records = [
            _make_case_record(task_intent=f"T{i}", quality_score=0.5, record_id=f"c{i}")
            for i in range(10)
        ]
        mock_repo.get_by_ids = AsyncMock(return_value=records)

        skill = _make_skill_record(source_case_ids=[f"c{i}" for i in range(10)])
        result = await extractor._load_case_history([skill], max_cases=3)
        assert len(result) == 3

    @pytest.mark.asyncio
    @patch("core.di.utils.get_bean_by_type")
    async def test_deduplicates_case_ids_across_skills(self, mock_get_bean):
        extractor = _build_extractor()
        mock_repo = AsyncMock()
        mock_get_bean.return_value = mock_repo
        mock_repo.get_by_ids = AsyncMock(return_value=[])

        skill1 = _make_skill_record(source_case_ids=["c1", "c2"])
        skill2 = _make_skill_record(source_case_ids=["c2", "c3"])
        await extractor._load_case_history([skill1, skill2])

        called_ids = set(mock_repo.get_by_ids.call_args[0][0])
        assert called_ids == {"c1", "c2", "c3"}

    @pytest.mark.asyncio
    @patch("core.di.utils.get_bean_by_type")
    async def test_filters_empty_case_ids(self, mock_get_bean):
        extractor = _build_extractor()
        mock_repo = AsyncMock()
        mock_get_bean.return_value = mock_repo
        mock_repo.get_by_ids = AsyncMock(return_value=[])

        skill = _make_skill_record(source_case_ids=["c1", "", None, "  ", "c2"])
        await extractor._load_case_history([skill])

        called_ids = set(mock_repo.get_by_ids.call_args[0][0])
        assert "" not in called_ids
        assert "None" not in called_ids
        assert "c1" in called_ids
        assert "c2" in called_ids

    @pytest.mark.asyncio
    @patch("core.di.utils.get_bean_by_type")
    async def test_db_failure_returns_empty(self, mock_get_bean):
        extractor = _build_extractor()
        mock_get_bean.side_effect = Exception("DB down")

        skill = _make_skill_record(source_case_ids=["c1"])
        result = await extractor._load_case_history([skill])
        assert result == []


# ===========================================================================
# _json_default tests
# ===========================================================================


class TestSkillJsonDefault:
    """Tests for AgentSkillExtractor._json_default."""

    def test_datetime_to_isoformat(self):
        dt = datetime(2025, 6, 15, 14, 30, 0)
        assert AgentSkillExtractor._json_default(dt) == "2025-06-15T14:30:00"

    def test_non_serializable_to_str(self):
        result = AgentSkillExtractor._json_default({1, 2})
        assert isinstance(result, str)

    def test_none_to_str(self):
        assert AgentSkillExtractor._json_default(None) == "None"


# ===========================================================================
# _compute_embedding tests
# ===========================================================================


class TestSkillComputeEmbedding:
    """Tests for AgentSkillExtractor._compute_embedding."""

    @pytest.mark.asyncio
    async def test_empty_text_returns_none(self):
        extractor = _build_extractor()
        result = await extractor._compute_embedding("")
        assert result is None

    @pytest.mark.asyncio
    async def test_none_text_returns_none(self):
        extractor = _build_extractor()
        result = await extractor._compute_embedding(None)
        assert result is None

    @pytest.mark.asyncio
    @patch("agentic_layer.vectorize_service.get_vectorize_service")
    async def test_successful_embedding(self, mock_vs):
        import numpy as np
        mock_service = MagicMock()
        mock_service.get_embedding = AsyncMock(return_value=np.array([0.1, 0.2, 0.3]))
        mock_service.get_model_name.return_value = "test-model"
        mock_vs.return_value = mock_service

        extractor = _build_extractor()
        result = await extractor._compute_embedding("Build REST API")
        assert result is not None
        assert result["embedding"] == [0.1, 0.2, 0.3]
        assert result["vector_model"] == "test-model"

    @pytest.mark.asyncio
    @patch("agentic_layer.vectorize_service.get_vectorize_service")
    async def test_embedding_exception_returns_none(self, mock_vs):
        mock_vs.side_effect = Exception("service unavailable")
        extractor = _build_extractor()
        result = await extractor._compute_embedding("some text")
        assert result is None


# ===========================================================================
# _rescore_maturity tests
# ===========================================================================


class TestRescoreMaturity:
    """Tests for AgentSkillExtractor._rescore_maturity."""

    @pytest.mark.asyncio
    @patch("memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor._evaluate_maturity")
    async def test_updates_dict_with_score(self, mock_eval):
        mock_eval.return_value = 0.85
        extractor = _build_extractor()
        record = _make_skill_record()
        updates = {"confidence": 0.8}
        await extractor._rescore_maturity(updates, "New Name", "New Desc", "New Content", record)
        assert updates["maturity_score"] == 0.85
        mock_eval.assert_called_once_with(
            name="New Name", description="New Desc",
            content="New Content", confidence=0.8,
        )

    @pytest.mark.asyncio
    @patch("memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor._evaluate_maturity")
    async def test_llm_returns_none_no_update(self, mock_eval):
        mock_eval.return_value = None
        extractor = _build_extractor()
        record = _make_skill_record()
        updates = {}
        await extractor._rescore_maturity(updates, "N", "D", "C", record)
        assert "maturity_score" not in updates

    @pytest.mark.asyncio
    @patch("memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor._evaluate_maturity")
    async def test_falls_back_to_record_fields(self, mock_eval):
        mock_eval.return_value = 0.5
        extractor = _build_extractor()
        record = _make_skill_record(name="Old", description="Old desc", content="Old content")
        record.confidence = 0.6
        updates = {}
        await extractor._rescore_maturity(updates, "", "", "", record)
        mock_eval.assert_called_once_with(
            name="Old", description="Old desc",
            content="Old content", confidence=0.6,
        )


# ===========================================================================
# _format_cases edge cases
# ===========================================================================


class TestFormatCasesEdgeCases:
    """Additional edge case tests for _format_cases."""

    def test_none_quality_score_defaults(self):
        extractor = _build_extractor()
        case = _make_case_record(quality_score=None)
        result = extractor._format_cases([case])
        parsed = json.loads(result)
        assert parsed[0]["quality_score"] == 0.5

    def test_zero_quality_score_defaults(self):
        extractor = _build_extractor()
        case = _make_case_record(quality_score=0)
        result = extractor._format_cases([case])
        parsed = json.loads(result)
        # 0 is falsy, so `or 0.5` applies
        assert parsed[0]["quality_score"] == 0.5

    def test_timestamp_format(self):
        extractor = _build_extractor()
        case = _make_case_record(timestamp=datetime(2025, 6, 15, 14, 30, 0))
        result = extractor._format_cases([case])
        parsed = json.loads(result)
        assert parsed[0]["timestamp"] == "2025-06-15T14:30:00"

    def test_empty_task_intent(self):
        extractor = _build_extractor()
        case = _make_case_record(task_intent="")
        result = extractor._format_cases([case])
        parsed = json.loads(result)
        assert parsed[0]["task_intent"] == ""


# ===========================================================================
# _format_existing_skills edge cases
# ===========================================================================


class TestFormatExistingSkillsEdgeCases:
    """Additional edge case tests for _format_existing_skills."""

    def test_skill_fields_included(self):
        extractor = _build_extractor()
        skill = _make_skill_record(
            name="Deploy", description="Deploy services", content="## Steps\n1. Build",
            confidence=0.9,
        )
        result = extractor._format_existing_skills([skill])
        parsed = json.loads(result)
        assert parsed[0]["name"] == "Deploy"
        assert parsed[0]["description"] == "Deploy services"
        assert parsed[0]["content"] == "## Steps\n1. Build"
        assert parsed[0]["confidence"] == 0.9

    def test_multiple_skills_correct_indices(self):
        extractor = _build_extractor()
        skills = [
            _make_skill_record(name="A"),
            _make_skill_record(name="B"),
        ]
        result = extractor._format_existing_skills(skills)
        parsed = json.loads(result)
        assert parsed[0]["index"] == 0
        assert parsed[0]["name"] == "A"
        assert parsed[1]["index"] == 1
        assert parsed[1]["name"] == "B"


# ===========================================================================
# _call_llm edge cases
# ===========================================================================


class TestCallLLMEdgeCases:
    """Additional edge case tests for _call_llm."""

    @pytest.mark.asyncio
    async def test_operations_not_list_retries(self):
        extractor = _build_extractor()
        extractor.llm_provider.generate = AsyncMock(
            return_value=json.dumps({"operations": "not a list"})
        )
        result = await extractor._call_llm("c", "s", "{new_case_json}{existing_skills_json}")
        assert result is None
        assert extractor.llm_provider.generate.call_count == 3

    @pytest.mark.asyncio
    async def test_empty_operations_list_accepted(self):
        resp = json.dumps({"operations": []})
        extractor = _build_extractor(llm_response=resp)
        result = await extractor._call_llm("c", "s", "{new_case_json}{existing_skills_json}")
        assert result is not None
        assert result["operations"] == []

    @pytest.mark.asyncio
    async def test_extra_fields_preserved(self):
        resp = json.dumps({"operations": [{"action": "none"}], "update_note": "No changes needed"})
        extractor = _build_extractor(llm_response=resp)
        result = await extractor._call_llm("c", "s", "{new_case_json}{existing_skills_json}")
        assert result["update_note"] == "No changes needed"


# ===========================================================================
# _evaluate_maturity edge cases
# ===========================================================================


class TestEvaluateMaturityAdditional:
    """Additional tests for _evaluate_maturity."""

    @pytest.mark.asyncio
    async def test_score_clamped_to_1(self):
        resp = json.dumps({
            "completeness": 5, "executability": 5, "evidence": 5, "clarity": 5
        })
        extractor = _build_extractor(llm_response=resp)
        score = await extractor._evaluate_maturity("n", "d", "c", 0.8)
        assert score == 1.0

    @pytest.mark.asyncio
    async def test_score_clamped_to_0(self):
        resp = json.dumps({
            "completeness": 0, "executability": 0, "evidence": 0, "clarity": 0
        })
        extractor = _build_extractor(llm_response=resp)
        score = await extractor._evaluate_maturity("n", "d", "c", 0.8)
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_missing_dimension_returns_none(self):
        resp = json.dumps({
            "completeness": 4, "executability": 4, "evidence": 4
            # missing "clarity"
        })
        extractor = _build_extractor(llm_response=resp)
        score = await extractor._evaluate_maturity("n", "d", "c", 0.8)
        assert score is None


# ===========================================================================
# _apply_add edge cases
# ===========================================================================


class TestApplyAddEdgeCases:
    """Additional edge case tests for _apply_add."""

    @pytest.mark.asyncio
    async def test_missing_data_key(self):
        extractor = _build_extractor()
        repo = _mock_skill_repo()
        op = {"action": "add"}  # no "data" key
        result = await extractor._apply_add(op, "c", "g", "u", repo)
        assert result is None

    @pytest.mark.asyncio
    @patch("memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor._compute_embedding")
    @patch("memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor._evaluate_maturity")
    async def test_embedding_failure_still_saves(self, mock_maturity, mock_embed):
        mock_embed.return_value = None
        mock_maturity.return_value = 0.6
        repo = _mock_skill_repo()
        extractor = _build_extractor()
        op = {
            "action": "add",
            "data": {
                "name": "Skill",
                "description": "Desc",
                "content": "## Steps\n1. Analyze the requirements\n2. Build the implementation\n3. Check edge cases\n4. Deploy to staging\n5. Verify production",
                "confidence": 0.7,
            },
        }
        import infra_layer.adapters.out.persistence.document.memory.agent_skill as skill_mod
        with patch.object(skill_mod, "AgentSkillRecord", return_value=MagicMock(id="s1")):
            result = await extractor._apply_add(op, "c", "g", "u", repo)
        assert result is not None

    @pytest.mark.asyncio
    @patch("memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor._compute_embedding")
    @patch("memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor._evaluate_maturity")
    async def test_maturity_none_defaults_to_0_6(self, mock_maturity, mock_embed):
        mock_embed.return_value = None
        mock_maturity.return_value = None  # LLM failed
        repo = _mock_skill_repo()
        extractor = _build_extractor()
        op = {
            "action": "add",
            "data": {
                "name": "S", "description": "D",
                "content": "## Steps\n1. Analyze the requirements\n2. Build the implementation\n3. Check edge cases\n4. Deploy to staging\n5. Verify production",
            },
        }
        import infra_layer.adapters.out.persistence.document.memory.agent_skill as skill_mod
        with patch.object(
            skill_mod, "AgentSkillRecord", return_value=MagicMock(id="s2")
        ) as mock_cls:
            await extractor._apply_add(op, "c", "g", "u", repo)
        assert mock_cls.call_args.kwargs["maturity_score"] == 0.6

    @pytest.mark.asyncio
    @patch("memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor._compute_embedding")
    @patch("memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor._evaluate_maturity")
    async def test_confidence_clamped(self, mock_maturity, mock_embed):
        mock_embed.return_value = None
        mock_maturity.return_value = 0.7
        repo = _mock_skill_repo()
        extractor = _build_extractor()
        op = {
            "action": "add",
            "data": {
                "name": "S", "description": "D",
                "content": "## Steps\n1. Analyze the requirements\n2. Build the implementation\n3. Check edge cases\n4. Deploy to staging\n5. Verify production",
                "confidence": 2.5,  # out of range
            },
        }
        import infra_layer.adapters.out.persistence.document.memory.agent_skill as skill_mod
        with patch.object(
            skill_mod, "AgentSkillRecord", return_value=MagicMock(id="s3")
        ) as mock_cls:
            await extractor._apply_add(op, "c", "g", "u", repo)
        assert mock_cls.call_args.kwargs["confidence"] == 1.0

    @pytest.mark.asyncio
    @patch("memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor._compute_embedding")
    @patch("memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor._evaluate_maturity")
    async def test_source_case_ids_passed(self, mock_maturity, mock_embed):
        mock_embed.return_value = None
        mock_maturity.return_value = 0.7
        repo = _mock_skill_repo()
        extractor = _build_extractor()
        op = {
            "action": "add",
            "data": {
                "name": "S", "description": "D",
                "content": "## Steps\n1. Analyze the requirements\n2. Build the implementation\n3. Check edge cases\n4. Deploy to staging\n5. Verify production",
            },
        }
        import infra_layer.adapters.out.persistence.document.memory.agent_skill as skill_mod
        with patch.object(
            skill_mod, "AgentSkillRecord", return_value=MagicMock(id="s4")
        ) as mock_cls:
            await extractor._apply_add(
                op, "c", "g", "u", repo, source_case_ids=["c1", "c2"]
            )
        assert mock_cls.call_args.kwargs["source_case_ids"] == ["c1", "c2"]


# ===========================================================================
# _apply_update edge cases
# ===========================================================================


class TestApplyUpdateEdgeCases:
    """Additional edge case tests for _apply_update."""

    @pytest.mark.asyncio
    @patch("memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor._compute_embedding")
    @patch("memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor._evaluate_maturity")
    async def test_update_description_triggers_re_embed(self, mock_maturity, mock_embed):
        mock_embed.return_value = {"embedding": [0.7, 0.8], "vector_model": "m"}
        mock_maturity.return_value = 0.8
        repo = _mock_skill_repo()
        existing = [_make_skill_record(description="Old Desc")]
        result_obj = SkillExtractionResult()
        extractor = _build_extractor()
        op = {"action": "update", "index": 0, "data": {"description": "New Desc"}}
        await extractor._apply_update(op, existing, repo, result_obj)
        mock_embed.assert_called_once()
        updates = repo.update_skill_by_id.call_args[0][1]
        assert "vector" in updates

    @pytest.mark.asyncio
    @patch("memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor._compute_embedding")
    @patch("memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor._evaluate_maturity")
    async def test_same_name_no_re_embed(self, mock_maturity, mock_embed):
        """If name is same as existing, no re-embedding."""
        mock_embed.return_value = None
        mock_maturity.return_value = None
        repo = _mock_skill_repo()
        existing = [_make_skill_record(name="Same Name")]
        result_obj = SkillExtractionResult()
        extractor = _build_extractor()
        op = {"action": "update", "index": 0, "data": {"name": "Same Name", "confidence": 0.9}}
        await extractor._apply_update(op, existing, repo, result_obj)
        mock_embed.assert_not_called()

    @pytest.mark.asyncio
    @patch("memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor._compute_embedding")
    @patch("memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor._evaluate_maturity")
    async def test_confidence_only_update(self, mock_maturity, mock_embed):
        mock_embed.return_value = None
        mock_maturity.return_value = None
        repo = _mock_skill_repo()
        existing = [_make_skill_record()]
        result_obj = SkillExtractionResult()
        extractor = _build_extractor()
        op = {"action": "update", "index": 0, "data": {"confidence": 0.95}}
        success = await extractor._apply_update(op, existing, repo, result_obj)
        assert success is True
        updates = repo.update_skill_by_id.call_args[0][1]
        assert updates["confidence"] == 0.95
        # No content change => no maturity re-eval
        mock_maturity.assert_not_called()

    @pytest.mark.asyncio
    @patch("memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor._compute_embedding")
    @patch("memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor._evaluate_maturity")
    async def test_invalid_confidence_ignored(self, mock_maturity, mock_embed):
        mock_embed.return_value = None
        mock_maturity.return_value = None
        repo = _mock_skill_repo()
        existing = [_make_skill_record()]
        result_obj = SkillExtractionResult()
        extractor = _build_extractor()
        op = {"action": "update", "index": 0, "data": {"confidence": "bad", "name": "New"}}
        success = await extractor._apply_update(op, existing, repo, result_obj)
        assert success is True
        updates = repo.update_skill_by_id.call_args[0][1]
        assert "confidence" not in updates
        assert updates["name"] == "New"

    @pytest.mark.asyncio
    @patch("memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor._compute_embedding")
    @patch("memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor._evaluate_maturity")
    async def test_multiple_source_case_ids_appended(self, mock_maturity, mock_embed):
        mock_embed.return_value = None
        mock_maturity.return_value = None
        repo = _mock_skill_repo()
        existing = [_make_skill_record(source_case_ids=["c1"])]
        result_obj = SkillExtractionResult()
        extractor = _build_extractor()
        op = {"action": "update", "index": 0, "data": {"confidence": 0.9}}
        await extractor._apply_update(
            op, existing, repo, result_obj, source_case_ids=["c2", "c3"]
        )
        updates = repo.update_skill_by_id.call_args[0][1]
        assert set(updates["source_case_ids"]) == {"c1", "c2", "c3"}

    @pytest.mark.asyncio
    async def test_retire_preserves_source_case_ids(self):
        repo = _mock_skill_repo()
        existing = [_make_skill_record(source_case_ids=["c1"])]
        result_obj = SkillExtractionResult()
        extractor = _build_extractor()
        op = {"action": "update", "index": 0, "data": {"confidence": 0.05}}
        await extractor._apply_update(
            op, existing, repo, result_obj, source_case_ids=["c2"]
        )
        updates = repo.update_skill_by_id.call_args[0][1]
        assert "source_case_ids" in updates
        assert set(updates["source_case_ids"]) == {"c1", "c2"}
        assert str(existing[0].id) in result_obj.deleted_ids

    @pytest.mark.asyncio
    @patch("memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor._compute_embedding")
    @patch("memory_layer.memory_extractor.agent_skill_extractor.AgentSkillExtractor._evaluate_maturity")
    async def test_update_sets_attributes_on_record(self, mock_maturity, mock_embed):
        """After successful update, record attributes are patched in-place."""
        mock_embed.return_value = None
        mock_maturity.return_value = None
        repo = _mock_skill_repo()
        existing = [_make_skill_record(name="Old")]
        result_obj = SkillExtractionResult()
        extractor = _build_extractor()
        op = {"action": "update", "index": 0, "data": {"name": "New"}}
        await extractor._apply_update(op, existing, repo, result_obj)
        # Record should be mutated in place
        assert existing[0].name == "New"
        assert existing[0] in result_obj.updated_records


# ===========================================================================
# extract_and_save edge cases
# ===========================================================================


class TestExtractAndSaveEdgeCases:
    """Additional edge case tests for extract_and_save."""

    @pytest.mark.asyncio
    async def test_source_case_ids_collected_from_records(self):
        """extract_and_save collects IDs from new_case_records for traceability."""
        response = json.dumps({
            "operations": [{"action": "add", "data": {"name": "S", "description": "D", "content": "C"}}]
        })
        extractor = _build_extractor(llm_response=response)
        repo = _mock_skill_repo()

        cases = [
            _make_case_record(record_id="case_A"),
            _make_case_record(record_id="case_B"),
        ]

        with patch.object(extractor, "_apply_add", new_callable=AsyncMock) as mock_add:
            mock_add.return_value = MagicMock(id="new_skill")
            await extractor.extract_and_save(
                cluster_id="c1", group_id="g1",
                new_case_records=cases,
                existing_skill_records=[],
                skill_repo=repo,
            )
            call_kwargs = mock_add.call_args.kwargs
            assert set(call_kwargs["source_case_ids"]) == {"case_A", "case_B"}

    @pytest.mark.asyncio
    async def test_none_record_ids_filtered(self):
        """Records with None id should not produce source_case_ids entries."""
        response = json.dumps({
            "operations": [{"action": "add", "data": {"name": "S", "description": "D", "content": "C"}}]
        })
        extractor = _build_extractor(llm_response=response)
        repo = _mock_skill_repo()

        cases = [
            _make_case_record(record_id=None),
            _make_case_record(record_id="case_X"),
        ]

        with patch.object(extractor, "_apply_add", new_callable=AsyncMock) as mock_add:
            mock_add.return_value = MagicMock(id="new_skill")
            await extractor.extract_and_save(
                cluster_id="c1", group_id="g1",
                new_case_records=cases,
                existing_skill_records=[],
                skill_repo=repo,
            )
            call_kwargs = mock_add.call_args.kwargs
            assert call_kwargs["source_case_ids"] == ["case_X"]

    @pytest.mark.asyncio
    async def test_mixed_operations(self):
        """Test add + update + none in a single response."""
        response = json.dumps({
            "operations": [
                {"action": "add", "data": {"name": "New", "description": "D", "content": "C"}},
                {"action": "update", "index": 0, "data": {"confidence": 0.9}},
                {"action": "none"},
            ]
        })
        extractor = _build_extractor(llm_response=response)
        repo = _mock_skill_repo()

        with patch.object(extractor, "_apply_add", new_callable=AsyncMock) as mock_add, \
             patch.object(extractor, "_apply_update", new_callable=AsyncMock) as mock_update:
            mock_add.return_value = MagicMock(id="new_id")
            mock_update.return_value = True
            result = await extractor.extract_and_save(
                cluster_id="c1", group_id="g1",
                new_case_records=[_make_case_record()],
                existing_skill_records=[_make_skill_record()],
                skill_repo=repo,
            )
            mock_add.assert_called_once()
            mock_update.assert_called_once()
            assert len(result.added_records) == 1

    @pytest.mark.asyncio
    async def test_case_history_loaded_after_top_k(self):
        """Case history should be loaded AFTER top-k selection."""
        response = json.dumps({"operations": [{"action": "none"}]})
        extractor = _build_extractor(llm_response=response)
        repo = _mock_skill_repo()
        many_skills = [_make_skill_record(name=f"S{i}", record_id=f"s{i}") for i in range(15)]
        selected = many_skills[:5]

        call_order = []

        async def mock_top_k(*args, **kwargs):
            call_order.append("top_k")
            return selected

        async def mock_load_history(skill_records, *args, **kwargs):
            call_order.append(("load_history", len(skill_records)))
            return []

        with patch.object(extractor, "_select_top_k_skills", side_effect=mock_top_k), \
             patch.object(extractor, "_load_case_history", side_effect=mock_load_history):
            await extractor.extract_and_save(
                cluster_id="c1", group_id="g1",
                new_case_records=[_make_case_record()],
                existing_skill_records=many_skills,
                skill_repo=repo,
                max_skills_in_prompt=10,
            )
        assert call_order[0] == "top_k"
        # load_history receives the filtered 5, not the original 15
        assert call_order[1] == ("load_history", 5)


# ===========================================================================
# _select_top_k_skills edge cases
# ===========================================================================


class TestSelectTopKSkillsEdgeCases:
    """Additional edge case tests for _select_top_k_skills."""

    @pytest.mark.asyncio
    async def test_fewer_skills_than_k(self):
        extractor = _build_extractor()
        skills = [_make_skill_record(name=f"S{i}") for i in range(3)]
        cases = [_make_case_record()]
        with patch.object(extractor, "_compute_embedding", new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = {"embedding": [0.1, 0.2, 0.3]}
            result = await extractor._select_top_k_skills(skills, cases, top_k=10)
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_empty_skills(self):
        extractor = _build_extractor()
        result = await extractor._select_top_k_skills([], [_make_case_record()], top_k=5)
        assert result == []

    @pytest.mark.asyncio
    async def test_no_vectors_returns_original_order(self):
        extractor = _build_extractor()
        skills = [_make_skill_record(name=f"S{i}", vector=None) for i in range(5)]
        cases = [_make_case_record()]
        with patch.object(extractor, "_compute_embedding", new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = {"embedding": [0.1, 0.2, 0.3]}
            result = await extractor._select_top_k_skills(skills, cases, top_k=3)
        assert len(result) == 3


# ===========================================================================
# _select_prompt edge cases
# ===========================================================================


class TestSelectPromptEdgeCases:
    """Additional tests for _select_prompt."""

    def test_empty_cases_uses_success(self):
        extractor = _build_extractor()
        result = extractor._select_prompt([])
        assert result == extractor.success_extract_prompt

    def test_all_none_quality_uses_success(self):
        extractor = _build_extractor()
        cases = [_make_case_record(quality_score=None)]
        result = extractor._select_prompt(cases)
        # None quality defaults to 0.5 which is > FAILURE_QUALITY_THRESHOLD (0.3)
        assert result == extractor.success_extract_prompt


# ===========================================================================
# _is_skill_content_sufficient edge cases
# ===========================================================================


class TestIsSkillContentSufficientEdgeCases:
    """Additional tests for _is_skill_content_sufficient."""

    def test_none_content(self):
        assert AgentSkillExtractor._is_skill_content_sufficient(None) is False

    def test_whitespace_only(self):
        assert AgentSkillExtractor._is_skill_content_sufficient("   \n\n  ") is False

    def test_numbered_steps_sufficient(self):
        content = "## Steps\n1. First do this\n2. Then do that\n3. Check results\n4. Validate output\n5. Deploy"
        assert AgentSkillExtractor._is_skill_content_sufficient(content) is True

    def test_bulleted_steps_sufficient(self):
        content = "## Process\n- Analyze requirements\n- Design architecture\n- Implement solution\n- Run tests\n- Deploy to production"
        assert AgentSkillExtractor._is_skill_content_sufficient(content) is True

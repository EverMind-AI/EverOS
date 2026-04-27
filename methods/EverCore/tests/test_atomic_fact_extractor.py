"""Unit tests for AtomicFactExtractor.

Tests cover:
- _parse_llm_response: new key, code blocks, invalid JSON, old key rejection
- _extract_atomic_fact: empty list acceptance, missing outer key rejection
- extract_atomic_fact: retry exhaustion returns None, successful first extraction
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memory_layer.memory_extractor.atomic_fact_extractor import AtomicFactExtractor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def extractor():
    """Create an AtomicFactExtractor with a mocked LLM provider."""
    mock_provider = MagicMock()
    mock_provider.generate = AsyncMock()
    return AtomicFactExtractor(
        llm_provider=mock_provider, atomic_fact_prompt="{{INPUT_TEXT}} {{TIME}}"
    )


VALID_RESPONSE = {
    "atomic_facts": {
        "time": "March 10, 2024(Sunday) at 12:00 AM",
        "atomic_fact": ["Alice likes hiking.", "Bob prefers reading."],
    }
}


# ---------------------------------------------------------------------------
# TestParseLlmResponse
# ---------------------------------------------------------------------------


class TestParseLlmResponse:
    """Tests for AtomicFactExtractor._parse_llm_response."""

    def test_parses_new_atomic_facts_key(self, extractor):
        """Direct JSON with 'atomic_facts' outer key parses correctly."""
        raw = json.dumps(VALID_RESPONSE)
        result = extractor._parse_llm_response(raw)
        assert "atomic_facts" in result
        assert result["atomic_facts"]["time"] == "March 10, 2024(Sunday) at 12:00 AM"
        assert result["atomic_facts"]["atomic_fact"] == [
            "Alice likes hiking.",
            "Bob prefers reading.",
        ]

    def test_parses_json_in_code_block(self, extractor):
        """JSON wrapped in ```json ... ``` code block parses correctly."""
        raw = "Here is the result:\n```json\n" + json.dumps(VALID_RESPONSE) + "\n```"
        result = extractor._parse_llm_response(raw)
        assert "atomic_facts" in result
        assert len(result["atomic_facts"]["atomic_fact"]) == 2

    def test_rejects_invalid_json(self, extractor):
        """Completely invalid text raises ValueError."""
        with pytest.raises(ValueError, match="Unable to parse LLM response"):
            extractor._parse_llm_response("this is not json at all!!!")

    def test_rejects_old_atomic_fact_key(self, extractor):
        """Old singular 'atomic_fact' outer key is not recognized as 'atomic_facts'."""
        old_format = {
            "atomic_fact": {
                "time": "March 10, 2024(Sunday) at 12:00 AM",
                "atomic_fact": ["Alice likes hiking."],
            }
        }
        # The parser will successfully parse the JSON (step 4: direct parse),
        # but the result will have "atomic_fact" not "atomic_facts".
        result = extractor._parse_llm_response(json.dumps(old_format))
        assert "atomic_facts" not in result
        assert "atomic_fact" in result


# ---------------------------------------------------------------------------
# TestExtractAtomicFactValidation
# ---------------------------------------------------------------------------


class TestExtractAtomicFactValidation:
    """Tests for _extract_atomic_fact validation logic."""

    @pytest.mark.asyncio
    async def test_empty_list_returns_valid_object(self, extractor):
        """LLM returns atomic_fact: [] -> returns AtomicFact with empty list, no exception."""
        llm_response = json.dumps(
            {
                "atomic_facts": {
                    "time": "March 10, 2024(Sunday) at 12:00 AM",
                    "atomic_fact": [],
                }
            }
        )
        extractor.llm_provider.generate.return_value = llm_response

        with patch("agentic_layer.vectorize_service.get_vectorize_service") as mock_vs:
            mock_service = MagicMock()
            mock_service.get_embeddings = AsyncMock(return_value=[])
            mock_service.get_model_name = MagicMock(return_value="test-model")
            mock_vs.return_value = mock_service

            result = await extractor._extract_atomic_fact(
                input_text="hello", timestamp="2024-03-10T00:00:00Z", user_id="user1"
            )

        assert result is not None
        assert result.atomic_fact == []
        assert result.time == "March 10, 2024(Sunday) at 12:00 AM"

    @pytest.mark.asyncio
    async def test_missing_outer_key_raises(self, extractor):
        """LLM returns wrong outer key -> raises ValueError mentioning 'atomic_facts'."""
        llm_response = json.dumps(
            {
                "wrong_key": {
                    "time": "March 10, 2024(Sunday) at 12:00 AM",
                    "atomic_fact": ["something"],
                }
            }
        )
        extractor.llm_provider.generate.return_value = llm_response

        with pytest.raises(ValueError, match="atomic_facts"):
            await extractor._extract_atomic_fact(
                input_text="hello", timestamp="2024-03-10T00:00:00Z", user_id="user1"
            )


# ---------------------------------------------------------------------------
# TestExtractAtomicFactErrorHandling
# ---------------------------------------------------------------------------


class TestExtractAtomicFactErrorHandling:
    """Tests for extract_atomic_fact (public method) retry and error handling."""

    @staticmethod
    def _make_memcell():
        return MagicMock(
            original_data=[
                {
                    "message": {
                        "sender_name": "Alice",
                        "content": [{"type": "text", "text": "hello"}],
                        "timestamp": "2024-03-10T00:00:00Z",
                    }
                }
            ],
            sender_ids=["alice"],
        )

    @pytest.mark.asyncio
    async def test_returns_none_after_retries_exhausted(self, extractor):
        """LLM provider raises Exception every time -> after 5 calls, returns None."""
        extractor.llm_provider.generate = AsyncMock(
            side_effect=Exception("LLM unavailable")
        )
        memcell = self._make_memcell()

        result = await extractor.extract_atomic_fact(
            memcell=memcell, timestamp="2024-03-10T00:00:00Z", user_id="user1"
        )

        assert result is None
        assert extractor.llm_provider.generate.call_count == 5

    @pytest.mark.asyncio
    async def test_returns_result_on_first_success(self, extractor):
        """Normal successful extraction returns AtomicFact object."""
        llm_response = json.dumps(VALID_RESPONSE)
        extractor.llm_provider.generate = AsyncMock(return_value=llm_response)
        memcell = self._make_memcell()

        with patch("agentic_layer.vectorize_service.get_vectorize_service") as mock_vs:
            mock_service = MagicMock()
            mock_service.get_embeddings = AsyncMock(
                return_value=[[0.1] * 1024, [0.2] * 1024]
            )
            mock_service.get_model_name = MagicMock(return_value="test-model")
            mock_vs.return_value = mock_service

            result = await extractor.extract_atomic_fact(
                memcell=memcell, timestamp="2024-03-10T00:00:00Z", user_id="user1"
            )

        assert result is not None
        assert result.atomic_fact == ["Alice likes hiking.", "Bob prefers reading."]
        assert result.time == "March 10, 2024(Sunday) at 12:00 AM"
        assert result.fact_embeddings is not None
        assert len(result.fact_embeddings) == 2
        assert extractor.llm_provider.generate.call_count == 1

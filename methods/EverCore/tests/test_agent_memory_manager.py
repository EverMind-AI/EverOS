"""
MemoryManager agent case dispatch tests.

Tests for:
- MemoryManager._extract_agent_case: delegation, non-agent memcell, None memcell
- MemoryManager.extract_memory: AGENT_CASE dispatch path

Usage:
    PYTHONPATH=src pytest tests/test_memory_manager_agent.py -v
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from api_specs.memory_types import MemCell, RawDataType, AgentCase
from api_specs.memory_models import MemoryType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent_memcell(**overrides):
    defaults = dict(
        user_id_list=["u1"],
        original_data=[
            {"message": {"role": "user", "content": "Deploy the app", "sender_id": "u1"}},
            {"message": {"role": "assistant", "content": "Done deploying."}},
        ],
        timestamp=datetime(2025, 6, 1, 10, 0, 0),
        event_id="evt_100",
        group_id="g1",
        type=RawDataType.AGENTCONVERSATION,
    )
    defaults.update(overrides)
    return MemCell(**defaults)


def _make_conv_memcell(**overrides):
    defaults = dict(
        user_id_list=["u1"],
        original_data=[
            {"message": {"role": "user", "content": "Hello", "sender_id": "u1"}},
        ],
        timestamp=datetime(2025, 6, 1, 10, 0, 0),
        event_id="evt_200",
        group_id="g1",
        type=RawDataType.CONVERSATION,
    )
    defaults.update(overrides)
    return MemCell(**defaults)


def _make_agent_case(**overrides):
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


def _make_memory_manager():
    """Create a MemoryManager with mocked LLM provider."""
    from memory_layer.memory_manager import MemoryManager
    with patch("memory_layer.memory_manager.build_default_provider", return_value=MagicMock()):
        return MemoryManager()


# ===========================================================================
# MemoryManager._extract_agent_case
# ===========================================================================


class TestMemoryManagerExtractAgentCase:
    """Tests for MemoryManager._extract_agent_case method."""

    @pytest.mark.asyncio
    async def test_agent_memcell_delegates_to_extractor(self):
        """Agent conversation memcell should delegate to AgentCaseExtractor."""
        memcell = _make_agent_memcell()
        expected_case = _make_agent_case()

        with patch("memory_layer.memory_extractor.agent_case_extractor.AgentCaseExtractor") as mock_cls:
            mock_extractor = AsyncMock()
            mock_extractor.extract_memory = AsyncMock(return_value=expected_case)
            mock_cls.return_value = mock_extractor

            mgr = _make_memory_manager()
            result = await mgr._extract_agent_case(memcell, user_id="u1", group_id="g1")

            assert result is expected_case
            mock_extractor.extract_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_non_agent_memcell_returns_none(self):
        """Regular conversation memcell should return None."""
        memcell = _make_conv_memcell()
        mgr = _make_memory_manager()
        result = await mgr._extract_agent_case(memcell, user_id="u1", group_id="g1")

        assert result is None

    @pytest.mark.asyncio
    async def test_none_memcell_returns_none(self):
        """None memcell should return None."""
        mgr = _make_memory_manager()
        result = await mgr._extract_agent_case(None, user_id="u1", group_id="g1")

        assert result is None


# ===========================================================================
# MemoryManager.extract_memory AGENT_CASE dispatch
# ===========================================================================


class TestMemoryManagerExtractMemoryDispatch:
    """Tests for MemoryManager.extract_memory dispatching to AGENT_CASE."""

    @pytest.mark.asyncio
    async def test_dispatch_agent_case(self):
        """extract_memory with AGENT_CASE type should call _extract_agent_case."""
        memcell = _make_agent_memcell()
        expected_case = _make_agent_case()

        mgr = _make_memory_manager()

        with patch.object(mgr, "_extract_agent_case", new_callable=AsyncMock) as mock_method:
            mock_method.return_value = expected_case

            with patch("memory_layer.memory_manager.record_extract_memory_call"), \
                 patch("memory_layer.memory_manager.get_space_id_for_metrics", return_value="test"):
                result = await mgr.extract_memory(
                    memcell=memcell,
                    memory_type=MemoryType.AGENT_CASE,
                    user_id="u1",
                    group_id="g1",
                )

            assert result is expected_case
            mock_method.assert_called_once_with(memcell, user_id="u1", group_id="g1")

    @pytest.mark.asyncio
    async def test_dispatch_agent_case_none_result(self):
        """extract_memory with AGENT_CASE returning None should still return None."""
        memcell = _make_agent_memcell()
        mgr = _make_memory_manager()

        with patch.object(mgr, "_extract_agent_case", new_callable=AsyncMock) as mock_method:
            mock_method.return_value = None

            with patch("memory_layer.memory_manager.record_extract_memory_call"), \
                 patch("memory_layer.memory_manager.get_space_id_for_metrics", return_value="test"):
                result = await mgr.extract_memory(
                    memcell=memcell,
                    memory_type=MemoryType.AGENT_CASE,
                )

            assert result is None

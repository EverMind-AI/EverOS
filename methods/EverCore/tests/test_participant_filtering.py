"""
tests/test_participant_filtering.py

Unit tests for participant extraction and downstream usage after
the role-based filtering change:
- _extract_participant_ids only extracts role='user' sender_ids
- Downstream functions trust participants without keyword filtering

Usage:
    PYTHONPATH=src pytest tests/test_participant_filtering.py -v
"""

from dataclasses import replace
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock

import pytest

from memory_layer.memcell_extractor.conv_memcell_extractor import ConvMemCellExtractor
from memory_layer.llm.llm_provider import LLMProvider
from api_specs.memory_types import EpisodeMemory
from biz_layer.mem_memorize import ExtractionState, _clone_episodes_for_users


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_extractor() -> ConvMemCellExtractor:
    provider = MagicMock(spec=LLMProvider)
    provider.generate = AsyncMock(return_value='{"boundaries": [], "should_wait": false}')
    return ConvMemCellExtractor(provider)


def _msg(role: str, sender_id: str) -> dict:
    return {"role": role, "sender_id": sender_id, "content": [{"type": "text", "text": "hi"}]}


# ---------------------------------------------------------------------------
# Tests for _extract_participant_ids
# ---------------------------------------------------------------------------

class TestExtractParticipantIds:

    def test_only_user_role_extracted(self):
        ext = _mock_extractor()
        messages = [
            _msg("user", "alice"),
            _msg("assistant", "robot_bot"),
            _msg("user", "bob"),
        ]
        result = ext._extract_participant_ids(messages)
        assert sorted(result) == ["alice", "bob"]

    def test_assistant_excluded(self):
        ext = _mock_extractor()
        messages = [
            _msg("assistant", "assistant_001"),
            _msg("assistant", "gpt_helper"),
        ]
        result = ext._extract_participant_ids(messages)
        assert result == []

    def test_system_role_excluded(self):
        ext = _mock_extractor()
        messages = [
            _msg("system", "sys_001"),
            _msg("user", "alice"),
        ]
        result = ext._extract_participant_ids(messages)
        assert result == ["alice"]

    def test_deduplication(self):
        ext = _mock_extractor()
        messages = [
            _msg("user", "alice"),
            _msg("user", "alice"),
            _msg("user", "bob"),
        ]
        result = ext._extract_participant_ids(messages)
        assert sorted(result) == ["alice", "bob"]

    def test_empty_sender_id_skipped(self):
        ext = _mock_extractor()
        messages = [
            {"role": "user", "sender_id": "", "content": []},
            {"role": "user", "sender_id": None, "content": []},
            {"role": "user", "content": []},
            _msg("user", "alice"),
        ]
        result = ext._extract_participant_ids(messages)
        assert result == ["alice"]

    def test_empty_messages(self):
        ext = _mock_extractor()
        assert ext._extract_participant_ids([]) == []

    def test_robot_keyword_in_user_role_kept(self):
        """User with 'robot' in sender_id but role='user' should be kept."""
        ext = _mock_extractor()
        messages = [_msg("user", "robot_tester")]
        result = ext._extract_participant_ids(messages)
        assert result == ["robot_tester"]


# ---------------------------------------------------------------------------
# Tests for _clone_episodes_for_users
# ---------------------------------------------------------------------------

class TestCloneEpisodesForUsers:

    def _make_state(self, participants: list) -> ExtractionState:
        ep = EpisodeMemory(
            memory_type="episodic_memory",
            user_id="group",
            user_name="group",
            timestamp=datetime(2026, 1, 1),
            episode="test episode",
        )
        state = MagicMock(spec=ExtractionState)
        state.participants = participants
        state.group_episode_memories = [ep]
        return state

    def test_clones_for_all_participants(self):
        state = self._make_state(["alice", "bob", "charlie"])
        result = _clone_episodes_for_users(state)
        assert len(result) == 3
        user_ids = [ep.user_id for ep in result]
        assert sorted(user_ids) == ["alice", "bob", "charlie"]

    def test_no_keyword_filtering(self):
        """Participants are trusted as-is, no keyword filtering applied."""
        state = self._make_state(["alice", "robot_tester"])
        result = _clone_episodes_for_users(state)
        assert len(result) == 2

    def test_empty_participants(self):
        state = self._make_state([])
        result = _clone_episodes_for_users(state)
        assert result == []

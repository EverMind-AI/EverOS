"""
Profile Memory & Extractor Unit Tests

Tests for:
- ProfileMemory: to_dict, from_dict, total_items, get_all_source_ids, readable output
- ProfileExtractRequest: construction and defaults
- ProfileExtractor: id mapping, LLM response parsing, operations application
- Unified extraction: no SOLO/TEAM branching

Usage:
    PYTHONPATH=src pytest tests/test_profile_memory.py -v
"""

import json
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from api_specs.memory_types import ProfileMemory, MemoryType, BaseMemory
from memory_layer.memory_extractor.profile_extractor import (
    ProfileExtractor,
    ProfileExtractRequest,
    ProfileAction,
    ProfileItemType,
    _create_id_mapping,
    _replace_sources,
    _get_short_id,
)
from memory_layer.memory_extractor.base_memory_extractor import MemoryExtractRequest


# ============================================================================
# ProfileMemory data model
# ============================================================================


class TestProfileMemory:
    """ProfileMemory dataclass: construction, serialization, utility methods."""

    def _make_profile(self, **kwargs) -> ProfileMemory:
        defaults = dict(
            memory_type=MemoryType.PROFILE,
            user_id="user_1",
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            explicit_info=[
                {"category": "Skill", "description": "Python", "evidence": "said so", "sources": ["ep1"]},
                {"category": "Interest", "description": "hiking", "evidence": "", "sources": []},
            ],
            implicit_traits=[
                {"trait": "Curious", "description": "asks many questions", "basis": "behavior", "evidence": "", "sources": ["ep1", "ep2"]},
            ],
        )
        defaults.update(kwargs)
        return ProfileMemory(**defaults)

    def test_inherits_base_memory(self):
        assert issubclass(ProfileMemory, BaseMemory)

    def test_total_items(self):
        p = self._make_profile()
        assert p.total_items() == 3

    def test_total_items_empty(self):
        p = self._make_profile(explicit_info=[], implicit_traits=[])
        assert p.total_items() == 0

    def test_to_dict_structure(self):
        p = self._make_profile()
        d = p.to_dict()
        assert d["user_id"] == "user_1"
        assert d["memory_type"] == "profile"
        assert len(d["explicit_info"]) == 2
        assert len(d["implicit_traits"]) == 1
        assert "processed_episode_ids" in d

    def test_to_dict_returns_plain_dicts(self):
        """explicit_info and implicit_traits should be plain dicts, not objects."""
        p = self._make_profile()
        d = p.to_dict()
        for info in d["explicit_info"]:
            assert isinstance(info, dict)
        for trait in d["implicit_traits"]:
            assert isinstance(trait, dict)

    def test_from_dict_roundtrip(self):
        p = self._make_profile()
        d = p.to_dict()
        p2 = ProfileMemory.from_dict(d)
        assert p2.user_id == "user_1"
        assert p2.total_items() == 3
        assert p2.explicit_info[0]["category"] == "Skill"
        assert p2.implicit_traits[0]["trait"] == "Curious"

    def test_from_dict_with_user_id_override(self):
        d = {"explicit_info": [{"category": "X", "description": "Y"}], "implicit_traits": []}
        p = ProfileMemory.from_dict(d, user_id="override_user", group_id="override_group")
        assert p.user_id == "override_user"
        assert p.group_id == "override_group"

    def test_from_dict_empty(self):
        p = ProfileMemory.from_dict({})
        assert p.total_items() == 0
        assert p.explicit_info == []
        assert p.implicit_traits == []

    def test_get_all_source_ids(self):
        p = self._make_profile()
        p.explicit_info[0]["sources"] = ["2024-01-01|conv_abc"]
        p.implicit_traits[0]["sources"] = ["conv_def", "2024-02-01|conv_ghi"]
        ids = p.get_all_source_ids()
        assert "conv_abc" in ids
        assert "conv_def" in ids
        assert "conv_ghi" in ids

    def test_get_all_source_ids_empty(self):
        p = self._make_profile(explicit_info=[], implicit_traits=[])
        assert p.get_all_source_ids() == set()

    def test_to_readable_document(self):
        p = self._make_profile()
        doc = p.to_readable_document()
        assert "Python" in doc
        assert "Skill" in doc
        assert "Curious" in doc
        assert "Total 3 items" in doc

    def test_to_readable_profile(self):
        p = self._make_profile()
        text = p.to_readable_profile()
        assert "Python" in text
        assert "Curious" in text

    def test_to_readable_profile_empty(self):
        p = self._make_profile(explicit_info=[], implicit_traits=[])
        assert p.to_readable_profile() == "No profile data yet."

    def test_post_init_sets_memory_type(self):
        p = self._make_profile()
        assert p.memory_type == MemoryType.PROFILE

    def test_post_init_sets_last_updated(self):
        p = self._make_profile()
        assert p.last_updated is not None

    def test_processed_episode_ids_default(self):
        p = self._make_profile()
        assert p.processed_episode_ids == []


# ============================================================================
# ID Mapper
# ============================================================================


class TestIdMapper:
    """ID mapping functions for token reduction."""

    def test_create_id_mapping(self):
        m = _create_id_mapping(["abc123", "def456", "ghi789"])
        assert m == {"abc123": "ep1", "def456": "ep2", "ghi789": "ep3"}

    def test_create_id_mapping_skips_empty(self):
        m = _create_id_mapping(["abc", "", None, "def"])
        assert "" not in m
        assert None not in m
        assert len(m) == 2

    def test_get_short_id(self):
        m = {"abc": "ep1"}
        assert _get_short_id("abc", m) == "ep1"
        assert _get_short_id("unknown", m) == "unknown"

    def test_replace_sources_forward(self):
        profile = {
            "explicit_info": [{"sources": ["abc", "def"]}],
            "implicit_traits": [{"sources": ["abc"]}],
        }
        m = {"abc": "ep1", "def": "ep2"}
        result = _replace_sources(profile, m)
        assert result["explicit_info"][0]["sources"] == ["ep1", "ep2"]
        assert result["implicit_traits"][0]["sources"] == ["ep1"]

    def test_replace_sources_reverse(self):
        profile = {
            "explicit_info": [{"sources": ["ep1"]}],
            "implicit_traits": [],
        }
        m = {"abc": "ep1"}
        result = _replace_sources(profile, m, reverse=True)
        assert result["explicit_info"][0]["sources"] == ["abc"]

    def test_replace_sources_with_timestamp_prefix(self):
        profile = {
            "explicit_info": [{"sources": ["2024-01-01|abc"]}],
            "implicit_traits": [],
        }
        m = {"abc": "ep1"}
        result = _replace_sources(profile, m)
        assert result["explicit_info"][0]["sources"] == ["2024-01-01|ep1"]

    def test_replace_sources_does_not_mutate_original(self):
        profile = {"explicit_info": [{"sources": ["abc"]}], "implicit_traits": []}
        m = {"abc": "ep1"}
        _replace_sources(profile, m)
        assert profile["explicit_info"][0]["sources"] == ["abc"]


# ============================================================================
# ProfileExtractRequest
# ============================================================================


class TestProfileExtractRequest:
    """ProfileExtractRequest construction and defaults."""

    def test_inherits_memory_extract_request(self):
        assert issubclass(ProfileExtractRequest, MemoryExtractRequest)

    def test_defaults(self):
        req = ProfileExtractRequest(user_id="u1", group_id="g1")
        assert req.new_episode is None
        assert req.cluster_episodes == []
        assert req.old_profile is None
        assert req.max_items == 25
        assert req.memcell_list == []
        assert req.episode_list == []

    def test_with_old_profile(self):
        p = ProfileMemory(
            memory_type=MemoryType.PROFILE,
            user_id="u1",
            timestamp=datetime.now(timezone.utc),
        )
        req = ProfileExtractRequest(user_id="u1", group_id="g1", old_profile=p)
        assert req.old_profile is p


# ============================================================================
# ProfileExtractor
# ============================================================================


class TestProfileExtractor:
    """ProfileExtractor: LLM response parsing, operations, dedup."""

    def _make_extractor(self):
        llm = MagicMock()
        return ProfileExtractor(llm_provider=llm)

    def test_parse_profile_response_json(self):
        ext = self._make_extractor()
        resp = '```json\n{"operations": [{"action": "none"}]}\n```'
        result = ext._parse_profile_response(resp)
        assert result["operations"] == [{"action": "none"}]

    def test_parse_profile_response_bare_json(self):
        ext = self._make_extractor()
        resp = '{"operations": []}'
        result = ext._parse_profile_response(resp)
        assert result["operations"] == []

    def test_parse_profile_response_invalid(self):
        ext = self._make_extractor()
        assert ext._parse_profile_response("not json at all") is None
        assert ext._parse_profile_response("") is None

    def test_format_profile_with_index_empty(self):
        ext = self._make_extractor()
        assert ext._format_profile_with_index({"explicit_info": [], "implicit_traits": []}) == ""

    def test_format_profile_with_index(self):
        ext = self._make_extractor()
        d = {
            ProfileItemType.EXPLICIT_INFO: [
                {"category": "Skill", "description": "Python", "evidence": "said so"},
            ],
            ProfileItemType.IMPLICIT_TRAITS: [
                {"trait": "Curious", "description": "asks why", "evidence": ""},
            ],
        }
        text = ext._format_profile_with_index(d)
        assert "[0] [Skill] Python" in text
        assert "[0] Curious: asks why" in text

    def test_attach_ts_already_has_timestamp(self):
        ext = self._make_extractor()
        assert ext._attach_ts("2024-01-01|ep1", {}) == "2024-01-01|ep1"

    def test_attach_ts_adds_timestamp(self):
        ext = self._make_extractor()
        result = ext._attach_ts("ep1", {"ep1": "2024-01-01"})
        assert result == "2024-01-01|ep1"

    def test_attach_ts_no_mapping(self):
        ext = self._make_extractor()
        assert ext._attach_ts("ep1", {}) == "ep1"

    @pytest.mark.asyncio
    async def test_extract_memory_skips_processed_episode(self):
        ext = self._make_extractor()
        old_profile = ProfileMemory(
            memory_type=MemoryType.PROFILE,
            user_id="u1",
            timestamp=datetime.now(timezone.utc),
            processed_episode_ids=["ep_already"],
        )
        req = ProfileExtractRequest(
            user_id="u1",
            group_id="g1",
            new_episode={"id": "ep_already", "original_data": []},
            old_profile=old_profile,
        )
        result = await ext.extract_memory(req)
        assert result is old_profile  # returned as-is, no LLM call

    @pytest.mark.asyncio
    async def test_extract_memory_creates_new_profile_when_no_old(self):
        ext = self._make_extractor()
        ext.llm_provider.generate = AsyncMock(return_value='{"operations": []}')
        req = ProfileExtractRequest(
            user_id="u1",
            group_id="g1",
            new_episode={"id": "ep1", "created_at": "2024-01-01", "original_data": []},
        )
        result = await ext.extract_memory(req)
        assert result is not None
        assert result.user_id == "u1"
        assert "ep1" in result.processed_episode_ids

    @pytest.mark.asyncio
    async def test_extract_memory_applies_add_operation(self):
        ext = self._make_extractor()
        llm_response = json.dumps({
            "operations": [
                {
                    "action": "add",
                    "type": "explicit_info",
                    "data": {
                        "category": "Hobby",
                        "description": "plays guitar",
                        "evidence": "mentioned in chat",
                        "sources": ["ep1"],
                    },
                }
            ]
        })
        ext.llm_provider.generate = AsyncMock(return_value=llm_response)
        req = ProfileExtractRequest(
            user_id="u1",
            group_id="g1",
            new_episode={"id": "ep1", "created_at": "2024-01-01", "original_data": []},
        )
        result = await ext.extract_memory(req)
        assert result.total_items() == 1
        assert result.explicit_info[0]["category"] == "Hobby"
        assert result.explicit_info[0]["description"] == "plays guitar"

    @pytest.mark.asyncio
    async def test_extract_memory_applies_delete_operation(self):
        ext = self._make_extractor()
        old = ProfileMemory(
            memory_type=MemoryType.PROFILE,
            user_id="u1",
            timestamp=datetime.now(timezone.utc),
            explicit_info=[
                {"category": "Old", "description": "outdated info", "evidence": "", "sources": []},
            ],
        )
        llm_response = json.dumps({
            "operations": [
                {"action": "delete", "type": "explicit_info", "index": 0, "reason": "no longer valid"},
            ]
        })
        ext.llm_provider.generate = AsyncMock(return_value=llm_response)
        req = ProfileExtractRequest(
            user_id="u1",
            group_id="g1",
            new_episode={"id": "ep2", "created_at": "2024-02-01", "original_data": []},
            old_profile=old,
        )
        result = await ext.extract_memory(req)
        assert result.total_items() == 0

    @pytest.mark.asyncio
    async def test_extract_memory_returns_old_on_no_episode(self):
        ext = self._make_extractor()
        old = ProfileMemory(
            memory_type=MemoryType.PROFILE,
            user_id="u1",
            timestamp=datetime.now(timezone.utc),
        )
        req = ProfileExtractRequest(user_id="u1", group_id="g1", old_profile=old)
        result = await ext.extract_memory(req)
        assert result is old


# ============================================================================
# _format_episodes_for_llm: tool-message filtering
# ============================================================================


class TestFormatEpisodesForLLM:
    """Verify tool messages and assistant tool_calls are filtered out when
    formatting episode original_data for the profile LLM prompt."""

    def _make_extractor(self):
        return ProfileExtractor(llm_provider=MagicMock())

    def _episode(self, original_data):
        return {
            "id": "ep1",
            "created_at": "2024-01-01T00:00:00Z",
            "original_data": original_data,
        }

    def test_user_and_assistant_messages_included(self):
        ext = self._make_extractor()
        episodes = [self._episode([
            {"role": "user", "sender_name": "Alice", "content": "hi", "timestamp": "t1"},
            {"role": "assistant", "sender_name": "Bot", "content": "hello", "timestamp": "t2"},
        ])]
        text = ext._format_episodes_for_llm(episodes, {"ep1": "ep1"})
        assert "【Alice】: hi" in text
        assert "【Bot】: hello" in text

    def test_tool_role_is_filtered(self):
        ext = self._make_extractor()
        episodes = [self._episode([
            {"role": "user", "sender_name": "Alice", "content": "run it", "timestamp": "t1"},
            {"role": "tool", "sender_name": "search_api", "content": "tool output", "timestamp": "t2"},
        ])]
        text = ext._format_episodes_for_llm(episodes, {"ep1": "ep1"})
        assert "【Alice】: run it" in text
        assert "tool output" not in text
        assert "search_api" not in text

    def test_assistant_with_tool_calls_is_filtered(self):
        ext = self._make_extractor()
        episodes = [self._episode([
            {
                "role": "assistant",
                "sender_name": "Bot",
                "content": "thinking...",
                "tool_calls": [{"id": "tc1", "function": {"name": "search"}}],
                "timestamp": "t1",
            },
            {"role": "assistant", "sender_name": "Bot", "content": "final answer", "timestamp": "t2"},
        ])]
        text = ext._format_episodes_for_llm(episodes, {"ep1": "ep1"})
        assert "thinking..." not in text
        assert "final answer" in text

    def test_assistant_with_empty_tool_calls_included(self):
        """tool_calls=[] is falsy; message should NOT be filtered."""
        ext = self._make_extractor()
        episodes = [self._episode([
            {
                "role": "assistant",
                "sender_name": "Bot",
                "content": "regular reply",
                "tool_calls": [],
                "timestamp": "t1",
            },
        ])]
        text = ext._format_episodes_for_llm(episodes, {"ep1": "ep1"})
        assert "regular reply" in text

    def test_assistant_with_none_tool_calls_included(self):
        ext = self._make_extractor()
        episodes = [self._episode([
            {
                "role": "assistant",
                "sender_name": "Bot",
                "content": "regular reply",
                "tool_calls": None,
                "timestamp": "t1",
            },
        ])]
        text = ext._format_episodes_for_llm(episodes, {"ep1": "ep1"})
        assert "regular reply" in text

    def test_missing_tool_calls_key_included(self):
        ext = self._make_extractor()
        episodes = [self._episode([
            {"role": "assistant", "sender_name": "Bot", "content": "regular reply", "timestamp": "t1"},
        ])]
        text = ext._format_episodes_for_llm(episodes, {"ep1": "ep1"})
        assert "regular reply" in text

    def test_nested_message_wrapper_is_unwrapped(self):
        ext = self._make_extractor()
        episodes = [self._episode([
            {"message": {
                "role": "tool",
                "sender_name": "search_api",
                "content": "tool result",
                "timestamp": "t1",
            }},
            {"message": {
                "role": "user",
                "sender_name": "Alice",
                "content": "kept message",
                "timestamp": "t2",
            }},
        ])]
        text = ext._format_episodes_for_llm(episodes, {"ep1": "ep1"})
        assert "tool result" not in text
        assert "【Alice】: kept message" in text

    def test_mixed_messages_preserves_only_conversational(self):
        ext = self._make_extractor()
        episodes = [self._episode([
            {"role": "user", "sender_name": "Alice", "content": "question", "timestamp": "t1"},
            {
                "role": "assistant",
                "sender_name": "Bot",
                "content": "let me check",
                "tool_calls": [{"id": "tc1"}],
                "timestamp": "t2",
            },
            {"role": "tool", "sender_name": "api", "content": "{\"data\": 1}", "timestamp": "t3"},
            {"role": "assistant", "sender_name": "Bot", "content": "the answer is 1", "timestamp": "t4"},
        ])]
        text = ext._format_episodes_for_llm(episodes, {"ep1": "ep1"})
        assert "question" in text
        assert "the answer is 1" in text
        assert "let me check" not in text
        assert "{\"data\": 1}" not in text


# ============================================================================
# Unified extraction (no SOLO/TEAM split)
# ============================================================================


class TestUnifiedExtraction:
    """Verify there is no SOLO/TEAM branching in the extraction pipeline."""

    def test_profile_manager_has_single_extract_method(self):
        """ProfileManager should have exactly one extract_profiles method, no extract_profiles_life."""
        from memory_layer.profile_manager.manager import ProfileManager
        assert hasattr(ProfileManager, "extract_profiles")
        assert not hasattr(ProfileManager, "extract_profiles_life")

    def test_profile_manager_config_has_no_scenario(self):
        """ProfileManagerConfig should not have a scenario field."""
        from memory_layer.profile_manager.config import ProfileManagerConfig
        config = ProfileManagerConfig()
        assert not hasattr(config, "scenario")

    def test_no_discriminator_module(self):
        """ValueDiscriminator (dead code) should be deleted."""
        import importlib
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("memory_layer.profile_manager.discriminator")

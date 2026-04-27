"""Profile Memory Extractor.

Extracts user profiles (explicit info + implicit traits) from conversations
using incremental LLM-based operations (add/update/delete).

Includes ID mapping to reduce token consumption and LLM hallucination.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from common_utils.datetime_utils import get_now_with_timezone
from core.observation.logger import get_logger
from memory_layer.llm.llm_provider import LLMProvider
from memory_layer.memory_extractor.base_memory_extractor import (
    MemoryExtractor,
    MemoryExtractRequest,
)
from memory_layer.prompts import get_prompt_by
from api_specs.memory_types import MemCell, MemoryType, ProfileMemory, ScenarioType, get_text_from_content_items, is_intermediate_agent_step

logger = get_logger(__name__)


# ============================================================================
# ID Mapper — Long ID <-> Short ID conversion to save tokens
# ============================================================================

def _create_id_mapping(long_ids: List[str]) -> Dict[str, str]:
    return {lid: f"ep{i+1}" for i, lid in enumerate(long_ids) if lid}


def _replace_sources(
    profile_dict: Dict[str, Any], id_map: Dict[str, str], reverse: bool = False
) -> Dict[str, Any]:
    mapping = {v: k for k, v in id_map.items()} if reverse else id_map
    result = copy.deepcopy(profile_dict)

    def _map_source(source: Any) -> Any:
        if not isinstance(source, str) or not source:
            return source
        if "|" in source:
            prefix, sid = source.rsplit("|", 1)
            sid = sid.strip()
            return f"{prefix}|{mapping.get(sid, sid)}"
        return mapping.get(source, source)

    for item in result.get("explicit_info", []):
        item["sources"] = [_map_source(s) for s in item.get("sources", [])]
    for item in result.get("implicit_traits", []):
        item["sources"] = [_map_source(s) for s in item.get("sources", [])]

    return result


def _get_short_id(long_id: str, id_map: Dict[str, str]) -> str:
    return id_map.get(long_id, long_id)


# ============================================================================
# Extract Request
# ============================================================================

class ProfileAction(str, Enum):
    NONE = "none"
    ADD = "add"
    UPDATE = "update"
    DELETE = "delete"


class ProfileItemType(str, Enum):
    EXPLICIT_INFO = "explicit_info"
    IMPLICIT_TRAITS = "implicit_traits"


@dataclass
class ProfileExtractRequest(MemoryExtractRequest):
    """Profile extraction request."""

    new_episode: Optional[Dict[str, Any]] = None
    referenced_episodes: Optional[List[Dict[str, Any]]] = None
    cluster_episodes: Optional[List[Dict[str, Any]]] = None
    old_profile: Optional[ProfileMemory] = None

    # Scene type
    scene: ScenarioType = ScenarioType.SOLO
    # Target user display name (for TEAM scene, to disambiguate speakers)
    target_user_name: Optional[str] = None

    # Legacy fields
    memcell: Optional[MemCell] = None
    memcell_list: Optional[List[MemCell]] = None
    episode_list: Optional[List[Dict[str, Any]]] = None

    max_items: int = 25

    def __post_init__(self):
        if self.memcell_list is None:
            self.memcell_list = []
        if self.episode_list is None:
            self.episode_list = []
        if self.referenced_episodes is None:
            self.referenced_episodes = []
        if self.cluster_episodes is None:
            self.cluster_episodes = []


# ============================================================================
# Profile Extractor
# ============================================================================

class ProfileExtractor(MemoryExtractor):
    """Extracts user profiles using incremental operations (add/update/delete)."""

    DEFAULT_MAX_ITEMS = 25

    def __init__(self, llm_provider: LLMProvider):
        super().__init__(MemoryType.PROFILE)
        self.llm_provider = llm_provider

    async def extract_memory(
        self, request: ProfileExtractRequest
    ) -> Optional[ProfileMemory]:
        """Extract profile from conversation episodes."""
        new_episode = request.new_episode
        cluster_episodes = request.cluster_episodes or []
        old_profile = request.old_profile
        max_items = request.max_items or self.DEFAULT_MAX_ITEMS

        # Backward compatibility with old episode_list mode
        if not new_episode and request.episode_list:
            episodes = request.episode_list
            if episodes:
                new_episode = episodes[-1]
                cluster_episodes = episodes[:-1] if len(episodes) > 1 else []

        if not new_episode:
            logger.warning("No new episode provided for profile extraction")
            return old_profile

        # Initialize profile
        if old_profile is None:
            logger.info(f"[ProfileExtractor] No old_profile for user={request.user_id}, creating new")
            current_profile = ProfileMemory(
                memory_type=MemoryType.PROFILE,
                user_id=request.user_id or "",
                group_id=request.group_id or "",
                timestamp=get_now_with_timezone(),
            )
        else:
            logger.info(
                f"[ProfileExtractor] Using old_profile for user={request.user_id}: "
                f"explicit={len(old_profile.explicit_info)}, implicit={len(old_profile.implicit_traits)}"
            )
            current_profile = old_profile

        # Check if already processed
        ep_id = new_episode.get("id")
        if ep_id in current_profile.processed_episode_ids:
            logger.info(f"Episode {ep_id} already processed, skipping")
            return current_profile

        # Create ID mapping
        all_ids = (
            list(current_profile.processed_episode_ids)
            + [ep.get("id") for ep in cluster_episodes]
            + [new_episode.get("id")]
        )
        id_map = _create_id_mapping(all_ids)

        logger.info(f"Processing profile: cluster={len(cluster_episodes)}, new=1")

        # Resolve target_user_name for TEAM scene
        target_user_name = request.target_user_name
        if not target_user_name and request.scene == ScenarioType.TEAM:
            target_user_name = self._resolve_user_name(
                request.user_id, [new_episode] + cluster_episodes
            )

        # Call LLM to update
        updated_dict = await self._llm_update_profile(
            current_profile=current_profile,
            cluster_episodes=cluster_episodes,
            new_episode=new_episode,
            id_map=id_map,
            scene=request.scene,
            target_user_name=target_user_name,
        )

        if updated_dict:
            current_profile.explicit_info = [
                d for d in updated_dict.get(ProfileItemType.EXPLICIT_INFO, [])
                if d.get("description", "").strip()
            ]
            current_profile.implicit_traits = [
                d for d in updated_dict.get(ProfileItemType.IMPLICIT_TRAITS, [])
                if d.get("description", "").strip()
            ]
            current_profile.last_updated = get_now_with_timezone()

        # Mark as processed
        new_ep_id = new_episode.get("id", "")
        if new_ep_id:
            current_profile.processed_episode_ids.append(new_ep_id)

        # Check capacity
        compact_threshold = int(max_items * 1.5)
        compact_target = int(max_items * 0.7)

        if current_profile.total_items() > compact_threshold:
            logger.info(
                f"Profile has {current_profile.total_items()} items (threshold={compact_threshold}), "
                f"compacting to {compact_target}..."
            )
            current_profile = await self._compact_profile(
                current_profile, compact_target, id_map
            )

        return current_profile

    async def _llm_update_profile(
        self,
        current_profile: ProfileMemory,
        cluster_episodes: List[Dict[str, Any]],
        new_episode: Dict[str, Any],
        id_map: Dict[str, str],
        scene: ScenarioType = ScenarioType.SOLO,
        target_user_name: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Call LLM for incremental update using operations (add/update/delete)."""

        profile_dict = current_profile.to_dict()
        profile_short = _replace_sources(profile_dict, id_map)
        profile_text = self._format_profile_with_index(profile_short)

        all_episodes = (cluster_episodes or []) + ([new_episode] if new_episode else [])
        conversations_text = self._format_episodes_for_llm(all_episodes, id_map)

        empty_profile = "(Empty, no records yet)"
        empty_conv = "(No conversations)"

        if scene == ScenarioType.TEAM and target_user_name:
            prompt_template = get_prompt_by("TEAM_PROFILE_UPDATE_PROMPT")
            prompt = prompt_template.format(
                target_user=target_user_name,
                current_profile=profile_text or empty_profile,
                conversations=conversations_text or empty_conv,
            )
        else:
            prompt_template = get_prompt_by("PROFILE_UPDATE_PROMPT")
            prompt = prompt_template.format(
                current_profile=profile_text or empty_profile,
                conversations=conversations_text or empty_conv,
            )

        try:
            response = await self.llm_provider.generate(prompt, temperature=0.3)
            result = self._parse_profile_response(response)
            if not result:
                return None

            operations = result.get("operations", [])

            explicit_list = list(current_profile.explicit_info)
            implicit_list = list(current_profile.implicit_traits)

            id_to_ts = self._build_timestamp_map(
                current_profile, cluster_episodes, new_episode
            )

            for op in operations:
                action = op.get("action", ProfileAction.NONE)

                if action == ProfileAction.NONE:
                    continue

                elif action == ProfileAction.ADD:
                    op_type = op.get("type")
                    data = op.get("data", {})
                    if not data.get("description", "").strip():
                        continue
                    data["sources"] = [
                        self._attach_ts(s, id_to_ts) for s in data.get("sources", [])
                    ]
                    if op_type == ProfileItemType.EXPLICIT_INFO:
                        explicit_list.append(data)
                        logger.info(f"[Profile] Added explicit_info: {data.get('description', '')[:30]}...")
                    elif op_type == ProfileItemType.IMPLICIT_TRAITS:
                        implicit_list.append(data)
                        logger.info(f"[Profile] Added implicit_trait: {data.get('trait', '')}...")

                elif action == ProfileAction.UPDATE:
                    op_type = op.get("type")
                    index = op.get("index", -1)
                    data = op.get("data", {})
                    target_list = (
                        explicit_list if op_type == ProfileItemType.EXPLICIT_INFO else implicit_list
                    )
                    if 0 <= index < len(target_list):
                        for key, val in data.items():
                            if val:
                                if key == "sources":
                                    old_sources = target_list[index].get("sources", [])
                                    new_sources = [self._attach_ts(s, id_to_ts) for s in val]
                                    target_list[index]["sources"] = list(set(old_sources + new_sources))
                                else:
                                    target_list[index][key] = val
                        logger.info(f"[Profile] Updated {op_type}[{index}]")

                elif action == ProfileAction.DELETE:
                    op_type = op.get("type")
                    index = op.get("index", -1)
                    reason = op.get("reason", "")
                    target_list = (
                        explicit_list if op_type == ProfileItemType.EXPLICIT_INFO else implicit_list
                    )
                    if 0 <= index < len(target_list) and reason:
                        target_list.pop(index)
                        logger.warning(f"[Profile] Deleted {op_type}[{index}]: {reason}")

            result_dict = {
                ProfileItemType.EXPLICIT_INFO: explicit_list,
                ProfileItemType.IMPLICIT_TRAITS: implicit_list,
            }
            return _replace_sources(result_dict, id_map, reverse=True)

        except Exception as e:
            logger.error(f"LLM update profile failed: {e}")
            return None

    def _build_timestamp_map(
        self,
        profile: ProfileMemory,
        cluster_episodes: List[Dict[str, Any]],
        new_episode: Dict[str, Any],
    ) -> Dict[str, str]:
        id_to_ts = {}

        for item in profile.explicit_info + profile.implicit_traits:
            for src in item.get("sources", []):
                if "|" in str(src):
                    ts, eid = str(src).rsplit("|", 1)
                    id_to_ts[eid.strip()] = ts.strip()

        for ep in (cluster_episodes or []) + ([new_episode] if new_episode else []):
            eid = ep.get("id")
            ts = self._format_timestamp(ep.get("created_at"))
            if eid and ts:
                id_to_ts[str(eid)] = ts

        return id_to_ts

    def _attach_ts(self, s: Any, id_to_ts: Dict[str, str]) -> str:
        if not isinstance(s, str) or not s:
            return s
        if "|" in s:
            return s
        sid = s.strip()
        ts = id_to_ts.get(sid)
        return f"{ts}|{sid}" if ts else sid

    def _format_profile_with_index(self, profile_dict: Dict[str, Any]) -> str:
        explicit = profile_dict.get(ProfileItemType.EXPLICIT_INFO, [])
        implicit = profile_dict.get(ProfileItemType.IMPLICIT_TRAITS, [])

        if not explicit and not implicit:
            return ""

        lines = []
        if explicit:
            lines.append("【Explicit Info】")
            for i, item in enumerate(explicit):
                lines.append(f"  [{i}] [{item.get('category', '')}] {item.get('description', '')}")
                if item.get("evidence"):
                    lines.append(f"      evidence: {item['evidence']}")

        if implicit:
            lines.append("\n【Implicit Traits】")
            for i, item in enumerate(implicit):
                lines.append(f"  [{i}] {item.get('trait', '')}: {item.get('description', '')}")
                if item.get("evidence"):
                    lines.append(f"      evidence: {item['evidence']}")

        return "\n".join(lines)

    async def _compact_profile(
        self, profile: ProfileMemory, max_items: int, id_map: Dict[str, str]
    ) -> ProfileMemory:
        profile_dict = profile.to_dict()
        profile_short = _replace_sources(profile_dict, id_map)
        profile_text = self._format_profile_for_llm(profile_short)
        total = profile.total_items()

        prompt_template = get_prompt_by("PROFILE_COMPACT_PROMPT")
        prompt = prompt_template.format(
            total_items=total, max_items=max_items, profile_text=profile_text
        )

        try:
            response = await self.llm_provider.generate(prompt, temperature=0.3)
            result = self._parse_profile_response(response)

            if result:
                result_long = _replace_sources(result, id_map, reverse=True)
                profile.explicit_info = [
                    d for d in result_long.get(ProfileItemType.EXPLICIT_INFO, [])
                    if d.get("description", "").strip()
                ]
                profile.implicit_traits = [
                    d for d in result_long.get(ProfileItemType.IMPLICIT_TRAITS, [])
                    if d.get("description", "").strip()
                ]
                profile.last_updated = get_now_with_timezone()

            return profile

        except Exception as e:
            logger.error(f"LLM compact profile failed: {e}")
            return profile

    def _format_profile_for_llm(self, profile_dict: Dict[str, Any]) -> str:
        explicit = profile_dict.get(ProfileItemType.EXPLICIT_INFO, [])
        implicit = profile_dict.get(ProfileItemType.IMPLICIT_TRAITS, [])

        if not explicit and not implicit:
            return ""

        lines = []
        if explicit:
            lines.append("【Explicit Info】")
            for i, item in enumerate(explicit, 1):
                lines.append(f"  {i}. [{item.get('category', '')}] {item.get('description', '')}")
                if item.get("evidence"):
                    lines.append(f"     evidence: {item['evidence']}")
                lines.append(f"     sources: {', '.join(item.get('sources', []))}")

        if implicit:
            lines.append("\n【Implicit Traits】")
            for i, item in enumerate(implicit, 1):
                lines.append(f"  {i}. {item.get('trait', '')}: {item.get('description', '')}")
                if item.get("basis"):
                    lines.append(f"     basis: {item['basis']}")
                if item.get("evidence"):
                    lines.append(f"     evidence: {item['evidence']}")
                lines.append(f"     sources: {', '.join(item.get('sources', []))}")

        return "\n".join(lines)

    def _format_episodes_for_llm(
        self, episodes: List[Dict[str, Any]], id_map: Dict[str, str]
    ) -> str:
        if not episodes:
            return ""

        lines = []
        for ep in episodes:
            short_id = _get_short_id(ep.get("id"), id_map)
            timestamp = self._format_timestamp(ep.get("created_at"))
            lines.append(f"[{short_id}] ({timestamp})")

            original_data = ep.get("original_data", [])
            if original_data and isinstance(original_data, list):
                for msg in original_data:
                    m = msg.get("message", msg)
                    if is_intermediate_agent_step(m):
                        continue
                    sender = m.get("sender_name", "Unknown")
                    content = get_text_from_content_items(m.get("content", []))
                    ts = m.get("timestamp", "")
                    if content:
                        lines.append(f"  [{ts}]【{sender}】: {content}\n\n")
            else:
                episode = ep.get("episode")
                if episode:
                    lines.append(f"  [Episode Memory] {episode}")

            lines.append("")

        return "\n".join(lines)

    def _format_timestamp(self, ts: Any) -> str:
        if not ts:
            return ""
        if isinstance(ts, datetime):
            return ts.strftime("%Y-%m-%d %H:%M")
        if isinstance(ts, str):
            return ts[:16] if len(ts) >= 16 else ts
        return str(ts)[:16]

    def _resolve_user_name(
        self, user_id: str, episodes: List[Dict[str, Any]]
    ) -> Optional[str]:
        """Resolve user_id to sender_name from episode original_data."""
        for ep in episodes:
            for msg in ep.get("original_data", []):
                m = msg.get("message", msg)
                if m.get("sender_id") == user_id:
                    name = m.get("sender_name")
                    if name:
                        return name
        # Fallback to user_id itself
        logger.warning(f"Could not resolve sender_name for user_id={user_id}, using user_id as fallback")
        return user_id

    def _parse_profile_response(self, response: str) -> Optional[Dict[str, Any]]:
        if not response:
            return None

        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
        raw_json = json_match.group(1) if json_match else response

        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError:
            brace_start = response.find("{")
            brace_end = response.rfind("}") + 1
            if brace_start >= 0 and brace_end > brace_start:
                try:
                    data = json.loads(response[brace_start:brace_end])
                except Exception:
                    logger.warning("Failed to parse profile response JSON")
                    return None
            else:
                logger.warning("No JSON found in profile response")
                return None

        update_note = data.get("update_note") or data.get("compact_note")
        if update_note:
            logger.info(f"Profile update: {update_note}")

        return data

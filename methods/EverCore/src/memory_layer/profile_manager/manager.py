"""ProfileManager - Pure computation component for profile extraction.

This module provides pure computation logic for extracting user profiles
from memcells. Storage is managed by the caller, not by ProfileManager itself.

Design:
- ProfileManager is a pure computation component
- Input: memcells + old_profiles
- Output: new_profiles
- Caller is responsible for loading/saving profiles
"""

import asyncio
from typing import Any, Dict, List, Optional

from memory_layer.llm.llm_provider import LLMProvider
from api_specs.memory_types import ProfileMemory, ScenarioType
from memory_layer.memory_extractor.profile_extractor import (
    ProfileExtractor,
    ProfileExtractRequest,
)
from memory_layer.profile_manager.config import ProfileManagerConfig
from core.observation.logger import get_logger

logger = get_logger(__name__)


class ProfileManager:
    """Pure computation component for profile extraction.

    ProfileManager extracts user profiles from memcells using LLM.
    It does NOT handle storage - the caller is responsible for loading
    old profiles and saving new profiles.

    Usage:
        ```python
        profile_mgr = ProfileManager(llm_provider, config)

        # Caller loads old profiles
        old_profiles = await storage.get_all_profiles()

        # Pure computation - extract profiles
        new_profiles = await profile_mgr.extract_profiles(
            memcells=memcell_list,
            old_profiles=list(old_profiles.values()),
            user_id_list=["user1", "user2"],
        )

        # Caller saves new profiles
        for profile in new_profiles:
            await storage.save_profile(profile.user_id, profile)
        ```
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        config: Optional[ProfileManagerConfig] = None,
        group_id: Optional[str] = None,
    ):
        self.llm_provider = llm_provider
        self.config = config or ProfileManagerConfig()
        self.group_id = group_id or "default"
        self._extractor = ProfileExtractor(llm_provider=llm_provider)
        self._stats = {
            "total_extractions": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
        }

    async def extract_profiles(
        self,
        memcells: List[Any],
        old_profiles: Optional[List[Any]] = None,
        user_id_list: Optional[List[str]] = None,
        group_id: Optional[str] = None,
        max_items: int = 25,
        scene: ScenarioType = ScenarioType.SOLO,
    ) -> List[ProfileMemory]:
        """Extract profiles from memcells (batch multi-user).

        The LLM will see 3 types of information:
        1. old_profile - Current user profile (each entry contains evidence + sources)
        2. cluster_memcells - MemCells from the same cluster (for context reference)
        3. new_memcell - The latest MemCell (last in the list)

        Args:
            memcells: List of MemCells (last one is new_memcell, others are cluster context)
            old_profiles: List of existing profiles (for incremental updates)
            user_id_list: List of user IDs to extract profiles for
            group_id: Group ID (optional)
            max_items: Maximum number of profile items

        Returns:
            List of ProfileMemory objects
        """
        self._stats["total_extractions"] += 1

        if not memcells:
            logger.error("No memcells provided for profile extraction")
            return []

        if not user_id_list:
            logger.error("No user_id_list provided for profile extraction")
            return []

        # Last memcell is new_memcell, others are cluster context
        new_memcell = memcells[-1]
        cluster_memcells = memcells[:-1] if len(memcells) > 1 else []

        # Convert memcells to episode dicts for LLM
        new_context = self._extract_context_from_memcell(new_memcell)
        cluster_contexts = [
            self._extract_context_from_memcell(mc) for mc in cluster_memcells
        ]

        # Convert old_profiles list to dict by user_id
        old_profiles_dict: Dict[str, ProfileMemory] = {}
        logger.info(f"[Profile] Processing {len(old_profiles or [])} old profiles")
        for p in old_profiles or []:
            uid = (
                p.get("user_id") if isinstance(p, dict) else getattr(p, "user_id", None)
            )
            p_dict = p if isinstance(p, dict) else p.to_dict()
            has_explicit = "explicit_info" in p_dict
            logger.info(
                f"[Profile] Old profile: user_id={uid}, has_explicit_info={has_explicit}, keys={list(p_dict.keys())[:5]}"
            )
            if uid and has_explicit:
                old_profiles_dict[uid] = ProfileMemory.from_dict(p_dict)
                logger.info(
                    f"[Profile] Loaded profile for {uid}: {old_profiles_dict[uid].total_items()} items"
                )

        results: List[ProfileMemory] = []
        logger.info(
            f"[Profile] user_id_list={user_id_list}, old_profiles_dict keys={list(old_profiles_dict.keys())}"
        )

        # Extract for each user
        for user_id in user_id_list:
            old_profile = old_profiles_dict.get(user_id)
            logger.info(
                f"[Profile] Looking for user_id={user_id}, found={old_profile is not None}"
            )

            # --- Per-user original_data filtering (Layer 2 of 2) ---
            # Layer 1 (mem_memorize.py) fetches memcells from all clusters that ANY user
            # in the group might need. This layer narrows it down per user:
            #   - Existing user: baseline = profile.last_updated → skip original_data
            #     already incorporated into their profile, only pass new ones to LLM.
            #   - New user: baseline = current memcell timestamp → start with just
            #     new_episode (original_data) for initial profile, avoids unbounded context.
            # new_episode is always passed separately and is not affected by this filter.
            if old_profile and old_profile.last_updated:
                user_baseline = old_profile.last_updated
            else:
                user_baseline = new_context.get("created_at")

            user_cluster_episodes = [
                ep for ep in cluster_contexts
                if ep.get("created_at") is None or ep.get("created_at") > user_baseline
            ]

            # Build request
            request = ProfileExtractRequest(
                new_episode=new_context,
                cluster_episodes=user_cluster_episodes,
                old_profile=old_profile,
                user_id=user_id,
                group_id=group_id or self.group_id,
                max_items=max_items,
                scene=scene,
            )

            # Extract with retry
            for attempt in range(self.config.max_retries):
                try:
                    logger.info(
                        f"Extracting profile for user {user_id} (attempt {attempt + 1})..."
                    )

                    result = await self._extractor.extract_memory(request)

                    if result:
                        self._stats["successful_extractions"] += 1
                        logger.info(
                            f"Profile extracted for {user_id}: {result.total_items()} items "
                            f"(explicit: {len(result.explicit_info)}, implicit: {len(result.implicit_traits)})"
                        )
                        results.append(result)
                    else:
                        logger.warning(
                            f"Profile extraction returned None for {user_id}"
                        )
                        if old_profile:
                            results.append(old_profile)
                    break

                except Exception as e:
                    logger.warning(
                        f"Profile extraction attempt {attempt + 1} for {user_id} failed: {e}"
                    )
                    if attempt < self.config.max_retries - 1:
                        await asyncio.sleep(0.5 * (attempt + 1))
                    else:
                        logger.error(
                            f"All profile extraction attempts failed for {user_id}"
                        )
                        if old_profile:
                            results.append(old_profile)

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get extraction statistics."""
        return dict(self._stats)

    def _extract_context_from_memcell(self, memcell: Any) -> Dict[str, Any]:
        """Extract context from MemCell for LLM."""
        if isinstance(memcell, dict):
            event_id = str(memcell.get("event_id", "") or memcell.get("id", ""))
            created_at = memcell.get("timestamp") or memcell.get("created_at")
            original_data = memcell.get("original_data", [])
        else:
            event_id = (
                str(memcell.event_id)
                if hasattr(memcell, 'event_id') and memcell.event_id
                else ""
            )
            created_at = memcell.timestamp if hasattr(memcell, 'timestamp') else None
            original_data = (
                memcell.original_data if hasattr(memcell, 'original_data') else []
            )

        return {
            "id": event_id,
            "created_at": created_at,
            "original_data": original_data,
        }

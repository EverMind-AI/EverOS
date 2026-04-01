"""
Batch Episode Worker
====================
Periodically generates high-quality episodic memories from accumulated MemCells.

Designed to work alongside ``REALTIME_EVENT_LOG_ONLY=true``.  In that mode,
real-time message processing extracts only event_logs immediately (no LLM
context needed for atomic facts).  This worker wakes up on a schedule, merges
several consecutive MemCells from the same group into one rich context window,
and then asks the episode LLM to produce a narrative – preserving the full
multi-turn quality that a single isolated message cannot provide.

Env vars (all optional):
  REALTIME_EVENT_LOG_ONLY=true      – turns on realtime mode (worker is needed)
  BATCH_EPISODE_INTERVAL=15         – worker run interval in minutes (default 15)
  BATCH_EPISODE_LOOKBACK=120        – lookback window in minutes (default 120)
  BATCH_EPISODE_MIN_CELLS=3         – min pending MemCells per group to trigger
                                      episode generation (default 3)
"""

from __future__ import annotations

import asyncio
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from api_specs.memory_types import EpisodeMemory, MemCell, MemoryType, RawDataType
from common_utils.datetime_utils import get_now_with_timezone
from core.di import get_bean_by_type
from core.observation.logger import get_logger
from infra_layer.adapters.out.persistence.repository.memcell_raw_repository import (
    MemCellRawRepository,
)
from memory_layer.memory_manager import MemoryManager

logger = get_logger(__name__)

_ROBOT_KEYWORDS = ("robot", "assistant")


class BatchEpisodeWorker:
    """Generates episodic memories from buffered MemCells in a background loop."""

    def __init__(
        self,
        interval_minutes: int = 15,
        lookback_minutes: int = 120,
        min_cells: int = 3,
    ) -> None:
        self.interval_minutes = int(
            os.getenv("BATCH_EPISODE_INTERVAL", str(interval_minutes))
        )
        self.lookback_minutes = int(
            os.getenv("BATCH_EPISODE_LOOKBACK", str(lookback_minutes))
        )
        self.min_cells = int(
            os.getenv("BATCH_EPISODE_MIN_CELLS", str(min_cells))
        )
        self._running = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run_once(self) -> int:
        """Scan recent MemCells and generate episodes for every pending group.

        Returns:
            Total number of episode documents saved across all groups.
        """
        now = get_now_with_timezone()
        start_time = now - timedelta(minutes=self.lookback_minutes)

        try:
            repo = get_bean_by_type(MemCellRawRepository)
            recent_cells: List[MemCell] = await repo.find_by_time_range(
                start_time=start_time, end_time=now
            )
        except Exception as exc:
            logger.error("[BatchEpisode] Failed to query MemCells: %s", exc)
            return 0

        # Group unprocessed cells by group_id
        by_group: Dict[str, List[MemCell]] = defaultdict(list)
        for mc in recent_cells:
            if (mc.extend or {}).get("batch_ep_done"):
                continue
            gid = mc.group_id or "__ungrouped__"
            by_group[gid].append(mc)

        if not by_group:
            logger.debug("[BatchEpisode] No pending MemCells found in lookback window")
            return 0

        total = 0
        for group_id, cells in by_group.items():
            if len(cells) < self.min_cells:
                logger.debug(
                    "[BatchEpisode] Group %s: %d cells < min=%d, skipping",
                    group_id,
                    len(cells),
                    self.min_cells,
                )
                continue
            try:
                count = await self._generate_episodes_for_group(group_id, cells)
                total += count
            except Exception as exc:
                logger.error(
                    "[BatchEpisode] Group %s episode generation failed: %s",
                    group_id,
                    exc,
                )

        logger.info(
            "[BatchEpisode] Run complete – %d episodes saved across %d groups",
            total,
            len(by_group),
        )
        return total

    async def start_periodic(self) -> None:
        """Blocking periodic loop – run as an asyncio background Task."""
        self._running = True
        logger.info(
            "[BatchEpisode] Worker started (interval=%dm, lookback=%dm, min_cells=%d)",
            self.interval_minutes,
            self.lookback_minutes,
            self.min_cells,
        )
        while self._running:
            try:
                await self.run_once()
            except Exception as exc:
                logger.error("[BatchEpisode] Unexpected error in periodic loop: %s", exc)
            await asyncio.sleep(self.interval_minutes * 60)

    def stop(self) -> None:
        self._running = False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _generate_episodes_for_group(
        self, group_id: str, memcells: List[MemCell]
    ) -> int:
        """Merge *memcells* for *group_id* and generate episodic memories.

        Returns number of episode documents saved.
        """
        # Sort ascending so the narrative is chronological
        # Use a timezone-aware sentinel so comparison with aware timestamps works
        _EPOCH = datetime.min.replace(tzinfo=timezone.utc)
        cells = sorted(
            memcells, key=lambda mc: mc.timestamp or _EPOCH
        )

        # Merge raw conversation messages
        merged_messages: list = []
        all_user_ids: list = []
        all_participants: list = []
        for mc in cells:
            merged_messages.extend(mc.original_data or [])
            all_user_ids.extend(mc.user_id_list or [])
            all_participants.extend(mc.participants or [])

        # Deduplicate while preserving order
        all_user_ids = list(dict.fromkeys(all_user_ids))
        all_participants = list(dict.fromkeys(all_participants))

        if not merged_messages:
            return 0

        real_group_id = group_id if group_id != "__ungrouped__" else None

        # Build a synthetic MemCell representing the merged context window
        merged_cell = MemCell(
            original_data=merged_messages,
            user_id_list=all_user_ids,
            timestamp=cells[-1].timestamp or get_now_with_timezone(),
            group_id=real_group_id,
            participants=all_participants,
            type=cells[0].type or RawDataType.CONVERSATION,
            summary="",
        )

        # --- Extract group episode ------------------------------------------
        memory_manager = MemoryManager()
        try:
            group_episode = await asyncio.wait_for(
                memory_manager.extract_memory(
                    memcell=merged_cell,
                    memory_type=MemoryType.EPISODIC_MEMORY,
                    user_id=None,  # None → group episode
                    group_id=real_group_id,
                ),
                timeout=120,
            )
        except asyncio.TimeoutError:
            logger.warning("[BatchEpisode] Group %s: episode extraction timed out", group_id)
            return 0
        except Exception as exc:
            logger.error("[BatchEpisode] Group %s: extraction error: %s", group_id, exc)
            return 0

        if group_episode is None or isinstance(group_episode, Exception):
            logger.debug("[BatchEpisode] Group %s: no episode extracted", group_id)
            return 0

        # Ensure metadata fields are populated
        if not getattr(group_episode, "group_id", None):
            group_episode.group_id = real_group_id
        if not getattr(group_episode, "user_name", None):
            group_episode.user_name = getattr(group_episode, "user_id", None)

        # --- Save episodes --------------------------------------------------
        # Defer imports to avoid circular dependencies
        from biz_layer.mem_db_operations import _convert_episode_memory_to_doc
        from biz_layer.mem_memorize import MemoryDocPayload, save_memory_docs
        from dataclasses import replace as dc_replace

        current_time = get_now_with_timezone()

        episodes_to_save: List[EpisodeMemory] = [group_episode]

        # Clone group episode for each real user (mirrors _clone_episodes_for_users)
        for uid in all_participants:
            if any(kw in uid.lower() for kw in _ROBOT_KEYWORDS):
                continue
            episodes_to_save.append(
                dc_replace(group_episode, user_id=uid, user_name=uid)
            )

        docs = [_convert_episode_memory_to_doc(ep, current_time) for ep in episodes_to_save]
        payloads = [MemoryDocPayload(MemoryType.EPISODIC_MEMORY, doc) for doc in docs]
        saved_map = await save_memory_docs(payloads)
        count = len(saved_map.get(MemoryType.EPISODIC_MEMORY, []))

        # --- Mark MemCells as batch-processed --------------------------------
        repo = get_bean_by_type(MemCellRawRepository)
        for mc in cells:
            try:
                extend = dict(mc.extend or {})
                extend["batch_ep_done"] = True
                await repo.update_by_event_id(str(mc.event_id), {"extend": extend})
            except Exception as exc:
                logger.warning(
                    "[BatchEpisode] Failed to mark MemCell %s as done: %s",
                    mc.event_id,
                    exc,
                )

        logger.info(
            "[BatchEpisode] Group %s: merged %d MemCells → %d episode docs saved",
            group_id,
            len(cells),
            count,
        )
        return count

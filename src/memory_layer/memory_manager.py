from dataclasses import dataclass
from datetime import datetime
import time
import os
import asyncio
from typing import List, Optional

from core.observation.logger import get_logger
from agentic_layer.metrics.memorize_metrics import (
    record_extract_memory_call,
    get_space_id_for_metrics,
)

from memory_layer.llm.llm_provider import LLMProvider
from memory_layer.memcell_extractor.conv_memcell_extractor import ConvMemCellExtractor
from memory_layer.memcell_extractor.base_memcell_extractor import RawData
from memory_layer.memcell_extractor.conv_memcell_extractor import (
    ConversationMemCellExtractRequest,
)
from api_specs.memory_types import (
    MemCell,
    RawDataType,
    MemoryType,
    Foresight,
    BaseMemory,
    EpisodeMemory,
    ParentType,
)
from memory_layer.memory_extractor.episode_memory_extractor import (
    EpisodeMemoryExtractor,
    EpisodeMemoryExtractRequest,
)
from memory_layer.memory_extractor.profile_memory_extractor import (
    ProfileMemoryExtractor,
    ProfileMemoryExtractRequest,
)
from memory_layer.memory_extractor.group_profile_memory_extractor import (
    GroupProfileMemoryExtractor,
    GroupProfileMemoryExtractRequest,
)
from memory_layer.memory_extractor.event_log_extractor import EventLogExtractor
from memory_layer.memory_extractor.foresight_extractor import ForesightExtractor
from memory_layer.memcell_extractor.base_memcell_extractor import StatusResult


logger = get_logger(__name__)


class MemoryManager:
    """
    Memory Manager - Responsible for orchestrating all memory extraction processes

    Responsibilities:
    1. Extract MemCell (boundary detection + raw data)
    2. Extract Episode/Foresight/EventLog/Profile and other memories (based on MemCell or episode)
    3. Manage the lifecycle of all Extractors
    4. Provide a unified memory extraction interface
    """

    def __init__(self):
        # Unified LLM Provider - shared by all extractors
        self.llm_provider = LLMProvider(
            provider_type=os.getenv("LLM_PROVIDER", "openai"),
            model=os.getenv("LLM_MODEL", "openai/gpt-4.1-mini"),  # skip-sensitive-check
            base_url=os.getenv("LLM_BASE_URL"),
            api_key=os.getenv("LLM_API_KEY", "your-api-key"),  # skip-sensitive-check
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.3")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "16384")),
        )

        # Episode Extractor - lazy initialization
        self._episode_extractor = None

    # TODO: add username
    async def extract_memcell(
        self,
        history_raw_data_list: list[RawData],
        new_raw_data_list: list[RawData],
        raw_data_type: RawDataType,
        group_id: Optional[str] = None,
        group_name: Optional[str] = None,
        user_id_list: Optional[List[str]] = None,
        old_memory_list: Optional[List[BaseMemory]] = None,
        force_boundary: bool = False,
    ) -> tuple[Optional[MemCell], Optional[StatusResult]]:
        """
        Extract MemCell (boundary detection + raw data)

        Args:
            history_raw_data_list: List of historical messages
            new_raw_data_list: List of new messages
            raw_data_type: Data type
            group_id: Group ID
            group_name: Group name
            user_id_list: List of user IDs
            old_memory_list: List of historical memories

        Returns:
            (MemCell, StatusResult) or (None, StatusResult)
        """
        now = time.time()

        # Boundary detection + create MemCell
        logger.debug(
            f"[MemoryManager] Starting boundary detection and creating MemCell"
        )

        # Enable smart_mask when history has more than 5 messages 
        smart_mask_flag = len(history_raw_data_list) > 5

        request = ConversationMemCellExtractRequest(
            history_raw_data_list,
            new_raw_data_list,
            user_id_list=user_id_list,
            group_id=group_id,
            group_name=group_name,
            old_memory_list=old_memory_list,
            smart_mask_flag=smart_mask_flag,
            force_boundary=force_boundary,
        )

        extractor = ConvMemCellExtractor(self.llm_provider)
        memcell, status_result = await extractor.extract_memcell(request)

        if not memcell:
            logger.debug(
                f"[MemoryManager] Boundary detection: no boundary reached, waiting for more messages"
            )
            return None, status_result

        logger.info(
            f"[MemoryManager] ✅ MemCell created successfully: "
            f"event_id={memcell.event_id}, "
            f"elapsed time: {time.time() - now:.2f} seconds"
        )

        return memcell, status_result

    async def extract_memory(
        self,
        memcell: MemCell,
        memory_type: MemoryType,
        user_id: Optional[
            str
        ] = None,  # None means group memory, with value means personal memory
        group_id: Optional[str] = None,
        group_name: Optional[str] = None,
        old_memory_list: Optional[List[BaseMemory]] = None,
        user_organization: Optional[List] = None,
    ):
        """
        Extract a single memory

        Args:
            memcell: Single MemCell (raw data container for memory)
            memory_type: Memory type
            user_id: User ID
                - None: Extract group Episode/group Profile
                - With value: Extract personal Episode/personal Profile
            group_id: Group ID
            group_name: Group name
            old_memory_list: List of historical memories
            user_organization: User organization information
            episodic_memory: Episodic memory (used to extract Foresight/EventLog)

        Returns:
            - EPISODIC_MEMORY: Returns Memory (group or personal)
            - FORESIGHT: Returns List[Foresight]
            - PERSONAL_EVENT_LOG: Returns EventLog
            - PROFILE/GROUP_PROFILE: Returns Memory
        """
        start_time = time.perf_counter()
        memory_type_str = memory_type.value if hasattr(memory_type, 'value') else str(memory_type)
        # Get metrics labels
        space_id = get_space_id_for_metrics()
        raw_data_type = memcell.type.value if memcell.type else 'unknown'
        result = None
        status = 'success'
        
        try:
            # Dispatch based on memory_type enum
            match memory_type:
                case MemoryType.EPISODIC_MEMORY:
                    result = await self._extract_episode(memcell, user_id, group_id)

                case MemoryType.FORESIGHT:
                    result = await self._extract_foresight(
                        memcell, user_id=user_id, group_id=group_id
                    )

                case MemoryType.EVENT_LOG:
                    result = await self._extract_event_log(
                        memcell, user_id=user_id, group_id=group_id
                    )

                case MemoryType.PROFILE:
                    result = await self._extract_profile(
                        memcell, user_id, group_id, old_memory_list
                    )

                case MemoryType.GROUP_PROFILE:
                    result = await self._extract_group_profile(
                        memcell,
                        user_id,
                        group_id,
                        group_name,
                        old_memory_list,
                        user_organization,
                    )

                case _:
                    logger.warning(f"[MemoryManager] Unknown memory_type: {memory_type}")
                    status = 'error'
                    return None
            
            # Determine status based on result
            if result is None:
                status = 'empty_result'
            elif isinstance(result, list) and len(result) == 0:
                status = 'empty_result'
            
            return result
            
        except Exception as e:
            status = 'error'
            raise
        finally:
            duration = time.perf_counter() - start_time
            record_extract_memory_call(
                space_id=space_id,
                raw_data_type=raw_data_type,
                memory_type=memory_type_str,
                status=status,
                duration_seconds=duration,
            )

    async def _extract_episode(
        self, memcell: MemCell, user_id: Optional[str], group_id: Optional[str]
    ) -> Optional[EpisodeMemory]:
        """Extract Episode (group or personal)"""
        if self._episode_extractor is None:
            self._episode_extractor = EpisodeMemoryExtractor(self.llm_provider)

        # Build extraction request
        from memory_layer.memory_extractor.base_memory_extractor import (
            MemoryExtractRequest,
        )

        request = MemoryExtractRequest(
            memcell=memcell,
            user_id=user_id,  # None=group, with value=personal
            group_id=group_id,
        )

        # Call extractor's extract_memory method
        # It will automatically determine whether to extract group or personal Episode based on user_id
        logger.debug(
            f"[MemoryManager] Extracting {'group' if user_id is None else 'personal'} Episode: user_id={user_id}"
        )

        return await self._episode_extractor.extract_memory(request)

    async def _extract_foresight(
        self,
        memcell: Optional[MemCell],
        user_id: Optional[str] = None,
        group_id: Optional[str] = None,
    ) -> List[Foresight]:
        """Extract Foresight (assistant scene uses raw conversation text)"""
        if not memcell:
            logger.warning("[MemoryManager] Missing memcell, cannot extract Foresight")
            return []
        uid = user_id
        gid = group_id
        # Build simple conversation transcript from memcell.original_data
        lines = []
        for msg in memcell.original_data or []:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or "").lower()
            if role == "assistant":
                continue
            speaker_name = msg.get("speaker_name")
            content = msg.get("content", "")
            ts = msg.get("timestamp")
            if ts:
                lines.append(f"[{ts}] {speaker_name}: {content}")
            else:
                lines.append(f"{speaker_name}: {content}")
        conversation_text = "\n".join(lines)

        # Best-effort resolve user_name from raw messages

        if uid is None:
            display_name = ",".join(
                set([msg.get("speaker_name") for msg in memcell.original_data or []])
            )
        else:
            for msg in memcell.original_data or []:
                speaker_id = msg.get("speaker_id")
                if speaker_id == uid:
                    display_name = msg.get("speaker_name")
                    break

        extractor = ForesightExtractor(llm_provider=self.llm_provider)
        foresights = await extractor.generate_foresights_for_conversation(
            conversation_text=conversation_text,
            timestamp=memcell.timestamp,
            user_id=uid,
            user_name=display_name,
            group_id=gid,
            ori_event_id_list=[memcell.event_id] if memcell.event_id else [],
        )
        # Set parent info after extraction
        for f in foresights:
            f.parent_type = ParentType.MEMCELL.value
            f.parent_id = memcell.event_id
        return foresights

    async def _extract_event_log(
        self,
        memcell: Optional[MemCell],
        user_id: Optional[str] = None,
        group_id: Optional[str] = None,
    ):
        """Extract Event Log"""
        if not memcell:
            logger.warning("[MemoryManager] Missing memcell, cannot extract EventLog")
            return None

        uid = user_id
        gid = group_id

        logger.debug(f"[MemoryManager] Extracting EventLog: user_id={uid}")

        extractor = EventLogExtractor(llm_provider=self.llm_provider)
        return await extractor.extract_event_log(
            memcell=memcell,
            timestamp=memcell.timestamp,
            user_id=uid,
            ori_event_id_list=[],
            group_id=gid,
        )

    async def _extract_profile(
        self,
        memcell: MemCell,
        user_id: Optional[str],
        group_id: Optional[str],
        old_memory_list: Optional[List[BaseMemory]],
    ) -> Optional[BaseMemory]:
        """Extract Profile"""
        if memcell.type != RawDataType.CONVERSATION:
            return None

        extractor = ProfileMemoryExtractor(self.llm_provider)
        request = ProfileMemoryExtractRequest(
            memcell_list=[memcell],
            user_id_list=[user_id] if user_id else [],
            group_id=group_id,
            old_memory_list=old_memory_list,
        )
        return await extractor.extract_memory(request)

    async def _extract_group_profile(
        self,
        memcell: MemCell,
        user_id: Optional[str],
        group_id: Optional[str],
        group_name: Optional[str],
        old_memory_list: Optional[List[BaseMemory]],
        user_organization: Optional[List],
    ) -> Optional[BaseMemory]:
        """Extract Group Profile"""
        extractor = GroupProfileMemoryExtractor(self.llm_provider)
        request = GroupProfileMemoryExtractRequest(
            memcell_list=[memcell],
            user_id_list=[user_id] if user_id else [],
            group_id=group_id,
            group_name=group_name,
            old_memory_list=old_memory_list,
            user_organization=user_organization,
        )
        return await extractor.extract_memory(request)

from dataclasses import dataclass
from datetime import datetime
import time
import asyncio
from typing import List, Optional, Dict, Any

from core.observation.logger import get_logger
from agentic_layer.metrics.memorize_metrics import (
    record_extract_memory_call,
    get_space_id_for_metrics,
)

from memory_layer.llm.llm_provider import (
    LLMProvider,
    build_default_provider,
    DEFAULT_PROVIDER_NAME,
    DEFAULT_LLM_TEMPERATURE,
    DEFAULT_LLM_MAX_TOKENS,
)
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
    get_text_from_content_items,
)
from memory_layer.memory_extractor.episode_memory_extractor import (
    EpisodeMemoryExtractor,
    EpisodeMemoryExtractRequest,
)
from memory_layer.memory_extractor.profile_extractor import (
    ProfileExtractor,
    ProfileExtractRequest,
)
from memory_layer.memory_extractor.atomic_fact_extractor import AtomicFactExtractor
from memory_layer.memory_extractor.foresight_extractor import ForesightExtractor
from memory_layer.memcell_extractor.base_memcell_extractor import StatusResult
from api_specs.memory_models import MessageSenderRole
from memory_layer.constants import EXTRACT_SCENES


logger = get_logger(__name__)


class MemoryManager:
    """
    Memory Manager - Responsible for orchestrating all memory extraction processes

    Responsibilities:
    1. Extract MemCell (boundary detection + raw data)
    2. Extract Episode/Foresight/AtomicFact/Profile and other memories (based on MemCell or episode)
    3. Manage the lifecycle of all Extractors
    4. Provide a unified memory extraction interface
    """

    SCENES = EXTRACT_SCENES

    def __init__(self, llm_config: Optional[Dict[str, Any]] = None):
        """
        Initialize MemoryManager

        Args:
            llm_config: Optional LLM configuration dict (e.g. from global settings llm_custom_setting)
                        Structure: {
                            "boundary": {"provider": "...", "model": "..."},
                            "extraction": {"provider": "...", "model": "..."},
                            "profile": {"provider": "...", "model": "..."}
                        }
        """
        self.llm_config = llm_config or {}
        self.providers_mapping: Dict[str, LLMProvider] = {}
        self._build_providers_mapping()

        # Episode Extractor - lazy initialization
        self._episode_extractor = None

    def _get_scene_config(self, scene: Optional[str]) -> Optional[Dict[str, Any]]:
        if not scene or not self.llm_config:
            return None
        cfg = self.llm_config.get(scene)
        if not cfg:
            return None
        if isinstance(cfg, dict):
            return cfg
        return {
            "provider": getattr(cfg, "provider", None),
            "model": getattr(cfg, "model", None),
            "temperature": getattr(cfg, "temperature", None),
            "max_tokens": getattr(cfg, "max_tokens", None),
            "extra": getattr(cfg, "extra", None),
        }

    def _get_scene_cfg_value(self, cfg: Dict[str, Any], key: str) -> Any:
        """Extract a value from scene config, checking 'extra' dict as fallback."""
        val = cfg.get(key)
        if val is None:
            extra = cfg.get("extra")
            if isinstance(extra, dict):
                val = extra.get(key)
        return val

    def _build_scene_provider(self, scene: str, cfg: Dict[str, Any]) -> LLMProvider:
        """Build an LLM provider from a single scene's config.

        api_key and base_url are resolved inside LLMProvider from env vars,
        not from llm_config.
        """
        provider_name = self._get_scene_cfg_value(cfg, "provider")
        if not provider_name:
            raise ValueError(f"missing provider in scene '{scene}' config")

        model = self._get_scene_cfg_value(cfg, "model")
        if not model:
            raise ValueError(
                f"missing model for provider '{provider_name}' "
                f"in scene '{scene}' config"
            )

        temperature = self._get_scene_cfg_value(cfg, "temperature")
        if temperature is None:
            temperature = DEFAULT_LLM_TEMPERATURE

        max_tokens = self._get_scene_cfg_value(cfg, "max_tokens")
        if max_tokens is None:
            max_tokens = DEFAULT_LLM_MAX_TOKENS

        return LLMProvider(
            provider_type=provider_name,
            model=model,  # skip-sensitive-check
            temperature=float(temperature),
            max_tokens=int(max_tokens),
        )

    def _build_providers_mapping(self) -> None:
        self.providers_mapping[DEFAULT_PROVIDER_NAME] = build_default_provider()
        for scene in self.SCENES:
            cfg = self._get_scene_config(scene)
            if not cfg:
                continue
            try:
                self.providers_mapping[scene] = self._build_scene_provider(scene, cfg)
            except Exception as e:
                logger.warning(
                    f"[MemoryManager] Failed to build provider for "
                    f"scene '{scene}': {e}, falling back to default"
                )

    def _get_provider_for_scene(self, scene: str) -> LLMProvider:
        provider = self.providers_mapping.get(scene)
        if provider is None:
            provider = self.providers_mapping.get(DEFAULT_PROVIDER_NAME)
        return provider

    # TODO: add username
    async def extract_memcell(
        self,
        history_raw_data_list: list[RawData],
        new_raw_data_list: list[RawData],
        raw_data_type: RawDataType,
        group_id: Optional[str] = None,
        user_id_list: Optional[List[str]] = None,
        old_memory_list: Optional[List[BaseMemory]] = None,
        flush: bool = False,
    ) -> tuple[List[MemCell], StatusResult]:
        """
        Extract MemCells using multi-split boundary detection.

        Args:
            history_raw_data_list: List of historical messages
            new_raw_data_list: List of new messages
            raw_data_type: Data type
            group_id: Group ID
            user_id_list: List of user IDs
            old_memory_list: List of historical memories
            flush: When True, remaining messages after boundary detection are packed into a final MemCell

        Returns:
            (list_of_memcells, StatusResult)
            Empty list means no boundary detected; StatusResult.should_wait for accumulation state
        """
        now = time.time()

        logger.debug(f"[MemoryManager] Starting boundary detection (flush={flush})")

        request = ConversationMemCellExtractRequest(
            history_raw_data_list,
            new_raw_data_list,
            user_id_list=user_id_list,
            group_id=group_id,
            old_memory_list=old_memory_list,
            flush=flush,
        )

        # Select extractor based on raw_data_type
        if raw_data_type == RawDataType.AGENTCONVERSATION:
            from memory_layer.memcell_extractor.agent_memcell_extractor import (
                AgentMemCellExtractor,
            )
            extractor = AgentMemCellExtractor(self._get_provider_for_scene("boundary"))
        else:
            extractor = ConvMemCellExtractor(self._get_provider_for_scene("boundary"))
        memcells, status_result = await extractor.extract_memcell(request)

        if not memcells:
            logger.debug(
                "[MemoryManager] Boundary detection: no boundary reached, waiting for more messages"
            )
            return [], status_result

        logger.info(
            f"[MemoryManager] ✅ {len(memcells)} MemCell(s) created, "
            f"elapsed time: {time.time() - now:.2f} seconds"
        )

        return memcells, status_result

    async def extract_memory(
        self,
        memcell: MemCell,
        memory_type: MemoryType,
        user_id: Optional[
            str
        ] = None,  # None means group memory, with value means personal memory
        group_id: Optional[str] = None,
        old_memory_list: Optional[List[BaseMemory]] = None,
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
            old_memory_list: List of historical memories

        Returns:
            - EPISODIC_MEMORY: Returns Memory (group or personal)
            - FORESIGHT: Returns List[Foresight]
            - PERSONAL_ATOMIC_FACT: Returns AtomicFact
            - PROFILE/GROUP_PROFILE: Returns Memory
        """
        start_time = time.perf_counter()
        memory_type_str = (
            memory_type.value if hasattr(memory_type, 'value') else str(memory_type)
        )
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

                case MemoryType.ATOMIC_FACT:
                    result = await self._extract_atomic_fact(
                        memcell, user_id=user_id, group_id=group_id
                    )

                case MemoryType.PROFILE:
                    result = await self._extract_profile(
                        memcell, user_id, group_id, old_memory_list
                    )

                case MemoryType.AGENT_CASE:
                    result = await self._extract_agent_case(
                        memcell, user_id=user_id, group_id=group_id
                    )

                case _:
                    logger.warning(
                        f"[MemoryManager] Unknown memory_type: {memory_type}"
                    )
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
            self._episode_extractor = EpisodeMemoryExtractor(
                self._get_provider_for_scene("extraction")
            )

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
        """Extract Foresight (solo scene uses raw conversation text)"""
        if not memcell:
            logger.warning("[MemoryManager] Missing memcell, cannot extract Foresight")
            return []
        uid = user_id
        gid = group_id
        # Build simple conversation transcript from memcell.conversation_data
        # (conversation_data already filters out tool calls/responses for agent conversations)
        lines = []
        for item in memcell.conversation_data or []:
            if not isinstance(item, dict):
                continue
            msg = item.get("message", item)
            role = msg.get("role")
            if role == MessageSenderRole.ASSISTANT.value:
                continue
            sender_name = msg.get("sender_name")
            content = get_text_from_content_items(msg.get("content", []))
            ts = msg.get("timestamp")
            if ts:
                lines.append(f"[{ts}] {sender_name}: {content}")
            else:
                lines.append(f"{sender_name}: {content}")
        conversation_text = "\n".join(lines)

        # Best-effort resolve user_name from raw messages

        if uid is None:
            display_name = ",".join(
                set(
                    [
                        item.get("message", item).get("sender_name")
                        for item in memcell.original_data or []
                    ]
                )
            )
        else:
            for item in memcell.original_data or []:
                msg = item.get("message", item)
                if msg.get("sender_id") == uid:
                    display_name = msg.get("sender_name")
                    break

        extractor = ForesightExtractor(
            llm_provider=self._get_provider_for_scene("extraction")
        )
        foresights = await extractor.generate_foresights_for_conversation(
            conversation_text=conversation_text,
            timestamp=memcell.timestamp,
            user_id=uid,
            user_name=display_name,
            group_id=gid,
        )
        return foresights

    async def _extract_atomic_fact(
        self,
        memcell: Optional[MemCell],
        user_id: Optional[str] = None,
        group_id: Optional[str] = None,
    ):
        """Extract Atomic Fact"""
        if not memcell:
            logger.warning("[MemoryManager] Missing memcell, cannot extract AtomicFact")
            return None

        uid = user_id
        gid = group_id

        logger.debug(f"[MemoryManager] Extracting AtomicFact: user_id={uid}")

        extractor = AtomicFactExtractor(
            llm_provider=self._get_provider_for_scene("extraction")
        )
        return await extractor.extract_atomic_fact(
            memcell=memcell, timestamp=memcell.timestamp, user_id=uid, group_id=gid
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

        extractor = ProfileExtractor(self._get_provider_for_scene("profile"))
        request = ProfileExtractRequest(
            memcell_list=[memcell],
            user_id_list=[user_id] if user_id else [],
            group_id=group_id,
        )
        return await extractor.extract_memory(request)

    async def _extract_agent_case(
        self,
        memcell: MemCell,
        user_id: Optional[str] = None,
        group_id: Optional[str] = None,
    ):
        """Extract AgentCase from an agent conversation MemCell."""
        from memory_layer.memory_extractor.agent_case_extractor import (
            AgentCaseExtractor,
            AgentCaseExtractRequest,
        )

        if not memcell or memcell.type != RawDataType.AGENTCONVERSATION:
            return None

        logger.debug("[MemoryManager] Extracting AgentCase")
        extractor = AgentCaseExtractor(
            llm_provider=self._get_provider_for_scene("extraction")
        )
        request = AgentCaseExtractRequest(
            memcell=memcell,
            user_id=user_id,
            group_id=group_id,
        )
        return await extractor.extract_memory(request)

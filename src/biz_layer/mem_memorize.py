from dataclasses import dataclass
import random
import time
import json
import traceback

from memory_layer.profile_manager.config import ScenarioType
from agentic_layer.metrics.memorize_metrics import (
    record_extraction_stage,
    record_memory_extracted,
    get_space_id_for_metrics,
)
from api_specs.dtos import MemorizeRequest
from memory_layer.memory_manager import MemoryManager
from api_specs.memory_types import (
    MemoryType,
    MemCell,
    BaseMemory,
    EpisodeMemory,
    RawDataType,
    Foresight,
)
from api_specs.memory_types import EventLog
from biz_layer.memorize_config import DEFAULT_MEMORIZE_CONFIG
from memory_layer.memory_extractor.profile_memory_extractor import ProfileMemory
from core.di import get_bean_by_type
from infra_layer.adapters.out.persistence.repository.episodic_memory_raw_repository import (
    EpisodicMemoryRawRepository,
)
from infra_layer.adapters.out.persistence.repository.foresight_record_repository import (
    ForesightRecordRawRepository,
)
from infra_layer.adapters.out.persistence.repository.event_log_record_raw_repository import (
    EventLogRecordRawRepository,
)
from infra_layer.adapters.out.persistence.repository.conversation_status_raw_repository import (
    ConversationStatusRawRepository,
)
from infra_layer.adapters.out.persistence.repository.conversation_meta_raw_repository import (
    ConversationMetaRawRepository,
)
from infra_layer.adapters.out.persistence.repository.memcell_raw_repository import (
    MemCellRawRepository,
)
from infra_layer.adapters.out.persistence.repository.core_memory_raw_repository import (
    CoreMemoryRawRepository,
)
from infra_layer.adapters.out.persistence.repository.group_user_profile_memory_raw_repository import (
    GroupUserProfileMemoryRawRepository,
)
from infra_layer.adapters.out.persistence.repository.group_profile_raw_repository import (
    GroupProfileRawRepository,
)
from infra_layer.adapters.out.persistence.repository.conversation_data_raw_repository import (
    ConversationDataRepository,
)
from api_specs.memory_types import RawDataType
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import uuid
from datetime import datetime, timedelta
import os
import asyncio
from collections import defaultdict
from common_utils.datetime_utils import get_now_with_timezone, to_iso_format
from memory_layer.memcell_extractor.base_memcell_extractor import StatusResult
import traceback

from core.observation.logger import get_logger
from infra_layer.adapters.out.search.elasticsearch.converter.episodic_memory_converter import (
    EpisodicMemoryConverter,
)
from infra_layer.adapters.out.search.milvus.converter.episodic_memory_milvus_converter import (
    EpisodicMemoryMilvusConverter,
)
from infra_layer.adapters.out.search.repository.episodic_memory_milvus_repository import (
    EpisodicMemoryMilvusRepository,
)
from infra_layer.adapters.out.search.repository.episodic_memory_es_repository import (
    EpisodicMemoryEsRepository,
)
from biz_layer.mem_sync import MemorySyncService
from core.context.context import get_current_app_info

logger = get_logger(__name__)


@dataclass
class MemoryDocPayload:
    memory_type: MemoryType
    doc: Any


from biz_layer.memorize_config import MemorizeConfig, DEFAULT_MEMORIZE_CONFIG


async def _trigger_clustering(
    group_id: str,
    memcell: MemCell,
    scene: Optional[str] = None,
    config: MemorizeConfig = DEFAULT_MEMORIZE_CONFIG,
) -> None:
    """Trigger MemCell clustering

    Args:
        group_id: Group ID
        memcell: The MemCell just saved
        scene: Conversation scene (used to determine Profile extraction strategy)
            - "group_chat": use group_chat scene
            - "assistant": use assistant scene
    """
    logger.info(
        f"[Clustering] Start triggering clustering: group_id={group_id}, event_id={memcell.event_id}, scene={scene}"
    )

    try:
        from memory_layer.cluster_manager import (
            ClusterManager,
            ClusterManagerConfig,
            ClusterState,
        )
        from memory_layer.profile_manager import ProfileManager, ProfileManagerConfig
        from infra_layer.adapters.out.persistence.repository.cluster_state_raw_repository import (
            ClusterStateRawRepository,
        )
        from infra_layer.adapters.out.persistence.repository.user_profile_raw_repository import (
            UserProfileRawRepository,
        )
        from memory_layer.llm.llm_provider import LLMProvider
        from core.di import get_bean_by_type
        import os

        logger.info(f"[Clustering] Retrieving ClusterStateRawRepository...")
        # Get MongoDB storage
        cluster_storage = get_bean_by_type(ClusterStateRawRepository)
        logger.info(
            f"[Clustering] ClusterStateRawRepository retrieved successfully: {type(cluster_storage)}"
        )

        # Create ClusterManager (pure computation component)
        cluster_config = ClusterManagerConfig(
            similarity_threshold=config.cluster_similarity_threshold,
            max_time_gap_days=config.cluster_max_time_gap_days,
        )
        cluster_manager = ClusterManager(config=cluster_config)
        logger.info(f"[Clustering] ClusterManager created successfully")

        # Load clustering state
        state_dict = await cluster_storage.load_cluster_state(group_id)
        cluster_state = (
            ClusterState.from_dict(state_dict) if state_dict else ClusterState()
        )
        logger.info(
            f"[Clustering] Loaded clustering state: {len(cluster_state.event_ids)} clustered events"
        )

        # Convert MemCell to dictionary format required for clustering
        memcell_dict = {
            "event_id": str(memcell.event_id),
            "episode": memcell.episode,
            "timestamp": memcell.timestamp.timestamp() if memcell.timestamp else None,
            "participants": memcell.participants or [],
            "group_id": group_id,
        }

        logger.info(
            f"[Clustering] Start clustering execution: {memcell_dict['event_id']}"
        )
        print(
            f"[Clustering] Start clustering execution: event_id={memcell_dict['event_id']}"
        )

        # Perform clustering (pure computation)
        cluster_id, cluster_state = await cluster_manager.cluster_memcell(
            memcell_dict, cluster_state
        )

        # Save clustering state
        await cluster_storage.save_cluster_state(group_id, cluster_state.to_dict())
        logger.info(f"[Clustering] Clustering state saved")

        print(f"[Clustering] Clustering completed: cluster_id={cluster_id}")

        if cluster_id:
            logger.info(
                f"[Clustering] ✅ MemCell {memcell.event_id} -> Cluster {cluster_id} (group: {group_id})"
            )
            print(f"[Clustering] ✅ MemCell {memcell.event_id} -> Cluster {cluster_id}")
        else:
            logger.warning(
                f"[Clustering] ⚠️ MemCell {memcell.event_id} clustering returned None (group: {group_id})"
            )
            print(f"[Clustering] ⚠️ Clustering returned None")

        # Profile extraction
        if cluster_id:
            await _trigger_profile_extraction(
                group_id=group_id,
                cluster_id=cluster_id,
                cluster_state=cluster_state,
                memcell=memcell,
                scene=scene,
                config=config,
            )

    except Exception as e:
        # Clustering failed, print detailed error and re-raise
        import traceback

        error_msg = f"[Clustering] ❌ Triggering clustering failed: {e}"
        logger.error(error_msg, exc_info=True)
        print(error_msg)  # Ensure visible in console
        print(traceback.format_exc())
        raise  # Re-raise exception so caller knows it failed


async def _trigger_profile_extraction(
    group_id: str,
    cluster_id: str,
    cluster_state,  # ClusterState
    memcell: MemCell,
    scene: Optional[str] = None,
    config: MemorizeConfig = DEFAULT_MEMORIZE_CONFIG,
) -> None:
    """Trigger Profile extraction

    Args:
        group_id: Group ID
        cluster_id: The cluster to which the current memcell was assigned
        cluster_state: Current clustering state
        memcell: The MemCell currently being processed
        scene: Conversation scene
        config: Memory extraction configuration
    """
    try:
        from memory_layer.profile_manager import ProfileManager, ProfileManagerConfig
        from infra_layer.adapters.out.persistence.repository.user_profile_raw_repository import (
            UserProfileRawRepository,
        )
        from memory_layer.llm.llm_provider import LLMProvider
        from core.di import get_bean_by_type
        import os

        # Get the number of memcells in the current cluster
        cluster_memcell_count = cluster_state.cluster_counts.get(cluster_id) or 0
        if cluster_memcell_count < config.profile_min_memcells:
            logger.debug(
                f"[Profile] Cluster {cluster_id} has only {cluster_memcell_count} memcells "
                f"(requires {config.profile_min_memcells}), skipping extraction"
            )
            return

        logger.info(
            f"[Profile] Start extracting Profile: cluster={cluster_id}, memcells={cluster_memcell_count}"
        )

        # Get Profile storage
        profile_repo = get_bean_by_type(UserProfileRawRepository)
        memcell_repo = get_bean_by_type(MemCellRawRepository)

        # Create LLM Provider
        llm_provider = LLMProvider(
            provider_type=os.getenv("LLM_PROVIDER", "openai"),
            model=os.getenv("LLM_MODEL", "gpt-4.1-mini"),  # skip-sensitive-check
            base_url=os.getenv("LLM_BASE_URL"),  # skip-sensitive-check
            api_key=os.getenv("LLM_API_KEY"),  # skip-sensitive-check
            temperature=float(
                os.getenv("LLM_TEMPERATURE", "0.3")
            ),  # skip-sensitive-check
            max_tokens=int(
                os.getenv("LLM_MAX_TOKENS", "16384")
            ),  # skip-sensitive-check
        )

        # Determine scenario
        profile_scenario = (
            ScenarioType(scene.lower()) if scene else ScenarioType.GROUP_CHAT
        )

        # Create ProfileManager (pure computation component)
        profile_config = ProfileManagerConfig(
            scenario=profile_scenario,
            min_confidence=config.profile_min_confidence,
            enable_versioning=config.profile_enable_versioning,
            auto_extract=True,
        )
        profile_manager = ProfileManager(
            llm_provider=llm_provider,
            config=profile_config,
            group_id=group_id,
            group_name=None,
        )

        # Get participant list (exclude robots)
        user_id_list = [
            u
            for u in (memcell.participants or [])
            if "robot" not in u.lower() and "assistant" not in u.lower()
        ]
        # ===== Common preprocessing: fetch all cluster memcells =====
        current_event_id = str(memcell.event_id) if memcell.event_id else cluster_id
        cluster_event_ids = set()
        if cluster_state and hasattr(cluster_state, 'eventid_to_cluster'):
            for event_id, cid in cluster_state.eventid_to_cluster.items():
                if cid == cluster_id and event_id != current_event_id:
                    cluster_event_ids.add(event_id)

        # Fetch cluster memcells + current memcell
        all_memcells = []
        if cluster_event_ids:
            try:
                cluster_memcells_dict = await memcell_repo.get_by_event_ids(
                    list(cluster_event_ids)
                )
                all_memcells = list(cluster_memcells_dict.values())
            except Exception as e:
                logger.warning(f"[Profile] Failed to fetch cluster memcells: {e}")

        # Append current memcell as the last one (new_memcell)
        all_memcells.append(memcell)
        logger.info(
            f"[Profile] Context: cluster={len(all_memcells) - 1}, new=1, users={len(user_id_list)}"
        )

        # ===== Extract and save profiles =====

        # Load old profiles (same for Work and Life)
        old_profiles_dict = await profile_repo.get_all_profiles(group_id=group_id)
        old_profiles = list(old_profiles_dict.values()) if old_profiles_dict else []
        logger.info(
            f"[Profile] Loaded {len(old_profiles)} existing profiles for group={group_id}"
        )
        if old_profiles:
            for uid, p in old_profiles_dict.items():
                keys = list(p.keys()) if isinstance(p, dict) else dir(p)
                logger.info(f"[Profile] Profile for {uid}: keys={keys[:8]}")

        # Extract profiles
        if profile_scenario == ScenarioType.ASSISTANT:
            new_profiles = await profile_manager.extract_profiles_life(
                memcells=all_memcells,
                old_profiles=old_profiles,
                user_id_list=user_id_list,
                group_id=group_id,
                max_items=config.profile_life_max_items,
            )
        else:
            new_profiles = await profile_manager.extract_profiles(
                memcells=all_memcells,
                old_profiles=old_profiles,
                user_id_list=user_id_list,
                group_id=group_id,
            )

        # Save profiles
        for profile in new_profiles:
            try:
                if profile_scenario == ScenarioType.ASSISTANT:
                    user_id = profile.user_id
                    profile_data = profile.to_dict()
                    metadata = {
                        "group_id": group_id,
                        "scenario": ScenarioType.ASSISTANT.value,
                        "cluster_id": cluster_id,
                        "memcell_count": cluster_memcell_count,
                        "total_items": profile.total_items(),
                    }
                else:
                    user_id = (
                        profile.get('user_id')
                        if isinstance(profile, dict)
                        else getattr(profile, 'user_id', None)
                    )
                    # Convert to dict if it's a ProfileMemory object
                    if hasattr(profile, 'to_dict'):
                        profile_data = profile.to_dict()
                    elif isinstance(profile, dict):
                        profile_data = profile
                    else:
                        profile_data = (
                            profile.__dict__
                            if hasattr(profile, '__dict__')
                            else profile
                        )
                    metadata = {
                        "group_id": group_id,
                        "scenario": "group_chat",
                        "cluster_id": cluster_id,
                        "memcell_count": cluster_memcell_count,
                        "confidence": config.profile_min_confidence,
                    }

                if user_id:
                    await profile_repo.save_profile(
                        user_id, profile_data, metadata=metadata
                    )
                    logger.info(f"[Profile] ✅ Saved: user={user_id}")
            except Exception as e:
                logger.warning(f"[Profile] Failed to save profile: {e}")

        logger.info(f"[Profile] ✅ Completed: {len(new_profiles)} profiles")

    except Exception as e:
        logger.error(f"[Profile] ❌ Profile extraction failed: {e}", exc_info=True)


from biz_layer.mem_db_operations import (
    _convert_timestamp_to_time,
    _convert_episode_memory_to_doc,
    _convert_foresight_to_doc,
    _convert_event_log_to_docs,
    _save_memcell_to_database,
    _save_profile_memory_to_core,
    _update_status_for_continuing_conversation,
    _update_status_after_memcell_extraction,
    _save_group_profile_memory,
    _save_profile_memory_to_group_user_profile_memory,
    _normalize_datetime_for_storage,
    _convert_projects_participated_list,
)
from typing import Tuple


def if_memorize(memcell: MemCell) -> bool:
    return True


# ==================== MemCell Processing Business Logic ====================


@dataclass
class ExtractionState:
    """Memory extraction state, stores intermediate results"""

    memcell: MemCell
    request: MemorizeRequest
    current_time: datetime
    scene: str
    is_assistant_scene: bool
    participants: List[str]
    parent_type: str = None
    parent_id: str = None
    group_episode: Optional[EpisodeMemory] = None
    group_episode_memories: List[EpisodeMemory] = None
    episode_memories: List[EpisodeMemory] = None
    parent_docs_map: Dict[str, Any] = None

    def __post_init__(self):
        self.group_episode_memories = []
        self.episode_memories = []
        self.parent_docs_map = {}
        # Set default parent info from memcell
        if self.parent_type is None:
            self.parent_type = DEFAULT_MEMORIZE_CONFIG.default_parent_type
        if self.parent_id is None:
            self.parent_id = self.memcell.event_id


async def process_memory_extraction(
    memcell: MemCell,
    request: MemorizeRequest,
    memory_manager: MemoryManager,
    current_time: datetime,
) -> int:
    """
    Main memory extraction process

    Starting from MemCell, extract all memory types including Episode, Foresight, EventLog, etc.

    Returns:
        int: Total number of memories extracted
    """
    # Get metrics labels
    space_id = get_space_id_for_metrics()
    raw_data_type = memcell.type.value if memcell.type else 'unknown'
    
    # 1. Initialize state
    init_start = time.perf_counter()
    state = await _init_extraction_state(memcell, request, current_time)
    record_extraction_stage(
        space_id=space_id,
        raw_data_type=raw_data_type,
        stage='init_state',
        duration_seconds=time.perf_counter() - init_start,
    )

    # 2. Parallel extract: Episode + (assistant scene) Foresight/EventLog
    foresight_memories, event_logs = [], []
    extract_start = time.perf_counter()
    
    # Wrapper functions to track individual stage durations
    async def _timed_extract_episodes():
        start = time.perf_counter()
        result = await _extract_episodes(state, memory_manager)
        record_extraction_stage(
            space_id=space_id,
            raw_data_type=raw_data_type,
            stage='extract_episodes',
            duration_seconds=time.perf_counter() - start,
        )
        return result
    
    async def _timed_extract_foresights():
        start = time.perf_counter()
        result = await _extract_foresights(state, memory_manager)
        record_extraction_stage(
            space_id=space_id,
            raw_data_type=raw_data_type,
            stage='extract_foresights',
            duration_seconds=time.perf_counter() - start,
        )
        return result
    
    async def _timed_extract_event_logs():
        start = time.perf_counter()
        result = await _extract_event_logs(state, memory_manager)
        record_extraction_stage(
            space_id=space_id,
            raw_data_type=raw_data_type,
            stage='extract_event_logs',
            duration_seconds=time.perf_counter() - start,
        )
        return result
    
    if state.is_assistant_scene:
        # Episode extraction modifies state in-place. Run it concurrently with
        # Foresight/EventLog, but cap each with a per-task timeout so a slow or
        # hanging LLM call never blocks episode persistence.
        _EXTRACTION_TIMEOUT = 120  # seconds per extraction task

        # ★ REALTIME_EVENT_LOG_ONLY mode: only extract event_log right now.
        #   Episodic memories are generated later by BatchEpisodeWorker which
        #   merges multiple MemCells into one rich context before calling the
        #   episode LLM, preserving full multi-turn quality.
        if os.getenv("REALTIME_EVENT_LOG_ONLY", "false").lower() == "true":
            try:
                event_logs = await asyncio.wait_for(
                    _timed_extract_event_logs(), timeout=_EXTRACTION_TIMEOUT
                )
                event_logs = event_logs or []
            except (asyncio.TimeoutError, Exception) as e:
                logger.warning("[Realtime] EventLog extraction failed: %s", e)
                event_logs = []
            record_extraction_stage(
                space_id=space_id,
                raw_data_type=raw_data_type,
                stage='extract_parallel',
                duration_seconds=time.perf_counter() - extract_start,
            )
            if event_logs:
                record_memory_extracted(
                    space_id=space_id,
                    raw_data_type=raw_data_type,
                    memory_type='event_log',
                    count=len(event_logs),
                )
            await _update_memcell_and_cluster(state)
            if if_memorize(memcell):
                rt_count = await _save_event_logs_without_episode(state, event_logs)
                await update_status_after_memcell(
                    state.request, state.memcell, state.current_time,
                    state.request.raw_data_type,
                )
                return rt_count
            return 0

        async def _safe_extract_foresights():
            try:
                return await asyncio.wait_for(_timed_extract_foresights(), timeout=_EXTRACTION_TIMEOUT)
            except asyncio.TimeoutError:
                logger.warning(
                    "[Extraction] Foresight extraction timed out after %ss, episode will still be saved",
                    _EXTRACTION_TIMEOUT,
                )
                return []
            except Exception as e:
                logger.error("[Extraction] Foresight extraction failed: %s", e)
                return []

        async def _safe_extract_event_logs():
            try:
                return await asyncio.wait_for(_timed_extract_event_logs(), timeout=_EXTRACTION_TIMEOUT)
            except asyncio.TimeoutError:
                logger.warning(
                    "[Extraction] EventLog extraction timed out after %ss, episode will still be saved",
                    _EXTRACTION_TIMEOUT,
                )
                return []
            except Exception as e:
                logger.error("[Extraction] EventLog extraction failed: %s", e)
                return []

        _, foresight_memories, event_logs = await asyncio.gather(
            _timed_extract_episodes(),
            _safe_extract_foresights(),
            _safe_extract_event_logs(),
        )
        foresight_memories = foresight_memories or []
        event_logs = event_logs or []
    else:
        await _timed_extract_episodes()
    record_extraction_stage(
        space_id=space_id,
        raw_data_type=raw_data_type,
        stage='extract_parallel',
        duration_seconds=time.perf_counter() - extract_start,
    )

    # Record extracted counts
    episodes_count = len(state.group_episode_memories) + len(state.episode_memories)
    if episodes_count > 0:
        record_memory_extracted(
            space_id=space_id,
            raw_data_type=raw_data_type,
            memory_type='episode',
            count=episodes_count,
        )
    if foresight_memories:
        record_memory_extracted(
            space_id=space_id,
            raw_data_type=raw_data_type,
            memory_type='foresight',
            count=len(foresight_memories),
        )
    if event_logs:
        record_memory_extracted(
            space_id=space_id,
            raw_data_type=raw_data_type,
            memory_type='event_log',
            count=len(event_logs),
        )

    # 3. Update MemCell and trigger clustering
    cluster_start = time.perf_counter()
    await _update_memcell_and_cluster(state)
    record_extraction_stage(
        space_id=space_id,
        raw_data_type=raw_data_type,
        stage='update_memcell_cluster',
        duration_seconds=time.perf_counter() - cluster_start,
    )

    # 4. Save memories
    memories_count = 0
    if if_memorize(memcell):
        save_start = time.perf_counter()
        memories_count = await _process_memories(state, foresight_memories, event_logs)
        record_extraction_stage(
            space_id=space_id,
            raw_data_type=raw_data_type,
            stage='process_memories',
            duration_seconds=time.perf_counter() - save_start,
        )

    return memories_count


async def _init_extraction_state(
    memcell: MemCell, request: MemorizeRequest, current_time: datetime
) -> ExtractionState:
    """Initialize extraction state"""
    conversation_meta_repo = get_bean_by_type(ConversationMetaRawRepository)
    conversation_meta = await conversation_meta_repo.get_by_group_id(request.group_id)
    scene = (
        conversation_meta.scene
        if conversation_meta and conversation_meta.scene
        else "assistant"
    )
    is_assistant_scene = scene.lower() == ScenarioType.ASSISTANT
    participants = list(set(memcell.participants)) if memcell.participants else []

    return ExtractionState(
        memcell=memcell,
        request=request,
        current_time=current_time,
        scene=scene,
        is_assistant_scene=is_assistant_scene,
        participants=participants,
    )


async def _extract_episodes(state: ExtractionState, memory_manager: MemoryManager):
    """Extract group and personal Episodes"""
    if state.is_assistant_scene:
        logger.info("[MemCell Processing] assistant scene, only extract group Episode")
        tasks = [_create_episode_task(state, memory_manager, None)]
    else:
        logger.info(
            f"[MemCell Processing] non-assistant scene, extract group + {len(state.participants)} personal Episodes"
        )
        tasks = [_create_episode_task(state, memory_manager, None)]
        tasks.extend(
            [
                _create_episode_task(state, memory_manager, uid)
                for uid in state.participants
            ]
        )

    results = await asyncio.gather(*tasks, return_exceptions=True)
    _process_episode_results(state, results)


def _create_episode_task(
    state: ExtractionState, memory_manager: MemoryManager, user_id: Optional[str]
):
    """Create Episode extraction task"""
    return memory_manager.extract_memory(
        memcell=state.memcell,
        memory_type=MemoryType.EPISODIC_MEMORY,
        user_id=user_id,
        group_id=state.request.group_id,
        group_name=state.request.group_name,
    )


def _process_episode_results(state: ExtractionState, results: List[Any]):
    """Process Episode extraction results"""
    # Group Episode
    group_episode = results[0] if results else None
    if isinstance(group_episode, Exception):
        logger.error(
            f"[MemCell Processing] ❌ Group Episode exception: {group_episode}"
        )
        group_episode = None
    elif group_episode:
        group_episode.ori_event_id_list = [state.memcell.event_id]
        group_episode.memcell_event_id_list = [state.memcell.event_id]
        state.group_episode_memories.append(group_episode)
        state.group_episode = group_episode
        state.memcell.episode = group_episode.episode
        state.memcell.subject = group_episode.subject
        logger.info("[MemCell Processing] ✅ Group Episode extracted successfully")

    # Personal Episodes
    if not state.is_assistant_scene:
        for user_id, result in zip(state.participants, results[1:]):
            if isinstance(result, Exception):
                logger.error(
                    f"[MemCell Processing] ❌ Personal Episode exception: user_id={user_id}"
                )
                continue
            if result:
                result.ori_event_id_list = [state.memcell.event_id]
                result.memcell_event_id_list = [state.memcell.event_id]
                state.episode_memories.append(result)
                logger.info(
                    f"[MemCell Processing] ✅ Personal Episode successful: user_id={user_id}"
                )


async def _update_memcell_and_cluster(state: ExtractionState):
    """Update MemCell's episode field and trigger clustering"""
    if not state.request.group_id or not state.group_episode:
        return

    # Update MemCell
    try:
        memcell_repo = get_bean_by_type(MemCellRawRepository)
        await memcell_repo.update_by_event_id(
            event_id=state.memcell.event_id,
            update_data={
                "episode": state.group_episode.episode,
                "subject": state.group_episode.subject,
            },
        )
        logger.info(
            f"[MemCell Processing] ✅ Updated MemCell episode: {state.memcell.event_id}"
        )
    except Exception as e:
        logger.error(f"[MemCell Processing] ❌ Failed to update MemCell: {e}")

    # Trigger clustering
    try:
        memcell_for_clustering = MemCell(
            event_id=state.memcell.event_id,
            user_id_list=state.memcell.user_id_list,
            original_data=state.memcell.original_data,
            timestamp=state.memcell.timestamp,
            summary=state.memcell.summary,
            group_id=state.memcell.group_id,
            group_name=state.memcell.group_name,
            participants=state.memcell.participants,
            type=state.memcell.type,
            episode=state.group_episode.episode,
        )
        await _trigger_clustering(
            state.request.group_id, memcell_for_clustering, state.scene
        )
        logger.info(
            f"[MemCell Processing] ✅ Clustering completed (scene={state.scene})"
        )
    except Exception as e:
        logger.error(f"[MemCell Processing] ❌ Failed to trigger clustering: {e}")


async def _process_memories(
    state: ExtractionState,
    foresight_memories: List[Foresight],
    event_logs: List[EventLog],
) -> int:
    """Save Episodes and Foresight/EventLog

    Returns:
        int: Total number of memories saved
    """
    await load_core_memories(state.request, state.participants, state.current_time)

    episodic_source = state.group_episode_memories + state.episode_memories
    episodes_to_save = list(episodic_source)

    # assistant scene: copy group Episode to each user
    if state.is_assistant_scene and state.group_episode_memories:
        episodes_to_save.extend(_clone_episodes_for_users(state))

    episodes_count = 0
    foresight_count = 0
    eventlog_count = 0

    if episodes_to_save:
        await _save_episodes(state, episodes_to_save, episodic_source)
        episodes_count = len(episodes_to_save)

    # Save foresight/eventlog (assistant scene only, already extracted)
    if state.is_assistant_scene and (foresight_memories or event_logs):
        await _save_foresight_and_eventlog(state, foresight_memories, event_logs)
        foresight_count = len(foresight_memories)
        eventlog_count = len(event_logs)

    await update_status_after_memcell(
        state.request, state.memcell, state.current_time, state.request.raw_data_type
    )

    return episodes_count + foresight_count + eventlog_count


async def _extract_foresights(
    state: ExtractionState, memory_manager: MemoryManager
) -> List[Foresight]:
    """Extract Foresight from memcell (assistant scene only)."""
    result = await memory_manager.extract_memory(
        memcell=state.memcell, memory_type=MemoryType.FORESIGHT, user_id=None
    )
    if isinstance(result, Exception) or not result:
        return []
    for mem in result:
        mem.group_id = state.request.group_id
        mem.group_name = state.request.group_name
        mem.parent_type = state.parent_type
        mem.parent_id = state.parent_id
    return result


async def _extract_event_logs(
    state: ExtractionState, memory_manager: MemoryManager
) -> List[EventLog]:
    """Extract EventLog from memcell (assistant scene only)."""
    result = await memory_manager.extract_memory(
        memcell=state.memcell, memory_type=MemoryType.EVENT_LOG, user_id=None
    )
    if isinstance(result, Exception) or not result:
        return []
    result.group_id = state.request.group_id
    result.group_name = state.request.group_name
    result.parent_type = state.parent_type
    result.parent_id = state.parent_id
    return [result]


async def _save_event_logs_without_episode(
    state: ExtractionState,
    event_logs: List[EventLog],
) -> int:
    """Save event logs against the MemCell directly (no episode parent).

    Used in REALTIME_EVENT_LOG_ONLY mode where episode extraction is deferred to
    BatchEpisodeWorker.  Each EventLogRecord will have:
      parent_type = "memcell"
      parent_id   = <memcell.event_id>
    """
    if not event_logs:
        return 0

    class _MemCellParent:
        """Minimal duck-type shim to satisfy _convert_event_log_to_docs."""
        timestamp = state.memcell.timestamp
        participants = state.memcell.participants or []
        type = (
            state.memcell.type.value
            if state.memcell.type
            else RawDataType.CONVERSATION.value
        )

    pseudo_parent = _MemCellParent()

    # Override parent pointers to MemCell
    for el in event_logs:
        el.parent_type = "memcell"
        el.parent_id = str(state.memcell.event_id)

    base_docs = []
    for el in event_logs:
        base_docs.extend(_convert_event_log_to_docs(el, pseudo_parent, state.current_time))

    if not base_docs:
        return 0

    all_docs = list(base_docs)
    # assistant scene: copy to each non-robot user
    if state.is_assistant_scene:
        user_ids = [
            u for u in state.participants
            if "robot" not in u.lower() and "assistant" not in u.lower()
        ]
        all_docs.extend([
            doc.model_copy(update={"user_id": uid, "user_name": uid})
            for doc in base_docs
            for uid in user_ids
        ])

    payloads = [MemoryDocPayload(MemoryType.EVENT_LOG, doc) for doc in all_docs]
    saved_map = await save_memory_docs(payloads)
    count = len(saved_map.get(MemoryType.EVENT_LOG, []))
    logger.info(
        "[Realtime] Saved %d event_log facts (parent=memcell %s)",
        count,
        state.memcell.event_id,
    )
    return count


def _clone_episodes_for_users(state: ExtractionState) -> List[EpisodeMemory]:
    """Copy group Episode to each user"""
    from dataclasses import replace

    cloned = []
    group_ep = state.group_episode_memories[0]
    for user_id in state.participants:
        if "robot" in user_id.lower() or "assistant" in user_id.lower():
            continue
        cloned.append(replace(group_ep, user_id=user_id, user_name=user_id))
    logger.info(f"[MemCell Processing] Copied group Episode to {len(cloned)} users")
    return cloned


async def _save_episodes(
    state: ExtractionState,
    episodes_to_save: List[EpisodeMemory],
    episodic_source: List[EpisodeMemory],
):
    """Save Episodes to database"""
    for ep in episodes_to_save:
        if getattr(ep, "group_name", None) is None:
            ep.group_name = state.request.group_name
        if getattr(ep, "user_name", None) is None:
            ep.user_name = ep.user_id

    docs = [
        _convert_episode_memory_to_doc(ep, state.current_time)
        for ep in episodes_to_save
    ]
    payloads = [MemoryDocPayload(MemoryType.EPISODIC_MEMORY, doc) for doc in docs]
    saved_map = await save_memory_docs(payloads)
    saved_docs = saved_map.get(MemoryType.EPISODIC_MEMORY, [])

    for ep, saved_doc in zip(episodic_source, saved_docs):
        ep.id = str(saved_doc.id)
        state.parent_docs_map[str(saved_doc.id)] = saved_doc


async def _save_foresight_and_eventlog(
    state: ExtractionState,
    foresight_memories: List[Foresight],
    event_logs: List[EventLog],
):
    """Save Foresight and EventLog (after episode saved)"""
    # Get the saved doc of group episode as parent_doc
    parent_doc = None
    if state.group_episode_memories:
        ep_id = state.group_episode_memories[0].id
        if ep_id:
            parent_doc = state.parent_docs_map.get(ep_id)

    if not parent_doc:
        logger.warning(
            "[MemCell Processing] No parent_doc for foresight/eventlog, skip saving"
        )
        return

    foresight_docs = [
        _convert_foresight_to_doc(mem, parent_doc, state.current_time)
        for mem in foresight_memories
    ]

    event_log_docs = []
    for el in event_logs:
        event_log_docs.extend(
            _convert_event_log_to_docs(el, parent_doc, state.current_time)
        )

    # assistant scene: copy to each user
    if state.is_assistant_scene:
        user_ids = [
            u
            for u in state.participants
            if "robot" not in u.lower() and "assistant" not in u.lower()
        ]
        foresight_docs.extend(
            [
                doc.model_copy(update={"user_id": uid, "user_name": uid})
                for doc in foresight_docs
                for uid in user_ids
            ]
        )
        event_log_docs.extend(
            [
                doc.model_copy(update={"user_id": uid, "user_name": uid})
                for doc in event_log_docs
                for uid in user_ids
            ]
        )
        logger.info(
            f"[MemCell Processing] Copied Foresight/EventLog to {len(user_ids)} users"
        )

    payloads = []
    payloads.extend(
        MemoryDocPayload(MemoryType.FORESIGHT, doc) for doc in foresight_docs
    )
    payloads.extend(
        MemoryDocPayload(MemoryType.EVENT_LOG, doc) for doc in event_log_docs
    )
    if payloads:
        await save_memory_docs(payloads)


def extract_message_time(raw_data):
    """
    Extract message time from RawData object

    Args:
        raw_data: RawData object

    Returns:
        datetime: Message time, return None if extraction fails
    """
    # Prioritize timestamp field
    if hasattr(raw_data, 'timestamp') and raw_data.timestamp:
        try:
            return _normalize_datetime_for_storage(raw_data.timestamp)
        except Exception as e:
            logger.debug(f"Failed to parse timestamp from raw_data.timestamp: {e}")
            pass

    # Extract from extend field
    if (
        hasattr(raw_data, 'extend')
        and raw_data.extend
        and isinstance(raw_data.extend, dict)
    ):
        timestamp_val = raw_data.extend.get('timestamp')
        if timestamp_val:
            try:
                return _normalize_datetime_for_storage(timestamp_val)
            except Exception as e:
                logger.debug(f"Failed to parse timestamp from extend field: {e}")
                pass

    return None


from core.observation.tracing.decorators import trace_logger


@trace_logger(operation_name="mem_memorize preprocess_conv_request", log_level="info")
async def preprocess_conv_request(
    request: MemorizeRequest, current_time: datetime
) -> MemorizeRequest:
    """
    Simplified request preprocessing:
    1. Get last_memcell_time from status table to determine current memcell start
    2. Read historical messages from conversation_data_repo (only messages after last_memcell_time)
    3. Set historical messages as history_raw_data_list
    4. Set current new message as new_raw_data_list
    5. Boundary detection handled by subsequent logic (will clear or retain after detection)
    """

    logger.info(f"[preprocess] Start processing: group_id={request.group_id}")

    # Check if there is new data
    if not request.new_raw_data_list:
        logger.info("[preprocess] No new data, skip processing")
        return None

    # Use conversation_data_repo for read-then-store operation
    conversation_data_repo = get_bean_by_type(ConversationDataRepository)
    status_repo = get_bean_by_type(ConversationStatusRawRepository)

    try:
        # Extract message_ids from new_raw_data_list to exclude them
        new_message_ids = [r.data_id for r in request.new_raw_data_list if r.data_id]

        # Step 0: Get last_memcell_time to filter history (only get current memcell's messages)
        start_time = None
        status = await status_repo.get_by_group_id(request.group_id)
        if status and status.last_memcell_time:
            start_time = status.last_memcell_time
            logger.info(f"[preprocess] Using last_memcell_time as start_time: {start_time}")

        # Step 1: Get historical messages, excluding current request's messages
        # Only get messages after last_memcell_time (current memcell's accumulated messages)
        history_raw_data_list = await conversation_data_repo.get_conversation_data(
            group_id=request.group_id,
            start_time=start_time,
            end_time=None,
            limit=1000,
            exclude_message_ids=new_message_ids,
        )

        logger.info(
            f"[preprocess] Read {len(history_raw_data_list)} historical messages (excluded {len(new_message_ids)} new, start_time={start_time})"
        )

        # Update request
        request.history_raw_data_list = history_raw_data_list
        # new_raw_data_list remains unchanged (the newly passed messages)

        logger.info(
            f"[preprocess] Completed: {len(history_raw_data_list)} historical, {len(request.new_raw_data_list)} new messages"
        )

        return request

    except Exception as e:
        logger.error(f"[preprocess] Data read failed: {e}")
        traceback.print_exc()
        # Use original request if read fails
        return request


async def update_status_when_no_memcell(
    request: MemorizeRequest,
    status_result: StatusResult,
    current_time: datetime,
    data_type: RawDataType,
):
    if data_type == RawDataType.CONVERSATION:
        # Try to update status table
        try:
            status_repo = get_bean_by_type(ConversationStatusRawRepository)

            if status_result.should_wait:
                logger.info(
                    f"[mem_memorize] Determined as unable to decide boundary, continue waiting, no status update"
                )
                return
            else:
                logger.info(
                    f"[mem_memorize] Determined as non-boundary, continue accumulating messages, update status table"
                )
                # Get latest message timestamp
                latest_time = _convert_timestamp_to_time(current_time, current_time)
                if request.new_raw_data_list:
                    last_msg = request.new_raw_data_list[-1]
                    if hasattr(last_msg, 'content') and isinstance(
                        last_msg.content, dict
                    ):
                        latest_time = last_msg.content.get('timestamp', latest_time)
                    elif hasattr(last_msg, 'timestamp'):
                        latest_time = last_msg.timestamp

                if not latest_time:
                    latest_time = min(latest_time, current_time)

                # Use encapsulated function to update conversation continuation status
                await _update_status_for_continuing_conversation(
                    status_repo, request, latest_time, current_time
                )

        except Exception as e:
            logger.error(f"Failed to update status table: {e}")
    else:
        pass


async def update_status_after_memcell(
    request: MemorizeRequest,
    memcell: MemCell,
    current_time: datetime,
    data_type: RawDataType,
):
    if data_type == RawDataType.CONVERSATION:
        # Update last_memcell_time in status table to memcell's timestamp
        try:
            status_repo = get_bean_by_type(ConversationStatusRawRepository)

            # Get MemCell's timestamp
            memcell_time = None
            if memcell and hasattr(memcell, 'timestamp'):
                memcell_time = memcell.timestamp
            else:
                memcell_time = current_time

            # Use encapsulated function to update status after MemCell extraction
            await _update_status_after_memcell_extraction(
                status_repo, request, memcell_time, current_time
            )

            logger.info(
                f"[mem_memorize] Memory extraction completed, status table updated"
            )

        except Exception as e:
            logger.error(f"Final status table update failed: {e}")
    else:
        pass


async def save_personal_profile_memory(
    profile_memories: List[ProfileMemory], version: Optional[str] = None
):
    logger.info(
        f"[mem_memorize] Saving {len(profile_memories)} personal profile memories to database"
    )
    # Initialize Repository instance
    core_memory_repo = get_bean_by_type(CoreMemoryRawRepository)

    # Save personal profile memories to GroupUserProfileMemoryRawRepository
    for profile_mem in profile_memories:
        await _save_profile_memory_to_core(profile_mem, core_memory_repo, version)
        # Remove individual operation success log


async def save_memory_docs(
    doc_payloads: List[MemoryDocPayload], version: Optional[str] = None
) -> Dict[MemoryType, List[Any]]:
    """
    Generic Doc saving function, automatically saves and synchronizes by MemoryType enum
    """

    grouped_docs: Dict[MemoryType, List[Any]] = defaultdict(list)
    for payload in doc_payloads:
        if payload and payload.doc:
            grouped_docs[payload.memory_type].append(payload.doc)

    saved_result: Dict[MemoryType, List[Any]] = {}

    # Episodic
    episodic_docs = grouped_docs.get(MemoryType.EPISODIC_MEMORY, [])
    if episodic_docs:
        episodic_repo = get_bean_by_type(EpisodicMemoryRawRepository)
        episodic_es_repo = get_bean_by_type(EpisodicMemoryEsRepository)
        episodic_milvus_repo = get_bean_by_type(EpisodicMemoryMilvusRepository)
        saved_episodic: List[Any] = []

        for doc in episodic_docs:
            saved_doc = await episodic_repo.append_episodic_memory(doc)
            saved_episodic.append(saved_doc)

            es_doc = EpisodicMemoryConverter.from_mongo(saved_doc)
            await episodic_es_repo.create(es_doc)

            milvus_entity = EpisodicMemoryMilvusConverter.from_mongo(saved_doc)
            vector = (
                milvus_entity.get("vector") if isinstance(milvus_entity, dict) else None
            )
            if vector and len(vector) > 0:
                await episodic_milvus_repo.insert(milvus_entity, flush=False)
            else:
                logger.warning(
                    "[mem_memorize] Skipping write to Milvus: vector empty or missing, event_id=%s",
                    getattr(saved_doc, "event_id", None),
                )

        saved_result[MemoryType.EPISODIC_MEMORY] = saved_episodic

    # Foresight
    foresight_docs = grouped_docs.get(MemoryType.FORESIGHT, [])
    if foresight_docs:
        foresight_repo = get_bean_by_type(ForesightRecordRawRepository)
        saved_foresight = await foresight_repo.create_batch(foresight_docs)
        saved_result[MemoryType.FORESIGHT] = saved_foresight

        sync_service = get_bean_by_type(MemorySyncService)
        await sync_service.sync_batch_foresights(
            saved_foresight, sync_to_es=True, sync_to_milvus=True
        )

    # Event Log
    event_log_docs = grouped_docs.get(MemoryType.EVENT_LOG, [])
    if event_log_docs:
        event_log_repo = get_bean_by_type(EventLogRecordRawRepository)
        saved_event_logs = await event_log_repo.create_batch(event_log_docs)
        saved_result[MemoryType.EVENT_LOG] = saved_event_logs

        sync_service = get_bean_by_type(MemorySyncService)
        await sync_service.sync_batch_event_logs(
            saved_event_logs, sync_to_es=True, sync_to_milvus=True
        )

    # Profile
    profile_docs = grouped_docs.get(MemoryType.PROFILE, [])
    if profile_docs:
        group_user_profile_repo = get_bean_by_type(GroupUserProfileMemoryRawRepository)
        saved_profiles = []
        for profile_mem in profile_docs:
            try:
                await _save_profile_memory_to_group_user_profile_memory(
                    profile_mem, group_user_profile_repo, version
                )
                saved_profiles.append(profile_mem)
            except Exception as exc:
                logger.error(f"Failed to save Profile memory: {exc}")
        if saved_profiles:
            saved_result[MemoryType.PROFILE] = saved_profiles

    group_profile_docs = grouped_docs.get(MemoryType.GROUP_PROFILE, [])
    if group_profile_docs:
        group_profile_repo = get_bean_by_type(GroupProfileRawRepository)
        saved_group_profiles = []
        for mem in group_profile_docs:
            try:
                await _save_group_profile_memory(mem, group_profile_repo, version)
                saved_group_profiles.append(mem)
            except Exception as exc:
                logger.error(f"Failed to save Group Profile memory: {exc}")
        if saved_group_profiles:
            saved_result[MemoryType.GROUP_PROFILE] = saved_group_profiles

    return saved_result


async def load_core_memories(
    request: MemorizeRequest, participants: List[str], current_time: datetime
):
    logger.info(f"[mem_memorize] Reading user data: {participants}")
    # Initialize Repository instance
    core_memory_repo = get_bean_by_type(CoreMemoryRawRepository)

    # Read user CoreMemory data
    user_core_memories = {}
    for user_id in participants:
        try:
            core_memory = await core_memory_repo.get_by_user_id(user_id)
            if core_memory:
                user_core_memories[user_id] = core_memory
            # Remove individual user success/failure logs
        except Exception as e:
            logger.error(f"Failed to get user {user_id} CoreMemory: {e}")

    logger.info(f"[mem_memorize] Retrieved {len(user_core_memories)} users' CoreMemory")

    # Directly convert CoreMemory to list of ProfileMemory objects
    old_memory_list = []
    if user_core_memories:
        for user_id, core_memory in user_core_memories.items():
            if core_memory:
                # Directly create ProfileMemory object
                profile_memory = ProfileMemory(
                    # Memory base class required fields
                    memory_type=MemoryType.CORE,
                    user_id=user_id,
                    timestamp=to_iso_format(current_time),
                    ori_event_id_list=[],
                    # Memory base class optional fields
                    subject=f"{getattr(core_memory, 'user_name', user_id)}'s personal profile",
                    summary=f"User {user_id}'s basic information: {getattr(core_memory, 'position', 'unknown role')}",
                    group_id=request.group_id,
                    participants=[user_id],
                    type=RawDataType.CONVERSATION,
                    # ProfileMemory specific fields - directly use original dictionary format
                    hard_skills=getattr(core_memory, 'hard_skills', None),
                    soft_skills=getattr(core_memory, 'soft_skills', None),
                    output_reasoning=getattr(core_memory, 'output_reasoning', None),
                    motivation_system=getattr(core_memory, 'motivation_system', None),
                    fear_system=getattr(core_memory, 'fear_system', None),
                    value_system=getattr(core_memory, 'value_system', None),
                    humor_use=getattr(core_memory, 'humor_use', None),
                    colloquialism=getattr(core_memory, 'colloquialism', None),
                    projects_participated=_convert_projects_participated_list(
                        getattr(core_memory, 'projects_participated', None)
                    ),
                )
                old_memory_list.append(profile_memory)

        logger.info(
            f"[mem_memorize] Directly converted {len(old_memory_list)} CoreMemory to ProfileMemory"
        )
    else:
        logger.info(f"[mem_memorize] No user CoreMemory data, old_memory_list is empty")


async def memorize(request: MemorizeRequest) -> int:
    """
    Main memory extraction process (global queue version)

    Flow:
    1. Save request logs and confirm them (sync_status: -1 -> 0)
    2. Get historical conversation data
    3. Extract MemCell (boundary detection)
    4. Save MemCell to database
    5. Process memory extraction

    Returns:
        int: Number of memories extracted (0 if no boundary detected or extraction failed)
    """
    logger.info(f"[mem_memorize] request.current_time: {request.current_time}")

    # Get current time
    if request.current_time:
        current_time = request.current_time
    else:
        current_time = get_now_with_timezone() + timedelta(seconds=1)
    logger.info(f"[mem_memorize] Current time: {current_time}")

    memory_manager = MemoryManager()
    conversation_data_repo = get_bean_by_type(ConversationDataRepository)

    # Note: Request logs are saved in controller layer for better timing control
    # (sync_status=-1, will be confirmed later based on boundary detection result)

    # ===== Preprocess and get historical data =====
    if request.raw_data_type == RawDataType.CONVERSATION:
        request = await preprocess_conv_request(request, current_time)
        if request == None:
            logger.warning(f"[mem_memorize] preprocess_conv_request returned None")
            return 0

    # Boundary detection
    # Get metrics labels
    space_id = get_space_id_for_metrics()
    raw_data_type = request.raw_data_type.value if request.raw_data_type else 'unknown'
    
    logger.info("=" * 80)
    logger.info(f"[Boundary Detection] Start detection: group_id={request.group_id}")
    logger.info(
        f"[Boundary Detection] Temporary stored historical messages: {len(request.history_raw_data_list)} messages"
    )
    logger.info(
        f"[Boundary Detection] New messages: {len(request.new_raw_data_list)} messages"
    )
    logger.info("=" * 80)

    memcell_start = time.perf_counter()
    memcell_result = await memory_manager.extract_memcell(
        request.history_raw_data_list,
        request.new_raw_data_list,
        request.raw_data_type,
        request.group_id,
        request.group_name,
        request.user_id_list,
    )
    record_extraction_stage(
        space_id=space_id,
        raw_data_type=raw_data_type,
        stage='extract_memcell',
        duration_seconds=time.perf_counter() - memcell_start,
    )
    logger.debug(f"[mem_memorize] Extracting MemCell took: {time.perf_counter() - memcell_start} seconds")

    if memcell_result == None:
        logger.warning(f"[mem_memorize] Skipped extracting MemCell")
        return 0

    memcell, status_result = memcell_result

    # Check boundary detection result
    logger.info("=" * 80)
    logger.info(f"[Boundary Detection Result] memcell is None: {memcell is None}")
    if memcell is None:
        logger.info(
            f"[Boundary Detection Result] Judgment: {'Need to wait for more messages' if status_result.should_wait else 'Non-boundary, continue accumulating'}"
        )
    else:
        logger.info(
            f"[Boundary Detection Result] Judgment: It's a boundary! event_id={memcell.event_id}"
        )
    logger.info("=" * 80)

    if memcell == None:
        # No boundary detected, confirm current messages to accumulation (sync_status: -1 -> 0)
        await conversation_data_repo.save_conversation_data(
            request.new_raw_data_list, request.group_id
        )
        logger.info(
            f"[mem_memorize] No boundary, confirmed {len(request.new_raw_data_list)} messages to accumulation"
        )
        await update_status_when_no_memcell(
            request, status_result, current_time, request.raw_data_type
        )
        logger.warning(f"[mem_memorize] No boundary detected, returning")
        return 0
    else:
        logger.info(f"[mem_memorize] Successfully extracted MemCell")
        # Judged as boundary, mark all accumulated data as used (restart accumulation)
        # Exclude current request's new messages so they can start the next accumulation
        try:
            new_message_ids = [
                r.data_id for r in request.new_raw_data_list if r.data_id
            ]
            delete_success = await conversation_data_repo.delete_conversation_data(
                request.group_id, exclude_message_ids=new_message_ids
            )
            if delete_success:
                logger.info(
                    f"[mem_memorize] Judged as boundary, history marked as used (excluded {len(new_message_ids)} new): group_id={request.group_id}"
                )
            else:
                logger.warning(
                    f"[mem_memorize] Failed to clear conversation history: group_id={request.group_id}"
                )
            # Confirm new messages to start the next accumulation cycle
            await conversation_data_repo.save_conversation_data(
                request.new_raw_data_list, request.group_id
            )
        except Exception as e:
            logger.error(
                f"[mem_memorize] Exception while marking conversation history: {e}"
            )
            traceback.print_exc()
    # TODO: Read status table, read accumulated MemCell data table, determine whether to perform memorize calculation

    # Save MemCell to table
    memcell = await _save_memcell_to_database(memcell, current_time)
    logger.info(f"[mem_memorize] Successfully saved MemCell: {memcell.event_id}")

    # Get current request_id

    app_info = get_current_app_info()
    request_id = app_info.get('request_id')

    # Directly execute memory extraction (blocking/asynchronous logic controlled by middleware layer request_process)
    try:
        memories_count = await process_memory_extraction(
            memcell, request, memory_manager, current_time
        )
        logger.info(
            f"[mem_memorize] ✅ Memory extraction completed, count={memories_count}, request_id={request_id}"
        )
        return memories_count
    except Exception as e:
        logger.error(f"[mem_memorize] ❌ Memory extraction failed: {e}")
        traceback.print_exc()
        return 0

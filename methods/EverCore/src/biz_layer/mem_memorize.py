from dataclasses import dataclass
import random
import time
import json
import traceback

from core.observation.stage_timer import timed, timed_parallel
from api_specs.memory_types import ScenarioType
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
    AgentCase,
)
from api_specs.memory_types import AtomicFact, get_text_from_content_items
from biz_layer.memorize_config import DEFAULT_MEMORIZE_CONFIG
from core.di import get_bean_by_type
from infra_layer.adapters.out.persistence.repository.episodic_memory_raw_repository import (
    EpisodicMemoryRawRepository,
)
from infra_layer.adapters.out.persistence.repository.foresight_record_raw_repository import (
    ForesightRecordRawRepository,
)
from infra_layer.adapters.out.persistence.repository.atomic_fact_record_raw_repository import (
    AtomicFactRecordRawRepository,
)
from infra_layer.adapters.out.persistence.repository.conversation_status_raw_repository import (
    ConversationStatusRawRepository,
)
from service.settings_service import SettingsService
from infra_layer.adapters.out.persistence.repository.memcell_raw_repository import (
    MemCellRawRepository,
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

logger = get_logger(__name__)


async def _load_llm_custom_setting() -> Optional[Dict[str, Any]]:
    """Load and normalize llm_custom_setting from global settings."""
    settings_service = get_bean_by_type(SettingsService)
    return await settings_service.get_llm_custom_setting()


@dataclass
class MemoryDocPayload:
    memory_type: MemoryType
    doc: Any


from biz_layer.memorize_config import (
    MemorizeConfig,
    DEFAULT_MEMORIZE_CONFIG,
)


def _is_agent_case_quality_sufficient(
    agent_case: AgentCase, config: MemorizeConfig
) -> bool:
    """Check if the AgentCase quality score meets the minimum threshold for skill extraction."""
    score = agent_case.quality_score
    if score is None or score < config.skill_min_quality_score:
        logger.info(
            f"[AgentSkill] Skipping skill extraction: quality_score={score} "
            f"< threshold={config.skill_min_quality_score}"
        )
        return False
    return True



async def _trigger_clustering(
    group_id: str,
    memcell: MemCell,
    scene: Optional[str] = None,
    config: MemorizeConfig = DEFAULT_MEMORIZE_CONFIG,
    episode_text: Optional[str] = None,
    agent_case: Optional[AgentCase] = None,
) -> None:
    """Trigger MemCell clustering

    Args:
        group_id: Group ID
        memcell: The MemCell just saved
        scene: Conversation scene (used to determine Profile extraction strategy)
            - "solo": 1 user + N agents scenario
            - "team": multi-user + agents scenario
        episode_text: Episode text extracted from this MemCell (used for clustering similarity)
        agent_case: Extracted AgentCase (if agent conversation), used for skill extraction
    """
    logger.info(
        f"[Clustering] Start triggering clustering: group_id={group_id}, event_id={memcell.event_id}, scene={scene}"
    )

    try:
        from memory_layer.cluster_manager import (
            ClusterManager,
            ClusterManagerConfig,
            MemSceneState,
        )
        from infra_layer.adapters.out.persistence.repository.mem_scene_raw_repository import (
            MemSceneRawRepository,
        )
        from core.di import get_bean_by_type

        logger.info(f"[Clustering] Retrieving MemSceneRawRepository...")
        # Get MongoDB storage
        cluster_storage = get_bean_by_type(MemSceneRawRepository)
        logger.info(
            f"[Clustering] MemSceneRawRepository retrieved successfully: {type(cluster_storage)}"
        )

        # Create ClusterManager (pure computation component)
        has_case = agent_case is not None
        cluster_config = ClusterManagerConfig(
            similarity_threshold=config.cluster_similarity_threshold,
            max_time_gap_days=config.cluster_max_time_gap_days,
        )

        # Build LLM provider and context fetcher for agent clustering
        llm_provider = None
        context_fetcher = None
        if has_case:
            from memory_layer.llm.llm_provider import build_default_provider
            from infra_layer.adapters.out.persistence.repository.agent_case_raw_repository import (
                AgentCaseRawRepository,
            )

            llm_provider = build_default_provider()
            agent_case_repo = get_bean_by_type(AgentCaseRawRepository)
            context_fetcher = agent_case_repo.fetch_task_intents_by_event_ids

        cluster_manager = ClusterManager(
            config=cluster_config,
            llm_provider=llm_provider,
            context_fetcher=context_fetcher,
        )

        # Clustering text: task_intent for agent case, episode for normal
        clustering_text = (
            agent_case.task_intent if has_case and agent_case.task_intent
            else episode_text
        )
        logger.info(
            f"[Clustering] ClusterManager created (has_case={has_case})"
        )

        # Convert MemCell to dictionary format required for clustering
        memcell_dict = {
            "event_id": str(memcell.event_id),
            "episode": episode_text,
            "clustering_text": clustering_text,
            "timestamp": memcell.timestamp.timestamp() if memcell.timestamp else None,
            "participants": memcell.participants or [],
            "group_id": group_id,
        }

        logger.debug(
            f"[Clustering] Start clustering execution: {memcell_dict['event_id']}"
        )

        from core.lock.redis_distributed_lock import distributed_lock

        # ===== Phase 1 + 2: Clustering + Profile extraction =====
        # Lock: trigger_clustering:{group_id}
        #
        # Protected shared state (read-modify-write):
        #   - mem_scene_state: loaded from DB, mutated by cluster_memcell(), saved back.
        #     Concurrent writes without this lock would cause lost updates.
        #   - Profile extraction (Phase 2) reads mem_scene_state snapshot taken in Phase 1
        #     and reads/writes user profiles in DB.
        #
        # Released before Phase 3 so the next request's Phase 1+2 is not blocked
        # by a slow LLM skill-extraction call.
        #
        # Data flow out of this lock:
        #   - cluster_id: determined by Phase 1, used as key for Phase 3 lock.
        #   - agent_case: passed through from caller, not modified here.
        #   Both are safe to use after lock release because Phase 3 re-reads
        #   its own shared state (existing_skills) from DB inside its own lock.
        cluster_id = None
        lock_resource = f"trigger_clustering:{group_id}"
        async with distributed_lock(
            resource=lock_resource,
            timeout=config.clustering_lock_timeout,
            blocking_timeout=config.clustering_lock_blocking_timeout,
        ) as acquired:
            if not acquired:
                logger.error(
                    f"[Clustering] Failed to acquire lock for group {group_id}, "
                    f"skipping memcell {memcell.event_id}"
                )
                return

            # ===== Phase 1: Clustering =====
            state_dict = await cluster_storage.load_mem_scene(group_id)
            mem_scene_state = (
                MemSceneState.from_dict(state_dict) if state_dict else MemSceneState()
            )
            logger.info(
                f"[Clustering] Loaded clustering state: {len(mem_scene_state.event_ids)} clustered events"
            )

            cluster_id, mem_scene_state = await cluster_manager.cluster_memcell(
                memcell_dict, mem_scene_state, has_case=has_case
            )

            await cluster_storage.save_mem_scene(group_id, mem_scene_state.to_dict())
            logger.info(f"[Clustering] Clustering state saved")

            if cluster_id:
                logger.debug(
                    f"[Clustering] ✅ MemCell {memcell.event_id} -> Cluster {cluster_id} (group: {group_id})"
                )
            else:
                logger.warning(
                    f"[Clustering] ⚠️ MemCell {memcell.event_id} clustering returned None (group: {group_id})"
                )

            # ===== Phase 2: Profile extraction (with interval-based throttling) =====
            if cluster_id and not config.skip_profile_extraction:
                total_memcell_count = sum(mem_scene_state.cluster_counts.values())
                should_extract = (
                    config.profile_extraction_interval <= 1
                    or total_memcell_count % config.profile_extraction_interval == 0
                )

                if should_extract:
                    # --- Group-level cluster selection (Layer 1 of 2) ---
                    # Profile extraction uses a two-layer filtering strategy:
                    #
                    # Layer 1 (here): Select which clusters to fetch from DB.
                    #   Uses min(last_updated_ts) across all users in the group as baseline.
                    #   This is intentionally broad — it covers the "slowest" user so no
                    #   cluster is missed for any user. Fetches ALL events from selected
                    #   clusters in one DB query.
                    #
                    # Layer 2 (manager.py, per-user loop): Filters fetched original_data per
                    #   user based on each user's own last_updated timestamp, so each user's
                    #   LLM prompt only contains data they haven't seen yet.
                    #   (Note: the code calls them "episodes" but the actual content is
                    #   memcell original_data — raw chat messages, not episode summaries.)
                    #
                    # For new groups with no profiles, defaults to current memcell timestamp
                    # to avoid selecting all historical clusters (cold-start protection).
                    from infra_layer.adapters.out.persistence.repository.user_profile_raw_repository import (
                        UserProfileRawRepository,
                    )
                    from core.di import get_bean_by_type

                    profile_repo = get_bean_by_type(UserProfileRawRepository)
                    existing_profiles = await profile_repo.get_all_by_group(group_id)

                    current_memcell_ts = memcell.timestamp.timestamp()

                    if existing_profiles:
                        timestamps = [
                            p.last_updated_ts
                            for p in existing_profiles
                            if p.last_updated_ts is not None
                        ]
                        last_profile_ts = (
                            min(timestamps) if timestamps else current_memcell_ts
                        )
                    else:
                        last_profile_ts = current_memcell_ts

                    target_cluster_ids = [
                        cid
                        for cid, ts in mem_scene_state.cluster_last_ts.items()
                        if ts is not None and ts > last_profile_ts
                    ]
                    if cluster_id not in target_cluster_ids:
                        target_cluster_ids.append(cluster_id)

                    logger.info(
                        f"[Profile] Timestamp-based selection: last_profile_ts={last_profile_ts}, "
                        f"target_clusters={target_cluster_ids}"
                    )

                    await _trigger_profile_extraction(
                        group_id=group_id,
                        cluster_ids=target_cluster_ids,
                        mem_scene_state=mem_scene_state,
                        memcell=memcell,
                        scene=scene,
                        config=config,
                    )
                else:
                    logger.debug(
                        f"[Profile] Skipping extraction: total_memcells={total_memcell_count}, "
                        f"interval={config.profile_extraction_interval}"
                    )

        # ===== Phase 3: Agent skill extraction =====
        # Lock: trigger_agent_skill:{group_id}:{cluster_id}
        #
        # Separate lock so Phase 1+2 of the next request is not blocked by this
        # slow LLM call.
        #
        # Data dependencies (all safe after Lock 1 release):
        #   - cluster_id: immutable identifier, determined in Phase 1.
        #   - agent_case: this request's own extraction result, not shared state.
        #   - existing_skills: re-read from DB inside _trigger_agent_skill_extraction,
        #     so it always reflects the latest state (including writes by prior requests).
        #
        # IMPORTANT for future maintainers:
        #   This function does NOT read memcells or agent_cases from DB. It only uses
        #   the passed-in agent_case (current request) + existing_skills (from DB).
        #   If you add logic that reads cluster memcells from DB here, you must
        #   consider that new memcells may have been added between Lock 1 release
        #   and Lock 2 acquisition.
        if cluster_id and agent_case and _is_agent_case_quality_sufficient(agent_case, config):
            skill_lock_resource = f"trigger_agent_skill:{group_id}:{cluster_id}"
            async with distributed_lock(
                resource=skill_lock_resource,
                timeout=config.skill_extraction_lock_timeout,
                blocking_timeout=config.skill_extraction_lock_blocking_timeout,
            ) as skill_acquired:
                if not skill_acquired:
                    logger.error(
                        f"[AgentSkill] Failed to acquire lock for group {group_id}, "
                        f"cluster {cluster_id}, skipping memcell {memcell.event_id}"
                    )
                    return
                await _trigger_agent_skill_extraction(
                    group_id=group_id,
                    cluster_id=cluster_id,
                    memcell=memcell,
                    agent_case=agent_case,
                    config=config,
                )

    except Exception as e:
        logger.error(
            f"[Clustering] ❌ Triggering clustering failed: {e}", exc_info=True
        )
        raise


async def _trigger_profile_extraction(
    group_id: str,
    cluster_ids: List[str],
    mem_scene_state,  # MemSceneState
    memcell: MemCell,
    scene: Optional[str] = None,
    config: MemorizeConfig = DEFAULT_MEMORIZE_CONFIG,
) -> None:
    """Trigger Profile extraction for one or more clusters.

    Args:
        group_id: Group ID
        cluster_ids: Cluster IDs to extract profiles from
        mem_scene_state: Current mem scene state
        memcell: The MemCell currently being processed (appended as new_memcell)
        scene: Conversation scene
        config: Memory extraction configuration
    """
    try:
        from memory_layer.profile_manager import ProfileManager, ProfileManagerConfig
        from infra_layer.adapters.out.persistence.repository.user_profile_raw_repository import (
            UserProfileRawRepository,
        )
        from memory_layer.llm.llm_provider import build_default_provider
        from core.di import get_bean_by_type
        import os

        total_memcell_count = sum(
            mem_scene_state.cluster_counts.get(cid, 0) for cid in cluster_ids
        )
        if total_memcell_count < config.profile_min_memcells:
            logger.debug(
                f"[Profile] Clusters {cluster_ids} have only {total_memcell_count} memcells "
                f"(requires {config.profile_min_memcells}), skipping extraction"
            )
            return

        logger.info(
            f"[Profile] Start extracting Profile: clusters={cluster_ids}, memcells={total_memcell_count}"
        )

        # Get Profile storage
        profile_repo = get_bean_by_type(UserProfileRawRepository)
        memcell_repo = get_bean_by_type(MemCellRawRepository)

        # Create LLM Provider
        llm_provider = build_default_provider()

        # Determine scenario (for metadata only, extraction logic is unified)
        profile_scenario = ScenarioType(scene.lower()) if scene else ScenarioType.TEAM

        # Create ProfileManager (pure computation component)
        profile_config = ProfileManagerConfig(
            min_confidence=config.profile_min_confidence,
            enable_versioning=config.profile_enable_versioning,
            auto_extract=True,
        )
        profile_manager = ProfileManager(
            llm_provider=llm_provider, config=profile_config, group_id=group_id
        )

        # ===== Fetch memcells from all target clusters =====
        current_event_id = str(memcell.event_id) if memcell.event_id else None
        target_cluster_set = set(cluster_ids)
        target_event_ids = set()
        if mem_scene_state and hasattr(mem_scene_state, 'eventid_to_cluster'):
            for event_id, cid in mem_scene_state.eventid_to_cluster.items():
                if cid in target_cluster_set and event_id != current_event_id:
                    target_event_ids.add(event_id)

        all_memcells = []
        if target_event_ids:
            try:
                fetched = await memcell_repo.get_by_event_ids(list(target_event_ids))
                all_memcells = list(fetched.values())
            except Exception as e:
                logger.warning(f"[Profile] Failed to fetch cluster memcells: {e}")

        # Append current memcell as the last one (new_memcell)
        all_memcells.append(memcell)

        # Merge participants from all memcells (deduplicated)
        all_participants: set = set()
        for mc in all_memcells:
            participants = (
                mc.participants
                if hasattr(mc, 'participants')
                else mc.get('participants', [])
            )
            all_participants.update(participants or [])
        user_id_list = list(all_participants)

        logger.info(
            f"[Profile] Context: clusters={len(cluster_ids)}, "
            f"memcells={len(all_memcells) - 1}, new=1, users={len(user_id_list)}"
        )

        # ===== Extract and save profiles =====
        # Caller (_trigger_clustering) holds trigger_clustering:{group_id} while calling
        # this function, so concurrent profile writes for the same group are serialized.

        # Load old profiles
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
        profile_scene = (
            ScenarioType.TEAM if scene == ScenarioType.TEAM.value else ScenarioType.SOLO
        )
        new_profiles = await profile_manager.extract_profiles(
            memcells=all_memcells,
            old_profiles=old_profiles,
            user_id_list=user_id_list,
            group_id=group_id,
            max_items=config.profile_max_items,
            scene=profile_scene,
        )

        # Save profiles
        for profile in new_profiles:
            try:
                memcell_ts = memcell.timestamp.timestamp() if memcell.timestamp else 0.0
                user_id = profile.user_id
                profile_data = profile.to_dict()
                metadata = {
                    "group_id": group_id,
                    "scenario": profile_scenario.value,
                    "memcell_count": total_memcell_count,
                    "total_items": profile.total_items(),
                    "last_updated_ts": memcell_ts,
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

        # Advance last_updated_ts even on failure to prevent repeated re-selection
        # of the same clusters. The data is "skipped" — acceptable tradeoff vs.
        # getting stuck in a loop retrying the same failing extraction.
        try:
            memcell_ts = memcell.timestamp.timestamp() if memcell.timestamp else 0.0
            for uid in user_id_list:
                existing = await profile_repo.get_by_user_and_group(uid, group_id)
                profile_data = (
                    existing.profile_data
                    if existing
                    else {"explicit_info": [], "implicit_traits": []}
                )
                await profile_repo.upsert(
                    user_id=uid,
                    group_id=group_id,
                    profile_data=profile_data,
                    metadata={"last_updated_ts": memcell_ts},
                    trigger_index=False,
                )
            logger.info(
                f"[Profile] Advanced last_updated_ts to {memcell_ts} for {len(user_id_list)} users despite failure"
            )
        except Exception as ts_err:
            logger.warning(
                f"[Profile] Failed to advance last_updated_ts on failure: {ts_err}"
            )


async def _trigger_agent_skill_extraction(
    group_id: str,
    cluster_id: str,
    memcell: MemCell,
    agent_case: AgentCase,
    config: MemorizeConfig = DEFAULT_MEMORIZE_CONFIG,
) -> None:
    """Trigger incremental AgentSkill extraction for a MemScene cluster.

    Args:
        group_id: Group ID
        cluster_id: The cluster (MemScene) to extract skills for
        memcell: The MemCell currently being processed (for user_id and event_id)
        agent_case: The extracted AgentCase BO
        config: Memory extraction configuration
    """
    try:
        from infra_layer.adapters.out.persistence.repository.agent_skill_raw_repository import (
            AgentSkillRawRepository,
        )
        from memory_layer.memory_extractor.agent_skill_extractor import (
            AgentSkillExtractor,
        )
        from memory_layer.llm.llm_provider import build_default_provider
        from infra_layer.adapters.out.search.milvus.converter.agent_skill_milvus_converter import (
            AgentSkillMilvusConverter,
        )
        from infra_layer.adapters.out.search.repository.agent_skill_milvus_repository import (
            AgentSkillMilvusRepository,
        )
        from infra_layer.adapters.out.search.elasticsearch.converter.agent_skill_converter import (
            AgentSkillConverter,
        )
        from infra_layer.adapters.out.search.repository.agent_skill_es_repository import (
            AgentSkillEsRepository,
        )

        # Caller (_trigger_clustering) acquires trigger_agent_skill:{group_id}:{cluster_id}
        # before calling this function, so concurrent skill writes for the same cluster are
        # serialized while different clusters within the same group can run in parallel.
        #
        # Concurrency safety of data used in this function:
        #   - existing_skills: read from DB below (inside the caller's lock), always fresh.
        #   - agent_case: passed in from the current request, not shared with other requests.
        #   - memcell: only used to extract user_id, no shared-state concern.
        #   - extract_and_save() does NOT read memcells or agent_cases from DB.
        #     It only merges new_case_records (passed-in) with existing_skill_records (from DB).
        #     If future changes add DB reads of memcells/cases here, re-evaluate the lock
        #     boundary — the gap between Lock 1 release and Lock 2 acquisition means
        #     new memcells may have been clustered in between.

        # Fetch existing skills for incremental merging
        skill_repo = get_bean_by_type(AgentSkillRawRepository)
        existing_skills = await skill_repo.get_by_cluster_id(
            cluster_id, group_id=group_id, min_confidence=config.skill_retire_confidence
        )

        logger.info(
            f"[AgentSkill] Incremental extraction: cluster={cluster_id}, "
            f"new_experience=1, existing_skills={len(existing_skills)}"
        )

        # Resolve user_id from the memcell's original conversation data
        user_id = _extract_user_id_from_memcell(memcell)

        # Run incremental skill extraction
        llm_provider = build_default_provider()
        extractor = AgentSkillExtractor(
            llm_provider=llm_provider,
            maturity_threshold=config.skill_maturity_threshold,
            retire_confidence=config.skill_retire_confidence,
            skip_maturity_scoring=config.skip_skill_maturity_scoring,
        )
        extraction_result = await extractor.extract_and_save(
            cluster_id=cluster_id,
            group_id=group_id,
            new_case_records=[agent_case],
            existing_skill_records=existing_skills,
            skill_repo=skill_repo,
            user_id=user_id,
        )

        if extraction_result.deleted_ids:
            logger.info(
                f"[AgentSkill] Retired skills for cluster={cluster_id}: "
                f"ids={extraction_result.deleted_ids}"
            )
        logger.info(
            f"[AgentSkill] Extraction result for cluster={cluster_id}: "
            f"added={len(extraction_result.added_records)}, "
            f"updated={len(extraction_result.updated_records)}, "
            f"retired={len(extraction_result.deleted_ids)}"
        )

        # Records that need insert into search engines (added + updated)
        upsert_records = (
            extraction_result.added_records + extraction_result.updated_records
        )
        # IDs of updated records that need their old entry removed first
        updated_ids = [str(r.id) for r in extraction_result.updated_records]
        # IDs to remove from search engines (deleted + updated-old-entries)
        remove_ids = extraction_result.deleted_ids + updated_ids

        if upsert_records or remove_ids:
            # Milvus sync: delete stale entries -> insert new/updated
            try:
                agent_skill_milvus_repo = get_bean_by_type(AgentSkillMilvusRepository)
                for old_id in remove_ids:
                    await agent_skill_milvus_repo.delete_by_id(old_id)
                inserted_count = 0
                for record in upsert_records:
                    milvus_entity = AgentSkillMilvusConverter.from_mongo(record)
                    if milvus_entity.get("vector"):
                        await agent_skill_milvus_repo.insert(milvus_entity, flush=False)
                        inserted_count += 1
                    else:
                        logger.warning(
                            f"[AgentSkill] Milvus skip (no vector): record={record.id}"
                        )
                logger.info(
                    f"[AgentSkill] Milvus synced for cluster={cluster_id}: "
                    f"inserted={inserted_count}, removed={len(remove_ids)}"
                )
            except Exception as milvus_exc:
                logger.warning(
                    f"[AgentSkill] Milvus sync failed for cluster={cluster_id}: {milvus_exc}"
                )

            # ES sync: delete stale entries -> insert new/updated
            try:
                agent_skill_es_repo = get_bean_by_type(AgentSkillEsRepository)
                for old_id in remove_ids:
                    await agent_skill_es_repo.delete_by_id(old_id)
                for record in upsert_records:
                    es_doc = AgentSkillConverter.from_mongo(record)
                    await agent_skill_es_repo.create(es_doc)
                logger.info(
                    f"[AgentSkill] ES synced for cluster={cluster_id}: "
                    f"inserted={len(upsert_records)}, removed={len(remove_ids)}"
                )
            except Exception as es_exc:
                logger.warning(
                    f"[AgentSkill] ES sync failed for cluster={cluster_id}: {es_exc}"
                )

    except Exception as e:
        logger.error(
            f"[AgentSkill] Skill extraction failed for cluster={cluster_id}: {e}",
            exc_info=True,
        )


from biz_layer.mem_db_operations import (
    _convert_episode_memory_to_doc,
    _convert_foresight_to_doc,
    _convert_atomic_fact_to_docs,
    _convert_agent_case_to_doc,
    _extract_user_id_from_memcell,
    _save_memcell_to_database,
    _update_status_for_continuing_conversation,
    _update_status_after_memcell_extraction,
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
    is_solo_scene: bool
    participants: List[str]
    episode_parent_type: str = None
    foresight_parent_type: str = None
    atomic_fact_parent_type: str = None
    parent_id: str = None
    group_episode: Optional[EpisodeMemory] = None
    group_episode_memories: List[EpisodeMemory] = None
    episode_memories: List[EpisodeMemory] = None
    agent_case: Optional[AgentCase] = None
    parent_docs_map: Dict[str, Any] = None

    @property
    def episode_saved(self) -> bool:
        """True when at least one episode has been saved to MongoDB.

        parent_docs_map is populated by _save_episodes only after a successful write,
        so non-empty means parent_doc is available for downstream linking.
        """
        return bool(self.parent_docs_map)

    def __post_init__(self):
        self.group_episode_memories = []
        self.episode_memories = []
        self.parent_docs_map = {}
        # Set default parent info from memcell
        if self.episode_parent_type is None:
            self.episode_parent_type = (
                DEFAULT_MEMORIZE_CONFIG.default_episode_parent_type
            )
        if self.foresight_parent_type is None:
            self.foresight_parent_type = (
                DEFAULT_MEMORIZE_CONFIG.default_foresight_parent_type
            )
        if self.atomic_fact_parent_type is None:
            self.atomic_fact_parent_type = (
                DEFAULT_MEMORIZE_CONFIG.default_atomic_fact_parent_type
            )
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

    Starting from MemCell, extract all memory types including Episode, Foresight, AtomicFact, etc.

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

    # 2. Parallel extract: Episode + (agent) AgentCase
    # foresight/atomic_fact moved to background task _foresight_and_atomic_facts_with_metrics
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

    async def _timed_extract_agent_case():
        start = time.perf_counter()
        result = await _extract_agent_case(state, memory_manager)
        record_extraction_stage(
            space_id=space_id,
            raw_data_type=raw_data_type,
            stage='extract_agent_case',
            duration_seconds=time.perf_counter() - start,
        )
        return result

    is_agent_conversation = (
        state.memcell.type == RawDataType.AGENTCONVERSATION
        if state.memcell.type
        else False
    )

    with timed("extract_memories"):
        if is_agent_conversation:
            # agent_case must be extracted synchronously: clustering fire-and-forget depends on it
            with timed_parallel("parallel_extraction"):
                await asyncio.gather(
                    _timed_extract_episodes(), _timed_extract_agent_case()
                )
        else:
            # solo non-agent: foresight/atomic_fact moved to background task
            # team scene: foresight/atomic_fact never extracted here either
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
    if state.agent_case:
        record_memory_extracted(
            space_id=space_id,
            raw_data_type=raw_data_type,
            memory_type='agent_case',
            count=1,
        )

    # 3. Fire-and-forget clustering + profile extraction (no data dependency on step 4)
    async def _clustering_with_metrics():
        cluster_start = time.perf_counter()
        try:
            await _update_memcell_and_cluster(state)
        except Exception as e:
            logger.error(f"[MemCell Processing] ❌ Background clustering failed: {e}")
        finally:
            record_extraction_stage(
                space_id=space_id,
                raw_data_type=raw_data_type,
                stage='update_memcell_cluster',
                duration_seconds=time.perf_counter() - cluster_start,
            )

    asyncio.create_task(_clustering_with_metrics())

    # 4. Save memories
    memories_count = 0
    if if_memorize(memcell):
        save_start = time.perf_counter()
        memories_count = await _process_memories(state)
        record_extraction_stage(
            space_id=space_id,
            raw_data_type=raw_data_type,
            stage='process_memories',
            duration_seconds=time.perf_counter() - save_start,
        )
        # Fire-and-forget: extract and save foresight/atomic_fact in background.
        # Solo scenes only; episode_saved confirms parent_doc is available for linking.
        if state.is_solo_scene and state.episode_saved and not DEFAULT_MEMORIZE_CONFIG.skip_foresight_and_eventlog:
            asyncio.create_task(
                _foresight_and_atomic_facts_with_metrics(state, memory_manager)
            )

    return memories_count


async def _init_extraction_state(
    memcell: MemCell, request: MemorizeRequest, current_time: datetime
) -> ExtractionState:
    """Initialize extraction state"""
    scene = request.scene or ScenarioType.SOLO.value
    is_solo_scene = scene.lower() == ScenarioType.SOLO
    participants = list(set(memcell.participants)) if memcell.participants else []

    return ExtractionState(
        memcell=memcell,
        request=request,
        current_time=current_time,
        scene=scene,
        is_solo_scene=is_solo_scene,
        participants=participants,
    )


async def _extract_episodes(state: ExtractionState, memory_manager: MemoryManager):
    """Extract group and personal Episodes"""
    if state.is_solo_scene:
        logger.info("[MemCell Processing] solo scene, only extract group Episode")
        tasks = [_create_episode_task(state, memory_manager, None)]
    else:
        logger.info(
            f"[MemCell Processing] team scene, extract group + {len(state.participants)} personal Episodes"
        )
        tasks = [_create_episode_task(state, memory_manager, None)]
        tasks.extend(
            [
                _create_episode_task(state, memory_manager, uid)
                for uid in state.participants
            ]
        )

    results = await asyncio.gather(*tasks, return_exceptions=True)
    from common_utils.async_utils import reraise_critical_errors

    reraise_critical_errors(results)
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
        group_episode.parent_type = state.episode_parent_type
        group_episode.parent_id = state.parent_id
        state.group_episode_memories.append(group_episode)
        state.group_episode = group_episode
        logger.info("[MemCell Processing] ✅ Group Episode extracted successfully")

    # Personal Episodes
    if not state.is_solo_scene:
        for user_id, result in zip(state.participants, results[1:]):
            if isinstance(result, Exception):
                logger.error(
                    f"[MemCell Processing] ❌ Personal Episode exception: user_id={user_id}"
                )
                continue
            if result:
                result.parent_type = state.episode_parent_type
                result.parent_id = state.parent_id
                state.episode_memories.append(result)
                logger.info(
                    f"[MemCell Processing] ✅ Personal Episode successful: user_id={user_id}"
                )


async def _update_memcell_and_cluster(state: ExtractionState):
    """Trigger clustering for the current MemCell"""
    if not state.request.group_id or not state.group_episode:
        return

    try:
        cluster_config = DEFAULT_MEMORIZE_CONFIG

        await _trigger_clustering(
            state.request.group_id,
            state.memcell,
            state.scene,
            config=cluster_config,
            episode_text=state.group_episode.episode,
            agent_case=state.agent_case,
        )
        logger.info(
            f"[MemCell Processing] ✅ Clustering completed (scene={state.scene})"
        )
    except Exception as e:
        logger.error(f"[MemCell Processing] ❌ Failed to trigger clustering: {e}")


async def _process_memories(state: ExtractionState) -> int:
    """Save Episodes and AgentCase.

    Foresight/AtomicFact are handled by the background task
    _foresight_and_atomic_facts_with_metrics (fire-and-forget after this returns).

    Returns:
        int: Total number of memories saved
    """
    # NOTE: load_core_memories disabled - CoreMemoryRawRepository not implemented
    # await load_core_memories(state.request, state.participants, state.current_time)

    episodic_source = state.group_episode_memories + state.episode_memories
    episodes_to_save = list(episodic_source)

    # solo scene: copy group Episode to each user
    if state.is_solo_scene and state.group_episode_memories:
        episodes_to_save.extend(_clone_episodes_for_users(state))

    episodes_count = 0
    agent_case_count = 0

    if episodes_to_save:
        with timed("persist_episodes"):
            await _save_episodes(state, episodes_to_save, episodic_source)
        episodes_count = len(episodes_to_save)

    # Save AgentCase (agent conversation only)
    if state.agent_case:
        with timed("persist_agent_case"):
            agent_case_count = await _save_agent_case(state)

    with timed("update_memcell_status"):
        await update_status_after_memcell(
            state.request,
            state.memcell,
            state.current_time,
            state.request.raw_data_type,
        )

    return episodes_count + agent_case_count


async def _extract_foresights(
    state: ExtractionState, memory_manager: MemoryManager
) -> List[Foresight]:
    """Extract Foresight from memcell (solo scene only)."""
    result = await memory_manager.extract_memory(
        memcell=state.memcell, memory_type=MemoryType.FORESIGHT, user_id=None
    )
    if isinstance(result, Exception) or not result:
        return []
    for mem in result:
        mem.group_id = state.request.group_id
        mem.parent_type = state.foresight_parent_type
        mem.parent_id = state.parent_id
    return result


def _should_skip_atomic_fact_for_agent(memcell: MemCell) -> bool:
    """Check if atomic fact extraction should be skipped for an agent conversation.

    Skip when there are tool calls and the cumulative assistant response
    is >= 1000 chars. Atomic facts from long tool-assisted conversations
    add little value but cost an extra LLM call; skipping speeds up the pipeline.
    """
    original_data = memcell.original_data or []

    # Unwrap from { "message": {...}, "parse_info": ... } format
    messages = []
    for item in original_data:
        if not isinstance(item, dict):
            continue
        msg = item.get("message", item)
        messages.append(msg)

    has_tool_calls = any(
        msg.get("tool_calls") or msg.get("role") == "tool" for msg in messages
    )
    if not has_tool_calls:
        return False

    total_length = 0
    for msg in messages:
        if msg.get("role") == "assistant" and not msg.get("tool_calls"):
            total_length += len(get_text_from_content_items(msg.get("content")))
    return total_length >= 1000


async def _extract_atomic_facts(
    state: ExtractionState, memory_manager: MemoryManager
) -> List[AtomicFact]:
    """Extract AtomicFact from memcell (solo scene only)."""
    result = await memory_manager.extract_memory(
        memcell=state.memcell, memory_type=MemoryType.ATOMIC_FACT, user_id=None
    )
    if isinstance(result, Exception) or not result:
        return []
    result.group_id = state.request.group_id
    result.parent_type = state.atomic_fact_parent_type
    result.parent_id = state.parent_id
    return [result]


async def _extract_agent_case(state: ExtractionState, memory_manager: MemoryManager):
    """Extract AgentCase from memcell (agent conversation only)."""
    result = await memory_manager.extract_memory(
        memcell=state.memcell,
        memory_type=MemoryType.AGENT_CASE,
        group_id=state.request.group_id,
    )
    if isinstance(result, Exception) or not result:
        return None
    state.agent_case = result
    return result


async def _save_agent_case(state: ExtractionState) -> int:
    """Save AgentCase to database.

    Returns:
        int: Number of agent experiences saved (0 or 1)
    """
    try:
        agent_case = state.agent_case
        doc = _convert_agent_case_to_doc(
            agent_case,
            state.memcell,
            state.current_time,
            session_id=state.request.session_id,
        )
        payloads = [MemoryDocPayload(MemoryType.AGENT_CASE, doc)]
        await save_memory_docs(payloads)
        logger.info(
            f"[MemCell Processing] AgentCase saved: intent='{agent_case.task_intent[:80]}'"
        )
        return 1
    except Exception as e:
        logger.error(f"[MemCell Processing] Failed to save AgentCase: {e}")
        return 0


def _clone_episodes_for_users(state: ExtractionState) -> List[EpisodeMemory]:
    """Copy group Episode to each user"""
    from dataclasses import replace

    cloned = []
    group_ep = state.group_episode_memories[0]
    for user_id in state.participants:
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
        if getattr(ep, "user_name", None) is None:
            ep.user_name = ep.user_id

    docs = [
        _convert_episode_memory_to_doc(
            ep, state.current_time, session_id=state.request.session_id
        )
        for ep in episodes_to_save
    ]
    payloads = [MemoryDocPayload(MemoryType.EPISODIC_MEMORY, doc) for doc in docs]
    saved_map = await save_memory_docs(payloads)
    saved_docs = saved_map.get(MemoryType.EPISODIC_MEMORY, [])

    for ep, saved_doc in zip(episodic_source, saved_docs):
        ep.id = str(saved_doc.id)
        state.parent_docs_map[str(saved_doc.id)] = saved_doc


async def _save_foresight_and_atomic_fact(
    state: ExtractionState,
    foresight_memories: List[Foresight],
    atomic_facts: List[AtomicFact],
):
    """Save Foresight and AtomicFact (after episode saved)"""
    # Get the saved doc of group episode as parent_doc
    parent_doc = None
    if state.group_episode_memories:
        ep_id = state.group_episode_memories[0].id
        if ep_id:
            parent_doc = state.parent_docs_map.get(ep_id)

    if not parent_doc:
        logger.warning(
            "[MemCell Processing] No parent_doc for foresight/atomic_fact, skip saving"
        )
        return

    session_id = state.request.session_id
    foresight_docs = [
        _convert_foresight_to_doc(
            mem, parent_doc, state.current_time, session_id=session_id
        )
        for mem in foresight_memories
    ]

    atomic_fact_docs = []
    for el in atomic_facts:
        atomic_fact_docs.extend(
            _convert_atomic_fact_to_docs(
                el, parent_doc, state.current_time, session_id=session_id
            )
        )

    # solo scene: copy to each user
    if state.is_solo_scene:
        user_ids = list(state.participants)
        foresight_docs.extend(
            [
                doc.model_copy(update={"user_id": uid, "user_name": uid})
                for doc in foresight_docs
                for uid in user_ids
            ]
        )
        atomic_fact_docs.extend(
            [
                doc.model_copy(update={"user_id": uid, "user_name": uid})
                for doc in atomic_fact_docs
                for uid in user_ids
            ]
        )
        logger.info(
            f"[MemCell Processing] Copied Foresight/AtomicFact to {len(user_ids)} users"
        )

    payloads = []
    payloads.extend(
        MemoryDocPayload(MemoryType.FORESIGHT, doc) for doc in foresight_docs
    )
    payloads.extend(
        MemoryDocPayload(MemoryType.ATOMIC_FACT, doc) for doc in atomic_fact_docs
    )
    if payloads:
        await save_memory_docs(payloads)


async def _foresight_and_atomic_facts_with_metrics(
    state: ExtractionState, memory_manager: MemoryManager
) -> None:
    """Background task: extract and save foresight and atomic_fact (solo scene only).

    Fired after _save_episodes completes so state.parent_docs_map is populated.
    Exceptions are caught and logged; never propagate to caller.

    space_id and raw_data_type are derived inside the function:
    - asyncio.create_task copies current contextvars.Context, so get_space_id_for_metrics() works.
    - raw_data_type is derived from state.memcell.type.
    """
    space_id = get_space_id_for_metrics()
    raw_data_type = state.memcell.type.value if state.memcell.type else 'unknown'
    start = time.perf_counter()
    try:
        should_skip = _should_skip_atomic_fact_for_agent(state.memcell)
        if should_skip:
            foresight_memories = await _extract_foresights(state, memory_manager)
            atomic_facts: List[AtomicFact] = []
        else:
            foresight_memories, atomic_facts = await asyncio.gather(
                _extract_foresights(state, memory_manager),
                _extract_atomic_facts(state, memory_manager),
            )
        if foresight_memories:
            record_memory_extracted(
                space_id=space_id,
                raw_data_type=raw_data_type,
                memory_type='foresight',
                count=len(foresight_memories),
            )
        if atomic_facts:
            record_memory_extracted(
                space_id=space_id,
                raw_data_type=raw_data_type,
                memory_type='atomic_fact',
                count=len(atomic_facts),
            )
        if foresight_memories or atomic_facts:
            await _save_foresight_and_atomic_fact(
                state, foresight_memories, atomic_facts
            )
    except Exception as e:
        logger.error(
            f"[ForesightAF] ❌ Background extraction/save failed: {e}", exc_info=True
        )
    finally:
        record_extraction_stage(
            space_id=space_id,
            raw_data_type=raw_data_type,
            stage='foresight_and_atomic_facts_bg',
            duration_seconds=time.perf_counter() - start,
        )


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
        if not request.flush:
            logger.info("[preprocess] No new data, skip processing")
            return None
        # flush=True with no new messages: load accumulated messages as history
        # so the extractor can flush them into a final MemCell
        logger.info(
            "[preprocess] Flush with no new data: loading accumulated messages as history"
        )
        conversation_data_repo = get_bean_by_type(ConversationDataRepository)
        status_repo = get_bean_by_type(ConversationStatusRawRepository)
        try:
            start_time = None
            status = await status_repo.get_by_group_id(
                request.group_id, session_id=request.session_id
            )
            if status and status.last_memcell_time:
                start_time = status.last_memcell_time
            accumulated = await conversation_data_repo.get_conversation_data(
                group_id=request.group_id,
                session_id=request.session_id,
                start_time=start_time,
                end_time=None,
                limit=1000,
                exclude_message_ids=[],
            )
            if not accumulated:
                logger.info(
                    "[preprocess] Flush: no accumulated messages, nothing to process"
                )
                return None
            request.history_raw_data_list = accumulated
            # new_raw_data_list stays empty; extractor handles flush+empty-new case
            logger.info(
                f"[preprocess] Flush: loaded {len(accumulated)} accumulated messages as history"
            )
            return request
        except Exception as e:
            logger.error(f"[preprocess] Flush data read failed: {e}")
            traceback.print_exc()
            return None

    # Use conversation_data_repo for read-then-store operation
    conversation_data_repo = get_bean_by_type(ConversationDataRepository)
    status_repo = get_bean_by_type(ConversationStatusRawRepository)

    try:
        # Extract message_ids from new_raw_data_list to exclude them
        new_message_ids = [r.data_id for r in request.new_raw_data_list if r.data_id]

        # Step 0: Get last_memcell_time to filter history (only get current memcell's messages)
        start_time = None
        status = await status_repo.get_by_group_id(
            request.group_id, session_id=request.session_id
        )
        if status and status.last_memcell_time:
            start_time = status.last_memcell_time
            logger.info(
                f"[preprocess] Using last_memcell_time as start_time: {start_time}"
            )

        # Step 1: Get historical messages, excluding current request's messages
        # Only get messages after last_memcell_time (current memcell's accumulated messages)
        history_raw_data_list = await conversation_data_repo.get_conversation_data(
            group_id=request.group_id,
            session_id=request.session_id,
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
    if data_type in (RawDataType.CONVERSATION, RawDataType.AGENTCONVERSATION):
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
                latest_time = to_iso_format(current_time)
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
    if data_type in (RawDataType.CONVERSATION, RawDataType.AGENTCONVERSATION):
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

    # Atomic Fact
    atomic_fact_docs = grouped_docs.get(MemoryType.ATOMIC_FACT, [])
    if atomic_fact_docs:
        atomic_fact_repo = get_bean_by_type(AtomicFactRecordRawRepository)
        saved_atomic_facts = await atomic_fact_repo.create_batch(atomic_fact_docs)
        saved_result[MemoryType.ATOMIC_FACT] = saved_atomic_facts

        sync_service = get_bean_by_type(MemorySyncService)
        await sync_service.sync_batch_atomic_facts(
            saved_atomic_facts, sync_to_es=True, sync_to_milvus=True
        )

    # Agent Case
    agent_case_docs = grouped_docs.get(MemoryType.AGENT_CASE, [])
    if agent_case_docs:
        from infra_layer.adapters.out.persistence.repository.agent_case_raw_repository import (
            AgentCaseRawRepository,
        )
        from infra_layer.adapters.out.search.elasticsearch.converter.agent_case_converter import (
            AgentCaseConverter,
        )
        from infra_layer.adapters.out.search.repository.agent_case_es_repository import (
            AgentCaseEsRepository,
        )
        from infra_layer.adapters.out.search.milvus.converter.agent_case_milvus_converter import (
            AgentCaseMilvusConverter,
        )
        from infra_layer.adapters.out.search.repository.agent_case_milvus_repository import (
            AgentCaseMilvusRepository,
        )

        agent_case_repo = get_bean_by_type(AgentCaseRawRepository)
        saved_agent_cases: List[Any] = []

        for doc in agent_case_docs:
            saved_doc = await agent_case_repo.append_experience(doc)
            saved_agent_cases.append(saved_doc)

            # ES sync
            try:
                agent_case_es_repo = get_bean_by_type(AgentCaseEsRepository)
                es_doc = AgentCaseConverter.from_mongo(saved_doc)
                await agent_case_es_repo.create(es_doc)
            except Exception as es_exc:
                logger.warning(f"[mem_memorize] AgentCase ES sync failed: {es_exc}")

            # Milvus sync
            try:
                agent_case_milvus_repo = get_bean_by_type(AgentCaseMilvusRepository)
                milvus_entity = AgentCaseMilvusConverter.from_mongo(saved_doc)
                vector = (
                    milvus_entity.get("vector")
                    if isinstance(milvus_entity, dict)
                    else None
                )
                if vector and len(vector) > 0:
                    await agent_case_milvus_repo.insert(milvus_entity, flush=False)
                else:
                    logger.warning(
                        "[mem_memorize] Skipping AgentCase Milvus write: vector empty or missing"
                    )
            except Exception as milvus_exc:
                logger.warning(
                    f"[mem_memorize] AgentCase Milvus sync failed: {milvus_exc}"
                )

        saved_result[MemoryType.AGENT_CASE] = saved_agent_cases

    # # Profile
    # profile_docs = grouped_docs.get(MemoryType.PROFILE, [])
    # if profile_docs:
    #     group_user_profile_repo = get_bean_by_type(GroupUserProfileMemoryRawRepository)
    #     saved_profiles = []
    #     for profile_mem in profile_docs:
    #         try:
    #             await _save_profile_memory_to_group_user_profile_memory(
    #                 profile_mem, group_user_profile_repo, version
    #             )
    #             saved_profiles.append(profile_mem)
    #         except Exception as exc:
    #             logger.error(f"Failed to save Profile memory: {exc}")
    #     if saved_profiles:
    #         saved_result[MemoryType.PROFILE] = saved_profiles

    # group_profile_docs = grouped_docs.get(
    #     "group_profile", []
    # )  # MemoryType.GROUP_PROFILE, [])
    # if group_profile_docs:
    #     group_profile_repo = get_bean_by_type(GroupProfileRawRepository)
    #     saved_group_profiles = []
    #     for mem in group_profile_docs:
    #         try:
    #             await _save_group_profile_memory(mem, group_profile_repo, version)
    #             saved_group_profiles.append(mem)
    #         except Exception as exc:
    #             logger.error(f"Failed to save Group Profile memory: {exc}")
    #     if saved_group_profiles:
    #         saved_result["group_profile"] = (
    #             saved_group_profiles  # MemoryType.GROUP_PROFILE] = saved_group_profiles
    #         )

    return saved_result


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

    conversation_data_repo = get_bean_by_type(ConversationDataRepository)

    # Note: Request logs are saved in controller layer for better timing control
    # (sync_status=-1, will be confirmed later based on boundary detection result)

    # ===== Preprocess and get historical data =====
    llm_custom_setting = None
    if request.raw_data_type in (
        RawDataType.CONVERSATION,
        RawDataType.AGENTCONVERSATION,
    ):
        with timed("validate_request"):
            request = await preprocess_conv_request(request, current_time)
        if request == None:
            logger.warning(f"[mem_memorize] preprocess_conv_request returned None")
            return 0

        # Fetch llm_custom_setting from global config (inherits automatically)
        # Note: llm_custom_setting is only stored in global config (group_id=None)
        llm_custom_setting = await _load_llm_custom_setting()
        if llm_custom_setting:
            logger.info(
                f"[mem_memorize] Using llm_custom_setting from global config for group {request.group_id}"
            )

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

    # Initialize MemoryManager with custom config
    memory_manager = MemoryManager(llm_config=llm_custom_setting)

    memcell_start = time.perf_counter()
    with timed("extract_memcell"):
        memcell_result = await memory_manager.extract_memcell(
            request.history_raw_data_list,
            request.new_raw_data_list,
            request.raw_data_type,
            request.group_id,
            [],
            flush=request.flush,
        )
    record_extraction_stage(
        space_id=space_id,
        raw_data_type=raw_data_type,
        stage='extract_memcell',
        duration_seconds=time.perf_counter() - memcell_start,
    )
    logger.debug(
        f"[mem_memorize] Extracting MemCell took: {time.perf_counter() - memcell_start} seconds"
    )

    memcells, status_result = memcell_result

    # Check boundary detection result
    logger.info("=" * 80)
    logger.info(
        f"[Boundary Detection Result] memcells={len(memcells)}, "
        f"should_wait={status_result.should_wait}"
    )
    logger.info("=" * 80)

    if not memcells:
        # No boundary detected, confirm current messages to accumulation (sync_status: -1 -> 0)
        with timed("update_message_status"):
            await conversation_data_repo.save_conversation_data(
                request.new_raw_data_list,
                request.group_id,
                session_id=request.session_id,
            )
            logger.info(
                f"[mem_memorize] No boundary, confirmed {len(request.new_raw_data_list)} messages to accumulation"
            )
            await update_status_when_no_memcell(
                request, status_result, current_time, request.raw_data_type
            )
        return 0

    # Determine which messages were consumed by the extracted MemCells.
    # MemCells are produced front-to-back, so consumed messages are a prefix of all_msgs.
    total_consumed = sum(len(mc.original_data) for mc in memcells if mc.original_data)
    all_raw_data = request.history_raw_data_list + request.new_raw_data_list
    remaining_raw_data = all_raw_data[total_consumed:]
    remaining_ids = [r.data_id for r in remaining_raw_data if r.data_id]

    try:
        if request.flush:
            # Flush: all messages have been consumed, clear the window
            delete_success = await conversation_data_repo.delete_conversation_data(
                request.group_id, session_id=request.session_id, exclude_message_ids=[]
            )
            if delete_success:
                logger.debug(
                    f"[mem_memorize] Flush mode: all messages marked as used, "
                    f"group_id={request.group_id}"
                )
            else:
                logger.warning(
                    f"[mem_memorize] Failed to clear conversation history: group_id={request.group_id}"
                )
        else:
            # Non-flush: consumed messages marked as used, remaining start next window
            delete_success = await conversation_data_repo.delete_conversation_data(
                request.group_id,
                session_id=request.session_id,
                exclude_message_ids=remaining_ids,
            )
            if delete_success:
                logger.debug(
                    f"[mem_memorize] Consumed messages marked as used "
                    f"(remaining={len(remaining_raw_data)}): group_id={request.group_id}"
                )
            else:
                logger.warning(
                    f"[mem_memorize] Failed to mark consumed messages: group_id={request.group_id}"
                )
            if remaining_raw_data:
                await conversation_data_repo.save_conversation_data(
                    remaining_raw_data, request.group_id, session_id=request.session_id
                )
    except Exception as e:
        logger.error(
            f"[mem_memorize] Exception while marking conversation history: {e}"
        )
        traceback.print_exc()

    # Save and process all extracted MemCells
    memories_count = 0

    try:
        for memcell in memcells:
            with timed("persist_memcell"):
                memcell = await _save_memcell_to_database(
                    memcell, current_time, session_id=request.session_id
                )
            logger.info(f"[mem_memorize] Saved MemCell: {memcell.event_id}")
            with timed("process_memory_extraction"):
                count = await process_memory_extraction(
                    memcell, request, memory_manager, current_time
                )
            memories_count += count

        logger.info(
            f"[mem_memorize] ✅ Memory extraction completed, "
            f"memcells={len(memcells)}, total_memories={memories_count}"
        )
        return memories_count
    except Exception as e:
        logger.error(f"[mem_memorize] ❌ Memory extraction failed: {e}")
        traceback.print_exc()
        return 0

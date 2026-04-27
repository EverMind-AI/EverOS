"""
Memory retrieval process configuration

Centralized management of all trigger conditions and thresholds for easy adjustment and maintenance.
"""

import logging
from dataclasses import dataclass
import os

from api_specs.memory_types import ParentType

logger = logging.getLogger(__name__)


@dataclass
class MemorizeConfig:
    """Memory retrieval process configuration"""

    # ===== Clustering configuration =====
    # Semantic similarity threshold; memcells exceeding this value will be clustered into the same cluster
    cluster_similarity_threshold: float = 0.3
    # Maximum time gap (days); memcells exceeding this gap will not be clustered together
    cluster_max_time_gap_days: int = 7

    # ===== Profile extraction configuration =====
    # Minimum number of memcells required to trigger Profile extraction
    profile_min_memcells: int = 1
    # Profile extraction interval: extract once every N memcells (1 = every time)
    profile_extraction_interval: int = 1
    # Minimum confidence required for Profile extraction
    profile_min_confidence: float = 0.6
    # Whether to enable version control
    profile_enable_versioning: bool = True
    # Profile maximum items
    profile_max_items: int = 25

    # ===== Parent type configuration =====
    # Default parent type for Episode (memcell or episode)
    default_episode_parent_type: str = ParentType.MEMCELL.value
    # Default parent type for Foresight (memcell or episode)
    default_foresight_parent_type: str = ParentType.MEMCELL.value
    # Default parent type for AtomicFact (memcell or episode)
    default_atomic_fact_parent_type: str = ParentType.MEMCELL.value

    # ===== Clustering lock configuration =====
    # Timeout (seconds) for acquiring the clustering lock
    clustering_lock_timeout: float = 600.0
    # Blocking timeout (seconds) for waiting to acquire the clustering lock
    clustering_lock_blocking_timeout: float = 2400.0

    # ===== Skill extraction lock configuration =====
    # Timeout (seconds) for acquiring the skill extraction lock
    skill_extraction_lock_timeout: float = 600.0
    # Blocking timeout (seconds) for waiting to acquire the skill extraction lock
    skill_extraction_lock_blocking_timeout: float = 2400.0

    # ===== Agent Skill extraction configuration =====
    # Minimum quality score (0.0-1.0) of the AgentCase required to trigger
    # skill extraction. Cases below this threshold are considered too low
    # quality to contribute to skill formation.
    skill_min_quality_score: float = 0.2
    # Minimum maturity score (0.0-1.0) for a skill to be retrievable
    skill_maturity_threshold: float = 0.6
    # Minimum confidence (0.0-1.0) for a skill to remain active.
    # Skills whose confidence drops below this threshold are kept in MongoDB
    # (data preserved) but removed from search engines and excluded from
    # future extraction context.
    skill_retire_confidence: float = 0.1

    # ===== Skip flags =====
    # Skip skill maturity scoring during skill extraction
    skip_skill_maturity_scoring: bool = False
    # Skip foresight and eventlog extraction
    skip_foresight_and_eventlog: bool = False
    # Skip profile extraction
    skip_profile_extraction: bool = False
    # Enable LLM-based relevance verification for skill search results
    enable_skill_llm_verify: bool = False


# Select config based on AGENT_MEMORIZE_MODE env var:
#   "online" (default) — full pipeline
#   "fast_skill" — skip profile/foresight/eventlog, skip maturity scoring
_AGENT_MEMORIZE_MODE = os.getenv("AGENT_MEMORIZE_MODE", "online").strip().lower()

if _AGENT_MEMORIZE_MODE == "fast_skill":
    DEFAULT_MEMORIZE_CONFIG = MemorizeConfig(
        skip_skill_maturity_scoring=True,
        skip_foresight_and_eventlog=True,
        skip_profile_extraction=True,
        clustering_lock_blocking_timeout=4800,
        skill_extraction_lock_blocking_timeout=4800,
        enable_skill_llm_verify=True
    )
else:
    if _AGENT_MEMORIZE_MODE != "online":
        logger.warning("Unknown AGENT_MEMORIZE_MODE=%r, falling back to 'online'", _AGENT_MEMORIZE_MODE)
    DEFAULT_MEMORIZE_CONFIG = MemorizeConfig()

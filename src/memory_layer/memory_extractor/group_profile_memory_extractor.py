"""Group Profile Memory Extraction for EverMemOS."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime
from enum import Enum
import hashlib
import os

from memory_layer.llm.llm_provider import LLMProvider
from memory_layer.memory_extractor.base_memory_extractor import MemoryExtractor, MemoryExtractRequest
from api_specs.memory_types import BaseMemory, MemoryType, MemCell
from common_utils.datetime_utils import (
    get_now_with_timezone,
    from_timestamp,
    from_iso_format,
    timezone,
)
from core.observation.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# Utility functions
# ============================================================================


def convert_to_datetime(timestamp, fallback_timestamp=None) -> datetime:
    """
    Convert various timestamp formats to datetime object with consistent timezone.

    Args:
        timestamp: The timestamp to convert
        fallback_timestamp: Fallback timestamp to use instead of current time

    Returns:
        datetime object with project consistent timezone
    """
    if isinstance(timestamp, datetime):
        # Ensure timezone consistency
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone)
        return timestamp.astimezone(timezone)
    elif isinstance(timestamp, (int, float)):
        # Use common function, automatically handle timezone and precision detection
        return from_timestamp(timestamp)
    elif isinstance(timestamp, str):
        try:
            # Use common function, unified timezone handling
            return from_iso_format(timestamp)
        except Exception as e:
            logger.exception(
                f"Failed to parse timestamp: {timestamp}, error: {e}, using fallback"
            )
            return fallback_timestamp if fallback_timestamp else get_now_with_timezone()
    else:
        logger.exception(f"Unknown timestamp format: {timestamp}, using fallback")
        return fallback_timestamp if fallback_timestamp else get_now_with_timezone()


# ============================================================================
# Data models - Keep in main file to ensure reference paths remain unchanged
# ============================================================================


class GroupRole(Enum):
    """7 key group roles in English."""

    DECISION_MAKER = "decision_maker"
    OPINION_LEADER = "opinion_leader"
    TOPIC_INITIATOR = "topic_initiator"
    EXECUTION_PROMOTER = "execution_promoter"
    CORE_CONTRIBUTOR = "core_contributor"
    COORDINATOR = "coordinator"
    INFO_SUMMARIZER = "info_summarizer"


class TopicStatus(Enum):
    """Topic status options."""

    EXPLORING = "exploring"
    DISAGREEMENT = "disagreement"
    CONSENSUS = "consensus"
    IMPLEMENTED = "implemented"


@dataclass
class TopicInfo:
    """
    Topic information for storage and output.

    Contains all information about the topic, including evidences and confidence.
    """

    name: str  # Topic name (phrased label)
    summary: str  # One-sentence overview
    status: str  # exploring/disagreement/consensus/implemented
    last_active_at: datetime  # Last active time (=updateTime)
    id: Optional[str] = (
        None  # Unique topic ID (system-generated, LLM does not need to provide)
    )
    update_type: Optional[str] = (
        None  # "new" | "update" (only used during incremental updates)
    )
    old_topic_id: Optional[str] = (
        None  # Points to old topic during update (only used during incremental updates)
    )
    evidences: Optional[List[str]] = field(
        default_factory=list
    )  # memcell_ids as evidence
    confidence: Optional[str] = None  # "strong" | "weak" - confidence level

    @classmethod
    def create_with_id(
        cls,
        name: str,
        summary: str,
        status: str,
        last_active_at: datetime,
        id: Optional[str] = None,
    ):
        """Create TopicInfo with generated or provided ID."""
        if not id:
            topic_id = hashlib.md5(name.encode('utf-8')).hexdigest()[:8]
            id = f"topic_{topic_id}"
        return cls(
            id=id,
            name=name,
            summary=summary,
            status=status,
            last_active_at=last_active_at,
        )


@dataclass
class GroupProfileMemory(BaseMemory):
    """
    Group Profile Memory aligned with design document.

    Contains group core information extracted from conversations.
    Evidences are now stored within topics and roles instead of separately.
    """

    # New fields, no conflict with base class
    group_name: Optional[str] = None

    # Extraction results (including strong + weak, sorted by last_active_at, limited to max_topics)
    # topics include evidences and confidence
    topics: Optional[List[TopicInfo]] = field(default_factory=list)
    # Each assignment in roles includes evidences and confidence
    # Format: role -> [{"user_id": "xxx", "user_name": "xxx", "confidence": "strong|weak", "evidences": [...]}]
    roles: Optional[Dict[str, List[Dict[str, str]]]] = field(default_factory=dict)

    # Note: summary and group_id are already defined as Optional in the base class, no need to redefine here

    def __post_init__(self):
        """Set memory_type to GROUP_PROFILE."""
        self.memory_type = MemoryType.GROUP_PROFILE
        # Ensure topics and roles are not None, preventing None values from historical data or exceptions
        if self.topics is None:
            self.topics = []
        if self.roles is None:
            self.roles = {}


@dataclass
class GroupProfileMemoryExtractRequest(MemoryExtractRequest):
    """
    Request for group profile memory extraction.

    Group Profile extraction may also need to process multiple MemCells (from clustering),
    so memcell_list support is provided
    """

    # Override base class field, optional single memcell
    memcell: Optional[MemCell] = None

    # Group Profile specific field
    memcell_list: Optional[List[MemCell]] = None
    user_id_list: Optional[List[str]] = None

    def __post_init__(self):
        # If memcell_list is provided, use it; otherwise use single memcell
        if self.memcell_list is None and self.memcell is not None:
            self.memcell_list = [self.memcell]
        elif self.memcell_list is None:
            self.memcell_list = []


# ============================================================================
# Main extractor class - Keep core logic
# ============================================================================


class GroupProfileMemoryExtractor(MemoryExtractor):
    """
    Extractor for group profile information from conversations.

    Uses helper processors for data processing, topic/role management, and LLM interaction.
    Core business logic remains in this class.
    """

    def __init__(
        self,
        llm_provider: LLMProvider | None = None,
        conversation_source: str = "original",
        max_topics: int = 10,
    ):
        """
        Initialize group profile extractor

        Args:
            llm_provider: LLM provider instance
            conversation_source: Conversation source, "original" or "episode"
            max_topics: Maximum number of topics
        """
        super().__init__(MemoryType.GROUP_PROFILE)
        self.llm_provider = llm_provider
        self.conversation_source = conversation_source
        self.max_topics = max_topics

        # Lazy initialization of helper processors
        self._data_processor = None
        self._topic_processor = None
        self._role_processor = None
        self._llm_handler = None

    # ========== Lazy load helper processors ==========

    @property
    def data_processor(self):
        """Lazy load data processor."""
        if self._data_processor is None:
            from memory_layer.memory_extractor.group_profile.data_processor import GroupProfileDataProcessor

            self._data_processor = GroupProfileDataProcessor(self.conversation_source)
        return self._data_processor

    @property
    def topic_processor(self):
        """Lazy load topic processor."""
        if self._topic_processor is None:
            from memory_layer.memory_extractor.group_profile.topic_processor import TopicProcessor

            self._topic_processor = TopicProcessor(self.data_processor)
        return self._topic_processor

    @property
    def role_processor(self):
        """Lazy load role processor."""
        if self._role_processor is None:
            from memory_layer.memory_extractor.group_profile.role_processor import RoleProcessor

            self._role_processor = RoleProcessor(self.data_processor)
        return self._role_processor

    @property
    def llm_handler(self):
        """Lazy load LLM handler."""
        if self._llm_handler is None:
            from memory_layer.memory_extractor.group_profile.llm_handler import GroupProfileLLMHandler

            self._llm_handler = GroupProfileLLMHandler(
                self.llm_provider, self.max_topics
            )
        return self._llm_handler

    # ========== Business logic methods ==========

    def _filter_group(
        self, group_name: Optional[str], user_id_list: Optional[List[str]]
    ) -> bool:
        """
        Filter groups that should not be processed.

        Args:
            group_name: Group name to check
            user_id_list: List of user IDs (currently unused but kept for API compatibility)

        Returns:
            True if group should be filtered out (not processed), False otherwise
        """
        # Control filtering logic based on environment variable ENV
        env_value = os.getenv('IGNORE_GROUP_NAME_FILTER', '').lower()

        if env_value == 'true':
            # If ENV variable is true, do not filter any groups
            return False
        else:
            # Otherwise, execute original filtering logic
            if group_name:
                return False
            return True

    # ========== Core extraction method ==========

    async def extract_memory(
        self, request: GroupProfileMemoryExtractRequest
    ) -> Optional[List[GroupProfileMemory]]:
        """
        Extract group profile memory from conversation memcells.

        【Core business process】Complete extraction task by combining various processors

        Args:
            request: Extract request containing memcells and related info

        Returns:
            List containing a single GroupProfileMemory object, or None
        """
        # ===== 1. Pre-checks =====
        if not request.memcell_list:
            return None

        group_id = request.group_id or ""
        group_name = request.group_name or ""
        memcell_list = request.memcell_list

        # Business filtering logic
        if self._filter_group(group_name, request.user_id_list):
            logger.info(
                f"[GroupProfileMemoryExtractor] Skipping group '{group_name}' - filtered out"
            )
            return None

        # ===== 2. Extract historical profile and build conversation text =====
        existing_profile = self.data_processor.extract_existing_group_profile(
            request.old_memory_list
        )
        conversation_text = self.data_processor.combine_conversation_text_with_ids(
            memcell_list
        )

        # ===== 3. Calculate time span =====
        start_time = convert_to_datetime(min(mc.timestamp for mc in memcell_list))
        end_time = convert_to_datetime(max(mc.timestamp for mc in memcell_list))
        timespan = f"{start_time.date()} to {end_time.date()}"

        try:
            # ===== 4. Execute LLM parallel analysis =====
            logger.info(
                f"[GroupProfileMemoryExtractor] Executing parallel analysis for group: {group_name}"
            )

            parsed_data = await self.llm_handler.execute_parallel_analysis(
                conversation_text=conversation_text,
                group_id=group_id,
                group_name=group_name,
                memcell_list=memcell_list,
                existing_profile=existing_profile,
                user_organization=None,
                timespan=timespan,
            )

            if not parsed_data:
                return None

            # ===== 5. Collect valid memcell IDs =====
            valid_memcell_ids = set(
                str(mc.event_id)
                for mc in memcell_list
                if hasattr(mc, 'event_id') and mc.event_id
            )
            logger.debug(
                f"[extract_memory] Valid memcell IDs count: {len(valid_memcell_ids)}"
            )

            # ===== 6. Process topics =====
            raw_topics = parsed_data.get("topics", [])
            existing_topics = (
                existing_profile.get("topics", []) if existing_profile else []
            )

            all_topics = self.topic_processor.apply_topic_incremental_updates(
                llm_topics=raw_topics,
                existing_topics_with_evidences=existing_topics,
                memcell_list=memcell_list,
                valid_memcell_ids=valid_memcell_ids,
                max_topics=self.max_topics,
            )
            logger.info(
                f"[extract_memory] Processed {len(all_topics)} topics (strong + weak)"
            )

            # ===== 7. Process roles =====
            raw_roles = parsed_data.get("roles", {})
            existing_roles = (
                existing_profile.get("roles", {}) if existing_profile else {}
            )

            # Build comprehensive speaker mapping
            comprehensive_mapping = (
                self.data_processor.get_comprehensive_speaker_mapping(
                    memcell_list, existing_roles
                )
            )

            all_roles = self.role_processor.process_roles_with_evidences(
                role_data=raw_roles,
                speaker_mapping=comprehensive_mapping,
                existing_roles=existing_roles,
                valid_memcell_ids=valid_memcell_ids,
                memcell_list=memcell_list,
            )
            logger.info(
                f"[extract_memory] Processed roles with {sum(len(v) for v in all_roles.values())} total assignments"
            )

            # ===== 8. Assemble final result =====
            group_profile = GroupProfileMemory(
                memory_type=MemoryType.GROUP_PROFILE,
                user_id="",
                timestamp=get_now_with_timezone(),
                ori_event_id_list=[
                    str(mc.event_id) for mc in memcell_list if hasattr(mc, 'event_id')
                ],
                group_id=group_id,
                group_name=group_name,
                topics=all_topics,  # All topics (strong + weak) with evidences, sorted by last_active_at
                roles=all_roles,  # All roles (strong + weak, strong first) with evidences
                summary=parsed_data.get("summary", ""),
                subject=parsed_data.get("subject", "not_found"),
            )

            return [group_profile]

        except Exception as e:
            logger.error(
                f"[GroupProfileMemoryExtractor] Extraction error: {e}", exc_info=True
            )
            return None


# ============================================================================
# Public API declaration
# ============================================================================

__all__ = [
    # Main class
    'GroupProfileMemoryExtractor',
    'GroupProfileMemoryExtractRequest',
    # Data models
    'GroupProfileMemory',
    'TopicInfo',
    # Enums
    'GroupRole',
    'TopicStatus',
    # Utility functions
    'convert_to_datetime',
]

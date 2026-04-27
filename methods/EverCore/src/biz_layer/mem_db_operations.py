"""
Database operations and data conversion functions.
Extracted from mem_memorize.py for database operations and data conversion logic.

This module contains the following features:
1. Time processing functions: Unified handling of various time formats to ensure consistency in database storage
2. Data conversion functions: Convert business layer objects to database document format
3. Database operation functions: Execute specific database CRUD operations
4. Status table operation functions: Manage the lifecycle of conversation status
"""

import time
from api_specs.dtos import MemorizeRequest
from api_specs.memory_types import MemCell, RawDataType
from core.di import get_bean_by_type
from infra_layer.adapters.out.persistence.repository.conversation_status_raw_repository import (
    ConversationStatusRawRepository,
)
from infra_layer.adapters.out.persistence.repository.memcell_raw_repository import (
    MemCellRawRepository,
)
from infra_layer.adapters.out.persistence.document.memory.episodic_memory import (
    EpisodicMemory,
)
from infra_layer.adapters.out.persistence.document.memory.memcell import (
    MemCell as DocMemCell,
    DataTypeEnum,
)
from typing import List, Any, Optional
from datetime import datetime, timedelta
from common_utils.datetime_utils import get_now_with_timezone, from_iso_format
from core.observation.logger import get_logger
from core.events import ApplicationEventPublisher
from infra_layer.adapters.out.event.memcell_created_event import MemCellCreatedEvent
from infra_layer.adapters.out.persistence.document.memory.foresight_record import (
    ForesightRecord,
)
from infra_layer.adapters.out.persistence.document.memory.atomic_fact_record import (
    AtomicFactRecord,
)
from api_specs.memory_types import RawDataType, AgentCase
from infra_layer.adapters.out.persistence.document.memory.agent_case import (
    AgentCaseRecord,
)

logger = get_logger(__name__)

# ==================== Data Conversion Functions ====================


def _convert_episode_memory_to_doc(
    episode_memory: Any,
    current_time: Optional[datetime] = None,
    session_id: Optional[str] = None,
) -> EpisodicMemory:
    """
    Convert EpisodeMemory business object to EpisodicMemory database document format.

    Use cases:
    - Format conversion before saving episodic memory to EpisodicMemoryRawRepository
    - Ensure business layer Memory objects meet database document model field requirements
    - Handle timestamp format and extension field mapping

    Args:
        episode_memory: Business layer EpisodeMemory object
        current_time: Current time, used as fallback when timestamp parsing fails

    Returns:
        EpisodicMemory: Episodic memory object in database document format
    """
    from infra_layer.adapters.out.persistence.document.memory.episodic_memory import (
        EpisodicMemory,
    )
    from agentic_layer.vectorize_service import get_vectorize_service

    # Parse timestamp to datetime object
    if current_time is None:
        current_time = get_now_with_timezone()

    # Default to using current_time
    timestamp_dt = current_time

    if hasattr(episode_memory, 'timestamp') and episode_memory.timestamp:
        try:
            timestamp_dt = from_iso_format(episode_memory.timestamp)
        except Exception as e:
            logger.debug(f"Timestamp conversion failed, using current time: {e}")
            timestamp_dt = current_time

    participants = episode_memory.participants
    return EpisodicMemory(
        user_id=episode_memory.user_id,
        group_id=episode_memory.group_id,
        session_id=session_id,
        timestamp=timestamp_dt,
        participants=participants,
        sender_ids=participants,
        summary=episode_memory.summary or "",
        subject=episode_memory.subject or "",
        episode=(
            episode_memory.episode
            if hasattr(episode_memory, 'episode')
            else episode_memory.summary or ""
        ),
        type=str(episode_memory.type.value) if episode_memory.type else "",
        parent_type=getattr(episode_memory, 'parent_type', None),
        parent_id=getattr(episode_memory, 'parent_id', None),
        vector_model=episode_memory.vector_model,
        vector=episode_memory.vector,
    )


def _convert_foresight_to_doc(
    foresight: Any,
    parent_doc: EpisodicMemory,
    current_time: Optional[datetime] = None,
    session_id: Optional[str] = None,
) -> ForesightRecord:
    """
    Convert Foresight business object to unified foresight document format.

    Args:
        foresight: Business layer Foresight object
        parent_doc: Parent episodic memory document
        current_time: Current time

    Returns:
        ForesightRecord: Foresight object in database document format
    """

    if current_time is None:
        current_time = get_now_with_timezone()

    participants = parent_doc.participants
    return ForesightRecord(
        user_id=getattr(foresight, "user_id", None),
        session_id=session_id,
        content=foresight.foresight,  # Foresight class uses 'foresight' field, but DB uses 'content'
        parent_type=foresight.parent_type,
        parent_id=foresight.parent_id,
        start_time=foresight.start_time,
        end_time=foresight.end_time,
        duration_days=foresight.duration_days,
        type=str(parent_doc.type) if parent_doc.type else None,
        group_id=parent_doc.group_id,
        participants=participants,
        sender_ids=participants,
        vector=foresight.vector,
        vector_model=foresight.vector_model,
        evidence=foresight.evidence,
    )


def _convert_atomic_fact_to_docs(
    atomic_fact_obj: Any,
    parent_doc: EpisodicMemory,
    current_time: Optional[datetime] = None,
    session_id: Optional[str] = None,
) -> List["AtomicFactRecord"]:
    """
    Convert AtomicFact business object to generic atomic fact document list.

    Args:
        atomic_fact_obj: Business layer AtomicFact object
        parent_doc: Parent episodic memory document
        current_time: Current time

    Returns:
        List[AtomicFactRecord]: List of atomic fact objects in database document format
    """
    if current_time is None:
        current_time = get_now_with_timezone()

    docs: List[AtomicFactRecord] = []
    if not atomic_fact_obj.atomic_fact or not atomic_fact_obj.fact_embeddings:
        return docs

    for i, fact in enumerate(atomic_fact_obj.atomic_fact):
        if i >= len(atomic_fact_obj.fact_embeddings):
            break

        vector = atomic_fact_obj.fact_embeddings[i]
        if hasattr(vector, 'tolist'):
            vector = vector.tolist()

        participants = parent_doc.participants
        doc = AtomicFactRecord(
            user_id=atomic_fact_obj.user_id,
            session_id=session_id,
            atomic_fact=fact,
            parent_type=atomic_fact_obj.parent_type,
            parent_id=atomic_fact_obj.parent_id,
            timestamp=parent_doc.timestamp or current_time,
            type=str(parent_doc.type) if parent_doc.type else None,
            group_id=atomic_fact_obj.group_id,
            participants=participants,
            sender_ids=participants,
            vector=vector,
            vector_model=getattr(atomic_fact_obj, 'vector_model', None),
        )
        docs.append(doc)

    return docs


def _extract_user_id_from_memcell(memcell: MemCell) -> Optional[str]:
    """Extract user_id from an agent conversation MemCell.

    Finds the first message with role='user' and returns its sender_id.
    original_data items are wrapped as {"message": msg, "parse_info": ...}.
    """
    for item in memcell.original_data or []:
        msg = item.get("message", item) if isinstance(item, dict) else item
        if isinstance(msg, dict) and msg.get("role") == "user":
            sender_id = msg.get("sender_id")
            if sender_id:
                return sender_id
    return None


def _convert_agent_case_to_doc(
    agent_case: AgentCase,
    memcell: MemCell,
    current_time: Optional[datetime] = None,
    session_id: Optional[str] = None,
) -> AgentCaseRecord:
    """Convert AgentCase business object to AgentCaseRecord database document."""
    if current_time is None:
        current_time = get_now_with_timezone()

    # Parse timestamp
    timestamp_dt = current_time
    if memcell.timestamp:
        try:
            if isinstance(memcell.timestamp, datetime):
                timestamp_dt = memcell.timestamp
            elif isinstance(memcell.timestamp, str):
                timestamp_dt = from_iso_format(memcell.timestamp)
        except Exception:
            timestamp_dt = current_time

    # Extract user_id from first role='user' message's sender_id
    user_id = _extract_user_id_from_memcell(memcell)

    return AgentCaseRecord(
        id=agent_case.id,
        user_id=user_id,
        group_id=memcell.group_id,
        session_id=session_id,
        timestamp=timestamp_dt,
        task_intent=agent_case.task_intent,
        approach=agent_case.approach,
        key_insight=agent_case.key_insight,
        quality_score=agent_case.quality_score,
        parent_type="memcell",
        parent_id=str(memcell.event_id) if memcell.event_id else None,
        vector=agent_case.vector,
        vector_model=agent_case.vector_model,
    )


# ==================== Database Operation Functions ====================
from core.observation.tracing.decorators import trace_logger


async def _save_memcell_to_database(
    memcell: MemCell, current_time: datetime, session_id: Optional[str] = None
) -> MemCell:
    """
    Convert business layer MemCell to document model and save to database.

    Args:
        memcell: Business layer MemCell object
        current_time: Current time, used as fallback when timestamp conversion fails
        session_id: Session identifier for conversation isolation

    Note:
        - Skips saving and logs when conversion fails
        - Does not interrupt flow when save fails
    """
    try:
        memcell_repo = get_bean_by_type(MemCellRawRepository)

        # Convert timestamp to timezone-aware datetime
        timestamp_dt = current_time
        if memcell.timestamp:
            try:
                timestamp_dt = from_iso_format(memcell.timestamp)
            except (ValueError, TypeError) as e:
                logger.debug(f"Timestamp conversion failed, using current time: {e}")

        # Convert data type enum
        doc_type = None
        if memcell.type and memcell.type == RawDataType.CONVERSATION:
            doc_type = DataTypeEnum.CONVERSATION
        elif memcell.type and memcell.type == RawDataType.AGENTCONVERSATION:
            doc_type = DataTypeEnum.AGENTCONVERSATION

        # Create document model
        doc_memcell = DocMemCell(
            timestamp=timestamp_dt,
            group_id=memcell.group_id,
            session_id=session_id,
            original_data=memcell.original_data or [],
            participants=memcell.participants,
            sender_ids=memcell.sender_ids,
            type=doc_type,
        )

        # Save to database
        result = await memcell_repo.append_memcell(doc_memcell)
        if result:
            memcell.event_id = str(result.event_id)
            logger.info(
                f"[mem_db_operations] MemCell saved successfully: {memcell.event_id}"
            )
            # Publish MemCellCreatedEvent
            try:
                publisher = get_bean_by_type(ApplicationEventPublisher)
                event = MemCellCreatedEvent(
                    memcell_id=memcell.event_id,
                    timestamp=int(current_time.timestamp() * 1000),
                )
                await publisher.publish(event)
                logger.debug(
                    f"[mem_db_operations] MemCellCreatedEvent published: {memcell.event_id}"
                )
            except Exception as e:
                logger.warning(
                    f"[mem_db_operations] Failed to publish MemCellCreatedEvent: {e}"
                )
        else:
            logger.info(f"[mem_db_operations] MemCell save failed: {memcell.event_id}")

    except Exception as e:
        logger.error(f"MemCell save failed: {e}")
        import traceback

        traceback.print_exc()
    return memcell


# ==================== Status Table Operation Functions ====================


async def _update_status_for_continuing_conversation(
    status_repo: ConversationStatusRawRepository,
    request: MemorizeRequest,
    latest_time: str,
    current_time: datetime,
) -> bool:
    """
    Update status record for continuing conversation (update new_msg_start_time).

    Use cases:
    - Called when MemCell extraction is judged as non-boundary
    - Conversation is still continuing, need to accumulate more messages
    - Update new_msg_start_time to latest message time to prepare for next processing

    Args:
        status_repo: ConversationStatusRawRepository instance
        request: Memorize request object
        latest_time: Timestamp of latest message
        current_time: Current time

    Returns:
        bool: Returns True if update successful, False otherwise
    """
    try:
        # First get existing status
        existing_status = await status_repo.get_by_group_id(
            request.group_id, session_id=request.session_id
        )
        if not existing_status:
            logger.info(
                f"Existing status not found, creating new status record: group_id={request.group_id}"
            )
            # Create new status record
            latest_dt = from_iso_format(latest_time)
            update_data = {
                "old_msg_start_time": None,
                "new_msg_start_time": latest_dt + timedelta(milliseconds=1),
                "last_memcell_time": None,
                "created_at": from_iso_format(current_time),
                "updated_at": from_iso_format(current_time),
            }
            result = await status_repo.upsert_by_group_id(
                request.group_id, update_data, session_id=request.session_id
            )
            if result:
                logger.info(
                    f"New status created successfully: group_id={request.group_id}"
                )
                return True
            else:
                logger.warning(
                    f"Failed to create new status: group_id={request.group_id}"
                )
                return False

        # Update new_msg_start_time to latest message time + 1 millisecond
        latest_dt = from_iso_format(latest_time)
        new_msg_start_time = latest_dt

        update_data = {
            "old_msg_start_time": (
                from_iso_format(existing_status.old_msg_start_time)
                if existing_status.old_msg_start_time
                else None
            ),
            "new_msg_start_time": new_msg_start_time + timedelta(milliseconds=1),
            "last_memcell_time": (
                from_iso_format(existing_status.last_memcell_time)
                if existing_status.last_memcell_time
                else None
            ),
            "created_at": from_iso_format(existing_status.created_at),
            "updated_at": current_time,
        }

        logger.debug(f"Conversation continuing, update new_msg_start_time")
        result = await status_repo.upsert_by_group_id(
            request.group_id, update_data, session_id=request.session_id
        )

        if result:
            logger.info(f"Conversation continuation status updated successfully")
            return True
        else:
            logger.warning(f"Conversation continuation status update failed")
            return False

    except Exception as e:
        logger.error(f"Conversation continuation status update failed: {e}")
        return False


async def _update_status_after_memcell_extraction(
    status_repo: ConversationStatusRawRepository,
    request: MemorizeRequest,
    memcell_time: str,
    current_time: datetime,
) -> bool:
    """
    Update status table after MemCell extraction (update old_msg_start_time and new_msg_start_time).

    Use cases:
    - Called after successfully extracting MemCell and completing memory extraction
    - Update processed message time boundary to avoid duplicate processing
    - Reset new_msg_start_time to current time to prepare for receiving new messages

    Args:
        status_repo: ConversationStatusRawRepository instance
        request: Memorize request object
        memcell_time: Timestamp of MemCell
        current_time: Current time

    Returns:
        bool: Returns True if update successful, False otherwise

    Note:
        - old_msg_start_time is updated to last history message time + 1ms
        - new_msg_start_time is reset to current time
        - last_memcell_time records the latest MemCell extraction time
    """
    try:
        # Get timestamp of last history data
        last_history_time = None
        if request.history_raw_data_list and request.history_raw_data_list[-1]:
            last_history_data = request.history_raw_data_list[-1]
            if hasattr(last_history_data, 'content') and isinstance(
                last_history_data.content, dict
            ):
                last_history_time = last_history_data.content.get('timestamp')
            elif hasattr(last_history_data, 'timestamp'):
                last_history_time = last_history_data.timestamp

        first_new_time = None
        if request.new_raw_data_list and request.new_raw_data_list[0]:
            first_new_data = request.new_raw_data_list[0]
            if hasattr(first_new_data, 'content') and isinstance(
                first_new_data.content, dict
            ):
                first_new_time = first_new_data.content.get('timestamp')
            elif hasattr(first_new_data, 'timestamp'):
                first_new_time = first_new_data.timestamp

        last_new_time = None
        if request.new_raw_data_list and request.new_raw_data_list[-1]:
            last_new_data = request.new_raw_data_list[-1]
            if hasattr(last_new_data, 'content') and isinstance(
                last_new_data.content, dict
            ):
                last_new_time = last_new_data.content.get('timestamp')
            elif hasattr(last_new_data, 'timestamp'):
                last_new_time = last_new_data.timestamp

        if last_new_time:
            last_new_dt = from_iso_format(last_new_time)
            new_msg_start_time = last_new_dt + timedelta(milliseconds=1)
        else:
            new_msg_start_time = from_iso_format(current_time)

        # Calculate old_msg_start_time (last history timestamp + 1 millisecond)
        if first_new_time:
            first_new_dt = from_iso_format(first_new_time)
            old_msg_start_time = first_new_dt
        elif last_history_time:
            last_history_dt = from_iso_format(last_history_time)
            old_msg_start_time = last_history_dt + timedelta(milliseconds=1)
        else:
            # If no history data, use existing current_time
            old_msg_start_time = from_iso_format(current_time)

        update_data = {
            "old_msg_start_time": old_msg_start_time,
            "new_msg_start_time": new_msg_start_time,  # Current time
            "last_memcell_time": from_iso_format(memcell_time),
            "updated_at": current_time,
        }

        # TODO : clear queue

        logger.debug(f"Update status table after MemCell extraction")
        result = await status_repo.upsert_by_group_id(
            request.group_id, update_data, session_id=request.session_id
        )

        if result:
            logger.info(f"Status update after MemCell extraction successful")
            return True
        else:
            logger.warning(f"Status update after MemCell extraction failed")
            return False

    except Exception as e:
        logger.error(f"Status update after MemCell extraction failed: {e}")
        return False

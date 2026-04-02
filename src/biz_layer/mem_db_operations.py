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
from memory_layer.memory_extractor.profile_memory_extractor import ProfileMemory
from memory_layer.memory_extractor.group_profile_memory_extractor import (
    GroupProfileMemory,
)
from memory_layer.memory_extractor.profile_memory_extractor import (
    GroupImportanceEvidence,
    ImportanceEvidence,
)
from core.di import get_bean_by_type
from infra_layer.adapters.out.persistence.repository.conversation_status_raw_repository import (
    ConversationStatusRawRepository,
)
from infra_layer.adapters.out.persistence.repository.group_user_profile_memory_raw_repository import (
    GroupUserProfileMemoryRawRepository,
)
from infra_layer.adapters.out.persistence.repository.group_profile_raw_repository import (
    GroupProfileRawRepository,
)
from infra_layer.adapters.out.persistence.repository.core_memory_raw_repository import (
    CoreMemoryRawRepository,
)
from infra_layer.adapters.out.persistence.repository.memcell_raw_repository import (
    MemCellRawRepository,
)
from infra_layer.adapters.out.persistence.document.memory.core_memory import CoreMemory
from infra_layer.adapters.out.persistence.document.memory.episodic_memory import (
    EpisodicMemory,
)
from infra_layer.adapters.out.persistence.document.memory.memcell import (
    MemCell as DocMemCell,
    RawData as DocRawData,
    DataTypeEnum,
)
from memory_layer.memory_extractor.profile_memory_extractor import ProjectInfo
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from common_utils.datetime_utils import (
    get_now_with_timezone,
    to_timezone,
    to_iso_format,
    from_iso_format,
    from_timestamp,
)
from core.observation.logger import get_logger
from core.events import ApplicationEventPublisher
from infra_layer.adapters.out.event.memcell_created_event import MemCellCreatedEvent
from infra_layer.adapters.out.persistence.document.memory.foresight_record import (
    ForesightRecord,
)
from infra_layer.adapters.out.persistence.document.memory.event_log_record import (
    EventLogRecord,
)
from api_specs.memory_types import RawDataType

logger = get_logger(__name__)

# ==================== Time Processing Functions ====================


def _normalize_datetime_for_storage(
    timestamp: Any, current_time: Optional[datetime] = None
) -> datetime:
    """
    Convert various time formats to local timezone datetime object (with timezone info, for database storage).

    Use cases:
    - Ensure uniform time field format before saving data to database
    - Handle time data from different sources (string, timestamp, datetime object)
    - Avoid data errors caused by timezone inconsistency

    Args:
        timestamp: Input time data, supports datetime, str, int, float types
        current_time: Fallback time, used when conversion fails

    Returns:
        datetime: Datetime object with timezone information
    """
    try:
        if not timestamp:
            return None
        if isinstance(timestamp, datetime):
            # If datetime object, use to_timezone to convert to local timezone
            return to_timezone(timestamp)
        elif isinstance(timestamp, str):
            # String format, use from_iso_format to parse
            return from_iso_format(timestamp)
        elif isinstance(timestamp, (int, float)):
            # Numeric timestamp, use from_timestamp to convert (milliseconds to seconds)
            return from_timestamp(timestamp / 1000)
        else:
            # Other types, return current local time
            return current_time if current_time else get_now_with_timezone()
    except Exception as e:
        logger.debug(f"Time formatting failed: {timestamp}, error: {e}")
        return current_time if current_time else get_now_with_timezone()


def _convert_timestamp_to_time(
    timestamp: Any, current_time: Optional[datetime] = None
) -> str:
    """
    Convert timestamp to ISO format time string, supports multiple input formats.

    Use cases:
    - Convert time data read from database to standard ISO format
    - Format output of time fields in business layer objects
    - Unified formatting of time fields in API responses

    Args:
        timestamp: Input time data, supports datetime, str, int, float types
        current_time: Fallback time, used when conversion fails

    Returns:
        str: ISO format time string
    """
    try:
        if not timestamp:
            return None
        if isinstance(timestamp, datetime):
            # If datetime object, use to_iso_format to convert
            return to_iso_format(timestamp)
        elif isinstance(timestamp, (int, float)):
            # If numeric timestamp (milliseconds), convert to datetime first then to ISO
            dt = from_timestamp(timestamp / 1000)
            return to_iso_format(dt)
        elif isinstance(timestamp, str):
            # If string, try to parse as datetime then convert to ISO
            try:
                dt = from_iso_format(timestamp)
                return to_iso_format(dt)
            except Exception:
                # If parsing fails, return string directly
                return timestamp
        else:
            # Other types, return current time in ISO format
            return to_iso_format(
                current_time if current_time else get_now_with_timezone()
            )
    except Exception as e:
        logger.debug(f"Timestamp conversion failed: {timestamp}, error: {e}")
        return to_iso_format(current_time if current_time else get_now_with_timezone())


# ==================== Data Conversion Functions ====================


def _convert_importance_evidence_to_document(
    importance_evidence_list: List[ImportanceEvidence],
) -> List[Dict[str, Any]]:
    """
    Convert ImportanceEvidence to database document format.
    """
    if not importance_evidence_list:
        return None
    return [
        {
            "user_id": importance_evidence.user_id,
            "group_id": importance_evidence.group_id,
            "speak_count": importance_evidence.speak_count,
            "refer_count": importance_evidence.refer_count,
            "conversation_count": importance_evidence.conversation_count,
        }
        for importance_evidence in importance_evidence_list
    ]


def _convert_document_to_importance_evidence(
    importance_evidence_list: List[Dict[str, Any]]
) -> List[ImportanceEvidence]:
    """
    Convert database document format to ImportanceEvidence.
    """
    if not importance_evidence_list:
        return None
    return [
        ImportanceEvidence(
            user_id=importance_evidence["user_id"],
            group_id=importance_evidence["group_id"],
            speak_count=importance_evidence["speak_count"],
            refer_count=importance_evidence["refer_count"],
            conversation_count=importance_evidence["conversation_count"],
        )
        for importance_evidence in importance_evidence_list
    ]


def _convert_group_importance_evidence_to_document(
    group_importance_evidence: GroupImportanceEvidence,
) -> Dict[str, Any]:
    """
    Convert GroupImportanceEvidence to database document format.
    """
    if not group_importance_evidence:
        return None
    return {
        "group_id": group_importance_evidence.group_id,
        "is_important": group_importance_evidence.is_important,
        "evidence_list": _convert_importance_evidence_to_document(
            group_importance_evidence.evidence_list
        ),
    }


def _convert_document_to_group_importance_evidence(
    group_importance_evidence: Dict[str, Any]
) -> GroupImportanceEvidence:
    """
    Convert database document format to GroupImportanceEvidence.
    """
    if not group_importance_evidence:
        return None
    return GroupImportanceEvidence(
        group_id=group_importance_evidence["group_id"],
        is_important=group_importance_evidence["is_important"],
        evidence_list=_convert_document_to_importance_evidence(
            group_importance_evidence["evidence_list"]
        ),
    )


def _convert_episode_memory_to_doc(
    episode_memory: Any, current_time: Optional[datetime] = None
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
            if isinstance(episode_memory.timestamp, datetime):
                timestamp_dt = episode_memory.timestamp
            elif isinstance(episode_memory.timestamp, str):
                timestamp_dt = from_iso_format(episode_memory.timestamp)
            elif isinstance(episode_memory.timestamp, (int, float)):
                # If numeric timestamp (milliseconds), convert to datetime
                timestamp_dt = from_timestamp(episode_memory.timestamp / 1000)
        except Exception as e:
            logger.debug(f"Timestamp conversion failed, using current time: {e}")
            timestamp_dt = current_time

    return EpisodicMemory(
        user_id=episode_memory.user_id,  # Keep None or actual value, do not convert to empty string
        user_name=episode_memory.user_name or '',
        group_id=episode_memory.group_id,
        group_name=episode_memory.group_name,
        timestamp=timestamp_dt,
        participants=episode_memory.participants,
        summary=episode_memory.summary or "",
        subject=episode_memory.subject or "",
        episode=(
            episode_memory.episode
            if hasattr(episode_memory, 'episode')
            else episode_memory.summary or ""
        ),
        type=str(episode_memory.type.value) if episode_memory.type else "",
        keywords=getattr(episode_memory, 'keywords', None),
        linked_entities=getattr(episode_memory, 'linked_entities', None),
        memcell_event_id_list=getattr(episode_memory, 'memcell_event_id_list', None),
        vector_model=episode_memory.vector_model,
        vector=episode_memory.vector,
        extend={
            "memory_type": episode_memory.memory_type.value,
            "ori_event_id": getattr(episode_memory, 'ori_event_id', None),
            "tags": getattr(episode_memory, 'tags', None),
        },
    )


def _convert_foresight_to_doc(
    foresight: Any, parent_doc: EpisodicMemory, current_time: Optional[datetime] = None
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

    return ForesightRecord(
        user_id=getattr(foresight, "user_id", None),
        user_name=getattr(
            foresight, "user_name", getattr(parent_doc, "user_name", None)
        ),
        content=foresight.foresight,  # Foresight class uses 'foresight' field, but DB uses 'content'
        parent_type=foresight.parent_type,
        parent_id=foresight.parent_id,
        start_time=foresight.start_time,
        end_time=foresight.end_time,
        duration_days=foresight.duration_days,
        group_id=parent_doc.group_id,
        group_name=parent_doc.group_name,
        participants=parent_doc.participants,
        vector=foresight.vector,
        vector_model=foresight.vector_model,
        evidence=foresight.evidence,
        extend={},
    )


def _convert_event_log_to_docs(
    event_log: Any, parent_doc: EpisodicMemory, current_time: Optional[datetime] = None
) -> List["EventLogRecord"]:
    """
    Convert EventLog business object to generic event log document list.

    Args:
        event_log: Business layer EventLog object
        parent_doc: Parent episodic memory document
        current_time: Current time

    Returns:
        List[EventLogRecord]: List of event log objects in database document format
    """
    if current_time is None:
        current_time = get_now_with_timezone()

    docs: List[EventLogRecord] = []
    if not event_log.atomic_fact or not event_log.fact_embeddings:
        return docs

    for i, fact in enumerate(event_log.atomic_fact):
        if i >= len(event_log.fact_embeddings):
            break

        vector = event_log.fact_embeddings[i]
        if hasattr(vector, 'tolist'):
            vector = vector.tolist()

        doc = EventLogRecord(
            user_id=event_log.user_id,
            user_name=event_log.user_name or '',
            atomic_fact=fact,
            parent_type=event_log.parent_type,
            parent_id=event_log.parent_id,
            timestamp=parent_doc.timestamp or current_time,
            group_id=event_log.group_id,
            group_name=event_log.group_name,
            participants=parent_doc.participants,
            vector=vector,
            vector_model=getattr(event_log, 'vector_model', None),
            event_type=parent_doc.type or RawDataType.CONVERSATION.value,
            extend={},
        )
        docs.append(doc)

    return docs


def _convert_group_profile_data_to_profile_format(
    group_profile_memory: GroupProfileMemory,
) -> Dict[str, Any]:
    """
    Convert GroupProfileMemory data format to the format expected by GroupProfile.

    Use cases:
    - Format conversion before saving GroupProfileMemory to GroupProfileRawRepository
    - Handle field mapping and type conversion between different data structures
    - Ensure timestamp format consistency

    Args:
        group_profile_memory: Business layer GroupProfileMemory object

    Returns:
        dict: Dictionary containing converted data, keys are GroupProfile field names
    """
    from infra_layer.adapters.out.persistence.document.memory.group_profile import (
        TopicInfo as DocTopicInfo,
    )

    # Handle topics conversion: from business TopicInfo to document TopicInfo
    # Fix: Initialize as empty list instead of None to avoid empty list being saved as None
    topics = []
    if (
        hasattr(group_profile_memory, 'topics')
        and group_profile_memory.topics is not None
    ):
        for topic in group_profile_memory.topics:
            if hasattr(topic, 'name'):  # Business layer TopicInfo object
                # Ensure last_active_at is datetime object
                last_active_at = topic.last_active_at
                if isinstance(last_active_at, str):
                    try:
                        from common_utils.datetime_utils import from_iso_format

                        last_active_at = from_iso_format(last_active_at)
                    except Exception:
                        from common_utils.datetime_utils import get_now_with_timezone

                        last_active_at = get_now_with_timezone()
                elif not isinstance(last_active_at, datetime):
                    from common_utils.datetime_utils import get_now_with_timezone

                    last_active_at = get_now_with_timezone()

                doc_topic = DocTopicInfo(
                    name=topic.name,
                    summary=topic.summary,
                    status=topic.status,
                    last_active_at=last_active_at,
                    id=getattr(topic, 'id', None),
                    update_type=getattr(topic, 'update_type', None),
                    old_topic_id=getattr(topic, 'old_topic_id', None),
                    evidences=getattr(topic, 'evidences', []),
                    confidence=getattr(topic, 'confidence', None),
                )
                topics.append(doc_topic)
            elif isinstance(topic, dict):
                # Already in dict format, create DocTopicInfo directly
                topics.append(DocTopicInfo(**topic))

    # Handle roles conversion: from Dict to RoleAssignment objects
    from infra_layer.adapters.out.persistence.document.memory.group_profile import (
        RoleAssignment,
    )

    # Fix: Initialize as empty dict instead of None to avoid empty dict being saved as None
    roles = {}
    if (
        hasattr(group_profile_memory, 'roles')
        and group_profile_memory.roles is not None
    ):
        for role_name, assignments in group_profile_memory.roles.items():
            role_assignments = []
            for assignment in assignments:
                if isinstance(assignment, dict):
                    # Create RoleAssignment object from dict
                    role_assignment = RoleAssignment(
                        user_id=assignment.get('user_id', ''),
                        user_name=assignment.get('user_name', ''),
                        confidence=assignment.get('confidence'),
                        evidences=assignment.get('evidences', []),
                    )
                    role_assignments.append(role_assignment)
                else:
                    # If already an object, add directly
                    role_assignments.append(assignment)
            if role_assignments:
                roles[role_name] = role_assignments

    # Handle timestamp: ensure it's integer milliseconds timestamp
    # TODO: Refactoring: timestamp should remain as datetime instead of converting to int
    timestamp = None
    if hasattr(group_profile_memory, 'timestamp') and group_profile_memory.timestamp:
        if isinstance(group_profile_memory.timestamp, datetime):
            timestamp = int(group_profile_memory.timestamp.timestamp() * 1000)
        elif isinstance(group_profile_memory.timestamp, (int, float)):
            timestamp = int(group_profile_memory.timestamp)
        elif isinstance(group_profile_memory.timestamp, str):
            try:
                from common_utils.datetime_utils import from_iso_format

                dt = from_iso_format(group_profile_memory.timestamp)
                timestamp = int(dt.timestamp() * 1000)
            except Exception:
                from common_utils.datetime_utils import get_now_with_timezone

                timestamp = int(get_now_with_timezone().timestamp() * 1000)
    else:
        # Use current time as default value
        from common_utils.datetime_utils import get_now_with_timezone

        timestamp = int(get_now_with_timezone().timestamp() * 1000)

    # Extract other fields
    group_name = getattr(group_profile_memory, 'group_name', None)
    subject = getattr(group_profile_memory, 'theme', None) or getattr(
        group_profile_memory, 'subject', None
    )
    summary = getattr(group_profile_memory, 'summary', None)
    extend = getattr(group_profile_memory, 'extend', None)

    return {
        "group_name": group_name,
        "topics": topics,
        "roles": roles,
        "timestamp": timestamp,
        "subject": subject,
        "summary": summary,
        "extend": extend,
    }


def _convert_document_to_project_info(project_info: Dict[str, str]) -> ProjectInfo:
    """
    Convert database document format to ProjectInfo.
    """
    if not project_info:
        return None

    def _process_field_with_evidences(value):
        """Process fields containing evidences, maintain List[Dict[str, Any]] format"""
        if value is None:
            return None

        # If already a dict list containing value/evidences, return directly
        if isinstance(value, list):
            if (
                value
                and isinstance(value[0], dict)
                and ("value" in value[0] or "evidences" in value[0])
            ):
                return value
            # If plain string list or other type list, convert to value/evidences format
            return [{"value": str(item), "evidences": []} for item in value if item]

        # If string, try to parse
        if isinstance(value, str):
            if not value.strip():
                return None
            try:
                import ast

                parsed_value = ast.literal_eval(value)
                if isinstance(parsed_value, list):
                    # Check if already in canonical format
                    if (
                        parsed_value
                        and isinstance(parsed_value[0], dict)
                        and (
                            "value" in parsed_value[0] or "evidences" in parsed_value[0]
                        )
                    ):
                        return parsed_value
                    # Otherwise convert to canonical format
                    return [
                        {"value": str(item), "evidences": []}
                        for item in parsed_value
                        if item
                    ]
            except (ValueError, SyntaxError):
                # Parsing failed, split by comma
                items = [item.strip() for item in value.split(',') if item.strip()]
                return [{"value": item, "evidences": []} for item in items]

        return None

    return ProjectInfo(
        project_id=project_info.get("project_id", ""),
        project_name=project_info.get("project_name", ""),
        entry_date=project_info.get("entry_date", ""),
        user_objective=_process_field_with_evidences(
            project_info.get("user_objective")
        ),
        contributions=_process_field_with_evidences(project_info.get("contributions")),
        subtasks=_process_field_with_evidences(project_info.get("subtasks")),
        user_concerns=_process_field_with_evidences(project_info.get("user_concerns")),
    )


def _convert_projects_participated_list(
    projects_participated: Optional[List[Dict[str, str]]]
) -> List[ProjectInfo]:
    """
    Convert projects_participated (List[Dict[str, str]]) from database to List[ProjectInfo].
    """
    if not projects_participated:
        return []

    result = []
    for project_dict in projects_participated:
        if isinstance(project_dict, dict):
            project_info = _convert_document_to_project_info(project_dict)
            if project_info:
                result.append(project_info)

    return result


def _convert_profile_data_to_core_format(profile_memory: ProfileMemory) -> CoreMemory:
    """
    Convert ProfileMemory data format to the format expected by CoreMemory.

    Use cases:
    - Data format conversion before saving user profile memory to CoreMemoryRawRepository
    - Handle data type conversion for fields like skills, personality, projects
    - Ensure data conforms to CoreMemory document model field definitions

    Args:
        profile_memory: Business layer ProfileMemory object

    Returns:
        dict: Dictionary containing converted data, keys are CoreMemory field names
    """

    # Convert hard_skills: use profile_memory.hard_skills directly
    hard_skills = None
    if hasattr(profile_memory, 'hard_skills') and profile_memory.hard_skills:
        hard_skills = profile_memory.hard_skills

    # Convert soft_skills: use profile_memory.soft_skills directly
    soft_skills = None
    if hasattr(profile_memory, 'soft_skills') and profile_memory.soft_skills:
        soft_skills = profile_memory.soft_skills

    output_reasoning = getattr(profile_memory, 'output_reasoning', None)

    motivation_system = None
    if (
        hasattr(profile_memory, 'motivation_system')
        and profile_memory.motivation_system
    ):
        motivation_system = profile_memory.motivation_system

    fear_system = None
    if hasattr(profile_memory, 'fear_system') and profile_memory.fear_system:
        fear_system = profile_memory.fear_system

    value_system = None
    if hasattr(profile_memory, 'value_system') and profile_memory.value_system:
        value_system = profile_memory.value_system

    humor_use = None
    if hasattr(profile_memory, 'humor_use') and profile_memory.humor_use:
        humor_use = profile_memory.humor_use

    colloquialism = None
    if hasattr(profile_memory, 'colloquialism') and profile_memory.colloquialism:
        colloquialism = profile_memory.colloquialism

    # Convert way_of_decision_making: use raw data directly (already contains evidences)
    way_of_decision_making = None
    if (
        hasattr(profile_memory, 'way_of_decision_making')
        and profile_memory.way_of_decision_making
    ):
        way_of_decision_making = profile_memory.way_of_decision_making

    # Convert personality: use raw data directly (already contains evidences)
    personality = None
    if hasattr(profile_memory, 'personality') and profile_memory.personality:
        personality = profile_memory.personality

    # Convert projects_participated: List[ProjectInfo] -> List[Dict[str, Any]]
    # Note: ProjectInfo fields now contain evidence-embedded data, use raw format directly
    projects_participated = None
    if (
        hasattr(profile_memory, 'projects_participated')
        and profile_memory.projects_participated
    ):
        if isinstance(profile_memory.projects_participated, list):
            projects_participated = []
            for project in profile_memory.projects_participated:
                if hasattr(project, 'project_id'):  # ProjectInfo object
                    # Use raw data directly, preserve evidence-embedded format
                    user_objective = getattr(project, 'user_objective', None)
                    contributions = getattr(project, 'contributions', None)
                    subtasks = getattr(project, 'subtasks', None)
                    user_concerns = getattr(project, 'user_concerns', None)

                    project_dict = {
                        "project_id": (
                            str(project.project_id) if project.project_id else ""
                        ),
                        "project_name": (
                            str(project.project_name) if project.project_name else ""
                        ),
                        "entry_date": (
                            str(project.entry_date) if project.entry_date else ""
                        ),
                        "user_objective": user_objective,
                        "contributions": contributions,
                        "subtasks": subtasks,
                        "user_concerns": user_concerns,
                    }
                    projects_participated.append(project_dict)
                elif isinstance(project, dict):
                    projects_participated.append(project)  # Already in correct format

    # Extract additional fields
    user_goal = getattr(profile_memory, 'user_goal', None)
    work_responsibility = getattr(profile_memory, 'work_responsibility', None)
    working_habit_preference = getattr(profile_memory, 'working_habit_preference', None)
    interests = getattr(profile_memory, 'interests', None)
    tendency = getattr(profile_memory, 'tendency', None)
    user_name = getattr(profile_memory, 'user_name', None)
    group_importance_evidence = getattr(
        profile_memory, 'group_importance_evidence', None
    )

    return {
        "user_name": user_name,
        "output_reasoning": output_reasoning,
        "hard_skills": hard_skills,
        "soft_skills": soft_skills,
        "way_of_decision_making": way_of_decision_making,
        "personality": personality,
        "projects_participated": projects_participated,
        "user_goal": user_goal,
        "work_responsibility": work_responsibility,
        "working_habit_preference": working_habit_preference,
        "interests": interests,
        "tendency": tendency,
        "motivation_system": motivation_system,
        "fear_system": fear_system,
        "value_system": value_system,
        "humor_use": humor_use,
        "colloquialism": colloquialism,
        "group_importance_evidence": _convert_group_importance_evidence_to_document(
            group_importance_evidence
        ),
    }


def _convert_memcell_to_document(
    memcell: MemCell, current_time: Optional[datetime] = None
) -> DocMemCell:
    """
    Convert business layer MemCell to document model MemCell.

    Use cases:
    - Format conversion before saving MemCell to MemCellRawRepository
    - Handle nested structure conversion of raw data to avoid infinite recursion
    - Unify timestamp format and data type enum conversion

    Args:
        memcell: Business layer MemCell object
        current_time: Current time, used as fallback when timestamp conversion fails

    Returns:
        DocMemCell: MemCell object in database document format

    Raises:
        Exception: Thrown when an error occurs during conversion
    """
    try:
        # Temporary solution: disable raw data conversion to avoid infinite recursion
        # Issue: Nested validation of BaseModel objects causes infinite recursion, even with simplest structures
        # TODO: Need to find a better solution to properly convert original_data
        doc_original_data = []
        if memcell.type == RawDataType.CONVERSATION:
            for raw_data_dict in memcell.original_data:
                # Actual data structure is: {'speaker_id': 'user_1', 'speaker_name': 'Alice', 'content': 'message content', 'timestamp': '...'}
                # Here content is the direct message string, not a nested dict
                # Helper function: convert various types to string
                def to_string(value):
                    if value is None:
                        return ''
                    elif isinstance(value, str):
                        return value
                    elif isinstance(value, datetime):
                        return value.isoformat()
                    elif isinstance(value, list):
                        return ','.join(str(item) for item in value) if value else ''
                    else:
                        return str(value)

                message = {
                    "content": raw_data_dict.get('content')
                    or '',  # Handle None content explicitly
                    "extend": {
                        "speaker_id": to_string(raw_data_dict.get('speaker_id', '')),
                        "speaker_name": to_string(
                            raw_data_dict.get('speaker_name', '')
                        ),
                        "timestamp": to_string(
                            _convert_timestamp_to_time(
                                raw_data_dict.get('timestamp', '')
                            )
                        ),
                        "message_id": to_string(raw_data_dict.get('data_id', '')),
                        "receiverId": to_string(raw_data_dict.get('receiverId', '')),
                        "roomId": to_string(raw_data_dict.get('roomId', '')),
                        "userIdList": to_string(raw_data_dict.get('userIdList', [])),
                        "createBy": to_string(raw_data_dict.get('createBy', '')),
                        "updateTime": to_string(raw_data_dict.get('updateTime', '')),
                        "msgType": to_string(raw_data_dict.get('msgType', '')),
                        "referList": to_string(raw_data_dict.get('referList', [])),
                        "orgId": to_string(raw_data_dict.get('orgId', '')),
                    },
                }

                # Create document model RawData
                doc_raw_data = DocRawData(
                    data_type=DataTypeEnum.CONVERSATION,  # Default to conversation type
                    messages=[message],  # Message list
                    # meta=raw_data_dict.get('metadata', {})  # Metadata
                )
                doc_original_data.append(doc_raw_data)

        # Convert timestamp to timezone-aware datetime to avoid infinite recursion
        if current_time is None:
            current_time = get_now_with_timezone()
        timestamp_dt = current_time
        if memcell.timestamp:
            try:
                # Check timestamp type and process
                # TODO: Refactoring: timestamp should remain as datetime, no type checking needed
                if isinstance(memcell.timestamp, datetime):
                    # If already datetime object, use directly
                    timestamp_dt = _normalize_datetime_for_storage(memcell.timestamp)
                else:
                    # If numeric timestamp, need to convert (assuming seconds timestamp)
                    timestamp_dt = _normalize_datetime_for_storage(
                        memcell.timestamp * 1000
                    )
            except (ValueError, TypeError) as e:
                logger.debug(f"Timestamp conversion failed, using current time: {e}")

        logger.debug(f"MemCell save timestamp: {timestamp_dt}")

        # Convert data type enum
        doc_type = None
        if memcell.type:
            try:
                # Convert RawDataType to DataTypeEnum
                if memcell.type == RawDataType.CONVERSATION:
                    doc_type = DataTypeEnum.CONVERSATION
            except Exception as e:
                logger.warning(f"Data type conversion failed: {e}")

        # MemCell itself is group memory, user_id is always None
        primary_user_id = None

        # Prepare extension fields - extract extension properties based on specific MemCell type
        email_fields = {}
        linkdoc_fields = {}

        # Prepare foresight_memories (convert to dict list)
        foresight_memories_list = None
        if hasattr(memcell, 'foresight_memories') and memcell.foresight_memories:
            foresight_memories_list = [
                (
                    sm.to_dict()
                    if hasattr(sm, 'to_dict')
                    else (sm if isinstance(sm, dict) else None)
                )
                for sm in memcell.foresight_memories
            ]
            foresight_memories_list = [
                sm for sm in foresight_memories_list if sm is not None
            ]

        # Prepare event_log (convert to dict)
        event_log_dict = None
        if hasattr(memcell, 'event_log') and memcell.event_log:
            if hasattr(memcell.event_log, 'to_dict'):
                event_log_dict = memcell.event_log.to_dict()
            elif isinstance(memcell.event_log, dict):
                event_log_dict = memcell.event_log

        # Prepare extend field (contains embedding and other extension info)
        extend_dict = {}
        if hasattr(memcell, 'extend') and memcell.extend:
            extend_dict = memcell.extend if isinstance(memcell.extend, dict) else {}

        # Add embedding to extend (if exists)
        if hasattr(memcell, 'embedding') and memcell.embedding:
            extend_dict['embedding'] = memcell.embedding

        # Create document model - pass timezone-aware datetime object directly instead of string
        # This avoids infinite recursion triggered by base class datetime validator
        doc_memcell = DocMemCell(
            user_id=primary_user_id,
            timestamp=timestamp_dt,  # Pass timezone-aware datetime directly
            summary=memcell.summary,
            group_id=memcell.group_id,
            original_data=doc_original_data,
            participants=memcell.participants,
            type=doc_type,
            subject=memcell.subject,
            keywords=memcell.keywords,
            linked_entities=memcell.linked_entities,
            episode=memcell.episode,
            foresight_memories=foresight_memories_list,  # ✅ Add foresight
            event_log=event_log_dict,  # ✅ Add event log
            extend=(
                extend_dict if extend_dict else None
            ),  # ✅ Add extend (contains embedding)
        )

        return doc_memcell

    except Exception as e:
        logger.error(f"MemCell conversion failed: {e}")
        import traceback

        traceback.print_exc()
        raise


# ==================== Database Operation Functions ====================
from core.observation.tracing.decorators import trace_logger


async def _save_memcell_to_database(
    memcell: MemCell, current_time: datetime
) -> MemCell:
    """
    Save MemCell to database.

    Use cases:
    - Persistence operation after successfully extracting MemCell in memorize flow
    - Ensure conversation segment memory units are saved
    - Provide data foundation for subsequent memory extraction

    Args:
        memcell: Business layer MemCell object

    Note:
        - Function internally performs automatic format conversion
        - Skips saving and logs when conversion fails
        - Prints error message but does not interrupt flow when save fails
    """
    try:
        # Initialize MemCell Repository
        memcell_repo = get_bean_by_type(MemCellRawRepository)
        # Convert business layer MemCell to document model
        doc_memcell = _convert_memcell_to_document(memcell, current_time)

        # Check if conversion was successful
        if doc_memcell is None:
            logger.warning(
                f"MemCell conversion skipped, cannot save: {memcell.event_id}"
            )
            return

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


async def _save_group_profile_memory(
    group_profile_memory: GroupProfileMemory,
    group_profile_raw_repo: GroupProfileRawRepository,
    version: Optional[str] = None,
) -> None:
    """
    Save GroupProfileMemory to GroupProfileRawRepository.
    """
    try:
        # Convert data format
        converted_data = _convert_group_profile_data_to_profile_format(
            group_profile_memory
        )

        # Full overwrite save GroupProfile (create or update)
        logger.debug(f"Save GroupProfile: {group_profile_memory.group_id}")

        # Prepare save data (separate timestamp, as upsert_by_group_id needs it passed separately)
        save_data = {}
        timestamp = None

        # Add non-null fields, but separate timestamp
        for k, v in converted_data.items():
            if v is not None:
                if k == "timestamp":
                    timestamp = v
                else:
                    save_data[k] = v

        save_data["version"] = version

        # Use upsert_by_group_id method (update if exists, create if not)
        await group_profile_raw_repo.upsert_by_group_id(
            group_profile_memory.group_id, save_data, timestamp=timestamp
        )

    except Exception as e:
        logger.error(f"GroupProfileMemory save failed: {e}")
        import traceback

        traceback.print_exc()


async def _save_profile_memory_to_core(
    profile_memory: ProfileMemory,
    core_memory_repo: CoreMemoryRawRepository,
    version: Optional[str] = None,
) -> None:
    """
    Save ProfileMemory to CoreMemoryRawRepository.

    Use cases:
    - When user profile memory extracted in memorize flow needs persistence
    - Full overwrite update of user's core memory information
    - Handle storage of user characteristic information like skills, personality, projects

    Args:
        profile_memory: Business layer ProfileMemory object
        core_memory_repo: CoreMemoryRawRepository instance

    Note:
        - Uses full overwrite strategy, directly replacing existing data with new data
        - Does not perform data merge, ensuring data consistency and accuracy

    Raises:
        Exception: Thrown when an error occurs during save
    """
    try:
        # Convert data format
        converted_data = _convert_profile_data_to_core_format(profile_memory)

        # Full overwrite save CoreMemory (create or update)
        logger.debug(f"Save CoreMemory: {profile_memory.user_id}")

        # Prepare save data (does not include user_id, as upsert_by_user_id handles it automatically)
        save_data = {"extend": getattr(profile_memory, 'extend', None)}
        # Add non-null fields
        for k, v in converted_data.items():
            if v is not None:
                save_data[k] = v

        save_data["version"] = version

        # Use upsert_by_user_id method (update if exists, create if not)
        await core_memory_repo.upsert_by_user_id(profile_memory.user_id, save_data)

    except Exception as e:
        logger.error(f"Save Profile Memory to CoreMemory failed: {e}")
        import traceback

        traceback.print_exc()
        raise


async def _save_profile_memory_to_group_user_profile_memory(
    profile_memory: ProfileMemory,
    group_user_profile_memory_repo: GroupUserProfileMemoryRawRepository,
    version: Optional[str] = None,
) -> None:
    """
    Save ProfileMemory to GroupUserProfileMemoryRawRepository.

    Use cases:
    - When user profile memory extracted in memorize flow needs persistence
    - Full overwrite update of user's core memory information
    - Handle storage of user characteristic information like skills, personality, projects

    Args:
        profile_memory: Business layer ProfileMemory object
        group_user_profile_memory_repo: GroupUserProfileMemoryRawRepository instance

    Note:
        - Uses full overwrite strategy, directly replacing existing data with new data
        - Does not perform data merge, ensuring data consistency and accuracy

    Raises:
        Exception: Thrown when an error occurs during save
    """
    try:
        # Convert data format
        converted_data = _convert_profile_data_to_core_format(profile_memory)

        # Full overwrite save CoreMemory (create or update)
        logger.debug(f"Save CoreMemory: {profile_memory.user_id}")

        # Prepare save data (does not include user_id, as upsert_by_user_id handles it automatically)
        save_data = {"extend": getattr(profile_memory, 'extend', None)}
        # Add non-null fields
        for k, v in converted_data.items():
            if v is not None:
                save_data[k] = v

        save_data["version"] = version

        # Use upsert_by_user_group method (update if exists, create if not)
        await group_user_profile_memory_repo.upsert_by_user_group(
            profile_memory.user_id, profile_memory.group_id, save_data
        )

    except Exception as e:
        logger.error(f"Save Profile Memory to GroupUserProfileMemory failed: {e}")
        import traceback

        traceback.print_exc()
        raise


# ==================== Status Table Operation Functions ====================


@dataclass
class ConversationStatus:
    """
    Conversation status table data structure.

    Used to track conversation processing status and time boundaries, ensuring continuity and consistency of message processing.

    Use cases:
    - Manage conversation lifecycle status
    - Record time boundaries of processed and pending messages
    - Support pause, continue and completion status management for conversations
    """

    group_id: str  # Group ID
    old_msg_start_time: Optional[str]  # Start time of processed messages
    new_msg_start_time: Optional[str]  # Start time of new messages
    last_memcell_time: Optional[str]  # Time of last MemCell extraction
    created_at: str  # Creation time
    updated_at: str  # Update time


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
        existing_status = await status_repo.get_by_group_id(request.group_id)
        if not existing_status:
            logger.info(
                f"Existing status not found, creating new status record: group_id={request.group_id}"
            )
            # Create new status record
            latest_dt = _normalize_datetime_for_storage(latest_time)
            update_data = {
                "old_msg_start_time": None,
                "new_msg_start_time": latest_dt + timedelta(milliseconds=1),
                "last_memcell_time": None,
                "created_at": _normalize_datetime_for_storage(current_time),
                "updated_at": _normalize_datetime_for_storage(current_time),
            }
            result = await status_repo.upsert_by_group_id(request.group_id, update_data)
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
        latest_dt = _normalize_datetime_for_storage(latest_time)
        new_msg_start_time = latest_dt

        update_data = {
            "old_msg_start_time": (
                _normalize_datetime_for_storage(existing_status.old_msg_start_time)
                if existing_status.old_msg_start_time
                else None
            ),
            "new_msg_start_time": new_msg_start_time + timedelta(milliseconds=1),
            "last_memcell_time": (
                _normalize_datetime_for_storage(existing_status.last_memcell_time)
                if existing_status.last_memcell_time
                else None
            ),
            "created_at": _normalize_datetime_for_storage(existing_status.created_at),
            "updated_at": current_time,
        }

        logger.debug(f"Conversation continuing, update new_msg_start_time")
        result = await status_repo.upsert_by_group_id(request.group_id, update_data)

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
            last_new_dt = _normalize_datetime_for_storage(last_new_time)
            new_msg_start_time = last_new_dt + timedelta(milliseconds=1)
        else:
            new_msg_start_time = _normalize_datetime_for_storage(current_time)

        # Calculate old_msg_start_time (last history timestamp + 1 millisecond)
        if first_new_time:
            first_new_dt = _normalize_datetime_for_storage(first_new_time)
            old_msg_start_time = first_new_dt
        elif last_history_time:
            last_history_dt = _normalize_datetime_for_storage(last_history_time)
            old_msg_start_time = last_history_dt + timedelta(milliseconds=1)
        else:
            # If no history data, use existing current_time
            old_msg_start_time = _normalize_datetime_for_storage(current_time)

        update_data = {
            "old_msg_start_time": old_msg_start_time,
            "new_msg_start_time": new_msg_start_time,  # Current time
            "last_memcell_time": _normalize_datetime_for_storage(memcell_time),
            "updated_at": current_time,
        }

        # TODO : clear queue

        logger.debug(f"Update status table after MemCell extraction")
        result = await status_repo.upsert_by_group_id(request.group_id, update_data)

        if result:
            logger.info(f"Status update after MemCell extraction successful")
            return True
        else:
            logger.warning(f"Status update after MemCell extraction failed")
            return False

    except Exception as e:
        logger.error(f"Status update after MemCell extraction failed: {e}")
        return False

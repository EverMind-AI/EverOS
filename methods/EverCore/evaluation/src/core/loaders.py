"""
Data loaders.

Provides loading functionality for different datasets.
Supports automatic conversion of non-Locomo format datasets.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

from evaluation.src.core.data_models import Dataset, Conversation, Message, QAPair
from evaluation.src.converters.registry import get_converter
from common_utils.datetime_utils import from_iso_format


def load_dataset(
    dataset_name: str, data_path: str, max_content_length: Optional[int] = None
) -> Dataset:
    """
    Smart dataset loading with automatic conversion support.

    Args:
        dataset_name: Dataset name (e.g., "locomo", "longmemeval", "personamem")
        data_path: Data file path or directory path
        max_content_length: Optional max content length for truncating long messages

    Returns:
        Dataset: Standard format dataset
    """
    data_path_obj = Path(data_path)

    # Check if conversion is needed
    converter = get_converter(dataset_name)

    if converter:
        # Dataset needs conversion
        if data_path_obj.is_file():
            # If given a file path, use its parent directory
            data_dir = data_path_obj.parent
        else:
            data_dir = data_path_obj

        # Check if conversion is needed
        if converter.needs_conversion(data_dir):
            print(f"📝 Converted file not found, converting {dataset_name}...")

            # Build input file paths
            input_files = converter.get_input_files()
            input_paths = {
                key: str(data_dir / filename) for key, filename in input_files.items()
            }

            # Execute conversion
            output_path = str(converter.get_converted_path(data_dir))
            converter.convert(input_paths, output_path)

        # Use converted file
        locomo_file = converter.get_converted_path(data_dir)
    else:
        # Native Locomo format, use directly
        if data_path_obj.is_file():
            locomo_file = data_path_obj
        else:
            # If directory, try to find .json file
            json_files = list(data_path_obj.glob("*.json"))
            if not json_files:
                raise FileNotFoundError(f"No JSON file found in {data_path_obj}")
            locomo_file = json_files[0]

    return load_locomo_dataset(
        str(locomo_file),
        dataset_name=dataset_name,
        max_content_length=max_content_length,
    )


def load_locomo_dataset(
    data_path: str,
    dataset_name: str = "locomo",
    max_content_length: Optional[int] = None,
) -> Dataset:
    """
    Load LoCoMo format dataset.

    Args:
        data_path: Locomo format data file path
        dataset_name: Dataset name (default "locomo", converted datasets should pass original name)
        max_content_length: Optional max content length for truncating long messages

    Returns:
        Dataset: Standard format dataset
    """
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    conversations = []
    qa_pairs = []

    for idx, item in enumerate(raw_data):
        # Add dataset prefix to avoid conversation_id conflicts between datasets
        # Example: locomo_0, longmemeval_0, personamem_0
        conv_id = f"{dataset_name}_{idx}"
        conversation_data = item.get("conversation", {})
        qa_data = item.get("qa", [])

        # Convert conversation
        conversation = _convert_locomo_conversation(
            conversation_data, conv_id, max_content_length=max_content_length
        )
        conversations.append(conversation)

        # Convert QA pairs
        for qa_idx, qa_item in enumerate(qa_data):
            qa_pair = _convert_locomo_qa_pair(qa_item, conv_id, qa_idx)
            qa_pairs.append(qa_pair)

    return Dataset(
        dataset_name=dataset_name,
        conversations=conversations,
        qa_pairs=qa_pairs,
        metadata={"total_conversations": len(conversations)},
    )


def _convert_locomo_conversation(
    conversation_data: dict, conv_id: str, max_content_length: Optional[int] = None
) -> Conversation:
    """
    Convert LoCoMo conversation.

    Args:
        conversation_data: LoCoMo format conversation data
        conv_id: Conversation ID
        max_content_length: Optional max content length for truncating long messages

    Returns:
        Conversation: Standard format conversation
    """
    messages = []

    # Get all session keys, sorted by numeric value
    session_keys = sorted(
        [
            key
            for key in conversation_data.keys()
            if key.startswith("session_") and not key.endswith("_date_time")
        ],
        key=lambda x: int(
            x.split("_")[1]
        ),  # Extract number X from session_X for sorting
    )

    # Generate fake start time for data without timestamps (for online APIs)
    # Use fixed baseline: 2024-01-01 00:00:00
    fake_base_time = datetime(2024, 1, 1, 0, 0, 0)

    # Step 1: Parse timestamps for all sessions
    session_times = []
    for session_idx, session_key in enumerate(session_keys):
        session_time_key = f"{session_key}_date_time"
        if session_time_key in conversation_data:
            session_time = _parse_locomo_timestamp(conversation_data[session_time_key])

            # If parse fails or is "Unknown", generate fake timestamp
            is_fake = session_time is None
            if is_fake:
                session_time = fake_base_time + timedelta(hours=session_idx)

            session_times.append({"time": session_time, "is_fake": is_fake})
        else:
            # No date_time field, generate fake timestamp
            session_times.append(
                {"time": fake_base_time + timedelta(hours=session_idx), "is_fake": True}
            )

    # Step 2: Assign message timestamps for each session
    for session_idx, session_key in enumerate(session_keys):
        session_messages = conversation_data[session_key]

        if not session_messages:
            continue

        # Get current session start time
        current_session_time = session_times[session_idx]["time"]
        is_fake_timestamp = session_times[session_idx]["is_fake"]

        # Calculate message time intervals
        # Strategy: prefer 30s intervals, only reduce if would exceed next session
        num_messages = len(session_messages)
        default_interval = 30  # Default 30s interval

        if num_messages > 1:
            # Calculate total duration needed with default interval
            required_duration = (num_messages - 1) * default_interval

            # Get available time span
            if session_idx + 1 < len(session_times):
                # Has next session: calculate time to next session
                next_session_time = session_times[session_idx + 1]["time"]
                available_duration = (
                    next_session_time - current_session_time
                ).total_seconds()

                # If time span is negative or too small (data issue), use default interval
                if available_duration <= 0:
                    time_interval = default_interval
                # Leave 10% buffer to avoid last message too close to next session
                elif required_duration > available_duration * 0.9:
                    # Need to reduce interval to fit all messages
                    time_interval = (available_duration * 0.9) / (num_messages - 1)
                else:
                    # Can use default interval
                    time_interval = default_interval
            else:
                # Last session: use default interval directly
                time_interval = default_interval
        else:
            # Only one message, place at session start
            time_interval = 0

        # Convert each message
        for msg_idx, msg in enumerate(session_messages):
            # Try to parse message-level timestamp first (priority 1)
            msg_timestamp = None
            timestamp_source = None

            if 'time' in msg and msg['time']:
                # Priority 1: Use message-level timestamp (strict parsing, raises on error)
                msg_timestamp = from_iso_format(msg['time'], strict=True)
                timestamp_source = "message_level"
            else:
                # Priority 2: Generate from session-level timestamp
                msg_timestamp = current_session_time + timedelta(
                    seconds=msg_idx * time_interval
                )
                timestamp_source = "fake" if is_fake_timestamp else "session_level"

            # Handle image information
            content = msg['text']
            if msg.get("img_url"):
                blip_caption = msg.get("blip_caption", "an image")
                sender_name = msg['speaker']
                content = f"[{sender_name} shared an image: {blip_caption}] {content}"

            # Apply content length limit (if specified)
            if max_content_length and len(content) > max_content_length:
                content = content[:max_content_length]

            message = Message(
                sender_id=f"{msg['speaker'].lower().replace(' ', '_')}_{conv_id}",
                sender_name=msg['speaker'],
                content=content,  # Use processed content
                timestamp=msg_timestamp,
                metadata={
                    "session": session_key,
                    "dia_id": msg.get("dia_id"),
                    "img_url": msg.get("img_url"),
                    "blip_caption": msg.get("blip_caption"),
                    "query": msg.get("query"),
                    "timestamp_source": timestamp_source,  # Mark timestamp source
                },
            )
            messages.append(message)

    return Conversation(
        conversation_id=conv_id,
        messages=messages,
        metadata={
            "speaker_a": conversation_data.get("speaker_a"),
            "speaker_b": conversation_data.get("speaker_b"),
        },
    )


def _convert_locomo_qa_pair(qa_item: dict, conv_id: str, qa_idx: int) -> QAPair:
    """Convert LoCoMo QA pair."""
    # Extract additional fields to metadata
    metadata = {"conversation_id": conv_id}

    # If has all_options (PersonaMem multiple choice), save to metadata
    if "all_options" in qa_item:
        metadata["all_options"] = qa_item["all_options"]

    # Prefer question_id from data if exists, otherwise generate unique ID
    question_id = qa_item.get("question_id")
    if not question_id:
        # Use conv_id + qa_idx to generate unique ID to avoid conflicts
        question_id = f"{conv_id}_qa{qa_idx}"

    # Normalize category to string (compatible with int and str)
    category = qa_item.get("category")
    if category is not None:
        category = str(category)

    return QAPair(
        question_id=question_id,
        question=qa_item.get("question", ""),
        answer=qa_item.get("answer", ""),
        category=category,
        evidence=qa_item.get("evidence", []),
        metadata=metadata,
    )


def _parse_locomo_timestamp(timestamp_str: str) -> Optional[datetime]:
    """
    Parse LoCoMo timestamp format.

    Input format: "6:07 pm on 13 January, 2023"
    Special value: "Unknown" or unparseable returns None
    Output: datetime object or None
    """
    # Clean string
    timestamp_str = timestamp_str.replace("\\s+", " ").strip()

    # Handle special cases: Unknown or empty string
    if timestamp_str.lower() == "unknown" or not timestamp_str:
        # No time information, return None
        return None

    try:
        return datetime.strptime(timestamp_str, "%I:%M %p on %d %B, %Y")
    except ValueError:
        # If parse fails, return None and print warning
        print(
            f"⚠️  Warning: Failed to parse timestamp '{timestamp_str}', no timestamp will be set"
        )
        return None

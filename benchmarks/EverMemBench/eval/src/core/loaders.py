"""
Data loaders for multi-person group chat datasets.

Loads JSON files in the format:
{
    "dialogues": {
        "2025-01-09": {
            "Group 1": [
                {"speaker": "Name", "time": "2025-01-09 09:32:15", "dialogue": "..."},
                ...
            ]
        },
        ...
    }
}
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List

from eval.src.core.data_models import Dataset, GroupChatDay, GroupChatMessage


def load_groupchat_dataset(
    file_path: str,
    dataset_name: Optional[str] = None,
    max_days: Optional[int] = None
) -> Dataset:
    """
    Load a multi-person group chat dataset from JSON file.
    
    Args:
        file_path: Path to the JSON file (e.g., dataset/004/dialogue.json)
        dataset_name: Optional name for the dataset (defaults to file stem)
        max_days: Optional limit on number of days to load
        
    Returns:
        Dataset object with parsed data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    # Determine dataset name
    if dataset_name is None:
        # Extract from path: dataset/004/dialogue.json -> groupchat_004
        parent_name = file_path.parent.name
        dataset_name = f"groupchat_{parent_name}"
    
    print(f"Loading dataset from: {file_path}")
    
    # Load JSON
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Validate structure
    if "dialogues" not in data:
        raise ValueError(f"Invalid format: missing 'dialogues' key in {file_path}")
    
    dialogues = data["dialogues"]
    
    # Parse days (sorted by date)
    days: List[GroupChatDay] = []
    sorted_dates = sorted(dialogues.keys())
    
    # Apply max_days limit if specified
    if max_days is not None:
        sorted_dates = sorted_dates[:max_days]
    
    for date_str in sorted_dates:
        groups_data = dialogues[date_str]
        groups = {}
        
        for group_name, messages_data in groups_data.items():
            messages = []
            for msg_data in messages_data:
                message = _parse_message(msg_data, group_name, date_str)
                messages.append(message)
            
            groups[group_name] = messages
        
        day = GroupChatDay(
            date=date_str,
            groups=groups,
            metadata={}
        )
        days.append(day)
    
    dataset = Dataset(
        name=dataset_name,
        days=days,
        metadata={
            "source_file": str(file_path),
            "total_days": len(days),
            "total_messages": sum(day.total_messages for day in days),
        }
    )
    
    print(f"  Loaded: {dataset.total_days} days, {dataset.total_messages} messages")
    if dataset.days:
        print(f"  Date range: {dataset.date_range[0]} to {dataset.date_range[1]}")
    
    return dataset


def _parse_message(
    msg_data: dict,
    group_name: str,
    date_str: str
) -> GroupChatMessage:
    """
    Parse a single message from raw JSON data.
    
    Args:
        msg_data: Raw message dict with speaker, time, dialogue
        group_name: Name of the group this message belongs to
        date_str: Date string for the message
        
    Returns:
        GroupChatMessage object
    """
    speaker = msg_data.get("speaker", "Unknown")
    content = msg_data.get("dialogue", "") or msg_data.get("text", "")
    time_str = msg_data.get("time", "")
    
    # Parse timestamp
    timestamp = _parse_timestamp(time_str, date_str)
    
    # Extract additional metadata
    metadata = {}
    for key in msg_data:
        if key not in ("speaker", "dialogue", "text", "time"):
            metadata[key] = msg_data[key]
    
    return GroupChatMessage(
        speaker=speaker,
        content=content,
        timestamp=timestamp,
        group=group_name,
        date=date_str,
        metadata=metadata
    )


def _parse_timestamp(time_str: str, date_str: str) -> datetime:
    """
    Parse timestamp string to datetime object.
    
    Supports formats:
    - "2025-01-09 09:32:15"
    - "2025-01-09T09:32:15"
    
    Args:
        time_str: Time string to parse
        date_str: Fallback date if time parsing fails
        
    Returns:
        datetime object
    """
    if not time_str or not isinstance(time_str, str):
        # Fallback: use date with midnight time
        return datetime.strptime(date_str, "%Y-%m-%d")
    
    time_str = time_str.strip()
    
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S.%f",
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(time_str, fmt)
        except ValueError:
            continue
    
    # Fallback: use date with midnight time
    return datetime.strptime(date_str, "%Y-%m-%d")


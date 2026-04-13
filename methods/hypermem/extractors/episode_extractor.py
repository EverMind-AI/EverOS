"""
Episode Extraction Base Classes

Provides the base class and data structures for detecting episode boundaries
in various types of content (conversations, emails, notes, etc.).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import re

from hypermem.llm.llm_provider import LLMProvider
from hypermem.types import RawDataType, Episode


iso_pattern = r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}'


@dataclass
class RawData:
    """Raw data structure for storing original content."""
    content: dict[str, Any]
    data_id: str
    data_type: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def _serialize_value(self, value: Any) -> Any:
        if isinstance(value, datetime):
            return value.isoformat()
        elif isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            return [self._serialize_value(item) for item in value]
        elif hasattr(value, '__dict__'):
            return self._serialize_value(value.__dict__)
        else:
            return value

    def _deserialize_value(self, value: Any, field_name: str = "") -> Any:
        if isinstance(value, str):
            if self._is_datetime_field(field_name) and self._is_iso_datetime(value):
                try:
                    from hypermem.utils.datetime_utils import from_iso_format
                    return from_iso_format(value)
                except (ValueError, ImportError):
                    return value
            return value
        elif isinstance(value, dict):
            return {k: self._deserialize_value(v, k) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._deserialize_value(item, field_name) for item in value]
        else:
            return value

    def _is_datetime_field(self, field_name: str) -> bool:
        if not isinstance(field_name, str):
            return False

        exact_datetime_fields = {
            'timestamp', 'createTime', 'updateTime', 'create_time', 'update_time',
            'sent_timestamp', 'received_timestamp', 'create_timestamp', 'last_update_timestamp',
            'modify_timestamp', 'readUpdateTime', 'created_at', 'updated_at',
            'joinTime', 'leaveTime', 'lastOnlineTime', 'sync_time', 'processed_at',
            'start_time', 'end_time', 'event_time', 'build_timestamp', 'datetime',
            'created', 'updated'
        }

        field_lower = field_name.lower()

        if field_name in exact_datetime_fields or field_lower in exact_datetime_fields:
            return True

        exclusions = {
            'runtime', 'timeout', 'timeline', 'timestamp_format', 'time_zone',
            'time_limit', 'timestamp_count', 'timestamp_enabled', 'time_sync',
            'playtime', 'lifetime', 'uptime', 'downtime'
        }

        if field_name in exclusions or field_lower in exclusions:
            return False

        time_suffixes = ['_time', '_timestamp', '_at', '_date']
        for suffix in time_suffixes:
            if field_name.endswith(suffix) or field_lower.endswith(suffix):
                return True

        if field_name.endswith('Time') and not field_name.endswith('runtime'):
            return True

        if field_name.endswith('Timestamp'):
            return True

        return False

    def _is_iso_datetime(self, value: str) -> bool:
        if not isinstance(value, str) or len(value) < 19:
            return False
        return bool(re.match(iso_pattern, value))

    def to_json(self) -> str:
        try:
            data = {
                'content': self._serialize_value(self.content),
                'data_id': self.data_id,
                'data_type': self.data_type,
                'metadata': self._serialize_value(self.metadata) if self.metadata else None
            }
            return json.dumps(data, ensure_ascii=False, separators=(',', ':'))
        except (TypeError, ValueError) as e:
            raise ValueError(f"Cannot serialize RawData to JSON: {e}") from e

    @classmethod
    def from_json_str(cls, json_str: str) -> 'RawData':
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}") from e

        if not isinstance(data, dict):
            raise ValueError("JSON must be an object")

        if 'content' not in data or 'data_id' not in data:
            raise ValueError("JSON missing required fields: content and data_id")

        instance = cls.__new__(cls)
        instance.content = instance._deserialize_value(data['content'], 'content')
        instance.data_id = data['data_id']
        instance.data_type = data.get('data_type')
        instance.metadata = instance._deserialize_value(data.get('metadata'), 'metadata') if data.get('metadata') else None

        return instance


@dataclass
class EpisodeExtractRequest:
    history_raw_data_list: List[RawData]
    new_raw_data_list: List[RawData]
    user_id_list: List[str]
    smart_mask_flag: Optional[bool] = False


@dataclass
class StatusResult:
    """Status control result."""
    should_wait: bool


class EpisodeExtractor(ABC):
    def __init__(self, raw_data_type: RawDataType, llm_provider=LLMProvider, **llm_kwargs):
        self.raw_data_type = raw_data_type
        self.llm_kwargs = llm_kwargs
        self._llm_provider = llm_provider

    @abstractmethod
    async def extract_episode(self, request: EpisodeExtractRequest) -> tuple[Optional[Episode], Optional[StatusResult]]:
        pass

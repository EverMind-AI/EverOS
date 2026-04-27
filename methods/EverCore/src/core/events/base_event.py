# -*- coding: utf-8 -*-
"""
Base event module

Provides the base abstract class for events, supporting JSON and BSON serialization/deserialization.
All business events should inherit from this base class.
"""

import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Type, TypeVar

import bson

from common_utils.datetime_utils import get_now_with_timezone, to_iso_format


T = TypeVar('T', bound='BaseEvent')


@dataclass
class BaseEvent(ABC):
    """
    Base event class

    All business events should inherit from this class. Subclasses need to define their own business fields,
    and can optionally override the `event_type` method to customize the event type name.

    The base class provides the following features:
    - Automatically generates event ID (event_id)
    - Automatically records event creation time (created_at)
    - JSON serialization/deserialization
    - BSON serialization/deserialization

    Attributes:
        event_id: Unique identifier for the event, automatically generated
        created_at: Event creation time (ISO format string), automatically generated

    Example:
        >>> @dataclass
        ... class UserCreatedEvent(BaseEvent):
        ...     user_id: str
        ...     username: str
        ...
        >>> event = UserCreatedEvent(user_id="123", username="alice")
        >>> print(event.event_type())  # "UserCreatedEvent"
        >>> json_str = event.to_json_str()
        >>> restored = UserCreatedEvent.from_json_str(json_str)
    """

    # Base class fields, using field to provide default factories
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(
        default_factory=lambda: to_iso_format(get_now_with_timezone())
    )

    @classmethod
    def event_type(cls) -> str:
        """
        Get the event type name

        Returns the class name by default. Subclasses can override this method to customize the event type name.

        Returns:
            str: Event type name
        """
        return cls.__name__

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert object to a serializable dictionary

        Note: Automatically adds the `_event_type` field, used during deserialization to determine the specific event type.

        Returns:
            Dict[str, Any]: Dictionary representation of the object
        """
        data = asdict(self)
        # Add event type field for identifying the event type during deserialization
        data['_event_type'] = self.event_type()
        return data

    @classmethod
    @abstractmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """
        Create an object instance from a dictionary

        Subclasses must implement this method to support deserialization.

        Args:
            data: Dictionary containing event data

        Returns:
            Event object instance

        Raises:
            KeyError: Missing required fields
            TypeError: Incorrect field types
        """
        pass

    def to_json_str(self) -> str:
        """
        Serialize object to JSON string

        Returns:
            str: JSON string
        """
        return json.dumps(self.to_dict(), ensure_ascii=False)

    def to_bson_bytes(self) -> bytes:
        """
        Serialize object to BSON bytes

        Returns:
            bytes: BSON byte data
        """
        return bson.encode(self.to_dict())

    @classmethod
    def from_json_str(cls: Type[T], json_str: str) -> T:
        """
        Deserialize object instance from JSON string

        Args:
            json_str: JSON string

        Returns:
            Event object instance

        Raises:
            ValueError: Invalid JSON format or data
        """
        try:
            data = json.loads(json_str)
            return cls.from_dict(data)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            raise ValueError(f"Invalid JSON data: {e}") from e

    @classmethod
    def from_bson_bytes(cls: Type[T], bson_bytes: bytes) -> T:
        """
        Deserialize object instance from BSON bytes

        Args:
            bson_bytes: BSON byte data

        Returns:
            Event object instance

        Raises:
            ValueError: Invalid BSON format or data
        """
        try:
            data = bson.decode(bson_bytes)
            return cls.from_dict(data)
        except Exception as e:
            raise ValueError(f"Invalid BSON data: {e}") from e

    def __repr__(self) -> str:
        """Return string representation of the object"""
        return f"{self.__class__.__name__}(event_id={self.event_id!r}, created_at={self.created_at!r})"

"""
Redis group queue item interface

Defines a standard interface for items stored in the queue, supporting JSON and BSON serialization and deserialization.
"""

import json
from abc import ABC, abstractmethod
from typing import Any, Dict
from enum import Enum
import bson


class SerializationMode(Enum):
    """Serialization mode enumeration"""

    JSON = "json"  # JSON string serialization
    BSON = "bson"  # BSON bytes serialization


class RedisGroupQueueItem(ABC):
    """Redis group queue item interface"""

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert object to dictionary

        Returns:
            Dict[str, Any]: Dictionary representation of the object
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_json_str(cls, json_str: str) -> 'RedisGroupQueueItem':
        """
        Create an object instance from a JSON string

        Args:
            json_str: JSON string

        Returns:
            RedisGroupQueueItem: Object instance

        Raises:
            ValueError: Invalid JSON format or data
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_bson_bytes(cls, bson_bytes: bytes) -> 'RedisGroupQueueItem':
        """
        Deserialize object from BSON bytes

        Args:
            bson_bytes: BSON byte data

        Returns:
            RedisGroupQueueItem: Object instance

        Raises:
            ValueError: Invalid BSON format or data
        """
        raise NotImplementedError

    def to_json_str(self) -> str:
        """
        Convert object to JSON string

        Returns:
            str: JSON string
        """
        return json.dumps(self.to_dict(), ensure_ascii=False)

    def to_bson_bytes(self) -> bytes:
        """
        Serialize object to BSON byte data

        Returns:
            bytes: BSON byte data
        """
        return bson.encode(self.to_dict())


class SimpleQueueItem(RedisGroupQueueItem):
    """Simple queue item implementation example"""

    def __init__(self, data: Any, item_type: str = "simple"):
        """
        Initialize simple queue item

        Args:
            data: Data content
            item_type: Item type identifier
        """
        self.data = data
        self.item_type = item_type

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {"data": self.data, "item_type": self.item_type}

    @classmethod
    def from_json_str(cls, json_str: str) -> 'SimpleQueueItem':
        """Create instance from JSON string"""
        try:
            json_dict = json.loads(json_str)
            return cls(
                data=json_dict["data"], item_type=json_dict.get("item_type", "simple")
            )
        except (json.JSONDecodeError, KeyError) as e:
            raise ValueError(f"Invalid JSON data: {e}") from e

    @classmethod
    def from_bson_bytes(cls, bson_bytes: bytes) -> 'SimpleQueueItem':
        """Create instance from BSON bytes"""
        try:
            data = bson.decode(bson_bytes)
            return cls(data=data["data"], item_type=data.get("item_type", "simple"))
        except (Exception, KeyError) as e:
            raise ValueError(f"Invalid BSON data: {e}") from e

    def __repr__(self) -> str:
        return f"SimpleQueueItem(data={self.data}, item_type={self.item_type})"

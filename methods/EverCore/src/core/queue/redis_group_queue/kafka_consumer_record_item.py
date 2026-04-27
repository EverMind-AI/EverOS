"""
Kafka ConsumerRecord queue item implementation

Provides serialization/deserialization between ConsumerRecord and RedisGroupQueueItem
Uses BSON format to handle binary data, ensuring data integrity
"""

import json
import base64
from typing import Optional, Sequence, Tuple, Any, Dict
from dataclasses import dataclass

import bson

from aiokafka import ConsumerRecord
from core.observation.logger import get_logger
from .redis_group_queue_item import RedisGroupQueueItem

logger = get_logger(__name__)


@dataclass
class KafkaConsumerRecordItem(RedisGroupQueueItem):
    """
    Kafka ConsumerRecord queue item

    Implements the RedisGroupQueueItem interface, providing serialization/deserialization functionality for ConsumerRecord
    """

    # ConsumerRecord fields
    topic: str
    partition: int
    offset: int
    timestamp: int
    timestamp_type: int
    key: Optional[str]
    value: Optional[Any]
    checksum: Optional[int]
    serialized_key_size: int
    serialized_value_size: int
    headers: Sequence[Tuple[str, bytes]]

    def __init__(self, consumer_record: ConsumerRecord):
        """
        Initialize from ConsumerRecord

        Args:
            consumer_record: aiokafka ConsumerRecord object
        """
        self.topic = consumer_record.topic
        self.partition = consumer_record.partition
        self.offset = consumer_record.offset
        self.timestamp = consumer_record.timestamp
        self.timestamp_type = consumer_record.timestamp_type
        # Handle key: if it's bytes, convert to base64 string
        self.key = (
            self._encode_bytes_to_base64(consumer_record.key)
            if isinstance(consumer_record.key, bytes)
            else consumer_record.key
        )
        # Handle value: keep original format, BSON can directly handle various types
        self.value = consumer_record.value
        self.checksum = consumer_record.checksum
        self.serialized_key_size = consumer_record.serialized_key_size
        self.serialized_value_size = consumer_record.serialized_value_size
        # Convert headers to serializable format, preserving binary data
        self.headers = [
            (
                name,
                (
                    self._encode_bytes_to_base64(data)
                    if isinstance(data, bytes)
                    else str(data)
                ),
            )
            for name, data in consumer_record.headers
        ]

    def _encode_bytes_to_base64(self, data: bytes) -> str:
        """Encode bytes data to base64 string"""
        return base64.b64encode(data).decode('utf-8')

    def _decode_base64_to_bytes(self, data: str) -> bytes:
        """Decode base64 string to bytes data"""
        return base64.b64decode(data.encode('utf-8'))

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert object to JSON-serializable dictionary

        Returns:
            Dict[str, Any]: Serializable dictionary
        """
        return {
            "topic": self.topic,
            "partition": self.partition,
            "offset": self.offset,
            "timestamp": self.timestamp,
            "timestamp_type": self.timestamp_type,
            "key": self.key,
            "value": self.value,
            "checksum": self.checksum,
            "serialized_key_size": self.serialized_key_size,
            "serialized_value_size": self.serialized_value_size,
            "headers": self.headers,
        }

    def to_json_str(self) -> str:
        """
        JSON serialization is not supported, please use BSON serialization
        """
        raise NotImplementedError(
            "KafkaConsumerRecordItem does not support JSON serialization, please use to_bson_bytes() method"
        )

    def to_bson_bytes(self) -> bytes:
        """
        Serialize to BSON byte data

        Returns:
            bytes: BSON byte data
        """
        try:
            data = self.to_dict()  # Get serializable dictionary
            return bson.encode(data)
        except Exception as e:
            logger.error("BSON serialization of KafkaConsumerRecordItem failed: %s", e)
            raise ValueError(f"BSON serialization failed: {e}") from e

    @classmethod
    def from_json_str(cls, json_str: str) -> 'KafkaConsumerRecordItem':
        """
        JSON deserialization is not supported, please use BSON deserialization
        """
        raise NotImplementedError(
            "KafkaConsumerRecordItem does not support JSON deserialization, please use from_bson_bytes() method"
        )

    @classmethod
    def from_bson_bytes(cls, bson_bytes: bytes) -> 'KafkaConsumerRecordItem':
        """
        Deserialize from BSON byte data

        Args:
            bson_bytes: BSON byte data

        Returns:
            KafkaConsumerRecordItem: Deserialized object
        """
        try:
            data = bson.decode(bson_bytes)

            # Create instance
            item = cls.__new__(cls)  # Bypass __init__
            item.topic = data["topic"]
            item.partition = data["partition"]
            item.offset = data["offset"]
            item.timestamp = data["timestamp"]
            item.timestamp_type = data["timestamp_type"]
            item.key = data["key"]
            item.value = data["value"]
            item.checksum = data["checksum"]
            item.serialized_key_size = data["serialized_key_size"]
            item.serialized_value_size = data["serialized_value_size"]
            item.headers = data["headers"]

            return item
        except Exception as e:
            logger.error(
                "BSON deserialization of KafkaConsumerRecordItem failed: %s", e
            )
            raise ValueError(f"BSON deserialization failed: {e}") from e

    def to_consumer_record(self) -> ConsumerRecord:
        """
        Convert to aiokafka ConsumerRecord object

        Returns:
            ConsumerRecord: aiokafka ConsumerRecord object
        """
        try:
            # Handle key: if it's a string, try to decode from base64, otherwise keep as is
            key = self.key
            if isinstance(key, str):
                try:
                    key = self._decode_base64_to_bytes(key)
                except Exception:
                    # If decoding fails, keep original string
                    pass

            # Handle headers: decode string data from base64 back to bytes
            headers_bytes = []
            for name, data in self.headers:
                if isinstance(data, str):
                    try:
                        # Try to decode from base64
                        headers_bytes.append((name, self._decode_base64_to_bytes(data)))
                    except Exception:
                        # If decoding fails, encode as UTF-8 bytes
                        headers_bytes.append((name, data.encode('utf-8')))
                else:
                    headers_bytes.append((name, bytes(data)))

            return ConsumerRecord(
                topic=self.topic,
                partition=self.partition,
                offset=self.offset,
                timestamp=self.timestamp,
                timestamp_type=self.timestamp_type,
                key=key,
                value=self.value,
                checksum=self.checksum,
                serialized_key_size=self.serialized_key_size,
                serialized_value_size=self.serialized_value_size,
                headers=headers_bytes,
            )
        except Exception as e:
            logger.error("Failed to convert to ConsumerRecord: %s", e)
            raise ValueError(f"Conversion failed: {e}") from e

    def __repr__(self) -> str:
        return (
            f"KafkaConsumerRecordItem(topic={self.topic}, partition={self.partition}, "
            f"offset={self.offset}, timestamp={self.timestamp})"
        )


def consumer_record_to_queue_item(
    consumer_record: ConsumerRecord,
) -> KafkaConsumerRecordItem:
    """
    Convert ConsumerRecord to queue item

    Args:
        consumer_record: aiokafka ConsumerRecord object

    Returns:
        KafkaConsumerRecordItem: Queue item
    """
    return KafkaConsumerRecordItem(consumer_record)


def queue_item_to_consumer_record(
    queue_item: KafkaConsumerRecordItem,
) -> ConsumerRecord:
    """
    Convert queue item to ConsumerRecord

    Args:
        queue_item: Queue item

    Returns:
        ConsumerRecord: aiokafka ConsumerRecord object
    """
    return queue_item.to_consumer_record()


def serialize_consumer_record_to_bson(consumer_record: ConsumerRecord) -> bytes:
    """
    Serialize ConsumerRecord to BSON byte data

    Args:
        consumer_record: aiokafka ConsumerRecord object

    Returns:
        bytes: BSON serialized byte data
    """
    queue_item = consumer_record_to_queue_item(consumer_record)
    return queue_item.to_bson_bytes()


def deserialize_bson_to_consumer_record(bson_bytes: bytes) -> ConsumerRecord:
    """
    Deserialize BSON byte data to ConsumerRecord

    Args:
        bson_bytes: BSON byte data

    Returns:
        ConsumerRecord: aiokafka ConsumerRecord object
    """
    queue_item = KafkaConsumerRecordItem.from_bson_bytes(bson_bytes)
    return queue_item_to_consumer_record(queue_item)

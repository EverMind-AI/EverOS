"""
Redis Data Processor

Provides unified data serialization and deserialization functionality, supporting:
1. JSON serialization (preferred)
2. Pickle serialization (fallback when JSON fails)
3. Automatic detection of deserialization type
"""

import json
import pickle
import uuid
from typing import Any, Union, Dict, List, Tuple
from core.observation.logger import get_logger

logger = get_logger(__name__)

# Configuration constants
UUID_LENGTH = 8  # UUID truncation length
PICKLE_MARKER = b"__PICKLE__"  # Pickle data marker


class RedisDataProcessor:
    """Redis Data Processor"""

    @staticmethod
    def serialize_data(data: Union[str, Dict, List, Any]) -> Union[str, bytes]:
        """
        Serialize data into string or binary data

        Prefer JSON serialization (returns string), use Pickle serialization (returns binary data) on failure

        Args:
            data: Data to be serialized

        Returns:
            Union[str, bytes]: JSON serialized string or Pickle serialized binary data

        Raises:
            ValueError: Serialization failed
        """
        # If already a string, return directly
        if isinstance(data, str):
            return data

        # Try JSON serialization first
        try:
            return json.dumps(data, ensure_ascii=False)
        except (TypeError, ValueError) as json_error:
            logger.debug(
                "JSON serialization failed, trying Pickle: %s", str(json_error)
            )

            # Use Pickle when JSON fails
            try:
                # Serialize with Pickle and add marker
                pickle_data = pickle.dumps(data)
                # Directly return binary data with marker (Redis is binary-safe)
                binary_data = PICKLE_MARKER + pickle_data
                logger.debug(
                    "Pickle serialization succeeded, data length: %d", len(binary_data)
                )
                return binary_data
            except Exception as pickle_error:
                logger.error("Pickle serialization also failed: %s", str(pickle_error))
                raise ValueError(
                    f"Data serialization failed: JSON error={json_error}, Pickle error={pickle_error}"
                ) from pickle_error

    @staticmethod
    def deserialize_data(data: Union[str, bytes]) -> Any:
        """
        Deserialize data

        Automatically detect whether the data is JSON or Pickle and perform corresponding deserialization

        Args:
            data: Serialized string or binary data

        Returns:
            Any: Deserialized data
        """
        # Handle binary data (from clients with decode_responses=False)
        if isinstance(data, bytes):
            # Check for Pickle marker
            if data.startswith(PICKLE_MARKER):
                logger.debug("Pickle binary data detected, performing deserialization")
                pickle_data = data[len(PICKLE_MARKER) :]
                try:
                    result = pickle.loads(pickle_data)
                    logger.debug("Pickle deserialization succeeded")
                    return result
                except Exception as e:
                    logger.error("Pickle deserialization failed: %s", str(e))
                    return data
            else:
                # Try decoding as string for JSON deserialization
                try:
                    data_str = data.decode('utf-8')
                    return json.loads(data_str)
                except UnicodeDecodeError:
                    logger.warning("Binary data cannot be decoded as UTF-8")
                    return data
                except json.JSONDecodeError:
                    # JSON parsing failed, but UTF-8 decoding succeeded, return decoded string
                    logger.debug(
                        "JSON parsing failed, returning decoded string: %s",
                        data_str[:50],
                    )
                    return data_str

        # Handle string data (from clients with decode_responses=True)
        if isinstance(data, str):
            # Try JSON deserialization
            try:
                return json.loads(data)
            except (json.JSONDecodeError, TypeError) as json_error:
                logger.debug("JSON deserialization failed: %s", str(json_error))
                # Return original string if JSON fails
                return data

        # Return other types directly
        return data

    @staticmethod
    def create_unique_member(data: Union[str, bytes]) -> Union[str, bytes]:
        """
        Create a unique member identifier

        Args:
            data: Serialized data (string or binary)

        Returns:
            Union[str, bytes]: Unique member identifier
        """
        unique_id = str(uuid.uuid4())[:UUID_LENGTH]

        if isinstance(data, bytes):
            # For binary data, use binary separator
            unique_id_bytes = unique_id.encode('utf-8')
            separator = b":"
            return unique_id_bytes + separator + data
        else:
            # For string data, use string format
            return f"{unique_id}:{data}"

    @staticmethod
    def parse_member_data(member: Union[str, bytes]) -> Tuple[str, Union[str, bytes]]:
        """
        Parse member data to extract unique ID and data content

        Args:
            member: Member data (string or binary format: unique_id:data)

        Returns:
            Tuple[str, Union[str, bytes]]: (unique_id, data)
        """
        if isinstance(member, bytes):
            # Handle binary data (from clients with decode_responses=False)
            separator = b":"
            if separator in member:
                unique_id_bytes, data = member.split(separator, 1)
                try:
                    unique_id = unique_id_bytes.decode('utf-8')
                except UnicodeDecodeError:
                    unique_id = "unknown"
            else:
                # Compatible with old format
                unique_id = "unknown"
                data = member
        else:
            # Handle string data (from clients with decode_responses=True)
            if ':' in member:
                unique_id, data = member.split(':', 1)
            else:
                # Compatible with old format
                unique_id = "unknown"
                data = member

        return unique_id, data

    @staticmethod
    def process_data_for_storage(
        data: Union[str, Dict, List, Any]
    ) -> Union[str, bytes]:
        """
        Process data for storage

        Serialize data and create unique identifier

        Args:
            data: Data to be processed

        Returns:
            Union[str, bytes]: Storable unique member data
        """
        serialized_data = RedisDataProcessor.serialize_data(data)
        return RedisDataProcessor.create_unique_member(serialized_data)

    @staticmethod
    def process_data_from_storage(member: Union[str, bytes]) -> Dict[str, Any]:
        """
        Process data read from storage

        Parse member data and deserialize

        Args:
            member: Member data read from Redis (string or binary)

        Returns:
            Dict[str, Any]: Dictionary containing parsed results, format:
                {
                    "id": str,                    # Unique identifier
                    "data": Any,                  # Deserialized original data
                    "raw_data": Union[str, bytes] # Serialized data
                }
        """
        # Redis now returns bytes, need to process first
        if isinstance(member, bytes):
            logger.debug("Binary data read from Redis, length: %d", len(member))

        unique_id, raw_data = RedisDataProcessor.parse_member_data(member)

        try:
            parsed_data = RedisDataProcessor.deserialize_data(raw_data)
        except Exception as e:
            logger.warning(
                "Failed to deserialize data: member=%s, error=%s",
                (
                    str(member)[:100]
                    if isinstance(member, str)
                    else f"bytes({len(member)})"
                ),
                str(e),
            )
            # Return raw data if deserialization fails
            parsed_data = raw_data

        return {"id": unique_id, "data": parsed_data, "raw_data": raw_data}


# For convenience, provide module-level functions
def serialize_data(data: Union[str, Dict, List, Any]) -> Union[str, bytes]:
    """Serialize data (module-level function)"""
    return RedisDataProcessor.serialize_data(data)


def deserialize_data(data: Union[str, bytes]) -> Any:
    """Deserialize data (module-level function)"""
    return RedisDataProcessor.deserialize_data(data)


def create_unique_member(data: Union[str, bytes]) -> Union[str, bytes]:
    """Create unique member identifier (module-level function)"""
    return RedisDataProcessor.create_unique_member(data)


def parse_member_data(member: Union[str, bytes]) -> Tuple[str, Union[str, bytes]]:
    """Parse member data (module-level function)"""
    return RedisDataProcessor.parse_member_data(member)


def process_data_for_storage(data: Union[str, Dict, List, Any]) -> Union[str, bytes]:
    """Process data for storage (module-level function)"""
    return RedisDataProcessor.process_data_for_storage(data)


def process_data_from_storage(member: Union[str, bytes]) -> Dict[str, Any]:
    """Process data read from storage (module-level function)"""
    return RedisDataProcessor.process_data_from_storage(member)

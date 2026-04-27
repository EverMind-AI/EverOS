"""
Kafka Consumer Factory

Provides AIOKafkaConsumer caching and management functionality based on kafka_topic, group_id, and server.
Supports force_new logic to create brand new consumer instances.
"""

import asyncio
import json
import ssl
import os
from typing import Dict, List, Optional, Any
from hashlib import md5

import bson
from aiokafka import AIOKafkaConsumer

from core.component.config_provider import ConfigProvider
from core.di.decorators import component
from core.observation.logger import get_logger
from common_utils.project_path import CURRENT_DIR
from common_utils.datetime_utils import from_iso_format, to_timestamp
from core.di.utils import get_bean_by_type

logger = get_logger(__name__)


def get_ca_file_path(ca_file_path: str) -> Optional[str]:
    """
    Get the full path of the CA certificate file

    Construct the path based on CURRENT_DIR + /config/kafka/ca/

    Returns:
        Optional[str]: CA certificate file path, returns None if it does not exist
    """
    # CURRENT_DIR points to the src directory, go up one level to the project root, then into the config directory
    ca_full_path = CURRENT_DIR / "config" / ca_file_path

    if ca_full_path.exists():
        logger.info("Using default CA certificate file: %s", ca_full_path)
        return str(ca_full_path)
    else:
        logger.warning("Default CA certificate file does not exist: %s", ca_full_path)
        return None


def get_default_kafka_config(env_prefix: str = "") -> Dict[str, Any]:
    """
    Get default Kafka configuration based on environment variables

    Args:
        env_prefix: Environment variable prefix to distinguish different configurations (e.g., "PRODUCER_" or "")
                   The prefix is prepended to KAFKA_, for example, "PRODUCER_" + "KAFKA_SERVERS" = "PRODUCER_KAFKA_SERVERS"

    Environment variables (using prefix=PRODUCER_ as an example):
    - {prefix}KAFKA_SERVERS: Kafka server list, comma-separated
    - {prefix}KAFKA_TOPIC: Kafka topic
    - {prefix}KAFKA_GROUP_ID: Consumer group ID
    - {prefix}MAX_POLL_INTERVAL_MS: Maximum poll interval (milliseconds)
    - {prefix}SESSION_TIMEOUT_MS: Session timeout (milliseconds)
    - {prefix}HEARTBEAT_INTERVAL_MS: Heartbeat interval (milliseconds)
    - {prefix}CA_FILE_PATH: CA certificate file path

    Returns:
        Dict[str, Any]: Configuration dictionary
    """

    def get_env(key: str, default: str = "") -> str:
        """Get environment variable with prefix"""
        return os.getenv(f"{env_prefix}{key}", default)

    # Get environment variables with default values
    kafka_servers_str = get_env("KAFKA_SERVERS", "")
    kafka_servers = [server.strip() for server in kafka_servers_str.split(",")]

    kafka_topic = get_env("KAFKA_TOPIC", "test_topic")
    kafka_group_id = get_env("KAFKA_GROUP_ID", "test_group")
    max_poll_interval_ms = int(get_env("MAX_POLL_INTERVAL_MS", "3600000"))
    session_timeout_ms = int(get_env("SESSION_TIMEOUT_MS", "10000"))
    heartbeat_interval_ms = int(get_env("HEARTBEAT_INTERVAL_MS", "3000"))

    # Handle CA certificate path
    ca_file_path = None
    ca_file_env = get_env("CA_FILE_PATH")
    if ca_file_env:
        ca_file_path = get_ca_file_path(ca_file_env)

    config = {
        "kafka_servers": kafka_servers,
        "kafka_topic": kafka_topic,
        "kafka_group_id": kafka_group_id,
        "max_poll_interval_ms": max_poll_interval_ms,
        "session_timeout_ms": session_timeout_ms,
        "heartbeat_interval_ms": heartbeat_interval_ms,
        "ca_file_path": ca_file_path,
        "auto_offset_reset": "earliest",
        "enable_auto_commit": True,
    }

    prefix_info = f" (prefix: {env_prefix})" if env_prefix else ""
    logger.info("Get default Kafka configuration%s:", prefix_info)
    logger.info("  Servers: %s", kafka_servers)
    logger.info("  Topic: %s", kafka_topic)
    logger.info("  Group ID: %s", kafka_group_id)
    logger.info("  Max poll interval: %s ms", max_poll_interval_ms)
    logger.info("  Session timeout: %s ms", session_timeout_ms)
    logger.info("  Heartbeat interval: %s ms", heartbeat_interval_ms)
    logger.info("  CA certificate: %s", ca_file_path or "None")

    return config


def get_cache_key(
    kafka_servers: List[str], kafka_topic: str, kafka_group_id: str
) -> str:
    """
    Generate cache key
    Create a unique identifier based on servers, topic, and group_id

    Args:
        kafka_servers: Kafka server list
        kafka_topic: Kafka topic
        kafka_group_id: Consumer group ID

    Returns:
        str: Cache key
    """
    servers_str = ",".join(sorted(kafka_servers))
    key_content = f"{servers_str}:{kafka_topic}:{kafka_group_id}"
    return md5(key_content.encode()).hexdigest()


def get_consumer_name(kafka_topic: str, kafka_group_id: str) -> str:
    """
    Get consumer name

    Args:
        kafka_topic: Kafka topic
        kafka_group_id: Consumer group ID

    Returns:
        str: Consumer name
    """
    # Use hyphens to join multiple topic names to avoid overly long names
    topic_str = "-".join(topic.strip() for topic in kafka_topic.split(","))
    return f"{topic_str}.{kafka_group_id}"


def bson_json_decode(value: bytes | None) -> Any:
    """
    BSON/JSON decoder
    Attempt BSON decoding first, fall back to JSON decoding if it fails
    """
    if not value or value == b"null":
        return value
    try:
        return bson.decode(value)
    except Exception:
        try:
            return json.loads(value.decode("utf-8"))
        except Exception as e:
            logger.error("JSON parsing error: %s", e)
            return value


@component(name="kafka_consumer_factory", primary=True)
class KafkaConsumerFactory:
    """
    Kafka Consumer Factory
    ### AIOKafkaConsumer is stateful, so the same instance cannot be used in multiple places ###

    Provides caching and management of AIOKafkaConsumer instances based on configuration
    Supports the force_new parameter to create completely new consumer instances
    """

    def __init__(self):
        """Initialize Kafka Consumer Factory"""
        self._consumers: Dict[str, AIOKafkaConsumer] = {}
        self._lock = asyncio.Lock()
        logger.info("KafkaConsumerFactory initialized")

    async def create_consumer(
        self,
        kafka_servers: List[str],
        kafka_topic: str,
        kafka_group_id: str,
        ca_file_path: Optional[str] = None,
        max_poll_interval_ms: int = 300000,
        session_timeout_ms: int = 10000,
        heartbeat_interval_ms: int = 3000,
        auto_offset_reset: str = "earliest",
        enable_auto_commit: bool = True,
    ) -> AIOKafkaConsumer:
        """
        Create an AIOKafkaConsumer instance

        Args:
            kafka_servers: Kafka server list
            kafka_topic: Kafka topic
            kafka_group_id: Consumer group ID
            ca_file_path: CA certificate file path
            max_poll_interval_ms: Maximum poll interval (milliseconds)
            session_timeout_ms: Session timeout (milliseconds)
            heartbeat_interval_ms: Heartbeat interval (milliseconds)
            auto_offset_reset: Auto offset reset strategy
            enable_auto_commit: Whether to enable auto commit

        Returns:
            AIOKafkaConsumer instance
        """
        # Create SSL context
        ssl_context = None
        if ca_file_path:
            config_provider = get_bean_by_type(ConfigProvider)
            ca_file_content = config_provider.get_raw_config(ca_file_path)
            ssl_context = ssl.create_default_context(purpose=ssl.Purpose.SERVER_AUTH)
            ssl_context.load_verify_locations(cadata=ca_file_content)

        # Handle multiple topics (comma-separated)
        topics = [topic.strip() for topic in kafka_topic.split(",")]

        # Create AIOKafkaConsumer
        consumer = AIOKafkaConsumer(
            *topics,  # Unpack topics list as individual arguments
            bootstrap_servers=kafka_servers,
            group_id=kafka_group_id,
            auto_offset_reset=auto_offset_reset,
            enable_auto_commit=enable_auto_commit,
            key_deserializer=lambda k: k.decode("utf-8") if k else None,
            value_deserializer=bson_json_decode,
            security_protocol="SSL" if ca_file_path else "PLAINTEXT",
            ssl_context=ssl_context,
            max_poll_interval_ms=max_poll_interval_ms,
            session_timeout_ms=session_timeout_ms,
            heartbeat_interval_ms=heartbeat_interval_ms,
        )

        consumer_name = get_consumer_name(kafka_topic, kafka_group_id)
        logger.info("Created AIOKafkaConsumer for %s", consumer_name)
        return consumer

    async def get_consumer(
        self,
        kafka_servers: List[str],
        kafka_topic: str,
        kafka_group_id: str,
        force_new: bool = False,
        **kwargs,
    ) -> AIOKafkaConsumer:
        """
        Get an AIOKafkaConsumer instance

        Args:
            kafka_servers: Kafka server list
            kafka_topic: Kafka topic
            kafka_group_id: Consumer group ID
            force_new: Whether to force creation of a new instance, default is False
            **kwargs: Additional configuration parameters

        Returns:
            AIOKafkaConsumer instance
        """
        cache_key = get_cache_key(kafka_servers, kafka_topic, kafka_group_id)
        consumer_name = get_consumer_name(kafka_topic, kafka_group_id)

        async with self._lock:
            # If forcing creation of a new instance, or it's not in cache
            if force_new or cache_key not in self._consumers:
                logger.info(
                    "Creating new consumer for %s (force_new=%s)",
                    consumer_name,
                    force_new,
                )

                # If forcing creation of a new instance, clean up the old one first
                if force_new and cache_key in self._consumers:
                    old_consumer = self._consumers[cache_key]
                    try:
                        await old_consumer.stop()
                    except Exception as e:
                        logger.error("Error stopping old consumer: %s", e)

                # Create a new consumer instance
                consumer = await self.create_consumer(
                    kafka_servers=kafka_servers,
                    kafka_topic=kafka_topic,
                    kafka_group_id=kafka_group_id,
                    **kwargs,
                )
                self._consumers[cache_key] = consumer

                logger.info(
                    "Consumer %s created and cached with key %s",
                    consumer_name,
                    cache_key,
                )
            else:
                consumer = self._consumers[cache_key]
                logger.debug("Using cached consumer for %s", consumer_name)

        return consumer

    async def get_default_consumer(
        self, force_new: bool = False, env_prefix: str = ""
    ) -> AIOKafkaConsumer:
        """
        Get the default AIOKafkaConsumer instance based on environment variable configuration

        Args:
            force_new: Whether to force creation of a new instance, default is False
            env_prefix: Environment variable prefix, default is "" (compatible with old configurations)
                       For example, env_prefix="CUSTOM_" will read CUSTOM_KAFKA_SERVERS, etc.

        Returns:
            AIOKafkaConsumer instance
        """
        # Always get configuration based on current env_prefix, do not use cache
        # Because different prefixes may correspond to different configurations
        config = get_default_kafka_config(env_prefix=env_prefix)

        return await self.get_consumer(
            kafka_servers=config["kafka_servers"],
            kafka_topic=config["kafka_topic"],
            kafka_group_id=config["kafka_group_id"],
            force_new=force_new,
            ca_file_path=config.get("ca_file_path"),
            max_poll_interval_ms=config.get("max_poll_interval_ms", 20 * 60 * 1000),
            session_timeout_ms=config.get("session_timeout_ms", 10000),
            heartbeat_interval_ms=config.get("heartbeat_interval_ms", 3000),
            auto_offset_reset=config.get("auto_offset_reset", "earliest"),
            enable_auto_commit=True,
        )

    async def remove_consumer(
        self, kafka_servers: List[str], kafka_topic: str, kafka_group_id: str
    ) -> bool:
        """
        Remove the specified consumer

        Args:
            kafka_servers: Kafka server list
            kafka_topic: Kafka topic
            kafka_group_id: Consumer group ID

        Returns:
            bool: Whether the removal was successful
        """
        cache_key = get_cache_key(kafka_servers, kafka_topic, kafka_group_id)
        consumer_name = get_consumer_name(kafka_topic, kafka_group_id)

        async with self._lock:
            if cache_key in self._consumers:
                consumer = self._consumers[cache_key]
                try:
                    await consumer.stop()
                except Exception as e:
                    logger.error("Error stopping consumer during removal: %s", e)

                del self._consumers[cache_key]
                logger.info("Consumer %s removed from cache", consumer_name)
                return True
            else:
                logger.warning("Consumer %s not found in cache", consumer_name)
                return False

    async def clear_all_consumers(self) -> None:
        """Clear all cached consumers"""
        async with self._lock:
            for cache_key, consumer in self._consumers.items():
                try:
                    await consumer.stop()
                except Exception as e:
                    logger.error("Error stopping consumer %s: %s", cache_key, e)

            self._consumers.clear()
            logger.info("All consumers cleared from cache")

    async def seek_to_datetime(
        self, offset_datetime: str, consumer: AIOKafkaConsumer
    ) -> bool:
        """
        Adjust Kafka Consumer's offset based on time format

        Args:
            offset_datetime: Time string, format "2025-09-23 15:21:12"
            consumer: AIOKafkaConsumer instance

        Returns:
            bool: Whether the offset adjustment was successful

        Raises:
            ValueError: Incorrect time format
            RuntimeError: Consumer not started or offset adjustment failed
        """
        try:
            # Parse time string into timezone-aware datetime object
            target_dt = from_iso_format(offset_datetime)
            # Convert to millisecond timestamp (Kafka uses millisecond timestamps)
            target_timestamp_ms = int(to_timestamp(target_dt) * 1000)

            logger.info(
                "Seeking consumer to datetime: %s (timestamp: %d)",
                offset_datetime,
                target_timestamp_ms,
            )

            # Check if consumer is started and get partition assignment
            try:
                # Attempt to get partition assignment, will raise exception if consumer not started
                partitions = consumer.assignment()
                if not partitions:
                    raise RuntimeError(
                        "Consumer has no assigned partitions. Make sure consumer is started and has subscribed to topics."
                    )
            except Exception as e:
                raise RuntimeError(
                    "Consumer must be started before seeking to timestamp"
                ) from e

            # Build timestamp map for each partition
            timestamp_map = {partition: target_timestamp_ms for partition in partitions}

            # Use offsets_for_times to get the offset at the corresponding timestamp
            offset_map = await consumer.offsets_for_times(timestamp_map)

            # Track processing statistics per topic
            topic_stats = {}
            seek_count = 0

            # Process each partition
            for partition in partitions:
                topic_name = partition.topic
                if topic_name not in topic_stats:
                    topic_stats[topic_name] = {
                        'total_partitions': 0,
                        'found_offsets': 0,
                        'used_latest': 0,
                    }
                topic_stats[topic_name]['total_partitions'] += 1

                # Get offset information for this partition
                offset_info = offset_map.get(partition) if offset_map else None

                if offset_info is not None:
                    # Found offset at the specified timestamp
                    target_offset = offset_info.offset
                    consumer.seek(partition, target_offset)
                    seek_count += 1
                    topic_stats[topic_name]['found_offsets'] += 1
                    logger.info(
                        "Seeked partition %s (topic: %s) to offset %d at timestamp %d",
                        partition,
                        topic_name,
                        target_offset,
                        target_timestamp_ms,
                    )
                else:
                    # No offset found at the specified timestamp, use latest offset
                    logger.warning(
                        "No offset found for partition %s (topic: %s) at timestamp %d, using latest offset",
                        partition,
                        topic_name,
                        target_timestamp_ms,
                    )

                    # Get the latest offset for this partition
                    latest_offset_map = await consumer.end_offsets([partition])
                    latest_offset = latest_offset_map[partition]
                    consumer.seek(partition, latest_offset)
                    seek_count += 1
                    topic_stats[topic_name]['used_latest'] += 1
                    logger.info(
                        "Seeked partition %s (topic: %s) to latest offset %d",
                        partition,
                        topic_name,
                        latest_offset,
                    )

            # Log processing statistics per topic
            for topic_name, stats in topic_stats.items():
                logger.info(
                    "Topic '%s': %d partitions total, %d found timestamp offsets, %d used latest offsets",
                    topic_name,
                    stats['total_partitions'],
                    stats['found_offsets'],
                    stats['used_latest'],
                )

            if seek_count > 0:
                logger.info(
                    "Successfully seeked %d partitions to datetime %s",
                    seek_count,
                    offset_datetime,
                )
                return True
            else:
                logger.warning(
                    "No partitions were seeked for datetime %s", offset_datetime
                )
                return False

        except ValueError as e:
            logger.error("Invalid datetime format '%s': %s", offset_datetime, e)
            raise ValueError(
                f"Invalid datetime format '{offset_datetime}'. Expected format: 'YYYY-MM-DD HH:MM:SS'"
            ) from e

        except Exception as e:
            logger.error(
                "Failed to seek consumer to datetime %s: %s", offset_datetime, e
            )
            raise RuntimeError(
                f"Failed to seek consumer to datetime {offset_datetime}"
            ) from e

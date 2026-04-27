"""
Kafka Producer Factory

Provides AIOKafkaProducer caching and management based on kafka_servers and kafka_topic.
Supports force_new logic to create brand new producer instances.
"""

import asyncio
import json
import os
import ssl
from typing import Dict, List, Optional, Any, Union
from hashlib import md5

import async_timeout
import bson
from aiokafka import AIOKafkaProducer
from aiokafka.producer.message_accumulator import MessageBatch

from core.component.config_provider import ConfigProvider
from core.component.kafka_consumer_factory import get_ca_file_path
from core.di.decorators import component
from core.di.utils import get_bean_by_type
from core.observation.logger import get_logger


# =============================================================================
# Monkey Patches for aiokafka performance and stability improvements
# =============================================================================


async def _optimized_wait_drain(self, timeout=None):
    """
    Optimized wait_drain: avoid asyncio.wait overhead for single Future.

    The original implementation uses asyncio.wait([single_future]) which is
    inefficient - it creates sets, registers/removes callbacks for each call.
    This patch uses async_timeout directly for zero overhead.
    """
    waiter = self._drain_waiter
    try:
        async with async_timeout.timeout(timeout):
            await waiter
    except asyncio.TimeoutError:
        pass
    if waiter.done():
        waiter.result()  # Check for exception


# Save original start method
_original_producer_start = AIOKafkaProducer.start


async def _idempotent_start(self):
    """
    Idempotent start: prevent creating multiple sender tasks.

    The original start() creates a sender task each time it's called.
    Multiple sender tasks sharing the same accumulator causes a busy loop.
    This patch makes start() idempotent - only the first call takes effect.
    """
    if getattr(self, "_memsys_started", False):
        return
    self._memsys_started = True
    await _original_producer_start(self)


# Apply patches
MessageBatch.wait_drain = _optimized_wait_drain
AIOKafkaProducer.start = _idempotent_start

logger = get_logger(__name__)

# Default prefix for producer environment variables
DEFAULT_PRODUCER_ENV_PREFIX = "PRODUCER_"


def get_default_producer_config(
    env_prefix: str = DEFAULT_PRODUCER_ENV_PREFIX,
) -> Dict[str, Any]:
    """
    Get default Kafka Producer configuration based on environment variables

    Args:
        env_prefix: Environment variable prefix, default is "PRODUCER_"
                   e.g., "PRODUCER_" + "KAFKA_SERVERS" = "PRODUCER_KAFKA_SERVERS"

    Environment variables (using prefix=PRODUCER_ as example):
    - {prefix}KAFKA_SERVERS: List of Kafka servers, comma-separated (required)
    - {prefix}CA_FILE_PATH: Path to CA certificate file (optional, for SSL connection)
    - {prefix}ACKS: Acknowledgment mode, default 1
        - 0: No wait for acknowledgment, fastest but may lose messages
        - 1: Leader acknowledgment only, balances performance and reliability
        - all: All replicas must acknowledge, most reliable but slowest
    - {prefix}COMPRESSION_TYPE: Compression type (optional)
        - gzip: High compression ratio, high CPU usage
        - snappy: Medium compression ratio, fast
        - lz4: Low compression ratio, fastest
        - zstd: High compression ratio, relatively fast (recommended)
    - {prefix}LINGER_MS: Send delay (milliseconds), default 0
        - Setting > 0 allows producer to wait for more messages to batch together, improving throughput
    - {prefix}MAX_BATCH_SIZE: Maximum bytes per batch, default 16384 (16KB)
    - {prefix}MAX_REQUEST_SIZE: Maximum bytes per request, default 1048576 (1MB)
    - {prefix}REQUEST_TIMEOUT_MS: Request timeout (milliseconds), default 30000 (30 seconds)

    Returns:
        Dict[str, Any]: Producer configuration dictionary
    """

    def get_env(key: str, default: str = "") -> str:
        """Get environment variable with prefix"""
        return os.getenv(f"{env_prefix}{key}", default)

    # Kafka server addresses (required)
    kafka_servers_str = get_env("KAFKA_SERVERS", "")
    kafka_servers = [
        server.strip() for server in kafka_servers_str.split(",") if server.strip()
    ]

    # Handle CA certificate path (for SSL connection)
    ca_file_path = None
    ca_file_env = get_env("CA_FILE_PATH")
    if ca_file_env:
        ca_file_path = get_ca_file_path(ca_file_env)

    # Producer-specific configurations
    acks_str = get_env("ACKS", "1")
    # Handle acks which might be numeric or string 'all'
    acks: Union[int, str] = acks_str if acks_str == "all" else int(acks_str)

    compression_type = get_env("COMPRESSION_TYPE") or None
    linger_ms = int(get_env("LINGER_MS", "300"))
    max_batch_size = int(get_env("MAX_BATCH_SIZE", "16384"))
    max_request_size = int(get_env("MAX_REQUEST_SIZE", "1048576"))
    request_timeout_ms = int(get_env("REQUEST_TIMEOUT_MS", "30000"))
    retry_backoff_ms = int(get_env("RETRY_BACKOFF_MS", "500"))

    config = {
        "kafka_servers": kafka_servers,
        "ca_file_path": ca_file_path,
        "acks": acks,
        "compression_type": compression_type,
        "linger_ms": linger_ms,
        "max_batch_size": max_batch_size,
        "max_request_size": max_request_size,
        "request_timeout_ms": request_timeout_ms,
        "retry_backoff_ms": retry_backoff_ms,
    }

    prefix_info = f" (prefix: {env_prefix})" if env_prefix else ""
    logger.info("Getting default Kafka Producer configuration%s:", prefix_info)
    logger.info("  Servers: %s", kafka_servers)
    logger.info("  CA certificate: %s", ca_file_path or "None")
    logger.info("  Acknowledgment mode (acks): %s", acks)
    logger.info("  Compression type: %s", compression_type or "None")
    logger.info("  Send delay (linger_ms): %s ms", linger_ms)
    logger.info("  Batch size (max_batch_size): %s bytes", max_batch_size)
    logger.info("  Max request (max_request_size): %s bytes", max_request_size)
    logger.info("  Request timeout (request_timeout_ms): %s ms", request_timeout_ms)
    logger.info("  Retry backoff (retry_backoff_ms): %s ms", retry_backoff_ms)

    return config


def get_producer_cache_key(kafka_servers: List[str], kafka_topic: str = "") -> str:
    """
    Generate Producer cache key
    Create a unique identifier based on servers and optional topic

    Args:
        kafka_servers: List of Kafka servers
        kafka_topic: Kafka topic (optional, producer may send to multiple topics)

    Returns:
        str: Cache key
    """
    servers_str = ",".join(sorted(kafka_servers))
    key_content = f"{servers_str}:{kafka_topic}" if kafka_topic else servers_str
    return md5(key_content.encode()).hexdigest()


def get_producer_name(kafka_servers: List[str], kafka_topic: str = "") -> str:
    """
    Get producer name

    Args:
        kafka_servers: List of Kafka servers
        kafka_topic: Kafka topic

    Returns:
        str: Producer name
    """
    servers_short = kafka_servers[0] if kafka_servers else "unknown"
    if kafka_topic:
        return f"producer-{kafka_topic}@{servers_short}"
    return f"producer@{servers_short}"


def json_serializer(value: Any) -> bytes:
    """
    JSON serializer
    Serialize value into JSON bytes
    """
    if value is None:
        return b"null"
    if isinstance(value, bytes):
        return value
    return json.dumps(value, ensure_ascii=False, default=str).encode("utf-8")


def bson_serializer(value: Any) -> bytes:
    """
    BSON serializer
    Serialize value into BSON bytes
    """
    if value is None:
        return b""
    if isinstance(value, bytes):
        return value
    if isinstance(value, dict):
        return bson.encode(value)
    # Non-dict types need to be wrapped into a dict
    return bson.encode({"data": value})


def bson_json_serializer(value: Any) -> bytes:
    """
    BSON/JSON serializer (default)
    Try BSON serialization first, fall back to JSON on failure
    Compatible with input as bytes (return directly)
    """
    if value is None:
        return b"null"
    if isinstance(value, bytes):
        # Already bytes, return directly (compatible with event directly converted to bson bytes)
        return value
    # Try BSON first
    if isinstance(value, dict):
        try:
            return bson.encode(value)
        except Exception:
            pass
    # Fall back to JSON
    try:
        return json.dumps(value, ensure_ascii=False, default=str).encode("utf-8")
    except Exception as e:
        logger.error("Serialization failed: %s", e)
        raise


def key_serializer(key: Any) -> Optional[bytes]:
    """
    Key serializer
    Serialize key into UTF-8 bytes
    """
    if key is None:
        return None
    if isinstance(key, bytes):
        return key
    return str(key).encode("utf-8")


@component(name="kafka_producer_factory", primary=True)
class KafkaProducerFactory:
    """
    Kafka Producer Factory

    Provides caching and management of AIOKafkaProducer instances based on configuration
    Supports force_new parameter to create brand new producer instances

    Note: AIOKafkaProducer.start() has been patched to be idempotent globally,
    so calling start() multiple times is safe (only the first call takes effect).
    """

    def __init__(self):
        """Initialize Kafka Producer Factory"""
        self._producers: Dict[str, AIOKafkaProducer] = {}
        self._lock = asyncio.Lock()
        logger.info("KafkaProducerFactory initialized")

    async def create_producer(
        self,
        kafka_servers: List[str],
        ca_file_path: Optional[str] = None,
        acks: Union[int, str] = 1,
        compression_type: Optional[str] = None,
        max_batch_size: int = 16384,
        linger_ms: int = 0,
        max_request_size: int = 1048576,
        request_timeout_ms: int = 30000,
        retry_backoff_ms: int = 500,
        value_serializer: Optional[callable] = None,
        start_timeout: float = 10.0,
    ) -> AIOKafkaProducer:
        """
        Create AIOKafkaProducer instance

        Args:
            kafka_servers: List of Kafka servers
            ca_file_path: Path to CA certificate file
            acks: Acknowledgment mode (0, 1, 'all')
            compression_type: Compression type ('gzip', 'snappy', 'lz4', 'zstd')
            max_batch_size: Maximum bytes for batch sending
            linger_ms: Send delay (milliseconds), used for batch sending
            max_request_size: Maximum bytes per request
            request_timeout_ms: Request timeout (milliseconds)
            retry_backoff_ms: Retry backoff (milliseconds)
            value_serializer: Value serializer, default is bson_json_serializer
            start_timeout: Timeout for starting producer (seconds), 0 to skip

        Returns:
            AIOKafkaProducer instance

        Raises:
            ConnectionError: If cannot connect to Kafka within timeout
        """
        # Create SSL context
        ssl_context = None
        if ca_file_path:
            config_provider = get_bean_by_type(ConfigProvider)
            ca_file_content = config_provider.get_raw_config(ca_file_path)
            ssl_context = ssl.create_default_context(purpose=ssl.Purpose.SERVER_AUTH)
            ssl_context.load_verify_locations(cadata=ca_file_content)

        # Use default BSON/JSON serializer
        if value_serializer is None:
            value_serializer = bson_json_serializer

        # Create AIOKafkaProducer
        producer = AIOKafkaProducer(
            bootstrap_servers=kafka_servers,
            key_serializer=key_serializer,
            value_serializer=value_serializer,
            acks=acks,
            compression_type=compression_type,
            max_batch_size=max_batch_size,
            linger_ms=linger_ms,
            max_request_size=max_request_size,
            request_timeout_ms=request_timeout_ms,
            retry_backoff_ms=retry_backoff_ms,
            security_protocol="SSL" if ca_file_path else "PLAINTEXT",
            ssl_context=ssl_context,
        )

        producer_name = get_producer_name(kafka_servers)

        # Start producer and verify connection
        if start_timeout > 0:
            try:
                await asyncio.wait_for(producer.start(), timeout=start_timeout)
                brokers = producer.client.cluster.brokers()
                logger.info(
                    "Producer %s started, connected to %d broker(s)",
                    producer_name,
                    len(brokers) if brokers else 0,
                )
            except asyncio.TimeoutError:
                logger.error(
                    "Cannot connect to Kafka %s within %ss, producer may not work",
                    kafka_servers,
                    start_timeout,
                )
            except Exception as e:
                logger.error("Failed to start Kafka producer %s: %s", producer_name, e)
        else:
            logger.info("Created AIOKafkaProducer for %s (not started)", producer_name)

        return producer

    async def get_producer(
        self,
        kafka_servers: List[str],
        force_new: bool = False,
        test_topic: Optional[str] = None,
        **kwargs,
    ) -> AIOKafkaProducer:
        """
        Get AIOKafkaProducer instance

        Args:
            kafka_servers: List of Kafka servers
            force_new: Whether to force creation of a new instance, default False
            test_topic: If provided, test connection by fetching partitions for this topic
            **kwargs: Additional configuration parameters

        Returns:
            AIOKafkaProducer instance
        """
        cache_key = get_producer_cache_key(kafka_servers)
        producer_name = get_producer_name(kafka_servers)

        async with self._lock:
            # If forcing new instance or not in cache
            if force_new or cache_key not in self._producers:
                logger.info(
                    "Creating new producer for %s (force_new=%s)",
                    producer_name,
                    force_new,
                )

                # If forcing new instance, clean up old one first
                if force_new and cache_key in self._producers:
                    old_producer = self._producers[cache_key]
                    try:
                        await old_producer.stop()
                    except Exception as e:
                        logger.error("Error stopping old producer: %s", e)

                # Create new producer instance
                producer = await self.create_producer(
                    kafka_servers=kafka_servers, **kwargs
                )
                self._producers[cache_key] = producer

                # Test connection if test_topic provided
                if test_topic:
                    try:
                        partitions = await asyncio.wait_for(
                            producer.partitions_for(test_topic), timeout=10.0
                        )
                        logger.info(
                            "Connection test passed: topic %s has %d partitions",
                            test_topic,
                            len(partitions) if partitions else 0,
                        )
                    except Exception as e:
                        logger.error(
                            "Connection test failed for %s: %s", producer_name, e
                        )
                        raise

                logger.info(
                    "Producer %s created and cached with key %s",
                    producer_name,
                    cache_key,
                )
            else:
                producer = self._producers[cache_key]
                logger.debug("Using cached producer for %s", producer_name)

        return producer

    async def get_default_producer(
        self, force_new: bool = False, env_prefix: str = DEFAULT_PRODUCER_ENV_PREFIX
    ) -> AIOKafkaProducer:
        """
        Get default AIOKafkaProducer instance based on environment variable configuration

        Args:
            force_new: Whether to force creation of a new instance, default False
            env_prefix: Environment variable prefix, default "PRODUCER_"
                       e.g., read PRODUCER_KAFKA_SERVERS, etc.

        Returns:
            AIOKafkaProducer instance
        """
        config = get_default_producer_config(env_prefix=env_prefix)

        return await self.get_producer(
            kafka_servers=config["kafka_servers"],
            force_new=force_new,
            ca_file_path=config.get("ca_file_path"),
            acks=config.get("acks", 1),
            compression_type=config.get("compression_type"),
            linger_ms=config.get("linger_ms", 0),
            max_batch_size=config.get("max_batch_size", 16384),
            max_request_size=config.get("max_request_size", 1048576),
            request_timeout_ms=config.get("request_timeout_ms", 30000),
            retry_backoff_ms=config.get("retry_backoff_ms", 500),
        )

    async def remove_producer(self, kafka_servers: List[str]) -> bool:
        """
        Remove specified producer

        Args:
            kafka_servers: List of Kafka servers

        Returns:
            bool: Whether removal was successful
        """
        cache_key = get_producer_cache_key(kafka_servers)
        producer_name = get_producer_name(kafka_servers)

        async with self._lock:
            if cache_key in self._producers:
                producer = self._producers[cache_key]
                try:
                    await producer.stop()
                except Exception as e:
                    logger.error("Error stopping producer during removal: %s", e)

                del self._producers[cache_key]
                logger.info("Producer %s removed from cache", producer_name)
                return True
            else:
                logger.warning("Producer %s not found in cache", producer_name)
                return False

    async def clear_all_producers(self) -> None:
        """Clear all cached producers"""
        async with self._lock:
            for cache_key, producer in self._producers.items():
                try:
                    await producer.stop()
                except Exception as e:
                    logger.error("Error stopping producer %s: %s", cache_key, e)

            self._producers.clear()
            logger.info("All producers cleared from cache")

    async def send(
        self,
        producer: AIOKafkaProducer,
        topic: str,
        value: Any,
        key: Optional[str] = None,
        partition: Optional[int] = None,
        timestamp_ms: Optional[int] = None,
        headers: Optional[List[tuple]] = None,
    ) -> Any:
        """
        Send message to Kafka

        Args:
            producer: AIOKafkaProducer instance
            topic: Target topic
            value: Message value
            key: Message key (optional)
            partition: Target partition (optional)
            timestamp_ms: Timestamp (optional, milliseconds)
            headers: Message headers (optional)

        Returns:
            RecordMetadata object
        """
        try:
            result = await producer.send_and_wait(
                topic=topic,
                value=value,
                key=key,
                partition=partition,
                timestamp_ms=timestamp_ms,
                headers=headers,
            )
            logger.debug(
                "Message sent to topic %s, partition %s, offset %s",
                result.topic,
                result.partition,
                result.offset,
            )
            return result
        except Exception as e:
            logger.error("Failed to send message to topic %s: %s", topic, e)
            raise

    async def send_batch(
        self, producer: AIOKafkaProducer, topic: str, messages: List[Dict[str, Any]]
    ) -> List[Any]:
        """
        Send messages in batch to Kafka

        Args:
            producer: AIOKafkaProducer instance
            topic: Target topic
            messages: List of messages, each message is a dictionary containing value, key (optional), etc.

        Returns:
            List of RecordMetadata objects
        """
        results = []
        for msg in messages:
            result = await self.send(
                producer=producer,
                topic=topic,
                value=msg.get("value"),
                key=msg.get("key"),
                partition=msg.get("partition"),
                timestamp_ms=msg.get("timestamp_ms"),
                headers=msg.get("headers"),
            )
            results.append(result)
        return results

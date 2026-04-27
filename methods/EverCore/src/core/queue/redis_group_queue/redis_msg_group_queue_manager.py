"""
Redis Message Group Queue Manager

Redis-based fixed partition queue manager.
Core features:
1. Fixed 50 partitions, numbered 001-050, not configurable
2. group_key routed to fixed partition via hash
3. Supports concurrent consumption of multiple queues, prevents conflicts using owner mechanism
4. Uses Redis sorted sets (ZSET) to store messages, supports sorting by score and time filtering

⚠️ Warning: Partition count is fixed at 50. Modifying this configuration will cause severe data routing errors and message loss!
"""

import asyncio
import time
import random
import hashlib
from typing import Any, Dict, List, Optional, Tuple, Callable, Type
from dataclasses import dataclass, field
from enum import Enum

import redis.asyncio as redis

from core.observation.logger import get_logger
from common_utils.datetime_utils import get_now_with_timezone, to_iso_format
from core.queue.redis_group_queue.redis_group_queue_item import SimpleQueueItem
from core.queue.redis_group_queue.redis_group_queue_item import (
    RedisGroupQueueItem,
    SerializationMode,
)
from core.queue.redis_group_queue.redis_group_queue_lua_scripts import (
    ENQUEUE_SCRIPT,
    GET_QUEUE_STATS_SCRIPT,
    GET_ALL_PARTITIONS_STATS_SCRIPT,
    REBALANCE_PARTITIONS_SCRIPT,
    JOIN_CONSUMER_SCRIPT,
    EXIT_CONSUMER_SCRIPT,
    KEEPALIVE_CONSUMER_SCRIPT,
    CLEANUP_INACTIVE_OWNERS_SCRIPT,
    FORCE_CLEANUP_SCRIPT,
    GET_MESSAGES_SCRIPT,
)
from core.rate_limit.rate_limiter import rate_limit

logger = get_logger(__name__)


class ShutdownMode(Enum):
    """Shutdown mode enumeration"""

    SOFT = "soft"  # Soft shutdown: Check if messages exist, with delay time control
    HARD = "hard"  # Hard shutdown: Shut down directly, record unprocessed message count


class ManagerState(Enum):
    """Manager state enumeration"""

    CREATED = "created"  # Created, not started
    STARTED = "started"  # Started, running
    SHUTDOWN = "shutdown"  # Shut down, cannot be restarted


@dataclass
class RedisPartitionStats:
    """Redis partition statistics"""

    partition: str
    current_size: int
    min_score: int
    max_score: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "partition": self.partition,
            "current_size": self.current_size,
            "min_score": self.min_score,
            "max_score": self.max_score,
        }


@dataclass
class RedisQueueStats:
    """Redis queue statistics"""

    queue_name: str
    current_size: int
    last_activity_time: float
    min_score: int
    max_score: int
    total_delivered: int = 0
    total_consumed: int = 0
    last_deliver_time: Optional[str] = None
    last_consume_time: Optional[str] = None
    partitions: Optional[List[RedisPartitionStats]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        result = {
            "queue_name": self.queue_name,
            "current_size": self.current_size,
            "last_activity_time": self.last_activity_time,
            "min_score": self.min_score,
            "max_score": self.max_score,
            "total_delivered": self.total_delivered,
            "total_consumed": self.total_consumed,
            "last_deliver_time": self.last_deliver_time,
            "last_consume_time": self.last_consume_time,
        }
        if self.partitions:
            result["partitions"] = [p.to_dict() for p in self.partitions]
        return result


@dataclass
class RedisManagerStats:
    """Redis manager overall statistics"""

    total_queues: int
    total_current_messages: int
    total_delivered_messages: int = 0
    total_consumed_messages: int = 0
    total_rejected_messages: int = 0
    start_time: str = field(
        default_factory=lambda: to_iso_format(get_now_with_timezone())
    )
    uptime_seconds: float = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "total_queues": self.total_queues,
            "total_current_messages": self.total_current_messages,
            "total_delivered_messages": self.total_delivered_messages,
            "total_consumed_messages": self.total_consumed_messages,
            "total_rejected_messages": self.total_rejected_messages,
            "start_time": self.start_time,
            "uptime_seconds": self.uptime_seconds,
        }


class RedisGroupQueueManager:
    """
    Redis message group queue manager (dynamic owner management version)

    Core features:
    1. Manages consumer active status based on owner_activate_time_zset
    2. Each owner has an independent queue_list recording assigned partitions
    3. Supports dynamic rebalance, automatically assigns partitions to active consumers
    4. Consumer join/exit automatically triggers rebalance
    5. Periodically cleans up inactive consumers (default 5 minutes inactive)
    6. Consumer keepalive mechanism (recommended to call every 30 seconds)
    7. Supports forced cleanup and reset
    8. Checks score difference threshold when consuming messages
    9. All operations ensure atomicity through Lua scripts
    """

    # Fixed partition count, configurable but recommended to keep at 50
    FIXED_PARTITION_COUNT = 50

    def __init__(
        self,
        redis_client: redis.Redis,
        key_prefix: str = "default",
        serialization_mode: SerializationMode = SerializationMode.JSON,
        item_class: Type[RedisGroupQueueItem] = None,
        sort_key_func: Optional[Callable[[RedisGroupQueueItem], int]] = None,
        max_total_messages: int = 20000,  # 2w
        queue_expire_seconds: int = 24 * 3600,  # 1 day
        activity_expire_seconds: int = 24 * 3600,  # 1 day
        enable_metrics: bool = True,
        log_interval_seconds: int = 600,  # 10 minutes
        owner_expire_seconds: int = 3600,  # owner expiration time, default 1 hour
        inactive_threshold_seconds: int = 300,  # inactive threshold, default 5 minutes
        cleanup_interval_seconds: int = 300,  # periodic cleanup interval, default 5 minutes
    ):
        """
        Initialize Redis message group queue manager

        Args:
            redis_client: Redis client
            key_prefix: Redis key prefix, used to distinguish different manager instances
            serialization_mode: Serialization mode (JSON or BSON)
            item_class: Queue item type, must inherit from RedisGroupQueueItem, default uses SimpleQueueItem
            sort_key_func: Sort key generation function, receives RedisGroupQueueItem returns int score
            max_total_messages: Maximum total message count limit
            queue_expire_seconds: Queue expiration time (seconds)
            activity_expire_seconds: Activity record expiration time (seconds)
            enable_metrics: Whether to enable statistics
            log_interval_seconds: Log printing interval (seconds)
            owner_expire_seconds: owner expiration time (seconds, default 1 hour)
            inactive_threshold_seconds: Inactive threshold (seconds, default 5 minutes)
            cleanup_interval_seconds: Periodic cleanup interval (seconds, default 5 minutes)
        """
        self.redis_client = redis_client
        self.key_prefix = key_prefix
        self.serialization_mode = serialization_mode
        # Set default item_class to SimpleQueueItem
        if item_class is None:
            self.item_class = SimpleQueueItem
        else:
            self.item_class = item_class
        self.sort_key_func = sort_key_func or self._default_sort_key
        self.max_total_messages = max_total_messages
        self.queue_expire_seconds = queue_expire_seconds
        self.activity_expire_seconds = activity_expire_seconds
        self.enable_metrics = enable_metrics
        self.log_interval_seconds = log_interval_seconds
        self.owner_expire_seconds = owner_expire_seconds
        self.inactive_threshold_seconds = inactive_threshold_seconds
        self.cleanup_interval_seconds = cleanup_interval_seconds

        # Redis key patterns - new dynamic owner management mode
        self.queue_prefix = (
            f"{key_prefix}:queue:"  # Queue key prefix, used in Lua scripts
        )
        self.queue_key_pattern = (
            f"{key_prefix}:queue:{{partition}}"  # partition is 001-050
        )
        self.owner_activate_time_zset_key = (
            f"{key_prefix}:owner_activate_time_zset"  # owner active time zset
        )
        self.queue_list_prefix = (
            f"{key_prefix}:queue_list:"  # owner's queue_list prefix
        )
        self.counter_key = f"{key_prefix}:counter"

        # Process-level owner ID (generated at startup, globally unique)
        self.owner_id = (
            f"{self.key_prefix}_{int(time.time())}_{random.randint(10000, 99999)}"
        )

        # Maintain owner last keepalive timestamp mapping (millisecond timestamp)
        self.owner_last_keepalive_time = {}

        # Generate fixed partition name list: 001, 002, ..., 050
        self.partition_names = [
            f"{i:03d}" for i in range(1, self.FIXED_PARTITION_COUNT + 1)
        ]

        # Manager statistics
        self._manager_stats = RedisManagerStats(
            total_queues=0, total_current_messages=0
        )

        # Async lock, protects statistics
        self._stats_lock = asyncio.Lock()

        # Start time
        self._start_time = time.time()

        # Periodic tasks
        self._log_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False

        # Manager state
        self._state = ManagerState.CREATED

        # Pre-compiled Lua scripts
        self._enqueue_script = None
        self._get_stats_script = None
        self._get_all_partitions_stats_script = None
        self._rebalance_partitions_script = None
        self._join_consumer_script = None
        self._exit_consumer_script = None
        self._keepalive_consumer_script = None
        self._cleanup_inactive_owners_script = None
        self._force_cleanup_script = None
        self._get_messages_script = None

        logger.info(
            "🚀 RedisGroupQueueManager[%s] Initialization completed: key_prefix=%s, max_total_messages=%d",
            self.key_prefix,
            self.key_prefix,
            self.max_total_messages,
        )

    def _default_sort_key(self, _item: RedisGroupQueueItem) -> int:
        """
        Default sort key generation function: Use current timestamp (milliseconds)

        Args:
            item: Queue item

        Returns:
            int: Sort score (millisecond timestamp)
        """
        return int(time.time() * 1000)  # Convert to millisecond integer

    def _hash_group_key_to_partition(self, group_key: str) -> str:
        """
        Route group_key to fixed partition via hash

        Args:
            group_key: Group key

        Returns:
            str: Partition name (001-100)
        """
        # Use MD5 hash to ensure even distribution
        hash_value = hashlib.md5(group_key.encode('utf-8')).hexdigest()
        # Take first 8 characters, convert to integer, then modulo
        partition_index = int(hash_value[:8], 16) % self.FIXED_PARTITION_COUNT
        return self.partition_names[partition_index]

    def _get_queue_key(self, partition: str) -> str:
        """Get queue Redis key"""
        return self.queue_key_pattern.format(partition=partition)

    def _get_queue_list_key(self, owner_id: Optional[str] = None) -> str:
        """Get owner's queue_list Redis key"""
        if owner_id is None:
            owner_id = self.owner_id
        return f"{self.queue_list_prefix}{owner_id}"

    def _parse_rebalance_result(
        self, result: Any, expected_count: int
    ) -> Tuple[bool, Tuple]:
        """
        Parse rebalance-related script return results

        Args:
            result: Lua script return result
            expected_count: Expected number of return values (2 for rebalance/join/exit, 3 for cleanup)

        Returns:
            Tuple[bool, Tuple]: (success or not, parsed result)
        """
        # Check return result format
        if not isinstance(result, (list, tuple)) or len(result) < expected_count:
            logger.error(
                "❌ RedisGroupQueueManager[%s] Script return format error: Expected %d values, got %s",
                self.key_prefix,
                expected_count,
                result,
            )
            return False, tuple([0] * expected_count)

        # Extract basic values
        if expected_count == 2:
            owner_count, assigned_partitions_flat = result
            parsed_result = (
                owner_count,
                self._convert_flat_to_dict(assigned_partitions_flat),
            )
        elif expected_count == 3:
            cleaned_count, owner_count, assigned_partitions_flat = result
            parsed_result = (
                cleaned_count,
                owner_count,
                self._convert_flat_to_dict(assigned_partitions_flat),
            )
        else:
            return False, tuple([0] * expected_count)

        return True, parsed_result

    def _convert_flat_to_dict(
        self, assigned_partitions_flat: Any
    ) -> Dict[str, List[str]]:
        """
        Convert flat array to dictionary format

        Args:
            assigned_partitions_flat: Flat array [owner_id1, [partitions1], owner_id2, [partitions2], ...]

        Returns:
            Dict[str, List[str]]: Assignment result dictionary
        """
        assigned_partitions = {}
        if (
            isinstance(assigned_partitions_flat, list)
            and len(assigned_partitions_flat) > 0
        ):
            for i in range(0, len(assigned_partitions_flat), 2):
                if i + 1 < len(assigned_partitions_flat):
                    owner_id = self._safe_decode_redis_value(
                        assigned_partitions_flat[i]
                    )
                    partitions_raw = assigned_partitions_flat[i + 1]
                    # Process partition list, each partition name needs decoding
                    if isinstance(partitions_raw, list):
                        partitions = [
                            self._safe_decode_redis_value(p) for p in partitions_raw
                        ]
                    else:
                        partitions = [self._safe_decode_redis_value(partitions_raw)]
                    assigned_partitions[owner_id] = partitions
        return assigned_partitions

    def _safe_decode_redis_value(self, value: Any) -> str:
        """
        Safely decode Redis return value, compatible with bytes and str types

        When Redis client uses decode_responses=False, return value is bytes type
        When Redis client uses decode_responses=True, return value is str type

        Args:
            value: Redis returned value, could be bytes or str

        Returns:
            str: Decoded string
        """
        if isinstance(value, bytes):
            return value.decode('utf-8')
        elif isinstance(value, str):
            return value
        else:
            return str(value)

    async def _check_and_keepalive_if_needed(self, owner_id: str) -> bool:
        """
        Check and perform keepalive if needed

        Check owner's last keepalive time, if no record exists or exceeds 30 seconds, trigger a keepalive.

        Args:
            owner_id: Consumer ID

        Returns:
            bool: Whether keepalive operation was performed
        """
        current_time_ms = int(time.time() * 1000)
        last_keepalive_time = self.owner_last_keepalive_time.get(owner_id, 0)

        # If no record exists or exceeds 30 seconds, trigger keepalive
        if (
            last_keepalive_time == 0 or (current_time_ms - last_keepalive_time) > 30000
        ):  # 30 seconds = 30000 milliseconds
            logger.debug(
                "💓 RedisGroupQueueManager[%s] Triggering keepalive on demand: owner_id=%s, time since last=%.1f seconds",
                self.key_prefix,
                owner_id,
                (current_time_ms - last_keepalive_time) / 1000.0,
            )
            # Trigger keepalive and update timestamp
            try:
                success = await self.keepalive_consumer(owner_id)
                if success:
                    self.owner_last_keepalive_time[owner_id] = current_time_ms
                    return True
                else:
                    logger.warning(
                        "⚠️ RedisGroupQueueManager[%s] On-demand keepalive failed: owner_id=%s, keepalive_consumer returned False",
                        self.key_prefix,
                        owner_id,
                    )
                    return False
            except (redis.RedisError, ValueError, TypeError) as e:
                logger.warning(
                    "⚠️ RedisGroupQueueManager[%s] On-demand keepalive exception: owner_id=%s, error=%s",
                    self.key_prefix,
                    owner_id,
                    e,
                )
                return False
        else:
            logger.debug(
                "💓 RedisGroupQueueManager[%s] No need for keepalive: owner_id=%s, time since last=%.1f seconds",
                self.key_prefix,
                owner_id,
                (current_time_ms - last_keepalive_time) / 1000.0,
            )
            return False

    async def _ensure_scripts_loaded(self):
        """Ensure Lua scripts are loaded"""
        if self._enqueue_script is None:
            self._enqueue_script = self.redis_client.register_script(ENQUEUE_SCRIPT)
            self._get_stats_script = self.redis_client.register_script(
                GET_QUEUE_STATS_SCRIPT
            )
            self._get_all_partitions_stats_script = self.redis_client.register_script(
                GET_ALL_PARTITIONS_STATS_SCRIPT
            )
            self._rebalance_partitions_script = self.redis_client.register_script(
                REBALANCE_PARTITIONS_SCRIPT
            )
            self._join_consumer_script = self.redis_client.register_script(
                JOIN_CONSUMER_SCRIPT
            )
            self._exit_consumer_script = self.redis_client.register_script(
                EXIT_CONSUMER_SCRIPT
            )
            self._keepalive_consumer_script = self.redis_client.register_script(
                KEEPALIVE_CONSUMER_SCRIPT
            )
            self._cleanup_inactive_owners_script = self.redis_client.register_script(
                CLEANUP_INACTIVE_OWNERS_SCRIPT
            )
            self._force_cleanup_script = self.redis_client.register_script(
                FORCE_CLEANUP_SCRIPT
            )
            self._get_messages_script = self.redis_client.register_script(
                GET_MESSAGES_SCRIPT
            )

    @rate_limit(max_rate=200, time_period=1)
    async def deliver_message(
        self,
        group_key: str,
        item: RedisGroupQueueItem,
        return_mode: str = "normal",
        max_total_messages: int = None,
    ) -> bool:
        """
        Deliver message to specified group queue

        Args:
            group_key: Group key, routed to fixed partition via hash
            item: Message data item, must implement RedisGroupQueueItem interface
            return_mode: Return mode, normal returns only bool, reject_reason also returns rejection reason
        Returns:
            bool: Whether delivery was successful
        """
        try:
            await self._ensure_scripts_loaded()

            # Route to fixed partition via hash
            partition = self._hash_group_key_to_partition(group_key)

            # Generate sort score
            sort_score = self.sort_key_func(item)

            # Serialize message based on serialization mode
            if self.serialization_mode == SerializationMode.BSON:
                message_data = item.to_bson_bytes()
            else:  # JSON mode
                message_data = item.to_json_str()

            # Get queue key
            queue_key = self._get_queue_key(partition)

            # Execute Lua script to deliver message
            result = await self._enqueue_script(
                keys=[queue_key, self.counter_key],
                args=[
                    message_data,
                    sort_score,
                    self.queue_expire_seconds,
                    self.activity_expire_seconds,
                    (
                        max_total_messages
                        if max_total_messages is not None
                        else self.max_total_messages
                    ),
                ],
            )

            success, new_count, message = result

            # Safely decode message content, compatible with bytes and str types
            message_str = self._safe_decode_redis_value(message)

            if success:
                # Update statistics
                async with self._stats_lock:
                    self._manager_stats.total_delivered_messages += 1
                    self._manager_stats.total_current_messages = new_count

                logger.debug(
                    "✅ RedisGroupQueueManager[%s] Message delivery successful: group_key=%s->partition=%s, score=%.3f, total retained=%d",
                    self.key_prefix,
                    group_key,
                    partition,
                    sort_score,
                    new_count,
                )
                if return_mode == "normal":
                    return True
                else:
                    return True, message_str
            else:
                # Delivery failed
                async with self._stats_lock:
                    self._manager_stats.total_rejected_messages += 1

                logger.warning(
                    "❌ RedisGroupQueueManager[%s] Delivery rejected: group_key=%s->partition=%s, reason=%s",
                    self.key_prefix,
                    group_key,
                    partition,
                    message_str,
                )
                if return_mode == "normal":
                    return False
                else:
                    return False, message_str

        except (redis.RedisError, ValueError, TypeError) as e:
            # Note: partition might be undefined here, need safe handling
            try:
                partition = self._hash_group_key_to_partition(group_key)
                logger.error(
                    "❌ RedisGroupQueueManager[%s] Message delivery failed: group_key=%s->partition=%s, error=%s",
                    self.key_prefix,
                    group_key,
                    partition,
                    e,
                )
            except (redis.RedisError, ValueError, TypeError):
                logger.error(
                    "❌ RedisGroupQueueManager[%s] Message delivery failed: group_key=%s, error=%s",
                    self.key_prefix,
                    group_key,
                    e,
                )
            if return_mode == "normal":
                return False
            else:
                return False, "Delivery error"

    @rate_limit(
        max_rate=4, time_period=1, key_func=lambda owner_id: f"get_messages_{owner_id}"
    )
    async def get_messages(
        self,
        score_threshold: int,
        current_score: Optional[int] = None,
        owner_id: Optional[str] = None,
        _retry_depth: int = 2,
    ) -> List[RedisGroupQueueItem]:
        """
        Get messages

        Iterate through all partitions assigned to this owner, attempt to get 1 message from each partition.
        On-demand keepalive mechanism: Check last keepalive time, trigger keepalive if exceeds 30 seconds.

        Args:
            score_threshold: Score difference threshold (milliseconds), required parameter
            current_score: Current score, used for threshold comparison when queue is empty, optional parameter
            owner_id: Consumer ID, default uses self.owner_id
            _retry_depth: Internal parameter, recursive retry depth limit, prevents infinite loop

        Returns:
            List[RedisGroupQueueItem]: Message list
        """
        try:
            await self._ensure_scripts_loaded()

            if owner_id is None:
                owner_id = self.owner_id

            # On-demand keepalive mechanism
            await self._check_and_keepalive_if_needed(owner_id)

            # Execute get messages script
            result = await self._get_messages_script(
                keys=[
                    self.owner_activate_time_zset_key,
                    self.queue_list_prefix,
                    self.queue_prefix,
                    self.counter_key,
                ],
                args=[
                    owner_id,
                    self.owner_expire_seconds,
                    score_threshold,
                    (
                        current_score
                        if current_score is not None
                        else self._default_sort_key(None)
                    ),
                ],
            )

            status, messages_data = result

            # Safely decode status value, compatible with bytes and str types
            status_str = self._safe_decode_redis_value(status)

            if status_str == "JOIN_REQUIRED":
                # Check recursion depth, prevent infinite loop
                if _retry_depth <= 0:
                    logger.error(
                        "❌ RedisGroupQueueManager[%s] JOIN_REQUIRED retry attempts exhausted: owner_id=%s",
                        self.key_prefix,
                        owner_id,
                    )
                    raise RuntimeError(
                        f"JOIN_REQUIRED retry attempts exhausted: owner_id={owner_id}"
                    )

                logger.info(
                    "🔄 RedisGroupQueueManager[%s] Consumer join required: owner_id=%s, remaining retries=%d",
                    self.key_prefix,
                    owner_id,
                    _retry_depth - 1,
                )
                # Automatically join consumer
                await self.join_consumer(owner_id)
                # Re-get messages, decrement retry depth
                return await self.get_messages(
                    score_threshold, current_score, owner_id, _retry_depth - 1
                )

            if status_str == "NO_QUEUES":
                logger.warning(
                    "📭 RedisGroupQueueManager[%s] Consumer has no assigned queues: owner_id=%s",
                    self.key_prefix,
                    owner_id,
                )
                return []

            # Parse message data
            messages = []
            for message_data in messages_data:
                try:
                    # Deserialize message based on serialization mode
                    if self.serialization_mode == SerializationMode.BSON:
                        # BSON byte data
                        item = self.item_class.from_bson_bytes(message_data)
                    else:
                        # JSON string
                        item = self.item_class.from_json_str(message_data)
                    messages.append(item)
                except (redis.RedisError, ValueError, TypeError) as e:
                    logger.warning(
                        "⚠️ RedisGroupQueueManager[%s] Message deserialization failed: %s",
                        self.key_prefix,
                        e,
                    )

            if messages:
                # Update statistics
                async with self._stats_lock:
                    self._manager_stats.total_consumed_messages += len(messages)

                logger.debug(
                    "📤 RedisGroupQueueManager[%s] Messages retrieved successfully: owner_id=%s, count=%d",
                    self.key_prefix,
                    owner_id,
                    len(messages),
                )
            else:
                logger.debug(
                    "📭 RedisGroupQueueManager[%s] No consumable messages: owner_id=%s",
                    self.key_prefix,
                    owner_id,
                )

            return messages

        except (redis.RedisError, ValueError, TypeError) as e:
            logger.error(
                "❌ RedisGroupQueueManager[%s] Failed to get messages: owner_id=%s, error=%s",
                self.key_prefix,
                owner_id,
                e,
            )
            return []

    # ==================== New dynamic owner management methods ====================

    @rate_limit(max_rate=1, time_period=1, key_func=lambda: "rebalance_partitions")
    async def rebalance_partitions(self) -> Tuple[int, Dict[str, List[str]]]:
        """
        Rebalance partitions

        Based on owner_activate_time_zset, clear all owners' queue_list,
        redistribute partitions evenly, assign a new queue_list to each owner.

        Returns:
            Tuple[int, Dict[str, List[str]]]: (owner count, assignment result dictionary)
        """
        try:
            await self._ensure_scripts_loaded()

            # Execute rebalance script
            result = await self._rebalance_partitions_script(
                keys=[self.owner_activate_time_zset_key, self.queue_list_prefix],
                args=[self.FIXED_PARTITION_COUNT, self.owner_expire_seconds],
            )

            # Parse return result
            success, (owner_count, assigned_partitions) = self._parse_rebalance_result(
                result, 2
            )
            if not success:
                return 0, {}

            logger.info(
                "🔄 RedisGroupQueueManager[%s] Rebalance partitions completed: owner count=%d, partition assignment=%s",
                self.key_prefix,
                owner_count,
                assigned_partitions,
            )

            return owner_count, assigned_partitions

        except (redis.RedisError, ValueError, TypeError) as e:
            logger.error(
                "❌ RedisGroupQueueManager[%s] Rebalance partitions failed: error=%s",
                self.key_prefix,
                e,
            )
            return 0, {}

    @rate_limit(
        max_rate=1, time_period=1, key_func=lambda owner_id: f"join_consumer_{owner_id}"
    )
    async def join_consumer(
        self, owner_id: Optional[str] = None
    ) -> Tuple[int, Dict[str, List[str]]]:
        """
        Join consumer

        Join owner_activate_time_zset, then rebalance partitions.

        Args:
            owner_id: Consumer ID, default uses self.owner_id

        Returns:
            Tuple[int, Dict[str, List[str]]]: (owner count, assignment result dictionary)
        """
        try:
            await self._ensure_scripts_loaded()

            if owner_id is None:
                owner_id = self.owner_id

            current_time = int(time.time() * 1000)  # Millisecond timestamp

            # Execute join consumer script
            result = await self._join_consumer_script(
                keys=[self.owner_activate_time_zset_key, self.queue_list_prefix],
                args=[
                    owner_id,
                    current_time,
                    self.owner_expire_seconds,
                    self.FIXED_PARTITION_COUNT,
                ],
            )

            # Parse return result
            success, (owner_count, assigned_partitions) = self._parse_rebalance_result(
                result, 2
            )
            if not success:
                return 0, {}

            # Initialize owner's keepalive timestamp
            current_time_ms = int(time.time() * 1000)
            self.owner_last_keepalive_time[owner_id] = current_time_ms

            logger.info(
                "✅ RedisGroupQueueManager[%s] Consumer joined successfully: owner_id=%s, owner count=%d, assignment result=%s",
                self.key_prefix,
                owner_id,
                owner_count,
                assigned_partitions,
            )

            return owner_count, assigned_partitions

        except (redis.RedisError, ValueError, TypeError) as e:
            logger.error(
                "❌ RedisGroupQueueManager[%s] Consumer join failed: owner_id=%s, error=%s",
                self.key_prefix,
                owner_id,
                e,
            )
            return 0, {}

    @rate_limit(
        max_rate=1, time_period=1, key_func=lambda owner_id: f"exit_consumer_{owner_id}"
    )
    async def exit_consumer(
        self, owner_id: Optional[str] = None
    ) -> Tuple[int, Dict[str, List[str]]]:
        """
        Consumer exit

        Remove from owner_activate_time_zset, then rebalance partitions.

        Args:
            owner_id: Consumer ID, default uses self.owner_id

        Returns:
            Tuple[int, Dict[str, List[str]]]: (owner count, assignment result dictionary)
        """
        try:
            await self._ensure_scripts_loaded()

            if owner_id is None:
                owner_id = self.owner_id

            # Execute consumer exit script
            result = await self._exit_consumer_script(
                keys=[self.owner_activate_time_zset_key, self.queue_list_prefix],
                args=[owner_id, self.owner_expire_seconds, self.FIXED_PARTITION_COUNT],
            )

            # Parse return result
            success, (owner_count, assigned_partitions) = self._parse_rebalance_result(
                result, 2
            )
            if not success:
                return 0, {}

            # Remove exiting consumer from keepalive timestamp mapping
            self.owner_last_keepalive_time.pop(owner_id, None)

            logger.info(
                "👋 RedisGroupQueueManager[%s] Consumer exited successfully: owner_id=%s, remaining owner count=%d, reassignment result=%s",
                self.key_prefix,
                owner_id,
                owner_count,
                assigned_partitions,
            )

            return owner_count, assigned_partitions

        except (redis.RedisError, ValueError, TypeError) as e:
            logger.error(
                "❌ RedisGroupQueueManager[%s] Consumer exit failed: owner_id=%s, error=%s",
                self.key_prefix,
                owner_id,
                e,
            )
            return 0, {}

    @rate_limit(
        max_rate=1,
        time_period=2,
        key_func=lambda owner_id: f"keepalive_consumer_{owner_id}",
    )
    async def keepalive_consumer(self, owner_id: Optional[str] = None) -> bool:
        """
        Consumer keepalive

        Consumer periodically updates owner_activate_time_zset time.
        Recommended to call every 30 seconds.

        Args:
            owner_id: Consumer ID

        Returns:
            bool: Whether keepalive was successful
        """
        try:
            await self._ensure_scripts_loaded()

            current_time = int(time.time() * 1000)  # Millisecond timestamp

            # Execute consumer keepalive script
            result = await self._keepalive_consumer_script(
                keys=[self.owner_activate_time_zset_key, self.queue_list_prefix],
                args=[owner_id, current_time, self.owner_expire_seconds],
            )

            success = bool(result)

            if success:
                logger.debug(
                    "💓 RedisGroupQueueManager[%s] Consumer keepalive successful: owner_id=%s",
                    self.key_prefix,
                    owner_id,
                )
            else:
                logger.warning(
                    "⚠️ RedisGroupQueueManager[%s] Consumer keepalive failed, queue_list does not exist: owner_id=%s",
                    self.key_prefix,
                    owner_id,
                )

            return success

        except (redis.RedisError, ValueError, TypeError) as e:
            logger.error(
                "❌ RedisGroupQueueManager[%s] Consumer keepalive failed: owner_id=%s, error=%s",
                self.key_prefix,
                owner_id,
                e,
            )
            return False

    @rate_limit(max_rate=1, time_period=5, key_func=lambda: "cleanup_inactive_owners")
    async def cleanup_inactive_owners(self) -> Tuple[int, int, Dict[str, List[str]]]:
        """
        Periodic cleanup and reset

        Traverse and clean up all inactive owners (e.g., no activity for 5 minutes),
        if any inactive owners exist, rebalance partitions.

        Returns:
            Tuple[int, int, Dict[str, List[str]]]: (cleaned owner count, remaining owner count, reassignment result)
        """
        try:
            await self._ensure_scripts_loaded()

            current_time = int(time.time() * 1000)  # Millisecond timestamp
            inactive_threshold = current_time - (
                self.inactive_threshold_seconds * 1000
            )  # Convert to milliseconds

            # Execute cleanup inactive owners script
            result = await self._cleanup_inactive_owners_script(
                keys=[
                    self.owner_activate_time_zset_key,
                    self.queue_list_prefix,
                    self.queue_prefix,
                    self.counter_key,
                ],
                args=[
                    inactive_threshold,
                    current_time,
                    self.owner_expire_seconds,
                    self.FIXED_PARTITION_COUNT,
                ],
            )

            # Parse return result
            success, (cleaned_count, owner_count, assigned_partitions) = (
                self._parse_rebalance_result(result, 3)
            )
            if not success:
                return 0, 0, {}

            if cleaned_count > 0:
                logger.info(
                    "🧹 RedisGroupQueueManager[%s] Cleanup inactive owners completed: cleaned count=%d, remaining owner count=%d, reassignment result=%s",
                    self.key_prefix,
                    cleaned_count,
                    owner_count,
                    assigned_partitions,
                )
            else:
                logger.debug(
                    "🧹 RedisGroupQueueManager[%s] Cleanup inactive owners completed: no cleanup needed",
                    self.key_prefix,
                )

            return cleaned_count, owner_count, assigned_partitions

        except (redis.RedisError, ValueError, TypeError) as e:
            logger.error(
                "❌ RedisGroupQueueManager[%s] Cleanup inactive owners failed: error=%s",
                self.key_prefix,
                e,
            )
            return 0, 0, {}

    @rate_limit(max_rate=1, time_period=5, key_func=lambda: "force_cleanup_and_reset")
    async def force_cleanup_and_reset(self, purge_all: bool = False) -> int:
        """
        Force cleanup and reset

        - purge_all=False (default): Clean owner_activate_time_zset and all owner queue_lists,
          do not delete partition queues, only recalculate counter.
        - purge_all=True: Additionally delete all partition queues and set counter to 0 (dangerous: full database purge).

        Returns:
            int: When purge_all=False returns cleaned owner count; when purge_all=True returns deleted partition count
        """
        try:
            await self._ensure_scripts_loaded()

            if purge_all:
                # Dangerous: Clear all partition queues + owners + reset counter (via unified script, purge_all='1')
                purged_partitions = await self._force_cleanup_script(
                    keys=[
                        self.owner_activate_time_zset_key,
                        self.queue_list_prefix,
                        self.queue_prefix,
                        self.counter_key,
                    ],
                    args=[self.FIXED_PARTITION_COUNT, "1"],
                )

                # Reset local statistics
                async with self._stats_lock:
                    self._manager_stats.total_current_messages = 0
                    self._manager_stats.total_delivered_messages = 0
                    self._manager_stats.total_consumed_messages = 0
                    self._manager_stats.total_rejected_messages = 0

                logger.warning(
                    "💥 RedisGroupQueueManager[%s] Cleared all queues and owners: partition count=%d",
                    self.key_prefix,
                    purged_partitions,
                )
                return int(purged_partitions or 0)
            else:
                # Only reset owners and queue assignments, do not delete partition queues
                cleaned_count = await self._force_cleanup_script(
                    keys=[
                        self.owner_activate_time_zset_key,
                        self.queue_list_prefix,
                        self.queue_prefix,
                        self.counter_key,
                    ],
                    args=[self.FIXED_PARTITION_COUNT, "0"],
                )

                logger.warning(
                    "💥 RedisGroupQueueManager[%s] Force cleanup and reset completed: cleaned owner count=%d",
                    self.key_prefix,
                    cleaned_count,
                )
                return cleaned_count

        except (redis.RedisError, ValueError, TypeError) as e:
            logger.error(
                "❌ RedisGroupQueueManager[%s] Force cleanup and reset failed: error=%s",
                self.key_prefix,
                e,
            )
            return 0

    @rate_limit(max_rate=1, time_period=5, key_func=lambda: "get_stats")
    async def get_stats(
        self,
        group_key: Optional[str] = None,
        include_all_partitions: bool = False,
        include_partition_details: bool = False,
        include_consumer_info: bool = False,
    ) -> Dict[str, Any]:
        """
        Get statistics (unified interface)

        Args:
            group_key: Group key, if provided get specific queue statistics, otherwise get manager overall statistics
            include_all_partitions: Whether to include statistics for all partitions
            include_partition_details: Whether to include partition detailed information
            include_consumer_info: Whether to include consumer information

        Returns:
            Dict[str, Any]: Statistics
        """
        try:
            await self._ensure_scripts_loaded()

            # If group_key is specified, return specific queue statistics
            if group_key is not None and not include_all_partitions:
                # Get statistics for single partition
                partition = self._hash_group_key_to_partition(group_key)
                queue_key = self._get_queue_key(partition)

                result = await self._get_stats_script(
                    keys=[queue_key, self.counter_key], args=[]
                )

                queue_size, _total_count, min_score, max_score = result

                return {
                    "type": "queue_stats",
                    "queue_name": f"{group_key}->partition={partition}",
                    "current_size": queue_size,
                    "last_activity_time": time.time(),
                    "min_score": min_score,
                    "max_score": max_score,
                    "partition": partition,
                }

            # Get statistics for all partitions (manager level or all partitions statistics)
            result = await self._get_all_partitions_stats_script(
                keys=[self.queue_prefix, self.counter_key],
                args=[str(self.FIXED_PARTITION_COUNT)],
            )

            (
                total_count,
                total_messages_in_queues,
                global_min_score,
                global_max_score,
                partition_stats_raw,
            ) = result

            # Build basic statistics
            async with self._stats_lock:
                # Update uptime and statistics
                self._manager_stats.uptime_seconds = time.time() - self._start_time
                self._manager_stats.total_current_messages = total_messages_in_queues
                self._manager_stats.total_queues = self.FIXED_PARTITION_COUNT

                stats = self._manager_stats.to_dict()

            # Add real-time statistics
            stats.update(
                {
                    "type": (
                        "manager_stats" if group_key is None else "all_partitions_stats"
                    ),
                    "counter_total_count": total_count,
                    "actual_messages_in_queues": total_messages_in_queues,
                    "global_min_score": global_min_score,
                    "global_max_score": global_max_score,
                    "key_prefix": self.key_prefix,
                }
            )

            # If consumer information is needed
            if include_consumer_info:
                try:
                    active_owners_raw = await self.redis_client.zrange(
                        self.owner_activate_time_zset_key, 0, -1
                    )
                    # Safely decode owner list
                    active_owners = [
                        self._safe_decode_redis_value(owner)
                        for owner in active_owners_raw
                    ]
                    stats["active_consumers_count"] = len(active_owners)
                    stats["active_consumers"] = active_owners

                    # Get partition assignments
                    partition_assignments = {}
                    for owner in active_owners:
                        queue_list_key = f"{self.queue_list_prefix}{owner}"
                        assigned_partitions_raw = await self.redis_client.lrange(
                            queue_list_key, 0, -1
                        )
                        # Safely decode partition list
                        assigned_partitions = [
                            self._safe_decode_redis_value(p)
                            for p in assigned_partitions_raw
                        ]
                        partition_assignments[owner] = assigned_partitions
                    stats["partition_assignments"] = partition_assignments

                except (redis.RedisError, ValueError, TypeError) as e:
                    logger.warning("Failed to get consumer information: %s", e)
                    stats["active_consumers_count"] = 0
                    stats["active_consumers"] = []
                    stats["partition_assignments"] = {}

            # If partition detailed information is needed
            if include_partition_details:
                partitions = []
                non_empty_partitions = 0
                max_partition_size = 0
                min_partition_size = float('inf')

                for i in range(0, len(partition_stats_raw), 4):
                    if i + 3 < len(partition_stats_raw):
                        partition_size = partition_stats_raw[i + 1]
                        partitions.append(
                            {
                                "partition": self._safe_decode_redis_value(
                                    partition_stats_raw[i]
                                ),
                                "current_size": partition_size,
                                "min_score": partition_stats_raw[i + 2],
                                "max_score": partition_stats_raw[i + 3],
                            }
                        )

                        if partition_size > 0:
                            non_empty_partitions += 1
                            max_partition_size = max(max_partition_size, partition_size)
                            min_partition_size = min(min_partition_size, partition_size)

                stats["partitions"] = partitions
                stats["non_empty_partitions"] = non_empty_partitions
                stats["max_partition_size"] = (
                    max_partition_size if max_partition_size != 0 else 0
                )
                stats["min_partition_size"] = (
                    min_partition_size if min_partition_size != float('inf') else 0
                )

            return stats

        except (redis.RedisError, ValueError, TypeError) as e:
            logger.error(
                "Failed to get statistics: group_key=%s, error=%s", group_key, e
            )

            # Fallback: Return basic statistics
            try:
                current_count = await self.redis_client.get(self.counter_key)
                total_current_messages = int(current_count) if current_count else 0
            except (redis.RedisError, ValueError, TypeError):
                total_current_messages = 0

            return {
                "type": "error_fallback",
                "total_current_messages": total_current_messages,
                "total_queues": self.FIXED_PARTITION_COUNT,
                "error": str(e),
            }

    @rate_limit(
        max_rate=1,
        time_period=5,
        key_func=lambda group_key: f"get_queue_stats_{group_key}",
    )
    async def get_queue_stats(self, group_key: str) -> Optional[Dict[str, Any]]:
        """Compatibility method: Get queue statistics"""
        result = await self.get_stats(group_key=group_key)
        return result if result.get("type") != "error_fallback" else None

    @rate_limit(max_rate=1, time_period=5, key_func=lambda: "get_manager_stats")
    async def get_manager_stats(self) -> Dict[str, Any]:
        """Compatibility method: Get manager statistics"""
        return await self.get_stats()

    async def start(self):
        """
        Start manager (start periodic tasks)

        Can only be started once, cannot restart after shutdown

        Raises:
            RuntimeError: If manager is already started or has been shut down
        """
        if self._state == ManagerState.STARTED:
            logger.warning(
                "⚠️ RedisGroupQueueManager[%s] Already started, ignoring duplicate start request",
                self.key_prefix,
            )
            return

        if self._state == ManagerState.SHUTDOWN:
            raise RuntimeError(
                f"RedisGroupQueueManager[{self.key_prefix}] has been shut down, cannot restart"
            )

        # State must be CREATED
        if self._state != ManagerState.CREATED:
            raise RuntimeError(
                f"RedisGroupQueueManager[{self.key_prefix}] state abnormal: {self._state}"
            )

        logger.info("🚀 RedisGroupQueueManager[%s] Starting...", self.key_prefix)

        await self.start_periodic_tasks()

        # Update state to started
        self._state = ManagerState.STARTED

        logger.info("✅ RedisGroupQueueManager[%s] Startup completed", self.key_prefix)

    async def start_periodic_tasks(self):
        """Start periodic tasks"""
        if self._running:
            return

        self._running = True

        # Execute cleanup immediately on startup
        try:
            await self.cleanup_inactive_owners()
            logger.info(
                "🧹 RedisGroupQueueManager[%s] Startup cleanup completed",
                self.key_prefix,
            )
        except (redis.RedisError, ValueError, TypeError) as e:
            logger.warning(
                "⚠️ RedisGroupQueueManager[%s] Startup cleanup failed: %s",
                self.key_prefix,
                e,
            )

        # Execute log immediately on startup
        try:
            await self._log_manager_details()
            logger.info(
                "🔥 RedisGroupQueueManager[%s] Startup log printing completed",
                self.key_prefix,
            )
        except (redis.RedisError, ValueError, TypeError) as e:
            logger.warning(
                "⚠️ RedisGroupQueueManager[%s] Startup log printing failed: %s",
                self.key_prefix,
                e,
            )

        # Start periodic tasks
        self._log_task = asyncio.create_task(self._periodic_log_worker())
        self._cleanup_task = asyncio.create_task(self._periodic_cleanup_worker())

        logger.info(
            "📊 RedisGroupQueueManager[%s] Periodic tasks started", self.key_prefix
        )

    async def stop_periodic_tasks(self):
        """Stop periodic tasks"""
        if not self._running:
            return

        self._running = False

        # Stop log task
        if self._log_task and not self._log_task.done():
            self._log_task.cancel()
            try:
                await self._log_task
            except asyncio.CancelledError:
                pass

        # Stop cleanup task
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        logger.info(
            "📊 RedisGroupQueueManager[%s] Periodic tasks stopped", self.key_prefix
        )

    async def _periodic_log_worker(self):
        """Periodic log printing worker coroutine"""
        try:
            while self._running:
                await asyncio.sleep(self.log_interval_seconds)
                if self._running:
                    await self._log_manager_details()
        except asyncio.CancelledError:
            logger.debug(
                "📊 RedisGroupQueueManager[%s] Periodic log task cancelled",
                self.key_prefix,
            )
        except (redis.RedisError, ValueError, TypeError) as e:
            logger.error(
                "📊 RedisGroupQueueManager[%s] Periodic log task exception: %s",
                self.key_prefix,
                e,
            )

    async def _periodic_cleanup_worker(self):
        """Periodic cleanup worker coroutine"""
        try:
            while self._running:
                # Add jitter to avoid all instances cleaning simultaneously, ensure non-negative
                jitter = self.cleanup_interval_seconds * 0.3
                delay = self.cleanup_interval_seconds + random.uniform(-jitter, jitter)
                await asyncio.sleep(max(1.0, delay))
                if self._running:
                    await self.cleanup_inactive_owners()
        except asyncio.CancelledError:
            logger.debug(
                "🧹 RedisGroupQueueManager[%s] Periodic cleanup task cancelled",
                self.key_prefix,
            )
        except (redis.RedisError, ValueError, TypeError) as e:
            logger.error(
                "🧹 RedisGroupQueueManager[%s] Periodic cleanup task exception: %s",
                self.key_prefix,
                e,
            )

    async def _log_manager_details(self):
        """Print manager details"""
        try:
            manager_stats = await self.get_manager_stats()

            # Print manager overall status
            logger.info(
                "📊 RedisGroupQueueManager[%s] Overall status: "
                "active queues=%d, total messages=%d, total delivered=%d, total consumed=%d, total rejected=%d, uptime=%.1f seconds",
                self.key_prefix,
                manager_stats["total_queues"],
                manager_stats["total_current_messages"],
                manager_stats["total_delivered_messages"],
                manager_stats["total_consumed_messages"],
                manager_stats["total_rejected_messages"],
                manager_stats["uptime_seconds"],
            )

            # Unified print all partitions' detailed information at once
            partitions = self.partition_names
            details_lines = []
            for partition in partitions:
                try:
                    queue_key = self._get_queue_key(partition)
                    queue_size = await self.redis_client.zcard(queue_key)
                    if queue_size > 0:
                        # Get min and max scores
                        min_result = await self.redis_client.zrange(
                            queue_key, 0, 0, withscores=True
                        )
                        max_result = await self.redis_client.zrange(
                            queue_key, -1, -1, withscores=True
                        )
                        min_score = min_result[0][1] if min_result else 0
                        max_score = max_result[0][1] if max_result else 0
                        details_lines.append(
                            f"   Partition[{partition}]: Size={queue_size}, Score range=[{min_score:.3f}, {max_score:.3f}]"
                        )
                    else:
                        details_lines.append(f"   Partition[{partition}]: Size=0")
                except (redis.RedisError, ValueError, TypeError) as e:
                    details_lines.append(
                        f"   Partition[{partition}]: Failed to get status: {e}"
                    )

            if details_lines:
                logger.info(
                    "🔥 Partition status summary: Total %d partitions\n%s",
                    len(partitions),
                    "\n".join(details_lines),
                )

        except (redis.RedisError, ValueError, TypeError) as e:
            logger.error(
                "📊 RedisGroupQueueManager[%s] Failed to print details: %s",
                self.key_prefix,
                e,
            )

    async def shutdown(self, mode: ShutdownMode = ShutdownMode.HARD) -> bool:
        """
        Shutdown manager

        Args:
            mode: Shutdown mode

        Returns:
            bool: Whether shutdown was successful
        """
        if self._state == ManagerState.SHUTDOWN:
            logger.warning(
                "⚠️ RedisGroupQueueManager[%s] Already shut down, ignoring duplicate shutdown request",
                self.key_prefix,
            )
            return True

        if self._state == ManagerState.CREATED:
            logger.info(
                "ℹ️ RedisGroupQueueManager[%s] Shutting down without having started",
                self.key_prefix,
            )
            self._state = ManagerState.SHUTDOWN
            return True

        # State must be STARTED
        if self._state != ManagerState.STARTED:
            logger.warning(
                "⚠️ RedisGroupQueueManager[%s] State abnormal, force shutdown: %s",
                self.key_prefix,
                self._state,
            )

        logger.info(
            "🔌 RedisGroupQueueManager[%s] Starting shutdown...", self.key_prefix
        )

        # Stop periodic tasks
        await self.stop_periodic_tasks()

        if mode == ShutdownMode.SOFT:
            # Soft shutdown: Check if messages exist
            stats = await self.get_manager_stats()
            remaining_messages = stats.get("total_current_messages", 0)
            if remaining_messages > 0:
                logger.warning(
                    "⚠️ RedisGroupQueueManager[%s] Soft shutdown detected remaining messages: %d messages",
                    self.key_prefix,
                    remaining_messages,
                )
                # Soft shutdown failed, but don't change state, allow retry
                return False

        # Final log details before shutdown
        try:
            await self._log_manager_details()
            logger.info(
                "🔥 RedisGroupQueueManager[%s] Final status log before shutdown completed",
                self.key_prefix,
            )
        except (redis.RedisError, ValueError, TypeError) as e:
            logger.warning(
                "⚠️ RedisGroupQueueManager[%s] Failed to print log before shutdown: %s",
                self.key_prefix,
                e,
            )

        # Update state to shut down
        self._state = ManagerState.SHUTDOWN

        logger.info("🔌 RedisGroupQueueManager[%s] Shut down", self.key_prefix)
        return True

    def get_state(self) -> ManagerState:
        """
        Get manager current state

        Returns:
            ManagerState: Current state
        """
        return self._state

    def __repr__(self) -> str:
        return (
            f"RedisGroupQueueManager(key_prefix={self.key_prefix}, "
            f"max_total_messages={self.max_total_messages})"
        )

"""
Redis Length-Limited Cache Manager

Length-limited cache implementation based on Redis Sorted Set, supporting:
- Append data to queue by key, prioritizing the provided timestamp as score
- Clean up by length, removing from the earliest data, retaining up to 100 records at most
- Queue expiration time is 60 minutes, extended on each append
"""

import time
import random
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

from core.di.decorators import component
from core.observation.logger import get_logger
from core.component.redis_provider import RedisProvider
from .redis_data_processor import RedisDataProcessor

# Configuration constants
DEFAULT_MAX_LENGTH = 100  # Default maximum length of 100 items
DEFAULT_EXPIRE_MINUTES = 60  # Default expiration time of 60 minutes
DEFAULT_CLEANUP_PROBABILITY = 0.1  # 10% probability of performing length cleanup

# Lua script: Clean up data by length
LENGTH_CLEANUP_LUA_SCRIPT = """
local queue_key = KEYS[1]
local max_length = tonumber(ARGV[1])

-- 1. Get queue length
local queue_length = redis.call('ZCARD', queue_key)

-- 2. If exceeding max length, delete excess elements from the earliest data
local cleaned_count = 0
if queue_length > max_length then
    local excess_count = queue_length - max_length
    -- Remove the earliest data (lowest score)
    cleaned_count = redis.call('ZREMRANGEBYRANK', queue_key, 0, excess_count - 1)
end

return cleaned_count
"""

# Lua script: Fetch data by timestamp range (with scores)
FETCH_BY_DATE_TIMESTAMP_RANGE_LUA_SCRIPT = """
local queue_key = KEYS[1]
local min_score = ARGV[1]  -- Minimum timestamp
local max_score = ARGV[2]  -- Maximum timestamp
local limit = tonumber(ARGV[3]) or -1  -- Limit number, -1 means no limit

-- Retrieve data and scores by score range, ordered by timestamp ascending
local messages
if limit > 0 then
    messages = redis.call('ZRANGEBYSCORE', queue_key, min_score, max_score, 'WITHSCORES', 'LIMIT', 0, limit)
else
    messages = redis.call('ZRANGEBYSCORE', queue_key, min_score, max_score, 'WITHSCORES')
end

return messages
"""

logger = get_logger(__name__)


@component(name="redis_length_cache_factory")
class RedisLengthCacheFactory:
    """Redis length-limited cache manager factory"""

    def __init__(self, redis_provider: RedisProvider):
        """
        Initialize cache factory

        Args:
            redis_provider: Redis connection provider
        """
        self.redis_provider = redis_provider
        self._length_cleanup_script = None
        self._timestamp_range_script = None
        logger.info("Redis length-limited cache factory initialized")

    async def _ensure_length_cleanup_script_registered(self):
        """Ensure length cleanup Lua script is registered (register only once)"""
        if self._length_cleanup_script is None:
            client = await self.redis_provider.get_client()
            self._length_cleanup_script = client.register_script(
                LENGTH_CLEANUP_LUA_SCRIPT
            )
            logger.info("Length cleanup Lua script registered")
        return self._length_cleanup_script

    async def _ensure_timestamp_range_script_registered(self):
        """Ensure timestamp range query Lua script is registered (register only once)"""
        if self._timestamp_range_script is None:
            # Use binary_cache client to register script, ensuring same connection as execution
            binary_client = await self.redis_provider.get_named_client(
                "binary_cache", decode_responses=False
            )
            self._timestamp_range_script = binary_client.register_script(
                FETCH_BY_DATE_TIMESTAMP_RANGE_LUA_SCRIPT
            )
            logger.info("Timestamp range query Lua script registered")
        return self._timestamp_range_script

    async def create_cache_manager(
        self,
        max_length: int = DEFAULT_MAX_LENGTH,
        expire_minutes: int = DEFAULT_EXPIRE_MINUTES,
        cleanup_probability: float = DEFAULT_CLEANUP_PROBABILITY,
    ) -> 'RedisLengthCacheManager':
        """
        Create cache manager instance

        Args:
            max_length: Maximum length
            expire_minutes: Expiration time (minutes)
            cleanup_probability: Length cleanup probability (0.0-1.0)

        Returns:
            RedisLengthCacheManager: Cache manager instance
        """
        length_cleanup_script = await self._ensure_length_cleanup_script_registered()
        timestamp_range_script = await self._ensure_timestamp_range_script_registered()
        return RedisLengthCacheManager(
            redis_provider=self.redis_provider,
            length_cleanup_script=length_cleanup_script,
            fetch_by_timestamp_range_script=timestamp_range_script,
            max_length=max_length,
            expire_minutes=expire_minutes,
            cleanup_probability=cleanup_probability,
        )


class RedisLengthCacheManager:
    """
    Redis Length-Limited Cache Manager

    Length-limited queue cache implemented using Redis Sorted Set (ZSET):
    - Score: Prefer provided time, otherwise use current timestamp
    - Member: Unique identifier:data content, ensuring data uniqueness
    - Supports length-based cleanup, removing from earliest data, retaining up to specified count
    - Queue expiration time is 60 minutes, renewed on each append
    """

    def __init__(
        self,
        redis_provider: RedisProvider,
        length_cleanup_script,
        fetch_by_timestamp_range_script,
        max_length: int = DEFAULT_MAX_LENGTH,
        expire_minutes: int = DEFAULT_EXPIRE_MINUTES,
        cleanup_probability: float = DEFAULT_CLEANUP_PROBABILITY,
    ):
        """
        Initialize Redis length-limited cache manager

        Args:
            redis_provider: Redis connection provider
            length_cleanup_script: Pre-registered length cleanup Lua script object
            fetch_by_timestamp_range_script: Pre-registered timestamp range query Lua script object
            max_length: Maximum length
            expire_minutes: Expiration time (minutes)
            cleanup_probability: Length cleanup probability (0.0-1.0)
        """
        self.redis_provider = redis_provider
        self.max_length = max_length
        self.expire_minutes = expire_minutes
        self.cleanup_probability = cleanup_probability
        self._length_cleanup_script = length_cleanup_script
        self._fetch_by_timestamp_range_script = fetch_by_timestamp_range_script

        logger.info(
            "Redis length-limited cache manager initialized: max_length=%d, expire=%d minutes, cleanup_prob=%.1f%%",
            max_length,
            expire_minutes,
            cleanup_probability * 100,
        )

    def _convert_timestamp(self, timestamp: Optional[Union[int, datetime]]) -> int:
        """
        Convert timestamp to millisecond-level integer

        Args:
            timestamp: Timestamp (milliseconds) or datetime object, use current time if not provided

        Returns:
            int: Millisecond-level timestamp
        """
        if timestamp is None:
            return int(time.time() * 1000)
        elif isinstance(timestamp, datetime):
            # If it's a datetime object, convert to millisecond timestamp
            return int(timestamp.timestamp() * 1000)
        else:
            # If it's an integer timestamp
            return int(timestamp)

    async def _cleanup_if_needed(self, key: str) -> int:
        """
        Decide whether to perform length cleanup based on probability

        Args:
            key: Cache key name

        Returns:
            int: Number of cleaned data items
        """
        if random.random() < self.cleanup_probability:
            # Use Lua script for atomic cleanup
            cleaned_count = await self._length_cleanup_script(
                keys=[key], args=[self.max_length]
            )
            return cleaned_count
        return 0

    async def append(
        self,
        key: str,
        data: Union[str, Dict, List, Any],
        timestamp: Optional[Union[int, datetime]] = None,
    ) -> bool:
        """
        Append data to the queue for the specified key

        Args:
            key: Cache key name
            data: Data to append, supports strings, dictionaries, lists, etc.
            timestamp: Timestamp (milliseconds) or datetime object, use current time if not provided

        Returns:
            bool: Whether operation succeeded

        Examples:
            # Append string data using current time
            await cache.append("user_actions", "user_login")

            # Append dictionary data with specified timestamp
            await cache.append("api_calls", {"method": "GET", "path": "/api/users"}, timestamp=1640995200000)

            # Append data using datetime object
            from common_utils.datetime_utils import get_now_with_timezone
            await cache.append("events", "user_action", timestamp=get_now_with_timezone())
        """
        try:
            client = await self.redis_provider.get_client()

            # 1. Data preprocessing
            score_timestamp = self._convert_timestamp(timestamp)
            unique_member = RedisDataProcessor.process_data_for_storage(data)

            # 2. Execute Redis operations
            add_result = await client.zadd(key, {unique_member: score_timestamp})

            expire_seconds = self.expire_minutes * 60
            expire_result = await client.expire(key, expire_seconds)

            zadd_result = add_result if add_result else None
            expire_result = expire_result if expire_result else None

            # 3. Perform cleanup based on probability
            cleaned_count = await self._cleanup_if_needed(key)

            # 4. Record result
            if zadd_result is not None and expire_result:
                if cleaned_count > 0:
                    logger.debug(
                        "Data appended successfully and earliest data cleaned: key=%s, member_length=%d, timestamp=%d, expire=%ds, cleaned=%d",
                        key,
                        len(unique_member),
                        score_timestamp,
                        expire_seconds,
                        cleaned_count,
                    )
                else:
                    logger.debug(
                        "Data appended successfully: key=%s, member_length=%d, timestamp=%d, expire=%ds",
                        key,
                        len(unique_member),
                        score_timestamp,
                        expire_seconds,
                    )
                return True
            else:
                logger.warning(
                    "Data append partially failed: key=%s, zadd_result=%s, expire_result=%s",
                    key,
                    zadd_result,
                    expire_result,
                )
                return False

        except (ConnectionError, TimeoutError, ValueError) as e:
            logger.error(
                "Failed to append data to Redis: key=%s, error=%s", key, str(e)
            )
            return False

    async def get_queue_size(self, key: str) -> int:
        """
        Get current size of the specified queue

        Args:
            key: Cache key name

        Returns:
            int: Number of elements in the queue
        """
        try:
            client = await self.redis_provider.get_client()
            size = await client.zcard(key)
            return size or 0
        except (ConnectionError, TimeoutError) as e:
            logger.error("Failed to get queue size: key=%s, error=%s", key, str(e))
            return 0

    async def clear_queue(self, key: str) -> bool:
        """
        Clear all data from the specified queue

        Args:
            key: Cache key name

        Returns:
            bool: Whether operation succeeded
        """
        try:
            client = await self.redis_provider.get_client()
            result = await client.delete(key)
            logger.info("Cleared queue: key=%s, result=%d", key, result)
            return result > 0
        except (ConnectionError, TimeoutError) as e:
            logger.error("Failed to clear queue: key=%s, error=%s", key, str(e))
            return False

    async def delete(self, key: str) -> bool:
        """
        Delete the specified cache key (alias of clear_queue)

        Args:
            key: Cache key name

        Returns:
            bool: Whether operation succeeded
        """
        return await self.clear_queue(key)

    async def cleanup_excess(self, key: str) -> int:
        """
        Manually clean up data exceeding length limit in the specified queue

        Args:
            key: Cache key name

        Returns:
            int: Number of cleaned data items
        """
        try:
            # Execute pre-registered Lua script for length cleanup
            cleaned_count = await self._length_cleanup_script(
                keys=[key], args=[self.max_length]
            )

            if cleaned_count > 0:
                logger.info(
                    "Manually cleaned earliest data: key=%s, cleaned=%d",
                    key,
                    cleaned_count,
                )

            return cleaned_count

        except (ConnectionError, TimeoutError) as e:
            logger.error("Failed to clean earliest data: key=%s, error=%s", key, str(e))
            return 0

    async def get_by_timestamp_range(
        self,
        key: str,
        start_timestamp: Optional[Union[int, datetime]] = None,
        end_timestamp: Optional[Union[int, datetime]] = None,
        limit: int = -1,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve queue data by timestamp range

        Args:
            key: Cache key name
            start_timestamp: Start timestamp (milliseconds) or datetime object, None means no restriction
            end_timestamp: End timestamp (milliseconds) or datetime object, None means no restriction
            limit: Limit number of returned items, -1 means no limit

        Returns:
            List[Dict[str, Any]]: List of data within timestamp range, each element contains:
                - id: Unique identifier
                - data: Original data
                - timestamp: Timestamp (milliseconds)
                - datetime: Formatted time string
        """
        try:
            # Convert timestamps
            min_score = "-inf"
            max_score = "+inf"

            if start_timestamp is not None:
                min_score = str(self._convert_timestamp(start_timestamp))

            if end_timestamp is not None:
                max_score = str(self._convert_timestamp(end_timestamp))

            # Execute pre-registered Lua script
            messages = await self._fetch_by_timestamp_range_script(
                keys=[key], args=[min_score, max_score, limit]
            )
            if not messages:
                logger.debug(
                    "No data in timestamp range: key=%s, range=[%s, %s]",
                    key,
                    min_score,
                    max_score,
                )
                return []

            # Parse message data (WITHSCORES returns format: [member1, score1, member2, score2, ...])
            result = []

            # Process WITHSCORES returned results, every two elements form a pair: message content and score
            if len(messages) % 2 != 0:
                logger.warning(
                    "WITHSCORES returned data length abnormal: %d, should be even",
                    len(messages),
                )
                return []

            for i in range(0, len(messages), 2):
                try:
                    if i + 1 >= len(messages):
                        logger.warning(
                            "WITHSCORES data index out of bounds: i=%d, len=%d",
                            i,
                            len(messages),
                        )
                        break

                    message = messages[i]  # Message content
                    score_raw = messages[i + 1]  # Score (could be bytes or string)

                    # Safely convert score to timestamp
                    try:
                        if isinstance(score_raw, bytes):
                            score_str = score_raw.decode('utf-8')
                        else:
                            score_str = str(score_raw)
                        timestamp = int(float(score_str))
                    except (ValueError, UnicodeDecodeError) as score_e:
                        logger.warning(
                            "Score conversion failed: score_raw=%s (type=%s), error=%s",
                            score_raw,
                            type(score_raw),
                            str(score_e),
                        )
                        timestamp = int(
                            time.time() * 1000
                        )  # Use current time as fallback

                    # Use data processor to parse data
                    processed_data = RedisDataProcessor.process_data_from_storage(
                        message
                    )

                    result.append(
                        {
                            "id": processed_data["id"],
                            "data": processed_data["data"],
                            "timestamp": timestamp,
                            "datetime": datetime.fromtimestamp(
                                timestamp / 1000
                            ).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                        }
                    )

                except Exception as e:
                    logger.warning(
                        "Failed to parse message: i=%d, message=%s, error=%s",
                        i,
                        message if i < len(messages) else "index out of bounds",
                        str(e),
                    )
                    continue

            # Sort by timestamp (newest first)
            result.sort(key=lambda x: x["timestamp"], reverse=True)

            logger.debug(
                "Successfully retrieved data by timestamp range: key=%s, range=[%s, %s], count=%d",
                key,
                min_score,
                max_score,
                len(result),
            )

            return result

        except Exception as e:
            logger.error(
                "Failed to retrieve data by timestamp range: key=%s, error=%s",
                key,
                str(e),
            )
            return []

    async def get_queue_stats(self, key: str) -> Dict[str, Any]:
        """
        Get queue statistics

        Args:
            key: Cache key name

        Returns:
            Dict[str, Any]: Queue statistics
        """
        try:
            client = await self.redis_provider.get_client()

            # Get queue size
            total_count = await client.zcard(key) or 0

            if total_count == 0:
                return {
                    "key": key,
                    "total_count": 0,
                    "max_length": self.max_length,
                    "oldest_timestamp": None,
                    "newest_timestamp": None,
                    "ttl_seconds": -1,
                }

            # Get oldest and newest timestamps
            oldest_data = await client.zrange(key, 0, 0, withscores=True)
            newest_data = await client.zrange(key, -1, -1, withscores=True)

            oldest_timestamp = int(oldest_data[0][1]) if oldest_data else None
            newest_timestamp = int(newest_data[0][1]) if newest_data else None

            # Get TTL
            ttl = await client.ttl(key)

            return {
                "key": key,
                "total_count": total_count,
                "max_length": self.max_length,
                "oldest_timestamp": oldest_timestamp,
                "newest_timestamp": newest_timestamp,
                "oldest_datetime": (
                    datetime.fromtimestamp(oldest_timestamp / 1000).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                    if oldest_timestamp
                    else None
                ),
                "newest_datetime": (
                    datetime.fromtimestamp(newest_timestamp / 1000).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                    if newest_timestamp
                    else None
                ),
                "ttl_seconds": ttl,
                "is_full": total_count >= self.max_length,
            }

        except (ConnectionError, TimeoutError) as e:
            logger.error(
                "Failed to get queue statistics: key=%s, error=%s", key, str(e)
            )
            return {
                "key": key,
                "total_count": 0,
                "max_length": self.max_length,
                "oldest_timestamp": None,
                "newest_timestamp": None,
                "ttl_seconds": -1,
                "error": str(e),
            }


# For backward compatibility, provide a default component instance
@component(name="redis_length_cache_manager")
class DefaultRedisLengthCacheManager:
    """Default Redis length-limited cache manager (backward compatibility)"""

    def __init__(self, redis_provider: RedisProvider):
        self.redis_provider = redis_provider
        self._manager = None
        self._factory = None

    async def _get_manager(self):
        """Lazy initialization of manager"""
        if self._manager is None:
            if self._factory is None:
                self._factory = RedisLengthCacheFactory(self.redis_provider)
            self._manager = await self._factory.create_cache_manager()
        return self._manager

    async def append(
        self,
        key: str,
        data: Union[str, Dict, List, Any],
        timestamp: Optional[Union[int, datetime]] = None,
    ) -> bool:
        manager = await self._get_manager()
        return await manager.append(key, data, timestamp)

    async def get_queue_size(self, key: str) -> int:
        manager = await self._get_manager()
        return await manager.get_queue_size(key)

    async def clear_queue(self, key: str) -> bool:
        manager = await self._get_manager()
        return await manager.clear_queue(key)

    async def delete(self, key: str) -> bool:
        manager = await self._get_manager()
        return await manager.delete(key)

    async def cleanup_excess(self, key: str) -> int:
        manager = await self._get_manager()
        return await manager.cleanup_excess(key)

    async def get_queue_stats(self, key: str) -> Dict[str, Any]:
        manager = await self._get_manager()
        return await manager.get_queue_stats(key)

    async def get_by_timestamp_range(
        self,
        key: str,
        start_timestamp: Optional[Union[int, datetime]] = None,
        end_timestamp: Optional[Union[int, datetime]] = None,
        limit: int = -1,
    ) -> List[Dict[str, Any]]:
        manager = await self._get_manager()
        return await manager.get_by_timestamp_range(
            key, start_timestamp, end_timestamp, limit
        )

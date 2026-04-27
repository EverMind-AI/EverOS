"""
Redis Time Window Cache Manager

Time window cache implemented using Redis Sorted Set, supporting:
- Appending data to time window queue by key
- Retrieving data within a specified time range
- Automatic expiration cleanup
- Random cleanup of expired data
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
DEFAULT_EXPIRE_MINUTES = 10  # Default expiration time: 10 minutes
DEFAULT_CLEANUP_PROBABILITY = 0.1  # 10% probability to perform random cleanup
DEFAULT_CLEANUP_MULTIPLIER = (
    2  # Cleanup threshold multiplier (clean data older than 2x expiration time)
)

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


@component(name="redis_windows_cache_factory")
class RedisWindowsCacheFactory:
    """Redis Time Window Cache Manager Factory"""

    def __init__(self, redis_provider: RedisProvider):
        """
        Initialize cache factory

        Args:
            redis_provider: Redis connection provider
        """
        self.redis_provider = redis_provider
        self._timestamp_range_script = None
        logger.info("Redis Time Window Cache Factory initialization completed")

    async def _ensure_timestamp_range_script_registered(self):
        """Ensure the timestamp range query Lua script is registered (register only once)"""
        if self._timestamp_range_script is None:
            # Register script using binary_cache client to ensure same connection used during execution
            binary_client = await self.redis_provider.get_named_client(
                "binary_cache", decode_responses=False
            )
            self._timestamp_range_script = binary_client.register_script(
                FETCH_BY_DATE_TIMESTAMP_RANGE_LUA_SCRIPT
            )
            logger.info("Timestamp range query Lua script registration completed")
        return self._timestamp_range_script

    async def create_cache_manager(
        self,
        expire_minutes: int = DEFAULT_EXPIRE_MINUTES,
        cleanup_probability: float = DEFAULT_CLEANUP_PROBABILITY,
    ) -> 'RedisWindowsCacheManager':
        """
        Create cache manager instance

        Args:
            expire_minutes: Default expiration time (minutes)
            cleanup_probability: Random cleanup probability (0.0-1.0)

        Returns:
            RedisWindowsCacheManager: Cache manager instance
        """
        timestamp_range_script = await self._ensure_timestamp_range_script_registered()
        return RedisWindowsCacheManager(
            redis_provider=self.redis_provider,
            fetch_by_timestamp_range_script=timestamp_range_script,
            expire_minutes=expire_minutes,
            cleanup_probability=cleanup_probability,
        )


class RedisWindowsCacheManager:
    """
    Redis Time Window Cache Manager

    Time window queue cache implemented using Redis Sorted Set (ZSET):
    - Score: Millisecond-level timestamp, used for time sorting
    - Member: Unique identifier:data content, ensuring data uniqueness
    - Supports custom expiration time, extended on each append
    - Provides random cleanup mechanism to prevent infinite memory growth
    """

    def __init__(
        self,
        redis_provider: RedisProvider,
        fetch_by_timestamp_range_script,
        expire_minutes: int = DEFAULT_EXPIRE_MINUTES,
        cleanup_probability: float = DEFAULT_CLEANUP_PROBABILITY,
    ):
        """
        Initialize Redis Time Window Cache Manager

        Args:
            redis_provider: Redis connection provider
            fetch_by_timestamp_range_script: Pre-registered timestamp range query Lua script object
            expire_minutes: Default expiration time (minutes)
            cleanup_probability: Random cleanup probability (0.0-1.0)
        """
        self.redis_provider = redis_provider
        self.default_expire_minutes = expire_minutes
        self.cleanup_probability = cleanup_probability
        self._fetch_by_timestamp_range_script = fetch_by_timestamp_range_script

        logger.info(
            "Redis Time Window Cache Manager initialization completed: expire=%d minutes, cleanup_prob=%.1f%%",
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

    async def _cleanup_expired_if_needed(self, key: str, client) -> int:
        """
        Decide whether to perform expired data cleanup based on probability

        Args:
            key: Cache key name
            client: Redis client

        Returns:
            int: Number of cleaned data entries
        """
        if random.random() < self.cleanup_probability:
            # Calculate cleanup threshold (data before default expiration time multiplier)
            current_timestamp = int(time.time() * 1000)
            cleanup_threshold = current_timestamp - (
                self.default_expire_minutes * DEFAULT_CLEANUP_MULTIPLIER * 60 * 1000
            )

            # Clean expired data
            cleaned_count = await client.zremrangebyscore(
                key, '-inf', cleanup_threshold
            )

            if cleaned_count > 0:
                logger.debug(
                    "Randomly cleaned expired data: key=%s, cleaned=%d",
                    key,
                    cleaned_count,
                )

            return cleaned_count
        return 0

    async def append(
        self,
        key: str,
        data: Union[str, Dict, List, Any],
        expire_minutes: Optional[int] = None,
    ) -> bool:
        """
        Append data to the time window queue for the specified key

        Args:
            key: Cache key name
            data: Data to append, supports strings, dictionaries, lists, etc.
            expire_minutes: Expiration time (minutes), default to instance configuration

        Returns:
            bool: Whether operation succeeded

        Examples:
            # Append string data
            await cache.append("user_actions", "user_login")

            # Append dictionary data
            await cache.append("api_calls", {"method": "GET", "path": "/api/users"})

            # Custom expiration time
            await cache.append("temp_data", "some_data", expire_minutes=5)
        """
        try:
            client = await self.redis_provider.get_client()

            # Get current millisecond-level timestamp
            current_timestamp = int(time.time() * 1000)

            # Process data using data processor
            unique_member = RedisDataProcessor.process_data_for_storage(data)

            # Use pipeline for atomic operations
            pipe = client.pipeline()

            # 1. Add data to sorted set
            pipe.zadd(key, {unique_member: current_timestamp})

            # 2. Set or extend expiration time
            expire_seconds = (expire_minutes or self.default_expire_minutes) * 60
            pipe.expire(key, expire_seconds)

            # 3. Randomly clean expired data (using independent cleanup function)
            # Note: Cannot call within pipeline because return value is needed
            # So execute pipeline first, then perform cleanup separately

            # Execute pipeline operations
            results = await pipe.execute()

            # Check operation results
            zadd_result = results[0]  # ZADD returns number of added members
            expire_result = results[1]  # EXPIRE returns whether successful

            # Perform random cleanup separately
            cleaned_count = await self._cleanup_expired_if_needed(key, client)

            if zadd_result is not None and expire_result:
                if cleaned_count > 0:
                    logger.debug(
                        "Data appended successfully and expired data cleaned: key=%s, member_length=%d, timestamp=%d, expire=%ds, cleaned=%d",
                        key,
                        len(unique_member),
                        current_timestamp,
                        expire_seconds,
                        cleaned_count,
                    )
                else:
                    logger.debug(
                        "Data appended successfully: key=%s, member_length=%d, timestamp=%d, expire=%ds",
                        key,
                        len(unique_member),
                        current_timestamp,
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

        except Exception as e:
            logger.error("Redis data append failed: key=%s, error=%s", key, str(e))
            return False

    async def get_queue_size(self, key: str) -> int:
        """
        Get current size of specified queue

        Args:
            key: Cache key name

        Returns:
            int: Number of elements in queue
        """
        try:
            client = await self.redis_provider.get_client()
            size = await client.zcard(key)
            return size or 0
        except Exception as e:
            logger.error("Failed to get queue size: key=%s, error=%s", key, str(e))
            return 0

    async def clear_queue(self, key: str) -> bool:
        """
        Clear all data from specified queue

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
        except Exception as e:
            logger.error("Failed to clear queue: key=%s, error=%s", key, str(e))
            return False

    async def cleanup_expired(self, key: str) -> int:
        """
        Manually clean expired data from specified queue

        Args:
            key: Cache key name

        Returns:
            int: Number of cleaned data entries
        """
        try:
            client = await self.redis_provider.get_client()

            # Force cleanup (ignore probability)
            # Calculate cleanup threshold (data older than default expiration time multiplier)
            current_timestamp = int(time.time() * 1000)
            cleanup_threshold = current_timestamp - (
                self.default_expire_minutes * DEFAULT_CLEANUP_MULTIPLIER * 60 * 1000
            )

            # Clean expired data
            cleaned_count = await client.zremrangebyscore(
                key, '-inf', cleanup_threshold
            )

            if cleaned_count > 0:
                logger.info(
                    "Manually cleaned expired data: key=%s, cleaned=%d",
                    key,
                    cleaned_count,
                )

            return cleaned_count

        except Exception as e:
            logger.error("Failed to clean expired data: key=%s, error=%s", key, str(e))
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
            }

        except Exception as e:
            logger.error(
                "Failed to get queue statistics: key=%s, error=%s", key, str(e)
            )
            return {
                "key": key,
                "total_count": 0,
                "oldest_timestamp": None,
                "newest_timestamp": None,
                "ttl_seconds": -1,
                "error": str(e),
            }


# For backward compatibility, provide a default component instance
@component(name="redis_windows_cache_manager")
class DefaultRedisWindowsCacheManager:
    """Default Redis Time Window Cache Manager (backward compatibility)"""

    def __init__(self, redis_provider: RedisProvider):
        self.redis_provider = redis_provider
        self._manager = None
        self._factory = None

    async def _get_manager(self):
        """Lazy initialization of manager"""
        if self._manager is None:
            if self._factory is None:
                self._factory = RedisWindowsCacheFactory(self.redis_provider)
            self._manager = await self._factory.create_cache_manager()
        return self._manager

    async def append(
        self,
        key: str,
        data: Union[str, Dict, List, Any],
        expire_minutes: Optional[int] = None,
    ) -> bool:
        manager = await self._get_manager()
        return await manager.append(key, data, expire_minutes)

    async def get_queue_size(self, key: str) -> int:
        manager = await self._get_manager()
        return await manager.get_queue_size(key)

    async def clear_queue(self, key: str) -> bool:
        manager = await self._get_manager()
        return await manager.clear_queue(key)

    async def cleanup_expired(self, key: str) -> int:
        manager = await self._get_manager()
        return await manager.cleanup_expired(key)

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

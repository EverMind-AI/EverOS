"""
Redis Connection Provider

Technical component providing Redis connection pool management and basic operations
"""

import os
import asyncio
from typing import Optional, Union
import redis.asyncio as redis
from redis.asyncio.connection import ConnectionPool

from core.di.decorators import component
from core.observation.logger import get_logger

logger = get_logger(__name__)


@component(name="redis_provider", primary=True)
class RedisProvider:
    """Redis connection provider"""

    def __init__(self):
        """Initialize Redis connection provider"""
        # Read Redis configuration from environment variables
        self.redis_host = os.getenv("REDIS_HOST", "localhost")
        self.redis_port = int(os.getenv("REDIS_PORT", "6379"))
        self.redis_db = int(os.getenv("REDIS_DB", "0"))
        self.redis_password = os.getenv("REDIS_PASSWORD")
        self.redis_ssl = os.getenv("REDIS_SSL", "false").lower() == "true"

        # Build Redis URL
        protocol = "rediss" if self.redis_ssl else "redis"
        if self.redis_password:
            self.redis_url = f"{protocol}://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        else:
            self.redis_url = (
                f"{protocol}://{self.redis_host}:{self.redis_port}/{self.redis_db}"
            )

        # Other configurations use default values
        self.max_connections = int(os.getenv("REDIS_MAX_CONNECTIONS", "400"))
        self.socket_timeout = int(os.getenv("REDIS_SOCKET_TIMEOUT", "15"))
        self.socket_connect_timeout = int(
            os.getenv("REDIS_SOCKET_CONNECT_TIMEOUT", "5")
        )

        # Named client cache
        self._named_clients = {}
        self._named_pools = {}
        self._named_initialized = set()
        self._lock = asyncio.Lock()

    async def get_client(self) -> redis.Redis:
        """
        Get Redis client (default client)

        Returns:
            redis.Redis: Redis client instance
        """
        return await self.get_named_client("default")

    async def get_named_client(self, name: str, **overrides) -> redis.Redis:
        """
        Get named Redis client, supports parameter override

        Args:
            name: Client name, used for caching
            **overrides: Override default parameters, such as decode_responses=False

        Returns:
            redis.Redis: Redis client instance
        """
        if name in self._named_initialized:
            return self._named_clients[name]

        async with self._lock:
            # Double-checked locking
            if name in self._named_initialized:
                return self._named_clients[name]

            try:
                # Build connection parameters, using default values + override parameters
                conn_params = {
                    "max_connections": self.max_connections,
                    "socket_timeout": self.socket_timeout,
                    "socket_connect_timeout": self.socket_connect_timeout,
                    "decode_responses": True,  # default value
                }
                conn_params.update(overrides)
                logger.info(
                    "Building Redis client connection params: %s, %s", name, conn_params
                )
                # Create named connection pool
                named_pool = ConnectionPool.from_url(self.redis_url, **conn_params)

                # Create named Redis client
                named_client = redis.Redis(connection_pool=named_pool)

                # Test connection
                await named_client.ping()

                # Cache client and connection pool
                self._named_clients[name] = named_client
                self._named_pools[name] = named_pool
                self._named_initialized.add(name)

                logger.info(
                    "Named Redis client initialized successfully: %s (param overrides: %s)",
                    name,
                    overrides if overrides else "None",
                )

                return named_client

            except Exception as e:
                logger.error(
                    "Named Redis client initialization failed: %s, error=%s",
                    name,
                    str(e),
                )
                # Clean up partially initialized resources
                if name in self._named_clients:
                    await self._named_clients[name].aclose()
                    del self._named_clients[name]
                if name in self._named_pools:
                    await self._named_pools[name].aclose()
                    del self._named_pools[name]
                self._named_initialized.discard(name)
                raise

    async def set(
        self,
        key: str,
        value: Union[str, bytes, int, float],
        ex: Optional[int] = None,
        nx: bool = False,
    ) -> bool:
        """
        Set key-value pair

        Args:
            key: Key name
            value: Value
            ex: Expiration time (seconds)
            nx: If True, set only if key does not exist

        Returns:
            bool: Whether set operation succeeded
        """
        client = await self.get_client()
        try:
            result = await client.set(key, value, ex=ex, nx=nx)
            return result is not None and result is not False
        except Exception as e:
            logger.error("Redis SET operation failed: key=%s, error=%s", key, str(e))
            return False

    async def get(self, key: str) -> Optional[str]:
        """
        Get key value

        Args:
            key: Key name

        Returns:
            Optional[str]: Key value, returns None if key does not exist
        """
        client = await self.get_client()
        try:
            return await client.get(key)
        except Exception as e:
            logger.error("Redis GET operation failed: key=%s, error=%s", key, str(e))
            return None

    async def exists(self, key: str) -> bool:
        """
        Check if key exists

        Args:
            key: Key name

        Returns:
            bool: Whether key exists
        """
        client = await self.get_client()
        try:
            result = await client.exists(key)
            return result > 0
        except Exception as e:
            logger.error("Redis EXISTS operation failed: key=%s, error=%s", key, str(e))
            return False

    async def delete(self, *keys: str) -> int:
        """
        Delete keys

        Args:
            keys: List of key names to delete

        Returns:
            int: Number of keys successfully deleted
        """
        if not keys:
            return 0

        client = await self.get_client()
        try:
            return await client.delete(*keys)
        except Exception as e:
            logger.error(
                "Redis DELETE operation failed: keys=%s, error=%s", keys, str(e)
            )
            return 0

    async def expire(self, key: str, seconds: int) -> bool:
        """
        Set key expiration time

        Args:
            key: Key name
            seconds: Expiration time (seconds)

        Returns:
            bool: Whether setting expiration succeeded
        """
        client = await self.get_client()
        try:
            return await client.expire(key, seconds)
        except Exception as e:
            logger.error(
                "Redis EXPIRE operation failed: key=%s, seconds=%s, error=%s",
                key,
                seconds,
                str(e),
            )
            return False

    async def ttl(self, key: str) -> int:
        """
        Get remaining time to live of key

        Args:
            key: Key name

        Returns:
            int: Remaining TTL in seconds, -1 means never expires, -2 means key does not exist
        """
        client = await self.get_client()
        try:
            return await client.ttl(key)
        except Exception as e:
            logger.error("Redis TTL operation failed: key=%s, error=%s", key, str(e))
            return -2

    async def keys(self, pattern: str) -> list:
        """
        Get list of keys matching pattern

        Args:
            pattern: Matching pattern (e.g., "prefix:*")

        Returns:
            list: List of matching keys
        """
        client = await self.get_client()
        try:
            return await client.keys(pattern)
        except Exception as e:
            logger.error(
                "Redis KEYS operation failed: pattern=%s, error=%s", pattern, str(e)
            )
            return []

    async def ping(self) -> bool:
        """
        Test Redis connection

        Returns:
            bool: Whether connection is healthy
        """
        try:
            client = await self.get_client()
            result = await client.ping()
            return result is True
        except Exception as e:
            logger.error("Redis PING failed: %s", str(e))
            return False

    async def lpush(self, key: str, *values: Union[str, bytes]) -> int:
        """
        Push elements to the left of the list

        Args:
            key: Key name
            values: List of values to push

        Returns:
            int: Length of list after push
        """
        if not values:
            return 0

        client = await self.get_client()
        try:
            return await client.lpush(key, *values)
        except Exception as e:
            logger.error("Redis LPUSH operation failed: key=%s, error=%s", key, str(e))
            return 0

    async def lrange(self, key: str, start: int, end: int) -> list:
        """
        Get elements in specified range of list

        Args:
            key: Key name
            start: Start index
            end: End index (-1 means to end of list)

        Returns:
            list: List of elements
        """
        client = await self.get_client()
        try:
            return await client.lrange(key, start, end)
        except Exception as e:
            logger.error("Redis LRANGE operation failed: key=%s, error=%s", key, str(e))
            return []

    async def close(self):
        """Close all Redis connection pools"""
        # Close all named clients
        for name, client in self._named_clients.items():
            try:
                await client.aclose()
                logger.info("Named Redis client closed: %s", name)
            except Exception as e:
                logger.error(
                    "Failed to close named Redis client: %s, error=%s", name, str(e)
                )

        for name, pool in self._named_pools.items():
            try:
                await pool.aclose()
                logger.info("Named Redis connection pool closed: %s", name)
            except Exception as e:
                logger.error(
                    "Failed to close named Redis connection pool: %s, error=%s",
                    name,
                    str(e),
                )

        # Clear cache
        self._named_clients.clear()
        self._named_pools.clear()
        self._named_initialized.clear()

    def is_initialized(self) -> bool:
        """Check if default client is initialized"""
        return "default" in self._named_initialized

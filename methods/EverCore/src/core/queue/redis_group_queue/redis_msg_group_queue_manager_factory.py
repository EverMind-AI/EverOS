"""
Redis message group queue manager factory

Provides caching and management functionality for RedisGroupQueueManager instances based on configuration.
Supports reading configuration from environment variables, provides default and named instances.
Follows the design pattern of mongodb_client_factory.py.
"""

import os
import asyncio
from typing import Dict, Optional, Callable, Type
from core.di.decorators import component
from core.observation.logger import get_logger
from core.component.redis_provider import RedisProvider
from .redis_msg_group_queue_manager import RedisGroupQueueManager
from .redis_group_queue_item import RedisGroupQueueItem, SerializationMode

logger = get_logger(__name__)


class RedisGroupQueueConfig:
    """Redis group queue configuration class"""

    def __init__(
        self,
        key_prefix: str = "default",
        serialization_mode: SerializationMode = SerializationMode.JSON,
        sort_key_func: Optional[Callable[[RedisGroupQueueItem], int]] = None,
        max_total_messages: int = 1000,
        queue_expire_seconds: int = 12 * 3600,  # 12 hours
        activity_expire_seconds: int = 7 * 24 * 3600,  # 7 days
        enable_metrics: bool = True,
        log_interval_seconds: int = 60,
        cleanup_interval_seconds: int = 300,  # 5 minutes
        **kwargs,
    ):
        self.key_prefix = key_prefix
        self.serialization_mode = serialization_mode
        self.sort_key_func = sort_key_func
        self.max_total_messages = max_total_messages
        self.queue_expire_seconds = queue_expire_seconds
        self.activity_expire_seconds = activity_expire_seconds
        self.enable_metrics = enable_metrics
        self.log_interval_seconds = log_interval_seconds
        self.cleanup_interval_seconds = cleanup_interval_seconds
        self.kwargs = kwargs

    def get_cache_key(self) -> str:
        """
        Get cache key

        Generate unique identifier based on core configuration parameters
        """
        # Use function name or default value for sort function
        sort_func_name = (
            getattr(self.sort_key_func, '__name__', 'default')
            if self.sort_key_func
            else 'default'
        )

        return (
            f"{self.key_prefix}:{self.serialization_mode.value}:{sort_func_name}:"
            f"{self.max_total_messages}:{self.queue_expire_seconds}:"
            f"{self.activity_expire_seconds}:{self.enable_metrics}:"
            f"{self.log_interval_seconds}:{self.cleanup_interval_seconds}"
        )

    @classmethod
    def from_env(cls, prefix: str = "") -> 'RedisGroupQueueConfig':
        """
        Create configuration from environment variables

        Prefix rule: if prefix is provided, variables will be read in the format "{prefix}_XXX", otherwise "XXX" is read.
        For example: prefix="CLIENT" will read "CLIENT_REDIS_QUEUE_KEY_PREFIX", "CLIENT_REDIS_QUEUE_MAX_TOTAL_MESSAGES", etc.

        Args:
            prefix: environment variable prefix

        Returns:
            RedisGroupQueueConfig: configuration instance
        """

        def _env(name: str, default: str) -> str:
            key = f"{prefix}_{name}" if prefix else name
            return os.getenv(key, default)

        # Read configuration items
        base_key_prefix = _env("REDIS_QUEUE_KEY_PREFIX", "default")
        # Support global Redis prefix
        global_redis_prefix = _env("GLOBAL_REDIS_PREFIX", "")
        key_prefix = (
            f"{global_redis_prefix}:{base_key_prefix}"
            if global_redis_prefix
            else base_key_prefix
        )
        # Serialization mode configuration
        serialization_mode_str = _env("REDIS_QUEUE_SERIALIZATION_MODE", "json").lower()
        serialization_mode = (
            SerializationMode.JSON
            if serialization_mode_str == "json"
            else SerializationMode.BSON
        )
        max_total_messages = int(_env("REDIS_QUEUE_MAX_TOTAL_MESSAGES", "20000"))
        queue_expire_seconds = int(_env("REDIS_QUEUE_EXPIRE_SECONDS", str(24 * 3600)))
        activity_expire_seconds = int(
            _env("REDIS_QUEUE_ACTIVITY_EXPIRE_SECONDS", str(24 * 3600))
        )
        enable_metrics = _env("REDIS_QUEUE_ENABLE_METRICS", "true").lower() == "true"
        log_interval_seconds = int(_env("REDIS_QUEUE_LOG_INTERVAL_SECONDS", "600"))
        cleanup_interval_seconds = int(
            _env("REDIS_QUEUE_CLEANUP_INTERVAL_SECONDS", "300")
        )

        return cls(
            key_prefix=key_prefix,
            serialization_mode=serialization_mode,
            max_total_messages=max_total_messages,
            queue_expire_seconds=queue_expire_seconds,
            activity_expire_seconds=activity_expire_seconds,
            enable_metrics=enable_metrics,
            log_interval_seconds=log_interval_seconds,
            cleanup_interval_seconds=cleanup_interval_seconds,
        )

    def __repr__(self) -> str:
        return (
            f"RedisGroupQueueConfig(key_prefix={self.key_prefix}, "
            f"max_total_messages={self.max_total_messages})"
        )


@component(name="redis_group_queue_manager_factory", primary=True)
class RedisGroupQueueManagerFactory:
    """Redis message group queue manager factory"""

    def __init__(self, redis_provider: RedisProvider):
        """
        Initialize factory

        Args:
            redis_provider: Redis connection provider
        """
        self.redis_provider = redis_provider
        self._managers: Dict[str, RedisGroupQueueManager] = {}
        self._default_config: Optional[RedisGroupQueueConfig] = None
        self._default_manager: Optional[RedisGroupQueueManager] = None
        self._lock = asyncio.Lock()

    async def get_manager(
        self,
        config: Optional[RedisGroupQueueConfig] = None,
        item_class: Optional[Type[RedisGroupQueueItem]] = None,
        auto_start: bool = True,
        redis_client_name: str = "default",
    ) -> RedisGroupQueueManager:
        """
        Get Redis message group queue manager

        Args:
            config: queue manager configuration, use default configuration if None
            item_class: queue item type, must inherit from RedisGroupQueueItem, default uses SimpleQueueItem
            auto_start: whether to automatically start the manager
            redis_client_name: Redis client name

        Returns:
            RedisGroupQueueManager: queue manager
        """
        if config is None:
            config = await self._get_default_config()

        # Generate cache key, including item_class information
        item_class_name = item_class.__name__ if item_class else 'default'
        cache_key = f"{config.get_cache_key()}:{item_class_name}:{redis_client_name}"

        async with self._lock:
            # Check cache
            if cache_key in self._managers:
                manager = self._managers[cache_key]
                return manager

            # Create new manager
            logger.info("Creating new RedisGroupQueueManager: %s", config)

            try:
                # Get Redis client based on serialization mode
                if config.serialization_mode == SerializationMode.BSON:
                    # BSON mode: use binary_cache, do not decode responses to support byte data
                    redis_client = await self.redis_provider.get_named_client(
                        "binary_cache", decode_responses=False
                    )
                else:
                    # JSON mode: use default client, automatically decode responses
                    redis_client = await self.redis_provider.get_client()

                manager = RedisGroupQueueManager(
                    redis_client=redis_client,
                    key_prefix=config.key_prefix,
                    serialization_mode=config.serialization_mode,
                    item_class=item_class,
                    sort_key_func=config.sort_key_func,
                    max_total_messages=config.max_total_messages,
                    queue_expire_seconds=config.queue_expire_seconds,
                    activity_expire_seconds=config.activity_expire_seconds,
                    enable_metrics=config.enable_metrics,
                    log_interval_seconds=config.log_interval_seconds,
                    cleanup_interval_seconds=config.cleanup_interval_seconds,
                    **config.kwargs,
                )

                if auto_start:
                    await manager.start()

                # Cache manager
                self._managers[cache_key] = manager
                logger.info(
                    "✅ RedisGroupQueueManager created successfully and cached: %s",
                    config,
                )

                return manager

            except Exception as e:
                logger.error(
                    "❌ Failed to create RedisGroupQueueManager: %s, error: %s",
                    config,
                    e,
                )
                raise

    async def _get_default_config(self) -> RedisGroupQueueConfig:
        """Get default configuration"""
        if self._default_config is None:
            self._default_config = RedisGroupQueueConfig.from_env()
            logger.info(
                "📋 Loaded default RedisGroupQueueManager configuration: %s",
                self._default_config,
            )

        return self._default_config

    async def get_manager_with_config(
        self,
        key_prefix: str = "default",
        serialization_mode: SerializationMode = SerializationMode.JSON,
        item_class: Optional[Type[RedisGroupQueueItem]] = None,
        sort_key_func: Optional[Callable[[RedisGroupQueueItem], int]] = None,
        max_total_messages: int = 2 * 10000,
        queue_expire_seconds: int = 24 * 3600,
        activity_expire_seconds: int = 24 * 3600,
        enable_metrics: bool = True,
        log_interval_seconds: int = 600,
        cleanup_interval_seconds: int = 300,
        auto_start: bool = True,
        redis_client_name: str = "default",
        **kwargs,
    ) -> RedisGroupQueueManager:
        """
        Create manager with specified configuration

        Args:
            key_prefix: Redis key prefix, used to distinguish different manager instances
            serialization_mode: serialization mode (JSON or BSON)
            item_class: queue item type, must inherit from RedisGroupQueueItem, default uses SimpleQueueItem
            sort_key_func: sort key generation function
            max_total_messages: maximum total message count
            queue_expire_seconds: queue expiration time
            activity_expire_seconds: activity record expiration time
            enable_metrics: whether to enable metrics
            log_interval_seconds: log interval
            cleanup_interval_seconds: cleanup interval
            auto_start: whether to auto start
            redis_client_name: Redis client name
            **kwargs: additional parameters

        Returns:
            RedisGroupQueueManager: queue manager
        """
        config = RedisGroupQueueConfig(
            key_prefix=key_prefix,
            serialization_mode=serialization_mode,
            sort_key_func=sort_key_func,
            max_total_messages=max_total_messages,
            queue_expire_seconds=queue_expire_seconds,
            activity_expire_seconds=activity_expire_seconds,
            enable_metrics=enable_metrics,
            log_interval_seconds=log_interval_seconds,
            cleanup_interval_seconds=cleanup_interval_seconds,
            **kwargs,
        )

        return await self.get_manager(config, item_class, auto_start, redis_client_name)

    async def stop_manager(
        self,
        config: Optional[RedisGroupQueueConfig] = None,
        item_class: Optional[Type[RedisGroupQueueItem]] = None,
        redis_client_name: str = "default",
    ):
        """
        Stop specified manager

        Args:
            config: configuration, if None then stop default manager
            item_class: queue item type, must inherit from RedisGroupQueueItem
            redis_client_name: Redis client name
        """
        if config is None:
            if self._default_manager:
                await self._default_manager.shutdown()
                return

        # Generate cache key, including item_class information
        item_class_name = item_class.__name__ if item_class else 'default'
        cache_key = f"{config.get_cache_key()}:{item_class_name}:{redis_client_name}"

        async with self._lock:
            if cache_key in self._managers:
                await self._managers[cache_key].shutdown()

    async def stop_all_managers(self):
        """Stop all managers"""
        async with self._lock:
            for manager in self._managers.values():
                await manager.shutdown()

            self._managers.clear()

            if self._default_manager:
                self._default_manager = None

            logger.info("🔌 All RedisGroupQueueManager instances have been stopped")

    def get_cached_managers_info(self) -> Dict[str, Dict]:
        """Get cached manager information"""
        return {
            cache_key: {
                "key_prefix": manager.key_prefix,
                "max_total_messages": manager.max_total_messages,
                "manager_stats": "Need to call get_manager_stats() asynchronously to retrieve",
            }
            for cache_key, manager in self._managers.items()
        }

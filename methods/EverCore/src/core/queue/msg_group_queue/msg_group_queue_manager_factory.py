"""
Message group queue manager factory

Provides caching and management of MsgGroupQueueManager instances based on configuration.
Supports reading configuration from environment variables, provides default and named instances.
Refer to the design pattern in mongodb_client_factory.py.
"""

import os
import asyncio
from typing import Dict, Optional
from core.di.decorators import component
from core.observation.logger import get_logger
from .msg_group_queue_manager import MsgGroupQueueManager

logger = get_logger(__name__)


class MsgGroupQueueConfig:
    """Message group queue configuration class"""

    def __init__(
        self,
        name: str = "default",
        num_queues: int = 10,
        max_total_messages: int = 100,
        enable_metrics: bool = True,
        log_interval_seconds: int = 30,
        **kwargs,
    ):
        self.name = name
        self.num_queues = num_queues
        self.max_total_messages = max_total_messages
        self.enable_metrics = enable_metrics
        self.log_interval_seconds = log_interval_seconds
        self.kwargs = kwargs

    def get_cache_key(self) -> str:
        """
        Get cache key

        Generate unique identifier based on core configuration parameters
        """
        return f"{self.name}:{self.num_queues}:{self.max_total_messages}:{self.enable_metrics}:{self.log_interval_seconds}"

    @classmethod
    def from_env(cls, prefix: str = "") -> 'MsgGroupQueueConfig':
        """
        Create configuration from environment variables

        Prefix rule: if prefix is provided, read variables in the format "{prefix}_XXX", otherwise read "XXX".
        For example: prefix="CLIENT" reads "CLIENT_MSG_QUEUE_NUM_QUEUES", "CLIENT_MSG_QUEUE_MAX_TOTAL_MESSAGES", etc.

        Args:
            prefix: environment variable prefix

        Returns:
            MsgGroupQueueConfig: configuration instance
        """

        def _env(name: str, default: str) -> str:
            key = f"{prefix}_{name}" if prefix else name
            return os.getenv(key, default)

        # Read configuration items
        name = _env("MSG_QUEUE_NAME", "default")
        num_queues = int(_env("MSG_QUEUE_NUM_QUEUES", "10"))
        max_total_messages = int(_env("MSG_QUEUE_MAX_TOTAL_MESSAGES", "100"))
        enable_metrics = _env("MSG_QUEUE_ENABLE_METRICS", "true").lower() == "true"
        log_interval_seconds = int(_env("MSG_QUEUE_LOG_INTERVAL_SECONDS", "30"))

        return cls(
            name=name,
            num_queues=num_queues,
            max_total_messages=max_total_messages,
            enable_metrics=enable_metrics,
            log_interval_seconds=log_interval_seconds,
        )

    def __repr__(self) -> str:
        return (
            f"MsgGroupQueueConfig(name={self.name}, "
            f"num_queues={self.num_queues}, "
            f"max_total_messages={self.max_total_messages})"
        )


@component(name="msg_group_queue_manager_factory", primary=True)
class MsgGroupQueueManagerFactory:
    """Message group queue manager factory"""

    def __init__(self):
        """Initialize factory"""
        self._managers: Dict[str, MsgGroupQueueManager] = {}
        self._default_config: Optional[MsgGroupQueueConfig] = None
        self._default_manager: Optional[MsgGroupQueueManager] = None
        self._lock = asyncio.Lock()

    async def get_manager(
        self, config: Optional[MsgGroupQueueConfig] = None, auto_start: bool = True
    ) -> MsgGroupQueueManager:
        """
        Get message group queue manager

        Args:
            config: queue manager configuration, use default if None
            auto_start: whether to automatically start the manager

        Returns:
            MsgGroupQueueManager: queue manager
        """
        if config is None:
            config = await self._get_default_config()

        cache_key = config.get_cache_key()

        async with self._lock:
            # Check cache
            if cache_key in self._managers:
                manager = self._managers[cache_key]
                return manager

            # Create new manager
            logger.info("Creating new MsgGroupQueueManager: %s", config)

            try:
                manager = MsgGroupQueueManager(
                    name=config.name,
                    num_queues=config.num_queues,
                    max_total_messages=config.max_total_messages,
                    enable_metrics=config.enable_metrics,
                    log_interval_seconds=config.log_interval_seconds,
                    **config.kwargs,
                )

                if auto_start:
                    await manager.start_periodic_logging()

                # Cache manager
                self._managers[cache_key] = manager
                logger.info(
                    "✅ MsgGroupQueueManager created successfully and cached: %s",
                    config,
                )

                return manager

            except Exception as e:
                logger.error(
                    "❌ Failed to create MsgGroupQueueManager: %s, error: %s", config, e
                )
                raise

    async def get_default_manager(
        self, auto_start: bool = True
    ) -> MsgGroupQueueManager:
        """
        Get default message group queue manager

        Args:
            auto_start: whether to automatically start the manager

        Returns:
            MsgGroupQueueManager: default queue manager
        """
        if self._default_manager is None:
            config = await self._get_default_config()
            self._default_manager = await self.get_manager(config, auto_start)

        return self._default_manager

    async def get_named_manager(
        self, name: str, auto_start: bool = True
    ) -> MsgGroupQueueManager:
        """
        Get message group queue manager by name

        Convention: use name as environment variable prefix, read configuration from "{name}_MSG_QUEUE_XXX".
        For example, when name="CLIENT", it will try to read "CLIENT_MSG_QUEUE_NUM_QUEUES", "CLIENT_MSG_QUEUE_MAX_TOTAL_MESSAGES", etc.

        Args:
            name: prefix name (i.e., environment variable prefix)
            auto_start: whether to automatically start the manager

        Returns:
            MsgGroupQueueManager: queue manager
        """
        if name.lower() == "default":
            return await self.get_default_manager(auto_start)

        config = MsgGroupQueueConfig.from_env(prefix=name)
        # Ensure configuration name matches requested name
        config.name = name.lower()

        logger.info(
            "📋 Loading named MsgGroupQueueManager configuration[name=%s]: %s",
            name,
            config,
        )
        return await self.get_manager(config, auto_start)

    async def _get_default_config(self) -> MsgGroupQueueConfig:
        """Get default configuration"""
        if self._default_config is None:
            self._default_config = MsgGroupQueueConfig.from_env()
            logger.info(
                "📋 Loaded default MsgGroupQueueManager configuration: %s",
                self._default_config,
            )

        return self._default_config

    async def create_manager_with_config(
        self,
        name: str = "default",
        num_queues: int = 10,
        max_total_messages: int = 100,
        enable_metrics: bool = True,
        log_interval_seconds: int = 30,
        auto_start: bool = True,
        **kwargs,
    ) -> MsgGroupQueueManager:
        """
        Create manager with specified configuration

        Args:
            name: manager name
            num_queues: number of queues
            max_total_messages: maximum total number of messages
            enable_metrics: whether to enable metrics
            log_interval_seconds: logging interval
            auto_start: whether to automatically start
            **kwargs: additional parameters

        Returns:
            MsgGroupQueueManager: queue manager
        """
        config = MsgGroupQueueConfig(
            name=name,
            num_queues=num_queues,
            max_total_messages=max_total_messages,
            enable_metrics=enable_metrics,
            log_interval_seconds=log_interval_seconds,
            **kwargs,
        )

        return await self.get_manager(config, auto_start)

    async def stop_manager(self, config: Optional[MsgGroupQueueConfig] = None):
        """
        Stop specified manager

        Args:
            config: configuration, if None then stop default manager
        """
        if config is None:
            if self._default_manager:
                await self._default_manager.shutdown()
                return

        cache_key = config.get_cache_key()

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

            logger.info("🔌 All MsgGroupQueueManager instances have been stopped")

    def get_cached_managers_info(self) -> Dict[str, Dict]:
        """Get cached manager information"""
        return {
            cache_key: {
                "name": manager.name,
                "num_queues": manager.num_queues,
                "max_total_messages": manager.max_total_messages,
                "manager_stats": "Need to call get_manager_stats() asynchronously to retrieve",
            }
            for cache_key, manager in self._managers.items()
        }

    async def get_default_msg_group_queue_manager(
        self, auto_start: bool = True
    ) -> MsgGroupQueueManager:
        """
        Convenience function to get default message group queue manager

        Args:
            auto_start: whether to automatically start the manager

        Returns:
            MsgGroupQueueManager: default queue manager
        """
        return await self.get_default_manager(auto_start)

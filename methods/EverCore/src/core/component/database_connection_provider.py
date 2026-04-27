"""
Database connection provider

Responsible for managing PostgreSQL connection pool and LangGraph checkpoint saver
"""

import os
from typing import Optional, Tuple
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row

from core.di.decorators import component
from core.observation.logger import get_logger
from common_utils.datetime_utils import get_timezone

logger = get_logger(__name__)


@component(name="database_connection_provider", primary=True)
class DatabaseConnectionProvider:
    """Database connection provider"""

    def __init__(self):
        """Initialize database connection provider"""
        self.database_url = os.getenv("DATABASE_URL")
        if not self.database_url:
            raise ValueError(
                "Database connection string DATABASE_URL is not configured"
            )

        # Read timezone configuration from environment variables
        self.timezone = get_timezone()

        # Connection pool configuration
        self.max_size = int(os.getenv("CHECKPOINTER_DB_POOL_SIZE", "20"))

        # Do not create connection pool during initialization, delay until needed
        self._connection_pool: Optional[AsyncConnectionPool] = None
        self._checkpointer: Optional[AsyncPostgresSaver] = None
        self._is_initialized = False

    async def _ensure_initialized(self):
        """Ensure connection pool is initialized"""
        if self._is_initialized:
            return

        logger.info("Initializing database connection pool...")

        # Connection parameters configuration
        connection_kwargs = {
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": dict_row,  # Add row_factory to match type
            "options": f"-c timezone={self.timezone}",  # Set connection timezone
        }

        # Create connection pool
        self._connection_pool = AsyncConnectionPool(
            conninfo=self.database_url,
            max_size=self.max_size,
            open=False,  # Do not open in constructor
            kwargs=connection_kwargs,
        )

        logger.info(
            "Database connection pool created successfully %s", self.database_url
        )

        # Explicitly open connection pool
        await self._connection_pool.open()
        logger.info(
            "Database connection pool initialized successfully, timezone set to: %s",
            self.timezone,
        )

        # Initialize checkpointer
        self._checkpointer = AsyncPostgresSaver(self._connection_pool)  # type: ignore
        await self._checkpointer.setup()
        logger.info("Checkpointer setup completed")

        self._is_initialized = True

    async def get_connection_pool(self) -> AsyncConnectionPool:
        """
        Get database connection pool

        Returns:
            AsyncConnectionPool: Database connection pool instance
        """
        await self._ensure_initialized()
        return self._connection_pool

    async def get_checkpointer(self) -> AsyncPostgresSaver:
        """
        Get LangGraph checkpoint saver

        Returns:
            AsyncPostgresSaver: Checkpoint saver instance
        """
        await self._ensure_initialized()
        return self._checkpointer

    async def get_connection_and_checkpointer(
        self,
    ) -> Tuple[AsyncConnectionPool, AsyncPostgresSaver]:
        """
        Get connection pool and checkpoint saver

        Returns:
            tuple: (connection pool, checkpoint saver)
        """
        await self._ensure_initialized()
        return self._connection_pool, self._checkpointer

    async def close(self):
        """Close database connection pool"""
        if self._connection_pool:
            await self._connection_pool.close()
            logger.info("Database connection pool has been closed")
            self._connection_pool = None
            self._checkpointer = None
            self._is_initialized = False

    def is_initialized(self) -> bool:
        """Check if initialized"""
        return self._is_initialized

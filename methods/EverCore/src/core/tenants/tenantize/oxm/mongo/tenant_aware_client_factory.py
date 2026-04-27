"""
Tenant-aware MongoDB Client Factory

This module provides a tenant-isolated MongoDB client factory implementation based on tenant context management.
The Factory is only responsible for creating and caching clients; mode determination is handled internally by TenantAwareMongoClient.
"""

import asyncio
from typing import Optional

from core.observation.logger import get_logger
from core.di.decorators import component
from core.component.mongodb_client_factory import (
    MongoDBClientFactory,
    MongoDBConfig,
    MongoDBClientWrapper,
)
from core.tenants.tenantize.oxm.mongo.config_utils import generate_tenant_database_name
from core.tenants.tenantize.oxm.mongo.tenant_aware_mongo_client import (
    TenantAwareMongoClient,
)

logger = get_logger(__name__)


@component(name="tenant_aware_mongodb_client_factory", primary=True)
class TenantAwareMongoDBClientFactory(MongoDBClientFactory):
    """
    Tenant-aware MongoDB Client Factory Implementation

    This factory class is responsible for creating and managing tenant-aware MongoDB clients.
    All logic regarding tenant mode vs non-tenant mode is handled internally by TenantAwareMongoClient.
    The Factory only handles simple creation, caching, and lifecycle management.

    Marked with primary=True, serving as the system's default MongoDB client factory.
    """

    def __init__(self):
        """Initialize the tenant-aware client factory"""
        # Tenant-aware client wrapper (global singleton)
        self._client_wrapper: Optional[MongoDBClientWrapper] = None

        # Lock for concurrent access protection
        self._lock = asyncio.Lock()

        logger.info("🏭 Tenant-aware MongoDB client factory initialized (primary=True)")

    async def get_client(
        self, config: Optional[MongoDBConfig] = None, **connection_kwargs
    ) -> MongoDBClientWrapper:
        """
        Get MongoDB client

        Returns a tenant-aware client wrapper. Mode determination is handled internally by TenantAwareMongoClient.

        Args:
            config: MongoDB configuration (retained for interface compatibility; actual configuration is provided via tenant context or environment variables)
            **connection_kwargs: Additional connection parameters

        Returns:
            MongoDBClientWrapper: MongoDB client wrapper
        """
        return await self._get_client_wrapper()

    async def _get_client_wrapper(self) -> MongoDBClientWrapper:
        """
        Get tenant-aware client wrapper (internal method)

        Singleton pattern: only one client wrapper instance is created for the entire factory.

        Returns:
            MongoDBClientWrapper: Wrapper containing the tenant-aware client
        """
        if self._client_wrapper is None:
            async with self._lock:
                # Double-check
                if self._client_wrapper is None:
                    logger.info("🔧 Creating tenant-aware MongoDB client wrapper")

                    # Create tenant-aware MongoDB client
                    tenant_aware_client = TenantAwareMongoClient()

                    # Create a dummy configuration (for interface compatibility)
                    dummy_config = MongoDBConfig(
                        host="tenant-aware",
                        port=27017,
                        database=generate_tenant_database_name(),
                    )

                    # Wrap into MongoDBClientWrapper
                    self._client_wrapper = TenantAwareClientWrapper(
                        tenant_aware_client, dummy_config
                    )

                    logger.info("✅ Tenant-aware MongoDB client wrapper created")

        return self._client_wrapper

    async def get_default_client(self) -> MongoDBClientWrapper:
        """
        Get default MongoDB client

        Returns:
            MongoDBClientWrapper: Default MongoDB client wrapper
        """
        return await self._get_client_wrapper()

    async def get_named_client(self, name: str) -> MongoDBClientWrapper:
        """
        Get MongoDB client by name

        Note: In the current implementation, the name parameter is ignored because tenant information is obtained from context.
        This method is retained for interface compatibility.

        Args:
            name: Prefix name (retained for interface compatibility)

        Returns:
            MongoDBClientWrapper: MongoDB client wrapper
        """
        logger.info("📋 Getting named client name=%s (tenant-aware mode)", name)
        return await self._get_client_wrapper()

    async def create_client_with_config(
        self,
        host: str = "localhost",
        port: int = 27017,
        username: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
        **kwargs,
    ) -> MongoDBClientWrapper:
        """
        Create client with specified configuration

        Note: In tenant-aware mode, configuration parameters are passed to TenantAwareMongoClient for non-tenant mode usage.
        In tenant mode, these parameters are ignored.

        Args:
            host: MongoDB host
            port: MongoDB port
            username: Username
            password: Password
            database: Database name
            **kwargs: Other connection parameters

        Returns:
            MongoDBClientWrapper: MongoDB client wrapper
        """
        if database is None:
            database = generate_tenant_database_name()
        logger.info(
            "📋 Creating client with specified configuration (tenant-aware mode): host=%s, port=%s, database=%s",
            host,
            port,
            database,
        )
        # In tenant-aware mode, configuration parameters are passed to TenantAwareMongoClient
        # If non-tenant mode is enabled, TenantAwareMongoClient will use these parameters
        return await self._get_client_wrapper()

    async def close_client(self, config: Optional[MongoDBConfig] = None):
        """
        Close specified client

        In tenant-aware mode, close the global client wrapper.

        Args:
            config: Configuration (retained for interface compatibility)
        """
        async with self._lock:
            if self._client_wrapper:
                await self._client_wrapper.close()
                self._client_wrapper = None
                logger.info("🔌 MongoDB client closed (tenant-aware factory)")

    async def close_all_clients(self):
        """Close all clients"""
        await self.close_client()


class TenantAwareClientWrapper(MongoDBClientWrapper):
    """
    Tenant-aware Client Wrapper

    Inherits from MongoDBClientWrapper, adapted for tenant-aware MongoDB client.
    Provides the same interface as MongoDBClientWrapper but uses TenantAwareMongoClient internally.
    """

    def __init__(
        self, tenant_aware_client: TenantAwareMongoClient, config: MongoDBConfig
    ):
        """
        Initialize tenant-aware client wrapper

        Args:
            tenant_aware_client: Tenant-aware MongoDB client
            config: MongoDB configuration (for compatibility)
        """
        # Directly set attributes without calling parent __init__
        self.client = tenant_aware_client
        self.config = config
        self._initialized = False
        self._document_models = []

    @property
    def database(self):
        """
        Get database object

        Returns the tenant-aware database proxy.
        """
        return self.client[self.config.database]

    async def test_connection(self) -> bool:
        """
        Test connection

        Note: In tenant mode, this must be called with a tenant context.
        In non-tenant mode, the provided configuration is used for testing.
        """
        try:
            # TenantAwareMongoClient will select the correct client based on configuration and context
            real_client = await self.client._get_real_client()
            await real_client.admin.command('ping')
            logger.info("✅ MongoDB connection test succeeded (tenant-aware)")
            return True
        except Exception as e:
            logger.error("❌ MongoDB connection test failed (tenant-aware): %s", e)
            return False

    async def close(self):
        """Close all connections"""
        if self.client:
            await self.client.close()
            logger.info("🔌 MongoDB connection closed (tenant-aware)")

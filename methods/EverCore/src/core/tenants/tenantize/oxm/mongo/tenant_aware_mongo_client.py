"""
Tenant-aware MongoDB Client Proxy

This module implements tenant-aware proxy versions of AsyncMongoClient and AsyncDatabase.
Core functionality: intercept all method calls and dynamically switch to the corresponding real client/database based on tenant context.
"""

from typing import Dict, Optional, Any
from pymongo.asynchronous.mongo_client import AsyncMongoClient
from pymongo.asynchronous.database import AsyncDatabase
from pymongo.asynchronous.collection import AsyncCollection

from core.observation.logger import get_logger
from core.tenants.tenant_contextvar import get_current_tenant
from core.tenants.tenant_config import get_tenant_config
from core.tenants.tenant_models import TenantPatchKey
from common_utils.datetime_utils import timezone
from core.tenants.tenantize.oxm.mongo.config_utils import (
    get_tenant_mongo_config,
    get_mongo_client_cache_key,
    load_mongo_config_from_env,
    generate_tenant_database_name,
)
from core.tenants.tenantize.tenant_cache_utils import get_or_compute_tenant_cache

logger = get_logger(__name__)


class TenantAwareMongoClient(AsyncMongoClient):
    """
    Tenant-aware AsyncMongoClient Proxy

    This class intercepts all calls to AsyncMongoClient via the proxy pattern,
    dynamically switching to the real MongoDB client corresponding to the current tenant context.

    Core features:
    1. Efficient caching: caches client instances per tenant to avoid redundant creation
    2. Tenant isolation: different tenants use separate client connections
    3. Non-tenant mode support: tenant functionality can be disabled via configuration, falling back to traditional mode
    4. Default client support: in tenant mode, automatically uses the default client (read from environment variables) when no tenant context exists
    5. Type compatibility: ensures compatibility with pymongo and beanie type checks through virtual subclass registration

    Usage examples:
        >>> # Tenant mode (reads configuration from tenant context)
        >>> client = TenantAwareMongoClient()
        >>> db = client["my_database"]

        >>> # Tenant mode without tenant context (uses default client)
        >>> # Reads default configuration from environment variables MONGODB_*
        >>> client = TenantAwareMongoClient()
        >>> db = client["my_database"]  # uses default client

        >>> # Non-tenant mode (uses traditional parameters)
        >>> client = TenantAwareMongoClient(
        ...     host="localhost",
        ...     port=27017,
        ...     username="admin",
        ...     password="password"
        ... )
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize tenant-aware client

        Args:
            host: MongoDB host address (used only in non-tenant mode)
            port: MongoDB port (used only in non-tenant mode)
            username: Username (used only in non-tenant mode)
            password: Password (used only in non-tenant mode)
            **kwargs: Other MongoDB client parameters

        Cache design:
            - self._client_cache: The actual storage location for client instances (main cache)
            - tenant_info_patch: Stores quick references (cache_key) for fast lookup of which cached client to use
        """
        # Client cache: based on connection parameters (host/port/username/password)
        # This is the main cache that actually stores client instances
        # Different tenants with the same configuration can reuse the same client instance
        # {cache_key: AsyncMongoClient}
        self._client_cache: Dict[str, AsyncMongoClient] = {}

        # Fallback client
        # Usage:
        # 1. Used in non-tenant mode (configuration from constructor parameters)
        # 2. Used in tenant mode when no tenant context exists (configuration from environment variables)
        # Note: An instance will only be in one mode, so these two cases won't occur simultaneously
        self._fallback_client: Optional[AsyncMongoClient] = None

        # Configuration for fallback client
        # Prioritize constructor parameters; if absent, read from environment variables
        self._fallback_config: Optional[Dict[str, Any]] = None
        if host or port or username or password:
            # Constructor parameters take precedence (for non-tenant mode)
            self._fallback_config = {
                "host": host or "localhost",
                "port": port or 27017,
                "username": username,
                "password": password,
                **kwargs,
            }

        # Configuration object
        self._config = get_tenant_config()

    def get_real_client(self) -> AsyncMongoClient:
        """
        Get the real MongoDB client (public method)

        Decides which client to return based on configuration and context:
        1. If non-tenant mode is enabled, return the traditional client
        2. If tenant mode is enabled, return the corresponding tenant client based on current tenant context
        3. If no tenant context exists in tenant mode, return the default client (read from environment variables)

        Optimization strategy:
        - Main cache: self._client_cache stores actual client instances (based on connection parameters)
        - Quick reference: tenant_info_patch stores client references for fast access
        - Different tenants with the same connection configuration will reuse the same client instance

        Note: Creating an AsyncMongoClient object itself is synchronous; only subsequent method calls are asynchronous.

        Returns:
            AsyncMongoClient: The real MongoDB client instance

        Raises:
            RuntimeError: When in non-tenant mode but connection parameters are not provided, or tenant configuration is missing
        """

        def compute_client() -> AsyncMongoClient:
            """Compute (get or create) the tenant's MongoDB client"""
            # Get MongoDB configuration from tenant configuration
            mongo_config = get_tenant_mongo_config()
            if not mongo_config:
                tenant_info = get_current_tenant()
                raise RuntimeError(
                    f"Tenant {tenant_info.tenant_id} is missing MongoDB configuration information. "
                    f"Ensure the tenant information includes storage_info.mongodb configuration."
                )

            # Generate cache key based on connection parameters
            cache_key = get_mongo_client_cache_key(mongo_config)

            # Get from main cache
            if cache_key in self._client_cache:
                logger.debug("🔍 Main cache hit [cache_key=%s]", cache_key)
                return self._client_cache[cache_key]

            # Double-check (prevent concurrent creation)
            if cache_key in self._client_cache:
                return self._client_cache[cache_key]

            # Create new client
            logger.info("🔧 Creating MongoDB client [cache_key=%s]", cache_key)
            client = self._create_client_from_config(mongo_config)

            # Cache in main cache
            self._client_cache[cache_key] = client
            logger.info("✅ MongoDB client cached [cache_key=%s]", cache_key)

            return client

        return get_or_compute_tenant_cache(
            patch_key=TenantPatchKey.MONGO_CLIENT_CACHE_KEY,
            compute_func=compute_client,
            fallback=lambda: self._get_fallback_client(),
            cache_description="MongoDB client",
        )

    def _get_fallback_client(self) -> AsyncMongoClient:
        """
        Get fallback client

        The fallback client is used in two scenarios:
        1. Non-tenant mode: uses configuration provided via constructor parameters
        2. Tenant mode without tenant context: uses configuration from environment variables

        Configuration priority:
        - If constructor parameter configuration exists (self._fallback_config), use it
        - Otherwise, load configuration from environment variables

        Returns:
            AsyncMongoClient: Fallback client instance

        Raises:
            RuntimeError: In non-tenant mode, if connection parameters are not provided and cannot be read from environment variables
        """
        # Check cache
        if self._fallback_client is not None:
            return self._fallback_client

        # Get fallback configuration
        if self._fallback_config is None:
            # No constructor parameter configuration, load from environment variables
            self._fallback_config = load_mongo_config_from_env()

        # Raise error if no configuration available
        if not self._fallback_config:
            raise RuntimeError(
                "MongoDB connection parameters not available. "
                "Please set environment variables MONGODB_* or pass host/port "
                "when creating TenantAwareMongoClient."
            )

        # Create fallback client
        logger.info("🔧 Creating fallback MongoDB client")
        self._fallback_client = self._create_client_from_config(self._fallback_config)
        logger.info("✅ Fallback MongoDB client created")

        return self._fallback_client

    def _create_client_from_config(self, config: Dict[str, Any]) -> AsyncMongoClient:
        """
        Create MongoDB client from configuration

        This is the SINGLE factory method for all real AsyncMongoClient instances
        (both tenant clients and fallback client). Tenant command interceptor and
        guard listener are automatically installed here to guarantee no client
        can escape tenant isolation.

        Args:
            config: Configuration dictionary containing fields like host, port, username, password, or uri

        Returns:
            AsyncMongoClient: Created client instance with interceptor installed
        """
        # Build connection parameters (including timezone and timeout settings)
        conn_kwargs = {
            "serverSelectionTimeoutMS": 10000,  # PyMongo AsyncMongoClient requires longer timeout
            "connectTimeoutMS": 10000,  # Connection timeout
            "socketTimeoutMS": 10000,  # Socket timeout
            "maxPoolSize": 50,
            "minPoolSize": 5,
            "tz_aware": True,  # Enable timezone awareness
            "tzinfo": timezone,  # Set timezone information
        }

        # Prioritize uri if provided
        uri = config.get("uri")
        if uri:
            # Merge extra parameters (excluding uri and database)
            extra_kwargs = {
                k: v for k, v in config.items() if k not in ("uri", "database")
            }
            # User-provided parameters have higher priority
            conn_kwargs.update(extra_kwargs)
            client = AsyncMongoClient(uri, **conn_kwargs)
        else:
            # Build connection parameters
            host = config.get("host", "localhost")
            port = config.get("port", 27017)
            username = config.get("username")
            password = config.get("password")

            # Build connection string
            if username and password:
                from urllib.parse import quote_plus

                encoded_username = quote_plus(username)
                encoded_password = quote_plus(password)
                connection_string = (
                    f"mongodb://{encoded_username}:{encoded_password}@{host}:{port}"
                )
            else:
                connection_string = f"mongodb://{host}:{port}"

            # Merge extra parameters
            extra_kwargs = {
                k: v
                for k, v in config.items()
                if k not in ("host", "port", "username", "password", "database")
            }
            # User-provided parameters have higher priority
            conn_kwargs.update(extra_kwargs)

            # Create client
            client = AsyncMongoClient(connection_string, **conn_kwargs)

        # Install tenant command interceptor + guard listener on every client.
        # This is the chokepoint: all clients (tenant + fallback) are created here,
        # so no client can escape without interceptor.
        self._install_interceptor(client)

        return client

    def _install_interceptor(self, client: AsyncMongoClient) -> None:
        """
        Install tenant command interceptor and guard listener on a client.

        Always installs — interceptor checks tenant context at runtime
        and passes through when no tenant_id is present.
        """
        from core.tenants.tenantize.oxm.mongo.tenant_field_command_interceptor import (
            install_tenant_interceptor,
        )

        install_tenant_interceptor(client)
        logger.info("Tenant interceptor installed on client %s", id(client))

    def __getitem__(self, key: str) -> "TenantAwareDatabase":
        """
        Support dictionary-style database access

        Returns a tenant-aware TenantAwareDatabase object.
        The database name will be dynamically determined according to tenant configuration; the key parameter is used only as a fallback.

        Args:
            key: Requested database name (used as fallback only when tenant configuration does not specify a database)

        Returns:
            TenantAwareDatabase: Tenant-aware MongoDB Database object
        """
        # Return tenant-aware database object (do not pass key, as it will be dynamically obtained)
        return TenantAwareDatabase(self)

    def __getattr__(self, name: str) -> Any:
        """
        Intercept attribute access (fallback mechanism)

        This method is called only when an attribute is not found, used to proxy to the real MongoDB client.
        This allows specific methods to be overridden without affecting proxy functionality.

        Args:
            name: Attribute name

        Returns:
            Any: Proxied attribute or method
        """
        # Get real client (synchronous)
        real_client = self.get_real_client()
        # Directly return the attribute or method from the real client
        return getattr(real_client, name)

    async def close(self):
        """
        Close all client connections

        Clean up all cached clients (main cache) and the fallback client.

        Note:
        - Main cache self._client_cache stores actual client instances and requires lifecycle management
        - tenant_info_patch only stores quick references (cache_key) and does not need cleanup
        """
        # Close all cached clients (main cache)
        for cache_key, client in self._client_cache.items():
            try:
                await client.close()
                logger.info("🔌 MongoDB client closed [cache_key=%s]", cache_key)
            except Exception as e:
                logger.error(
                    "❌ Failed to close client [cache_key=%s]: %s", cache_key, e
                )

        self._client_cache.clear()

        # Close fallback client
        if self._fallback_client:
            try:
                await self._fallback_client.close()
                logger.info("🔌 Fallback MongoDB client closed")
            except Exception as e:
                logger.error("❌ Failed to close fallback client: %s", e)

            self._fallback_client = None


class TenantAwareDatabase(AsyncDatabase):
    """
    Tenant-aware AsyncDatabase Proxy

    This class intercepts all calls to AsyncDatabase via the proxy pattern,
    dynamically switching to the real database object corresponding to the current tenant context.

    Core features:
    1. Tenant isolation: different tenants use different database instances
    2. Transparent proxy: all database operations are automatically routed to the correct tenant database
    3. Dynamic database name: database name is dynamically obtained based on tenant context
    4. Type compatibility: inherits from AsyncDatabase, ensuring compatibility with pymongo and beanie type checks

    Usage examples:
        >>> client = TenantAwareMongoClient()
        >>> db = client["my_database"]  # Returns TenantAwareDatabase
        >>> collection = db["my_collection"]  # Automatically routed to correct tenant
        >>> # In different tenant contexts, db.name will return different database names
    """

    def __init__(self, client: TenantAwareMongoClient):
        """
        Initialize tenant-aware database

        Args:
            client: Tenant-aware MongoDB client

        Note:
            - Database name is not stored; it is dynamically obtained on each access
            - This ensures the correct database is used in different tenant contexts
        """
        # Save client reference
        # Note: Do not call parent class __init__ as we want to fully proxy behavior
        self._tenant_aware_client = client

    def _get_real_database(self) -> AsyncDatabase:
        """
        Get the real MongoDB database object (with caching)

        Obtain the real client through the tenant-aware client, then access the corresponding database.
        The database name is dynamically obtained according to tenant configuration, ensuring each tenant uses the correct database.

        Optimization: The database object is cached in tenant_info_patch to avoid repeated creation

        Note: A tenant has only one database configuration, so a fixed patch_key is used

        Returns:
            AsyncDatabase: The real MongoDB Database object
        """

        def compute_database() -> AsyncDatabase:
            """Compute database object"""
            actual_database_name = self._get_actual_database_name()
            real_client = self._tenant_aware_client.get_real_client()
            return real_client[actual_database_name]

        return get_or_compute_tenant_cache(
            patch_key=TenantPatchKey.MONGO_REAL_DATABASE,
            compute_func=compute_database,
            fallback=compute_database,  # fallback logic is the same, reuse directly
            cache_description="MongoDB database object",
        )

    def _get_actual_database_name(self) -> str:
        """
        Get the actual database name (dynamically obtained, with caching)

        Dynamically obtain the real database name based on current tenant configuration:
        1. In tenant mode, read the database name from tenant configuration (must be specified, no fallback)
        2. In non-tenant mode, read the default database name from environment variables
        3. If no tenant context exists, read the default database name from environment variables

        Optimization: The database name is cached in tenant_info_patch to avoid repeated computation

        Returns:
            str: The actual database name

        Raises:
            RuntimeError: In tenant mode if tenant configuration is missing or database name is not specified
        """

        def compute_database_name() -> str:
            """Compute database name"""
            # Use common function to get tenant MongoDB configuration
            mongo_config = get_tenant_mongo_config()
            if not mongo_config:
                tenant_info = get_current_tenant()
                raise RuntimeError(
                    f"Tenant {tenant_info.tenant_id} is missing MongoDB configuration information. "
                    f"Ensure the tenant information includes storage_info.mongodb configuration."
                )

            # Get database name from configuration
            database_name = mongo_config.get("database")
            if not database_name:
                # In tenant mode, database name must be specified; cannot fall back to default
                tenant_info = get_current_tenant()
                raise RuntimeError(
                    f"Database name not specified in MongoDB configuration for tenant {tenant_info.tenant_id}. "
                    f"Please specify the database name in storage_info.mongodb.database of the tenant configuration."
                )

            return database_name

        return get_or_compute_tenant_cache(
            patch_key=TenantPatchKey.ACTUAL_DATABASE_NAME,
            compute_func=compute_database_name,
            fallback=lambda: generate_tenant_database_name(),  # Lazy: returns b0001_memsys when no tenant context
            cache_description="database name",
        )

    def __getitem__(self, key: str) -> AsyncCollection:
        """
        Support dictionary-style collection access

        Args:
            key: Collection name

        Returns:
            AsyncCollection: MongoDB Collection object
        """
        # Get real database, then access collection
        return AsyncCollection(self, key)

    def __getattr__(self, name: str) -> Any:
        """
        Intercept attribute access (fallback mechanism)

        This method is called only when an attribute is not found, used to proxy to the real MongoDB database object.

        Args:
            name: Attribute name

        Returns:
            Any: Proxied attribute or method
        """
        # Get real database
        real_database = self._get_real_database()
        logger.debug(
            "🔍 Getting real MongoDB database object attribute or method: %s", name
        )
        # Directly return the attribute or method from the real database
        return getattr(real_database, name)

    @property
    def name(self) -> str:
        """
        Get database name (dynamically obtained)

        Dynamically return the actual database name based on the current tenant context.
        The name of the same TenantAwareDatabase object may differ in different tenant contexts.

        Returns:
            str: The actual database name
        """
        return self._get_actual_database_name()

    @property
    def _name(self) -> str:
        """
        Get database name

        Returns:
            str: Database name
        """
        return self._get_actual_database_name()

    @property
    def client(self) -> AsyncMongoClient:
        """
        Get client reference (return real client)

        Since TenantAwareDatabase is already in a specific tenant context,
        return the real MongoDB client directly to avoid unnecessary secondary proxying.

        Returns:
            AsyncMongoClient: The real MongoDB client
        """
        return self._tenant_aware_client.get_real_client()

    def __bool__(self) -> bool:
        """
        Boolean evaluation of database object

        Returns:
            bool: Always returns True (database object is always truthy)
        """
        return True

    def __repr__(self) -> str:
        """
        String representation of database object

        Returns:
            str: Description of the database object
        """
        return (
            f"TenantAwareDatabase(client={self._tenant_aware_client}, name={self.name})"
        )

    def __eq__(self, other: Any) -> bool:
        """
        Equality comparison of database objects

        Only compare client references, as the database name is dynamic.

        Args:
            other: Object to compare

        Returns:
            bool: Whether they are equal
        """
        if isinstance(other, TenantAwareDatabase):
            return self._tenant_aware_client == other._tenant_aware_client
        return False

    def __hash__(self) -> int:
        """
        Hash value of database object

        Generate hash value based only on client reference.

        Returns:
            int: Hash value
        """
        return hash(id(self._tenant_aware_client))

"""
Tenant-aware Milvus Collection

This module implements tenant awareness by inheriting from pymilvus.Collection and overriding the _get_connection method.
Core idea: Dynamically return the correct connection handler based on tenant context.
"""

from typing import Optional
from pymilvus import Collection, CollectionSchema
from pymilvus.orm.connections import connections

from core.observation.logger import get_logger
from core.tenants.tenant_contextvar import get_current_tenant
from core.tenants.tenant_models import TenantPatchKey
from core.tenants.tenantize.oxm.milvus.config_utils import (
    get_tenant_milvus_config,
    get_milvus_connection_cache_key,
    load_milvus_config_from_env,
    get_tenant_aware_collection_name,
)
from core.tenants.tenantize.tenant_cache_utils import get_or_compute_tenant_cache
from core.component.milvus_client_factory import MilvusClientFactory
from core.di.utils import get_bean_by_type

logger = get_logger(__name__)


class TenantAwareCollection(Collection):
    """
    Tenant-aware Milvus Collection

    Implements tenant awareness by inheriting from pymilvus.Collection and overriding the _get_connection method.
    Core functionality: Automatically selects and returns the correct Milvus connection based on the current tenant context.

    Key features:
    1. Tenant isolation: Different tenants use different Milvus connections (distinguished by using alias)
    2. Connection reuse: Tenants with the same configuration share the same connection (cached via cache_key)
    3. Automatic registration: Automatically registers tenant connection upon first access
    4. Fallback connection: Uses default connection when not in tenant mode or without tenant context

    Usage:
        >>> # Used in MilvusCollectionBase
        >>> class MyCollectionBase(MilvusCollectionBase):
        ...     def load_collection(self) -> Collection:
        ...         # Use TenantAwareCollection instead of the original Collection
        ...         return TenantAwareCollection(
        ...             name=self.name,
        ...             using="default",  # using parameter will be ignored, actual tenant-aware connection is used
        ...             schema=self._SCHEMA,
        ...         )

    Notes:
    - The passed using parameter will be ignored; the actual tenant-aware connection alias is used
    - Connection is automatically registered upon first access (via MilvusClientFactory)
    - Connection alias and configuration are cached in tenant_info_patch to avoid redundant computation
    """

    def __init__(
        self,
        name: str,
        schema: Optional[CollectionSchema] = None,
        using: str = "default",
        **kwargs,
    ):
        """
        Initialize tenant-aware Collection

        Args:
            name: Collection name
            schema: Collection schema (optional)
            using: Connection alias (will be ignored, actual tenant-aware connection is used)
            **kwargs: Other parameters (passed to parent class)

        Note:
            - The using parameter will be overridden by the tenant-aware connection alias
            - Ensures tenant connection is registered upon first access
            - _original_name stores the original name value for property usage
        """
        # Save the original name (before calling parent class __init__)
        # This allows _name to be implemented as a property if tenant-aware table names are needed
        self._original_name = name
        self._original_using = using

        # Call parent constructor (using tenant-aware using)
        # Parent class will set self._name = name
        super().__init__(name=name, schema=schema, using=using, **kwargs)

        logger.debug("Creating TenantAwareCollection [name=%s, using=%s]", name, using)

    def _get_connection(self):
        """
        Override parent method to return tenant-aware connection

        This is the core method: called every time a Milvus connection is needed.
        Here we dynamically return the correct connection handler based on tenant context.

        Returns:
            Milvus connection handler

        Note:
            - This method is called on every operation (search, insert, query, etc.)
            - We re-fetch tenant using to support connection switching across requests
        """
        # Dynamically get current tenant's connection alias (supports switching across requests)
        tenant_using = self._get_tenant_aware_using()

        # Ensure connection is registered
        self._ensure_connection_registered(tenant_using)

        # Return corresponding connection handler
        return connections._fetch_handler(tenant_using)

    @staticmethod
    def _get_tenant_aware_using() -> str:
        """
        Get tenant-aware connection alias

        Determines which connection alias to return based on configuration and context:
        1. If non-tenant mode is enabled, return "default"
        2. If tenant mode is enabled, return the corresponding connection alias based on current tenant configuration
        3. If no tenant context exists under tenant mode, return "default"

        Returns:
            str: pymilvus connection alias (using)
        """

        def compute_using() -> str:
            """Compute tenant connection alias"""
            # Get Milvus configuration from tenant config
            milvus_config = get_tenant_milvus_config()
            if not milvus_config:
                raise RuntimeError("Tenant missing Milvus configuration")

            # Generate unique connection alias based on connection parameters
            cache_key = get_milvus_connection_cache_key(milvus_config)
            return f"tenant_{cache_key}"

        return get_or_compute_tenant_cache(
            patch_key=TenantPatchKey.MILVUS_CONNECTION_CACHE_KEY,
            compute_func=compute_using,
            fallback="default",  # Concrete value, no need for lazy evaluation
            cache_description="Milvus connection alias",
        )

    @staticmethod
    def _ensure_connection_registered(using: str) -> None:
        """
        Ensure the specified connection alias is registered

        If the connection is not yet registered, it will be automatically registered (via MilvusClientFactory).

        Args:
            using: Connection alias

        Note:
            - For "default" connection, assume it's already registered at application startup
            - For tenant connections (tenant_*), register automatically if not already registered
        """
        # Check if connection already exists
        try:
            connections._fetch_handler(using)
            # Connection exists, return directly
            return
        except Exception:
            # Connection does not exist, needs registration
            pass

        # If it's the default connection, try to register from environment variables
        if using == "default":
            logger.info("📋 Registering default Milvus connection")
            config = load_milvus_config_from_env()
            TenantAwareCollection._register_connection(config, using)
            return

        # Tenant connection: register from tenant configuration
        try:
            tenant_info = get_current_tenant()
            if not tenant_info:
                raise RuntimeError(
                    "Cannot register tenant connection: tenant context not set"
                )

            milvus_config = get_tenant_milvus_config()
            if not milvus_config:
                raise RuntimeError(
                    f"Cannot register tenant connection: tenant {tenant_info.tenant_id} missing Milvus configuration"
                )

            logger.info(
                "📋 Registering Milvus connection for tenant [%s] [using=%s]",
                tenant_info.tenant_id,
                using,
            )

            TenantAwareCollection._register_connection(milvus_config, using)

        except Exception as e:
            logger.error(
                "Failed to register tenant connection [using=%s]: %s", using, e
            )
            raise

    @staticmethod
    def _register_connection(config: dict, using: str) -> None:
        """
        Register Milvus connection

        Args:
            config: Milvus connection configuration
            using: Connection alias

        Note:
            - Use MilvusClientFactory to create the connection
            - This reuses existing connection pool management logic
        """
        try:
            # Create connection via MilvusClientFactory
            # This reuses existing connection pool management
            factory = get_bean_by_type(MilvusClientFactory)

            # Build URI
            host = config.get("host", "localhost")
            port = config.get("port", 19530)
            uri = (
                f"{host}:{port}" if host.startswith("http") else f"http://{host}:{port}"
            )

            # Create client (this automatically registers the connection)
            # Note: Do not pass db_name, tenant isolation is achieved through Collection name
            factory.get_client(
                uri=uri,
                user=config.get("user", ""),
                password=config.get("password", ""),
                alias=using,
            )

            logger.info(
                "✅ Milvus connection registered [using=%s, host=%s, port=%s]",
                using,
                host,
                port,
            )

        except Exception as e:
            logger.error(
                "Failed to register Milvus connection [using=%s]: %s", using, e
            )
            raise

    # ============================================================
    # Tenant-aware table name support (optional feature)
    # ============================================================
    # Uncomment the @property below if tenant-aware table names are needed.
    # This way, different tenants will use different table names, achieving table-level isolation.
    #
    # Note: After enabling this feature, ensure:
    # 1. Each tenant has an independent table
    # 2. Table names comply with Milvus naming conventions
    # 3. Consider table name length limits
    #
    @property
    def _name(self) -> str:
        """
        Tenant-aware table name

        Override parent class _name attribute to add tenant identifier to table name.

        Example:
            Original table name: "my_collection"
            Tenant A: "tenant_001_my_collection"
            Tenant B: "tenant_002_my_collection"

        Returns:
            str: Tenant-aware table name
        """
        return self.get_tenant_aware_name(self._original_name)

    @classmethod
    def get_tenant_aware_name(cls, original_name: str) -> str:
        """
        Get tenant-aware table name
        """
        return get_tenant_aware_collection_name(original_name)

    @_name.setter
    def _name(self, value: str) -> None:
        """
        Set table name (setter)

        The parent class Collection in pymilvus may attempt to set the _name attribute.
        Here we capture the set operation and update _original_name.

        Args:
            value: Table name to be set
        """
        # Update original table name
        # Note: We store the original value, not the tenant-aware value
        # Because the getter will automatically add the tenant prefix
        self._original_name = value

    @property
    def using(self) -> str:
        """
        Get tenant-aware connection alias
        """
        return self._get_tenant_aware_using()

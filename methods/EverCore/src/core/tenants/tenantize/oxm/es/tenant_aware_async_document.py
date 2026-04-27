"""
Tenant-aware Elasticsearch AsyncDocument

This module implements tenant awareness by inheriting from AliasSupportDoc and overriding key methods.
Core idea: Dynamically return the correct connection and index names based on tenant context.
"""

from typing import Optional, Any, Dict, Type
from fnmatch import fnmatch
from elasticsearch import AsyncElasticsearch
from elasticsearch.dsl import field as e_field
from elasticsearch.dsl.async_connections import connections as async_connections

from core.observation.logger import get_logger
from core.tenants.tenant_contextvar import get_current_tenant
from core.tenants.tenant_models import TenantPatchKey
from core.tenants.tenantize.oxm.es.config_utils import (
    get_tenant_es_config,
    get_es_connection_cache_key,
    load_es_config_from_env,
    get_tenant_aware_index_name,
)
from core.tenants.tenantize.tenant_cache_utils import get_or_compute_tenant_cache
from core.oxm.es.doc_base import AliasSupportDoc
from core.oxm.es.mapping_templates import DYNAMIC_TEMPLATES

logger = get_logger(__name__)


class TenantAwareAsyncDocument(AliasSupportDoc):
    """
    Tenant-aware Elasticsearch AsyncDocument

    Implements tenant awareness by inheriting from AliasSupportDoc and overriding key methods.
    Core functionality: Automatically selects and returns the correct ES connection and index based on current tenant context.

    Key features:
    1. Tenant isolation: Different tenants use different ES connections (distinguished by alias)
    2. Index isolation: Different tenants use different index names (with tenant prefix)
    3. Connection reuse: Tenants with the same configuration share the same connection (cached via cache_key)
    4. Automatic registration: Automatically registers tenant connection upon first access
    5. Fallback connection: Uses default connection in non-tenant mode or when no tenant context exists

    Usage:
        >>> # Define a tenant-aware document class
        >>> class MyDoc(TenantAwareAsyncDocument):
        ...     title = field.Text()
        ...
        ...     class Index:
        ...         name = "my_index"

    Notes:
    - The provided 'using' parameter is ignored; the actual connection alias used is tenant-aware
    - Connection is automatically registered on first access
    - Connection alias and configuration are cached in tenant_info_patch to avoid redundant computation
    """

    # Logical tenant isolation field.
    # Keyword type for exact matching + filter cache.
    # Injected by TenantAwareAsyncElasticsearch interceptor at write time.
    tenant_id = e_field.Keyword()

    class Meta:
        abstract = True

    # ============================================================
    # Tenant-aware connection management
    # ============================================================

    @classmethod
    def _get_using(cls, using: Optional[str] = None) -> str:
        """
        Override parent method to return tenant-aware connection alias

        Ignore the provided 'using' parameter and return the correct connection alias based on tenant context.

        Args:
            using: Connection alias (will be ignored)

        Returns:
            str: Tenant-aware connection alias
        """
        return cls._get_tenant_aware_using()

    @classmethod
    def _get_connection(cls, using: Optional[str] = None) -> AsyncElasticsearch:
        """
        Override parent method to return tenant-aware connection

        This is the core method: called every time an ES connection is needed.
        Here we dynamically return the correct connection based on tenant context.

        Args:
            using: Connection alias (will be ignored)

        Returns:
            AsyncElasticsearch: Tenant-aware ES client
        """
        # Dynamically get the connection alias for the current tenant
        tenant_using = cls._get_tenant_aware_using()

        # Return the corresponding connection
        return async_connections.get_connection(tenant_using)

    @classmethod
    def _get_tenant_aware_using(cls) -> str:
        """
        Get tenant-aware connection alias

        Decide which connection alias to return based on configuration and context:
        1. If non-tenant mode is enabled, return "default"
        2. If tenant mode is enabled, return the corresponding connection alias based on current tenant configuration
        3. If in tenant mode but no tenant context exists, return "default"

        Returns:
            str: elasticsearch-dsl connection alias (using)
        """

        def compute_using() -> str:
            """Compute tenant connection alias"""
            # Get ES configuration from tenant settings
            es_config = get_tenant_es_config()
            if not es_config:
                raise RuntimeError("Tenant is missing Elasticsearch configuration")

            # Generate a unique connection alias based on connection parameters
            cache_key = get_es_connection_cache_key(es_config)
            using = f"tenant_{cache_key}"
            # es connection is registered in _register_connection
            cls._ensure_connection_registered(using)
            return using

        return get_or_compute_tenant_cache(
            patch_key=TenantPatchKey.ES_CONNECTION_CACHE_KEY,
            compute_func=compute_using,
            fallback="default",  # Concrete value, no need for deferred computation
            cache_description="Elasticsearch connection alias",
        )

    @classmethod
    def _ensure_connection_registered(cls, using: str) -> None:
        """
        Ensure the specified connection alias is registered

        If the connection is not yet registered, it will be automatically registered.

        Args:
            using: Connection alias

        Note:
            - For "default" connection, assume it's already registered at application startup
            - For tenant connections (tenant_*), automatically register if not already registered
        """
        # Check if connection already exists
        try:
            async_connections.get_connection(using)
            # Connection exists, return directly
            return
        except Exception:
            # Connection does not exist, need to register
            pass

        # If it's the default connection, try to register from environment variables
        if using == "default":
            logger.info("📋 Registering default Elasticsearch connection")
            config = load_es_config_from_env()
            cls._register_connection(config, using)
            return

        # Tenant connection: register from tenant configuration
        try:
            tenant_info = get_current_tenant()
            if not tenant_info:
                raise RuntimeError(
                    "Cannot register tenant connection: tenant context not set"
                )

            es_config = get_tenant_es_config()
            if not es_config:
                raise RuntimeError(
                    f"Cannot register tenant connection: tenant {tenant_info.tenant_id} is missing Elasticsearch configuration"
                )

            logger.info(
                "📋 Registering Elasticsearch connection for tenant [%s] [using=%s]",
                tenant_info.tenant_id,
                using,
            )

            cls._register_connection(es_config, using)

        except Exception as e:
            logger.error(
                "Failed to register tenant connection [using=%s]: %s", using, e
            )
            raise

    @classmethod
    def _register_connection(cls, config: Dict[str, Any], using: str) -> None:
        """
        Register Elasticsearch connection

        Args:
            config: Elasticsearch connection configuration
            using: Connection alias

        Note:
            - Use elasticsearch-dsl's connections manager to create the connection
            - This allows reuse of existing connection pool management logic
        """
        try:
            # Build connection parameters
            conn_params = {
                "hosts": config.get("hosts", ["http://localhost:9200"]),
                "timeout": config.get("timeout", 120),
                "max_retries": 3,
                "retry_on_timeout": True,
                "verify_certs": config.get("verify_certs", False),
                "ssl_show_warn": False,
            }

            # Add authentication info
            api_key = config.get("api_key")
            username = config.get("username")
            password = config.get("password")

            if api_key:
                conn_params["api_key"] = api_key
            elif username and password:
                conn_params["basic_auth"] = (username, password)

            # Create tenant-aware client with Guard transport (dual-layer defense)
            # Layer 1: TenantAwareAsyncElasticsearch intercepts at perform_request()
            # Layer 2: TenantGuardTransport verifies at transport level
            from core.tenants.tenantize.oxm.es.tenant_field_es_interceptor import (
                TenantAwareAsyncElasticsearch,
                TenantGuardTransport,
            )

            conn_params["transport_class"] = TenantGuardTransport
            client = TenantAwareAsyncElasticsearch(**conn_params)
            async_connections.add_connection(alias=using, conn=client)

            logger.info(
                "✅ Tenant-aware Elasticsearch connection registered [using=%s, hosts=%s]",
                using,
                conn_params["hosts"],
            )

        except Exception as e:
            logger.error(
                "Failed to register Elasticsearch connection [using=%s]: %s", using, e
            )
            raise

    # ============================================================
    # Tenant-aware index management
    # ============================================================

    @classmethod
    def get_original_index_name(cls) -> str:
        """
        Get original index name (without tenant prefix)

        Retrieve the originally configured index name from cls._index._name.

        Returns:
            str: Original index name
        """
        if hasattr(cls, '_index') and hasattr(cls._index, '_name'):
            return cls._index._name
        raise ValueError(
            f"Document class {cls.__name__} does not have correct index configuration"
        )

    @classmethod
    def get_index_name(cls) -> str:
        """
        Override parent method to return tenant-aware index name

        Add tenant prefix to index name based on current tenant context.
        If in non-tenant mode or no tenant context exists, return original name.

        Returns:
            str: Tenant-aware index name
        """
        original_name = cls.get_original_index_name()
        return get_tenant_aware_index_name(original_name)

    @classmethod
    def _matches(cls, hit: Dict[str, Any]) -> bool:
        """
        Override parent method to match tenant-aware index patterns

        Used to filter documents belonging to the current tenant from ES responses.

        Args:
            hit: ES hit result

        Returns:
            bool: Whether it matches the current document class
        """
        # Get tenant-aware index name
        tenant_index_name = cls.get_index_name()

        # Build matching pattern
        pattern = f"{tenant_index_name}-*"

        return fnmatch(hit.get("_index", ""), pattern)

    @classmethod
    def _default_index(cls, index: Optional[str] = None) -> str:
        """
        Override parent method to return tenant-aware default index name

        Args:
            index: Optional index name override

        Returns:
            str: Tenant-aware index name
        """
        if index:
            return index
        return cls.get_index_name()

    def _get_index(
        self, index: Optional[str] = None, required: bool = True
    ) -> Optional[str]:
        """
        Override parent method to return tenant-aware index name

        Args:
            index: Optional index name override
            required: Whether an index name must be returned

        Returns:
            Optional[str]: Index name

        Raises:
            ValidationException: If required=True and index name cannot be obtained
        """
        # If index is explicitly provided, use it directly
        if index is not None:
            return index

        # Try to get from meta
        if hasattr(self, 'meta') and hasattr(self.meta, 'index'):
            meta_index = getattr(self.meta, 'index', None)
            if meta_index:
                return meta_index

        # Return tenant-aware default index name
        tenant_index = self.get_index_name()
        if tenant_index:
            return tenant_index

        # If index name is required but cannot be obtained, raise exception
        if required:
            from elasticsearch.dsl.exceptions import ValidationException

            raise ValidationException("No index")

        return None

    @classmethod
    def dest(cls) -> str:
        """
        Override parent method to generate tenant-aware destination index name (with timestamp)

        Returns:
            str: Destination index name with timestamp
        """
        # Use tenant-aware index name to generate destination name
        tenant_index_name = cls.get_index_name()
        from common_utils.datetime_utils import get_now_with_timezone

        now = get_now_with_timezone()
        return f"{tenant_index_name}-{now.strftime('%Y%m%d%H%M%S%f')}"


def TenantAwareAliasDoc(
    doc_name: str,
    number_of_shards: int = 3,
    number_of_replicas: int = 1,
    refresh_interval: str = "10s",
) -> Type[TenantAwareAsyncDocument]:
    """
    Create a tenant-aware ES document class that supports alias pattern

    This is a factory function for creating tenant-aware document classes.
    Automatically handles timezone for datetime fields and tenant isolation.

    Args:
        doc_name: Document name (original index name)
        number_of_shards: Number of shards

    Returns:
        Tenant-aware document class

    Examples:
        >>> # Create a tenant-aware document class
        >>> class MyDoc(TenantAwareAliasDoc("my_docs")):
        ...     title = field.Text()
        ...     content = field.Text()
    """
    from elasticsearch.dsl import MetaField
    from core.oxm.es.es_utils import get_index_ns

    # If there is a namespace, append it to the document name
    if get_index_ns():
        doc_name = f"{doc_name}-{get_index_ns()}"

    class GeneratedTenantAwareDoc(TenantAwareAsyncDocument):
        # Save original document name for use in tenant-aware methods
        _ORIGINAL_DOC_NAME = doc_name
        PATTERN = f"{doc_name}-*"

        class Index:
            name = doc_name
            settings = {
                "number_of_shards": number_of_shards,
                "number_of_replicas": number_of_replicas,
                "refresh_interval": refresh_interval,
                "max_ngram_diff": 50,
                "max_shingle_diff": 10,
            }

        class Meta:
            dynamic = MetaField("true")
            # Disable date auto-detection to prevent "2023/10/01" from being incorrectly converted and causing subsequent errors
            date_detection = MetaField(False)
            # Disable numeric detection to prevent string numbers from being confused
            numeric_detection = MetaField(False)
            # Dynamic mapping rules based on field suffixes (see mapping_templates.py)
            dynamic_templates = MetaField(DYNAMIC_TEMPLATES)
            # NOTE: _routing.required is NOT set here globally because:
            # - Exclusive mode: routing=tenant_id would concentrate all docs on one shard,
            #   breaking multi-shard load balancing for large tenants
            # - Exclusive mode: get/delete by ID passthrough without routing, would be rejected
            # _routing.required should be configured per-index by the routing layer
            # for shared-mode indexes only.

        @classmethod
        def get_original_index_name(cls) -> str:
            """Get original index name (without tenant prefix)"""
            return doc_name

    return GeneratedTenantAwareDoc

# skip-sensitive-file
"""
Tenant-aware Milvus Collection Management Class with Suffix and Alias Mechanism

This module combines the functionalities of TenantAwareCollection and MilvusCollectionWithSuffix:
1. Tenant awareness: Automatically selects the correct connection and table name based on tenant context
2. Dynamic table names: Supports dynamic suffix setting via suffix parameter or environment variables
3. Alias mechanism: Real table names include timestamps, accessed via alias
"""

import os
from typing import Optional
from pymilvus import connections, Collection, DataType, FieldSchema
from pymilvus.client.types import ConsistencyLevel

from core.observation.logger import get_logger
from core.oxm.milvus.milvus_collection_base import (
    MilvusCollectionWithSuffix,
    generate_new_collection_name,
)
from core.tenants.tenantize.oxm.milvus.tenant_aware_collection import (
    TenantAwareCollection,
)
from core.tenants.tenantize.oxm.milvus.config_utils import (
    get_tenant_aware_collection_name,
)
from core.tenants.tenant_constants import TENANT_ID_FIELD, TENANT_ID_MAX_LENGTH
from core.tenants.tenant_contextvar import get_current_tenant
from pymilvus import utility

logger = get_logger(__name__)

# Standard tenant_id FieldSchema for Milvus collections.
# is_partition_key=True enables automatic partition routing by tenant_id,
# so queries with tenant_id filter only scan the relevant partition.
TENANT_ID_FIELD_SCHEMA = FieldSchema(
    name=TENANT_ID_FIELD,
    dtype=DataType.VARCHAR,
    max_length=TENANT_ID_MAX_LENGTH,
    is_partition_key=True,
)


class TenantAwareMilvusCollectionWithSuffix(MilvusCollectionWithSuffix):
    """
    Tenant-aware Milvus Collection Management Class with Suffix and Alias Mechanism

    Inherits from MilvusCollectionWithSuffix, adding tenant awareness capabilities:
    1. Automatically selects the correct Milvus connection based on tenant context
    2. Supports tenant-aware table names (automatically adds tenant prefix)
    3. Retains all functionalities of MilvusCollectionWithSuffix (suffix, alias, creation management, etc.)

    Partition key:
    - tenant_id is auto-appended as partition_key to all subclass schemas
    - _NUM_PARTITIONS defaults to 256 for balanced tenant isolation in shared mode
    - Subclasses can override _NUM_PARTITIONS if needed

    Key features:
    - Tenant isolation: Different tenants use different connections and table names
    - Dynamic table names: Supports suffix and environment variables
    - Alias mechanism: Real table names include timestamps, accessed via alias
    - Version management: Can create new versions and perform gradual switching
    - Tenant prefix: All operations automatically add tenant prefix (e.g., tenant_001_movies)

    Table naming rules:
    - Original base name: movies
    - With suffix: movies_production
    - With tenant prefix: tenant_001_movies_production (alias)
    - Real name: tenant_001_movies_production-20231015123456789000

    Usage:
    1. Subclass definition:
       - _COLLECTION_NAME: Base name of the Collection (required)
       - _SCHEMA: Schema definition of the Collection (required)
       - _INDEX_CONFIGS: List of index configurations (optional)
       - _DB_USING: Milvus connection alias (optional, will be overridden by tenant-aware connection)

    2. Instantiation:
       mgr = TenantAwareMovieCollection(suffix="customer_a")
       # Within tenant context:
       # - Uses tenant's Milvus connection
       # - Alias: tenant_001_movies_customer_a
       # - Real name: tenant_001_movies_customer_a-20231015123456789000

    3. Initialization:
       with tenant_context(tenant_info):
           mgr.ensure_all()  # One-click initialization

    4. Usage:
       with tenant_context(tenant_info):
           mgr.collection.insert([...])
           mgr.collection.search(...)

    Example:
        class MovieCollection(TenantAwareMilvusCollectionWithSuffix):
            _COLLECTION_NAME = "movies"
            _SCHEMA = CollectionSchema(fields=[...])
            _INDEX_CONFIGS = [
                IndexConfig(field_name="embedding", index_type="IVF_FLAT", ...),
            ]

        # Multi-tenant scenario usage
        tenant_a = TenantInfo(tenant_id="tenant_001", ...)
        tenant_b = TenantInfo(tenant_id="tenant_002", ...)

        mgr = MovieCollection(suffix="production")

        # Tenant A operations
        with tenant_context(tenant_a):
            mgr.ensure_all()
            mgr.collection.insert([...])

        # Tenant B operations
        with tenant_context(tenant_b):
            mgr.ensure_all()
            mgr.collection.insert([...])
    """

    # Default partition count for partition_key routing.
    # Configurable via MILVUS_NUM_PARTITIONS env var (default 256).
    # Milvus max is 4096. Subclasses can override.
    _NUM_PARTITIONS: int = int(os.getenv("MILVUS_NUM_PARTITIONS", "256"))

    @staticmethod
    def _resolve_num_partitions(class_default: Optional[int]) -> Optional[int]:
        """Resolve num_partitions: storage_info > class attribute > env var.

        Priority:
        1. tenant storage_info.milvus.num_partitions (set per-tenant at init time)
        2. class attribute _NUM_PARTITIONS (set via env var or subclass override)
        """
        tenant = get_current_tenant()
        if tenant:
            milvus_config = tenant.get_storage_info("milvus")
            if milvus_config and "num_partitions" in milvus_config:
                return int(milvus_config["num_partitions"])
        return class_default

    def __init_subclass__(cls, **kwargs):
        """Auto-append tenant_id FieldSchema to _SCHEMA when a subclass is defined.

        Always appends regardless of tenant mode, so schema is consistent
        across all environments (aligned with ES and MongoDB behavior).
        In non-tenant mode the field exists but is never populated.
        """
        super().__init_subclass__(**kwargs)

        # Auto-append tenant_id FieldSchema (partition_key)
        schema = getattr(cls, "_SCHEMA", None)
        if schema is not None:
            existing_names = {f.name for f in schema.fields}
            if TENANT_ID_FIELD not in existing_names:
                schema.add_field(
                    field_name=TENANT_ID_FIELD_SCHEMA.name,
                    datatype=TENANT_ID_FIELD_SCHEMA.dtype,
                    max_length=TENANT_ID_FIELD_SCHEMA.max_length,
                    is_partition_key=TENANT_ID_FIELD_SCHEMA.is_partition_key,
                )
                logger.info(
                    "Auto-appended tenant_id (partition_key) to %s._SCHEMA",
                    cls.__name__,
                )

        # Auto-append tenant_id scalar index for filter performance
        index_configs = getattr(cls, "_INDEX_CONFIGS", None)
        if index_configs is not None:
            existing_index_fields = {cfg.field_name for cfg in index_configs}
            if TENANT_ID_FIELD not in existing_index_fields:
                from core.oxm.milvus.milvus_collection_base import IndexConfig

                index_configs.append(
                    IndexConfig(
                        field_name=TENANT_ID_FIELD,
                        index_type="AUTOINDEX",
                        index_name="idx_tenant_id",
                    )
                )
                logger.info(
                    "Auto-appended tenant_id scalar index to %s._INDEX_CONFIGS",
                    cls.__name__,
                )

    def __init__(self, suffix: Optional[str] = None):
        """
        Initialize the tenant-aware Collection manager

        Args:
            suffix: Collection name suffix; if not provided, read from environment variable

        Note:
            - Save the original _alias_name (without tenant prefix)
            - The actual table name will dynamically add tenant prefix at runtime
        """
        super().__init__(suffix=suffix)
        # Save the original alias name (without tenant prefix)
        # Used in the name property to dynamically compute tenant-aware names
        self._original_alias_name = self._alias_name

    @property
    def name(self) -> str:
        """
        Get the tenant-aware Collection name (alias)

        Override parent class's name property to dynamically add tenant prefix.
        This ensures all places using self.name automatically get tenant-aware table names.

        Returns:
            str: Tenant-aware alias name

        Example:
            Original alias: movies_production
            Tenant A: tenant_001_movies_production
            Tenant B: tenant_002_movies_production
        """
        return TenantAwareCollection.get_tenant_aware_name(self._original_alias_name)

    @property
    def using(self) -> str:
        """
        Get the tenant-aware connection alias
        """
        return TenantAwareCollection._get_tenant_aware_using()

    def ensure_connection_registered(self) -> None:
        """
        Ensure the tenant-aware connection is registered
        """
        TenantAwareCollection._ensure_connection_registered(self.using)

    def load_collection(self) -> TenantAwareCollection:
        """
        Load or create a tenant-aware Collection

        Override parent class method, using TenantAwareCollection instead of regular Collection.
        This ensures all Collection operations are tenant-aware.

        Args:
            name: Collection name (alias name, already includes tenant prefix)

        Returns:
            TenantAwareCollection instance

        Note:
            - Use TenantAwareCollection to automatically handle tenant connections
            - Maintain MilvusCollectionWithSuffix's alias mechanism
            - If alias does not exist, create a new timestamped Collection
            - The name parameter should already be tenant-aware (passed via self.name)
        """
        using = self.using
        origin_alias_name = self._original_alias_name
        tenant_aware_alias_name = get_tenant_aware_collection_name(origin_alias_name)
        new_real_name = generate_new_collection_name(origin_alias_name)
        tenant_aware_new_real_name = get_tenant_aware_collection_name(new_real_name)

        # First check if alias exists (using tenant-aware connection)
        # Note: TenantAwareCollection automatically handles the using parameter
        self.ensure_connection_registered()

        if not utility.has_collection(tenant_aware_alias_name, using=using):
            # Collection does not exist, create a new tenant-aware Collection
            logger.info(
                "Collection '%s' does not exist, creating new tenant-aware Collection: %s",
                origin_alias_name,
                tenant_aware_new_real_name,
            )

            # Create tenant-aware Collection
            # Use native Collection, need to explicitly pass using parameter
            create_kwargs = {
                "name": tenant_aware_new_real_name,
                "schema": self._SCHEMA,
                "consistency_level": ConsistencyLevel.Bounded,
                "using": using,
            }
            num_partitions = self._resolve_num_partitions(
                getattr(self, "_NUM_PARTITIONS", None)
            )
            if num_partitions is not None:
                create_kwargs["num_partitions"] = num_partitions
            logger.info(
                "Creating tenant-aware Collection: %s (num_partitions=%s)",
                tenant_aware_new_real_name,
                num_partitions,
            )
            Collection(**create_kwargs)

            # Create alias pointing to new Collection
            # Note: First delete any existing old alias
            try:
                utility.drop_alias(tenant_aware_alias_name, using=using)
            except Exception:
                pass  # alias does not exist, ignore

            utility.create_alias(
                collection_name=tenant_aware_new_real_name,
                alias=tenant_aware_alias_name,
                using=using,
            )
            logger.info(
                "Alias '%s' -> '%s' created",
                tenant_aware_alias_name,
                tenant_aware_new_real_name,
            )

        # Uniformly load tenant-aware Collection via alias
        coll = TenantAwareCollection(
            name=origin_alias_name,
            schema=self._SCHEMA,
            consistency_level=ConsistencyLevel.Bounded,
        )

        return coll

    def ensure_create(self) -> None:
        """
        Ensure Collection has been created

        Override parent class method, using tenant-aware alias name.

        This method triggers lazy loading of Collection; if alias does not exist, creates a new Collection.
        """
        if self._collection_instance is None:
            # Use tenant-aware alias name
            self._collection_instance = self.load_collection()
        logger.info("Collection '%s' is ready", self.name)

    def create_new_collection(self) -> TenantAwareCollection:
        """
        Create a new tenant-aware real Collection (without switching alias)

        Override parent class method, using TenantAwareCollection and tenant-aware names.

        Returns:
            New tenant-aware Collection instance (with indexes created and loaded)

        Note:
            - Use native Collection for creation (need to explicitly pass using parameter)
            - New Collection name includes tenant prefix and timestamp
            - Return TenantAwareCollection instance to ensure tenant isolation
            - Automatically create indexes and load into memory
        """
        if not self._SCHEMA:
            raise NotImplementedError(
                f"{self.__class__.__name__} must define '_SCHEMA' to support collection creation"
            )

        # Use tenant-aware alias name
        using = self.using
        origin_alias_name = self._original_alias_name
        tenant_aware_alias_name = get_tenant_aware_collection_name(origin_alias_name)
        new_real_name = generate_new_collection_name(origin_alias_name)
        tenant_aware_new_real_name = get_tenant_aware_collection_name(new_real_name)

        # Create new tenant-aware collection
        # Use native Collection, need to explicitly pass using parameter
        create_kwargs = {
            "name": tenant_aware_new_real_name,
            "schema": self._SCHEMA,
            "consistency_level": ConsistencyLevel.Bounded,
            "using": using,
        }
        num_partitions = self._resolve_num_partitions(
            getattr(self, "_NUM_PARTITIONS", None)
        )
        if num_partitions is not None:
            create_kwargs["num_partitions"] = num_partitions
        _coll = Collection(**create_kwargs)

        logger.info(
            "New tenant-aware Collection created: %s (num_partitions=%s)",
            tenant_aware_new_real_name,
            num_partitions,
        )

        # Create indexes for new collection and load
        try:
            self._create_indexes_for_collection(_coll)
            _coll.load()
            logger.info(
                "Indexes created and loading completed for new Collection '%s'",
                new_real_name,
            )
        except Exception as e:
            logger.warning("Error creating indexes for new collection: %s", e)
            raise

        # Return TenantAwareCollection instance, using original alias name
        # Note: Use _original_alias_name here, TenantAwareCollection will automatically add tenant prefix
        new_coll = TenantAwareCollection(
            name=new_real_name,
            schema=self._SCHEMA,
            consistency_level=ConsistencyLevel.Bounded,
        )

        return new_coll

    def switch_alias(
        self, new_collection: TenantAwareCollection, drop_old: bool = False
    ) -> None:
        """
        Switch alias to specified new collection, optionally delete old collection

        Override parent class method, using tenant-aware alias name.

        Args:
            new_collection: New Collection instance
            drop_old: Whether to delete old collection (default False)

        Note:
            - Use tenant-aware alias name for switching
            - Prefer alter_alias, fall back to drop/create if failed
            - Refresh class-level cache after switching
        """
        # Use tenant-aware alias name
        using = self.using
        origin_alias_name = self._original_alias_name
        tenant_aware_alias_name = get_tenant_aware_collection_name(origin_alias_name)
        tenant_aware_new_real_name = new_collection.name

        # Get old collection real name (if exists)
        old_real_name: Optional[str] = None
        try:
            conn = connections._fetch_handler(using)
            desc = conn.describe_alias(tenant_aware_alias_name)
            old_real_name = (
                desc.get("collection_name") if isinstance(desc, dict) else None
            )
        except Exception:
            old_real_name = None

        # Alias switching
        try:
            conn = connections._fetch_handler(using)
            conn.alter_alias(tenant_aware_new_real_name, tenant_aware_alias_name)
            logger.info(
                "Alias '%s' switched to '%s'",
                tenant_aware_alias_name,
                tenant_aware_new_real_name,
            )
        except Exception as e:
            logger.warning("alter_alias failed, trying drop/create: %s", e)
            try:
                utility.drop_alias(tenant_aware_alias_name, using=using)
            except Exception:
                pass
            utility.create_alias(
                collection_name=tenant_aware_new_real_name,
                alias=tenant_aware_alias_name,
                using=using,
            )
            logger.info(
                "Alias '%s' -> '%s' created",
                tenant_aware_alias_name,
                tenant_aware_new_real_name,
            )

        # Optionally delete old collection (after switching completes)
        if drop_old and old_real_name:
            try:
                utility.drop_collection(old_real_name, using=using)
                logger.info("Old collection deleted: %s", old_real_name)
            except Exception as e:
                logger.warning(
                    "Failed to delete old collection (can handle manually): %s", e
                )

        # Refresh class-level cache to alias collection
        try:
            self.__class__._collection_instance = TenantAwareCollection(
                name=origin_alias_name,
                schema=self._SCHEMA,
                consistency_level=ConsistencyLevel.Bounded,
            )
        except Exception:
            pass

    # ==================== Tenant Field Isolation ====================

    @classmethod
    def async_collection(cls):
        """Get asynchronous Collection instance wrapped with tenant field isolation proxy.

        Override parent to wrap AsyncCollection in TenantFieldCollectionProxy.
        This ensures all data operations (insert/search/query/delete) automatically
        inject tenant_id, while control-plane operations pass through.

        In non-tenant mode, the proxy is a transparent passthrough.
        """
        inner = super().async_collection()

        from core.tenants.tenantize.oxm.milvus.tenant_field_collection_proxy import (
            TenantFieldCollectionProxy,
        )

        return TenantFieldCollectionProxy(inner)

    @classmethod
    def collection(cls):
        """Get synchronous Collection instance.

        Blocked: synchronous Collection bypasses the TenantFieldCollectionProxy
        and cannot enforce tenant isolation. Use async_collection() instead.
        """
        raise RuntimeError(
            f"{cls.__name__}.collection() is blocked. "
            f"Synchronous Collection cannot enforce tenant field isolation. "
            f"Use async_collection() instead."
        )

    # ==================== Collection Management ====================

    def exists(self) -> bool:
        """
        Check if Collection exists (via alias)

        Override parent class method, using tenant-aware name and using.

        Returns:
            bool: Whether Collection exists
        """
        name = self.name
        using = self.using
        return utility.has_collection(name, using=using)

    def drop(self) -> None:
        """
        Delete current Collection (including alias and real Collection)

        Override parent class method, using tenant-aware name, using, and TenantAwareCollection.

        Note:
            - Use tenant-aware connection alias
            - Use TenantAwareCollection to ensure tenant isolation
            - Delete the real Collection (not the alias)
        """
        using = self.using
        name = self.name
        try:
            utility.drop_collection(name, using=using)
            logger.info("Collection '%s' deleted", name)
        except Exception as e:  # pylint: disable=broad-except
            logger.warning(
                "Collection '%s' does not exist or deletion failed: %s", name, e
            )

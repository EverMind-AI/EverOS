"""
Tenant information data model

This module defines data classes related to tenants for unified management of tenant information.
"""

import json
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from enum import Enum


class TenantPatchKey(str, Enum):
    """
    Tenant information Patch cache key enumeration

    Used to manage cache keys in tenant_info_patch uniformly, avoiding key name conflicts caused by string hardcoding.

    Design notes:
    - All patch cache keys should be defined here
    - Using enumeration provides better type hints and prevents spelling errors
    - Inherits from str, allowing it to be used directly as a dictionary key

    Cache key descriptions:
    - MONGO_CLIENT_CACHE_KEY: Used to cache the cache_key of MongoDB client (hash value of connection parameters)
    - ACTUAL_DATABASE_NAME: Used to cache the actual database name used
    - REAL_DATABASE_PREFIX: Used to cache the key prefix of the database object (needs to be concatenated with the database name during actual use)
    - ES_CONNECTION_CACHE_KEY: Used to cache the cache_key of Elasticsearch connection alias
    """

    # MongoDB client related
    MONGO_CLIENT_CACHE_KEY = "mongo_client_cache_key"

    # MongoDB database related
    ACTUAL_DATABASE_NAME = "actual_database_name"
    MONGO_REAL_DATABASE = "mongo_real_database"  # Real MongoDB database object

    # Milvus connection related
    MILVUS_CONNECTION_CACHE_KEY = "milvus_connection_cache_key"

    # Elasticsearch connection related
    ES_CONNECTION_CACHE_KEY = "es_connection_cache_key"


@dataclass
class TenantDetail:
    """
    Tenant detailed information data class

    This class is used to store adapted tenant detailed information; external tenant information will be adapted into this data model.

    Attributes:
        tenant_info: Dictionary containing tenant-related information, storing converted tenant data
                    Example structure: {
                        "hash_key": "...",
                        "account_id": "...",
                        "space_id": "...",
                        "organization_id": "..."
                    }
        storage_info: Dictionary containing storage configuration information, optional field
                     Example structure: {
                         "mongodb": {"host": "...", "port": 27017, ...},
                         "redis": {"host": "...", "port": 6379, ...}
                     }
    """

    tenant_info: Optional[Dict[str, Any]] = field(default=None)
    storage_info: Optional[Dict[str, Any]] = field(default=None)

    # Isolation mode: "shared" (logical isolation, default) or "exclusive" (physical isolation).
    # Both modes inject tenant_id on all reads and writes (interceptor behavior is identical).
    # The difference is storage-level: shared uses common tables/indices, exclusive uses dedicated ones.
    # Default "shared" — all new tenants use shared mode.
    isolation_mode: str = field(default="shared")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert TenantDetail to dictionary

        Returns:
            Dict[str, Any]: Dictionary containing tenant_info and storage_info
        """
        return {
            'tenant_info': self.tenant_info,
            'storage_info': self.storage_info,
            'isolation_mode': self.isolation_mode,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TenantDetail':
        """
        Create TenantDetail instance from dictionary

        Args:
            data: Dictionary containing tenant_info and storage_info

        Returns:
            TenantDetail: Newly created instance
        """
        return cls(
            tenant_info=data.get('tenant_info'),
            storage_info=data.get('storage_info'),
            isolation_mode=data.get('isolation_mode', 'shared'),
        )


@dataclass
class TenantInfo:
    """
    Tenant information data class

    This class is the main data model for tenant information, containing core tenant details.

    Attributes:
        tenant_id: Unique identifier for the tenant
        tenant_detail: Adapted detailed tenant information
        origin_tenant_data: Original tenant data directly passed from external sources, kept unchanged without adaptation
        tenant_info_patch: Cache data related to the tenant, used to store computed values (e.g., actual_database_name, real_client, etc.)
                          Lifecycle aligns with tenant_info, avoiding redundant computation
    """

    tenant_id: str
    tenant_detail: TenantDetail
    origin_tenant_data: Dict[str, Any] = field(default_factory=dict)
    tenant_info_patch: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_shared_mode(self) -> bool:
        """Whether this tenant uses shared (logical) isolation mode."""
        return self.tenant_detail.isolation_mode == "shared"

    def get_storage_info(self, storage_type: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve storage configuration information of the specified type

        This method retrieves the configuration for the specified storage type from tenant_detail.storage_info.

        Args:
            storage_type: Storage type, such as "mongodb", "redis", "elasticsearch", etc.

        Returns:
            Configuration dictionary for the specified storage type, or None if not found

        Examples:
            >>> tenant_info = TenantInfo(
            ...     tenant_id="tenant_001",
            ...     tenant_detail=TenantDetail(storage_info={
            ...         "mongodb": {"host": "localhost", "port": 27017}
            ...     }),
            ...     origin_tenant_data={}
            ... )
            >>> mongo_config = tenant_info.get_storage_info("mongodb")
            >>> print(mongo_config)
            {'host': 'localhost', 'port': 27017}
        """
        # Check if tenant_detail.storage_info exists
        if self.tenant_detail.storage_info is None:
            return None

        # Retrieve configuration of the specified type from storage_info
        return self.tenant_detail.storage_info.get(storage_type)

    def get_patch_value(self, key: str, default: Any = None) -> Any:
        """
        Retrieve cached value from tenant_info_patch

        This method is used to retrieve computed values cached in tenant_info_patch,
        avoiding redundant computation and improving performance.

        Args:
            key: Cache key name
            default: Default return value if the key does not exist

        Returns:
            Cached value, or default if the key does not exist

        Examples:
            >>> tenant_info = TenantInfo(
            ...     tenant_id="tenant_001",
            ...     tenant_detail=TenantDetail(),
            ...     origin_tenant_data={}
            ... )
            >>> # Retrieve cached database name
            >>> db_name = tenant_info.get_patch_value(TenantPatchKey.ACTUAL_DATABASE_NAME)
        """
        return self.tenant_info_patch.get(key, default)

    def set_patch_value(self, key: str, value: Any) -> None:
        """
        Set cached value in tenant_info_patch

        This method is used to cache computation results, avoiding redundant computation later.
        The cache lifecycle is consistent with the TenantInfo instance.

        Args:
            key: Cache key name
            value: Value to be cached

        Examples:
            >>> tenant_info = TenantInfo(
            ...     tenant_id="tenant_001",
            ...     tenant_detail=TenantDetail(),
            ...     origin_tenant_data={}
            ... )
            >>> # Cache database name
            >>> tenant_info.set_patch_value(TenantPatchKey.ACTUAL_DATABASE_NAME, "tenant_001_db")
            >>> # Cache client's cache_key
            >>> tenant_info.set_patch_value(TenantPatchKey.MONGO_CLIENT_CACHE_KEY, "cache_key_value")
        """
        self.tenant_info_patch[key] = value

    def has_patch_value(self, key: str) -> bool:
        """
        Check if a specific cached value exists in tenant_info_patch

        Args:
            key: Cache key name

        Returns:
            Returns True if exists, otherwise False

        Examples:
            >>> tenant_info = TenantInfo(
            ...     tenant_id="tenant_001",
            ...     tenant_detail=TenantDetail(),
            ...     origin_tenant_data={}
            ... )
            >>> tenant_info.set_patch_value(TenantPatchKey.ACTUAL_DATABASE_NAME, "tenant_001_db")
            >>> tenant_info.has_patch_value(TenantPatchKey.ACTUAL_DATABASE_NAME)
            True
            >>> tenant_info.has_patch_value("non_existent_key")
            False
        """
        return key in self.tenant_info_patch

    def clear_patch_cache(self) -> None:
        """
        Clear all cached data

        In certain cases, it might be necessary to clear the cache (e.g., when tenant configuration is updated),
        this method clears all data in tenant_info_patch.

        Examples:
            >>> tenant_info = TenantInfo(
            ...     tenant_id="tenant_001",
            ...     tenant_detail=TenantDetail(),
            ...     origin_tenant_data={}
            ... )
            >>> tenant_info.set_patch_value("key1", "value1")
            >>> tenant_info.clear_patch_cache()
            >>> tenant_info.has_patch_value("key1")
            False
        """
        self.tenant_info_patch.clear()

    def invalidate_patch(self, key: Optional[str] = None) -> bool:
        """
        Invalidate cache (delete specific key or all)

        Used to refresh or delete cached objects in tenant_info_patch.
        Call this method when cached resources need rebuilding (e.g., connection lost, configuration changed).

        Args:
            key: Cache key name to delete. If None, clears all cache.

        Returns:
            bool: If key is specified, returns whether the key existed and was deleted;
                  if key is None (clear all), always returns True.

        Examples:
            >>> tenant_info = TenantInfo(
            ...     tenant_id="tenant_001",
            ...     tenant_detail=TenantDetail(),
            ...     origin_tenant_data={}
            ... )
            >>> tenant_info.set_patch_value(TenantPatchKey.MONGO_REAL_DATABASE, some_db_obj)
            >>> # Delete specific cache
            >>> tenant_info.invalidate_patch(TenantPatchKey.MONGO_REAL_DATABASE)
            True
            >>> # Clear all cache
            >>> tenant_info.invalidate_patch()
            True
        """
        if key is None:
            # Clear all cache
            self.tenant_info_patch.clear()
            return True
        else:
            # Delete specific key
            if key in self.tenant_info_patch:
                del self.tenant_info_patch[key]
                return True
            return False

    # ==================== Serialization/Deserialization methods ====================
    # Used to pass tenant context to other processes via asynchronous tasks

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert TenantInfo to dictionary (for serialization)

        Note: tenant_info_patch will not be serialized because it may contain non-serializable objects (e.g., database connections).
        After deserialization, tenant_info_patch will be initialized as an empty dictionary.

        Returns:
            Dict[str, Any]: Dictionary containing tenant_id, tenant_detail, and origin_tenant_data
        """
        return {
            'tenant_id': self.tenant_id,
            'tenant_detail': self.tenant_detail.to_dict(),
            'origin_tenant_data': self.origin_tenant_data,
        }

    def to_json(self) -> str:
        """
        Serialize TenantInfo into a JSON string

        Used to pass tenant context to other processes via asynchronous tasks.

        Note: tenant_info_patch will not be serialized.

        Returns:
            str: JSON string

        Examples:
            >>> tenant_info = TenantInfo(
            ...     tenant_id="t1234567890abcdef",
            ...     tenant_detail=TenantDetail(tenant_info={"account_id": "acc_001"}),
            ...     origin_tenant_data={}
            ... )
            >>> json_str = tenant_info.to_json()
        """
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TenantInfo':
        """
        Create TenantInfo instance from dictionary

        Args:
            data: Dictionary containing tenant_id, tenant_detail, and origin_tenant_data

        Returns:
            TenantInfo: Newly created instance (tenant_info_patch initialized as empty dictionary)
        """
        tenant_detail_data = data.get('tenant_detail', {})
        return cls(
            tenant_id=data['tenant_id'],
            tenant_detail=TenantDetail.from_dict(tenant_detail_data),
            origin_tenant_data=data.get('origin_tenant_data', {}),
            # tenant_info_patch is not restored from serialized data, initialized as empty
        )

    @classmethod
    def from_json(cls, json_str: str) -> 'TenantInfo':
        """
        Deserialize TenantInfo instance from JSON string

        Used to receive tenant context from other processes via asynchronous tasks.

        Args:
            json_str: JSON string

        Returns:
            TenantInfo: Newly created instance (tenant_info_patch initialized as empty dictionary)

        Examples:
            >>> json_str = '{"tenant_id": "t1234567890abcdef", ...}'
            >>> tenant_info = TenantInfo.from_json(json_str)
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

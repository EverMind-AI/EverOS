"""
Tenant information service module

This module defines the tenant information service interface and its default implementation,
used to retrieve tenant information based on tenant_id (typically single_tenant_id from config).

Uses DI mechanism to manage TenantInfoService implementations.
"""

import os
from abc import ABC, abstractmethod
from typing import Optional

from core.tenants.tenant_models import TenantInfo, TenantDetail
from core.tenants.tenant_constants import ISOLATION_MODE_EXCLUSIVE
from core.di.decorators import component


class TenantInfoService(ABC):
    """
    Tenant information service interface

    This interface defines standard methods for retrieving tenant information.
    Different implementations can retrieve tenant information from various data sources (e.g., database, API, configuration files).

    Using DI mechanism:
    - Multiple implementations can be registered
    - Use primary=True to mark the default implementation
    - Obtain instances through the container
    """

    @abstractmethod
    def get_tenant_info(self, tenant_id: str) -> Optional[TenantInfo]:
        """
        Retrieve tenant information by tenant ID

        Args:
            tenant_id: Unique identifier of the tenant

        Returns:
            Tenant information object, or None if not found
        """
        raise NotImplementedError


@component("default_tenant_info_service")
class DefaultTenantInfoService(TenantInfoService):
    """
    Default tenant information service implementation

    This implementation provides basic tenant information containing only the tenant_id,
    without detailed information such as storage configurations. Suitable for simple scenarios or as the default implementation.

    Uses the @component decorator to register into the DI container and mark as primary.
    """

    def get_tenant_info(self, tenant_id: str) -> Optional[TenantInfo]:
        """
        Create tenant information for single-tenant (local dev) mode.

        Builds a complete TenantInfo with explicit storage_info read from
        environment variables, so the behavior is identical to a multi-tenant
        exclusive deployment — no "fallback coincidence" relied upon.

        Args:
            tenant_id: Unique identifier of the tenant

        Returns:
            TenantInfo object with storage_info populated from env

        Examples:
            >>> from core.di.container import get_container
            >>> service = get_container().get_bean_by_type(TenantInfoService)
            >>> tenant_info = service.get_tenant_info("dev")
            >>> print(tenant_info.tenant_detail.isolation_mode)
            exclusive
        """
        if not tenant_id:
            return None

        # Build resource names with tenant_id as prefix, same pattern as
        # EnterpriseTenantRouter._build_storage_info() in exclusive mode.
        # e.g. tenant_id="dev" → database="dev_memsys", index_prefix="dev"
        base_db = os.getenv("MONGODB_DATABASE", "memsys")

        storage_info = {
            "mongodb": {"database": f"{tenant_id}_{base_db}"},
            "elasticsearch": {"index_prefix": tenant_id},
            "milvus": {"collection_prefix": tenant_id},
        }

        tenant_detail = TenantDetail(
            tenant_info={"tenant_id": tenant_id},
            storage_info=storage_info,
            isolation_mode=ISOLATION_MODE_EXCLUSIVE,
        )

        return TenantInfo(
            tenant_id=tenant_id, tenant_detail=tenant_detail, origin_tenant_data={}
        )

"""
Tenant context management module

This module provides tenant context management functionality based on contextvars,
used to safely store and access current tenant information in asynchronous environments.
"""

from contextvars import ContextVar
from typing import Optional

from core.tenants.tenant_models import TenantInfo
from core.tenants.tenant_config import get_tenant_config
from core.tenants.tenant_info_provider import TenantInfoService
from core.di.container import get_container
from core.di.exceptions import BeanNotFoundError


# Global tenant context variable
# Using ContextVar ensures each task has an independent tenant context in asynchronous environments
current_tenant_contextvar: ContextVar[Optional[TenantInfo]] = ContextVar(
    'current_tenant', default=None
)


def set_current_tenant(tenant_info: Optional[TenantInfo]) -> None:
    """
    Set the tenant information for the current request

    This method sets the tenant information into the current context. In asynchronous environments,
    each request/task has an independent context, without interference.

    Args:
        tenant_info: Tenant information object, if None, clears the current tenant information

    Examples:
        >>> from core.tenants.tenant_models import TenantInfo, TenantDetail
        >>> tenant = TenantInfo(
        ...     tenant_id="tenant_001",
        ...     tenant_detail=TenantDetail(),
        ...     origin_tenant_data={}
        ... )
        >>> set_current_tenant(tenant)
    """
    current_tenant_contextvar.set(tenant_info)


def get_current_tenant() -> Optional[TenantInfo]:
    """
    Get the tenant information for the current request

    This method retrieves tenant information from the current context. If no tenant information is set in the current context,
    it attempts to get the single tenant ID from the tenant configuration and retrieve tenant information via the tenant info service.

    Retrieval logic:
    1. First, try to get tenant information from contextvar
    2. If not in contextvar, check if SINGLE_TENANT_ID is set in the configuration
    3. If SINGLE_TENANT_ID is configured, get TenantInfoService from DI container and retrieve tenant information

    Returns:
        Tenant information in the current context, returns None if not set

    Examples:
        >>> tenant = get_current_tenant()
        >>> if tenant:
        ...     print(f"Current tenant ID: {tenant.tenant_id}")
        ... else:
        ...     print("Tenant information not set")
    """
    # 1. First try to get from contextvar
    tenant_info = current_tenant_contextvar.get()
    if tenant_info is not None:
        return tenant_info

    # 2. If not in contextvar, try to get single_tenant_id from configuration
    tenant_config = get_tenant_config()
    single_tenant_id = tenant_config.single_tenant_id

    # 3. If single_tenant_id is configured, get TenantInfoService from DI container
    if single_tenant_id:
        try:
            # Get TenantInfoService instance from DI container (automatically selects primary implementation)
            service = get_container().get_bean_by_type(TenantInfoService)
            tenant_info = service.get_tenant_info(single_tenant_id)
            set_current_tenant(tenant_info)
            return tenant_info
        except BeanNotFoundError:
            # If TenantInfoService is not registered in DI container, return None
            # This usually occurs during early application startup or in test environments
            return None

    # 4. If none of the above, return None
    return None


def clear_current_tenant() -> None:
    """
    Clear the tenant information for the current request

    This method sets the tenant information in the current context to None,
    equivalent to set_current_tenant(None).

    Examples:
        >>> clear_current_tenant()
    """
    current_tenant_contextvar.set(None)


def get_current_tenant_id() -> Optional[str]:
    """
    Get the ID of the current tenant

    This is a convenience method that directly returns the tenant_id of the current tenant.
    If no tenant information is currently set, returns None.

    Returns:
        The ID of the current tenant, returns None if tenant information is not set

    Examples:
        >>> tenant_id = get_current_tenant_id()
        >>> if tenant_id:
        ...     print(f"Current tenant ID: {tenant_id}")
    """
    tenant = get_current_tenant()
    return tenant.tenant_id if tenant else None

"""
Tenant Router

Responsible for resolving tenant identity from requests and building tenant context
(TenantInfo with storage_info, isolation_mode, etc.).

This is the SINGLE SOURCE OF TRUTH for "which tenant is this request for" and
"where does this tenant's data live". The routing decision includes:
1. Resolve tenant identity from the request (headers, tokens, etc.)
2. Determine isolation mode (shared vs exclusive)
3. Build storage_info (database names, index prefixes, collection prefixes)
4. Construct TenantInfo and set it into the request context

Open-source version: no-op (non-tenant mode, no routing needed).
Enterprise version: extracts org/space from headers, generates tenant_id,
builds storage_info with proper naming, sets isolation_mode.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from fastapi import Request

from core.di.decorators import component
from core.observation.logger import get_logger
from core.tenants.tenant_models import TenantInfo

logger = get_logger(__name__)


class TenantRouter(ABC):
    """
    Tenant router interface.

    Implementations resolve tenant identity from HTTP requests and build
    the full TenantInfo (including storage_info and isolation_mode).

    The router is called by AppLogicProvider.setup_app_context() on every request.
    """

    @abstractmethod
    def resolve_tenant(self, request: Request) -> Optional[TenantInfo]:
        """
        Resolve tenant from the incoming request and set tenant context.

        Implementations should:
        1. Extract tenant identity from request (headers, auth tokens, etc.)
        2. Build TenantDetail with storage_info and isolation_mode
        3. Construct TenantInfo
        4. Call set_current_tenant() to set context
        5. Return the TenantInfo (or None if no tenant context)

        Args:
            request: FastAPI request object

        Returns:
            TenantInfo if tenant was resolved and context set, None otherwise
        """
        raise NotImplementedError


@component("default_tenant_router")
class DefaultTenantRouter(TenantRouter):
    """
    Default tenant router (open-source version).

    No-op implementation — returns None (no tenant routing).
    In non-tenant mode or open-source deployments, tenant context
    is either not needed or handled by single-tenant auto-activation.
    """

    def resolve_tenant(self, request: Request) -> Optional[TenantInfo]:
        """No tenant routing in open-source version."""
        return None

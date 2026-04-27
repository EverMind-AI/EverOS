# -*- coding: utf-8 -*-
"""
Tenant context check decorator

Ensures that the API request carries a valid tenant context.
"""

from functools import wraps
from typing import Callable, Any

from fastapi import HTTPException

from core.tenants.tenant_contextvar import get_current_tenant_id


def require_tenant(func: Callable) -> Callable:
    """
    Require tenant context decorator

    Used to decorate controller interface methods to ensure the request carries a valid tenant context.
    If the tenant context is missing, returns a 400 error.

    Example usage:
        @post("/init-db")
        @require_tenant
        async def init_tenant_database(self) -> TenantInitResponse:
            tenant_id = get_current_tenant_id()
            # tenant_id is guaranteed to be non-None
            ...

    Args:
        func: The decorated asynchronous function

    Returns:
        Callable: The wrapped function
    """

    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Check tenant context
        tenant_id = get_current_tenant_id()
        if not tenant_id:
            raise HTTPException(status_code=400, detail="Missing tenant context.")

        # Call the original function
        return await func(*args, **kwargs)

    return wrapper

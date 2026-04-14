"""
Redis tenant key utility functions module

Provides tenant isolation for Redis key names by prepending the tenant ID to achieve multi-tenant data isolation.
"""

from typing import Optional

from core.tenants.tenant_contextvar import get_current_tenant_id


def build_tenant_redis_key(prefix: str, tenant_id: str, key: str) -> str:
    """
    Build a tenant-scoped Redis key with an explicit tenant_id.

    Format: {prefix}:{tenant_id}:{key}

    Args:
        prefix: Key namespace prefix (e.g. "task_status")
        tenant_id: Tenant identifier
        key: Business key (e.g. task_id, request_id)

    Returns:
        str: "{prefix}:{tenant_id}:{key}"

    Examples:
        >>> build_tenant_redis_key("task_status", "t3a7b2c1d9e", "abc123")
        'task_status:t3a7b2c1d9e:abc123'
    """
    return f"{prefix}:{tenant_id}:{key}"

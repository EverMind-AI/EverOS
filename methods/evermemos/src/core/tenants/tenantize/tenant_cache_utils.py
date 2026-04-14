"""
Tenant cache utility functions

Provides tenant-aware caching mechanism for caching computation results within tenant context.
This is a generic cache pattern implementation to avoid duplicating the same caching logic across different modules.

Core design patterns:
- Cache Pattern
- Lazy Initialization Pattern
- Memoization Pattern

Use cases:
- Tenant-aware connection alias computation
- Tenant-aware database name retrieval
- Tenant-aware configuration information retrieval
- Any computation result that needs to be cached per tenant
"""

from typing import TypeVar, Callable, Optional, Union
from core.constants.exceptions import CriticalError
from core.observation.logger import get_logger
from core.tenants.tenant_contextvar import get_current_tenant
from core.tenants.tenant_config import (
    get_tenant_config,
)  # still needed for app_ready check
from core.tenants.tenant_models import TenantPatchKey

logger = get_logger(__name__)

T = TypeVar("T")


class TenantContextMissingError(CriticalError):
    """Raised when tenant context is missing after app startup (strict check).

    Inherits from CriticalError so that ``reraise_critical_errors()`` will
    propagate it out of ``asyncio.gather(return_exceptions=True)`` result processing.
    """

    pass


def get_or_compute_tenant_cache(
    patch_key: TenantPatchKey,
    compute_func: Callable[[], T],
    fallback: Optional[Union[T, Callable[[], T]]] = None,
    cache_description: str = "value",
) -> T:
    """
    Get or compute tenant cache value (supports lazy evaluation fallback)

    This is a generic tenant-aware cache function that encapsulates common caching patterns:
    1. Get tenant information -> if not available, return fallback (lazy evaluation)
    2. Check patch cache -> if hit, return cached value
    3. Call compute_func to calculate new value -> cache to patch -> return new value

    Performance optimization:
    - Fallback supports lazy evaluation: fallback function is only called when actually needed
    - Avoid unnecessary fallback computation in tenant mode

    Args:
        patch_key: TenantPatchKey enum value, used to identify the cache item
        compute_func: Computation function, called when cache miss occurs. Should be a parameterless Callable
        fallback: Fallback value when no tenant context is available (optional)
                 - Can be a concrete value (e.g., "default")
                 - Or a parameterless function (lazy evaluation, e.g., lambda: get_default_database_name())
        cache_description: Description of the cache item, used for logging (optional, default is "value")

    Returns:
        T: Cached value or computed value

    Raises:
        RuntimeError: If fallback is None and no tenant context is available

    Usage examples:
        >>> # Example 1: Get tenant-aware connection alias (fallback is a concrete value)
        >>> def compute_using():
        ...     milvus_config = get_tenant_milvus_config()
        ...     cache_key = get_milvus_connection_cache_key(milvus_config)
        ...     return f"tenant_{cache_key}"
        >>>
        >>> using = get_or_compute_tenant_cache(
        ...     patch_key=TenantPatchKey.MILVUS_CONNECTION_CACHE_KEY,
        ...     compute_func=compute_using,
        ...     fallback="default",  # Concrete value, no lazy evaluation needed
        ...     cache_description="Milvus connection alias"
        ... )

        >>> # Example 2: Get tenant-aware database name (fallback is a function, lazy evaluation)
        >>> def compute_database_name():
        ...     mongo_config = get_tenant_mongo_config()
        ...     return mongo_config.get("database")
        >>>
        >>> db_name = get_or_compute_tenant_cache(
        ...     patch_key=TenantPatchKey.ACTUAL_DATABASE_NAME,
        ...     compute_func=compute_database_name,
        ...     fallback=lambda: get_default_database_name(),  # Lazy evaluation, only called when needed
        ...     cache_description="database name"
        ... )
    """
    try:
        config = get_tenant_config()

        # Step 1: Get tenant information
        tenant_info = get_current_tenant()
        if not tenant_info:
            # Strict check mode: after app startup, tenant context must exist in tenant mode
            if config.app_ready:
                raise TenantContextMissingError(
                    f"🚨 Strict tenant check failed: app is ready but tenant context is missing!"
                    f"This usually indicates a serious code issue, please check the call chain."
                    f"[cache_key={patch_key.value}, cache_description={cache_description}]"
                )

            # During app startup, allow using fallback
            fallback_value = _resolve_fallback(fallback, cache_description)
            if fallback_value is None:
                raise RuntimeError(
                    f"Tenant context not set in tenant mode and no fallback provided [cache_key={patch_key.value}]"
                )
            logger.debug(
                "⚠️ Tenant context not set in tenant mode, using fallback %s [fallback=%s]",
                cache_description,
                fallback_value,
            )
            return fallback_value

        tenant_id = tenant_info.tenant_id

        # Step 3: Check patch cache
        cached_value = tenant_info.get_patch_value(patch_key)
        if cached_value is not None:
            logger.debug(
                "🔍 Cache hit in tenant_info_patch for %s [tenant_id=%s, value=%s]",
                cache_description,
                tenant_id,
                cached_value,
            )
            return cached_value

        # Step 4: Compute new value
        computed_value = compute_func()

        # Step 5: Cache to patch
        tenant_info.set_patch_value(patch_key, computed_value)

        logger.debug(
            "✅ Computed and cached %s for tenant [%s] [value=%s]",
            cache_description,
            tenant_id,
            computed_value,
        )

        return computed_value

    except CriticalError:
        raise
    except Exception as e:
        # Exception handling: try to use fallback (lazy evaluation)
        fallback_value = _resolve_fallback(fallback, cache_description)
        if fallback_value is not None:
            logger.error(
                "Failed to get tenant cache %s, using fallback value: %s [fallback=%s]",
                cache_description,
                e,
                fallback_value,
            )
            return fallback_value
        else:
            logger.error("Failed to get tenant cache %s: %s", cache_description, e)
            raise


def _resolve_fallback(
    fallback: Optional[Union[T, Callable[[], T]]], description: str
) -> Optional[T]:
    """
    Resolve fallback value (supports lazy evaluation)

    Args:
        fallback: Can be a concrete value or a function
        description: Description for logging

    Returns:
        Resolved value
    """
    if fallback is None:
        return None

    # If it's a Callable, call it (lazy evaluation)
    if callable(fallback):
        try:
            return fallback()
        except Exception as e:
            logger.error("Failed to compute fallback %s: %s", description, e)
            return None

    # Otherwise, return the concrete value directly
    return fallback

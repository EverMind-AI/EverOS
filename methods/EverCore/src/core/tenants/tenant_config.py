"""
Tenant configuration module

This module provides tenant-related configuration management, including configuration options such as the non-tenant mode switch.
Configuration items are loaded from environment variables and support caching to improve performance.
"""

import os
from typing import Optional
from functools import lru_cache

from core.observation.logger import get_logger

logger = get_logger(__name__)


class TenantConfig:
    """
    Tenant configuration class

    This class manages tenant-related configuration options, including:
    - Non-tenant mode switch: controls whether tenant functionality is enabled
    - Single tenant ID: tenant identifier used to activate single-tenant mode
    - Other tenant-related configuration options

    Configuration items are loaded from environment variables and provide a caching mechanism to improve performance.
    """

    def __init__(self):
        """Initialize tenant configuration"""
        self._single_tenant_id: Optional[str] = None
        self._app_ready: bool = (
            False  # Application startup completion status, used for strict tenant checks
        )

    @property
    def single_tenant_id(self) -> Optional[str]:
        """
        Get single tenant ID configuration

        Read configuration from environment variable TENANT_SINGLE_TENANT_ID.
        When this environment variable is set, the system will automatically activate tenant logic for this tenant ID.
        Suitable for single-tenant deployment scenarios.

        Returns:
            Single tenant ID, returns None if not set

        Examples:
            >>> config = get_tenant_config()
            >>> tenant_id = config.single_tenant_id
            >>> if tenant_id:
            ...     print(f"Single tenant mode, tenant ID: {tenant_id}")
        """
        if self._single_tenant_id is None:
            self._single_tenant_id = os.getenv("TENANT_SINGLE_TENANT_ID", "").strip()
            # If empty string, set to None
            if not self._single_tenant_id:
                self._single_tenant_id = None
            else:
                logger.info(
                    "🏢 Single tenant mode activated, tenant ID: %s",
                    self._single_tenant_id,
                )

        return self._single_tenant_id

    @property
    def app_ready(self) -> bool:
        """
        Get application startup completion status

        This status is used for strict tenant checking mode:
        - False: Application is starting, operations without tenant context are allowed (using fallback)
        - True: Application is ready, tenant context is required in tenant mode, otherwise raise error directly

        This is a fallback mechanism used in production environments to catch code errors that miss tenant context.

        Returns:
            bool: True means application is ready, False means application is starting
        """
        return self._app_ready

    def mark_app_ready(self) -> None:
        """
        Mark application startup as complete

        This method should be called after all lifespan providers have started.
        After calling, missing tenant context in tenant mode will raise error directly instead of using fallback logic.

        Note: This method can only be set once; repeated calls will log a warning.
        """
        if self._app_ready:
            logger.warning(
                "⚠️ Application is already ready, mark_app_ready() called repeatedly"
            )
            return

        self._app_ready = True
        logger.info(
            "✅ Application startup completed, tenant strict check mode enabled"
        )

    def reload(self):
        """
        Reload configuration

        Clear cached configuration items and force re-read from environment variables.
        Typically used after testing or configuration changes.

        Note: reload does not reset the app_ready state, as it reflects runtime status rather than configuration.
        """
        self._single_tenant_id = None
        logger.info("🔄 Tenant configuration reloaded")

    def reset_app_ready(self) -> None:
        """
        Reset application ready state (for testing only)

        Warning: This method should only be used in testing scenarios and should not be called in production.
        """
        self._app_ready = False
        logger.warning("⚠️ Application ready state has been reset (for testing only)")


@lru_cache(maxsize=1)
def get_tenant_config() -> TenantConfig:
    """
    Get tenant configuration singleton

    Uses lru_cache to ensure only one configuration instance is created during the application lifecycle.

    Returns:
        TenantConfig: Tenant configuration object

    Examples:
        >>> config = get_tenant_config()
        >>> config = get_tenant_config()
        >>> if config.single_tenant_id:
        ...     print("Single tenant mode")
        ... else:
        ...     print("Multi-tenant mode")
    """
    return TenantConfig()

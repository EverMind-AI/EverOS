# -*- coding: utf-8 -*-
"""
Request history configuration module

Provides configuration interface and default implementation for request history logging.
The default implementation disables request history logging (for opensource version).
Enterprise version can override this to enable the feature.
"""

from abc import ABC, abstractmethod
from typing import Optional

from core.di.decorators import component
from core.di.utils import get_bean_by_type
from core.observation.logger import get_logger

logger = get_logger(__name__)


class RequestHistoryConfig(ABC):
    """
    Request history configuration interface

    Defines the configuration for request history logging feature.
    Different implementations can enable/disable the feature based on deployment needs.

    Using DI mechanism:
    - Default implementation (opensource) disables the feature
    - Enterprise can register a new implementation to enable it
    """

    @abstractmethod
    def is_enabled(self) -> bool:
        """
        Check if request history logging is enabled

        Returns:
            bool: True if enabled, False if disabled
        """
        raise NotImplementedError

    def get_config_name(self) -> str:
        """
        Get the configuration provider name

        Returns:
            str: Configuration provider name
        """
        return self.__class__.__name__


@component("default_request_history_config", primary=True)
class DefaultRequestHistoryConfig(RequestHistoryConfig):
    """
    Default request history configuration (opensource version)

    This implementation disables request history logging by default.
    Suitable for opensource deployments where request replay is not needed.
    """

    def is_enabled(self) -> bool:
        """
        Check if request history logging is enabled

        Returns:
            bool: Always True for opensource version
        """
        return True


# ============================================================================
# Utility functions with caching
# ============================================================================

# Cache for the config instance
_config_cache: Optional[RequestHistoryConfig] = None
_enabled_cache: Optional[bool] = None


def _get_config_instance() -> RequestHistoryConfig:
    """
    Get the RequestHistoryConfig instance from DI container

    Returns:
        RequestHistoryConfig: The configuration instance
    """
    global _config_cache
    if _config_cache is not None:
        return _config_cache

    return get_bean_by_type(RequestHistoryConfig)


def is_request_history_enabled() -> bool:
    """
    Check if request history logging is enabled (with caching)

    This function caches the result for performance.
    The cache is populated on first call and reused thereafter.

    Returns:
        bool: True if request history logging is enabled, False otherwise

    Example:
        >>> from core.request import is_request_history_enabled
        >>> if is_request_history_enabled():
        ...     # Log the request
        ...     pass
    """
    global _enabled_cache
    if _enabled_cache is not None:
        return _enabled_cache

    config = _get_config_instance()
    _enabled_cache = config.is_enabled()

    logger.info(
        f"Request history logging is {'enabled' if _enabled_cache else 'disabled'} "
        f"(config: {config.get_config_name()})"
    )

    return _enabled_cache


def clear_request_history_cache() -> None:
    """
    Clear the request history configuration cache

    Call this if the configuration needs to be reloaded,
    for example during testing or dynamic configuration changes.
    """
    global _config_cache, _enabled_cache
    _config_cache = None
    _enabled_cache = None
    logger.debug("Request history configuration cache cleared")


def get_request_history_config() -> RequestHistoryConfig:
    """
    Get the current RequestHistoryConfig instance

    Returns:
        RequestHistoryConfig: The current configuration instance
    """
    return _get_config_instance()

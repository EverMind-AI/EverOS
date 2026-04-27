"""
Metrics Registry

Centralized management of Prometheus metrics registry with singleton access.
"""
from prometheus_client import CollectorRegistry, REGISTRY, generate_latest
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Global registry instance (uses default REGISTRY)
_metrics_registry: Optional[CollectorRegistry] = None


def get_metrics_registry() -> CollectorRegistry:
    """
    Get the global metrics registry
    
    Returns:
        CollectorRegistry: Prometheus registry instance
    
    Notes:
        - Uses prometheus_client's global REGISTRY by default
        - All metrics are automatically registered to this registry
        - Used by start_metrics_server() for exposing /metrics endpoint
    """
    global _metrics_registry
    if _metrics_registry is None:
        _metrics_registry = REGISTRY
    return _metrics_registry


def set_metrics_registry(registry: CollectorRegistry) -> None:
    """
    Set custom registry (mainly for testing)
    
    Args:
        registry: Custom CollectorRegistry instance
    """
    global _metrics_registry
    _metrics_registry = registry


def generate_metrics_response() -> bytes:
    """
    Generate metrics response content (for testing/debugging)
    
    Returns:
        bytes: Prometheus format metrics data
    """
    return generate_latest(get_metrics_registry())


def reset_metrics_registry() -> None:
    """
    Reset metrics registry (mainly for testing)
    
    Warning: Do not call this method in production
    """
    global _metrics_registry
    _metrics_registry = None
    logger.warning("Metrics registry has been reset")


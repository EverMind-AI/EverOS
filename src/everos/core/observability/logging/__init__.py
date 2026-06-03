"""structlog-based logging factory.

External usage:
    from everos.core.observability.logging import get_logger, configure_logging

    logger = get_logger(__name__)
    logger.info("event_name", key=value)
"""

from .factory import configure_logging as configure_logging
from .factory import get_logger as get_logger

__all__ = ["configure_logging", "get_logger"]

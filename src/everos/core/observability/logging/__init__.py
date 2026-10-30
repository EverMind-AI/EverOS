"""structlog-based logging factory.

External usage:
    from everos.core.observability.logging import get_logger, configure_logging

    logger = get_logger(__name__)
    logger.info("event_name", key=value)

    with bound_log_context(request_id=rid):
        ...  # every event logged here carries request_id
"""

from .factory import bound_log_context as bound_log_context
from .factory import configure_logging as configure_logging
from .factory import get_logger as get_logger

__all__ = ["bound_log_context", "configure_logging", "get_logger"]

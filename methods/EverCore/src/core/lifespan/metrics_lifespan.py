"""
Metrics lifecycle provider implementation

Starts standalone Prometheus metrics server on a separate port (default: 9090).
"""
import os
from fastapi import FastAPI
from typing import Any, Tuple

from core.observation.logger import get_logger
from core.di.decorators import component
from core.observation.metrics import start_metrics_server, is_metrics_server_running, get_metrics_url
from .lifespan_interface import LifespanProvider

logger = get_logger(__name__)


@component(name="metrics_lifespan_provider")
class MetricsLifespanProvider(LifespanProvider):
    """Metrics lifecycle provider - starts Prometheus metrics server"""

    def __init__(self, name: str = "metrics", order: int = 5):
        """
        Initialize the metrics lifecycle provider

        Args:
            name (str): Provider name
            order (int): Execution order, metrics should start early (before database)
        """
        super().__init__(name, order)

    async def startup(self, app: FastAPI) -> Tuple[Any, ...]:
        """
        Start metrics server

        Args:
            app (FastAPI): FastAPI application instance

        Returns:
            Tuple containing metrics server info
        """
        # Get port from environment variable or default to 9090
        port = int(os.getenv("METRICS_PORT", "9090"))
        
        logger.info("Starting Prometheus metrics server on port %d...", port)
        
        try:
            success = start_metrics_server(port=port)
            
            if success:
                logger.info("✅ Metrics server started: %s", get_metrics_url(port=port))
                app.state.metrics_port = port
            else:
                logger.warning("Metrics server already running or failed to start")
            
            return (port, success)
            
        except Exception as e:
            logger.error("Failed to start metrics server: %s", str(e))
            # Don't raise - metrics failure shouldn't prevent app startup
            return (port, False)

    async def shutdown(self, app: FastAPI) -> None:
        """
        Cleanup metrics server (daemon thread auto-stops with main process)

        Args:
            app (FastAPI): FastAPI application instance
        """
        logger.info("Metrics server will stop with main process (daemon thread)")
        
        # Clean up app.state
        if hasattr(app.state, 'metrics_port'):
            delattr(app.state, 'metrics_port')


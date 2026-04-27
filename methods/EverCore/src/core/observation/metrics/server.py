"""
Standalone Metrics Server

Runs Prometheus metrics endpoint on a separate port (default: 9090).
This isolates metrics from business API for security and operational benefits.

Usage:
    from core.observation.metrics.server import start_metrics_server
    
    # Start metrics server on port 9090
    start_metrics_server(port=9090)
    
    # Or use environment variable METRICS_PORT
    start_metrics_server()  # reads from METRICS_PORT or defaults to 9090

Benefits:
    - Security: Metrics endpoint can be firewalled separately
    - Availability: Metrics available even if main app is overloaded
    - Operations: Can expose to internal network only
"""
import os
import logging
from typing import Optional
from prometheus_client import start_http_server
from .registry import get_metrics_registry

logger = logging.getLogger(__name__)

# Global server state
_metrics_server_started: bool = False


def start_metrics_server(
    port: Optional[int] = None,
    addr: str = "0.0.0.0",
) -> bool:
    """
    Start standalone Prometheus metrics HTTP server
    
    Args:
        port: Port to listen on (default: from METRICS_PORT env or 9090)
        addr: Address to bind to (default: 0.0.0.0)
    
    Returns:
        bool: True if server started successfully, False if already running
    
    Example:
        # Start on default port 9090
        start_metrics_server()
        
        # Start on custom port
        start_metrics_server(port=9091)
        
        # Prometheus can scrape: http://your-host:9090/metrics
    """
    global _metrics_server_started
    
    if _metrics_server_started:
        logger.warning("Metrics server already running")
        return False
    
    # Get port from parameter, env var, or default
    if port is None:
        port = int(os.getenv("METRICS_PORT", "9090"))
    
    try:
        # Start HTTP server using prometheus_client's built-in server
        # This creates a daemon thread that serves /metrics endpoint
        start_http_server(
            port=port,
            addr=addr,
            registry=get_metrics_registry(),
        )
        
        _metrics_server_started = True
        logger.info(f"✅ Metrics server started on {addr}:{port}/metrics")
        return True
        
    except Exception as e:
        logger.error(f"Failed to start metrics server: {e}")
        return False


def is_metrics_server_running() -> bool:
    """Check if metrics server is running"""
    return _metrics_server_started


def get_metrics_url(host: str = "localhost", port: Optional[int] = None) -> str:
    """
    Get the metrics endpoint URL
    
    Args:
        host: Hostname (default: localhost)
        port: Port (default: from METRICS_PORT env or 9090)
    
    Returns:
        str: Full URL to metrics endpoint
    """
    if port is None:
        port = int(os.getenv("METRICS_PORT", "9090"))
    return f"http://{host}:{port}/metrics"


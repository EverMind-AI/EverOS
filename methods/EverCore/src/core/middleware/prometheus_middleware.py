"""
Prometheus HTTP Metrics Middleware

Auto-instrumentation middleware for HTTP request metrics.
Inspired by Kratos, Hertz, and other microservice frameworks.

Usage:
    from fastapi import FastAPI
    from core.middleware.prometheus_middleware import PrometheusMiddleware
    
    app = FastAPI()
    app.add_middleware(PrometheusMiddleware)
"""
from core.observation.logger import get_logger
import time
from typing import Callable
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from prometheus_client import Counter, Histogram
from core.observation.metrics.registry import get_metrics_registry

logger = get_logger(__name__)


# Pre-defined HTTP metrics (following Prometheus naming conventions)
_http_requests_total = Counter(
    name='http_requests_total',
    documentation='Total number of HTTP requests',
    labelnames=['method', 'path', 'status'],
    namespace='evermemos',
    registry=get_metrics_registry(),
)

_http_request_duration_seconds = Histogram(
    name='http_request_duration_seconds',
    documentation='HTTP request duration in seconds',
    labelnames=['method', 'path'],
    namespace='evermemos',
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    registry=get_metrics_registry(),
)

_http_request_size_bytes = Histogram(
    name='http_request_size_bytes',
    documentation='HTTP request size in bytes',
    labelnames=['method', 'path'],
    namespace='evermemos',
    buckets=(100, 1000, 10000, 100000, 1000000, 10000000),
    registry=get_metrics_registry(),
)

_http_response_size_bytes = Histogram(
    name='http_response_size_bytes',
    documentation='HTTP response size in bytes',
    labelnames=['method', 'path'],
    namespace='evermemos',
    buckets=(100, 1000, 10000, 100000, 1000000, 10000000),
    registry=get_metrics_registry(),
)


def _get_fastapi_route_template(request: Request) -> str:
    """
    Get the actual route template from FastAPI request.

    Args:
        request: FastAPI request object

    Returns:
        str: Route template string, empty string if not available
    """
    try:
        # Get route info from request.scope (available after request processing)
        if hasattr(request, 'scope') and 'route' in request.scope:
            route = request.scope['route']
            if hasattr(route, 'path'):
                return route.path

        # If no route in scope, try to infer from path_params
        if hasattr(request, 'path_params') and request.path_params:
            path = request.url.path
            for param_name, param_value in request.path_params.items():
                if str(param_value) in path:
                    path = path.replace(str(param_value), f"{{{param_name}}}")
            return path

    except Exception as e:
        logger.debug("Failed to get FastAPI route template: %s", str(e))

    return ""


def _normalize_path(request: Request) -> str:
    """
    Get normalized path label, prefer FastAPI route template.

    Strategy:
    1. Try to get actual route template from FastAPI route info
    2. Mark unmatched paths as {unmatched}

    Examples:
    - /api/users/123 -> /api/users/{user_id} (FastAPI route template)
    - /unknown/path -> {unmatched} (unmatched path)

    Args:
        request: FastAPI request object

    Returns:
        str: Normalized path
    """
    route_template = _get_fastapi_route_template(request)
    if route_template:
        return route_template

    return '{unmatched}'



class PrometheusMiddleware(BaseHTTPMiddleware):
    """
    Prometheus HTTP Metrics Middleware
    
    Automatically records:
    - http_requests_total (Counter): Total requests by method, path, status
    - http_request_duration_seconds (Histogram): Request latency
    - http_request_size_bytes (Histogram): Request body size
    - http_response_size_bytes (Histogram): Response body size
    
    Design inspired by:
    - Kratos (Bilibili): middleware.Middleware pattern
    - Hertz (ByteDance): promhttp.InstrumentHandler pattern
    - Go kit: endpoint.Middleware pattern
    
    Usage:
        app = FastAPI()
        app.add_middleware(PrometheusMiddleware)
    """
    
    # Paths to skip metrics collection
    SKIP_PATHS = {'/metrics', '/health', '/healthz', '/ready', '/favicon.ico'}
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip metrics for certain paths
        if request.url.path in self.SKIP_PATHS:
            return await call_next(request)
        
        method = request.method
        
        # Record request size (before processing)
        request_size = 0
        if request.headers.get('content-length'):
            request_size = int(request.headers.get('content-length', 0))
        
        # Time the request
        start_time = time.perf_counter()
        status = '500'  # Default to 500 in case of unhandled exception
        response = None
        
        try:
            response = await call_next(request)
            status = str(response.status_code)
        except Exception:
            raise
        finally:
            # Get path AFTER call_next - route info is now available
            path = _normalize_path(request)
            
            # Record metrics
            duration = time.perf_counter() - start_time
            
            _http_requests_total.labels(
                method=method,
                path=path,
                status=status,
            ).inc()
            
            _http_request_duration_seconds.labels(
                method=method,
                path=path,
            ).observe(duration)
            
            # Record request size
            if request_size > 0:
                _http_request_size_bytes.labels(method=method, path=path).observe(request_size)
        
        # Record response size
        if response and hasattr(response, 'headers') and response.headers.get('content-length'):
            response_size = int(response.headers.get('content-length', 0))
            _http_response_size_bytes.labels(method=method, path=path).observe(response_size)
        
        return response


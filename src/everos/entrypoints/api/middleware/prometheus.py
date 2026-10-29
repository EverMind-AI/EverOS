"""Prometheus HTTP metrics middleware.

Auto-instruments incoming HTTP requests with a request counter and a
duration histogram. Mounted via ``app.add_middleware(PrometheusMiddleware)``.

Skips internal endpoints (``/metrics``, ``/health``, etc.) so they do not
inflate cardinality or pollute their own statistics.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from everos.core.observability.logging import get_logger
from everos.core.observability.metrics import Counter, Histogram, HistogramBuckets

logger = get_logger(__name__)


_http_requests_total = Counter(
    name="http_requests_total",
    description="Total number of HTTP requests handled.",
    labelnames=("method", "path", "status"),
    namespace="everos",
)

_http_request_duration_seconds = Histogram(
    name="http_request_duration_seconds",
    description="HTTP request duration in seconds.",
    labelnames=("method", "path"),
    namespace="everos",
    buckets=HistogramBuckets.DEFAULT,
)


_SKIP_PATHS = frozenset({"/metrics", "/health", "/healthz", "/favicon.ico"})


def _normalize_path(request: Request) -> str:
    """Resolve the full route template (e.g. ``/api/v1/users/{user_id}``).

    Built from the concrete request path with matched path params folded
    back to ``{name}`` placeholders — NOT from ``route.path``. A route
    mounted under a prefix (the ``/api/vN`` version aliases) only carries
    its router-relative path on the route object (e.g. ``/memory/get``);
    the prefix lives on the mounted router. Using ``route.path`` would drop
    the version prefix and collapse v1/v2 traffic into one label, so the
    label is rebuilt from ``url.path`` instead. Unmatched requests (no
    ``route`` in scope) fold to a single bucket to bound cardinality.
    """
    scope = getattr(request, "scope", {})
    route = scope.get("route") if isinstance(scope, dict) else None
    if route is None:
        return "{unmatched}"
    path = request.url.path
    for name, value in request.path_params.items():
        value_str = str(value)
        if value_str:
            path = path.replace(value_str, f"{{{name}}}")
    return path


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Records ``http_requests_total`` and ``http_request_duration_seconds``."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        if request.url.path in _SKIP_PATHS:
            return await call_next(request)

        method = request.method
        start = time.perf_counter()
        status = "500"
        response: Response | None = None
        try:
            response = await call_next(request)
            status = str(response.status_code)
            return response
        finally:
            duration = time.perf_counter() - start
            path = _normalize_path(request)
            _http_requests_total.labels(method=method, path=path, status=status).inc()
            _http_request_duration_seconds.labels(method=method, path=path).observe(
                duration
            )

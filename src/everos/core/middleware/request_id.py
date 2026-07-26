"""Request-context middleware.

Establishes per-request context at HTTP entry:

* Mints a W3C-compatible ``request_id`` and binds it for the request's
  lifetime — via ``request.state``, the ``core.context`` contextvar
  (readable by service / infra), and structlog contextvars (so every log
  line carries it). Echoed on the ``X-Request-Id`` response header.
* Continues an upstream **distributed trace**: if the request carries a
  W3C ``traceparent`` header, our first span nests under that trace instead
  of rooting a new one. Absent → we root our own trace (the common case).

This is the single place a request id enters the system; ``extract_request_id``
(``entrypoints/api/utils.py``) reads what this middleware sets.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from everos.core.context import reset_request_id, set_request_id
from everos.core.observability.tracing import gen_request_id, use_traceparent

_HEADER = "X-Request-Id"


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Assigns a request id + continues any upstream trace for the request."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = gen_request_id()
        request.state.request_id = request_id
        # Bind before ``call_next`` so the downstream task (endpoint) inherits
        # the value; reset in ``finally`` so it never leaks to the next request.
        token = set_request_id(request_id)
        structlog.contextvars.bind_contextvars(request_id=request_id)
        try:
            # Continue the upstream trace when a traceparent header is present
            # (no-op otherwise). Attached before call_next so downstream spans
            # inherit it across the middleware task boundary.
            with use_traceparent(request.headers.get("traceparent")):
                response = await call_next(request)
            response.headers[_HEADER] = request_id
            return response
        finally:
            structlog.contextvars.unbind_contextvars("request_id")
            reset_request_id(token)

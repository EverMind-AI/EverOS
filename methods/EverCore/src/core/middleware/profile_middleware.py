"""
Performance Profiling Middleware

Provides performance profiling functionality for HTTP requests based on the pyinstrument library.

Features:
1. URL parameter trigger: enable profiling by adding ?profile=true to the request URL
2. HTML report: returns a visual performance analysis HTML report
3. Environment variable control: enable/disable via the PROFILING_ENABLED environment variable
4. Graceful degradation: automatically disabled if pyinstrument is not installed, without affecting normal requests

Environment Variables:
- PROFILING_ENABLED: whether to enable profiling (default: false)
- PROFILING: same as PROFILING_ENABLED (alternative environment variable name)

Usage:
1. Set environment variable: export PROFILING_ENABLED=true
2. Install dependency: uv add pyinstrument
3. Add parameter when accessing endpoint: http://localhost:8000/api/endpoint?profile=true
"""

import os
from typing import Callable, Optional

from fastapi import Request
from fastapi.responses import HTMLResponse
from starlette.responses import Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from core.observation.logger import get_logger

logger = get_logger(__name__)


class ProfileMiddleware(BaseHTTPMiddleware):
    """
    Performance profiling middleware

    Enables performance profiling when the request URL contains the ?profile=true parameter and returns an HTML-formatted analysis report
    """

    def __init__(self, app: ASGIApp):
        """
        Initialize the performance profiling middleware

        Args:
            app: ASGI application instance
        """
        super().__init__(app)

        # Read from environment variable whether profiling is enabled
        profiling_env = os.getenv(
            'PROFILING_ENABLED', os.getenv('PROFILING', 'true')
        ).lower()
        self._profiling_enabled = profiling_env in ('true', '1', 'yes')

        # Check if pyinstrument is available
        self._profiler_available = False
        if self._profiling_enabled:
            try:
                import pyinstrument

                self._profiler_available = True
                logger.info("✅ Performance profiling middleware enabled")
                logger.info(
                    "Tip: Add ?profile=true parameter to the request URL to enable profiling"
                )
            except ImportError:
                logger.warning(
                    "⚠️ pyinstrument not installed, profiling feature will be disabled"
                )
                logger.warning("Please run: uv add pyinstrument")
                self._profiling_enabled = False
        else:
            logger.debug(
                "Performance profiling is not enabled (set environment variable PROFILING_ENABLED=true to enable)"
            )

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Handle HTTP requests and perform performance profiling when needed

        Args:
            request: FastAPI request object
            call_next: next middleware or route handler

        Returns:
            Response: response object (normal response or profiling report)
        """
        # If feature is not enabled, pass through directly
        if not self._profiling_enabled or not self._profiler_available:
            return await call_next(request)

        # Check if profiling is required
        profiling = request.query_params.get("profile", "").lower() in (
            "true",
            "1",
            "yes",
        )

        if not profiling:
            # No profiling needed, process request normally
            return await call_next(request)

        # Profiling is needed
        try:
            # Dynamic import (although availability has been checked, import when used for type safety)
            from pyinstrument import Profiler

            # Create and start profiler
            profiler = Profiler()
            profiler.start()

            logger.info("Profiling started: %s %s", request.method, request.url.path)

            try:
                # Execute request (note: original response will be discarded and replaced with profiler report)
                await call_next(request)
            except Exception as e:
                # Even if the request fails, stop profiler and return profiling report
                logger.error("Request failed during profiling: %s", str(e))
                # Continue generating profiling report

            # Stop profiler
            profiler.stop()

            # Generate HTML report
            html_output = profiler.output_html()

            logger.info("Profiling completed: %s %s", request.method, request.url.path)

            # Return HTML-formatted profiling report
            return HTMLResponse(content=html_output, status_code=200)

        except Exception as e:
            logger.error("Error occurred during profiling: %s", str(e))
            # If profiling fails, re-execute normal request
            return await call_next(request)

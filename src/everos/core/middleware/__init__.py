"""Cross-cutting HTTP middleware components.

External usage::

    from everos.core.middleware import (
        DEFAULT_CORS_ALLOW_CREDENTIALS,
        DEFAULT_CORS_ALLOW_HEADERS,
        DEFAULT_CORS_ALLOW_METHODS,
        DEFAULT_CORS_ORIGINS,
        ProfileMiddleware,
        PrometheusMiddleware,
    )
"""

from .cors import DEFAULT_CORS_ALLOW_CREDENTIALS as DEFAULT_CORS_ALLOW_CREDENTIALS
from .cors import DEFAULT_CORS_ALLOW_HEADERS as DEFAULT_CORS_ALLOW_HEADERS
from .cors import DEFAULT_CORS_ALLOW_METHODS as DEFAULT_CORS_ALLOW_METHODS
from .cors import DEFAULT_CORS_ORIGINS as DEFAULT_CORS_ORIGINS
from .profile import ProfileMiddleware as ProfileMiddleware
from .prometheus import PrometheusMiddleware as PrometheusMiddleware

__all__ = [
    "DEFAULT_CORS_ALLOW_CREDENTIALS",
    "DEFAULT_CORS_ALLOW_HEADERS",
    "DEFAULT_CORS_ALLOW_METHODS",
    "DEFAULT_CORS_ORIGINS",
    "ProfileMiddleware",
    "PrometheusMiddleware",
]

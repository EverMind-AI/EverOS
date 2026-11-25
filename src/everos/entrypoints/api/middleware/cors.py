"""CORS configuration defaults.

The CORS middleware itself is FastAPI's stock ``CORSMiddleware``; this module
centralises the default policy values used by the application factory. No
origin is allowed by default: browser pages of other sites may not call the
API unless ``[api] cors_origins`` lists them.
"""

from __future__ import annotations

DEFAULT_CORS_ALLOW_CREDENTIALS: bool = False
DEFAULT_CORS_ALLOW_HEADERS: list[str] = ["*"]
DEFAULT_CORS_ALLOW_METHODS: list[str] = ["*"]
DEFAULT_CORS_ORIGINS: list[str] = []

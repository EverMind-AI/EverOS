"""Response envelopes shared by every route.

A success answers ``{"request_id": ..., "data": ...}`` (``SuccessEnvelope``);
an error answers ``{"request_id": ..., "error": {...}}`` (``ErrorResponse``),
built by :mod:`everos.entrypoints.api.exception_handlers`.
"""

from __future__ import annotations

from pydantic import BaseModel

from everos.core.errors import ErrorCode


class SuccessEnvelope[T](BaseModel):
    """200 wrapper: ``request_id`` sits at the top level, not inside ``data``."""

    request_id: str
    data: T


class ErrorDetail(BaseModel):
    """Inner ``error`` object in the canonical error envelope."""

    code: ErrorCode
    message: str
    timestamp: str
    path: str


class ErrorResponse(BaseModel):
    """Canonical error envelope returned by all error handlers."""

    request_id: str
    error: ErrorDetail

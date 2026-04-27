"""Base DTO types for API responses.

This module contains common base types used across all API endpoints.
"""

from __future__ import annotations

from typing import Generic, TypeVar

from pydantic import BaseModel, Field


# Generic type for API response result
T = TypeVar("T")


class BaseApiResponse(BaseModel, Generic[T]):
    """Base API response wrapper

    Unified response format for all API endpoints.
    """

    data: T = Field(description="Response result data")

    model_config = {"arbitrary_types_allowed": True}


class ErrorApiResponse(BaseModel):
    """Unified error response model

    Used by global_exception_handler to construct consistent error responses
    for both HTTPException and unhandled exceptions.
    """

    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    request_id: str = Field(default="unknown", description="Request ID")
    timestamp: str = Field(..., description="ISO format timestamp")
    path: str = Field(..., description="Request path")

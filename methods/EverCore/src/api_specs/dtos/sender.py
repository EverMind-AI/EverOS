"""Sender resource DTOs.

This module contains DTOs for sender CRUD operations:
- Create (POST /api/v1/senders)
- Get (GET /api/v1/senders/{sender_id})
- Patch (PATCH /api/v1/senders/{sender_id})
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from api_specs.dtos.base import BaseApiResponse


# =============================================================================
# Request DTOs
# =============================================================================


class CreateSenderRequest(BaseModel):
    """Create sender request body

    Used for POST /api/v1/senders endpoint.
    If a sender with the same sender_id exists, it will be updated (upsert).
    """

    sender_id: str = Field(
        ..., description="Sender identifier (unique)", examples=["user_123"]
    )
    name: Optional[str] = Field(
        default=None, description="Sender display name", examples=["Alice"]
    )

    model_config = {
        "json_schema_extra": {"example": {"sender_id": "user_123", "name": "Alice"}}
    }


class PatchSenderRequest(BaseModel):
    """Partial update sender request body

    Used for PATCH /api/v1/senders/{sender_id} endpoint.
    """

    name: Optional[str] = Field(
        default=None, description="New sender display name", examples=["Alice Updated"]
    )

    model_config = {"json_schema_extra": {"example": {"name": "Alice Updated"}}}


# =============================================================================
# Response DTOs
# =============================================================================


class SenderResponse(BaseModel):
    """Sender response DTO

    Returned by all sender endpoints.
    """

    sender_id: str = Field(..., description="Sender identifier")
    name: Optional[str] = Field(default=None, description="Sender display name")
    created_at: str = Field(..., description="Creation time (ISO 8601)")
    updated_at: str = Field(..., description="Last update time (ISO 8601)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "sender_id": "user_123",
                "name": "Alice",
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-02-26T10:00:00+00:00",
            }
        }
    }


# =============================================================================
# API Response Wrappers
# =============================================================================


class CreateSenderApiResponse(BaseApiResponse[SenderResponse]):
    """Create sender API response"""

    data: SenderResponse = Field(description="Created/updated sender data")


class GetSenderApiResponse(BaseApiResponse[SenderResponse]):
    """Get sender API response"""

    data: SenderResponse = Field(description="Sender data")


class PatchSenderApiResponse(BaseApiResponse[SenderResponse]):
    """Patch sender API response"""

    data: SenderResponse = Field(description="Updated sender data")

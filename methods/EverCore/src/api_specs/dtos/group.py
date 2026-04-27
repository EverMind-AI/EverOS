"""Group resource DTOs.

This module contains DTOs for group CRUD operations:
- Create (POST /api/v1/groups)
- Get (GET /api/v1/groups/{group_id})
- Patch (PATCH /api/v1/groups/{group_id})
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, model_validator

from api_specs.dtos.base import BaseApiResponse


# =============================================================================
# Request DTOs
# =============================================================================


class CreateGroupRequest(BaseModel):
    """Create group request body

    Used for POST /api/v1/groups endpoint.
    If a group with the same group_id exists, it will be updated (upsert).
    """

    group_id: str = Field(
        ..., description="Group identifier (unique)", examples=["group_abc"]
    )
    name: Optional[str] = Field(
        default=None, description="Group display name", examples=["Project Discussion"]
    )
    description: Optional[str] = Field(
        default=None,
        description="Group description",
        examples=["Weekly sync on Project X"],
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "group_id": "group_abc",
                "name": "Project Discussion",
                "description": "Weekly sync on Project X",
            }
        }
    }


class PatchGroupRequest(BaseModel):
    """Partial update group request body

    Used for PATCH /api/v1/groups/{group_id} endpoint.
    At least one field must be provided.
    """

    name: Optional[str] = Field(
        default=None,
        description="New group display name",
        examples=["Updated Group Name"],
    )
    description: Optional[str] = Field(
        default=None,
        description="New group description",
        examples=["Updated description"],
    )

    @model_validator(mode="after")
    def validate_at_least_one_field(self):
        """At least one field must be provided for patch"""
        if self.name is None and self.description is None:
            raise ValueError("At least one of 'name' or 'description' must be provided")
        return self

    model_config = {"json_schema_extra": {"example": {"name": "Updated Group Name"}}}


# =============================================================================
# Response DTOs
# =============================================================================


class GroupResponse(BaseModel):
    """Group response DTO

    Returned by all group endpoints.
    """

    group_id: str = Field(..., description="Group identifier")
    name: Optional[str] = Field(default=None, description="Group display name")
    description: Optional[str] = Field(default=None, description="Group description")
    created_at: str = Field(..., description="Creation time (ISO 8601)")
    updated_at: str = Field(..., description="Last update time (ISO 8601)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "group_id": "group_abc",
                "name": "Project Discussion",
                "description": "Weekly sync on Project X",
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-02-26T10:00:00+00:00",
            }
        }
    }


# =============================================================================
# API Response Wrappers
# =============================================================================


class CreateGroupApiResponse(BaseApiResponse[GroupResponse]):
    """Create group API response"""

    data: GroupResponse = Field(description="Created/updated group data")


class GetGroupApiResponse(BaseApiResponse[GroupResponse]):
    """Get group API response"""

    data: GroupResponse = Field(description="Group data")


class PatchGroupApiResponse(BaseApiResponse[GroupResponse]):
    """Patch group API response"""

    data: GroupResponse = Field(description="Updated group data")

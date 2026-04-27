"""DTOs for POST /api/v1/memories/delete endpoint."""

from typing import Optional

from pydantic import BaseModel, Field, model_validator

from core.oxm.constants import MAGIC_ALL


class DeleteMemoriesRequest(BaseModel):
    """Delete memories request body.

    Two mutually exclusive modes:
    - By ID: provide memory_id only (no other fields allowed)
    - By filters: provide user_id and/or group_id (session_id, sender_id optional)

    Three-state filter semantics:
    - MAGIC_ALL (default/not provided): skip this filter
    - None or "": match null/empty records
    - "alice": exact match
    """

    memory_id: Optional[str] = Field(
        default=None, description="MemCell ID for single delete"
    )
    user_id: Optional[str] = Field(
        default=MAGIC_ALL, description="User ID scope for batch delete"
    )
    group_id: Optional[str] = Field(
        default=MAGIC_ALL, description="Group ID scope for batch delete"
    )
    session_id: Optional[str] = Field(
        default=MAGIC_ALL, description="Session filter (batch delete only)"
    )
    sender_id: Optional[str] = Field(
        default=MAGIC_ALL,
        description="Sender filter, matches participants array (batch delete only)",
    )

    @model_validator(mode="after")
    def validate_mode_exclusivity(self):
        filter_fields = [self.user_id, self.group_id, self.session_id, self.sender_id]
        if self.memory_id is not None:
            if any(f != MAGIC_ALL for f in filter_fields):
                raise ValueError(
                    "When memory_id is provided, no other fields "
                    "(user_id, group_id, session_id, sender_id) are allowed"
                )
        else:
            if self.user_id == MAGIC_ALL and self.group_id == MAGIC_ALL:
                raise ValueError(
                    "Either memory_id must be provided, or at least one of "
                    "user_id / group_id is required"
                )
        return self

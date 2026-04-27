"""Settings resource DTOs.

This module contains DTOs for Settings CRUD operations:
- Get (GET /api/v1/settings)
- Update (PUT /api/v1/settings)

The Settings API manages the singleton global configuration per space.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from api_specs.dtos.base import BaseApiResponse


# =============================================================================
# LLM Configuration Types
# =============================================================================


class LlmProviderConfig(BaseModel):
    """LLM provider configuration

    Defines the provider and model for a specific LLM task.
    """

    provider: str = Field(
        ...,
        description="LLM provider name",
        examples=["openai", "openrouter", "anthropic"],
    )
    model: str = Field(
        ...,
        description="Model name",
        examples=["gpt-4.1-mini", "qwen/qwen3-235b-a22b-2507"],
    )
    extra: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional provider-specific configuration",
        examples=[{"temperature": 0.7, "max_tokens": 1024}],
    )


class LlmCustomSetting(BaseModel):
    """LLM custom settings for algorithm control

    Allows configuring different LLM providers/models for different tasks.

    Example:
        {
            "boundary": {"provider": "openai", "model": "gpt-4.1-mini"},
            "extraction": {"provider": "openrouter", "model": "qwen/qwen3-235b-a22b-2507"}
        }
    """

    boundary: Optional[LlmProviderConfig] = Field(
        default=None,
        description="LLM config for boundary detection (fast, cheap model recommended)",
        examples=[{"provider": "openai", "model": "gpt-4.1-mini"}],
    )
    extraction: Optional[LlmProviderConfig] = Field(
        default=None,
        description="LLM config for memory extraction (high quality model recommended)",
        examples=[{"provider": "openrouter", "model": "qwen/qwen3-235b-a22b-2507"}],
    )
    extra: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional task-specific LLM configurations"
    )


# =============================================================================
# Request DTOs
# =============================================================================


class UpdateSettingsRequest(BaseModel):
    """Update settings request body

    Used for PUT /api/v1/settings endpoint.
    Handles both initialization (first call) and updates.

    """

    llm_custom_setting: Optional[LlmCustomSetting] = Field(
        default=None,
        description="LLM custom settings for algorithm control. "
        "Validated against provider whitelist.",
        examples=[
            {
                "boundary": {"provider": "openai", "model": "gpt-4.1-mini"},
                "extraction": {
                    "provider": "openrouter",
                    "model": "qwen/qwen3-235b-a22b-2507",
                },
            }
        ],
    )
    # Hidden fields: not yet implemented, uncomment when ready
    # timezone: Optional[str] = Field(
    #     default=None,
    #     description="IANA timezone identifier",
    #     examples=["UTC", "Asia/Shanghai"],
    # )
    # boundary_detection_timeout: Optional[int] = Field(
    #     default=None,
    #     description="MemCell auto-flush idle timeout in seconds",
    #     examples=[3600],
    # )
    # extraction_mode: Optional[str] = Field(
    #     default=None,
    #     description="Extraction mode: 'default' or 'pro'",
    #     examples=["default", "pro"],
    # )
    # offline_profile_extraction_interval: Optional[int] = Field(
    #     default=None,
    #     description="Offline profile extraction interval in seconds",
    #     examples=[86400],
    # )


# =============================================================================
# Response DTOs
# =============================================================================


class SettingsResponse(BaseModel):
    """Settings response DTO

    Returned by GET and PUT /api/v1/settings endpoints.
    """

    llm_custom_setting: Optional[Dict[str, Any]] = Field(
        default=None, description="LLM custom settings (serialized)"
    )
    # Hidden fields: not yet implemented, uncomment when ready
    # timezone: str = Field(..., description="IANA timezone identifier")
    # boundary_detection_timeout: int = Field(
    #     ..., description="MemCell auto-flush idle timeout in seconds"
    # )
    # extraction_mode: str = Field(..., description="Extraction mode")
    # offline_profile_extraction_interval: int = Field(
    #     ..., description="Offline profile extraction interval in seconds"
    # )
    created_at: str = Field(..., description="Creation time (ISO 8601)")
    updated_at: str = Field(..., description="Last update time (ISO 8601)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "llm_custom_setting": {
                    "boundary": {"provider": "openai", "model": "gpt-4.1-mini"},
                    "extraction": {"provider": "openai", "model": "gpt-4o"},
                },
                # "timezone": "UTC",
                # "boundary_detection_timeout": 3600,
                # "extraction_mode": "default",
                # "offline_profile_extraction_interval": 86400,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-03-05T10:00:00+00:00",
            }
        }
    }


# =============================================================================
# API Response Wrappers
# =============================================================================


class GetSettingsApiResponse(BaseApiResponse[SettingsResponse]):
    """Get settings API response"""

    data: SettingsResponse = Field(description="Settings data")


class UpdateSettingsApiResponse(BaseApiResponse[SettingsResponse]):
    """Update settings API response"""

    data: SettingsResponse = Field(description="Updated settings data")

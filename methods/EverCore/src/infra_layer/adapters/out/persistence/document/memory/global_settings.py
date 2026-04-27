"""
GlobalSettings Beanie ODM model

Space-level global configuration singleton document.
"""

from datetime import datetime
from typing import Optional, Dict, Any
from core.tenants.tenantize.oxm.mongo.tenant_aware_document import (
    TenantAwareDocumentBase,
)
from core.oxm.mongo.audit_base import AuditBase
from pydantic import BaseModel, Field, ConfigDict
from pymongo import IndexModel, ASCENDING, DESCENDING


class LlmProviderConfigModel(BaseModel):
    """LLM provider configuration model

    Defines the provider and model for a specific LLM task
    """

    provider: str = Field(
        ..., description="LLM provider name, e.g.: openai, openrouter."
    )
    model: str = Field(
        ...,
        description="Model name, e.g.: qwen/qwen3-235b-a22b-2507(openrouter), gpt-4.1-mini(openai), etc.",
    )
    extra: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional provider-specific configuration"
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict representation"""
        result = {"provider": self.provider, "model": self.model}
        if self.extra:
            result["extra"] = self.extra
        return result

    @classmethod
    def from_any(cls, data: Any) -> Optional["LlmProviderConfigModel"]:
        """Create from dict or DTO object"""
        if data is None:
            return None
        if isinstance(data, cls):
            return data
        if isinstance(data, dict):
            return cls(
                provider=data.get("provider", ""),
                model=data.get("model", ""),
                extra=data.get("extra"),
            )
        if hasattr(data, "provider") and hasattr(data, "model"):
            return cls(
                provider=data.provider,
                model=data.model,
                extra=getattr(data, "extra", None),
            )
        return None


class LlmCustomSettingModel(BaseModel):
    """LLM custom settings model for algorithm control

    Allows configuring different LLM providers/models for different tasks.

    Example:
        {
            "boundary": {"provider": "openai", "model": "gpt-4.1-mini"},
            "extraction": {"provider": "openrouter", "model": "qwen/qwen3-235b-a22b-2507"}
        }
    """

    boundary: Optional[LlmProviderConfigModel] = Field(
        default=None,
        description="LLM config for boundary detection (fast, cheap model recommended)",
    )
    extraction: Optional[LlmProviderConfigModel] = Field(
        default=None,
        description="LLM config for memory extraction (high quality model recommended)",
    )
    extra: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional task-specific LLM configurations"
    )

    def to_dict(self) -> Optional[Dict[str, Any]]:
        """Convert to dict representation for response"""
        result: Dict[str, Any] = {}
        if self.boundary:
            result["boundary"] = self.boundary.to_dict()
        if self.extraction:
            result["extraction"] = self.extraction.to_dict()
        if self.extra:
            result["extra"] = self.extra
        return result if result else None

    @classmethod
    def from_any(cls, data: Any) -> Optional["LlmCustomSettingModel"]:
        """Create from dict or DTO object"""
        if data is None:
            return None
        if isinstance(data, cls):
            return data
        if isinstance(data, dict):
            boundary = LlmProviderConfigModel.from_any(data.get("boundary"))
            extraction = LlmProviderConfigModel.from_any(data.get("extraction"))
            extra = data.get("extra")
            if boundary is None and extraction is None and extra is None:
                return None
            return cls(boundary=boundary, extraction=extraction, extra=extra)
        if hasattr(data, "boundary") or hasattr(data, "extraction"):
            boundary = LlmProviderConfigModel.from_any(getattr(data, "boundary", None))
            extraction = LlmProviderConfigModel.from_any(
                getattr(data, "extraction", None)
            )
            extra = getattr(data, "extra", None)
            if boundary is None and extraction is None and extra is None:
                return None
            return cls(boundary=boundary, extraction=extraction, extra=extra)
        return None


class GlobalSettings(TenantAwareDocumentBase, AuditBase):
    """
    Global settings document model

    Singleton document per space/namespace storing space-level configuration.
    Singleton document per space/namespace storing space-level settings.
    """

    # LLM configuration
    llm_custom_setting: Optional[LlmCustomSettingModel] = Field(
        default=None,
        description="LLM config: {boundary: {provider, model, extra}, extraction: {provider, model, extra}}",
    )

    # Timezone
    timezone: str = Field(
        default="UTC", description="IANA timezone identifier, e.g. Asia/Shanghai"
    )

    # Memory extraction settings
    boundary_detection_timeout: int = Field(
        default=3600, description="MemCell auto-flush idle timeout in seconds"
    )
    extraction_mode: str = Field(
        default="default", description="Extraction mode: 'default' or 'pro'"
    )
    offline_profile_extraction_interval: int = Field(
        default=86400, description="Offline profile extraction interval in seconds"
    )

    model_config = ConfigDict(
        collection="v1_global_settings",
        validate_assignment=True,
        json_encoders={datetime: lambda dt: dt.isoformat()},
        json_schema_extra={
            "example": {
                "llm_custom_setting": {
                    "boundary": {"provider": "openai", "model": "gpt-4o-mini"},
                    "extraction": {"provider": "openai", "model": "gpt-4o"},
                },
                "timezone": "UTC",
                "boundary_detection_timeout": 3600,
                "extraction_mode": "default",
                "offline_profile_extraction_interval": 86400,
            }
        },
        extra="allow",
    )

    class Settings:
        """Beanie settings"""

        name = "v1_global_settings"
        indexes = [
            IndexModel(
                [("tenant_id", ASCENDING), ("created_at", DESCENDING)],
                name="idx_tenant_created_at",
            ),
            IndexModel(
                [("tenant_id", ASCENDING), ("updated_at", DESCENDING)],
                name="idx_tenant_updated_at",
            ),
        ]
        validate_on_save = True
        use_state_management = True

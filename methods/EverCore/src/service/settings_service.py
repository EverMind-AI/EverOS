"""
Settings service

Provides business logic for global settings operations.
Handles both initialization (first PUT) and subsequent updates.
Includes LLM provider/model whitelist validation.
"""

import logging
import os
from typing import Any, Dict, Optional

from core.di import service
from core.di.utils import get_bean_by_type
from core.constants.exceptions import ValidationException
from infra_layer.adapters.out.persistence.repository.settings_raw_repository import (
    GlobalSettingsRawRepository,
)
from infra_layer.adapters.out.persistence.document.memory.global_settings import (
    GlobalSettings,
    LlmCustomSettingModel,
)
from api_specs.dtos.settings import SettingsResponse, UpdateSettingsRequest

logger = logging.getLogger(__name__)


@service("settings_service")
class SettingsService:
    """
    Settings service

    Provides:
    - Get global settings (singleton)
    - Update or initialize global settings (PUT semantics)
    - Internal helper methods for business layer (get_llm_custom_setting)
    - LLM provider/model whitelist validation
    """

    def __init__(self):
        self._repository: Optional[GlobalSettingsRawRepository] = None

    def _get_repository(self) -> GlobalSettingsRawRepository:
        """Get repository (lazy loading)"""
        if self._repository is None:
            self._repository = get_bean_by_type(GlobalSettingsRawRepository)
        return self._repository

    def _to_response(self, doc: GlobalSettings) -> SettingsResponse:
        """Convert GlobalSettings document to response DTO"""
        llm_setting_dict = None
        if doc.llm_custom_setting:
            llm_setting_dict = doc.llm_custom_setting.to_dict()

        return SettingsResponse(
            llm_custom_setting=llm_setting_dict,
            # Hidden fields: not yet implemented, uncomment when ready
            # timezone=doc.timezone,
            # boundary_detection_timeout=doc.boundary_detection_timeout,
            # extraction_mode=doc.extraction_mode,
            # offline_profile_extraction_interval=doc.offline_profile_extraction_interval,
            created_at=doc.created_at.isoformat() if doc.created_at else "",
            updated_at=doc.updated_at.isoformat() if doc.updated_at else "",
        )

    async def get(self) -> Optional[SettingsResponse]:
        """Get global settings

        Returns:
            SettingsResponse or None if not initialized
        """
        repo = self._get_repository()
        doc = await repo.get_global_settings()
        if not doc:
            return None
        return self._to_response(doc)

    async def update(
        self, request: UpdateSettingsRequest, raw_data: Optional[Dict[str, Any]] = None
    ) -> SettingsResponse:
        """Update or initialize global settings

        PUT semantics:
        - If settings don't exist: initialize
        - If settings exist: update provided fields
        - If a field is explicitly set to null in raw_data: clear it

        Args:
            request: Update settings request
            raw_data: Raw request JSON dict, used to distinguish
                      "field absent" from "field explicitly null"

        Returns:
            SettingsResponse with updated data

        Raises:
            ValueError: When validation fails
            ValidationException: When LLM whitelist validation fails
        """
        repo = self._get_repository()
        existing = await repo.get_global_settings()

        # Validate LLM whitelist
        if request.llm_custom_setting:
            self._validate_llm_custom_setting(request.llm_custom_setting)

        if existing is None:
            data = self._build_data(request, exclude_none=True)
            # Convert LlmCustomSetting DTO to LlmCustomSettingModel for storage
            if "llm_custom_setting" in data and data["llm_custom_setting"] is not None:
                data["llm_custom_setting"] = LlmCustomSettingModel.from_any(
                    data["llm_custom_setting"]
                )

            doc = await repo.upsert_global_settings(data)
            if not doc:
                raise ValueError("Failed to initialize settings")

            logger.info("Settings initialized")
            return self._to_response(doc)
        else:

            update_data = self._build_update_data(request, raw_data=raw_data)
            if not update_data:
                # No fields to update, return current state
                return self._to_response(existing)

            # Convert LlmCustomSetting DTO to LlmCustomSettingModel for storage
            if (
                "llm_custom_setting" in update_data
                and update_data["llm_custom_setting"] is not None
            ):
                update_data["llm_custom_setting"] = LlmCustomSettingModel.from_any(
                    update_data["llm_custom_setting"]
                )

            doc = await repo.update_global_settings(update_data)
            if not doc:
                raise ValueError("Failed to update settings")

            logger.info("Settings updated: fields=%s", list(update_data.keys()))
            return self._to_response(doc)

    # =========================================================================
    # Internal helper methods (for business layer)
    # =========================================================================

    async def get_llm_custom_setting(self) -> Optional[Dict[str, Any]]:
        """Get LLM custom setting as a dictionary

        Returns:
            LLM custom setting dict or None if not configured
        """
        repo = self._get_repository()
        doc = await repo.get_global_settings()
        if not doc or not doc.llm_custom_setting:
            return None
        if hasattr(doc.llm_custom_setting, "model_dump"):
            return doc.llm_custom_setting.model_dump()
        if hasattr(doc.llm_custom_setting, "dict"):
            return doc.llm_custom_setting.dict()
        return doc.llm_custom_setting

    # =========================================================================
    # Validation
    # =========================================================================

    @classmethod
    def _validate_llm_custom_setting(cls, llm_setting: Any) -> None:
        """
        Validate LLM custom setting provider/model against whitelist.

        Reads {PROVIDER}_WHITE_LIST env var (comma-separated model names).
        If the env var is not set or empty, no restriction is applied.

        Args:
            llm_setting: Object with boundary/extraction attributes, each having provider/model

        Raises:
            ValidationException: If a model is not in the provider's whitelist
        """
        if not llm_setting:
            return

        from memory_layer.constants import EXTRACT_SCENES

        for task_name in EXTRACT_SCENES:
            config = getattr(llm_setting, task_name, None)
            if config is None:
                continue
            provider = getattr(config, "provider", None)
            model = getattr(config, "model", None)
            if not provider or not model:
                continue
            cls._validate_model_whitelist(provider, model, task_name)

    @staticmethod
    def _validate_model_whitelist(provider: str, model: str, task_name: str) -> None:
        """
        Validate model against the provider's whitelist from environment variable.

        Reads {PROVIDER}_WHITE_LIST env var (comma-separated model names).
        If the env var is not set or empty, no restriction is applied.

        Args:
            provider: Provider name (e.g., "openai", "openrouter")
            model: Model name
            task_name: Task name for error context (e.g., "boundary", "extraction")

        Raises:
            ValidationException: If model is not in the whitelist
        """
        env_key = f"{provider.upper()}_WHITE_LIST"
        raw = os.getenv(env_key, "").strip()
        if not raw:
            return
        allowed_models = {m.strip() for m in raw.split(",") if m.strip()}
        if not allowed_models:
            return
        if model not in allowed_models:
            raise ValidationException(
                message=f"Model '{model}' is not allowed for provider '{provider}' "
                f"(task: {task_name}). "
                f"Allowed models: {', '.join(sorted(allowed_models))}.",
                field=f"llm_custom_setting.{task_name}.model",
                details={"error_code": "MODEL_NOT_IN_WHITELIST"},
            )

    # =========================================================================
    # Data building helpers
    # =========================================================================

    @staticmethod
    def _build_data(
        request: UpdateSettingsRequest, exclude_none: bool = True
    ) -> Dict[str, Any]:
        """Build data dict from request for initialization"""
        data = {}
        fields = [
            "llm_custom_setting",
            # Hidden fields: not yet implemented, uncomment when ready
            # "timezone",
            # "boundary_detection_timeout",
            # "extraction_mode",
            # "offline_profile_extraction_interval",
        ]
        for field in fields:
            value = getattr(request, field, None)
            if exclude_none and value is None:
                continue
            # Convert Pydantic model to dict for storage
            if hasattr(value, "model_dump"):
                value = value.model_dump(exclude_none=True)
            data[field] = value
        return data

    @staticmethod
    def _build_update_data(
        request: UpdateSettingsRequest, raw_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Build update data dict from request.

        Uses raw_data (the original JSON dict) to distinguish between
        "field not provided" (skip) and "field explicitly set to null" (clear).
        """
        data = {}
        fields = [
            "llm_custom_setting",
            # Hidden fields: not yet implemented, uncomment when ready
            # "timezone",
            # "boundary_detection_timeout",
            # "extraction_mode",
            # "offline_profile_extraction_interval",
        ]
        for field in fields:
            value = getattr(request, field, None)
            if value is None:
                # Check if the field was explicitly sent as null
                if (
                    raw_data is not None
                    and field in raw_data
                    and raw_data[field] is None
                ):
                    data[field] = None
                continue
            # Convert Pydantic model to dict for storage
            if hasattr(value, "model_dump"):
                value = value.model_dump(exclude_none=True)
            data[field] = value
        return data

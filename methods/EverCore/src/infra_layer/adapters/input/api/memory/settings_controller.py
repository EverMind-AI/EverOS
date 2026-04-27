"""
Settings Controller - Global settings management controller

Provides RESTful API routes for:
- Settings retrieval (GET /settings): get singleton global settings
- Settings update (PUT /settings): update or initialize global settings
"""

import logging

from fastapi import HTTPException, Request as FastAPIRequest

from core.di.decorators import controller
from core.interface.controller.base_controller import BaseController, get, put
from core.constants.exceptions import ValidationException
from api_specs.dtos.settings import (
    UpdateSettingsRequest,
    GetSettingsApiResponse,
    UpdateSettingsApiResponse,
)
from service.settings_service import SettingsService

logger = logging.getLogger(__name__)


@controller("settings_controller", primary=True)
class SettingsController(BaseController):
    """
    Settings Controller

    Handles global settings operations.
    Settings is a singleton per space (no ID in path).
    """

    def __init__(self, settings_service: SettingsService):
        """Initialize controller"""
        super().__init__(
            prefix="/api/v1/settings", tags=["Settings Controller"], default_auth="none"
        )
        self.settings_service = settings_service
        logger.info("SettingsController initialized")

    @get(
        "",
        response_model=GetSettingsApiResponse,
        summary="Get global settings",
        description="""
        Get the singleton global settings for this space.
        Returns 404 if settings have not been initialized yet.
        """,
        responses={
            404: {
                "description": "Settings not initialized",
                "content": {
                    "application/json": {
                        "example": {
                            "error": {
                                "code": "ResourceNotFound",
                                "message": "Settings not initialized",
                                "param": "",
                                "type": "NotFound",
                            }
                        }
                    }
                },
            }
        },
    )
    async def get_settings(self, request: FastAPIRequest) -> GetSettingsApiResponse:
        """Get global settings"""
        try:
            result = await self.settings_service.get()

            if not result:
                raise HTTPException(status_code=404, detail="Settings not initialized")

            return {"data": result.model_dump()}

        except HTTPException:
            raise
        except Exception as e:
            logger.error("Settings get failed: %s", e, exc_info=True)
            raise HTTPException(
                status_code=500, detail="Failed to retrieve settings"
            ) from e

    @put(
        "",
        response_model=UpdateSettingsApiResponse,
        summary="Update or initialize global settings",
        description="""
        Update the global settings, or initialize them if they don't exist yet.

        ## Initialization (first call):
        - All fields are optional and will use defaults

        ## Update (subsequent calls):
        - Only provided fields are updated (null fields are ignored)
        """,
        responses={
            422: {
                "description": "Request parameter error",
                "content": {
                    "application/json": {
                        "example": {
                            "error": {
                                "code": "InvalidParameter",
                                "message": "Invalid parameter value",
                                "param": "",
                                "type": "BadRequest",
                            }
                        }
                    }
                },
            }
        },
    )
    async def update_settings(
        self, request: FastAPIRequest, request_body: UpdateSettingsRequest = None
    ) -> UpdateSettingsApiResponse:
        """Update or initialize global settings"""
        del request_body  # Used for OpenAPI documentation only
        try:
            request_data = await request.json()
            update_request = UpdateSettingsRequest(**request_data)

            logger.info("Received settings update request")

            result = await self.settings_service.update(
                update_request, raw_data=request_data
            )
            return {"data": result.model_dump()}

        except (ValueError, ValidationException) as e:
            logger.error("Settings update parameter error: %s", e)
            raise HTTPException(status_code=422, detail=str(e)) from e
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Settings update failed: %s", e, exc_info=True)
            raise HTTPException(
                status_code=500, detail="Failed to update settings"
            ) from e

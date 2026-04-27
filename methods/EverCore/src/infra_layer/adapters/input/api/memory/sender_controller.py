"""
Sender Controller - Sender management controller

Provides RESTful API routes for:
- Sender creation (POST /senders): create or upsert a sender
- Sender retrieval (GET /senders/{sender_id}): get sender by sender_id
- Sender update (PATCH /senders/{sender_id}): partial update sender fields
"""

import logging

from fastapi import HTTPException, Request as FastAPIRequest

from core.di.decorators import controller
from core.interface.controller.base_controller import BaseController, get, post, patch
from api_specs.dtos.sender import (
    CreateSenderRequest,
    PatchSenderRequest,
    CreateSenderApiResponse,
    GetSenderApiResponse,
    PatchSenderApiResponse,
)
from service.sender_service import SenderService

logger = logging.getLogger(__name__)


@controller("sender_controller", primary=True)
class SenderController(BaseController):
    """
    Sender Controller

    Handles sender CRUD operations.
    """

    def __init__(self, sender_service: SenderService):
        """Initialize controller"""
        super().__init__(
            prefix="/api/v1/senders", tags=["Sender Controller"], default_auth="none"
        )
        self.sender_service = sender_service
        logger.info("SenderController initialized")

    @post(
        "",
        response_model=CreateSenderApiResponse,
        summary="Create or update a sender",
        description="""
        Create a new sender or update an existing one (upsert by sender_id).

        ## Fields:
        - **sender_id** (required): Sender identifier (unique)
        - **name** (optional): Sender display name
        """,
        responses={
            422: {
                "description": "Request parameter error",
                "content": {
                    "application/json": {
                        "example": {
                            "error": {
                                "code": "InvalidParameter",
                                "message": "sender_id is required",
                                "param": "sender_id",
                                "type": "BadRequest",
                            }
                        }
                    }
                },
            }
        },
    )
    async def create_sender(
        self, request: FastAPIRequest, request_body: CreateSenderRequest = None
    ) -> CreateSenderApiResponse:
        """Create or update a sender"""
        del request_body  # Used for OpenAPI documentation only
        try:
            request_data = await request.json()
            create_request = CreateSenderRequest(**request_data)

            logger.info(
                "Received sender create request: sender_id=%s", create_request.sender_id
            )

            result = await self.sender_service.create_or_update(
                sender_id=create_request.sender_id, name=create_request.name
            )

            if not result:
                raise HTTPException(status_code=500, detail="Failed to create sender")

            return {"data": result.model_dump()}

        except ValueError as e:
            logger.error("Sender create parameter error: %s", e)
            raise HTTPException(status_code=422, detail=str(e)) from e
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Sender create failed: %s", e, exc_info=True)
            raise HTTPException(
                status_code=500, detail="Failed to create sender"
            ) from e

    @get(
        "/{sender_id}",
        response_model=GetSenderApiResponse,
        summary="Get sender by sender_id",
        description="Retrieve a sender's details by its sender_id.",
        responses={
            404: {
                "description": "Sender not found",
                "content": {
                    "application/json": {
                        "example": {
                            "error": {
                                "code": "ResourceNotFound",
                                "message": "Sender not found: user_123",
                                "param": "sender_id",
                                "type": "NotFound",
                            }
                        }
                    }
                },
            }
        },
    )
    async def get_sender(
        self, request: FastAPIRequest, sender_id: str
    ) -> GetSenderApiResponse:
        """Get sender by sender_id"""
        try:
            logger.info("Received sender get request: sender_id=%s", sender_id)

            result = await self.sender_service.get_by_sender_id(sender_id)

            if not result:
                raise HTTPException(
                    status_code=404, detail=f"Sender not found: {sender_id}"
                )

            return {"data": result.model_dump()}

        except HTTPException:
            raise
        except Exception as e:
            logger.error("Sender get failed: %s", e, exc_info=True)
            raise HTTPException(
                status_code=500, detail="Failed to retrieve sender"
            ) from e

    @patch(
        "/{sender_id}",
        response_model=PatchSenderApiResponse,
        summary="Partially update sender",
        description="Update a sender's display name.",
        responses={
            404: {
                "description": "Sender not found",
                "content": {
                    "application/json": {
                        "example": {
                            "error": {
                                "code": "ResourceNotFound",
                                "message": "Sender not found: user_123",
                                "param": "sender_id",
                                "type": "NotFound",
                            }
                        }
                    }
                },
            }
        },
    )
    async def patch_sender(
        self,
        request: FastAPIRequest,
        sender_id: str,
        request_body: PatchSenderRequest = None,
    ) -> PatchSenderApiResponse:
        """Partially update sender fields"""
        del request_body  # Used for OpenAPI documentation only
        try:
            request_data = await request.json()
            patch_request = PatchSenderRequest(**request_data)

            logger.info("Received sender patch request: sender_id=%s", sender_id)

            result = await self.sender_service.patch(
                sender_id=sender_id, name=patch_request.name
            )

            if not result:
                raise HTTPException(
                    status_code=404, detail=f"Sender not found: {sender_id}"
                )

            return {"data": result.model_dump()}

        except ValueError as e:
            logger.error("Sender patch parameter error: %s", e)
            raise HTTPException(status_code=422, detail=str(e)) from e
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Sender patch failed: %s", e, exc_info=True)
            raise HTTPException(
                status_code=500, detail="Failed to update sender"
            ) from e

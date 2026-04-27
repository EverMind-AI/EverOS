"""
Group Controller - Group management controller

Provides RESTful API routes for:
- Group creation (POST /groups): create or upsert a group
- Group retrieval (GET /groups/{group_id}): get group by group_id
- Group update (PATCH /groups/{group_id}): partial update group fields
"""

import logging

from fastapi import HTTPException, Request as FastAPIRequest

from core.di.decorators import controller
from core.interface.controller.base_controller import BaseController, get, post, patch
from api_specs.dtos.group import (
    CreateGroupRequest,
    PatchGroupRequest,
    CreateGroupApiResponse,
    GetGroupApiResponse,
    PatchGroupApiResponse,
)
from service.group_service import GroupService

logger = logging.getLogger(__name__)


@controller("group_controller", primary=True)
class GroupController(BaseController):
    """
    Group Controller

    Handles group CRUD operations.
    """

    def __init__(self, group_service: GroupService):
        """Initialize controller"""
        super().__init__(
            prefix="/api/v1/groups", tags=["Group Controller"], default_auth="none"
        )
        self.group_service = group_service
        logger.info("GroupController initialized")

    @post(
        "",
        response_model=CreateGroupApiResponse,
        summary="Create or update a group",
        description="""
        Create a new group or update an existing one (upsert by group_id).

        ## Fields:
        - **group_id** (required): Group identifier (unique)
        - **name** (optional): Group display name
        - **description** (optional): Group description
        """,
        responses={
            422: {
                "description": "Request parameter error",
                "content": {
                    "application/json": {
                        "example": {
                            "error": {
                                "code": "InvalidParameter",
                                "message": "group_id is required",
                                "param": "group_id",
                                "type": "BadRequest",
                            }
                        }
                    }
                },
            }
        },
    )
    async def create_group(
        self, request: FastAPIRequest, request_body: CreateGroupRequest = None
    ) -> CreateGroupApiResponse:
        """Create or update a group"""
        del request_body  # Used for OpenAPI documentation only
        try:
            request_data = await request.json()
            create_request = CreateGroupRequest(**request_data)

            logger.info(
                "Received group create request: group_id=%s", create_request.group_id
            )

            result = await self.group_service.create_or_update(
                group_id=create_request.group_id,
                name=create_request.name,
                description=create_request.description,
            )

            if not result:
                raise HTTPException(status_code=500, detail="Failed to create group")

            return {"data": result.model_dump()}

        except ValueError as e:
            logger.error("Group create parameter error: %s", e)
            raise HTTPException(status_code=422, detail=str(e)) from e
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Group create failed: %s", e, exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to create group") from e

    @get(
        "/{group_id}",
        response_model=GetGroupApiResponse,
        summary="Get group by group_id",
        description="Retrieve a group's details by its group_id.",
        responses={
            404: {
                "description": "Group not found",
                "content": {
                    "application/json": {
                        "example": {
                            "error": {
                                "code": "ResourceNotFound",
                                "message": "Group not found: group_abc",
                                "param": "group_id",
                                "type": "NotFound",
                            }
                        }
                    }
                },
            }
        },
    )
    async def get_group(
        self, request: FastAPIRequest, group_id: str
    ) -> GetGroupApiResponse:
        """Get group by group_id"""
        try:
            logger.info("Received group get request: group_id=%s", group_id)

            result = await self.group_service.get_by_group_id(group_id)

            if not result:
                raise HTTPException(
                    status_code=404, detail=f"Group not found: {group_id}"
                )

            return {"data": result.model_dump()}

        except HTTPException:
            raise
        except Exception as e:
            logger.error("Group get failed: %s", e, exc_info=True)
            raise HTTPException(
                status_code=500, detail="Failed to retrieve group"
            ) from e

    @patch(
        "/{group_id}",
        response_model=PatchGroupApiResponse,
        summary="Partially update group",
        description="""
        Partially update a group's fields. At least one of name or description must be provided.
        """,
        responses={
            422: {
                "description": "Request parameter error",
                "content": {
                    "application/json": {
                        "example": {
                            "error": {
                                "code": "InvalidParameter",
                                "message": "At least one of 'name' or 'description' must be provided",
                                "param": "",
                                "type": "BadRequest",
                            }
                        }
                    }
                },
            },
            404: {
                "description": "Group not found",
                "content": {
                    "application/json": {
                        "example": {
                            "error": {
                                "code": "ResourceNotFound",
                                "message": "Group not found: group_abc",
                                "param": "group_id",
                                "type": "NotFound",
                            }
                        }
                    }
                },
            },
        },
    )
    async def patch_group(
        self,
        request: FastAPIRequest,
        group_id: str,
        request_body: PatchGroupRequest = None,
    ) -> PatchGroupApiResponse:
        """Partially update group fields"""
        del request_body  # Used for OpenAPI documentation only
        try:
            request_data = await request.json()
            patch_request = PatchGroupRequest(**request_data)

            logger.info("Received group patch request: group_id=%s", group_id)

            result = await self.group_service.patch(
                group_id=group_id,
                name=patch_request.name,
                description=patch_request.description,
            )

            if not result:
                raise HTTPException(
                    status_code=404, detail=f"Group not found: {group_id}"
                )

            return {"data": result.model_dump()}

        except ValueError as e:
            logger.error("Group patch parameter error: %s", e)
            raise HTTPException(status_code=422, detail=str(e)) from e
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Group patch failed: %s", e, exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to update group") from e

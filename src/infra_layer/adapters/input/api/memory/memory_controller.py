"""
Memory Controller - Unified memory management controller

Provides RESTful API routes for:
- Memory ingestion (POST /memories): accept a single-message payload and create memories
- Memory fetch (GET /memories): fetch by memory_type with optional user/group/time filters (query params or JSON body)
- Memory search (GET /memories/search): keyword/vector/hybrid/rrf/agentic retrieval with grouped results
- Conversation metadata (GET/POST/PATCH /conversation-meta): get with default fallback, upsert, and partial update
- Memory deletion (DELETE /memories): soft delete by combined filters
"""

import json
import logging
import time
from contextlib import suppress
from typing import Any, Dict
from fastapi import HTTPException, Request as FastAPIRequest

from core.di.decorators import controller
from core.di import get_bean_by_type
from core.interface.controller.base_controller import (
    BaseController,
    get,
    post,
    patch,
    delete,
)
from core.constants.errors import ErrorCode, ErrorStatus
from core.constants.exceptions import ValidationException
from agentic_layer.memory_manager import MemoryManager
from api_specs.request_converter import (
    convert_simple_message_to_memorize_request,
    convert_dict_to_fetch_mem_request,
    convert_dict_to_retrieve_mem_request,
)
from infra_layer.adapters.input.api.dto.memory_dto import (
    # Request DTOs
    MemorizeMessageRequest,
    FetchMemRequest,
    RetrieveMemRequest,
    ConversationMetaCreateRequest,
    ConversationMetaGetRequest,
    ConversationMetaPatchRequest,
    DeleteMemoriesRequestDTO,
    # Response DTOs
    MemorizeResponse,
    FetchMemoriesResponse,
    SearchMemoriesResponse,
    GetConversationMetaResponse,
    SaveConversationMetaResponse,
    PatchConversationMetaResponse,
    DeleteMemoriesResponse,
    RestoreMemoriesResponse,
)
from core.request.timeout_background import timeout_to_background
from core.request import log_request
from core.component.redis_provider import RedisProvider
from service.memory_request_log_service import MemoryRequestLogService
from service.memcell_delete_service import MemCellDeleteService
from service.memcell_restore_service import MemCellRestoreService
from service.conversation_meta_service import ConversationMetaService
from api_specs.memory_types import RawDataType
from agentic_layer.metrics.memorize_metrics import (
    record_memorize_request,
    record_memorize_error,
    record_memorize_message,
    classify_memorize_error,
    get_space_id_for_metrics,
    get_raw_data_type_label,
)

logger = logging.getLogger(__name__)


@controller("memory_controller", primary=True)
class MemoryController(BaseController):
    """
    Memory Controller
    """

    def __init__(self, conversation_meta_service: ConversationMetaService):
        """Initialize controller"""
        super().__init__(
            prefix="/api/v1/memories",
            tags=["Memory Controller"],
            default_auth="none",  # Adjust authentication strategy based on actual needs
        )
        self.memory_manager = MemoryManager()
        self.conversation_meta_service = conversation_meta_service
        # Get RedisProvider
        self.redis_provider = get_bean_by_type(RedisProvider)
        logger.info(
            "MemoryController initialized with MemoryManager and ConversationMetaService"
        )

    @staticmethod
    async def _collect_request_params(
        fastapi_request: FastAPIRequest,
    ) -> Dict[str, Any]:
        """Merge query parameters with optional JSON body parameters."""
        params: Dict[str, Any] = dict(fastapi_request.query_params)
        if body := await fastapi_request.body():
            with suppress(json.JSONDecodeError, TypeError):
                if isinstance(body_data := json.loads(body), dict):
                    params.update(body_data)
        return params

    @post(
        "",
        response_model=MemorizeResponse,
        summary="Store single message",
        description="""
        Store a single message into memory.
        
        ## Fields:
        - **message_id** (required): Unique identifier for the message
        - **create_time** (required): Message creation time (ISO 8601 format)
        - **sender** (required): Sender user ID
        - **content** (required): Message content
        - **group_id** (optional): Group ID
        - **group_name** (optional): Group name
        - **sender_name** (optional): Sender display name (defaults to sender if empty)
        - **role** (optional): Sender role ("user" or "assistant")
        - **refer_list** (optional): List of referenced message IDs
        
        ## Functionality:
        - Accepts raw single-message data
        - Automatically creates memories when sufficient context is available
        - Returns extraction count and status
        """,
        responses={
            400: {
                "description": "Request parameter error",
                "content": {
                    "application/json": {
                        "example": {
                            "status": ErrorStatus.FAILED.value,
                            "code": ErrorCode.INVALID_PARAMETER.value,
                            "message": "Data format error: Required field message_id is missing",
                            "timestamp": "2025-01-15T10:30:00+00:00",
                            "path": "/api/v1/memories",
                        }
                    }
                },
            },
            500: {
                "description": "Internal server error",
                "content": {
                    "application/json": {
                        "example": {
                            "status": ErrorStatus.FAILED.value,
                            "code": ErrorCode.SYSTEM_ERROR.value,
                            "message": "Failed to store memory, please try again later",
                            "timestamp": "2025-01-15T10:30:00+00:00",
                            "path": "/api/v1/memories",
                        }
                    }
                },
            },
        },
    )
    @log_request()
    @timeout_to_background()
    async def memorize_single_message(
        self,
        request: FastAPIRequest,
        request_body: MemorizeMessageRequest = None,  # OpenAPI documentation only
    ) -> MemorizeResponse:
        """
        Store single message memory data

        Convert a single-message payload to a memory request and persist it.
        If no memory is extracted, the message remains pending for later processing.

        Args:
            request: FastAPI request object
            request_body: Message request body (used for OpenAPI documentation only)

        Returns:
            Dict[str, Any]: Memory storage response with extraction count and status info

        Raises:
            HTTPException: When request processing fails
        """
        del request_body  # Used for OpenAPI documentation only
        start_time = time.perf_counter()
        memory_count = 0
        # Get space_id for metrics (available from tenant context)
        space_id = get_space_id_for_metrics()
        # Default raw_data_type, will be updated after conversion
        raw_data_type = get_raw_data_type_label(None)

        try:
            # 1. Get JSON body from request (simple direct format)
            message_data = await request.json()
            logger.info("Received memorize request (single message)")

            # 2. Convert directly to MemorizeRequest (unified single-step conversion)
            logger.info(
                "Starting conversion from simple message format to MemorizeRequest"
            )
            memorize_request = await convert_simple_message_to_memorize_request(
                message_data
            )

            # Update raw_data_type from request (for subsequent metrics)
            if memorize_request.raw_data_type:
                raw_data_type = get_raw_data_type_label(
                    memorize_request.raw_data_type.value
                )

            record_memorize_message(
                space_id=space_id,
                raw_data_type=raw_data_type,
                status='received',
                count=1,
            )

            # Extract metadata for logging
            group_name = memorize_request.group_name
            group_id = memorize_request.group_id

            logger.info(
                "Conversion completed: group_id=%s, group_name=%s", group_id, group_name
            )

            # 3. Save request logs first (sync_status=-1) for better timing control
            if (
                memorize_request.raw_data_type == RawDataType.CONVERSATION
                and memorize_request.new_raw_data_list
            ):
                log_service = get_bean_by_type(MemoryRequestLogService)
                await log_service.save_request_logs(
                    request=memorize_request,
                    version="1.0.0",
                    endpoint_name="memorize_single_message",
                    method=request.method,
                    url=str(request.url),
                    raw_input_dict=message_data,
                )
                logger.info(
                    "Saved %d request logs: group_id=%s",
                    len(memorize_request.new_raw_data_list),
                    group_id,
                )

            # 4. Call memory_manager to process the request
            logger.info("Starting to process memory request")
            # memorize returns count of extracted memories (int)
            memory_count = await self.memory_manager.memorize(memorize_request)

            # 5. Return unified response format
            logger.info(
                "Memory request processing completed, extracted %s memories",
                memory_count,
            )

            # Optimize return message to help users understand runtime status
            if memory_count > 0:
                message = f"Extracted {memory_count} memories"
                status = 'extracted'
            else:
                message = "Message queued, awaiting boundary detection"
                status = 'accumulated'

            # Record success metrics
            record_memorize_request(
                space_id=space_id,
                raw_data_type=raw_data_type,
                status=status,
                duration_seconds=time.perf_counter() - start_time,
            )

            return {
                "status": ErrorStatus.OK.value,
                "message": message,
                "result": {
                    "saved_memories": [],  # Memories saved to DB, fetch via API
                    "count": memory_count,
                    "status_info": "accumulated" if memory_count == 0 else "extracted",
                },
            }

        except ValueError as e:
            logger.error("memorize request parameter error: %s", e)
            record_memorize_error(
                space_id=space_id,
                raw_data_type=raw_data_type,
                stage='conversion',
                error_type='validation_error',
            )
            record_memorize_request(
                space_id=space_id,
                raw_data_type=raw_data_type,
                status='error',
                duration_seconds=time.perf_counter() - start_time,
            )
            raise HTTPException(status_code=400, detail=str(e)) from e
        except HTTPException:
            # Re-raise HTTPException (already handled errors)
            record_memorize_request(
                space_id=space_id,
                raw_data_type=raw_data_type,
                status='error',
                duration_seconds=time.perf_counter() - start_time,
            )
            raise
        except Exception as e:
            logger.error("memorize request processing failed: %s", e, exc_info=True)
            error_type = classify_memorize_error(e)
            record_memorize_error(
                space_id=space_id,
                raw_data_type=raw_data_type,
                stage='memorize_process',
                error_type=error_type,
            )
            record_memorize_request(
                space_id=space_id,
                raw_data_type=raw_data_type,
                status='error',
                duration_seconds=time.perf_counter() - start_time,
            )
            raise HTTPException(
                status_code=500, detail="Failed to store memory, please try again later"
            ) from e

    @get(
        "",
        response_model=FetchMemoriesResponse,
        summary="Fetch user memories",
        description="""
        Retrieve memory records by memory_type with optional filters
        
        ## Fields:
        - **user_id** (required): User ID
        - **memory_type** (optional): Memory type (default: episodic_memory)
            - profile: user profile
            - episodic_memory: episodic memory
            - foresight: prospective memory
            - event_log: event log (atomic facts)
        - **limit** (optional): Max records to return (default: 10, max: 100)
        - **offset** (optional): Pagination offset (default: 0)
        - **version_range** (optional): Version range filter [start, end]
        
        ## Use cases:
        - User profile display
        - Personalized recommendation systems
        - Conversation history review
        """,
        responses={
            400: {
                "description": "Request parameter error",
                "content": {
                    "application/json": {
                        "example": {
                            "status": ErrorStatus.FAILED.value,
                            "code": ErrorCode.INVALID_PARAMETER.value,
                            "message": "user_id cannot be empty",
                            "timestamp": "2024-01-15T10:30:00+00:00",
                            "path": "/api/v1/memories",
                        }
                    }
                },
            },
            500: {
                "description": "Internal server error",
                "content": {
                    "application/json": {
                        "example": {
                            "status": ErrorStatus.FAILED.value,
                            "code": ErrorCode.SYSTEM_ERROR.value,
                            "message": "Failed to retrieve memory, please try again later",
                            "timestamp": "2024-01-15T10:30:00+00:00",
                            "path": "/api/v1/memories",
                        }
                    }
                },
            },
        },
    )
    async def fetch_memories(
        self,
        fastapi_request: FastAPIRequest,
        request_body: FetchMemRequest = None,  # For OpenAPI request body documentation
    ) -> FetchMemoriesResponse:
        """
        Retrieve user memory data

        Fetch memory records by memory_type with optional user/group/time filters.
        Parameters are accepted from query params or request body (GET with body is supported).

        Args:
            fastapi_request: FastAPI request object
            request_body: Request body parameters (used for OpenAPI documentation only)

        Returns:
            Dict[str, Any]: Memory retrieval response

        Raises:
            HTTPException: When request processing fails
        """
        del request_body  # Used for OpenAPI documentation only
        try:
            params = await self._collect_request_params(fastapi_request)

            logger.info(
                "Received fetch request: user_id=%s, memory_type=%s",
                params.get("user_id"),
                params.get("memory_type"),
            )

            # Directly use converter to transform
            fetch_request = convert_dict_to_fetch_mem_request(params)

            # Call memory_manager's fetch_mem method
            response = await self.memory_manager.fetch_mem(fetch_request)

            # Return unified response format
            memory_count = len(response.memories) if response.memories else 0
            logger.info(
                "Fetch request processing completed: user_id=%s, returned %s memories",
                params.get("user_id"),
                memory_count,
            )
            return {
                "status": ErrorStatus.OK.value,
                "message": f"Memory retrieval successful, retrieved {memory_count} memories",
                "result": response,
            }

        except ValueError as e:
            logger.error("Fetch request parameter error: %s", e)
            raise HTTPException(status_code=400, detail=str(e)) from e
        except HTTPException:
            # Re-raise HTTPException
            raise
        except Exception as e:
            logger.error("Fetch request processing failed: %s", e, exc_info=True)
            raise HTTPException(
                status_code=500,
                detail="Failed to retrieve memory, please try again later",
            ) from e

    @get(
        "/search",
        response_model=SearchMemoriesResponse,
        summary="Search relevant memories (keyword/vector/hybrid/rrf/agentic)",
        description="""
        Retrieve relevant memory data based on query text using multiple retrieval methods
        
        ## Fields:
        - **query** (optional): Search query text
        - **user_id** (optional): User ID (at least one of user_id/group_id required)
        - **group_id** (optional): Group ID (at least one of user_id/group_id required)
        - **retrieve_method** (optional): Retrieval method (default: keyword)
            - keyword: keyword retrieval (BM25)
            - vector: vector semantic retrieval
            - hybrid: hybrid retrieval (keyword + vector)
            - rrf: RRF fusion retrieval
            - agentic: LLM-guided multi-round retrieval
        - **radius** (optional): Similarity threshold (0.0-1.0) for vector search
        - **memory_types** (optional): List of memory types to search
            - episodic_memory
            - foresight
            - event_log
        - **start_time** (optional): Start time (ISO 8601)
        - **end_time** (optional): End time (ISO 8601)
        - **top_k** (optional): Max results (default: 10, max: 100)
        - **include_metadata** (optional): Whether to include metadata (default: true)
        
        ## Result description:
        - Memories returned organized by group
        - Each group contains multiple relevant memories sorted by time
        - Groups sorted by importance score, most important group first
        - Each memory has a relevance score indicating match degree with query
        """,
        responses={
            400: {
                "description": "Request parameter error",
                "content": {
                    "application/json": {
                        "example": {
                            "status": ErrorStatus.FAILED.value,
                            "code": ErrorCode.INVALID_PARAMETER.value,
                            "message": "query cannot be empty",
                            "timestamp": "2024-01-15T10:30:00+00:00",
                            "path": "/api/v1/memories/search",
                        }
                    }
                },
            },
            500: {
                "description": "Internal server error",
                "content": {
                    "application/json": {
                        "example": {
                            "status": ErrorStatus.FAILED.value,
                            "code": ErrorCode.SYSTEM_ERROR.value,
                            "message": "Failed to retrieve memory, please try again later",
                            "timestamp": "2024-01-15T10:30:00+00:00",
                            "path": "/api/v1/memories/search",
                        }
                    }
                },
            },
        },
    )
    async def search_memories(
        self,
        fastapi_request: FastAPIRequest,
        request_body: RetrieveMemRequest = None,  # For OpenAPI request body documentation
    ) -> SearchMemoriesResponse:
        """
        Search relevant memory data

        Retrieve relevant memory data based on query text using keyword, vector, hybrid, RRF, or agentic methods.
        Parameters are passed via request body (GET with body, similar to Elasticsearch style).

        Args:
            fastapi_request: FastAPI request object
            request_body: Request body parameters (used for OpenAPI documentation only)

        Returns:
            Dict[str, Any]: Memory search response

        Raises:
            HTTPException: When request processing fails
        """
        del request_body  # Used for OpenAPI documentation only
        try:
            query_params = await self._collect_request_params(fastapi_request)

            query_text = query_params.get("query")
            logger.info(
                "Received search request: user_id=%s, query=%s, retrieve_method=%s",
                query_params.get("user_id"),
                query_text,
                query_params.get("retrieve_method"),
            )

            # Directly use converter to transform
            retrieve_request = convert_dict_to_retrieve_mem_request(
                query_params, query=query_text
            )
            logger.info(
                "After conversion: retrieve_method=%s", retrieve_request.retrieve_method
            )

            # Use retrieve_mem method (supports keyword, vector, hybrid, rrf, agentic)
            response = await self.memory_manager.retrieve_mem(retrieve_request)

            # Return unified response format
            group_count = len(response.memories) if response.memories else 0
            logger.info(
                "Search request complete: user_id=%s, returned %s groups",
                query_params.get("user_id"),
                group_count,
            )
            return {
                "status": ErrorStatus.OK.value,
                "message": f"Memory search successful, retrieved {group_count} groups",
                "result": response,
            }

        except ValueError as e:
            logger.error("Search request parameter error: %s", e)
            raise HTTPException(status_code=400, detail=str(e)) from e
        except HTTPException:
            # Re-raise HTTPException
            raise
        except Exception as e:
            logger.error("Search request processing failed: %s", e, exc_info=True)
            raise HTTPException(
                status_code=500,
                detail="Failed to retrieve memory, please try again later",
            ) from e

    @get(
        "/conversation-meta",
        response_model=GetConversationMetaResponse,
        summary="Get conversation metadata",
        description="""
        Retrieve conversation metadata by group_id with fallback to default config
        
        ## Functionality:
        - Query by group_id to get conversation metadata
        - If group_id not found, fallback to default config
        - If group_id not provided, returns default config
        
        ## Fallback Logic:
        - Try exact group_id first, then use default config
        
        ## Use Cases:
        - Get specific group's metadata
        - Get default settings (group_id not provided or null)
        - Auto-fallback to defaults when group config not set
        """,
        responses={
            404: {
                "description": "Conversation metadata not found",
                "content": {
                    "application/json": {
                        "example": {
                            "status": "failed",
                            "message": "Conversation metadata not found for group_id: group_123",
                        }
                    }
                },
            }
        },
    )
    async def get_conversation_meta(
        self,
        fastapi_request: FastAPIRequest,
        request_body: ConversationMetaGetRequest = None,  # OpenAPI documentation only
    ) -> GetConversationMetaResponse:
        """
        Get conversation metadata by group_id with fallback support

        Args:
            fastapi_request: FastAPI request object
            request_body: Get request parameters (used for OpenAPI documentation only)

        Returns:
            Dict[str, Any]: Conversation metadata response

        Raises:
            HTTPException: When request processing fails
        """
        del request_body  # Used for OpenAPI documentation only
        try:
            params = await self._collect_request_params(fastapi_request)

            group_id = params.get("group_id")

            logger.info("Received conversation-meta get request: group_id=%s", group_id)

            # Query via service (fallback to default is handled internally)
            result = await self.conversation_meta_service.get_by_group_id(group_id)

            if not result:
                raise HTTPException(
                    status_code=404,
                    detail=f"Conversation metadata not found for group_id: {group_id}",
                )

            message = (
                "Using default config"
                if result.is_default and group_id
                else "Conversation metadata retrieved successfully"
            )

            return {
                "status": ErrorStatus.OK.value,
                "message": message,
                "result": result.model_dump(),
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(
                "conversation-meta get request processing failed: %s", e, exc_info=True
            )
            raise HTTPException(
                status_code=500,
                detail="Failed to retrieve conversation metadata, please try again later",
            ) from e

    @post(
        "/conversation-meta",
        response_model=SaveConversationMetaResponse,
        summary="Save conversation metadata",
        description="""
        Save conversation metadata information, including scene, participants, tags, etc.
        
        ## Functionality:
        - If group_id exists, update the entire record (upsert)
        - If group_id does not exist, create a new record
        - If group_id is omitted, save as default config for the scene
        - All fields must be provided with complete data
        
        ## Default Config:
        - Default config is used as fallback when specific group_id config not found
        
        ## Notes:
        - This is a full update interface that will replace the entire record
        - If you only need to update partial fields, use the PATCH /conversation-meta interface
        """,
        responses={
            400: {
                "description": "Request parameter error",
                "content": {
                    "application/json": {
                        "example": {
                            "status": ErrorStatus.FAILED.value,
                            "code": ErrorCode.INVALID_PARAMETER.value,
                            "message": "Field 'scene': invalid scene value",
                            "timestamp": "2025-01-15T10:30:00+00:00",
                            "path": "/api/v1/memories/conversation-meta",
                        }
                    }
                },
            },
            500: {
                "description": "Internal server error",
                "content": {
                    "application/json": {
                        "example": {
                            "status": ErrorStatus.FAILED.value,
                            "code": ErrorCode.SYSTEM_ERROR.value,
                            "message": "Failed to save conversation metadata, please try again later",
                            "timestamp": "2025-01-15T10:30:00+00:00",
                            "path": "/api/v1/memories/conversation-meta",
                        }
                    }
                },
            },
        },
    )
    async def save_conversation_meta(
        self,
        fastapi_request: FastAPIRequest,
        request_body: ConversationMetaCreateRequest = None,  # OpenAPI documentation only
    ) -> SaveConversationMetaResponse:
        """
        Save conversation metadata

        Save conversation metadata to MongoDB via service

        Args:
            fastapi_request: FastAPI request object
            request_body: Conversation metadata request body (used for OpenAPI documentation only)

        Returns:
            Dict[str, Any]: Save response, containing saved metadata information

        Raises:
            HTTPException: When request processing fails
        """
        del request_body  # Used for OpenAPI documentation only
        try:
            # 1. Parse request body into DTO
            request_data = await fastapi_request.json()
            create_request = ConversationMetaCreateRequest(**request_data)

            logger.info(
                "Received conversation-meta save request: group_id=%s",
                create_request.group_id,
            )

            # 2. Save via service
            result = await self.conversation_meta_service.save(create_request)

            if not result:
                raise HTTPException(
                    status_code=500, detail="Failed to save conversation metadata"
                )

            # 3. Return success response
            return {
                "status": ErrorStatus.OK.value,
                "message": "Conversation metadata saved successfully",
                "result": result.model_dump(),
            }

        except ValidationException as e:
            logger.error(
                "conversation-meta validation failed: %s", e.message, exc_info=True
            )
            raise HTTPException(status_code=400, detail=e.message) from e
        except ValueError as e:
            logger.error("conversation-meta request parameter error: %s", e)
            raise HTTPException(status_code=400, detail=str(e)) from e
        except HTTPException:
            raise
        except Exception as e:
            logger.error(
                "conversation-meta request processing failed: %s", e, exc_info=True
            )
            raise HTTPException(
                status_code=500,
                detail="Failed to save conversation metadata, please try again later",
            ) from e

    @patch(
        "/conversation-meta",
        response_model=PatchConversationMetaResponse,
        summary="Partially update conversation metadata",
        description="""
        Partially update conversation metadata, only updating provided fields
        
        ## Functionality:
        - Locate the conversation metadata to update by group_id
        - When group_id is null or not provided, updates the default config
        - Only update fields provided in the request, keep unchanged fields as-is
        - Suitable for scenarios requiring modification of partial information
        
        ## Fields that can be updated:
        - **name**: Conversation name
        - **description**: Conversation description
        - **scene_desc**: Scene description
        - **tags**: Tag list
        - **user_details**: User details (will completely replace existing user_details)
        - **default_timezone**: Default timezone
        
        ## Notes:
        - group_id can be a specific value or omitted (for default config)
        - If user_details field is provided, it will completely replace existing user details
        - Not allowed to modify core fields such as version, scene, group_id, conversation_created_at
        """,
        responses={
            400: {
                "description": "Request parameter error",
                "content": {
                    "application/json": {
                        "example": {
                            "status": ErrorStatus.FAILED.value,
                            "code": ErrorCode.INVALID_PARAMETER.value,
                            "message": "Missing required field group_id",
                            "timestamp": "2025-01-15T10:30:00+00:00",
                            "path": "/api/v1/memories/conversation-meta",
                        }
                    }
                },
            },
            404: {
                "description": "Conversation metadata not found",
                "content": {
                    "application/json": {
                        "example": {
                            "status": ErrorStatus.FAILED.value,
                            "code": ErrorCode.RESOURCE_NOT_FOUND.value,
                            "message": "Specified conversation metadata not found: group_123",
                            "timestamp": "2025-01-15T10:30:00+00:00",
                            "path": "/api/v1/memories/conversation-meta",
                        }
                    }
                },
            },
            500: {
                "description": "Internal server error",
                "content": {
                    "application/json": {
                        "example": {
                            "status": ErrorStatus.FAILED.value,
                            "code": ErrorCode.SYSTEM_ERROR.value,
                            "message": "Failed to update conversation metadata, please try again later",
                            "timestamp": "2025-01-15T10:30:00+00:00",
                            "path": "/api/v1/memories/conversation-meta",
                        }
                    }
                },
            },
        },
    )
    async def patch_conversation_meta(
        self,
        fastapi_request: FastAPIRequest,
        request_body: ConversationMetaPatchRequest = None,  # OpenAPI documentation only
    ) -> PatchConversationMetaResponse:
        """
        Partially update conversation metadata

        Locate record by group_id, only update fields provided in the request

        Args:
            fastapi_request: FastAPI request object
            request_body: Patch request body (used for OpenAPI documentation only)

        Returns:
            Dict[str, Any]: Update response, containing updated metadata information

        Raises:
            HTTPException: When request processing fails
        """
        del request_body  # Used for OpenAPI documentation only
        try:
            # 1. Parse request body into DTO
            request_data = await fastapi_request.json()
            patch_request = ConversationMetaPatchRequest(**request_data)

            logger.info(
                "Received conversation-meta partial update request: group_id=%s",
                patch_request.group_id,
            )

            # 2. Patch via service
            result, updated_fields = await self.conversation_meta_service.patch(
                patch_request
            )

            if result is None:
                detail_msg = (
                    f"Specified conversation metadata not found: group_id={patch_request.group_id}"
                    if patch_request.group_id
                    else "Default config not found"
                )
                raise HTTPException(status_code=404, detail=detail_msg)

            # 3. Return success response
            if not updated_fields:
                return {
                    "status": ErrorStatus.OK.value,
                    "message": "No fields need updating",
                    "result": {
                        "id": result.id,
                        "group_id": result.group_id,
                        "updated_fields": [],
                    },
                }

            return {
                "status": ErrorStatus.OK.value,
                "message": f"Conversation metadata updated successfully, updated {len(updated_fields)} fields",
                "result": {
                    "id": result.id,
                    "group_id": result.group_id,
                    "scene": result.scene,
                    "name": result.name,
                    "updated_fields": updated_fields,
                    "updated_at": result.updated_at,
                },
            }

        except ValidationException as e:
            logger.error(
                "conversation-meta partial update validation failed: %s",
                e.message,
                exc_info=True,
            )
            raise HTTPException(status_code=400, detail=e.message) from e
        except HTTPException:
            # Re-raise HTTPException
            raise
        except KeyError as e:
            logger.error(
                "conversation-meta partial update request missing required field: %s", e
            )
            raise HTTPException(
                status_code=400, detail=f"Missing required field: {str(e)}"
            ) from e
        except ValueError as e:
            logger.error(
                "conversation-meta partial update request parameter error: %s", e
            )
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            logger.error(
                "conversation-meta partial update request processing failed: %s",
                e,
                exc_info=True,
            )
            raise HTTPException(
                status_code=500,
                detail="Failed to update conversation metadata, please try again later",
            ) from e

    @delete(
        "",
        response_model=DeleteMemoriesResponse,
        summary="Delete memories (soft delete)",
        description="""
        Soft delete memory records based on combined filter criteria
        
        ## Functionality:
        - Soft delete records matching combined filter conditions
        - If multiple conditions provided, ALL must be satisfied (AND logic)
        - At least one filter must be specified
        
        ## Filter Parameters (combined with AND):
        - **event_id**: Filter by specific event_id
        - **user_id**: Filter by user ID
        - **group_id**: Filter by group ID
        
        ## Examples:
        - event_id only: Delete specific memory
        - user_id only: Delete all user's memories
        - user_id + group_id: Delete user's memories in specific group
        - event_id + user_id + group_id: Delete if all conditions match
        
        ## Soft Delete:
        - Records are marked as deleted, not physically removed
        - Deleted records can be restored if needed
        - Deleted records won't appear in regular queries
        
        ## Use cases:
        - User requests data deletion
        - Group chat cleanup
        - Privacy compliance (GDPR, etc.)
        - Conversation history management
        """,
        responses={
            400: {
                "description": "Request parameter error",
                "content": {
                    "application/json": {
                        "example": {
                            "status": ErrorStatus.FAILED.value,
                            "code": ErrorCode.INVALID_PARAMETER.value,
                            "message": "At least one of event_id, user_id, or group_id must be provided",
                            "timestamp": "2025-01-15T10:30:00+00:00",
                            "path": "/api/v1/memories",
                        }
                    }
                },
            },
            404: {
                "description": "Memory not found",
                "content": {
                    "application/json": {
                        "example": {
                            "status": ErrorStatus.FAILED.value,
                            "code": ErrorCode.RESOURCE_NOT_FOUND.value,
                            "message": "No memories found matching the criteria or already deleted",
                            "timestamp": "2025-01-15T10:30:00+00:00",
                            "path": "/api/v1/memories",
                        }
                    }
                },
            },
            500: {
                "description": "Internal server error",
                "content": {
                    "application/json": {
                        "example": {
                            "status": ErrorStatus.FAILED.value,
                            "code": ErrorCode.SYSTEM_ERROR.value,
                            "message": "Failed to delete memories, please try again later",
                            "timestamp": "2025-01-15T10:30:00+00:00",
                            "path": "/api/v1/memories",
                        }
                    }
                },
            },
        },
    )
    async def delete_memories(
        self,
        fastapi_request: FastAPIRequest,
        request_body: DeleteMemoriesRequestDTO = None,  # OpenAPI documentation (body params)
    ) -> DeleteMemoriesResponse:
        """
        Soft delete memory data based on combined filter criteria

        Filters are combined with AND logic. Omit any filter you do not want to apply.

        Args:
            fastapi_request: FastAPI request object
            request_body: Request body parameters (used for OpenAPI documentation only)

        Returns:
            Dict[str, Any]: Delete result response

        Raises:
            HTTPException: When request processing fails
        """
        del request_body  # Used for OpenAPI documentation only

        try:
            from core.oxm.constants import MAGIC_ALL

            params = await self._collect_request_params(fastapi_request)

            # Extract and validate parameters using DeleteMemoriesRequestDTO
            try:
                delete_request = DeleteMemoriesRequestDTO(
                    event_id=params.get("event_id", MAGIC_ALL),
                    user_id=params.get("user_id", MAGIC_ALL),
                    group_id=params.get("group_id", MAGIC_ALL),
                )
            except ValueError as e:
                logger.error("Delete request validation failed: %s", e)
                raise HTTPException(status_code=400, detail=str(e)) from e

            logger.info(
                "Received delete request: event_id=%s, user_id=%s, group_id=%s",
                delete_request.event_id,
                delete_request.user_id,
                delete_request.group_id,
            )

            # Get delete service
            delete_service = get_bean_by_type(MemCellDeleteService)

            # Execute delete operation (combined filters)
            result = await delete_service.delete_by_combined_criteria(
                event_id=delete_request.event_id,
                user_id=delete_request.user_id,
                group_id=delete_request.group_id,
            )

            # Check if deletion was successful
            if not result["success"]:
                error_msg = result.get(
                    "error",
                    "No memories found matching the criteria or already deleted",
                )
                logger.warning("Delete operation returned no results: %s", result)
                raise HTTPException(status_code=404, detail=error_msg)

            # Log successful deletion
            logger.info(
                "Delete request completed successfully: filters=%s, count=%d",
                result["filters"],
                result["count"],
            )

            # Return success response
            return {
                "status": ErrorStatus.OK.value,
                "message": f"Successfully deleted {result['count']} {'memory' if result['count'] == 1 else 'memories'}",
                "result": {"filters": result["filters"], "count": result["count"]},
            }

        except HTTPException:
            # Re-raise HTTPException
            raise
        except Exception as e:
            logger.error("Delete request processing failed: %s", e, exc_info=True)
            raise HTTPException(
                status_code=500,
                detail="Failed to delete memories, please try again later",
            ) from e

    @post(
        "/restore",
        response_model=RestoreMemoriesResponse,
        summary="Restore soft-deleted memories",
        description="""
        Restore soft-deleted memory records based on filter criteria
        
        ## Functionality:
        - Restore previously soft-deleted memories
        - Supports restore by event_id (single memory) or user_id (batch)
        - At least one filter must be specified
        
        ## Filter Parameters (provide one):
        - **event_id**: Restore a specific memory by its event_id
        - **user_id**: Restore all soft-deleted memories of a user
        
        ## Use Cases:
        - Undo accidental deletion
        - Recover user data
        - Data restoration after testing
        """,
        responses={
            400: {
                "description": "Request parameter error",
                "content": {
                    "application/json": {
                        "example": {
                            "status": ErrorStatus.FAILED.value,
                            "code": ErrorCode.INVALID_PARAMETER.value,
                            "message": "At least one of event_id or user_id must be provided",
                        }
                    }
                },
            },
            500: {
                "description": "Internal server error",
                "content": {
                    "application/json": {
                        "example": {
                            "status": ErrorStatus.FAILED.value,
                            "code": ErrorCode.SYSTEM_ERROR.value,
                            "message": "Failed to restore memories, please try again later",
                        }
                    }
                },
            },
        },
    )
    async def restore_memories(
        self,
        fastapi_request: FastAPIRequest,
        request_body=None,
    ) -> RestoreMemoriesResponse:
        """
        Restore soft-deleted memory data based on filter criteria
        """
        del request_body

        try:
            from core.oxm.constants import MAGIC_ALL

            params = await self._collect_request_params(fastapi_request)

            event_id = params.get("event_id", MAGIC_ALL)
            user_id = params.get("user_id", MAGIC_ALL)

            # Validate: at least one filter required
            if (not event_id or event_id == MAGIC_ALL) and (
                not user_id or user_id == MAGIC_ALL
            ):
                raise HTTPException(
                    status_code=400,
                    detail="At least one of event_id or user_id must be provided",
                )

            logger.info(
                "Received restore request: event_id=%s, user_id=%s",
                event_id,
                user_id,
            )

            # Get restore service
            restore_service = get_bean_by_type(MemCellRestoreService)

            # Execute restore
            result = await restore_service.restore_by_combined_criteria(
                event_id=event_id,
                user_id=user_id,
            )

            if not result["success"]:
                error_msg = result.get(
                    "error", "No soft-deleted memories found matching the criteria"
                )
                logger.warning("Restore operation returned no results: %s", result)
                raise HTTPException(status_code=404, detail=error_msg)

            logger.info(
                "Restore request completed successfully: filters=%s, count=%d",
                result["filters"],
                result["count"],
            )

            return {
                "status": ErrorStatus.OK.value,
                "message": f"Successfully restored {result['count']} {'memory' if result['count'] == 1 else 'memories'}",
                "result": {"filters": result["filters"], "count": result["count"]},
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error("Restore request processing failed: %s", e, exc_info=True)
            raise HTTPException(
                status_code=500,
                detail="Failed to restore memories, please try again later",
            ) from e

"""
Memory Controller - Unified memory management controller

Provides RESTful API routes for:
- Personal add (POST /memories): batch messages for personal scene
- Group add (POST /memories/group): batch messages for group scene
- Personal flush (POST /memories/flush): trigger boundary detection for personal scene
- Group flush (POST /memories/group/flush): trigger boundary detection for group scene
- Memory search (GET /memories/search): keyword/vector/hybrid/rrf/agentic retrieval with grouped results
- Memory deletion (POST /memories/delete): soft delete by ID or filter conditions
"""

import asyncio
import json
import logging
import time

from contextlib import suppress
from fastapi import HTTPException, Request as FastAPIRequest

from core.di.decorators import controller
from core.di import get_bean_by_type
from core.interface.controller.base_controller import BaseController, get, post
from core.observation.stage_timer import stage_timed, timed
from core.constants.errors import ErrorCode, ErrorStatus
from agentic_layer.memory_manager import MemoryManager
from api_specs.request_converter import (
    convert_dict_to_retrieve_mem_request,
    convert_personal_add_to_memorize_request,
    convert_group_add_to_memorize_request,
    convert_personal_flush_to_memorize_request,
    convert_group_flush_to_memorize_request,
    convert_agent_add_to_memorize_request,
    convert_agent_flush_to_memorize_request,
)
from api_specs.id_generator import DEFAULT_SESSION_ID
from infra_layer.adapters.out.event.personal_memorize_event import PersonalMemorizeEvent
from infra_layer.adapters.out.event.group_memorize_event import GroupMemorizeEvent
from core.events.event_publisher import ApplicationEventPublisher
from infra_layer.adapters.input.api.dto.memory_dto import (
    # Request DTOs
    # Add / Flush DTOs
    PersonalAddRequest,
    GroupAddRequest,
    PersonalFlushRequest,
    GroupFlushRequest,
    AddResponse,
    FlushResponse,
)
from api_specs.dtos.memory import AgentAddRequest, AgentFlushRequest
from api_specs.dtos.memory_delete import DeleteMemoriesRequest
from core.request import log_request
from core.request.app_logic_provider import AppLogicProvider
from core.component.redis_provider import RedisProvider
from service.content_enrich_provider import ContentEnrichProvider
from service.raw_message_service import RawMessageService
from service.sender_service import SenderService
from service.memcell_delete_service import MemCellDeleteService
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

    def __init__(self):
        """Initialize controller"""
        super().__init__(
            prefix="/api/v1/memories", tags=["Memory Controller"], default_auth="none"
        )
        self.memory_manager = MemoryManager()
        self.redis_provider = get_bean_by_type(RedisProvider)
        self._content_enrich = get_bean_by_type(ContentEnrichProvider)
        self._app_logic = get_bean_by_type(AppLogicProvider)
        logger.info("MemoryController initialized with MemoryManager")

    # =========================================================================
    # Add Endpoints
    # =========================================================================

    @post(
        "",
        response_model=AddResponse,
        summary="Store messages (personal)",
        description="Store batch messages into personal memory space.",
    )
    @log_request()
    @stage_timed("add")
    async def add_personal_memories(
        self, request: FastAPIRequest, request_body: PersonalAddRequest = None
    ) -> AddResponse:
        """POST /api/v1/memories - Personal add endpoint."""
        del request_body
        start_time = time.perf_counter()
        space_id = get_space_id_for_metrics()
        raw_data_type = get_raw_data_type_label(None)

        try:
            request_data = await request.json()
            logger.info(
                "Received personal add request: user_id=%s", request_data.get("user_id")
            )

            memorize_request = convert_personal_add_to_memorize_request(request_data)
            raw_data_type = get_raw_data_type_label(
                memorize_request.raw_data_type.value
            )
            msg_count = len(memorize_request.new_raw_data_list)

            record_memorize_message(
                space_id=space_id,
                raw_data_type=raw_data_type,
                status='received',
                count=msg_count,
            )

            group_id = memorize_request.group_id
            session_id = memorize_request.session_id

            # Auto-register group
            if group_id:
                asyncio.create_task(self._ensure_group_exists(group_id=group_id))

            # Auto-register session (skip for default sentinel)
            if session_id and session_id != DEFAULT_SESSION_ID:
                asyncio.create_task(self._ensure_session_exists(session_id=session_id))

            # Auto-register senders from converted data (includes auto-filled sender_ids)
            self._auto_register_senders(memorize_request.new_raw_data_list)

            # Enrich sender_name from DB for messages that didn't provide one
            messages = request_data.get("messages", [])
            with timed("enrich_sender_names"):
                await self._enrich_sender_names(
                    messages, memorize_request.new_raw_data_list
                )

            # Content enrichment (e.g. multimodal parsing, no-op by default)
            # Must run BEFORE save_request_logs so that parsed multimodal text
            # is included in the flat content saved to RawMessage.
            with timed("enrich_content"):
                await self._content_enrich.enrich(memorize_request.new_raw_data_list)

            # Save request logs
            with timed("persist_raw_messages"):
                await self._save_raw_messages(
                    memorize_request, request, "add_personal_memories"
                )

            # Process
            memory_count = await self.memory_manager.memorize(memorize_request)

            status = 'extracted' if memory_count > 0 else 'accumulated'
            record_memorize_request(
                space_id=space_id,
                raw_data_type=raw_data_type,
                status=status,
                duration_seconds=time.perf_counter() - start_time,
            )

            # Publish personal memorize event (fire-and-forget)
            asyncio.create_task(
                self._publish_event(
                    PersonalMemorizeEvent(
                        user_id=request_data.get("user_id", ""),
                        session_id=session_id,
                        group_id=group_id,
                    )
                )
            )

            return {
                "data": {
                    "request_id": self._app_logic.get_current_request_id(),
                    "message_count": msg_count,
                    "status": status,
                    "message": "Messages accepted",
                }
            }

        except ValueError as e:
            logger.error("Personal add parameter error: %s", e)
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
            raise HTTPException(status_code=422, detail=str(e)) from e
        except HTTPException:
            record_memorize_request(
                space_id=space_id,
                raw_data_type=raw_data_type,
                status='error',
                duration_seconds=time.perf_counter() - start_time,
            )
            raise
        except Exception as e:
            logger.error("Personal add failed: %s", e, exc_info=True)
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

    @post(
        "/group",
        response_model=AddResponse,
        summary="Store messages (group)",
        description="Store batch messages into group memory space.",
    )
    @log_request()
    @stage_timed("add")
    async def add_group_memories(
        self, request: FastAPIRequest, request_body: GroupAddRequest = None
    ) -> AddResponse:
        """POST /api/v1/memories/group - Group add endpoint."""
        del request_body
        start_time = time.perf_counter()
        space_id = get_space_id_for_metrics()
        raw_data_type = get_raw_data_type_label(None)

        try:
            request_data = await request.json()
            logger.info(
                "Received group add request: group_id=%s", request_data.get("group_id")
            )

            memorize_request = convert_group_add_to_memorize_request(request_data)
            raw_data_type = get_raw_data_type_label(
                memorize_request.raw_data_type.value
            )
            msg_count = len(memorize_request.new_raw_data_list)

            record_memorize_message(
                space_id=space_id,
                raw_data_type=raw_data_type,
                status='received',
                count=msg_count,
            )

            group_id = memorize_request.group_id

            # Auto-register group (with optional metadata)
            if group_id:
                group_meta = request_data.get("group_meta") or {}
                asyncio.create_task(
                    self._ensure_group_exists(
                        group_id=group_id,
                        name=group_meta.get("name"),
                        description=group_meta.get("description"),
                    )
                )

            # Auto-register senders from converted data (includes sender_ids from request)
            self._auto_register_senders(memorize_request.new_raw_data_list)

            # Enrich sender_name from DB for messages that didn't provide one
            messages = request_data.get("messages", [])
            with timed("enrich_sender_names"):
                await self._enrich_sender_names(
                    messages, memorize_request.new_raw_data_list
                )

            # Content enrichment (e.g. multimodal parsing, no-op by default)
            # Must run BEFORE save_request_logs so that parsed multimodal text
            # is included in the flat content saved to RawMessage.
            with timed("enrich_content"):
                await self._content_enrich.enrich(memorize_request.new_raw_data_list)

            # Save request logs
            with timed("persist_raw_messages"):
                await self._save_raw_messages(
                    memorize_request, request, "add_group_memories"
                )

            # Process
            memory_count = await self.memory_manager.memorize(memorize_request)

            status = 'extracted' if memory_count > 0 else 'accumulated'
            record_memorize_request(
                space_id=space_id,
                raw_data_type=raw_data_type,
                status=status,
                duration_seconds=time.perf_counter() - start_time,
            )

            # Publish group memorize event (fire-and-forget)
            sender_ids = list(
                {
                    msg.get("sender_id")
                    for msg in request_data.get("messages", [])
                    if msg.get("sender_id")
                }
            )
            asyncio.create_task(
                self._publish_event(
                    GroupMemorizeEvent(group_id=group_id, sender_ids=sender_ids)
                )
            )

            return {
                "data": {
                    "request_id": self._app_logic.get_current_request_id(),
                    "message_count": msg_count,
                    "status": status,
                    "message": "Messages accepted",
                }
            }

        except ValueError as e:
            logger.error("Group add parameter error: %s", e)
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
            raise HTTPException(status_code=422, detail=str(e)) from e
        except HTTPException:
            record_memorize_request(
                space_id=space_id,
                raw_data_type=raw_data_type,
                status='error',
                duration_seconds=time.perf_counter() - start_time,
            )
            raise
        except Exception as e:
            logger.error("Group add failed: %s", e, exc_info=True)
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

    # =========================================================================
    # Flush Endpoints
    # =========================================================================

    @post(
        "/flush",
        response_model=FlushResponse,
        summary="Flush personal memories",
        description="Trigger boundary detection on accumulated personal messages.",
    )
    @log_request()
    @stage_timed("flush")
    async def flush_personal_memories(
        self, request: FastAPIRequest, request_body: PersonalFlushRequest = None
    ) -> FlushResponse:
        """POST /api/v1/memories/flush - Personal flush endpoint."""
        del request_body
        start_time = time.perf_counter()
        space_id = get_space_id_for_metrics()
        raw_data_type = get_raw_data_type_label("Conversation")

        try:
            request_data = await request.json()
            logger.info(
                "Received personal flush: user_id=%s", request_data.get("user_id")
            )

            memorize_request = convert_personal_flush_to_memorize_request(request_data)

            memory_count = await self.memory_manager.memorize(memorize_request)

            status = 'extracted' if memory_count > 0 else 'no_extraction'
            record_memorize_request(
                space_id=space_id,
                raw_data_type=raw_data_type,
                status='flush',
                duration_seconds=time.perf_counter() - start_time,
            )

            return {
                "data": {
                    "request_id": self._app_logic.get_current_request_id(),
                    "status": status,
                    "message": "Flush completed",
                }
            }

        except ValueError as e:
            logger.error("Personal flush parameter error: %s", e)
            raise HTTPException(status_code=422, detail=str(e)) from e
        except Exception as e:
            logger.error("Personal flush failed: %s", e, exc_info=True)
            raise HTTPException(
                status_code=500, detail="Flush failed, please try again later"
            ) from e

    @post(
        "/group/flush",
        response_model=FlushResponse,
        summary="Flush group memories",
        description="Trigger boundary detection on accumulated group messages.",
    )
    @log_request()
    @stage_timed("flush")
    async def flush_group_memories(
        self, request: FastAPIRequest, request_body: GroupFlushRequest = None
    ) -> FlushResponse:
        """POST /api/v1/memories/group/flush - Group flush endpoint."""
        del request_body
        start_time = time.perf_counter()
        space_id = get_space_id_for_metrics()
        raw_data_type = get_raw_data_type_label("Conversation")

        try:
            request_data = await request.json()
            logger.info(
                "Received group flush: group_id=%s", request_data.get("group_id")
            )

            memorize_request = convert_group_flush_to_memorize_request(request_data)

            memory_count = await self.memory_manager.memorize(memorize_request)

            status = 'extracted' if memory_count > 0 else 'no_extraction'
            record_memorize_request(
                space_id=space_id,
                raw_data_type=raw_data_type,
                status='flush',
                duration_seconds=time.perf_counter() - start_time,
            )

            return {
                "data": {
                    "request_id": self._app_logic.get_current_request_id(),
                    "status": status,
                    "message": "Flush completed",
                }
            }

        except ValueError as e:
            logger.error("Group flush parameter error: %s", e)
            raise HTTPException(status_code=422, detail=str(e)) from e
        except Exception as e:
            logger.error("Group flush failed: %s", e, exc_info=True)
            raise HTTPException(
                status_code=500, detail="Flush failed, please try again later"
            ) from e

    @post(
        "/agent/flush",
        response_model=FlushResponse,
        summary="Flush agent memories",
        description="""Trigger agent-aware boundary detection on accumulated agent trajectory messages.

        Flushes buffered agent messages for the specified user, triggering memory extraction
        (agent cases and skills) if a conversation boundary is detected.

        ## Request Body Fields:
        - **user_id** (required, string): Owner user ID
        - **session_id** (optional, string): Target session to flush. If omitted, flushes all sessions for the user.

        ## Request Examples:

        **Flush all sessions:**
        ```json
        {"user_id": "user_01"}
        ```

        **Flush specific session:**
        ```json
        {"user_id": "user_01", "session_id": "sess_agent_001"}
        ```

        ## Response:
        - **data.request_id**: Request tracking ID (reserved)
        - **data.status**: `extracted` (memory extraction triggered) or `no_extraction` (no boundary detected)
        - **data.message**: Human-readable status description
        """,
        responses={
            200: {
                "description": "Flush completed",
                "content": {
                    "application/json": {
                        "examples": {
                            "no_extraction": {
                                "summary": "No boundary detected",
                                "value": {
                                    "data": {
                                        "request_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                                        "status": "no_extraction",
                                        "message": "Flush completed",
                                    }
                                },
                            },
                            "extracted": {
                                "summary": "Memory extraction triggered",
                                "value": {
                                    "data": {
                                        "request_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                                        "status": "extracted",
                                        "message": "Flush completed",
                                    }
                                },
                            },
                        }
                    }
                },
            },
            422: {
                "description": "Request parameter error",
                "content": {
                    "application/json": {
                        "examples": {
                            "missing_user_id": {
                                "summary": "Missing required user_id",
                                "value": {"detail": "Missing required field: user_id"},
                            }
                        }
                    }
                },
            },
        },
    )
    @log_request()
    @stage_timed("agent_flush")
    async def flush_agent_memories(
        self, request: FastAPIRequest, request_body: AgentFlushRequest = None
    ) -> FlushResponse:
        """POST /api/v1/memories/agent/flush - Agent flush endpoint."""
        del request_body
        start_time = time.perf_counter()
        space_id = get_space_id_for_metrics()
        raw_data_type = get_raw_data_type_label("AgentConversation")

        try:
            request_data = await request.json()
            logger.info("Received agent flush: user_id=%s", request_data.get("user_id"))

            with timed("convert_request"):
                memorize_request = convert_agent_flush_to_memorize_request(request_data)

            with timed("memorize"):
                memory_count = await self.memory_manager.memorize(memorize_request)

            status = 'extracted' if memory_count > 0 else 'no_extraction'
            record_memorize_request(
                space_id=space_id,
                raw_data_type=raw_data_type,
                status='flush',
                duration_seconds=time.perf_counter() - start_time,
            )

            return {
                "data": {
                    "request_id": self._app_logic.get_current_request_id(),
                    "status": status,
                    "message": "Flush completed",
                }
            }

        except ValueError as e:
            logger.error("Agent flush parameter error: %s", e)
            raise HTTPException(status_code=422, detail=str(e)) from e
        except Exception as e:
            logger.error("Agent flush failed: %s", e, exc_info=True)
            raise HTTPException(
                status_code=500, detail="Flush failed, please try again later"
            ) from e

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _auto_register_senders(self, raw_data_list: list) -> None:
        """Fire-and-forget auto-register senders from converted raw data list.

        Uses the converted RawData objects (not the original request JSON)
        so that auto-filled sender_ids are included.
        """
        seen = set()
        for raw_data in raw_data_list:
            content = raw_data.content
            sender_id = content.get("sender_id")
            if sender_id and sender_id not in seen:
                seen.add(sender_id)
                asyncio.create_task(
                    self._ensure_sender_exists(
                        sender_id=sender_id, sender_name=content.get("sender_name")
                    )
                )

    # =========================================================================
    # Agent Add
    # =========================================================================

    @post(
        "/agent",
        response_model=AddResponse,
        summary="Store agent trajectory messages",
        description="""Store agent trajectory messages (user/assistant/tool) into memory.
        Supports tool_calls and tool_call_id for OpenAI-format function calling.

        ## Request Body Fields:
        - **user_id** (required, string): Owner user ID
        - **session_id** (optional, string): Session identifier for conversation isolation
        - **messages** (required, array, 1-500): Agent trajectory messages

        ## Message Fields:
        - **role** (required): `user`, `assistant`, or `tool`
        - **timestamp** (required, int): Message timestamp in unix milliseconds
        - **content** (required, string or array): Accepts plain string shorthand `"hello"` or array of content items `[{type: "text", text: "hello"}]`
        - **message_id** (optional): Message unique ID
        - **sender_id** (optional): Sender identifier
        - **sender_name** (optional): Sender display name
        - **tool_calls** (optional): Tool calls made by the assistant (OpenAI format). Only when role='assistant'
        - **tool_call_id** (optional): ID of the tool call this message responds to. Required when role='tool'

        ## Request Example (with tool calls):
        ```json
        {
          "user_id": "user_01",
          "session_id": "sess_agent_001",
          "messages": [
            {
              "role": "user",
              "timestamp": 1710835200000,
              "content": "What is the weather in Tokyo?"
            },
            {
              "role": "assistant",
              "timestamp": 1710835201000,
              "content": "Let me check the weather for you.",
              "tool_calls": [
                {
                  "id": "call_abc123",
                  "type": "function",
                  "function": {"name": "get_weather", "arguments": "{\\"city\\": \\"Tokyo\\"}"}
                }
              ]
            },
            {
              "role": "tool",
              "timestamp": 1710835202000,
              "tool_call_id": "call_abc123",
              "content": [{"type": "text", "text": "Tokyo: 18C, partly cloudy"}]
            },
            {
              "role": "assistant",
              "timestamp": 1710835203000,
              "content": [{"type": "text", "text": "The weather in Tokyo is 18C and partly cloudy."}]
            }
          ]
        }
        ```

        ## Response:
        - **data.request_id**: Request tracking ID (reserved)
        - **data.message_count**: Number of messages accepted
        - **data.status**: `accumulated` (buffered) or `extracted` (memory extraction triggered)
        - **data.message**: Human-readable status description
        """,
        responses={
            200: {
                "description": "Messages accepted",
                "content": {
                    "application/json": {
                        "example": {
                            "data": {
                                "request_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                                "message_count": 4,
                                "status": "accumulated",
                                "message": "Messages accepted",
                            }
                        }
                    }
                },
            },
            422: {
                "description": "Request parameter error",
                "content": {
                    "application/json": {
                        "examples": {
                            "missing_user_id": {
                                "summary": "Missing required user_id",
                                "value": {"detail": "Missing required field: user_id"},
                            },
                            "invalid_role": {
                                "summary": "Invalid message role",
                                "value": {
                                    "detail": "Invalid value for messages[].role: 'invalid'. Must be 'user', 'assistant', or 'tool'"
                                },
                            },
                            "missing_tool_call_id": {
                                "summary": "Missing tool_call_id for tool message",
                                "value": {
                                    "detail": "Missing required field: messages[].tool_call_id (required when role='tool')"
                                },
                            },
                        }
                    }
                },
            },
        },
    )
    @log_request()
    @stage_timed("agent_add")
    async def add_agent_memories(
        self, request: FastAPIRequest, request_body: AgentAddRequest = None
    ) -> AddResponse:
        """POST /api/v1/memories/agent - Agent add endpoint."""
        del request_body
        start_time = time.perf_counter()
        space_id = get_space_id_for_metrics()
        raw_data_type = get_raw_data_type_label(None)

        try:
            request_data = await request.json()
            logger.info(
                "Received agent add request: user_id=%s", request_data.get("user_id")
            )

            with timed("convert_request"):
                memorize_request = convert_agent_add_to_memorize_request(request_data)
            raw_data_type = get_raw_data_type_label(
                memorize_request.raw_data_type.value
            )
            msg_count = len(memorize_request.new_raw_data_list)

            record_memorize_message(
                space_id=space_id,
                raw_data_type=raw_data_type,
                status='received',
                count=msg_count,
            )

            group_id = memorize_request.group_id
            session_id = memorize_request.session_id

            # Auto-register group
            if group_id:
                asyncio.create_task(self._ensure_group_exists(group_id=group_id))

            # Auto-register session (skip for default sentinel)
            if session_id and session_id != DEFAULT_SESSION_ID:
                asyncio.create_task(self._ensure_session_exists(session_id=session_id))

            # Auto-register senders from converted data (skip role=tool)
            for raw_data in memorize_request.new_raw_data_list:
                content = raw_data.content
                if content.get("role") != "tool":
                    sender_id = content.get("sender_id")
                    if sender_id:
                        asyncio.create_task(
                            self._ensure_sender_exists(
                                sender_id=sender_id,
                                sender_name=content.get("sender_name"),
                            )
                        )

            # Enrich sender_name from DB for messages that didn't provide one
            messages = request_data.get("messages", [])
            await self._enrich_sender_names(
                messages, memorize_request.new_raw_data_list
            )

            # Content enrichment (e.g. multimodal parsing, no-op by default)
            # Must run BEFORE save_request_logs so that parsed multimodal text
            # is included in the flat content saved to RawMessage.
            await self._content_enrich.enrich(memorize_request.new_raw_data_list)

            # Save request logs
            with timed("persist_raw_messages"):
                await self._save_raw_messages(
                    memorize_request, request, "add_agent_memories"
                )

            # Process
            with timed("memorize"):
                memory_count = await self.memory_manager.memorize(memorize_request)

            status = 'extracted' if memory_count > 0 else 'accumulated'
            record_memorize_request(
                space_id=space_id,
                raw_data_type=raw_data_type,
                status=status,
                duration_seconds=time.perf_counter() - start_time,
            )

            # Publish personal memorize event (agent is solo scene)
            asyncio.create_task(
                self._publish_event(
                    PersonalMemorizeEvent(
                        user_id=request_data.get("user_id", ""),
                        session_id=session_id,
                        group_id=group_id,
                    )
                )
            )

            return {
                "data": {
                    "request_id": self._app_logic.get_current_request_id(),
                    "message_count": msg_count,
                    "status": status,
                    "message": "Messages accepted",
                }
            }

        except ValueError as e:
            logger.error("Agent add parameter error: %s", e)
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
            raise HTTPException(status_code=422, detail=str(e)) from e
        except HTTPException:
            record_memorize_request(
                space_id=space_id,
                raw_data_type=raw_data_type,
                status='error',
                duration_seconds=time.perf_counter() - start_time,
            )
            raise
        except Exception as e:
            logger.error("Agent add failed: %s", e, exc_info=True)
            record_memorize_error(
                space_id=space_id,
                raw_data_type=raw_data_type,
                stage='memorize',
                error_type=classify_memorize_error(e),
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

    # =========================================================================
    # Helper methods
    # =========================================================================

    async def _save_raw_messages(
        self, memorize_request, request, endpoint_name: str
    ) -> None:
        """Save individual messages from the request as RawMessage documents."""
        if memorize_request.new_raw_data_list:
            raw_message_service = get_bean_by_type(RawMessageService)
            await raw_message_service.save_raw_messages(
                request=memorize_request,
                version="1.0.0",
                endpoint_name=endpoint_name,
                method=request.method,
                url=str(request.url),
            )

    async def _ensure_group_exists(
        self, group_id: str, name: str | None = None, description: str | None = None
    ) -> None:
        """Auto-register group when memory is ingested."""
        try:
            from service.group_service import GroupService

            group_service = get_bean_by_type(GroupService)
            await group_service.ensure_group_exists(
                group_id=group_id, name=name, description=description
            )
        except Exception as e:
            logger.warning(
                "Failed to auto-register group: group_id=%s, error=%s", group_id, e
            )

    async def _enrich_sender_names(self, messages: list, raw_data_list: list) -> None:
        """Enrich sender_name from DB for messages that didn't provide one.

        Collects sender_ids where sender_name was not explicitly provided,
        batch queries the Sender collection, and updates the corresponding
        RawData content dicts in place.

        Args:
            messages: Original request messages list
            raw_data_list: RawData objects from MemorizeRequest.new_raw_data_list
        """
        # Collect sender_ids that need enrichment
        missing_sender_ids = set()
        for msg in messages:
            if not msg.get("sender_name") and msg.get("sender_id"):
                missing_sender_ids.add(msg["sender_id"])

        if not missing_sender_ids:
            return

        try:
            sender_service = get_bean_by_type(SenderService)
            name_map = await sender_service.batch_get_sender_names(
                list(missing_sender_ids)
            )
            if not name_map:
                return

            # Update RawData content dicts in place
            for raw_data in raw_data_list:
                sid = raw_data.content.get("sender_id")
                if sid in name_map:
                    raw_data.content["sender_name"] = name_map[sid]
        except Exception as e:
            logger.warning("Failed to enrich sender names: %s", e)

    async def _ensure_sender_exists(
        self, sender_id: str, sender_name: str = None
    ) -> None:
        """Auto-register sender when memory is ingested."""
        try:
            from service.sender_service import SenderService

            sender_service = get_bean_by_type(SenderService)
            await sender_service.ensure_sender_exists(
                sender_id=sender_id, name=sender_name
            )
        except Exception as e:
            logger.warning(
                "Failed to auto-register sender: sender_id=%s, error=%s", sender_id, e
            )

    async def _publish_event(self, event) -> None:
        """Publish an event via ApplicationEventPublisher (fire-and-forget)."""
        try:
            publisher = get_bean_by_type(ApplicationEventPublisher)
            await publisher.publish(event)
        except Exception as e:
            logger.warning("Failed to publish event %s: %s", type(event).__name__, e)

    async def _ensure_session_exists(self, session_id: str) -> None:
        """Auto-register session when memory is ingested."""
        try:
            from service.session_service import SessionService

            session_service = get_bean_by_type(SessionService)
            await session_service.ensure_session_exists(session_id=session_id)
        except Exception as e:
            logger.warning(
                "Failed to auto-register session: session_id=%s, error=%s",
                session_id,
                e,
            )

    @post(
        "/delete",
        status_code=204,
        summary="Delete memories (soft delete)",
        description="Soft delete memories by ID or by filter conditions. "
        "Two modes: single delete (by memory_id) or batch delete (by filters). "
        "Returns 204 No Content on success.",
    )
    @stage_timed("delete")
    async def delete_memories(self, request_body: DeleteMemoriesRequest) -> None:
        """Soft delete memories by ID or by filter conditions."""
        delete_service = get_bean_by_type(MemCellDeleteService)

        if request_body.memory_id is not None:
            await delete_service.delete_by_id(request_body.memory_id)
        else:
            await delete_service.delete_by_filters(
                user_id=request_body.user_id,
                group_id=request_body.group_id,
                session_id=request_body.session_id,
                sender_id=request_body.sender_id,
            )

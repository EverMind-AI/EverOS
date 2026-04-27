"""
Memory GET Controller

Provides POST /api/v1/memories/get endpoint for fetching memories
with structured filters DSL. Delegates to MemoryManager for business logic.
"""

import logging

from fastapi import HTTPException, Request as FastAPIRequest
from pydantic import ValidationError

from core.di.decorators import controller
from core.interface.controller.base_controller import BaseController, post
from core.observation.stage_timer import stage_timed
from agentic_layer.memory_manager import MemoryManager
from agentic_layer.get_mem_service import InvalidScopeError
from api_specs.dtos.memory import GetMemRequest, GetMemoriesResponse

logger = logging.getLogger(__name__)


@controller("memory_get_controller", primary=True)
class MemoryGetController(BaseController):
    """Memory GET Controller"""

    def __init__(self):
        super().__init__(
            prefix="/api/v1/memories",
            tags=["Memory Get Controller"],
            default_auth="none",
        )
        self._manager = MemoryManager()

    @post(
        "/get",
        response_model=GetMemoriesResponse,
        summary="Get memories with filters",
        description="""
        Get episodic memories or profiles using structured filters DSL.

        ## Request Body Fields:
        - **memory_type** (required, string): Memory type to query
            - `episodic_memory`: episodic memory (derived from conversations)
            - `profile`: user profile (explicit_info and implicit_traits)
            - `agent_case`: agent experience (task intent + approach + quality score)
            - `agent_skill`: agent skill (reusable skills from clustered cases)
        - **filters** (required, object): Filter conditions using MongoDB-style DSL
            - Must contain at least one of `user_id` or `group_id` at first level
            - See **Filters DSL** section below for full syntax
        - **page** (optional, int): Page number, starts from 1 (default: 1, min: 1)
        - **page_size** (optional, int): Items per page (default: 20, min: 1, max: 100)
        - **rank_by** (optional, string): Sort field (default: `timestamp`). Profile type auto-fallbacks to `updated_at`
        - **rank_order** (optional, string): Sort order (default: `desc`)
            - `asc`: oldest first
            - `desc`: newest first

        ## Filters DSL:
        Allowlist-based: only the following fields are processed, unknown fields are silently ignored.

        | Field | Type | Operators | Description |
        |-------|------|-----------|-------------|
        | `user_id` | string | eq, in | User ID filter (conditional required) |
        | `group_id` | string | eq, in | Group ID filter (conditional required) |
        | `session_id` | string | eq, in, gt, gte, lt, lte | Session ID filter |
        | `timestamp` | int (epoch ms/s) or ISO string | eq, gt, gte, lt, lte | Time range filter. Epoch >1e12 treated as ms |
        | `AND` | array of filter objects | - | All conditions must match |
        | `OR` | array of filter objects | - | Any condition must match |

        **Operator syntax**: plain value = eq, `{"in": [...]}`, `{"gte": v, "lt": v}`

        ## Filter Examples:

        **Simple eq:**
        ```json
        {"filters": {"user_id": "user_01", "group_id": "group_01"}}
        ```

        **in operator:**
        ```json
        {"filters": {"group_id": {"in": ["group_01", "group_02"]}}}
        ```

        **Timestamp range with AND:**
        ```json
        {"filters": {"user_id": "user_01", "AND": [{"timestamp": {"gte": 1768469400000}}, {"timestamp": {"lt": 1768473000000}}]}}
        ```

        **Nested AND/OR:**
        ```json
        {"filters": {"AND": [{"user_id": "user_01"}, {"OR": [{"group_id": "group_01"}, {"group_id": "group_02"}]}]}}
        ```

        ## Response:
        - **data.episodes**: List of episodic memory items (populated when memory_type=episodic_memory, otherwise [])
            - Each item contains: id, user_id, group_id, session_id, timestamp, participants, summary, subject, episode, type, parent_type, parent_id
        - **data.profiles**: List of profile items (populated when memory_type=profile, otherwise [])
            - Each item contains: id, user_id, group_id, profile_data (explicit_info, implicit_traits), scenario, cluster_ids, memcell_count
        - **data.agent_cases**: List of agent case items (populated when memory_type=agent_case, otherwise [])
            - Each item contains: id, user_id, group_id, session_id, task_intent, approach, quality_score, timestamp, parent_type, parent_id
        - **data.agent_skills**: List of agent skill items (populated when memory_type=agent_skill, otherwise [])
            - Each item contains: id, user_id, group_id, cluster_id, name, description, content, confidence, maturity_score
        - **data.total_count**: Total records matching filters (for pagination calculation)
        - **data.count**: Number of records in current page

        ## Use cases:
        - User profile display and personalized recommendations
        - Conversation history review with time range filtering
        - Cross-group memory aggregation using group_id in operator
        - Paginated memory browsing with custom sort order
        """,
        responses={
            422: {
                "description": "Request parameter error",
                "content": {
                    "application/json": {
                        "examples": {
                            "missing_scope": {
                                "summary": "Missing required scope filter",
                                "value": {
                                    "code": "HTTP_ERROR",
                                    "message": "filters must contain at least one of 'user_id' or 'group_id' at first level",
                                    "request_id": "req_abc123",
                                    "timestamp": "2026-01-01T00:00:00+00:00",
                                    "path": "/api/v1/memories/get",
                                },
                            },
                            "invalid_memory_type": {
                                "summary": "Invalid memory_type value",
                                "value": {
                                    "code": "HTTP_ERROR",
                                    "message": "memory_type: Value error, memory_type must be one of: agent_case, agent_skill, episodic_memory, profile",
                                    "request_id": "req_abc123",
                                    "timestamp": "2026-01-01T00:00:00+00:00",
                                    "path": "/api/v1/memories/get",
                                },
                            },
                            "missing_filters": {
                                "summary": "Missing required filters field",
                                "value": {
                                    "code": "HTTP_ERROR",
                                    "message": "filters: Field required",
                                    "request_id": "req_abc123",
                                    "timestamp": "2026-01-01T00:00:00+00:00",
                                    "path": "/api/v1/memories/get",
                                },
                            },
                        }
                    }
                },
            },
            500: {
                "description": "Internal server error",
                "content": {
                    "application/json": {
                        "example": {
                            "code": "SYSTEM_ERROR",
                            "message": "Internal server error",
                            "request_id": "req_abc123",
                            "timestamp": "2026-01-01T00:00:00+00:00",
                            "path": "/api/v1/memories/get",
                        }
                    }
                },
            },
        },
    )
    @stage_timed("get")
    async def get_memories(
        self, fastapi_request: FastAPIRequest, request_body: GetMemRequest = None
    ) -> GetMemoriesResponse:
        """Get memories by type with filters."""
        del request_body  # Used for OpenAPI documentation only

        # 1. Parse and validate request body
        try:
            body = await fastapi_request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON request body")

        try:
            request = GetMemRequest(**body)
        except ValidationError as e:
            first_error = e.errors()[0]
            field = ".".join(str(loc) for loc in first_error.get("loc", []))
            msg = first_error.get("msg", "Validation error")
            raise HTTPException(
                status_code=422, detail=f"{field}: {msg}" if field else msg
            )

        # 2. Delegate to MemoryManager
        try:
            response = await self._manager.get_mem(request)
        except InvalidScopeError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        except Exception as e:
            logger.error("get_memories failed: %s", e, exc_info=True)
            raise HTTPException(
                status_code=500,
                detail="Failed to retrieve memory, please try again later",
            ) from e

        return GetMemoriesResponse(data=response)

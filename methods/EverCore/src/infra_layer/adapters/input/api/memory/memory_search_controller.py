"""
Memory Search Controller (v1)

Provides POST /api/v1/memories/search endpoint for searching memories
with structured query, filters, and multiple retrieval methods. Delegates
to SearchMemoryService for business logic.
"""

import logging

from fastapi import HTTPException, Request as FastAPIRequest
from pydantic import ValidationError

from core.di.decorators import controller
from core.interface.controller.base_controller import BaseController, post
from core.observation.stage_timer import stage_timed
from api_specs.dtos.memory import SearchMemoriesRequest, SearchMemoriesResponse
from agentic_layer.search_mem_service import SearchMemoryService

logger = logging.getLogger(__name__)


@controller("memory_search_controller", primary=True)
class MemorySearchController(BaseController):
    """Memory Search Controller (v1)"""

    def __init__(self):
        super().__init__(
            prefix="/api/v1/memories",
            tags=["Memory Search Controller"],
            default_auth="none",
        )
        self._service = SearchMemoryService()

    @post(
        "/search",
        response_model=SearchMemoriesResponse,
        summary="Search memories (v1)",
        description="""
        V1 unified memory search endpoint supporting multiple memory types and retrieval methods.

        ## Memory Types:
        - **episodic_memory**: Episodic memories (ES + Milvus)
        - **profile**: User profiles (Milvus only)
        - **raw_message**: Raw unprocessed messages (ES only) - pending messages not yet extracted into memories
        - **agent_memory**: Agent memory - cases and skills (ES + Milvus)

        ## Retrieval Methods:
        - **keyword**: BM25 keyword search (ES only)
        - **vector**: Vector semantic search (Milvus only)
        - **hybrid**: Hybrid retrieval (default). episodic_memory uses hierarchical retrieval, others use ES + Milvus + Rerank
        - **agentic**: LLM-guided multi-round retrieval

        ## Filters DSL:
        Allowlist-based: only the following fields are processed, unknown fields are silently ignored.

        | Field | Type | Operators | Description |
        |-------|------|-----------|-------------|
        | `user_id` | string | eq, in | User ID filter (conditional required) |
        | `group_id` | string | eq, in | Group ID filter (conditional required) |
        | `session_id` | string | eq, in | Session ID filter |
        | `timestamp` | int (epoch ms/s) or ISO string | eq, gt, gte, lt, lte | Time range filter. Epoch >1e12 treated as ms |
        | `AND` | array of filter objects | - | All conditions must match |
        | `OR` | array of filter objects | - | Any condition must match |

        **Operator syntax**: plain value = eq, `{"in": [...]}`, `{"gte": v, "lt": v}`

        ## Filter Examples:

        **Simple filters:**
        ```json
        {"filters": {"user_id": "user_123", "group_id": "group_abc"}}
        ```

        **in operator:**
        ```json
        {"filters": {"group_id": {"in": ["group_a", "group_b"]}}}
        ```

        **AND with timestamp range:**
        ```json
        {"filters": {"AND": [{"user_id": "user_123"}, {"timestamp": {"gte": 1704067200000}}]}}
        ```

        **Search agent memory:**
        ```json
        {"query": "How to handle timeout errors", "method": "hybrid", "memory_types": ["agent_memory"], "filters": {"user_id": "user_01"}}
        ```

        ## Response:
        - **data.episodes**: List of episodic memory search results
            - Each item contains: id, user_id, group_id, session_id, timestamp, participants, summary, subject, episode, type, parent_type, parent_id, score
        - **data.profiles**: List of profile search results
            - Each item contains: id, user_id, group_id, profile_data, scenario, memcell_count, score
        - **data.raw_messages**: List of raw message search results (when memory_types includes "raw_message")
        - **data.agent_memory**: Agent cases and skills (when memory_types includes "agent_memory"). Returns both cases and skills in a single container.
            - **data.agent_memory.cases**: List of agent case search results (id, user_id, group_id, session_id, task_intent, approach, quality_score, timestamp, parent_type, parent_id, score)
            - **data.agent_memory.skills**: List of agent skill search results (id, user_id, group_id, cluster_id, name, description, content, confidence, maturity_score, score)
        - **data.query**: Query information echoed from request
        - **data.original_data**: Original data (if include_original_data=true)
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
                                    "message": "filters must contain at least one of 'user_id' or 'group_id'",
                                    "request_id": "req_abc123",
                                    "timestamp": "2026-01-01T00:00:00+00:00",
                                    "path": "/api/v1/memories/search",
                                },
                            },
                            "invalid_method": {
                                "summary": "Invalid retrieval method",
                                "value": {
                                    "code": "HTTP_ERROR",
                                    "message": "method: Value error, method must be one of: keyword, vector, hybrid, rrf, agentic",
                                    "request_id": "req_abc123",
                                    "timestamp": "2026-01-01T00:00:00+00:00",
                                    "path": "/api/v1/memories/search",
                                },
                            },
                            "invalid_memory_types": {
                                "summary": "Invalid memory type",
                                "value": {
                                    "code": "HTTP_ERROR",
                                    "message": "memory_types: Value error, memory_types must be from: agent_memory, episodic_memory, profile, raw_message",
                                    "request_id": "req_abc123",
                                    "timestamp": "2026-01-01T00:00:00+00:00",
                                    "path": "/api/v1/memories/search",
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
                            "path": "/api/v1/memories/search",
                        }
                    }
                },
            },
        },
    )
    @stage_timed("search")
    async def search_memories(
        self,
        fastapi_request: FastAPIRequest,
        request_body: SearchMemoriesRequest = None,
    ) -> SearchMemoriesResponse:
        """Search memories using v1 unified search interface.

        Args:
            fastapi_request: FastAPI request object
            request_body: SearchMemoriesRequest (used for OpenAPI documentation only)

        Returns:
            SearchMemoriesResponse with search results
        """
        del request_body  # Used for OpenAPI documentation only

        # 1. Parse and validate request body
        try:
            body = await fastapi_request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON request body")

        try:
            request = SearchMemoriesRequest(**body)
        except ValidationError as e:
            first_error = e.errors()[0]
            field = ".".join(str(loc) for loc in first_error.get("loc", []))
            msg = first_error.get("msg", "Validation error")
            raise HTTPException(
                status_code=422, detail=f"{field}: {msg}" if field else msg
            )

        # TODO: to optimize in future, never remove this dependency (MR!199) before that
        fastapi_request.state.search_method = request.method

        # 2. Delegate to SearchMemoryService
        try:
            response_data = await self._service.search_memories(
                query=request.query,
                method=request.method,
                memory_types=request.memory_types,
                filters=request.filters,
                top_k=request.top_k,
                radius=request.radius,
                include_original_data=request.include_original_data,
            )
        except ValueError as e:
            logger.error("search_memories validation error: %s", e)
            raise HTTPException(status_code=422, detail=str(e)) from e
        except Exception as e:
            logger.error("search_memories failed: %s", e, exc_info=True)
            raise HTTPException(
                status_code=500,
                detail="Failed to search memories, please try again later",
            ) from e

        agent_case_count = (
            len(response_data.agent_memory.cases) if response_data.agent_memory else 0
        )
        agent_skill_count = (
            len(response_data.agent_memory.skills) if response_data.agent_memory else 0
        )
        logger.info(
            "search_memories completed: episodes=%d, profiles=%d, agent_cases=%d, agent_skills=%d",
            len(response_data.episodes),
            len(response_data.profiles),
            agent_case_count,
            agent_skill_count,
        )

        return SearchMemoriesResponse(data=response_data)

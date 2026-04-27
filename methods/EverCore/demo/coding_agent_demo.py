"""Coding Agent Memory Demo - Code Generation + Testing Trajectory

Demonstrates how the agent memory system works with a coding agent:
1. Send two similar coding agent trajectories (OpenAI chat completion format)
2. Wait for memory extraction (MemCell segmentation + AgentCase extraction)
3. The second trajectory triggers MemScene clustering -> AgentSkill extraction
4. Fetch extracted agent experiences AND agent skills
5. Hybrid search over agent_case and agent_skill (retrieve_method=hybrid)

The mock trajectories simulate a coding agent that:
- Trajectory 1: Adds input validation to a FastAPI endpoint, then adds custom error messages
- Trajectory 2: Adds rate limiting middleware, then adds per-endpoint configuration

Both follow the same coding pattern (read existing code -> implement -> run tests -> refine),
so they cluster together and trigger AgentSkill extraction.

Prerequisites:
    Start the API server first (in another terminal):
    uv run python src/run.py

Run the demo:
    uv run python src/bootstrap.py demo/coding_agent_demo.py
"""

import asyncio
import json

from demo.utils.agent_demo_helpers import (
    AgentDemoRunner,
    MEMORY_TYPE_PRINTERS,
    print_separator,
    print_search_case_results,
    print_search_skill_results,
)


# ==================== Mock Coding Agent Trajectory ====================

def build_mock_trajectory():
    """Build a mock coding agent trajectory in OpenAI chat completion format.

    Simulates a two-turn interaction:
    Turn 1: User asks to add input validation -> Agent reads code, writes validation, runs tests
    Turn 2: User asks for custom error messages -> Agent searches patterns, updates code, runs tests
    """
    messages = [
        # ===== Turn 1: Add input validation to a FastAPI endpoint =====
        {
            "role": "user",
            "content": "Add input validation to the POST /api/users endpoint. It should validate email format, password strength (min 8 chars, 1 uppercase, 1 digit), and username length (3-20 chars).",
        },
        {
            "role": "assistant",
            "content": "Let me first read the existing endpoint code to understand the current structure.",
            "tool_calls": [
                {
                    "id": "call_001",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": json.dumps({
                            "path": "src/api/users.py"
                        }),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_001",
            "content": json.dumps({
                "content": (
                    "from fastapi import APIRouter, HTTPException\n"
                    "from pydantic import BaseModel\n\n"
                    "router = APIRouter()\n\n"
                    "class CreateUserRequest(BaseModel):\n"
                    "    username: str\n"
                    "    email: str\n"
                    "    password: str\n\n"
                    "@router.post('/api/users')\n"
                    "async def create_user(req: CreateUserRequest):\n"
                    "    # TODO: add validation\n"
                    "    user = await db.create_user(req.username, req.email, req.password)\n"
                    "    return {'id': user.id, 'username': user.username}\n"
                ),
            }),
        },
        {
            "role": "assistant",
            "content": "I see the endpoint uses Pydantic models but has no field-level validation. I'll add validators using Pydantic's field_validator.",
            "tool_calls": [
                {
                    "id": "call_002",
                    "type": "function",
                    "function": {
                        "name": "write_file",
                        "arguments": json.dumps({
                            "path": "src/api/users.py",
                            "content": (
                                "import re\n"
                                "from fastapi import APIRouter, HTTPException\n"
                                "from pydantic import BaseModel, field_validator\n\n"
                                "router = APIRouter()\n\n"
                                "class CreateUserRequest(BaseModel):\n"
                                "    username: str\n"
                                "    email: str\n"
                                "    password: str\n\n"
                                "    @field_validator('email')\n"
                                "    @classmethod\n"
                                "    def validate_email(cls, v):\n"
                                "        pattern = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+$'\n"
                                "        if not re.match(pattern, v):\n"
                                "            raise ValueError('Invalid email format')\n"
                                "        return v\n\n"
                                "    @field_validator('password')\n"
                                "    @classmethod\n"
                                "    def validate_password(cls, v):\n"
                                "        if len(v) < 8:\n"
                                "            raise ValueError('Password must be at least 8 characters')\n"
                                "        if not re.search(r'[A-Z]', v):\n"
                                "            raise ValueError('Password must contain at least 1 uppercase letter')\n"
                                "        if not re.search(r'\\d', v):\n"
                                "            raise ValueError('Password must contain at least 1 digit')\n"
                                "        return v\n\n"
                                "    @field_validator('username')\n"
                                "    @classmethod\n"
                                "    def validate_username(cls, v):\n"
                                "        if not 3 <= len(v) <= 20:\n"
                                "            raise ValueError('Username must be 3-20 characters')\n"
                                "        return v\n\n"
                                "@router.post('/api/users')\n"
                                "async def create_user(req: CreateUserRequest):\n"
                                "    user = await db.create_user(req.username, req.email, req.password)\n"
                                "    return {'id': user.id, 'username': user.username}\n"
                            ),
                        }),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_002",
            "content": json.dumps({"status": "ok", "path": "src/api/users.py", "bytes_written": 1247}),
        },
        {
            "role": "assistant",
            "content": "Validation code written. Let me run the tests to make sure everything passes.",
            "tool_calls": [
                {
                    "id": "call_003",
                    "type": "function",
                    "function": {
                        "name": "run_tests",
                        "arguments": json.dumps({
                            "command": "pytest tests/test_users.py -v"
                        }),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_003",
            "content": json.dumps({
                "exit_code": 0,
                "output": (
                    "tests/test_users.py::test_create_user_valid PASSED\n"
                    "tests/test_users.py::test_create_user_invalid_email PASSED\n"
                    "tests/test_users.py::test_create_user_weak_password PASSED\n"
                    "tests/test_users.py::test_create_user_short_username PASSED\n"
                    "\n4 passed in 0.32s"
                ),
            }),
        },
        {
            "role": "assistant",
            "content": (
                "I've added input validation to the POST /api/users endpoint using Pydantic field_validator:\n\n"
                "1. **Email validation** - regex pattern check for valid email format\n"
                "2. **Password validation** - minimum 8 chars, at least 1 uppercase letter, at least 1 digit\n"
                "3. **Username validation** - length between 3-20 characters\n\n"
                "All 4 tests pass. The validators raise `ValueError` with descriptive messages, "
                "which Pydantic automatically converts to 422 Unprocessable Entity responses."
            ),
        },

        # ===== Turn 2: Add custom error response format =====
        {
            "role": "user",
            "content": "The default Pydantic 422 error format is hard to parse on the frontend. Add a custom exception handler that returns a clean JSON error format like {\"error\": \"validation_error\", \"details\": [{\"field\": \"email\", \"message\": \"Invalid email format\"}]}",
        },
        {
            "role": "assistant",
            "content": "Let me check if there's an existing exception handler pattern in the project.",
            "tool_calls": [
                {
                    "id": "call_004",
                    "type": "function",
                    "function": {
                        "name": "search_code",
                        "arguments": json.dumps({
                            "query": "exception_handler RequestValidationError",
                            "file_pattern": "*.py",
                        }),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_004",
            "content": json.dumps({
                "matches": [
                    {
                        "file": "src/main.py",
                        "line": 15,
                        "content": "# TODO: add custom exception handlers",
                    }
                ],
            }),
        },
        {
            "role": "assistant",
            "content": "No existing handler found, just a TODO. I'll add a custom validation exception handler to main.py.",
            "tool_calls": [
                {
                    "id": "call_005",
                    "type": "function",
                    "function": {
                        "name": "write_file",
                        "arguments": json.dumps({
                            "path": "src/exceptions.py",
                            "content": (
                                "from fastapi import Request\n"
                                "from fastapi.responses import JSONResponse\n"
                                "from fastapi.exceptions import RequestValidationError\n\n"
                                "async def validation_exception_handler(request: Request, exc: RequestValidationError):\n"
                                "    details = []\n"
                                "    for error in exc.errors():\n"
                                "        field = error['loc'][-1] if error['loc'] else 'unknown'\n"
                                "        details.append({'field': str(field), 'message': error['msg']})\n"
                                "    return JSONResponse(\n"
                                "        status_code=422,\n"
                                "        content={'error': 'validation_error', 'details': details},\n"
                                "    )\n"
                            ),
                        }),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_005",
            "content": json.dumps({"status": "ok", "path": "src/exceptions.py", "bytes_written": 512}),
        },
        {
            "role": "assistant",
            "content": "Now let me run the tests to verify the custom error format.",
            "tool_calls": [
                {
                    "id": "call_006",
                    "type": "function",
                    "function": {
                        "name": "run_tests",
                        "arguments": json.dumps({
                            "command": "pytest tests/test_users.py tests/test_exceptions.py -v"
                        }),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_006",
            "content": json.dumps({
                "exit_code": 0,
                "output": (
                    "tests/test_users.py::test_create_user_valid PASSED\n"
                    "tests/test_users.py::test_create_user_invalid_email PASSED\n"
                    "tests/test_exceptions.py::test_validation_error_format PASSED\n"
                    "tests/test_exceptions.py::test_multiple_validation_errors PASSED\n"
                    "\n4 passed in 0.28s"
                ),
            }),
        },
        {
            "role": "assistant",
            "content": (
                "Done! I've added a custom validation exception handler:\n\n"
                "1. Created `src/exceptions.py` with a `validation_exception_handler` that catches "
                "`RequestValidationError`\n"
                "2. It transforms Pydantic's default 422 response into a clean format:\n"
                "   ```json\n"
                "   {\"error\": \"validation_error\", \"details\": [{\"field\": \"email\", \"message\": \"Invalid email format\"}]}\n"
                "   ```\n"
                "3. Register it in main.py: `app.add_exception_handler(RequestValidationError, validation_exception_handler)`\n\n"
                "All tests pass including the new exception format tests."
            ),
        },
    ]

    return messages


def build_second_trajectory():
    """Build a second coding agent trajectory to trigger clustering + skill extraction.

    Simulates a different coding task but same pattern:
    Turn 1: User asks to add rate limiting middleware -> Agent reads code, implements, tests
    Turn 2: User asks for per-endpoint config -> Agent searches patterns, updates code, tests

    This trajectory follows the same coding pattern as the first (read -> implement -> test ->
    search -> refine -> test), so it should cluster into the same MemScene and trigger
    AgentSkill extraction.
    """
    messages = [
        # ===== Turn 1: Add rate limiting middleware =====
        {
            "role": "user",
            "content": "Add rate limiting middleware to the API. Default limit should be 100 requests per minute per IP. Use an in-memory store with sliding window.",
        },
        {
            "role": "assistant",
            "content": "Let me check the existing middleware setup first.",
            "tool_calls": [
                {
                    "id": "call_101",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": json.dumps({
                            "path": "src/middleware/__init__.py"
                        }),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_101",
            "content": json.dumps({
                "content": (
                    "from .cors import CORSMiddleware\n"
                    "from .logging import LoggingMiddleware\n\n"
                    "MIDDLEWARE_STACK = [CORSMiddleware, LoggingMiddleware]\n"
                ),
            }),
        },
        {
            "role": "assistant",
            "content": "I see there's a middleware stack pattern. I'll create a rate limiter middleware that fits into this architecture.",
            "tool_calls": [
                {
                    "id": "call_102",
                    "type": "function",
                    "function": {
                        "name": "write_file",
                        "arguments": json.dumps({
                            "path": "src/middleware/rate_limit.py",
                            "content": (
                                "import time\n"
                                "from collections import defaultdict\n"
                                "from fastapi import Request, Response\n"
                                "from starlette.middleware.base import BaseHTTPMiddleware\n\n"
                                "class RateLimitMiddleware(BaseHTTPMiddleware):\n"
                                "    def __init__(self, app, max_requests: int = 100, window_seconds: int = 60):\n"
                                "        super().__init__(app)\n"
                                "        self.max_requests = max_requests\n"
                                "        self.window_seconds = window_seconds\n"
                                "        self.requests = defaultdict(list)\n\n"
                                "    async def dispatch(self, request: Request, call_next):\n"
                                "        client_ip = request.client.host\n"
                                "        now = time.time()\n"
                                "        window_start = now - self.window_seconds\n"
                                "        # Sliding window: remove expired entries\n"
                                "        self.requests[client_ip] = [\n"
                                "            t for t in self.requests[client_ip] if t > window_start\n"
                                "        ]\n"
                                "        if len(self.requests[client_ip]) >= self.max_requests:\n"
                                "            return Response(\n"
                                "                content='{\"error\": \"rate_limit_exceeded\"}',\n"
                                "                status_code=429,\n"
                                "                headers={'Retry-After': str(self.window_seconds)},\n"
                                "            )\n"
                                "        self.requests[client_ip].append(now)\n"
                                "        return await call_next(request)\n"
                            ),
                        }),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_102",
            "content": json.dumps({"status": "ok", "path": "src/middleware/rate_limit.py", "bytes_written": 1089}),
        },
        {
            "role": "assistant",
            "content": "Rate limiter written. Let me run the middleware tests.",
            "tool_calls": [
                {
                    "id": "call_103",
                    "type": "function",
                    "function": {
                        "name": "run_tests",
                        "arguments": json.dumps({
                            "command": "pytest tests/test_rate_limit.py -v"
                        }),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_103",
            "content": json.dumps({
                "exit_code": 0,
                "output": (
                    "tests/test_rate_limit.py::test_under_limit_passes PASSED\n"
                    "tests/test_rate_limit.py::test_over_limit_returns_429 PASSED\n"
                    "tests/test_rate_limit.py::test_sliding_window_resets PASSED\n"
                    "\n3 passed in 0.18s"
                ),
            }),
        },
        {
            "role": "assistant",
            "content": (
                "I've added rate limiting middleware to the API:\n\n"
                "1. Created `src/middleware/rate_limit.py` with `RateLimitMiddleware`\n"
                "2. Uses an in-memory sliding window algorithm:\n"
                "   - Tracks request timestamps per client IP\n"
                "   - Removes expired entries outside the window on each request\n"
                "   - Returns 429 with `Retry-After` header when limit exceeded\n"
                "3. Default: 100 requests per 60-second sliding window\n\n"
                "All 3 tests pass. Add it to the middleware stack: "
                "`MIDDLEWARE_STACK.append(RateLimitMiddleware)`"
            ),
        },

        # ===== Turn 2: Add per-endpoint rate limit configuration =====
        {
            "role": "user",
            "content": "Some endpoints like /api/auth/login should have a stricter limit (10/min) and /api/health should be unlimited. Add per-endpoint rate limit configuration.",
        },
        {
            "role": "assistant",
            "content": "Let me look at how routes are structured to find the best way to add per-endpoint config.",
            "tool_calls": [
                {
                    "id": "call_104",
                    "type": "function",
                    "function": {
                        "name": "search_code",
                        "arguments": json.dumps({
                            "query": "APIRouter include_router prefix",
                            "file_pattern": "*.py",
                        }),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_104",
            "content": json.dumps({
                "matches": [
                    {
                        "file": "src/main.py",
                        "line": 22,
                        "content": "app.include_router(auth_router, prefix='/api/auth')",
                    },
                    {
                        "file": "src/main.py",
                        "line": 23,
                        "content": "app.include_router(users_router, prefix='/api/users')",
                    },
                    {
                        "file": "src/main.py",
                        "line": 24,
                        "content": "app.include_router(health_router, prefix='/api/health')",
                    },
                ],
            }),
        },
        {
            "role": "assistant",
            "content": "Good, routes use standard prefix-based structure. I'll add a config dict to the rate limiter for per-path overrides.",
            "tool_calls": [
                {
                    "id": "call_105",
                    "type": "function",
                    "function": {
                        "name": "write_file",
                        "arguments": json.dumps({
                            "path": "src/middleware/rate_limit.py",
                            "content": (
                                "import time\n"
                                "from collections import defaultdict\n"
                                "from typing import Dict, Optional\n"
                                "from fastapi import Request, Response\n"
                                "from starlette.middleware.base import BaseHTTPMiddleware\n\n"
                                "class RateLimitConfig:\n"
                                "    def __init__(self, max_requests: int, window_seconds: int = 60):\n"
                                "        self.max_requests = max_requests\n"
                                "        self.window_seconds = window_seconds\n\n"
                                "UNLIMITED = RateLimitConfig(max_requests=0)  # 0 means no limit\n\n"
                                "class RateLimitMiddleware(BaseHTTPMiddleware):\n"
                                "    def __init__(self, app, default_limit: int = 100,\n"
                                "                 window_seconds: int = 60,\n"
                                "                 path_configs: Optional[Dict[str, RateLimitConfig]] = None):\n"
                                "        super().__init__(app)\n"
                                "        self.default_config = RateLimitConfig(default_limit, window_seconds)\n"
                                "        self.path_configs = path_configs or {}\n"
                                "        self.requests = defaultdict(list)\n\n"
                                "    def _get_config(self, path: str) -> RateLimitConfig:\n"
                                "        for prefix, config in self.path_configs.items():\n"
                                "            if path.startswith(prefix):\n"
                                "                return config\n"
                                "        return self.default_config\n\n"
                                "    async def dispatch(self, request: Request, call_next):\n"
                                "        config = self._get_config(request.url.path)\n"
                                "        if config.max_requests == 0:  # unlimited\n"
                                "            return await call_next(request)\n"
                                "        client_ip = request.client.host\n"
                                "        key = f'{client_ip}:{request.url.path}'\n"
                                "        now = time.time()\n"
                                "        window_start = now - config.window_seconds\n"
                                "        self.requests[key] = [t for t in self.requests[key] if t > window_start]\n"
                                "        if len(self.requests[key]) >= config.max_requests:\n"
                                "            return Response(\n"
                                "                content='{\"error\": \"rate_limit_exceeded\"}',\n"
                                "                status_code=429,\n"
                                "                headers={'Retry-After': str(config.window_seconds)},\n"
                                "            )\n"
                                "        self.requests[key].append(now)\n"
                                "        return await call_next(request)\n"
                            ),
                        }),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_105",
            "content": json.dumps({"status": "ok", "path": "src/middleware/rate_limit.py", "bytes_written": 1856}),
        },
        {
            "role": "assistant",
            "content": "Updated. Let me run the full test suite to verify.",
            "tool_calls": [
                {
                    "id": "call_106",
                    "type": "function",
                    "function": {
                        "name": "run_tests",
                        "arguments": json.dumps({
                            "command": "pytest tests/test_rate_limit.py -v"
                        }),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_106",
            "content": json.dumps({
                "exit_code": 0,
                "output": (
                    "tests/test_rate_limit.py::test_under_limit_passes PASSED\n"
                    "tests/test_rate_limit.py::test_over_limit_returns_429 PASSED\n"
                    "tests/test_rate_limit.py::test_sliding_window_resets PASSED\n"
                    "tests/test_rate_limit.py::test_per_endpoint_strict_limit PASSED\n"
                    "tests/test_rate_limit.py::test_unlimited_endpoint_bypasses PASSED\n"
                    "\n5 passed in 0.24s"
                ),
            }),
        },
        {
            "role": "assistant",
            "content": (
                "Done! Added per-endpoint rate limit configuration:\n\n"
                "1. Added `RateLimitConfig` class for per-path settings\n"
                "2. Added `UNLIMITED` constant for endpoints with no limit\n"
                "3. The middleware now accepts `path_configs` dict for prefix-based overrides:\n"
                "   ```python\n"
                "   RateLimitMiddleware(app, path_configs={\n"
                "       '/api/auth/login': RateLimitConfig(max_requests=10),\n"
                "       '/api/health': UNLIMITED,\n"
                "   })\n"
                "   ```\n"
                "4. Rate limit keys now include the path for per-endpoint tracking\n\n"
                "All 5 tests pass including the new per-endpoint tests."
            ),
        },
    ]

    return messages


# ==================== Main ====================

async def main():
    runner = AgentDemoRunner(
        group_id_prefix="coding_agent_demo",
        group_name="Coding Agent Demo Session",
        description="Coding Agent Memory Demo",
        tags=["demo", "agent", "coding"],
        msg_prefix="coding_msg",
    )

    print_separator("Coding Agent Memory Demo - Code Implementation + Testing")

    # Step 1: Initialize conversation metadata
    print("\nStep 1: Initialize conversation metadata")
    print_separator()
    await runner.save_conversation_meta()

    # Step 2: Send first agent trajectory (input validation)
    print("\nStep 2: Send first coding agent trajectory (input validation)")
    print_separator()

    messages1 = build_mock_trajectory()
    print(f"  Sending {len(messages1)} messages (2 interaction turns)...\n")

    for idx, msg in enumerate(messages1):
        is_last = (idx == len(messages1) - 1)
        await runner.send_agent_message(msg, idx, flush=is_last)
        await asyncio.sleep(0.5)

    # Step 3: Wait for first extraction to complete
    print("\nStep 3: Wait for first extraction pipeline")
    print_separator()
    print("  Waiting 60 seconds for first AgentCase extraction...")
    await asyncio.sleep(60)

    # Step 4: Send second trajectory (rate limiting - different feature, same pattern)
    print("\nStep 4: Send second trajectory (rate limiting middleware)")
    print_separator()
    print("  Different feature, but same coding pattern (read -> implement -> test -> refine).")
    print("  It will cluster into the same MemScene and trigger AgentSkill extraction.\n")

    messages2 = build_second_trajectory()
    offset = len(messages1)
    print(f"  Sending {len(messages2)} messages (2 interaction turns)...\n")

    for idx, msg in enumerate(messages2):
        is_last = (idx == len(messages2) - 1)
        await runner.send_agent_message(msg, offset + idx, flush=is_last)
        await asyncio.sleep(0.5)

    # Step 5: Wait for second extraction + skill extraction
    print("\nStep 5: Wait for second extraction pipeline + AgentSkill extraction")
    print_separator()
    print("  Waiting 60 seconds for AgentCase + AgentSkill extraction...")
    await asyncio.sleep(60)

    # Step 6: Fetch and print all extracted memories
    print("\nStep 6: Fetch all extracted memories")

    for memory_type, label, printer in MEMORY_TYPE_PRINTERS:
        print(f"\n--- {label} ---")
        memories = await runner.fetch_memories(memory_type)
        print(f"  ({len(memories)} records)")
        printer(memories)

    # Step 7: Hybrid search demo (vector + BM25 keyword search)
    print("\nStep 7: Hybrid search (agent_memory)")

    search_queries = [
        "how to add validation and error handling to API endpoints",
        "implement rate limiting for a REST API",
        "how to return structured JSON errors from FastAPI",
        "best practices for implementing and testing API features with code review",
        "how to add input validation to a Pydantic model",
        "steps to build rate limiting middleware with per-endpoint config",
        "how to handle validation errors in a web API",
    ]

    for query in search_queries:
        print(f"\n--- Hybrid Search: Agent Memory ---")
        print(f"  Query: \"{query}\"")
        result = await runner.search_memories(query, "agent_memory")
        cases = result.get("cases", [])
        skills = result.get("skills", [])
        print(f"  ({len(cases)} cases, {len(skills)} skills)")
        if cases:
            print("  [Cases]")
            print_search_case_results(cases)
        if skills:
            print("  [Skills]")
            print_search_skill_results(skills)

    # Step 8: Agentic search demo (LLM-guided multi-round retrieval)
    print("\nStep 8: Agentic search for Agent Memory (LLM-guided multi-round retrieval)")

    agentic_queries = [
        "best practices for implementing and testing API features with code review",
        "how to add input validation to a Pydantic model",
        "steps to build rate limiting middleware with per-endpoint config",
    ]

    for query in agentic_queries:
        print(f"\n--- Agentic Search: Agent Memory ---")
        print(f"  Query: \"{query}\"")
        result = await runner.search_memories(
            query, "agent_memory", retrieve_method="agentic"
        )
        cases = result.get("cases", [])
        skills = result.get("skills", [])
        print(f"  ({len(cases)} cases, {len(skills)} skills)")
        if cases:
            print("  [Cases]")
            print_search_case_results(cases)
        if skills:
            print("  [Skills]")
            print_search_skill_results(skills)

    print_separator("Demo completed!")


if __name__ == "__main__":
    asyncio.run(main())

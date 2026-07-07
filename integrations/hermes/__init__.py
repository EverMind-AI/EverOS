"""EverOS memory provider for the Hermes Agent.

Wires the Phase 1 plugin modules (``_client`` / ``_config`` / ``_formatting``
/ ``_setup`` / ``_types`` / ``_constants``) into the Hermes ``MemoryProvider``
ABC. This is the only module in the bundle that imports Hermes symbols
(``agent.memory_provider``, ``hermes_constants``, ``tools.registry``,
``utils``); everything else is Hermes-agnostic so it can be unit-tested in
isolation.

The provider mirrors each turn into EverOS via a background sync thread,
prefetches relevant memory before the agent answers, exposes four tools
(``everos_search`` / ``everos_list`` / ``everos_add`` / ``everos_flush``),
and trips a circuit breaker after repeated transient failures (mem0 parity).
"""

from __future__ import annotations

import atexit
import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Any

from agent.memory_provider import MemoryProvider
from hermes_constants import get_hermes_home
from tools.registry import tool_error
from utils import atomic_json_write

from ._client import EverOSApiClient
from ._config import (
    get_scope_ids,
    is_configured,
    load_config,
    resolve_agent_id,
    resolve_user_id,
)
from ._constants import (
    _BREAKER_COOLDOWN_SECS,
    _BREAKER_THRESHOLD,
    _DEFAULT_AGENT_ID,
    _DEFAULT_API_URL,
    _DEFAULT_SEARCH_METHOD,
    _DEFAULT_TOOL_SEARCH_TOP_K,
    _MIRROR_TARGETS,
    _PREFETCH_WAIT_SECS,
    TOOL_ADD,
    TOOL_FLUSH,
    TOOL_LIST,
    TOOL_SEARCH,
)
from ._formatting import (
    format_memory_write_message,
    format_prefetch,
    format_tool_result,
)
from ._setup import post_setup as _run_setup
from ._types import EverOSClientError, MessageItem, ScopeIds

logger = logging.getLogger(__name__)

# Trivial prompts that should not trigger a prefetch (mem0 parity).
_TRIVIAL_PROMPT_RE = re.compile(
    r"^(yes|no|ok|okay|sure|thanks|thank you|y|n|yep|nope|yeah|nah|"
    r"continue|go ahead|do it|proceed|got it|cool|nice|great|done|next|"
    r"lgtm|k)$",
    re.IGNORECASE,
)

# EverOS error codes that warrant circuit-breaker trips (transient). Client
# errors (INVALID_INPUT / NOT_FOUND / ...) are not transient and do not count.
_TRANSIENT_CODES: frozenset[str] = frozenset(
    {
        "",
        "CLIENT_CLOSED",
        "CONFIGURATION_ERROR",
        "EXTERNAL_SERVICE_UNAVAILABLE",
        "INTERNAL_ERROR",
    }
)

# ── OpenAI function-calling tool schemas ────────────────────────────────────

_SEARCH_SCHEMA: dict[str, Any] = {
    "name": TOOL_SEARCH,
    "description": (
        "Search the EverOS memory store for relevant episodes, atomic "
        "facts, and the user profile."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural-language search query.",
            },
            "top_k": {
                "type": "integer",
                "default": _DEFAULT_TOOL_SEARCH_TOP_K,
                "description": "Maximum number of results (1-100).",
            },
            "method": {
                "type": "string",
                "enum": ["keyword", "vector", "hybrid", "agentic"],
                "default": "hybrid",
            },
            "owner": {
                "type": "string",
                "enum": ["user", "agent"],
                "default": "user",
            },
        },
        "required": ["query"],
    },
}

_LIST_SCHEMA: dict[str, Any] = {
    "name": TOOL_LIST,
    "description": "List memories of a given type from the EverOS store.",
    "parameters": {
        "type": "object",
        "properties": {
            "memory_type": {
                "type": "string",
                "enum": ["episode", "profile", "agent_case", "agent_skill"],
                "default": "episode",
            },
            "page": {
                "type": "integer",
                "default": 1,
                "minimum": 1,
            },
            "page_size": {
                "type": "integer",
                "default": 20,
                "minimum": 1,
                "maximum": 100,
            },
            "owner": {
                "type": "string",
                "enum": ["user", "agent"],
                "default": "user",
            },
        },
        "required": [],
    },
}

_ADD_SCHEMA: dict[str, Any] = {
    "name": TOOL_ADD,
    "description": "Store a fact in the EverOS memory store for the user.",
    "parameters": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The fact to remember.",
            },
        },
        "required": ["content"],
    },
}

_FLUSH_SCHEMA: dict[str, Any] = {
    "name": TOOL_FLUSH,
    "description": "Flush the buffered session messages for extraction.",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}


class EverosMemoryProvider(MemoryProvider):
    """Hermes ``MemoryProvider`` backed by an EverOS server."""

    def __init__(self) -> None:
        self._config: dict[str, Any] | None = None
        self._client: EverOSApiClient | None = None
        self._session_id: str | None = None
        self._user_id: str | None = None
        self._agent_id: str | None = None
        self._scope: ScopeIds | None = None
        self._sync_thread: threading.Thread | None = None
        self._mirror_thread: threading.Thread | None = None
        self._prefetch_thread: threading.Thread | None = None
        self._prefetch_query: str | None = None
        self._prefetch_result: str | None = None
        self._prefetch_done: bool = False
        self._consecutive_failures: int = 0
        self._breaker_open_until: float = 0.0
        self._breaker_lock: threading.Lock = threading.Lock()
        self._sync_lock: threading.Lock = threading.Lock()
        self._prefetch_lock: threading.Lock = threading.Lock()
        self._atexit_registered: bool = False
        self._init_error: str = ""

    # ── identity / availability ───────────────────────────────────────────

    @property
    def name(self) -> str:
        return "everos"

    def is_available(self) -> bool:
        cfg = load_config(get_hermes_home())
        return is_configured(cfg)

    # ── lifecycle ─────────────────────────────────────────────────────────

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        self._config = load_config(get_hermes_home())
        self._user_id = resolve_user_id(
            self._config, kwargs.get("user_id"), kwargs.get("user_id_alt")
        )
        self._agent_id = resolve_agent_id(self._config)
        self._scope = get_scope_ids(self._config)
        try:
            self._client = EverOSApiClient(
                self._config.get("api_url") or _DEFAULT_API_URL
            )
        except Exception as exc:
            self._init_error = str(exc)
            self._client = None
            logger.warning("EverOS client init failed: %s", exc)
        self._session_id = session_id
        if not self._atexit_registered:
            atexit.register(self._shutdown_client)
            self._atexit_registered = True

    # ── circuit breaker ───────────────────────────────────────────────────

    def _is_breaker_open(self) -> bool:
        with self._breaker_lock:
            if self._consecutive_failures < _BREAKER_THRESHOLD:
                return False
            if time.monotonic() >= self._breaker_open_until:
                self._consecutive_failures = 0
                self._breaker_open_until = 0.0
                return False
            return True

    def _record_success(self) -> None:
        with self._breaker_lock:
            self._consecutive_failures = 0

    def _record_failure(self) -> None:
        with self._breaker_lock:
            self._consecutive_failures += 1
            if self._consecutive_failures >= _BREAKER_THRESHOLD:
                self._breaker_open_until = time.monotonic() + _BREAKER_COOLDOWN_SECS
                logger.warning(
                    "EverOS circuit breaker opened after %d failures",
                    self._consecutive_failures,
                )

    @staticmethod
    def _is_transient(code: str) -> bool:
        return code in _TRANSIENT_CODES

    # ── turn sync ─────────────────────────────────────────────────────────

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: list[dict[str, Any]] | None = None,
    ) -> None:
        if self._client is None or self._is_breaker_open():
            return
        now_ms = time.time_ns() // 1_000_000
        user_msg: MessageItem = {
            "sender_id": self._user_id,
            "role": "user",
            "timestamp": now_ms,
            "content": user_content,
        }
        asst_msg: MessageItem = {
            "sender_id": self._agent_id,
            "role": "assistant",
            "timestamp": now_ms + 1,
            "content": assistant_content,
        }
        sid = session_id or self._session_id
        with self._sync_lock:
            prev = self._sync_thread
            if prev is not None:
                prev.join(timeout=5.0)
                if prev.is_alive():
                    return  # avoid duplicate writes
            thread = threading.Thread(
                target=self._sync_worker,
                args=(sid, [user_msg, asst_msg]),
                name="everos-sync",
                daemon=True,
            )
            self._sync_thread = thread
            thread.start()

    def _sync_worker(self, session_id: str, messages: list[MessageItem]) -> None:
        client = self._client
        scope = self._scope
        if client is None or scope is None:
            return
        try:
            client.add_messages(session_id, scope.app_id, scope.project_id, messages)
            self._record_success()
        except EverOSClientError as exc:
            if self._is_transient(exc.code):
                self._record_failure()
            logger.warning("EverOS sync_turn add failed: %s", exc)
        except Exception:
            logger.warning("EverOS sync_worker error", exc_info=True)

    # ── prefetch ──────────────────────────────────────────────────────────

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if self._client is None or self._is_breaker_open():
            return ""
        cached = self._consume_prefetch_result(query)
        if cached is not None:
            return cached
        self._start_prefetch(query)
        thread = self._prefetch_thread
        if thread is not None:
            thread.join(timeout=_PREFETCH_WAIT_SECS)
        return self._consume_prefetch_result(query) or ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        if self._is_trivial_prompt(query):
            return
        self._start_prefetch(query)

    def _start_prefetch(self, query: str) -> None:
        with self._prefetch_lock:
            self._prefetch_query = query
            self._prefetch_result = None
            self._prefetch_done = False
            thread = threading.Thread(
                target=self._prefetch_worker,
                args=(query,),
                name="everos-prefetch",
                daemon=True,
            )
            self._prefetch_thread = thread
            thread.start()

    def _prefetch_worker(self, query: str) -> None:
        client = self._client
        scope = self._scope
        if client is None or scope is None:
            return
        try:
            data = client.search(
                self._user_id,
                None,
                scope.app_id,
                scope.project_id,
                query,
                include_profile=True,
                top_k=_DEFAULT_TOOL_SEARCH_TOP_K,
                method=_DEFAULT_SEARCH_METHOD,
            )
            body = format_prefetch(query, data)
            with self._prefetch_lock:
                if self._prefetch_query == query:
                    self._prefetch_result = body
                    self._prefetch_done = True
            self._record_success()
        except EverOSClientError as exc:
            if self._is_transient(exc.code):
                self._record_failure()
            else:
                logger.warning("EverOS prefetch failed: %s", exc)
        except Exception:
            logger.warning("EverOS prefetch error", exc_info=True)

    def _consume_prefetch_result(self, query: str) -> str | None:
        with self._prefetch_lock:
            if self._prefetch_query == query and self._prefetch_done:
                result = self._prefetch_result
                self._prefetch_result = None
                self._prefetch_done = False
                self._prefetch_query = None
                return result
            return None

    @staticmethod
    def _is_trivial_prompt(query: str) -> bool:
        q = (query or "").strip().lower()
        if not q or q.startswith("/"):
            return True
        return bool(_TRIVIAL_PROMPT_RE.match(q))

    # ── session / memory-write hooks ──────────────────────────────────────

    def on_session_end(self, messages: list[dict[str, Any]]) -> None:
        if self._sync_thread is not None:
            self._sync_thread.join(timeout=10.0)
        if (
            self._client is not None
            and not self._is_breaker_open()
            and self._scope is not None
        ):
            try:
                self._client.flush_session(
                    self._session_id,
                    self._scope.app_id,
                    self._scope.project_id,
                )
            except EverOSClientError as exc:
                logger.warning("EverOS flush on session end failed: %s", exc)
            except Exception:
                logger.warning("EverOS on_session_end error", exc_info=True)
        self._shutdown_client()

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        effective = "add" if action in ("add", "replace") else action
        if (
            effective != "add"
            or target not in _MIRROR_TARGETS
            or not content
            or self._client is None
            or self._is_breaker_open()
            or self._scope is None
        ):
            return
        msg = format_memory_write_message(
            content, self._user_id, time.time_ns() // 1_000_000
        )
        sid = self._session_id
        scope = self._scope
        client = self._client

        def _mirror() -> None:
            try:
                client.add_messages(sid, scope.app_id, scope.project_id, [msg])
                client.flush_session(sid, scope.app_id, scope.project_id)
                self._record_success()
            except EverOSClientError as exc:
                if self._is_transient(exc.code):
                    self._record_failure()
                logger.warning("EverOS memory-write mirror failed: %s", exc)
            except Exception:
                logger.warning("EverOS mirror error", exc_info=True)

        self._mirror_thread = threading.Thread(
            target=_mirror, name="everos-mirror", daemon=True
        )
        self._mirror_thread.start()

    # ── tools ─────────────────────────────────────────────────────────────

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return [_SEARCH_SCHEMA, _LIST_SCHEMA, _ADD_SCHEMA, _FLUSH_SCHEMA]

    def handle_tool_call(
        self, tool_name: str, args: dict[str, Any], **kwargs: Any
    ) -> str:
        if self._client is None:
            return tool_error(f"EverOS backend not initialized: {self._init_error}")
        if self._is_breaker_open():
            return tool_error(
                "EverOS temporarily unavailable (circuit breaker open). "
                "Will retry automatically."
            )
        scope = self._scope
        client = self._client
        if scope is None:
            return tool_error("EverOS scope not initialized")

        if tool_name == TOOL_SEARCH:
            return self._tool_search(client, scope, args)
        if tool_name == TOOL_LIST:
            return self._tool_list(client, scope, args)
        if tool_name == TOOL_ADD:
            return self._tool_add(client, scope, args)
        if tool_name == TOOL_FLUSH:
            return self._tool_flush(client, scope)
        return tool_error(f"Unknown tool: {tool_name}")

    def _tool_search(
        self,
        client: EverOSApiClient,
        scope: ScopeIds,
        args: dict[str, Any],
    ) -> str:
        query = args.get("query")
        if not isinstance(query, str) or not query:
            return tool_error("Missing required parameter: query")
        top_k = args.get("top_k") or _DEFAULT_TOOL_SEARCH_TOP_K
        if not isinstance(top_k, int):
            top_k = _DEFAULT_TOOL_SEARCH_TOP_K
        top_k = max(1, min(100, top_k))
        method = args.get("method") or _DEFAULT_SEARCH_METHOD
        owner = args.get("owner") or "user"
        user_id, agent_id, include_profile = (
            (self._user_id, None, True)
            if owner == "user"
            else (None, self._agent_id, False)
        )
        try:
            data = client.search(
                user_id,
                agent_id,
                scope.app_id,
                scope.project_id,
                query,
                top_k=top_k,
                method=method,
                include_profile=include_profile,
            )
        except EverOSClientError as exc:
            if self._is_transient(exc.code):
                self._record_failure()
            return tool_error(f"EverOS search failed: {exc}")
        return format_tool_result(
            {
                "results": data,
                "count": len(data.get("episodes") or []),
            }
        )

    def _tool_list(
        self,
        client: EverOSApiClient,
        scope: ScopeIds,
        args: dict[str, Any],
    ) -> str:
        memory_type = args.get("memory_type") or "episode"
        page = args.get("page") or 1
        if not isinstance(page, int) or page < 1:
            page = 1
        page_size = args.get("page_size") or 20
        if not isinstance(page_size, int):
            page_size = 20
        page_size = max(1, min(100, page_size))
        owner = args.get("owner") or "user"
        user_id, agent_id = (
            (self._user_id, None) if owner == "user" else (None, self._agent_id)
        )
        try:
            data = client.get(
                user_id,
                agent_id,
                scope.app_id,
                scope.project_id,
                memory_type,
                page=page,
                page_size=page_size,
            )
        except EverOSClientError as exc:
            if self._is_transient(exc.code):
                self._record_failure()
            return tool_error(f"EverOS list failed: {exc}")
        items = data.get(f"{memory_type}s") or []
        return format_tool_result(
            {
                "results": items,
                "total_count": data.get("total_count", 0),
                "count": len(items),
            }
        )

    def _tool_add(
        self,
        client: EverOSApiClient,
        scope: ScopeIds,
        args: dict[str, Any],
    ) -> str:
        content = args.get("content")
        if not isinstance(content, str) or not content:
            return tool_error("Missing required parameter: content")
        msg = format_memory_write_message(
            content, self._user_id, time.time_ns() // 1_000_000
        )
        try:
            client.add_messages(self._session_id, scope.app_id, scope.project_id, [msg])
            client.flush_session(self._session_id, scope.app_id, scope.project_id)
        except EverOSClientError as exc:
            if self._is_transient(exc.code):
                self._record_failure()
            return tool_error(f"EverOS add failed: {exc}")
        return format_tool_result({"result": "Fact stored."})

    def _tool_flush(self, client: EverOSApiClient, scope: ScopeIds) -> str:
        try:
            resp = client.flush_session(
                self._session_id, scope.app_id, scope.project_id
            )
        except EverOSClientError as exc:
            if self._is_transient(exc.code):
                self._record_failure()
            return tool_error(f"EverOS flush failed: {exc}")
        return format_tool_result({"status": resp.get("status")})

    # ── prompt / config ───────────────────────────────────────────────────

    def system_prompt_block(self) -> str:
        mode = (self._config or {}).get("mode", "oss")
        user = self._user_id or "unknown"
        return (
            "## EverOS Memory Active\n"
            f"- Mode: {mode}\n"
            f"- User: {user}\n"
            "- Call `everos_search` before answering context-dependent "
            "questions about the user or prior conversations.\n"
        )

    def get_config_schema(self) -> list[dict[str, Any]]:
        return [
            {
                "key": "api_url",
                "description": "EverOS API base URL.",
                "default": _DEFAULT_API_URL,
            },
            {
                "key": "mode",
                "description": "EverOS deployment mode.",
                "choices": ["platform", "oss"],
                "default": "oss",
            },
            {
                "key": "user_id",
                "description": "User identifier for memory scoping.",
                "default": "hermes-user",
            },
            {
                "key": "agent_id",
                "description": "Agent identifier for memory scoping.",
                "default": _DEFAULT_AGENT_ID,
            },
            {
                "key": "app_id",
                "description": "Application identifier for memory scoping.",
                "default": "default",
            },
            {
                "key": "project_id",
                "description": "Project identifier for memory scoping.",
                "default": "default",
            },
            {
                "key": "agent_track_enabled",
                "description": "Whether agent-scoped memory tracking is enabled.",
                "choices": ["true", "false"],
                "default": "false",
            },
        ]

    def save_config(self, values: dict[str, Any], hermes_home: str) -> None:
        path = Path(hermes_home) / "everos.json"
        existing: dict[str, Any] = {}
        if path.exists():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    existing = raw
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Failed to read existing %s: %s", path, exc)
        merged = {**existing, **values}
        atomic_json_write(path, merged, mode=0o600)

    def post_setup(self, hermes_home: str, config: dict[str, Any]) -> None:
        _run_setup(Path(hermes_home), config, interactive=True)

    def backup_paths(self) -> list[str]:
        root = Path.home() / ".everos"
        return [str(root)] if root.exists() else []

    # ── teardown ──────────────────────────────────────────────────────────

    def shutdown(self) -> None:
        if self._prefetch_thread is not None:
            self._prefetch_thread.join(timeout=5.0)
        if self._sync_thread is not None:
            self._sync_thread.join(timeout=5.0)
        if self._mirror_thread is not None and self._mirror_thread.is_alive():
            self._mirror_thread.join(timeout=5.0)
        self._shutdown_client()

    def _shutdown_client(self) -> None:
        client = self._client
        if client is not None:
            try:
                client.close()
            except Exception:
                logger.warning("Error closing EverOS client", exc_info=True)
        self._client = None

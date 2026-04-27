"""Agent Demo Helpers (v1 API)

Shared utilities for agent demo scripts (search_agent_demo, coding_agent_demo, etc.).
Provides:
- AgentDemoRunner: stateful helper for v1 API calls (send messages, fetch, search)
- Print helpers: stateless formatters for various memory types
"""

import uuid
from typing import List, Optional

import httpx
from common_utils.datetime_utils import get_now_with_timezone


DEFAULT_BASE_URL = "http://localhost:1995"


# ==================== Print Helpers ====================


def print_separator(text: str = ""):
    if text:
        print(f"\n{'='*60}")
        print(f"{text}")
        print('=' * 60)
    else:
        print('-' * 60)


def print_episodic_memories(memories: list):
    """Print episodic memories."""
    if not memories:
        print("  (none)")
        return
    for i, m in enumerate(memories, 1):
        print(f"\n  [{i}] {m.get('summary') or m.get('episode') or 'N/A'}")
        if m.get("keywords"):
            print(f"      Keywords : {', '.join(m['keywords'])}")
        if m.get("timestamp"):
            print(f"      Time     : {m['timestamp']}")


def print_event_logs(memories: list):
    """Print event log memories (atomic facts)."""
    if not memories:
        print("  (none)")
        return
    for i, m in enumerate(memories, 1):
        print(f"\n  [{i}] {m.get('atomic_fact', 'N/A')}")
        if m.get("timestamp"):
            print(f"      Time : {m['timestamp']}")


def print_foresights(memories: list):
    """Print foresight memories."""
    if not memories:
        print("  (none)")
        return
    for i, m in enumerate(memories, 1):
        content = m.get("content") or m.get("foresight") or "N/A"
        print(f"\n  [{i}] {content}")
        validity = " ~ ".join(filter(None, [m.get("start_time"), m.get("end_time")]))
        if validity:
            print(f"      Validity : {validity}")
        if m.get("evidence"):
            print(f"      Evidence : {m['evidence']}")


def print_agent_cases(memories: list):
    """Print agent experience memories."""
    if not memories:
        print("  (none)")
        return
    for i, exp in enumerate(memories, 1):
        print(f"\n  [{i}] {exp.get('task_intent', 'N/A')}")
        print(f"      Parent   : {exp.get('parent_id', 'N/A')}")
        approach = exp.get("approach", "")
        if approach:
            print(f"      Approach : {approach}")
        if exp.get("quality_score") is not None:
            print(f"      Quality  : {exp['quality_score']}")


def print_agent_skills(memories: list):
    """Print agent skills."""
    if not memories:
        print("  (none)")
        return

    for i, m in enumerate(memories, 1):
        print(f"\n  [{i}] {m.get('name') or 'Unnamed'}")
        if m.get("description"):
            print(f"      Description: {m['description']}")
        print(f"      Content    : {m.get('content', 'N/A')}")
        print(f"      Confidence : {m.get('confidence', 0):.2f}")
        print(f"      Cluster    : {m.get('cluster_id', 'N/A')}")


def print_search_case_results(hits: list):
    """Print search results for agent_case."""
    if not hits:
        print("  (no results)")
        return
    for i, h in enumerate(hits, 1):
        score = h.get("score", 0.0)
        task_intent = h.get("task_intent") or ""
        print(f"\n  [{i}] score={score:.4f}")
        print(f"      Intent   : {task_intent}")


def print_search_skill_results(hits: list):
    """Print search results for agent_skill."""
    if not hits:
        print("  (no results)")
        return
    for i, h in enumerate(hits, 1):
        score = h.get("score", 0.0)
        name = h.get("name") or "Unnamed"
        content = h.get("content") or ""
        print(f"\n  [{i}] score={score:.4f}  {name}")
        print(f"      {content}")
        if h.get("description"):
            print(f"      Description: {h['description']}")
        print(f"      Confidence : {h.get('confidence', 0.0):.2f}")


# Memory type -> (label, printer) mapping for fetch step
MEMORY_TYPE_PRINTERS = [
    ("episodic_memory",  "Episodic Memory",   print_episodic_memories),
    ("agent_case", "Agent Case",   print_agent_cases),
    ("agent_skill",      "Agent Skill",        print_agent_skills),
]


# ==================== AgentDemoRunner ====================


class AgentDemoRunner:
    """Stateful helper for running agent demo scripts (v1 API).

    Encapsulates user/session config and provides v1 API call methods.
    Each demo creates its own runner with unique session_id.

    Usage:
        runner = AgentDemoRunner(
            session_prefix="search_agent_demo",
            user_id="demo_user",
        )
        await runner.send_agent_message(msg, 0, flush=True)
    """

    def __init__(
        self,
        session_prefix: str = "agent_demo",
        user_id: str = "demo_user",
        msg_prefix: str = "agent_msg",
        base_url: str = DEFAULT_BASE_URL,
        # Kept for backward compat — old demos pass these but they are unused in v1
        group_id_prefix: str = "",
        group_name: str = "",
        description: str = "",  # noqa: ARG002
        tags: Optional[List[str]] = None,  # noqa: ARG002
    ):
        self.run_id = uuid.uuid4().hex[:8]
        self.session_id = f"{session_prefix}_{self.run_id}"
        # v1 auto-generates group_id from user_id, but demos may want to reference it
        self.group_id = f"{group_id_prefix or session_prefix}_{self.run_id}"
        self.group_name = group_name
        self.msg_prefix = msg_prefix
        self.user_id = user_id
        self.base_url = base_url

        self.agent_url = f"{base_url}/api/v1/memories/agent"
        self.flush_url = f"{base_url}/api/v1/memories/agent/flush"
        self.get_url = f"{base_url}/api/v1/memories/get"
        self.search_url = f"{base_url}/api/v1/memories/search"

    async def save_conversation_meta(self):
        """No-op in v1 — conversation meta is auto-created.

        Kept for backward compatibility with existing demo scripts.
        """
        print(f"  v1 API: group auto-registered (user_id={self.user_id}, session={self.session_id})")

    async def send_agent_message(
        self, msg: dict, msg_index: int, flush: bool = False
    ) -> bool:
        """Send a single agent message via POST /api/v1/memories/agent."""
        now = get_now_with_timezone()
        timestamp_ms = int(now.timestamp() * 1000)

        role = msg.get("role", "user")
        sender_id = self.user_id if role == "user" else "assistant"

        message_item = {
            "message_id": f"{self.msg_prefix}_{self.run_id}_{msg_index:03d}",
            "sender_id": sender_id,
            "sender_name": sender_id,
            "role": role,
            "timestamp": timestamp_ms,
            "content": msg.get("content") or "",
        }

        if msg.get("tool_calls"):
            message_item["tool_calls"] = msg["tool_calls"]
        if msg.get("tool_call_id"):
            message_item["tool_call_id"] = msg["tool_call_id"]

        payload = {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "messages": [message_item],
        }

        try:
            async with httpx.AsyncClient(timeout=500.0) as client:
                resp = await client.post(self.agent_url, json=payload)
                resp.raise_for_status()
                result = resp.json()

                data = result.get("data", {})
                status = data.get("status", "")
                role_label = f"[{role}]".ljust(12)
                raw_content = msg.get("content")
                if isinstance(raw_content, list):
                    content_preview = (raw_content[0].get("text", "") if raw_content else "")[:50]
                else:
                    content_preview = (raw_content or "(tool_calls)")[:50]
                if status == "extracted":
                    print(f"  {role_label} {content_preview}  -> Extracted memories")
                else:
                    print(f"  {role_label} {content_preview}")

                # Handle flush after message if requested
                if flush:
                    await self._flush()

                return True
        except httpx.ConnectError:
            print(f"  Cannot connect to API server ({self.base_url})")
            print(f"  Please start first: uv run python src/run.py")
            return False
        except Exception as e:
            print(f"  Error: {e}")
            return False

    async def _flush(self) -> bool:
        """Trigger flush via POST /api/v1/memories/agent/flush."""
        payload = {
            "user_id": self.user_id,
            "session_id": self.session_id,
        }
        try:
            async with httpx.AsyncClient(timeout=500.0) as client:
                resp = await client.post(self.flush_url, json=payload)
                resp.raise_for_status()
                result = resp.json()
                data = result.get("data", {})
                if data.get("status") == "extracted":
                    print(f"  [flush]      -> Extracted memories")
                else:
                    print(f"  [flush]      -> {data.get('status', 'done')}")
                return True
        except Exception as e:
            print(f"  Flush error: {e}")
            return False

    async def fetch_memories(self, memory_type: str) -> list:
        """Fetch memories via POST /api/v1/memories/get."""
        payload = {
            "memory_type": memory_type,
            "page": 1,
            "page_size": 20,
            "rank_by": "timestamp",
            "rank_order": "desc",
            "filters": {
                "user_id": self.user_id,
            },
        }
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(self.get_url, json=payload)
                resp.raise_for_status()
                result = resp.json()
                data = result.get("data", {})
                # v1 GetMemResponse has typed arrays: episodes, profiles, agent_cases, agent_skills
                if memory_type == "episodic_memory":
                    return data.get("episodes", [])
                elif memory_type == "profile":
                    return data.get("profiles", [])
                elif memory_type == "agent_case":
                    return data.get("agent_cases", [])
                elif memory_type == "agent_skill":
                    return data.get("agent_skills", [])
                else:
                    return []
        except Exception as e:
            print(f"  [{memory_type}] Fetch error: {e}")
            return []

    async def search_memories(
        self,
        query: str,
        memory_type: str,
        top_k: int = 5,
        retrieve_method: str = "hybrid",
    ) -> list | dict:
        """Search memories via POST /api/v1/memories/search.

        Args:
            query: Search query text.
            memory_type: One of "agent_memory", "episodic_memory", "profile", "raw_message".
            top_k: Max results.
            retrieve_method: Retrieval method.

        Returns:
            For agent_memory: dict with "cases" and "skills" lists.
            For other types: list of results.
        """
        payload = {
            "query": query,
            "method": retrieve_method,
            "memory_types": [memory_type],
            "top_k": top_k,
            "filters": {
                "user_id": self.user_id,
            },
        }
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(self.search_url, json=payload)
                resp.raise_for_status()
                result = resp.json()
                data = result.get("data", {})

                if memory_type == "agent_memory":
                    agent_mem = data.get("agent_memory") or {}
                    return {
                        "cases": agent_mem.get("cases", []),
                        "skills": agent_mem.get("skills", []),
                    }
                elif memory_type == "episodic_memory":
                    return data.get("episodes", [])
                else:
                    return data.get("memories", [])
        except Exception as e:
            print(f"  Search error: {e}")
            return [] if memory_type != "agent_memory" else {"cases": [], "skills": []}

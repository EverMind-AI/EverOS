"""
EverMemClient - Thin wrapper around the EverMemOS REST API.

Core endpoints:
    POST   /api/v1/memories          → Store a message (MemorizeMessageRequest format)
    GET    /api/v1/memories          → Fetch memories by type
    GET    /api/v1/memories/search   → Search memories
    DELETE /api/v1/memories          → Delete memories

All requests use Content-Type: application/json.
GET requests use JSON body (server merges with query params).
"""

import os
import uuid
import requests
from datetime import datetime, timezone
from typing import Any


class EverMemClient:
    """
    Thin wrapper around the EverMemOS REST API.

    Key parameters:
        retrieve_method : keyword | vector | hybrid | rrf | agentic
        memory_types    : episodic_memory | foresight | event_log (profile NOT on search)
        role            : user | assistant
    """

    def __init__(self):
        self.base_url = os.getenv("EVERMEM_BASE_URL", "http://localhost:1995")
        self.user_id = os.getenv("EVERMEM_USER_ID", "ap_agent_system")
        self.group_id = os.getenv("EVERMEM_GROUP_ID")  # For group-scoped memories (seed data)
        self.headers = {"Content-Type": "application/json"}

    def write_memory(
        self,
        content: str,
        role: str = "assistant",
        memory_type: str = "episodic_memory",
        metadata: dict | None = None,
        *,
        group_id: str | None = None,
        group_name: str | None = None,
        sender: str | None = None,
        sync_mode: bool = False,
    ) -> dict:
        """
        Store a memory in EverMemOS.

        EverMemOS expects MemorizeMessageRequest format. memory_type and metadata
        are for internal use only — fold context into content. The system extracts
        memories from the message content.

        role: "assistant" for agent-generated memories, "user" for user inputs.
        sync_mode: If True, add ?sync_mode=false to wait for extraction (slower but ensures memories are indexed).
        """
        del memory_type, metadata  # Not sent to API; structure content instead
        effective_sender = sender or self.user_id
        payload = {
            "message_id": f"ap_{uuid.uuid4().hex[:12]}",
            "create_time": datetime.now(timezone.utc).isoformat(),
            "sender": effective_sender,
            "content": content,
            "role": role,
        }
        if group_id:
            payload["group_id"] = group_id
        if group_name:
            payload["group_name"] = group_name

        url = f"{self.base_url}/api/v1/memories"
        if sync_mode:
            url += "?sync_mode=false"

        response = requests.post(url, json=payload, headers=self.headers, timeout=120)
        response.raise_for_status()
        return response.json()

    def write_vendor_profile(self, vendor: str, profile_content: str) -> dict:
        """
        Write or update stable vendor profile. Folds vendor into content
        since EverMemOS has no metadata field on MemorizeMessageRequest.
        """
        content = f"[Vendor profile for {vendor}] {profile_content}"
        return self.write_memory(
            content=content,
            role="assistant",
            memory_type="profile",
        )

    def search_memories(
        self,
        query: str,
        retrieve_method: str = "agentic",
        memory_types: list[str] | None = None,
        top_k: int = 10,
    ) -> list[dict]:
        """
        Search vendor memory. memory_types: episodic_memory | foresight | event_log.
        Profile is NOT supported on /search — use fetch_memories_by_type for profile.
        """
        if memory_types is None:
            memory_types = ["episodic_memory", "event_log"]
        payload = {
            "query": query,
            "retrieve_method": retrieve_method,
            "memory_types": memory_types,
            "top_k": top_k,
        }
        if self.group_id:
            payload["group_id"] = self.group_id
        else:
            payload["user_id"] = self.user_id
        response = requests.get(
            f"{self.base_url}/api/v1/memories/search",
            json=payload,
            headers=self.headers,
        )
        response.raise_for_status()
        data = response.json()
        raw = data.get("result", data)
        memories = raw.get("memories", raw.get("results", []))
        return self._flatten_search_results(memories)

    def _flatten_search_results(self, memories: Any) -> list[dict]:
        """Flatten nested {group_id: [records]} into a single list."""
        if not memories or not isinstance(memories, list):
            return []
        flat: list[dict] = []
        for item in memories:
            if isinstance(item, dict):
                for group_id, records in item.items():
                    if isinstance(records, list):
                        for r in records:
                            rec = dict(r) if isinstance(r, dict) else {"content": str(r)}
                            rec.setdefault("group_id", group_id)
                            flat.append(rec)
                        break
                else:
                    flat.append(item)
        return flat if flat else list(memories)

    def fetch_memories_by_type(
        self,
        memory_types: list[str] | None = None,
        top_k: int = 20,
    ) -> list[dict]:
        """
        Fetch memories by type (profile, episodic_memory, etc.).
        EverMemOS uses memory_type (singular) and limit per request.
        """
        if memory_types is None:
            memory_types = ["episodic_memory"]
        all_memories: list[dict] = []
        for memory_type in memory_types:
            payload = {
                "memory_type": memory_type,
                "limit": top_k,
                "offset": 0,
            }
            if self.group_id:
                payload["group_id"] = self.group_id
            else:
                payload["user_id"] = self.user_id
            response = requests.get(
                f"{self.base_url}/api/v1/memories",
                json=payload,
                headers=self.headers,
            )
            response.raise_for_status()
            data = response.json()
            raw = data.get("result", data)
            memories = raw.get("memories", raw.get("results", []))
            if isinstance(memories, list):
                for m in memories:
                    rec = dict(m) if isinstance(m, dict) else {"content": str(m)}
                    rec.setdefault("memory_type", memory_type)
                    all_memories.append(rec)
        return all_memories

    def format_memory_context(self, memories: list[dict]) -> str:
        """Convert raw memory results into a clean string for Claude's context."""
        if not memories:
            return "No prior history found for this vendor."
        lines = []
        for i, mem in enumerate(memories, 1):
            content = (
                mem.get("episode")
                or mem.get("content")
                or mem.get("memory")
                or mem.get("summary")
                or mem.get("atomic_fact")
                or mem.get("subject")
                or str(mem)
            )
            if isinstance(content, dict):
                content = str(content)
            profile_data = mem.get("profile_data")
            if profile_data and isinstance(profile_data, dict):
                content = str(profile_data) if not content else content
            mem_type = mem.get("memory_type", mem.get("type", "memory"))
            timestamp = mem.get("created_at", mem.get("timestamp", "unknown date"))
            lines.append(f"[{str(mem_type).upper()} {i} | {timestamp}]: {content}")
        return "\n".join(lines)

    def delete_memories(self, user_id: str | None = None, group_id: str | None = None) -> dict:
        """Delete memories. Uses group_id if set, else user_id."""
        payload: dict[str, str] = {}
        if group_id or self.group_id:
            payload["group_id"] = group_id or self.group_id or ""
        else:
            payload["user_id"] = user_id or self.user_id
        response = requests.delete(
            f"{self.base_url}/api/v1/memories",
            json=payload,
            headers=self.headers,
        )
        response.raise_for_status()
        return response.json()

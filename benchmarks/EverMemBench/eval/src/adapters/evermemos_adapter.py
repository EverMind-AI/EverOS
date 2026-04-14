"""
EverMemOS (EverMind) Adapter for multi-person group chat evaluation.

Implements Add functionality using EverMemOS Cloud API:
POST /api/v1/memories

Docs reference:
https://docs.evermind.ai/api-reference/endpoint/add_memories

We ingest *single messages* (EverMemOS endpoint is single-message payload).
To preserve multi-person group chat information we map:
- role: always "user" (dataset is human-only)
- sender: speaker id (we use speaker name)
- sender_name: speaker name
- group_id: simplified format `${user_id}_${groupId}` (e.g. 004_1)
- create_time: message timestamp (ISO 8601)
- message_id: global counter (msg_00001, msg_00002, ...)

Processing:
- Iterate day->group, never mix across day/group.
- Within each group/day, send messages sequentially (with optional add_interval delay).
"""

import asyncio
import re
from datetime import timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import time

import aiohttp
from aiolimiter import AsyncLimiter

from eval.src.adapters.base import BaseAdapter
from eval.src.core.data_models import Dataset, GroupChatMessage, AddResult, SearchResult
from eval.src.utils.logger import get_console, print_success, print_warning


class EverMemosAdapter(BaseAdapter):
    """
    EverMemOS adapter.

    Config example:
    ```yaml
    name: "evermemos"
    base_url: "${EVERMEMOS_BASE_URL:https://api.evermind.ai}"
    api_key: "${EVERMEMOS_API_KEY}"
    max_retries: 5
    requests_per_second: 5
    add_interval: 0.0
    ```
    """

    def __init__(self, config: Dict[str, Any], output_dir: Optional[Path] = None):
        super().__init__(config, output_dir)

        self.base_url = (config.get("base_url") or "").rstrip("/")
        if not self.base_url:
            raise ValueError("EverMemOS base_url is required. Set 'base_url' in config or EVERMEMOS_BASE_URL env var.")

        api_key = config.get("api_key", "")
        
        # API key is optional for local deployments
        self.headers = {
            "Content-Type": "application/json",
        }
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

        self.max_retries = config.get("max_retries", 5)
        self.add_interval = config.get("add_interval", 0.0)

        rps = config.get("requests_per_second", 5)
        self.rate_limiter = AsyncLimiter(max_rate=rps, time_period=1.0)

        # Search configuration
        self.search_config = config.get("search", {})

        # HTTP configuration
        self.http_timeout = config.get("http_timeout", 60)

        self._session: Optional[aiohttp.ClientSession] = None
        self.console = get_console()
        
        # Global message counter
        self._message_counter = 0

        self.console.print("✅ EverMemosAdapter initialized", style="bold green")
        self.console.print(f"   Base URL: {self.base_url}")
        self.console.print(f"   Rate Limit: {rps} req/s")

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.http_timeout)
            self._session = aiohttp.ClientSession(headers=self.headers, timeout=timeout)
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def add(
        self,
        dataset: Dataset,
        user_id: str,
        days_to_process: Optional[List[str]] = None,
        **kwargs,
    ) -> AddResult:
        self.console.print(f"\n{'='*60}", style="bold cyan")
        self.console.print("Stage: Add (EverMemOS)", style="bold cyan")
        self.console.print(f"{'='*60}", style="bold cyan")
        self.console.print(f"User ID (logical): {user_id}")
        self.console.print(f"Dataset: {dataset.name}")

        if days_to_process:
            days = [d for d in dataset.days if d.date in days_to_process]
        else:
            days = dataset.days

        self.console.print(f"Days to process: {len(days)}")

        total_messages = 0
        total_errors: List[str] = []

        for day in days:
            self.console.print(f"\n📅 Processing {day.date}...", style="dim")
            for group_name, messages in day.groups.items():
                self.console.print(f"   👥 Group: {group_name}", style="dim")
                self.console.print(f"      Messages: {len(messages)}")

                for idx, msg in enumerate(messages):
                    try:
                        # Pass user_id to format message for group_id construction
                        payload = self._format_message(msg, user_id=user_id, group_name=group_name)
                        await self._send_one(payload)
                        total_messages += 1
                    except Exception as e:
                        error_msg = f"[{day.date}][{group_name}] msg#{idx} failed: {e}"
                        total_errors.append(error_msg)
                        self.console.print(f"      ❌ {error_msg}", style="red")

                    if self.add_interval and self.add_interval > 0:
                        await asyncio.sleep(self.add_interval)

        success = len(total_errors) == 0
        self.console.print(f"\n{'='*60}", style="bold cyan")
        if success:
            print_success(f"Add completed: {total_messages} messages sent")
        else:
            print_warning(f"Add completed with errors: {total_messages} messages, {len(total_errors)} errors")

        return AddResult(
            success=success,
            days_processed=len(days),
            messages_sent=total_messages,
            errors=total_errors,
            metadata={"dataset": dataset.name, "user_id": user_id},
        )

    def _format_message(
        self,
        msg: GroupChatMessage,
        user_id: str,
        group_name: str,
    ) -> Dict[str, Any]:
        ts = msg.timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        # EverMemOS expects ISO 8601 string with timezone offset
        create_time = ts.isoformat()

        # Simplified group id: ${user_id}_${groupId} (e.g. 004_1)
        m = re.match(r"^\s*Group\s*(\d+)\s*$", group_name, flags=re.IGNORECASE)
        group_suffix = m.group(1) if m else re.sub(r"\s+", "_", group_name.strip()) or "group"
        group_id = f"{user_id}_{group_suffix}"

        # Global counter message_id: msg_00001
        self._message_counter += 1
        message_id = f"msg_{self._message_counter:05d}"

        return {
            "content": msg.content,
            "create_time": create_time,
            "group_id": group_id,
            "message_id": message_id,
            "refer_list": None,
            "role": "user",
            "sender": msg.speaker,
            "sender_name": msg.speaker,
            "sync": False
        }

    async def _send_one(self, payload: Dict[str, Any]) -> None:
        session = await self._get_session()
        url = f"{self.base_url}/api/v1/memories"

        for attempt in range(self.max_retries):
            try:
                async with self.rate_limiter:
                    async with session.post(url, json=payload) as resp:
                        text = await resp.text()
                        # Cloud may return 200 OK, local deployment may return 202 Accepted
                        # (queued/processing in background). 202 is not an error.
                        if resp.status not in (200, 202):
                            raise Exception(f"HTTP {resp.status}: {text}")

                        if resp.status == 202:
                            # Typical local response:
                            # {"message":"Request accepted, processing in background","request_id":"..."}
                            # We treat it as success and don't poll (no stable status endpoint assumed here).
                            try:
                                data = await resp.json()
                                req_id = data.get("request_id")
                                if req_id:
                                    self.console.print(
                                        f"      ⏳ Accepted (202), processing in background (request_id={req_id})",
                                        style="dim"
                                    )
                            except Exception:
                                # Best-effort logging only; still treat as success.
                                pass
                        return
            except Exception as e:
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise
    
    async def search(
        self,
        query: str,
        user_id: str,
        top_k: int = 10,
        **kwargs
    ) -> SearchResult:
        """
        Search episodic memories in EverMemOS.

        Uses GET /api/v1/memories/search with JSON body.
        Searches all groups for the given user.

        Args:
            query: Search query (usually question text)
            user_id: User ID for EverMemOS
            top_k: Number of memories to retrieve
            **kwargs: Additional parameters:
                - retrieve_method: "keyword", "vector", "hybrid", "agentic" (default: "hybrid")

        Returns:
            SearchResult with retrieved memories and formatted context
        """
        start_time = time.time()

        session = await self._get_session()

        # Read search params from config, allow kwargs override
        effective_top_k = top_k if top_k is not None else self.search_config.get("top_k", 10)
        retrieve_method = kwargs.get(
            "retrieve_method",
            self.search_config.get("retrieve_method", "hybrid")
        )

        raw_memories = await self._search_episodic(
            session, query, effective_top_k, retrieve_method
        )

        # Format memories
        memory_strings = []
        for mem in raw_memories:
            formatted = self._format_memory(mem)
            memory_strings.append(formatted)

        # Build context
        context = self._format_search_context(memory_strings)

        duration_ms = (time.time() - start_time) * 1000

        return SearchResult(
            question_id=kwargs.get("question_id", ""),
            query=query,
            retrieved_memories=memory_strings,
            context=context,
            search_duration_ms=duration_ms,
            metadata={
                "user_id": user_id,
                "retrieve_method": retrieve_method,
                "total_memories": len(raw_memories),
            }
        )

    async def _search_episodic(
        self,
        session: aiohttp.ClientSession,
        query: str,
        top_k: int,
        retrieve_method: str,
    ) -> List[Dict[str, Any]]:
        """
        Search episodic memories.

        Args:
            session: aiohttp session
            query: Search query
            top_k: Number of results to retrieve
            retrieve_method: Search method ("keyword", "vector", "hybrid", "agentic")

        Returns:
            List of episodic memory dicts with: timestamp, group_id, content
        """
        url = f"{self.base_url}/api/v1/memories/search"

        payload = {
            "memory_types": ["episodic_memory"],
            "query": query,
            "retrieve_method": retrieve_method,
            "top_k": top_k,
            "user_id": "",
            "include_metadata": False,
        }

        memories = []

        for attempt in range(self.max_retries):
            try:
                async with self.rate_limiter:
                    async with session.request("GET", url, json=payload) as resp:
                        if resp.status == 200:
                            result = await resp.json()

                            if result.get("status") != "ok":
                                self.console.print(
                                    f"      ⚠️  Search API returned non-ok status: {result.get('status')}",
                                    style="yellow"
                                )

                            memories_list = result.get("result", {}).get("memories", [])

                            for group_memories in memories_list:
                                if isinstance(group_memories, dict):
                                    for group_id, mem_list in group_memories.items():
                                        for mem in mem_list:
                                            if isinstance(mem, dict):
                                                if mem.get("memory_type", "") != "episodic_memory":
                                                    continue

                                                timestamp = mem.get("timestamp", "")
                                                episode = mem.get("episode", "")
                                                summary = mem.get("summary", "")
                                                subject = mem.get("subject", "")

                                                # Prefer episode (more complete), fallback to summary, then subject
                                                content = episode or summary or subject
                                                if content:
                                                    memories.append({
                                                        "timestamp": timestamp,
                                                        "group_id": group_id,
                                                        "content": content,
                                                    })
                        elif resp.status == 404:
                            self.console.print(
                                f"      ⚠️  Memories not found",
                                style="yellow"
                            )
                        else:
                            text = await resp.text()
                            self.console.print(
                                f"      ⚠️  Search error: HTTP {resp.status}: {text[:200]}",
                                style="yellow"
                            )
                break
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    self.console.print(
                        f"      ⚠️  Search retry {attempt + 1}/{self.max_retries} in {wait_time}s: {e}",
                        style="yellow"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    raise

        return memories

    def _format_memory(self, mem: Dict[str, Any]) -> str:
        """
        Format a single episodic memory.

        Args:
            mem: Memory dict with timestamp, group_id, content

        Returns:
            Formatted memory string
        """
        content = mem.get("content", "")
        timestamp = mem.get("timestamp", "")
        group_id = mem.get("group_id", "")
        if timestamp and group_id:
            return f"[{timestamp}][Group: {group_id}] {content}"
        elif timestamp:
            return f"[{timestamp}] {content}"
        return content

    def _format_search_context(
        self,
        memories: List[str]
    ) -> str:
        """
        Format retrieved memories into context string for LLM.

        Args:
            memories: List of memory strings

        Returns:
            Formatted context string with [MEMORIES] section
        """
        if not memories:
            return "(No memories retrieved)"

        return "\n".join(f"- {mem}" for mem in memories)



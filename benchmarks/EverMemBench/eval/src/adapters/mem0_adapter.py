"""
Mem0 Adapter for multi-person group chat evaluation.

Implements Add functionality for Mem0 memory system.
Uses Mem0 "Group Chat" feature by providing `name` per message so Mem0 can
attribute memories to individual speakers.

All messages are sent with role="user" and formatted as:
- name: "<Speaker>"
- content: "content" (no group prefix needed; group is encoded in run_id)
"""
import asyncio
import re
from datetime import timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import time

from eval.src.adapters.base import BaseAdapter
from eval.src.core.data_models import Dataset, GroupChatMessage, AddResult, SearchResult
from eval.src.utils.logger import get_console, print_success, print_warning


class Mem0Adapter(BaseAdapter):
    """
    Mem0 memory system adapter for multi-person group chat.
    
    Formats all messages as:
    - role: "user" (always)
    - name: speaker name (Mem0 group chat attribution)
    - content: "original_content" (no group prefix; run_id identifies group session)
    
    Timestamp support:
    - Mem0 API supports `timestamp` (Unix seconds) on add().
    - For efficiency, we send one API call per batch and set timestamp to the FIRST message's
      timestamp in that batch.
    
    Config example:
    ```yaml
    name: "mem0"
    api_key: "${MEM0_API_KEY}"
    batch_size: 5
    max_retries: 5
    ```
    """
    
    def __init__(self, config: Dict[str, Any], output_dir: Optional[Path] = None):
        super().__init__(config, output_dir)
        
        # Import Mem0 async client
        try:
            from mem0 import AsyncMemoryClient
        except ImportError:
            raise ImportError(
                "Mem0 client not installed. "
                "Please install: pip install mem0ai"
            )
        
        # API configuration
        api_key = config.get("api_key", "")
        if not api_key:
            raise ValueError("Mem0 API key is required. Set 'api_key' in config or MEM0_API_KEY env var.")
        
        self.client = AsyncMemoryClient(api_key=api_key)
        
        # Batch configuration
        self.batch_size = config.get("batch_size", 5)
        self.max_retries = config.get("max_retries", 5)
        self.add_interval = config.get("add_interval", 0.5)

        # Graph configuration (Mem0 Knowledge Graph)
        # Keep defaults False for backwards compatibility.
        self.enable_graph_add = bool(config.get("enable_graph_add", False))
        self.enable_graph_search = bool(config.get("enable_graph_search", False))

        # Search configuration
        self.search_config = config.get("search", {})
        
        self.console = get_console()
        
        self.console.print("✅ Mem0Adapter initialized", style="bold green")
        self.console.print(f"   Batch Size: {self.batch_size}")
        self.console.print(f"   Add Interval: {self.add_interval}s")
        self.console.print(f"   Enable Graph (add): {self.enable_graph_add}")
        self.console.print(f"   Enable Graph (search): {self.enable_graph_search}")
    
    async def add(
        self,
        dataset: Dataset,
        user_id: str,
        days_to_process: Optional[List[str]] = None,
        **kwargs
    ) -> AddResult:
        """
        Add dataset to Mem0 memory system.
        
        Args:
            dataset: Dataset with group chat data
            user_id: User ID for Mem0
            days_to_process: Optional list of dates to process (None = all)
            **kwargs: Additional parameters
            
        Returns:
            AddResult with statistics
        """
        self.console.print(f"\n{'='*60}", style="bold cyan")
        self.console.print("Stage: Add (Mem0)", style="bold cyan")
        self.console.print(f"{'='*60}", style="bold cyan")
        self.console.print(f"User ID: {user_id}")
        self.console.print(f"Dataset: {dataset.name}")
        
        # Determine which days to process
        if days_to_process:
            days = [d for d in dataset.days if d.date in days_to_process]
        else:
            days = dataset.days
        
        self.console.print(f"Days to process: {len(days)}")
        
        total_messages = 0
        total_errors = []
        
        # Process each day
        for day in days:
            self.console.print(f"\n📅 Processing {day.date}...", style="dim")

            # Process each group separately to avoid cross-group batching
            for group_name, messages in day.groups.items():
                self.console.print(f"   👥 Group: {group_name}", style="dim")

                run_id = self._make_run_id(user_id=user_id, group_name=group_name)

                if self.enable_graph_add:
                    group_messages = [self._format_graph_message(m) for m in messages]
                    self.console.print(f"      Messages: {len(group_messages)}")
                    self.console.print("      Graph mode: send one message per request", style="dim")

                    for msg_idx, message in enumerate(group_messages):
                        try:
                            await self._send_graph_message(message, run_id=run_id)
                            total_messages += 1
                            self.console.print(
                                f"      ✅ Message {msg_idx + 1}/{len(group_messages)} sent",
                                style="dim green",
                            )
                        except Exception as e:
                            error_msg = f"[{day.date}][{group_name}] Message {msg_idx + 1} failed: {e}"
                            total_errors.append(error_msg)
                            self.console.print(f"      ❌ {error_msg}", style="red")
                else:
                    group_messages = [self._format_message(m) for m in messages]
                    self.console.print(f"      Messages: {len(group_messages)}")

                    batches = self._split_into_batches(group_messages)
                    self.console.print(f"      Batches: {len(batches)}")

                    for batch_idx, batch in enumerate(batches):
                        try:
                            await self._send_batch(batch, run_id=run_id)
                            total_messages += len(batch)
                            self.console.print(
                                f"      ✅ Batch {batch_idx + 1}/{len(batches)} sent ({len(batch)} messages)",
                                style="dim green",
                            )
                        except Exception as e:
                            error_msg = f"[{day.date}][{group_name}] Batch {batch_idx + 1} failed: {e}"
                            total_errors.append(error_msg)
                            self.console.print(f"      ❌ {error_msg}", style="red")
        
        # Summary
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
            metadata={
                "user_id": user_id,
                "dataset": dataset.name,
            }
        )
    
    def _make_run_id(self, user_id: str, group_name: str) -> str:
        """
        Build a stable run_id for Mem0 group chat sessions.

        Per user request, run_id should be `${user_id}_${groupId}` (e.g. `004_1`).
        Group ids are stable within a dataset batch, so we don't need to embed day or dataset.
        """
        # Typical dataset group names look like "Group 1" / "Group 2".
        m = re.match(r"^\s*Group\s*(\d+)\s*$", group_name, flags=re.IGNORECASE)
        group_id = m.group(1) if m else re.sub(r"\s+", "_", group_name.strip()) or "group"
        return f"{user_id}_{group_id}"

    def _format_message(self, msg: GroupChatMessage) -> Dict[str, Any]:
        """
        Format a GroupChatMessage for Mem0 API.
        
        Input: GroupChatMessage with speaker, content, timestamp, group
        Output: {
            "role": "user",
            "name": "Alice",
            "content": "content",
            "__timestamp": 1721577600
        }
        """
        # Mem0 group chat: use `name` for speaker attribution
        formatted_content = msg.content
        ts = msg.timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        # Mem0 expects Unix seconds (UTC) at request-level, not per-message field.
        # We keep it alongside the message dict under a private key and pass it in add(timestamp=...).
        timestamp_unix = int(ts.timestamp())

        return {"role": "user", "name": msg.speaker, "content": formatted_content, "__timestamp": timestamp_unix}

    def _format_graph_message(self, msg: GroupChatMessage) -> Dict[str, Any]:
        """
        Format a GroupChatMessage for Mem0 graph add.

        Graph mode requires:
        - one message per request
        - user_id set to the speaker (no name field in message)
        """
        formatted_content = msg.content
        ts = msg.timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        timestamp_unix = int(ts.timestamp())

        return {
            "user_id": msg.speaker,
            "content": formatted_content,
            "__timestamp": timestamp_unix,
        }
    
    def _split_into_batches(self, messages: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Split messages into batches.
        
        Args:
            messages: List of formatted messages
            
        Returns:
            List of batches
        """
        batches = []
        for i in range(0, len(messages), self.batch_size):
            batches.append(messages[i:i + self.batch_size])
        return batches
    
    async def _send_batch(self, messages: List[Dict[str, Any]], run_id: str):
        """
        Send a batch of messages to Mem0 API.
        
        Args:
            messages: List of formatted messages
            run_id: Group chat session identifier for Mem0
            
        Raises:
            Exception: If API call fails after retries
        """
        if not messages:
            return

        # Use the FIRST message timestamp as the batch timestamp.
        batch_timestamp_unix = messages[0].get("__timestamp")
        payload_messages = [{"role": m["role"], "name": m.get("name"), "content": m["content"]} for m in messages]

        for attempt in range(self.max_retries):
            try:
                await self.client.add(
                    messages=payload_messages,
                    run_id=run_id,
                    timestamp=batch_timestamp_unix,
                    enable_graph=self.enable_graph_add,
                )

                # Add interval between batches
                if self.add_interval > 0:
                    await asyncio.sleep(self.add_interval)

                return  # Success
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    self.console.print(
                        f"      ⚠️  Retry {attempt + 1}/{self.max_retries} in {wait_time}s: {e}",
                        style="yellow",
                    )
                    await asyncio.sleep(wait_time)
                else:
                    raise

    async def _send_graph_message(self, message: Dict[str, Any], run_id: str):
        """
        Send a single message to Mem0 in graph mode.
        """
        if not message:
            return

        payload_messages = [{"role": "user", "content": message["content"]}]
        message_timestamp_unix = message.get("__timestamp")
        message_user_id = message.get("user_id")

        for attempt in range(self.max_retries):
            try:
                await self.client.add(
                    messages=payload_messages,
                    run_id=run_id,
                    user_id=message_user_id,
                    timestamp=message_timestamp_unix,
                    enable_graph=True,
                )

                if self.add_interval > 0:
                    await asyncio.sleep(self.add_interval)

                return
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    self.console.print(
                        f"      ⚠️  Retry {attempt + 1}/{self.max_retries} in {wait_time}s: {e}",
                        style="yellow",
                    )
                    await asyncio.sleep(wait_time)
                else:
                    raise
    
    async def close(self):
        """Close client resources."""
        pass  # Mem0 client doesn't require explicit cleanup
    
    async def search(
        self,
        query: str,
        user_id: str,
        top_k: int = 10,
        **kwargs
    ) -> SearchResult:
        """
        Search memories in Mem0.
        
        Uses client.search() with run_id filters to search across all groups
        for a given user batch. For example, batch 004 uses filters:
        {"AND": [{"user_id": "*"}, {"run_id": {"in": ["004_1", "004_2", "004_3"]}}]}
        
        Args:
            query: Search query (usually question text)
            user_id: User ID (batch ID like "004")
            top_k: Number of memories to retrieve
            **kwargs: Additional parameters:
                - group_ids: List of group IDs to search (e.g., ["1", "2", "3"])
                - run_ids: Explicit list of run_ids to filter
                
        Returns:
            SearchResult with retrieved memories and formatted context
        """
        start_time = time.time()

        # Read search params from config, allow kwargs override
        effective_top_k = top_k if top_k is not None else self.search_config.get("top_k", 10)

        # Build run_id filter based on user_id and group_ids
        run_ids = kwargs.get("run_ids")
        if not run_ids:
            # Default: search groups 1, 2, 3 for the user_id
            group_ids = kwargs.get(
                "group_ids",
                self.search_config.get("group_ids", ["1", "2", "3"])
            )
            run_ids = [f"{user_id}_{gid}" for gid in group_ids]
        
        # Construct filters for Mem0 search
        filters = {
            "AND": [
                {"user_id": "*"},
                {"run_id": {"in": run_ids}}
            ]
        }
        
        for attempt in range(self.max_retries):
            try:
                results = await self.client.search(
                    query=query,
                    top_k=effective_top_k,
                    filters=filters,
                    version="v2",
                    enable_graph=self.enable_graph_search,
                )
                
                # Parse results
                memories = []
                memory_details = []
                
                for mem in results.get("results", []):
                    memory_text = mem.get("memory", "")
                    attrs = mem.get("structured_attributes", {})

                    memories.append(memory_text)
                    memory_details.append({
                        "memory": memory_text,
                        "timestamp": mem.get("created_at", ""),
                        "score": round(mem.get("score", 0), 2),
                        "group_id": mem.get("session_id", ""),
                        "user": mem.get("user_id", ""),
                        "day_of_week": attrs.get("day_of_week", ""),
                    })
                
                # Format context for LLM
                context = self._format_search_context(user_id, memory_details)
                
                duration_ms = (time.time() - start_time) * 1000
                
                return SearchResult(
                    question_id=kwargs.get("question_id", ""),
                    query=query,
                    retrieved_memories=memories,
                    context=context,
                    search_duration_ms=duration_ms,
                    metadata={
                        "filters": filters,
                        "memory_details": memory_details,
                    }
                )
                
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    self.console.print(
                        f"      ⚠️  Search retry {attempt + 1}/{self.max_retries} in {wait_time}s: {e}",
                        style="yellow",
                    )
                    await asyncio.sleep(wait_time)
                else:
                    raise
    
    def _format_search_context(
        self,
        user_id: str,
        memory_details: List[Dict[str, Any]]
    ) -> str:
        """
        Format retrieved memories into context string for LLM.
        
        Args:
            user_id: User identifier
            memory_details: List of memory dicts with memory, timestamp, score
            
        Returns:
            Formatted context string
        """
        if not memory_details:
            return "(No memories retrieved)"

        lines = []
        for item in memory_details:
            timestamp = item.get("timestamp", "")
            memory = item.get("memory", "")
            group_id = item.get("group_id", "")
            day_of_week = item.get("day_of_week", "")
            user = item.get("user", "")
            if timestamp:
                lines.append(f"- [Group: {group_id}] [Day: {day_of_week}] [User: {user}] {timestamp}: {memory}")
            else:
                lines.append(f"- [Group: {group_id}][User: {user}] {memory}")

        return "\n".join(lines)


"""
Mem0 Adapter - adapt Mem0 online API for evaluation framework.
Reference: https://mem0.ai/

Key features:
- Dual-perspective handling: separate storage and retrieval for speaker_a and speaker_b
- Supports custom instructions
"""

import asyncio
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List

from rich.console import Console

from evaluation.src.adapters.online_base import OnlineAPIAdapter
from evaluation.src.adapters.registry import register_adapter
from evaluation.src.core.data_models import Conversation, SearchResult


@register_adapter("mem0")
class Mem0Adapter(OnlineAPIAdapter):
    """
    Mem0 online API adapter.

    Supports:
    - Standard memory storage and retrieval

    Config example:
    ```yaml
    adapter: "mem0"
    api_key: "${MEM0_API_KEY}"
    batch_size: 2
    display_timezone_offset: 8  # Optional: convert timestamps to UTC+8 for display
    ```
    """

    def __init__(self, config: dict, output_dir: Path = None):
        super().__init__(config, output_dir)

        # Import Mem0 async client
        try:
            from mem0 import AsyncMemoryClient
        except ImportError:
            raise ImportError(
                "Mem0 client not installed. " "Please install: pip install mem0ai"
            )

        # Initialize Mem0 async client
        api_key = config.get("api_key", "")
        if not api_key:
            raise ValueError("Mem0 API key is required. Set 'api_key' in config.")

        self.client = AsyncMemoryClient(api_key=api_key)
        self.batch_size = config.get("batch_size", 2)
        self.max_retries = config.get("max_retries", 5)
        self.max_content_length = config.get("max_content_length", 12000)
        self.add_interval = config.get("add_interval", 0.0)
        self.search_interval = config.get("search", {}).get("search_interval", 0.0)
        self.console = Console()

        # Set custom instructions (loaded from prompts.yaml)
        # Prioritize config settings (backward compatible), otherwise load from prompts
        custom_instructions = config.get("custom_instructions", None)
        if not custom_instructions:
            # Load from prompts.yaml
            custom_instructions = (
                self._prompts.get("add_stage", {})
                .get("mem0", {})
                .get("custom_instructions", None)
            )
            print(f"   ✅ Custom instructions set (from prompts.yaml)")

        # Store custom_instructions for async initialization
        self._custom_instructions = custom_instructions

        print(f"   Batch Size: {self.batch_size}")
        print(f"   Max Content Length: {self.max_content_length}")
        if self.add_interval > 0:
            print(f"   Add Interval: {self.add_interval}s (rate limiting)")
        if self.search_interval > 0:
            print(f"   Search Interval: {self.search_interval}s (rate limiting)")

    def _convert_timestamp_to_display_timezone(self, timestamp_str: str) -> str:
        """
        Convert mem0's timestamp to display timezone.

        Default behavior (if display_timezone_offset not set):
        - Convert to system local timezone (symmetric with add stage where naive datetime
          is treated as local timezone by Python's .timestamp() method)

        Optional behavior (if display_timezone_offset is set):
        - Convert to specified timezone (e.g., UTC for explicit UTC handling)

        Args:
            timestamp_str: ISO format timestamp string with timezone (e.g., "2023-05-07T22:56:00-07:00")

        Returns:
            Formatted timestamp string in display timezone or original if conversion fails
        """
        if not timestamp_str:
            return timestamp_str

        try:
            # Parse ISO format timestamp (with timezone)
            dt = datetime.fromisoformat(timestamp_str)

            dt_display = dt.astimezone(None)

            # Format as readable string (YYYY-MM-DD HH:MM:SS)
            return dt_display.strftime("%Y-%m-%d %H:%M:%S")
        except Exception as e:
            # If conversion fails, return original string
            self.console.print(
                f"⚠️  Failed to convert timestamp '{timestamp_str}': {e}", style="yellow"
            )
            return timestamp_str

    async def prepare(self, conversations: List[Conversation], **kwargs) -> None:
        """
        Preparation stage: update project configuration and clean existing data.

        Args:
            conversations: Standard format conversation list
            **kwargs: Extra parameters
        """
        # Update project with custom instructions (if set)
        if self._custom_instructions:
            try:
                await self.client.update_project(
                    custom_instructions=self._custom_instructions
                )
                self.console.print("   ✅ Custom instructions set", style="green")
            except Exception as e:
                self.console.print(
                    f"   ⚠️  Failed to set custom instructions: {e}", style="yellow"
                )

        # Check if need to clean existing data
        clean_before_add = self.config.get("clean_before_add", False)

        if not clean_before_add:
            self.console.print(
                "   ⏭️  Skipping data cleanup (clean_before_add=false)", style="dim"
            )
            return

        self.console.print(f"\n{'='*60}", style="bold yellow")
        self.console.print(f"Preparation: Cleaning existing data", style="bold yellow")
        self.console.print(f"{'='*60}", style="bold yellow")

        # Collect all user_ids to clean
        user_ids_to_clean = set()

        for conv in conversations:
            # Get user_id for speaker_a and speaker_b
            speaker_a = conv.metadata.get("speaker_a", "")
            speaker_b = conv.metadata.get("speaker_b", "")

            need_dual = self._need_dual_perspective(speaker_a, speaker_b)

            user_ids_to_clean.add(self._extract_user_id(conv, speaker="speaker_a"))

            if need_dual:
                user_ids_to_clean.add(self._extract_user_id(conv, speaker="speaker_b"))

        # Clean all user data
        self.console.print(
            f"\n🗑️  Cleaning data for {len(user_ids_to_clean)} user(s)...",
            style="yellow",
        )

        cleaned_count = 0
        failed_count = 0

        for user_id in user_ids_to_clean:
            try:
                # Use async client for delete operation
                await self.client.delete_all(user_id=user_id)
                cleaned_count += 1
                self.console.print(f"   ✅ Cleaned: {user_id}", style="green")
            except Exception as e:
                failed_count += 1
                self.console.print(
                    f"   ⚠️  Failed to clean {user_id}: {e}", style="yellow"
                )

        self.console.print(
            f"\n✅ Cleanup completed: {cleaned_count} succeeded, {failed_count} failed",
            style="bold green",
        )

    async def _add_user_messages(
        self, conv: Conversation, messages: List[Dict[str, Any]], speaker: str, **kwargs
    ) -> Any:
        """
        Add messages for a single user to Mem0.

        Args:
            conv: Original conversation object
            messages: Formatted message list
            speaker: "speaker_a" or "speaker_b"
            **kwargs: Extra parameters

        Returns:
            None
        """
        # Extract user_id
        user_id = self._extract_user_id(conv, speaker=speaker)

        # Handle content truncation (Mem0 specific)
        truncated_count = 0
        for msg in messages:
            if len(msg["content"]) > self.max_content_length:
                msg["content"] = msg["content"][: self.max_content_length]
                truncated_count += 1

        # Log info
        sender_name = conv.metadata.get(speaker, speaker)
        is_fake_timestamp = (
            conv.messages[0].metadata.get("is_fake_timestamp", False)
            if conv.messages
            else False
        )

        self.console.print(
            f"   📤 Adding for {sender_name} ({user_id}): {len(messages)} messages",
            style="dim",
        )
        if is_fake_timestamp:
            self.console.print(f"   ⚠️  Using fake timestamp", style="yellow")
        if truncated_count > 0:
            self.console.print(
                f"   ⚠️  Truncated {truncated_count} messages (>{self.max_content_length} chars)",
                style="yellow",
            )

        # Add messages in batches with retry
        # Note: messages list corresponds to conv.messages in order
        for i in range(0, len(messages), self.batch_size):
            batch_messages = messages[i : i + self.batch_size]

            # Use the timestamp of the first message in this batch
            timestamp = None
            if i < len(conv.messages) and conv.messages[i].timestamp:
                timestamp = int(conv.messages[i].timestamp.timestamp())

            for attempt in range(self.max_retries):
                try:
                    # Use async client for add operation
                    await self.client.add(
                        messages=batch_messages, timestamp=timestamp, user_id=user_id
                    )
                    # Wait between add requests to avoid rate limits
                    if self.add_interval > 0:
                        await asyncio.sleep(self.add_interval)
                    break
                except Exception as e:
                    if attempt < self.max_retries - 1:
                        self.console.print(
                            f"   ⚠️  [{sender_name} (user_id={user_id})] Retry {attempt + 1}/{self.max_retries}: {e}",
                            style="yellow",
                        )
                        await asyncio.sleep(2**attempt)  # Use async sleep
                    else:
                        self.console.print(
                            f"   ❌ [{sender_name} (user_id={user_id})] Failed after {self.max_retries} retries: {e}",
                            style="red",
                        )
                        raise e

        return None

    async def _search_single_user(
        self, query: str, conversation_id: str, user_id: str, top_k: int, **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Search memories for a single user (Mem0-specific with timezone conversion).

        Calls Mem0 search API and converts results to standard format,
        applying timezone conversion to timestamps.

        Args:
            query: Query text
            conversation_id: Conversation ID (not used by Mem0)
            user_id: User ID to search for
            top_k: Number of results to retrieve
            **kwargs: Additional parameters

        Returns:
            List of search results with timezone-converted timestamps
        """
        # Add interval before search to avoid rate limiting (429 errors)
        if self.search_interval > 0:
            await asyncio.sleep(self.search_interval)

        try:
            # Use async client for search operation
            raw_results = await self.client.search(
                query=query,
                top_k=top_k,
                user_id=user_id,
                filters={"AND": [{"user_id": f"{user_id}"}]},
            )

            # Debug: print raw search results
            self.console.print(f"\n[DEBUG] Mem0 Search Results:", style="yellow")
            self.console.print(f"  Query: {query}", style="dim")
            self.console.print(f"  User ID: {user_id}", style="dim")
            self.console.print(
                f"  Results: {json.dumps(raw_results, indent=2, ensure_ascii=False)}",
                style="dim",
            )

        except Exception as e:
            self.console.print(f"❌ Mem0 search error: {e}", style="red")
            return []

        # Convert to standard format with timezone conversion
        results = []
        for memory in raw_results.get("results", []):
            # Apply timezone conversion to timestamp
            created_at_original = memory.get("created_at", "")
            created_at_display = self._convert_timestamp_to_display_timezone(
                created_at_original
            )

            results.append(
                {
                    "content": f"{created_at_display}: {memory['memory']}",  # Add timestamp prefix
                    "score": memory.get("score", 0.0),
                    "user_id": user_id,
                    "metadata": {
                        "id": memory.get("id", ""),
                        "created_at": created_at_original,
                        "created_at_display": created_at_display,
                        "memory": memory.get("memory", ""),
                        "user_id": memory.get("user_id", ""),
                    },
                }
            )

        return results

    def _build_single_search_result(
        self,
        query: str,
        conversation_id: str,
        results: List[Dict[str, Any]],
        user_id: str,
        top_k: int,
        **kwargs,
    ) -> SearchResult:
        """
        Build SearchResult for single perspective (Mem0: simple metadata).

        Args:
            query: Query text
            conversation_id: Conversation ID
            results: Search results from _search_single_user
            user_id: User ID
            top_k: Number of results requested
            **kwargs: Additional parameters

        Returns:
            SearchResult (no formatted_context, uses fallback)
        """
        return SearchResult(
            query=query,
            conversation_id=conversation_id,
            results=results,
            retrieval_metadata={
                "system": "mem0",
                "top_k": top_k,
                "dual_perspective": False,
                "user_ids": [user_id],
            },
        )

    def _build_dual_search_result(
        self,
        query: str,
        conversation_id: str,
        all_results: List[Dict[str, Any]],
        results_a: List[Dict[str, Any]],
        results_b: List[Dict[str, Any]],
        speaker_a: str,
        speaker_b: str,
        speaker_a_user_id: str,
        speaker_b_user_id: str,
        top_k: int,
        **kwargs,
    ) -> SearchResult:
        """
        Build SearchResult for dual perspective (Mem0: use template).

        Formats memories using the default template for dual-speaker scenarios.

        Args:
            query: Query text
            conversation_id: Conversation ID
            all_results: Merged results (for fallback)
            results_a: Speaker A's search results
            results_b: Speaker B's search results
            speaker_a: Speaker A name
            speaker_b: Speaker B name
            speaker_a_user_id: Speaker A user ID
            speaker_b_user_id: Speaker B user ID
            top_k: Number of results per user
            **kwargs: Additional parameters

        Returns:
            SearchResult with formatted_context
        """
        # Extract content from results (already includes timezone-converted timestamps)
        speaker_a_memories_text = (
            "\n".join([r["content"] for r in results_a])
            if results_a
            else "(No memories found)"
        )
        speaker_b_memories_text = (
            "\n".join([r["content"] for r in results_b])
            if results_b
            else "(No memories found)"
        )

        # Use default template
        template = self._prompts["online_api"].get("templates", {}).get("default", "")
        formatted_context = template.format(
            speaker_1=speaker_a,
            speaker_1_memories=speaker_a_memories_text,
            speaker_2=speaker_b,
            speaker_2_memories=speaker_b_memories_text,
        )

        return SearchResult(
            query=query,
            conversation_id=conversation_id,
            results=all_results,
            retrieval_metadata={
                "system": "mem0",
                "top_k": top_k,
                "dual_perspective": True,
                "user_ids": [speaker_a_user_id, speaker_b_user_id],
                "formatted_context": formatted_context,
                "speaker_a_memories_count": len(results_a),
                "speaker_b_memories_count": len(results_b),
            },
        )

    def _get_answer_prompt(self) -> str:
        """
        Return answer prompt.

        Uses generic default prompt (loaded from YAML).
        """
        return self._prompts["online_api"]["default"]["answer_prompt_mem0"]

    def get_system_info(self) -> Dict[str, Any]:
        """Return system info."""
        return {
            "name": "Mem0",
            "type": "online_api",
            "description": "Mem0 - Personalized AI Memory Layer",
            "adapter": "Mem0Adapter",
        }

"""
Memu Adapter - adapt Memu online API for evaluation framework.
Uses HTTP RESTful API instead of Python SDK to avoid dependency conflicts.
Reference: https://memu.so/
"""

import json
import time
import requests
from pathlib import Path
from typing import Any, Dict, List

from rich.console import Console

from evaluation.src.adapters.online_base import OnlineAPIAdapter
from evaluation.src.adapters.registry import register_adapter
from evaluation.src.core.data_models import Conversation, SearchResult
from common_utils.datetime_utils import get_now_with_timezone


@register_adapter("memu")
class MemuAdapter(OnlineAPIAdapter):
    """
    Memu online API adapter.

    Uses HTTP RESTful API directly to avoid Python SDK dependency conflicts.

    Supports:
    - Memory ingestion (based on conversation context)
    - Async task status monitoring
    - Memory retrieval

    Config example:
    ```yaml
    adapter: "memu"
    api_key: "${MEMU_API_KEY}"
    base_url: "https://api.memu.so"  # Optional, defaults to official API
    agent_id: "default_agent"  # Optional, default agent ID
    agent_name: "Assistant"  # Optional, default agent name
    task_check_interval: 3  # Optional, task status check interval (seconds)
    task_timeout: 90  # Optional, task timeout (seconds)
    ```
    """

    def __init__(self, config: dict, output_dir: Path = None):
        super().__init__(config, output_dir)

        # Get configuration
        api_key = config.get("api_key", "")
        if not api_key:
            raise ValueError("Memu API key is required. Set 'api_key' in config.")

        self.base_url = config.get("base_url", "https://api.memu.so").rstrip('/')
        self.agent_id = config.get("agent_id", "default_agent")
        self.agent_name = config.get("agent_name", "Assistant")
        self.task_check_interval = config.get("task_check_interval", 3)
        self.task_timeout = config.get("task_timeout", 90)
        self.max_retries = config.get("max_retries", 5)

        # Get valid_users list for filtering (used for retrying failed tasks)
        self.valid_users = config.get("valid_users", None)

        # Mock mode for testing (skip actual API calls)
        self.mock_mode = config.get("mock_mode", False)

        # HTTP headers
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        self.console = Console()
        self.console.print(f"   Base URL: {self.base_url}", style="dim")
        self.console.print(
            f"   Agent: {self.agent_name} ({self.agent_id})", style="dim"
        )
        if self.valid_users:
            self.console.print(
                f"   Valid Users Filter: {len(self.valid_users)} user(s)",
                style="yellow",
            )
        if self.mock_mode:
            self.console.print(
                f"   🧪 Mock Mode: ENABLED (API calls will be simulated)",
                style="bold yellow",
            )

        # Force sequential processing (override num_workers)
        self.console.print(
            f"   🔄 Sequential Mode: ENABLED (all operations are serial)",
            style="bold cyan",
        )

    async def add(self, conversations: List[Conversation], **kwargs) -> Dict[str, Any]:
        """
        Ingest conversation data (call online API) in pure sequential mode.

        Override parent's add() method to enforce sequential processing:
        - Process conversations one by one (no concurrency)
        - Process users within each conversation one by one (no concurrency)
        - Wait for each task to complete before proceeding to the next

        This ensures Memu API is not overwhelmed with concurrent requests.
        """
        conversation_ids = []
        add_results = []

        # Process conversations sequentially (one by one)
        for conv in conversations:
            conv_id = conv.conversation_id

            # Extract conversation info (speaker names, user_ids, perspective mode)
            conv_info = self._extract_conversation_info(
                conversation=conv, conversation_id=conv_id
            )

            # Get format type
            format_type = self._get_format_type()

            # Organize messages based on perspective
            if conv_info["need_dual_perspective"]:
                # Dual perspective: prepare messages for both speakers
                speaker_a_messages = self._conversation_to_messages(
                    conv, format_type=format_type, perspective="speaker_a"
                )
                speaker_b_messages = self._conversation_to_messages(
                    conv, format_type=format_type, perspective="speaker_b"
                )

                # Add messages for speaker_a first (sequential)
                result_a = await self._add_user_messages(
                    conv, speaker_a_messages, speaker="speaker_a", **kwargs
                )

                # Wait for speaker_a's task to complete
                await self._wait_for_conversation_tasks(
                    [result_a], conversation_id=conv_id, **kwargs
                )

                # Add messages for speaker_b second (sequential)
                result_b = await self._add_user_messages(
                    conv, speaker_b_messages, speaker="speaker_b", **kwargs
                )

                # Wait for speaker_b's task to complete
                await self._wait_for_conversation_tasks(
                    [result_b], conversation_id=conv_id, **kwargs
                )

                # Collect results
                conversation_ids.append(conv_id)
                add_results.extend([result_a, result_b])
            else:
                # Single perspective: prepare messages for speaker_a only
                messages = self._conversation_to_messages(
                    conv, format_type=format_type, perspective=None
                )

                # Add messages for single user
                result = await self._add_user_messages(
                    conv, messages, speaker="speaker_a", **kwargs
                )

                # Wait for task to complete
                await self._wait_for_conversation_tasks(
                    [result], conversation_id=conv_id, **kwargs
                )

                # Collect results
                conversation_ids.append(conv_id)
                add_results.append(result)

        # Post-processing (already waited for all tasks, so this is a no-op)
        await self._post_add_process(add_results, **kwargs)

        # Build and return result
        return self._build_add_result(conversation_ids, add_results, **kwargs)

    async def _add_user_messages(
        self, conv: Conversation, messages: List[Dict[str, Any]], speaker: str, **kwargs
    ) -> Any:
        """
        Add messages for a single user to Memu.

        Args:
            conv: Original conversation object
            messages: Formatted message list
            speaker: "speaker_a" or "speaker_b"
            **kwargs: Extra parameters

        Returns:
            task_id: Task ID for tracking async task
        """
        # Extract user_id and user_name
        user_id = self._extract_user_id(conv, speaker=speaker)
        user_name = conv.metadata.get(
            speaker, "User" if speaker == "speaker_a" else "Assistant"
        )

        # Check if user_id is in valid_users list (if valid_users is set)
        if self.valid_users is not None and user_id not in self.valid_users:
            self.console.print(
                f"   ⏭️  Skipping {user_name} ({user_id}): not in valid_users",
                style="dim yellow",
            )
            return None

        # Get session_date (ISO format date)
        session_date = None
        if conv.messages and conv.messages[0].timestamp:
            session_date = conv.messages[0].timestamp.strftime("%Y-%m-%d")
        else:
            session_date = get_now_with_timezone().strftime("%Y-%m-%d")

        # Validate that all messages have name field
        # Note: messages already contain name and time from base class _conversation_to_messages
        for msg in messages:
            if not msg.get("name"):
                raise ValueError(f"Message missing 'name' field: {msg}")

        self.console.print(
            f"   📤 Adding for {user_name} ({user_id}): {len(messages)} messages",
            style="dim",
        )

        # Construct request payload
        payload = {
            "conversation": messages,
            "user_id": user_id,
            "user_name": user_name,
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "session_date": session_date,
        }

        # Mock mode: Skip actual API call
        if self.mock_mode:
            self.console.print(
                f"      🧪 [MOCK] Would add {len(messages)} messages for {user_name} ({user_id})",
                style="cyan",
            )
            self.console.print(
                f"      🧪 [MOCK] Payload: user_id={user_id}, agent_id={self.agent_id}, "
                f"session_date={session_date}, messages={len(messages)}",
                style="dim cyan",
            )
            self.console.print(f"      🧪 [MOCK] Returning task_id=None", style="cyan")
            return None

        # Submit task (with retry)
        import asyncio

        task_id = None
        for attempt in range(self.max_retries):
            try:
                url = f"{self.base_url}/api/v0/memory/memorize"
                # Use run_in_executor to avoid blocking event loop
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None, lambda: requests.post(url, headers=self.headers, json=payload)
                )
                response.raise_for_status()

                result = response.json()
                task_id = result.get("task_id")
                status = result.get("status")

                self.console.print(
                    f"      ✅ Task created: {task_id} (status: {status})",
                    style="green",
                )
                break

            except Exception as e:
                if attempt < self.max_retries - 1:
                    self.console.print(
                        f"      ⚠️  [{user_name}] Retry {attempt + 1}/{self.max_retries}: {e}",
                        style="yellow",
                    )
                    time.sleep(2**attempt)
                else:
                    self.console.print(
                        f"      ❌ [{user_name}] Failed after {self.max_retries} retries: {e}",
                        style="red",
                    )
                    raise e

        return task_id

    async def _wait_for_conversation_tasks(
        self, task_results: List[Any], **kwargs
    ) -> None:
        """
        Wait for tasks from a single conversation to complete.

        This is called per-conversation, before releasing the semaphore.
        This ensures that Memu respects the num_workers limit on concurrent tasks.

        Args:
            task_results: List of task_ids from this conversation
            **kwargs: Extra parameters (including conversation_id)
        """
        # Filter out None values
        task_ids = [task_id for task_id in task_results if task_id is not None]

        # Extract conversation_id for logging
        conversation_id = kwargs.get("conversation_id", "unknown")

        if task_ids:
            # Wait for this conversation's tasks to complete
            await self._wait_for_all_tasks(task_ids, conversation_id)

    async def _post_add_process(self, add_results: List[Any], **kwargs) -> None:
        """
        Post-processing hook.

        For Memu, all tasks have already been waited for in _wait_for_conversation_tasks,
        so this is now a no-op.

        Args:
            add_results: List of task_ids returned from _add_user_messages
            **kwargs: Extra parameters
        """
        # All tasks already waited for in _wait_for_conversation_tasks
        # This is now a no-op
        pass

    def _build_add_result(
        self, conversation_ids: List[str], add_results: List[Any], **kwargs
    ) -> Dict[str, Any]:
        """
        Build the final result dict with task_ids for Memu.

        Args:
            conversation_ids: List of conversation IDs
            add_results: List of task_ids
            **kwargs: Extra parameters

        Returns:
            Result dictionary with task_ids
        """
        # Filter out None values to get actual task_ids
        task_ids = [task_id for task_id in add_results if task_id is not None]

        return {
            "type": "online_api",
            "system": "memu",
            "conversation_ids": conversation_ids,
            "task_ids": task_ids,
        }

    async def _wait_for_all_tasks(
        self, task_ids: List[str], conversation_id: str = "unknown"
    ) -> bool:
        """
        Wait for all tasks to complete.

        Args:
            task_ids: Task ID list
            conversation_id: Conversation ID for logging

        Returns:
            Whether all tasks completed successfully
        """
        import asyncio

        if not task_ids:
            return True

        start_time = time.time()
        pending_tasks = set(task_ids)

        # Show progress
        total_tasks = len(task_ids)
        # Create a short label for logging
        conv_label = f"[{conversation_id}]"

        while time.time() - start_time < self.task_timeout:
            completed_in_round = []
            failed_in_round = []

            for task_id in list(pending_tasks):
                try:
                    url = f"{self.base_url}/api/v0/memory/memorize/status/{task_id}"
                    # Use run_in_executor to avoid blocking event loop
                    loop = asyncio.get_event_loop()
                    response = await loop.run_in_executor(
                        None, lambda: requests.get(url, headers=self.headers)
                    )
                    response.raise_for_status()
                    result = response.json()
                    status = result.get("status")

                    # Memu API returns uppercase status: PENDING/PROCESSING/SUCCESS/FAILED
                    if status in ["SUCCESS", "COMPLETED"]:
                        completed_in_round.append(task_id)
                    elif status in ["FAILED", "FAILURE"]:
                        failed_in_round.append(task_id)
                        self.console.print(
                            f"   {conv_label} ❌ Task {task_id} failed: {result.get('detail_info', 'Unknown error')}",
                            style="red",
                        )

                except Exception as e:
                    self.console.print(
                        f"   {conv_label} ⚠️  Error checking task {task_id}: {e}",
                        style="yellow",
                    )

            # Remove completed/failed tasks
            for task_id in completed_in_round + failed_in_round:
                pending_tasks.remove(task_id)

            # Update progress
            completed_count = total_tasks - len(pending_tasks)
            if completed_in_round or failed_in_round:
                self.console.print(
                    f"   {conv_label} 📊 Progress: {completed_count}/{total_tasks} tasks completed",
                    style="cyan",
                )

            # If all tasks completed
            if not pending_tasks:
                self.console.print(
                    f"   {conv_label} ✅ All {total_tasks} tasks completed!",
                    style="bold green",
                )
                return len(failed_in_round) == 0

            # Wait before retry
            if pending_tasks:
                elapsed = time.time() - start_time
                self.console.print(
                    f"   {conv_label} ⏳ {len(pending_tasks)} task(s) still processing... ({elapsed:.0f}s elapsed)",
                    style="dim",
                )
                await asyncio.sleep(self.task_check_interval)

        # Timeout
        self.console.print(
            f"   {conv_label} ⚠️  Timeout: {len(pending_tasks)} task(s) not completed within {self.task_timeout}s",
            style="yellow",
        )
        return False

    async def _search_single_user(
        self, query: str, conversation_id: str, user_id: str, top_k: int, **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Search memories for a single user (Memu-specific with categories summary).

        Calls Memu search API and fetches categories summary.

        Args:
            query: Query text
            conversation_id: Conversation ID (not used by Memu)
            user_id: User ID to search for
            top_k: Number of results to retrieve
            **kwargs: Additional parameters (e.g., min_similarity)

        Returns:
            List of search results with categories_summary in metadata
        """
        import asyncio

        min_similarity = kwargs.get("min_similarity", 0.3)

        try:
            url = f"{self.base_url}/api/v0/memory/retrieve/related-memory-items"
            payload = {
                "user_id": user_id,
                "agent_id": self.agent_id,
                "query": query,
                "top_k": top_k,
                "min_similarity": min_similarity,
            }

            # Use run_in_executor to avoid blocking event loop
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, lambda: requests.post(url, headers=self.headers, json=payload)
            )
            response.raise_for_status()
            result = response.json()

        except Exception as e:
            self.console.print(f"❌ Memu search error: {e}", style="red")
            return []

        # Get categories summary (fail silently if error)
        categories_summary = await self._get_categories_summary(user_id)

        # Convert to standard format
        results = []
        related_memories = result.get("related_memories", [])

        for item in related_memories:
            memory = item.get("memory", {})
            results.append(
                {
                    "content": memory.get("content", ""),
                    "score": item.get("similarity_score", 0.0),
                    "user_id": user_id,
                    "metadata": {
                        "id": memory.get("memory_id", ""),
                        "category": memory.get("category", ""),
                        "created_at": memory.get("created_at", ""),
                        "happened_at": memory.get("happened_at", ""),
                        "categories_summary": categories_summary,  # Store summary for this user
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
        Build SearchResult for single perspective (Memu: custom context with summary).

        Args:
            query: Query text
            conversation_id: Conversation ID
            results: Search results from _search_single_user
            user_id: User ID
            top_k: Number of results requested
            **kwargs: Additional parameters (e.g., min_similarity)

        Returns:
            SearchResult with custom formatted_context
        """
        min_similarity = kwargs.get("min_similarity", 0.3)

        # Extract categories_summary from first result's metadata
        categories_summary = (
            results[0]["metadata"]["categories_summary"] if results else ""
        )

        # Build custom context using Memu-specific logic
        formatted_context = self._format_user_memories_with_summary(
            results=results,
            categories_summary=categories_summary,
            top_k=top_k,
            memory_separator="\n\n",
        )

        return SearchResult(
            query=query,
            conversation_id=conversation_id,
            results=results,
            retrieval_metadata={
                "system": "memu",
                "user_ids": [user_id],
                "top_k": top_k,
                "min_similarity": min_similarity,
                "total_found": len(results),
                "formatted_context": formatted_context,
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
        Build SearchResult for dual perspective (Memu: custom context with summaries).

        Formats memories using Memu-specific dual perspective logic,
        including categories summaries for both speakers.

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
            **kwargs: Additional parameters (e.g., min_similarity)

        Returns:
            SearchResult with custom formatted_context
        """
        min_similarity = kwargs.get("min_similarity", 0.3)

        # Extract categories summaries from results' metadata
        categories_summary_a = (
            results_a[0]["metadata"]["categories_summary"] if results_a else ""
        )
        categories_summary_b = (
            results_b[0]["metadata"]["categories_summary"] if results_b else ""
        )

        # Build dual perspective context using Memu-specific logic
        speaker_a_memories_text = self._format_user_memories_with_summary(
            results=results_a,
            categories_summary=categories_summary_a,
            top_k=top_k,
            memory_separator="\n",
        )

        speaker_b_memories_text = self._format_user_memories_with_summary(
            results=results_b,
            categories_summary=categories_summary_b,
            top_k=top_k,
            memory_separator="\n",
        )

        # Wrap using default template
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
                "system": "memu",
                "user_ids": [speaker_a_user_id, speaker_b_user_id],
                "top_k": top_k,
                "min_similarity": min_similarity,
                "total_found": len(all_results),
                "formatted_context": formatted_context,
                "dual_perspective": True,
                "categories_summary_a": categories_summary_a,
                "categories_summary_b": categories_summary_b,
            },
        )

    def _format_user_memories_with_summary(
        self,
        results: List[Dict[str, Any]],
        categories_summary: str = "",
        top_k: int = 10,
        memory_separator: str = "\n\n",
    ) -> str:
        """
        Format a single user's memories with categories summary.

        This is a helper method to avoid code duplication in building contexts.

        Args:
            results: Search results list
            categories_summary: Categories summary (optional)
            top_k: Number of results to show
            memory_separator: Separator between memories (default: "\n\n")

        Returns:
            Formatted text combining summary and memories
        """
        content_parts = []

        # Add categories summary first (if available)
        if categories_summary:
            content_parts.append(categories_summary)

        # Add search results
        if results:
            if categories_summary:
                content_parts.append("\n## Related Memories\n")

            memories = []
            for idx, result in enumerate(results[:top_k], 1):
                content = result.get("content", "")
                metadata = result.get("metadata", {})
                happened_at = metadata.get("happened_at", "")
                category = metadata.get("category", "")

                # Build format for each memory
                memory_text = f"{idx}. {content}"

                # Add time and category information (if available)
                metadata_parts = []
                if happened_at:
                    # Only show date part (YYYY-MM-DD)
                    date_str = (
                        happened_at.split("T")[0] if "T" in happened_at else happened_at
                    )
                    metadata_parts.append(f"Date: {date_str}")
                if category:
                    metadata_parts.append(f"Category: {category}")

                if metadata_parts:
                    memory_text += f" ({', '.join(metadata_parts)})"

                memories.append(memory_text)

            content_parts.append(memory_separator.join(memories))
        elif not categories_summary:
            # No categories summary and no search results
            return ""

        return "".join(content_parts)

    async def _get_all_memories(self, user_id: str) -> Dict[str, Any]:
        """
        Get all memories (categories with summaries) for a user.

        This method calls the Memu API to retrieve default categories and their summaries.
        This provides a high-level overview of the user's memory landscape.

        Args:
            user_id: User ID

        Returns:
            API response containing categories and their summaries
            Returns empty dict if error occurs (fail silently)
        """
        import asyncio

        try:
            url = f"{self.base_url}/api/v0/memory/retrieve/default-categories"
            payload = {
                "user_id": user_id,
                "agent_id": self.agent_id,
                "want_memory_items": True,
            }

            # Use run_in_executor to avoid blocking event loop
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, lambda: requests.post(url, headers=self.headers, json=payload)
            )
            response.raise_for_status()
            result = response.json()

            return result

        except Exception as e:
            # Fail silently - categories summary is optional context
            self.console.print(
                f"   ⚠️  Failed to get categories for {user_id}: {e}", style="dim yellow"
            )
            return {}

    def _format_categories_summary(self, memories: Dict[str, Any]) -> str:
        """
        Format categories summary from get_all_memories response.

        Extracts category names and summaries and formats them into a readable string.
        This provides a structured overview that helps the LLM understand the memory landscape.

        Args:
            memories: Response from _get_all_memories()

        Returns:
            Formatted categories summary string
            Returns empty string if no valid categories found
        """
        if not memories or 'categories' not in memories:
            return ""

        summary_parts = ["## Memory Overview (by Category)\n"]

        categories = memories.get('categories', [])
        has_content = False

        for category in categories:
            category_name = category.get('name', '')
            category_summary = category.get('summary', '')

            if category_name and category_summary:
                summary_parts.append(f"**{category_name}:** {category_summary}\n\n")
                has_content = True

        if not has_content:
            return ""

        return "".join(summary_parts)

    async def _get_categories_summary(self, user_id: str) -> str:
        """
        Get and format categories summary for a user.

        This is a convenience method that combines _get_all_memories and _format_categories_summary.
        It's designed to be called during search to augment context with memory overview.

        Args:
            user_id: User ID

        Returns:
            Formatted categories summary string
            Returns empty string if error occurs or no categories found
        """
        memories = await self._get_all_memories(user_id)
        return self._format_categories_summary(memories)

    def get_system_info(self) -> Dict[str, Any]:
        """Return system info."""
        return {
            "name": "Memu",
            "type": "online_api",
            "description": "Memu - Memory Management System (HTTP RESTful API)",
            "adapter": "MemuAdapter",
            "base_url": self.base_url,
            "agent_id": self.agent_id,
        }

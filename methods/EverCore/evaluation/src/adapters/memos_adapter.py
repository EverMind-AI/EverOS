"""
Memos Adapter - adapt Memos online API for evaluation framework.
Reference: https://www.memos.so/
"""

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp
from aiolimiter import AsyncLimiter
from rich.console import Console

from evaluation.src.adapters.online_base import OnlineAPIAdapter
from evaluation.src.adapters.registry import register_adapter
from evaluation.src.core.data_models import Conversation, SearchResult


@register_adapter("memos")
class MemosAdapter(OnlineAPIAdapter):
    """
    Memos online API adapter.

    Supports:
    - Memory ingestion (supports conversation context)
    - Memory retrieval

    Official API supported parameters:
    - user_id (required) - Format: {conv_id}_{speaker}, already contains session info
    - query (required)
    - memory_limit_number (optional, default 6)

    Note: Does not use conversation_id parameter, as user_id already contains session info

    Config example:
    ```yaml
    adapter: "memos"
    api_url: "${MEMOS_URL}"
    api_key: "${MEMOS_KEY}"
    ```
    """

    def __init__(self, config: dict, output_dir: Path = None):
        super().__init__(config, output_dir)

        # Get API configuration
        self.api_url = config.get("api_url", "")
        if not self.api_url:
            raise ValueError("Memos API URL is required. Set 'api_url' in config.")

        api_key = config.get("api_key", "")
        if not api_key:
            raise ValueError("Memos API key is required. Set 'api_key' in config.")

        self.headers = {"Content-Type": "application/json", "Authorization": api_key}

        # Retrieval configuration (only keep batch_size and max_retries, other params not supported by official API)
        self.batch_size = config.get("batch_size", 9999)  # Memos supports large batches
        self.max_retries = config.get("max_retries", 5)

        # Rate limiting configuration (default: 10 requests/second)
        requests_per_second = config.get("requests_per_second", 10)
        self.rate_limiter = AsyncLimiter(max_rate=requests_per_second, time_period=1.0)

        # Create aiohttp session (will be initialized on first use)
        self._session: Optional[aiohttp.ClientSession] = None

        self.console = Console()

        print(f"   API URL: {self.api_url}")
        print(f"   Rate Limit: {requests_per_second} requests/second (async)")

    async def _get_session(self) -> aiohttp.ClientSession:
        """
        Get or create aiohttp session (lazy initialization).

        Returns:
            aiohttp.ClientSession instance
        """
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=60)
            self._session = aiohttp.ClientSession(headers=self.headers, timeout=timeout)
        return self._session

    async def close(self):
        """
        Close aiohttp session.

        Should be called when adapter is no longer needed.
        """
        if self._session and not self._session.closed:
            await self._session.close()

    async def _add_user_messages(
        self, conv: Conversation, messages: List[Dict[str, Any]], speaker: str, **kwargs
    ) -> Any:
        """
        Add messages for a single user to Memos.

        Args:
            conv: Original conversation object
            messages: Formatted message list
            speaker: "speaker_a" or "speaker_b"
            **kwargs: Extra parameters

        Returns:
            None
        """
        # Extract user_id and conv_id
        user_id = self._extract_user_id(conv, speaker=speaker)
        conv_id = conv.conversation_id

        # Log info
        sender_name = conv.metadata.get(speaker, speaker)
        self.console.print(
            f"   📤 Adding for {sender_name} ({user_id}): {len(messages)} messages",
            style="dim",
        )

        # Get session
        session = await self._get_session()

        # Send messages in batches with retry
        url = f"{self.api_url}/add/message"

        for i in range(0, len(messages), self.batch_size):
            batch_messages = messages[i : i + self.batch_size]

            # Try to send the batch with automatic batch size reduction on token limit error
            await self._send_message_batch(
                url=url,
                batch_messages=batch_messages,
                user_id=user_id,
                conv_id=conv_id,
                sender_name=sender_name,
                session=session,
            )

        return None

    async def _send_message_batch(
        self,
        url: str,
        batch_messages: List[Dict[str, Any]],
        user_id: str,
        conv_id: str,
        sender_name: str,
        session: aiohttp.ClientSession,
    ) -> None:
        """
        Send a batch of messages to Memos API.

        Handles token limit exceeded errors by automatically reducing batch size to 2.

        Args:
            url: API endpoint URL
            batch_messages: Messages to send in this batch
            user_id: User ID
            conv_id: Conversation ID
            sender_name: Sender name (for logging)
            session: aiohttp session
        """
        payload_dict = {
            "messages": batch_messages,
            "user_id": user_id,
            "conversation_id": conv_id,
        }

        for attempt in range(self.max_retries):
            try:
                # Apply rate limiting
                async with self.rate_limiter:
                    async with session.post(url, json=payload_dict) as response:
                        if response.status != 200:
                            text = await response.text()
                            raise Exception(f"HTTP {response.status}: {text}")

                        result = await response.json()

                        # Check for token limit exceeded error
                        if (
                            result.get("code") == 40302
                            and result.get("message") == "Input token limit exceeded"
                        ):
                            # If batch size > 1, try splitting into smaller batches
                            if len(batch_messages) > 1:
                                # Determine new batch size: if current > 2, use 2; otherwise use 1
                                new_batch_size = 2 if len(batch_messages) > 2 else 1
                                self.console.print(
                                    f"   ⚠️  [{sender_name}] Token limit exceeded, splitting batch of {len(batch_messages)} into smaller batches (size={new_batch_size})",
                                    style="yellow",
                                )
                                # Recursively send in smaller batches
                                for j in range(0, len(batch_messages), new_batch_size):
                                    sub_batch = batch_messages[j : j + new_batch_size]
                                    await self._send_message_batch(
                                        url=url,
                                        batch_messages=sub_batch,
                                        user_id=user_id,
                                        conv_id=conv_id,
                                        sender_name=sender_name,
                                        session=session,
                                    )
                                return  # Success
                            else:
                                # Batch size is 1, cannot split further
                                # Try truncating the message content by removing last 1000 characters
                                message = batch_messages[0]
                                original_content = message.get("content", "")

                                if len(original_content) > 1000:
                                    self.console.print(
                                        f"   ⚠️  [{sender_name}] Single message token limit exceeded, truncating content (removing last 1000 chars)",
                                        style="yellow",
                                    )
                                    # Create a truncated version of the message
                                    truncated_message = message.copy()
                                    truncated_message["content"] = original_content[
                                        :-1000
                                    ]

                                    # Try sending the truncated message
                                    await self._send_message_batch(
                                        url=url,
                                        batch_messages=[truncated_message],
                                        user_id=user_id,
                                        conv_id=conv_id,
                                        sender_name=sender_name,
                                        session=session,
                                    )
                                    return  # Success
                                else:
                                    # Content is already short, cannot truncate further
                                    raise Exception(
                                        f"API error (token limit, single message too large, content length={len(original_content)}): {result}"
                                    )

                        if result.get("message") != "ok":
                            raise Exception(f"API error: {result}")

                # Success - break retry loop
                break

            except Exception as e:
                if attempt < self.max_retries - 1:
                    self.console.print(
                        f"   ⚠️  [{sender_name}] Retry {attempt + 1}/{self.max_retries}: {e}",
                        style="yellow",
                    )
                    await asyncio.sleep(2**attempt)  # Exponential backoff
                else:
                    self.console.print(
                        f"   ❌ [{sender_name}] Failed after {self.max_retries} retries: {e}",
                        style="red",
                    )
                    raise e

    async def _search_single_user(
        self, query: str, conversation_id: str, user_id: str, top_k: int, **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Search memories for a single user (Memos-specific with preference extraction).

        Calls Memos HTTP API and extracts preference information.

        Args:
            query: Query text
            conversation_id: Conversation ID (not used by Memos, user_id contains this info)
            user_id: User ID to search for (format: {conv_id}_{speaker})
            top_k: Number of results to retrieve
            **kwargs: Additional parameters

        Returns:
            List of search results with preference information in metadata

        Note:
            user_id already contains session info (format: {conv_id}_{speaker}).
            Example: user_id="locomo_0_Caroline" uniquely identifies the locomo_0 conversation.
        """
        # Get session
        session = await self._get_session()

        # Prepare HTTP request
        url = f"{self.api_url}/search/memory"
        payload_dict = {
            "query": query,
            "user_id": user_id,
            "memory_limit_number": top_k,
        }

        # Call API with retry mechanism
        text_mem_res = []
        pref_string = ""

        for attempt in range(self.max_retries):
            try:
                # Apply rate limiting
                async with self.rate_limiter:
                    async with session.post(url, json=payload_dict) as response:
                        if response.status != 200:
                            text = await response.text()
                            raise Exception(f"HTTP {response.status}: {text}")

                        result = await response.json()
                        if result.get("message") != "ok":
                            raise Exception(f"API error: {result}")

                        data = result.get("data", {})
                        text_mem_res = data.get("memory_detail_list", [])
                        pref_mem_res = data.get("preference_detail_list", [])
                        preference_note = data.get("preference_note", "")

                        # Standardize field names: rename memory_value to memory
                        for i in text_mem_res:
                            i.update(
                                {"memory": i.pop("memory_value", i.get("memory", ""))}
                            )

                        # Format preference string
                        explicit_prefs = [
                            p["preference"]
                            for p in pref_mem_res
                            if p.get("preference_type", "") == "explicit_preference"
                        ]
                        implicit_prefs = [
                            p["preference"]
                            for p in pref_mem_res
                            if p.get("preference_type", "") == "implicit_preference"
                        ]

                        pref_parts = []
                        if explicit_prefs:
                            pref_parts.append(
                                "Explicit Preference:\n"
                                + "\n".join(
                                    f"{i + 1}. {p}"
                                    for i, p in enumerate(explicit_prefs)
                                )
                            )
                        if implicit_prefs:
                            pref_parts.append(
                                "Implicit Preference:\n"
                                + "\n".join(
                                    f"{i + 1}. {p}"
                                    for i, p in enumerate(implicit_prefs)
                                )
                            )

                        pref_string = "\n".join(pref_parts) + preference_note

                # Success - break retry loop
                break

            except Exception as e:
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2**attempt)  # Exponential backoff
                else:
                    self.console.print(f"❌ Memos search error: {e}", style="red")
                    return []

        # Convert to standard format
        results = []
        for item in text_mem_res:
            created_at = item.get("memory_time") or item.get("create_time", "")
            results.append(
                {
                    "content": item.get("memory", ""),
                    "score": item.get("relativity", item.get("score", 0.0)),
                    "user_id": user_id,
                    "metadata": {
                        "memory_id": item.get("id", ""),
                        "created_at": str(created_at) if created_at else "",
                        "memory_type": item.get("memory_type", ""),
                        "confidence": item.get("confidence", 0.0),
                        "tags": item.get("tags", []),
                        "pref_string": pref_string,  # Store preference for this user
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
        Build SearchResult for single perspective (Memos: include preference).

        Args:
            query: Query text
            conversation_id: Conversation ID
            results: Search results from _search_single_user
            user_id: User ID
            top_k: Number of results requested
            **kwargs: Additional parameters

        Returns:
            SearchResult with preference metadata (no formatted_context, uses fallback)
        """
        # Extract pref_string from first result's metadata (all results share same pref_string)
        pref_string = results[0]["metadata"]["pref_string"] if results else ""

        return SearchResult(
            query=query,
            conversation_id=conversation_id,
            results=results,
            retrieval_metadata={
                "system": "memos",
                "preferences": {"pref_string": pref_string},
                "top_k": top_k,
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
        Build SearchResult for dual perspective (Memos: use template + preference).

        Formats memories using the default template, including preference information
        for both speakers.

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
            SearchResult with formatted_context and preferences
        """
        # Extract preferences from results' metadata
        pref_a = results_a[0]["metadata"]["pref_string"] if results_a else ""
        pref_b = results_b[0]["metadata"]["pref_string"] if results_b else ""

        # Build context for each speaker (memories + preferences)
        speaker_a_memories = (
            "\n".join([r["content"] for r in results_a])
            if results_a
            else "(No memories found)"
        )
        speaker_b_memories = (
            "\n".join([r["content"] for r in results_b])
            if results_b
            else "(No memories found)"
        )

        speaker_a_context = speaker_a_memories + (f"\n{pref_a}" if pref_a else "")
        speaker_b_context = speaker_b_memories + (f"\n{pref_b}" if pref_b else "")

        # Use default template
        template = self._prompts["online_api"].get("templates", {}).get("default", "")
        formatted_context = template.format(
            speaker_1=speaker_a,
            speaker_1_memories=speaker_a_context,
            speaker_2=speaker_b,
            speaker_2_memories=speaker_b_context,
        )

        return SearchResult(
            query=query,
            conversation_id=conversation_id,
            results=all_results,
            retrieval_metadata={
                "system": "memos",
                "dual_perspective": True,
                "formatted_context": formatted_context,
                "top_k": top_k,
                "user_ids": [speaker_a_user_id, speaker_b_user_id],
                "preferences": {"speaker_a_pref": pref_a, "speaker_b_pref": pref_b},
            },
        )

    def _get_answer_prompt(self) -> str:
        """
        Get answer prompt.

        Subclasses can override this method to return their own prompt.
        Defaults to generic default prompt.
        """
        return self._prompts["online_api"]["default"]["answer_prompt_memos"]

    def get_system_info(self) -> Dict[str, Any]:
        """Return system info."""
        return {
            "name": "Memos",
            "type": "online_api",
            "description": "Memos - Memory System with Preference Support",
            "adapter": "MemosAdapter",
        }

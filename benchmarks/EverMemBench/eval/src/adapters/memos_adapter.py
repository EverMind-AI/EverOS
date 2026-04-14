"""
Memos Adapter for multi-person group chat evaluation.

Implements Add functionality for Memos memory system.
All messages are sent with role="user" and formatted as:
[Group: X][Speaker: Name]content
"""
import asyncio
from datetime import timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import time

import aiohttp
from aiolimiter import AsyncLimiter

from eval.src.adapters.base import BaseAdapter
from eval.src.core.data_models import Dataset, GroupChatMessage, AddResult, SearchResult
from eval.src.utils.logger import get_console, print_success, print_warning


class MemosAdapter(BaseAdapter):
    """
    Memos memory system adapter for multi-person group chat.
    
    Formats all messages as:
    - role: "user" (always)
    - content: "[Group: X][Speaker: Name]original_content"
    - chat_time: ISO format timestamp
    
    Config example:
    ```yaml
    name: "memos"
    api_url: "${MEMOS_API_URL}"
    api_key: "${MEMOS_API_KEY}"
    batch_size: 100
    char_limit: 30000
    max_retries: 5
    requests_per_second: 10
    ```
    """
    
    def __init__(self, config: Dict[str, Any], output_dir: Optional[Path] = None):
        super().__init__(config, output_dir)
        
        # API configuration
        self.api_url = config.get("api_url", "")
        if not self.api_url:
            raise ValueError("Memos API URL is required. Set 'api_url' in config.")
        
        api_key = config.get("api_key", "")
        if not api_key:
            raise ValueError("Memos API key is required. Set 'api_key' in config.")
        
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": api_key
        }
        
        # Batch configuration
        self.batch_size = config.get("batch_size", 100)
        self.char_limit = config.get("char_limit", 30000)
        self.max_retries = config.get("max_retries", 5)
        
        # Rate limiting
        requests_per_second = config.get("requests_per_second", 10)
        self.rate_limiter = AsyncLimiter(max_rate=requests_per_second, time_period=1.0)

        # Search configuration
        self.search_config = config.get("search", {})

        # HTTP configuration
        self.http_timeout = config.get("http_timeout", 60)

        # HTTP session (lazy init)
        self._session: Optional[aiohttp.ClientSession] = None
        
        self.console = get_console()
        
        self.console.print("✅ MemosAdapter initialized", style="bold green")
        self.console.print(f"   API URL: {self.api_url}")
        self.console.print(f"   Batch Size: {self.batch_size}")
        self.console.print(f"   Rate Limit: {requests_per_second} req/s")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.http_timeout)
            self._session = aiohttp.ClientSession(
                headers=self.headers,
                timeout=timeout
            )
        return self._session
    
    async def close(self):
        """Close HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def add(
        self,
        dataset: Dataset,
        user_id: str,
        days_to_process: Optional[List[str]] = None,
        **kwargs
    ) -> AddResult:
        """
        Add dataset to Memos memory system.
        
        Args:
            dataset: Dataset with group chat data
            user_id: User ID for Memos
            days_to_process: Optional list of dates to process (None = all)
            **kwargs: Additional parameters
            
        Returns:
            AddResult with statistics
        """
        self.console.print(f"\n{'='*60}", style="bold cyan")
        self.console.print("Stage: Add (Memos)", style="bold cyan")
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

                group_messages = [self._format_message(m) for m in messages]
                self.console.print(f"      Messages: {len(group_messages)}")

                batches = self._split_into_batches(group_messages)
                self.console.print(f"      Batches: {len(batches)}")

                for batch_idx, batch in enumerate(batches):
                    try:
                        await self._send_batch(batch, user_id)
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
    
    def _format_message(self, msg: GroupChatMessage) -> Dict[str, Any]:
        """
        Format a GroupChatMessage for Memos API.
        
        Input: GroupChatMessage with speaker, content, timestamp, group
        Output: {
            "role": "user",
            "content": "[Group: X][Speaker: Name]content",
            "chat_time": "2025-01-09T09:32:15+00:00"
        }
        """
        # Format content with group and speaker prefix
        formatted_content = f"[Group: {msg.group}][Speaker: {msg.speaker}]{msg.content}"
        
        # Format timestamp as ISO with timezone
        # Add UTC timezone if naive
        ts = msg.timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        chat_time = ts.isoformat()
        
        return {
            "role": "user",
            "content": formatted_content,
            "chat_time": chat_time
        }
    
    def _split_into_batches(self, messages: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Split messages into batches respecting size and character limits.
        
        Args:
            messages: List of formatted messages
            
        Returns:
            List of batches
        """
        batches = []
        current_batch = []
        current_chars = 0
        
        for msg in messages:
            msg_chars = len(msg["content"])
            
            # Check if adding this message would exceed limits
            if (len(current_batch) >= self.batch_size or 
                current_chars + msg_chars > self.char_limit):
                # Start new batch
                if current_batch:
                    batches.append(current_batch)
                current_batch = [msg]
                current_chars = msg_chars
            else:
                current_batch.append(msg)
                current_chars += msg_chars
        
        # Add final batch
        if current_batch:
            batches.append(current_batch)
        
        return batches
    
    async def _send_batch(self, messages: List[Dict[str, Any]], user_id: str):
        """
        Send a batch of messages to Memos API.
        
        Args:
            messages: List of formatted messages
            user_id: User ID for Memos
            
        Raises:
            Exception: If API call fails after retries
        """
        session = await self._get_session()
        url = f"{self.api_url}/add/message"
        
        payload = {
            "messages": messages,
            "user_id": user_id,
            "conversation_id": user_id
        }
        
        for attempt in range(self.max_retries):
            try:
                async with self.rate_limiter:
                    async with session.post(url, json=payload) as response:
                        if response.status != 200:
                            text = await response.text()
                            raise Exception(f"HTTP {response.status}: {text}")
                        
                        result = await response.json()
                        if result.get("message") != "ok":
                            raise Exception(f"API error: {result}")
                        
                        return  # Success
                        
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    self.console.print(
                        f"      ⚠️  Retry {attempt + 1}/{self.max_retries} in {wait_time}s: {e}",
                        style="yellow"
                    )
                    await asyncio.sleep(wait_time)
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
        Search memories in Memos.
        
        Uses /search/memory endpoint with:
        - query: search query
        - user_id: user identifier
        - memory_limit_number: number of memories to retrieve
        - preference_limit_number: number of preference memories
        
        Args:
            query: Search query (usually question text)
            user_id: User ID for Memos
            top_k: Number of memories to retrieve
            **kwargs: Additional parameters (preference_limit_number, etc.)
            
        Returns:
            SearchResult with retrieved memories and formatted context
        """
        start_time = time.time()

        session = await self._get_session()
        url = f"{self.api_url}/search/memory"

        # Read search params from config, allow kwargs override
        effective_top_k = top_k if top_k is not None else self.search_config.get("top_k", 10)
        preference_limit = kwargs.get(
            "preference_limit_number",
            self.search_config.get("preference_limit_number", 6)
        )

        payload = {
            "query": query,
            "user_id": user_id,
            "memory_limit_number": effective_top_k,
            "preference_limit_number": preference_limit
        }
        
        for attempt in range(self.max_retries):
            try:
                async with self.rate_limiter:
                    async with session.post(url, json=payload) as response:
                        if response.status != 200:
                            text = await response.text()
                            raise Exception(f"HTTP {response.status}: {text}")
                        
                        result = await response.json()
                        # Check for success: code == 0
                        if result.get("code") != 0:
                            raise Exception(f"API error: {result}")
                        
                        # Parse search results from memory_detail_list
                        data = result.get("data", {})
                        memory_list = data.get("memory_detail_list", [])
                        
                        # Extract memories
                        all_memories = []
                        for mem in memory_list:
                            memory_value = mem.get("memory_value", "")
                            memory_key = mem.get("memory_key", "")
                            if memory_value:
                                all_memories.append(memory_value)
                        
                        # Format context for LLM
                        context = self._format_search_context(
                            memories=all_memories,
                            pref_string=""
                        )
                        
                        duration_ms = (time.time() - start_time) * 1000
                        
                        return SearchResult(
                            question_id=kwargs.get("question_id", ""),
                            query=query,
                            retrieved_memories=all_memories,
                            context=context,
                            search_duration_ms=duration_ms,
                            metadata={
                                "memories_count": len(all_memories),
                            }
                        )
                        
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
    
    def _format_search_context(
        self,
        memories: List[str],
        pref_string: str = ""
    ) -> str:
        """
        Format retrieved memories into context string for LLM.

        Args:
            memories: List of memory strings
            pref_string: Preference string from API

        Returns:
            Formatted context string
        """
        if not memories and not pref_string:
            return "(No memories retrieved)"

        memory_text = "\n".join(f"- {mem}" for mem in memories if mem)
        if pref_string:
            memory_text += f"\n{pref_string}"

        return memory_text


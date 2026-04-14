"""
Memobase Adapter for multi-person group chat evaluation.

Implements Add functionality for Memobase memory system.
All messages are sent with role="user" and formatted as:
[Group: X][Speaker: Name]content

Uses ChatBlob for batch message ingestion with created_at timestamps.

Note: Memobase requires a UUID user_id (returned by add_user). During add stage,
we save the mapping to a file. During search, we load it to get the actual UUID.
"""
import asyncio
import json
from datetime import timezone, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import time

from eval.src.adapters.base import BaseAdapter
from eval.src.core.data_models import Dataset, GroupChatMessage, AddResult, SearchResult
from eval.src.utils.logger import get_console, print_success, print_warning


class MemobaseAdapter(BaseAdapter):
    """
    Memobase memory system adapter for multi-person group chat.
    
    Formats all messages as:
    - role: "user" (always)
    - content: "[Group: X][Speaker: Name]original_content"
    - created_at: ISO timestamp
    - alias: speaker name (Memobase character alias)
    
    Uses ChatBlob for batch ingestion.
    
    Config example:
    ```yaml
    name: "memobase"
    project_url: "${MEMOBASE_PROJECT_URL}"
    api_key: "${MEMOBASE_API_KEY}"
    batch_size: 20
    ```
    """
    
    def __init__(self, config: Dict[str, Any], output_dir: Optional[Path] = None):
        super().__init__(config, output_dir)
        
        # Import Memobase client
        try:
            from memobase import MemoBaseClient, ChatBlob
            self.ChatBlob = ChatBlob
        except ImportError:
            raise ImportError(
                "Memobase client not installed. "
                "Please install: pip install memobase"
            )
        
        # API configuration
        project_url = config.get("project_url", "")
        if not project_url:
            raise ValueError("Memobase project URL is required. Set 'project_url' in config or MEMOBASE_PROJECT_URL env var.")
        
        api_key = config.get("api_key", "")
        if not api_key:
            raise ValueError("Memobase API key is required. Set 'api_key' in config or MEMOBASE_API_KEY env var.")
        
        self.client = MemoBaseClient(
            project_url=project_url,
            api_key=api_key
        )
        
        # Batch configuration
        self.batch_size = config.get("batch_size", 20)
        self.batch_delay = config.get("batch_delay", 2.0)  # Delay between batches
        self.max_retries = config.get("max_retries", 5)

        # Search configuration
        self.search_config = config.get("search", {})
        
        self.console = get_console()
        
        # User mapping file to store user_id -> memobase_uuid mapping
        self.user_mapping_file = self.output_dir / "memobase_users.json"
        
        self.console.print("✅ MemobaseAdapter initialized", style="bold green")
        self.console.print(f"   Project URL: {project_url}")
        self.console.print(f"   Batch Size: {self.batch_size}")
        self.console.print(f"   Batch Delay: {self.batch_delay}s")
    
    def _save_user_mapping(self, original_id: str, memobase_uuid: str):
        """Save user_id -> memobase_uuid mapping to file."""
        mapping = {}
        if self.user_mapping_file.exists():
            with open(self.user_mapping_file, "r", encoding="utf-8") as f:
                mapping = json.load(f)
        
        mapping[original_id] = memobase_uuid
        
        with open(self.user_mapping_file, "w", encoding="utf-8") as f:
            json.dump(mapping, f, indent=2)
    
    def _load_user_mapping(self, original_id: str) -> Optional[str]:
        """Load memobase_uuid for given user_id from mapping file."""
        if not self.user_mapping_file.exists():
            return None
        
        with open(self.user_mapping_file, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        
        return mapping.get(original_id)
    
    def _find_user_by_name(self, name: str) -> Optional[str]:
        """
        Find Memobase user UUID by name using get_all_users API.
        
        Args:
            name: User name to search for (e.g., "004")
            
        Returns:
            UUID if found, None otherwise
        """
        try:
            users = self.client.get_all_users()
            if not users:
                return None
            for user in users:
                if user is None:
                    continue
                # Handle both dict-like and object-like user representations
                if isinstance(user, dict):
                    additional_fields = user.get("additional_fields", {})
                    user_name = additional_fields.get("name") if additional_fields else None
                    user_id = user.get("id")
                else:
                    # Object-like: try attribute access
                    additional_fields = getattr(user, "additional_fields", None)
                    if additional_fields:
                        user_name = additional_fields.get("name") if isinstance(additional_fields, dict) else getattr(additional_fields, "name", None)
                    else:
                        user_name = None
                    user_id = getattr(user, "id", None)
                
                if user_name == name and user_id:
                    # Save to mapping for future use
                    self._save_user_mapping(name, user_id)
                    return user_id
            return None
        except Exception as e:
            self.console.print(f"      ⚠️  get_all_users failed: {e}", style="yellow")
            return None
    
    async def add(
        self,
        dataset: Dataset,
        user_id: str,
        days_to_process: Optional[List[str]] = None,
        **kwargs
    ) -> AddResult:
        """
        Add dataset to Memobase memory system.
        
        Args:
            dataset: Dataset with group chat data
            user_id: User ID for Memobase
            days_to_process: Optional list of dates to process (None = all)
            **kwargs: Additional parameters
            
        Returns:
            AddResult with statistics
        """
        self.console.print(f"\n{'='*60}", style="bold cyan")
        self.console.print("Stage: Add (Memobase)", style="bold cyan")
        self.console.print(f"{'='*60}", style="bold cyan")
        self.console.print(f"User ID: {user_id}")
        self.console.print(f"Dataset: {dataset.name}")
        
        # Optional resume behavior: skip all days prior to start_date (inclusive resume).
        # Example: start_date="2025-05-22" => process days where day.date >= "2025-05-22".
        start_date = kwargs.get("start_date")
        start_date_obj = None
        if start_date:
            try:
                start_date_obj = datetime.strptime(start_date, "%Y-%m-%d").date()
            except Exception:
                raise ValueError(f"Invalid --start-date '{start_date}'. Expected format YYYY-MM-DD, e.g. 2025-05-22")

        # Determine which days to process
        if days_to_process:
            days = [d for d in dataset.days if d.date in days_to_process]
        else:
            days = dataset.days

        if start_date_obj is not None:
            days = [
                d for d in days
                if datetime.strptime(d.date, "%Y-%m-%d").date() >= start_date_obj
            ]
            self.console.print(f"Resume start date: {start_date} (inclusive)", style="dim")
        
        self.console.print(f"Days to process: {len(days)}")
        
        # Create or get user (resume-safe):
        # 1) Prefer local mapping file (stable UUID) to avoid accidental new user creation
        # 2) If missing, try to find by name via get_all_users()
        # 3) Only if still missing, create a new user via add_user()
        memobase_user_id = self._load_user_mapping(user_id)
        if memobase_user_id:
            self.console.print(f"   ✅ User mapped: {user_id} -> {memobase_user_id}", style="dim green")
        else:
            memobase_user_id = self._find_user_by_name(user_id)
            if memobase_user_id:
                self.console.print(f"   ✅ User found: {user_id} -> {memobase_user_id}", style="dim green")
            else:
                memobase_user_id = self.client.add_user({"name": user_id})
                self.console.print(f"   ✅ User created: {memobase_user_id}", style="dim green")
                # Save mapping for later search / future resume
                self._save_user_mapping(user_id, memobase_user_id)
        
        # Get user object ONCE and reuse it (avoids extra API calls per batch)
        memobase_user = self.client.get_user(memobase_user_id)
        
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
                        await self._send_batch(batch, memobase_user)
                        total_messages += len(batch)
                        self.console.print(
                            f"      ✅ Batch {batch_idx + 1}/{len(batches)} sent ({len(batch)} messages)",
                            style="dim green",
                        )
                        # Delay between batches to avoid rate limiting
                        if batch_idx < len(batches) - 1 and self.batch_delay > 0:
                            await asyncio.sleep(self.batch_delay)
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
                "memobase_user_id": memobase_user_id,
                "dataset": dataset.name,
            }
        )
    
    def _format_message(self, msg: GroupChatMessage) -> Dict[str, Any]:
        """
        Format a GroupChatMessage for Memobase API.
        
        Input: GroupChatMessage with speaker, content, timestamp, group
        Output: {
            "role": "user",
            "content": "[Group: X][Speaker: Name]content",
            "created_at": "2025-01-09T09:32:15",
            "alias": "Alice"
        }
        """
        # Format content with group and speaker prefix
        formatted_content = f"[Group: {msg.group}][Speaker: {msg.speaker}]{msg.content}"
        
        # Format timestamp
        ts = msg.timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        created_at = ts.strftime("%Y-%m-%dT%H:%M:%S")
        
        return {
            "role": "user",
            "content": formatted_content,
            "created_at": created_at,
            # Memobase character alias: use speaker name for multi-person attribution.
            "alias": msg.speaker,
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
    
    async def _send_batch(self, messages: List[Dict[str, Any]], memobase_user):
        """
        Send a batch of messages to Memobase API using ChatBlob.
        
        Args:
            messages: List of formatted messages
            memobase_user: Memobase user object (pre-fetched, reused)
            
        Raises:
            Exception: If API call fails after retries
        """
        for attempt in range(self.max_retries):
            try:
                # Run synchronous Memobase operations in executor
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    self._sync_send_batch,
                    messages,
                    memobase_user
                )
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
    
    def _sync_send_batch(self, messages: List[Dict[str, Any]], memobase_user):
        """
        Synchronous batch send for Memobase.
        
        Args:
            messages: List of formatted messages
            memobase_user: Memobase user object (reused)
        """
        blob = self.ChatBlob(messages=messages)
        memobase_user.insert(blob)
        memobase_user.flush(sync=True)
    
    async def close(self):
        """Close client resources."""
        pass  # Memobase client doesn't require explicit cleanup
    
    async def search(
        self,
        query: str,
        user_id: str,
        top_k: int = 10,
        **kwargs
    ) -> SearchResult:
        """
        Search memories in Memobase using context() method.
        
        The context() method returns relevant memories based on the query,
        formatted as a context string ready for LLM consumption.
        
        Note: user_id should be the original ID (e.g., "004"). We will look up
        the actual Memobase UUID from the mapping file saved during add stage.
        
        Args:
            query: Search query (usually question text)
            user_id: Original user ID (we look up the UUID from mapping)
            top_k: Not directly used; max_token_size controls context length
            **kwargs: Additional parameters:
                - max_token_size: Max tokens for context (default: 1000)
                - event_similarity_threshold: Similarity threshold (default: 0.2)
                - memobase_user_id: Override UUID directly if known
                
        Returns:
            SearchResult with retrieved memories and formatted context
        """
        start_time = time.time()

        # Read search params from config, allow kwargs override
        max_token_size = kwargs.get(
            "max_token_size",
            self.search_config.get("max_token_size", 1000)
        )
        event_similarity_threshold = kwargs.get(
            "event_similarity_threshold",
            self.search_config.get("event_similarity_threshold", 0.2)
        )
        
        # Look up actual Memobase UUID from mapping or by searching users
        memobase_uuid = kwargs.get("memobase_user_id")
        if not memobase_uuid:
            memobase_uuid = self._load_user_mapping(user_id)
        if not memobase_uuid:
            # Try to find user by name using get_all_users API
            memobase_uuid = self._find_user_by_name(user_id)
        if not memobase_uuid:
            raise ValueError(
                f"No Memobase UUID found for user_id '{user_id}'. "
                f"User not found in Memobase."
            )
        
        self.console.print(f"      Using Memobase UUID: {memobase_uuid}", style="dim")
        
        for attempt in range(self.max_retries):
            try:
                # Run synchronous Memobase operation in executor
                loop = asyncio.get_event_loop()
                context = await loop.run_in_executor(
                    None,
                    self._sync_search,
                    query,
                    memobase_uuid,  # Use UUID, not original user_id
                    max_token_size,
                    event_similarity_threshold
                )
                
                # Parse memories from context
                # Memobase context format has memories as lines starting with "- "
                memories = [
                    line.strip()[2:]  # Remove "- " prefix
                    for line in context.split("\n")
                    if line.strip().startswith("- ")
                ]
                
                duration_ms = (time.time() - start_time) * 1000
                
                return SearchResult(
                    question_id=kwargs.get("question_id", ""),
                    query=query,
                    retrieved_memories=memories,
                    context=context,
                    search_duration_ms=duration_ms,
                    metadata={
                        "max_token_size": max_token_size,
                        "event_similarity_threshold": event_similarity_threshold,
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
    
    def _sync_search(
        self,
        query: str,
        user_id: str,
        max_token_size: int,
        event_similarity_threshold: float
    ) -> str:
        """
        Synchronous search for Memobase.
        
        Args:
            query: Search query
            user_id: User ID
            max_token_size: Max tokens for context
            event_similarity_threshold: Similarity threshold
            
        Returns:
            Context string from Memobase
        """
        u = self.client.get_user(user_id)
        context = u.context(
            max_token_size=max_token_size,
            chats=[{"role": "user", "content": query}],
            event_similarity_threshold=event_similarity_threshold,
            fill_window_with_events=True
        )
        return context


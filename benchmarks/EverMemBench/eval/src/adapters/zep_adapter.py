"""
Zep Adapter for multi-person group chat evaluation.

Implements Add functionality for Zep memory system using Graph API.
Uses graph.add() for multi-person support with timestamp prefix:
[2025-01-09T09:32:15][Group: X][Speaker: Name]content

Per Zep docs (group chat / multi-person), messages should not be associated with a
single user. We therefore write to a standalone graph (`graph_id`) created via
`graph.create()`, and add messages via `graph.add(graph_id=...)`.

Timestamps:
Zep Cloud Graph API supports `created_at` on `graph.add()`, so we send timestamps
via that field (instead of embedding the timestamp into the message content).
"""
import asyncio
from datetime import timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import time

from eval.src.adapters.base import BaseAdapter
from eval.src.core.data_models import Dataset, GroupChatMessage, AddResult, SearchResult
from eval.src.utils.logger import get_console, print_success, print_warning


class ZepAdapter(BaseAdapter):
    """
    Zep memory system adapter for multi-person group chat.
    
    Uses Graph API (graph.add) for multi-person support.
    
    Writes to a standalone graph (graph_id) so the chat isn't tied to a single user.

    Formats all messages with group/speaker prefix:
    - data: "[Group: X][Speaker: Name]original_content"
    - type: "message"
    
    Config example:
    ```yaml
    name: "zep"
    api_key: "${ZEP_API_KEY}"
    batch_size: 10
    ```
    """
    
    def __init__(self, config: Dict[str, Any], output_dir: Optional[Path] = None):
        super().__init__(config, output_dir)
        
        # Import Zep async client
        try:
            from zep_cloud.client import AsyncZep
        except ImportError:
            raise ImportError(
                "Zep client not installed. "
                "Please install: pip install zep-cloud"
            )
        
        # API configuration
        api_key = config.get("api_key", "")
        if not api_key:
            raise ValueError("Zep API key is required. Set 'api_key' in config or ZEP_API_KEY env var.")
        
        self.client = AsyncZep(api_key=api_key)
        
        # Zep graph.add is per-message; batch_size controls concurrent group processing
        self.batch_size = config.get("batch_size", 10)
        self.max_retries = config.get("max_retries", 5)
        self.add_interval = config.get("add_interval", 0.1)  # Interval between messages

        # Search configuration
        self.search_config = config.get("search", {})
        
        self.console = get_console()
        
        self.console.print("✅ ZepAdapter initialized", style="bold green")
        self.console.print(f"   Batch Size: {self.batch_size}")
        self.console.print(f"   Add Interval: {self.add_interval}s")
    
    async def add(
        self,
        dataset: Dataset,
        user_id: str,
        days_to_process: Optional[List[str]] = None,
        **kwargs
    ) -> AddResult:
        """
        Add dataset to Zep memory system using Graph API.
        
        Args:
            dataset: Dataset with group chat data
            user_id: User ID for Zep graph
            days_to_process: Optional list of dates to process (None = all)
            **kwargs: Additional parameters
            
        Returns:
            AddResult with statistics
        """
        self.console.print(f"\n{'='*60}", style="bold cyan")
        self.console.print("Stage: Add (Zep Graph)", style="bold cyan")
        self.console.print(f"{'='*60}", style="bold cyan")
        graph_id = kwargs.get("graph_id") or user_id
        self.console.print(f"Graph ID: {graph_id}")
        self.console.print(f"Dataset: {dataset.name}")
        
        # Determine which days to process
        if days_to_process:
            days = [d for d in dataset.days if d.date in days_to_process]
        else:
            days = dataset.days
        
        self.console.print(f"Days to process: {len(days)}")

        # Create standalone graph for group chat (do not associate with a single user)
        try:
            await self.client.graph.create(graph_id=graph_id)
            self.console.print(f"   ✅ Graph {graph_id} created", style="dim green")
        except Exception as e:
            error_msg = str(e).lower()
            if "already exists" in error_msg or "conflict" in error_msg or "409" in str(e):
                self.console.print(f"   ℹ️  Graph {graph_id} already exists", style="dim")
            else:
                self.console.print(f"   ⚠️  Graph create warning: {e}", style="yellow")
        
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
                        await self._send_batch(batch, graph_id)
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
                "graph_id": graph_id,
                "dataset": dataset.name,
            }
        )
    
    def _format_message(self, msg: GroupChatMessage) -> Dict[str, str]:
        """
        Format a GroupChatMessage for Zep Graph API.

        Input: GroupChatMessage with speaker, content, timestamp, group
        Output:
          {
            "data": "[Group: X][Speaker: Name]content",
            "created_at": "2025-01-09T09:32:15Z"
          }
        """
        # created_at: send via API field (Zep expects ISO 8601 string)
        ts = msg.timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        created_at = ts.isoformat(timespec="seconds").replace("+00:00", "Z")

        data = f"[Group: {msg.group}][Speaker: {msg.speaker}]{msg.content}"
        return {"data": data, "created_at": created_at}
    
    def _split_into_batches(self, messages: List[Dict[str, str]]) -> List[List[Dict[str, str]]]:
        """
        Split messages into batches.
        
        Args:
            messages: List of formatted message dicts
            
        Returns:
            List of batches
        """
        batches = []
        for i in range(0, len(messages), self.batch_size):
            batches.append(messages[i:i + self.batch_size])
        return batches
    
    async def _send_batch(self, messages: List[Dict[str, str]], graph_id: str):
        """
        Send a batch of messages to Zep Graph API.
        
        Each message is sent individually to graph.add().
        
        Args:
            messages: List of formatted message dicts: {"data": ..., "created_at": ...}
            graph_id: Standalone graph id
            
        Raises:
            Exception: If API call fails after retries
        """
        for msg in messages:
            for attempt in range(self.max_retries):
                try:
                    await self.client.graph.add(
                        graph_id=graph_id,
                        type="message",
                        data=msg["data"],
                        created_at=msg.get("created_at"),
                    )
                    
                    # Add interval between messages
                    if self.add_interval > 0:
                        await asyncio.sleep(self.add_interval)
                    
                    break  # Success, move to next message
                    
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
    
    async def close(self):
        """Close client resources."""
        pass  # Zep client doesn't require explicit cleanup
    
    async def search(
        self,
        query: str,
        user_id: str,
        top_k: int = 10,
        **kwargs
    ) -> SearchResult:
        """
        Search memories in Zep using Graph Search API.
        
        Uses graph.search() with both edges (facts) and nodes (entities) scopes.
        The graph_id is the user_id passed to this method.
        
        Args:
            query: Search query (usually question text)
            user_id: User ID which is used as graph_id
            top_k: Total number of results (split between edges and nodes)
            **kwargs: Additional parameters:
                - graph_id: Override graph_id (default: user_id)
                - reranker: Reranker to use (default: "cross_encoder" for edges, "rrf" for nodes)
                - max_query_length: Max query length (default: 400, Zep API limit)
                
        Returns:
            SearchResult with retrieved memories and formatted context
        """
        start_time = time.time()

        graph_id = kwargs.get("graph_id") or user_id

        # Read search params from config, allow kwargs override
        effective_top_k = top_k if top_k is not None else self.search_config.get("top_k", 10)
        edges_limit = effective_top_k // 2
        nodes_limit = effective_top_k - edges_limit

        # Zep API limit: query cannot be longer than 400 characters
        max_query_length = kwargs.get(
            "max_query_length",
            self.search_config.get("max_query_length", 400)
        )
        original_query = query
        if len(query) > max_query_length:
            # Keep the tail of the query (most recent / most specific part).
            # This preserves suffix constraints (e.g., options, dates, names) which are often
            # more informative than generic prefixes.
            query = query[-max_query_length:]
            self.console.print(
                f"      ⚠️  Query truncated from {len(original_query)} to {max_query_length} chars",
                style="yellow dim"
            )
        
        for attempt in range(self.max_retries):
            try:
                # Search edges (facts)
                reranker_edges = kwargs.get(
                    "reranker_edges",
                    self.search_config.get("reranker_edges", "cross_encoder")
                )
                edges_results = await self.client.graph.search(
                    graph_id=graph_id,
                    query=query,
                    scope="edges",
                    limit=edges_limit,
                    reranker=reranker_edges
                )

                # Search nodes (entities)
                reranker_nodes = kwargs.get(
                    "reranker_nodes",
                    self.search_config.get("reranker_nodes", "rrf")
                )
                nodes_results = await self.client.graph.search(
                    graph_id=graph_id,
                    query=query,
                    scope="nodes",
                    limit=nodes_limit,
                    reranker=reranker_nodes
                )
                
                # Parse results
                facts = []
                for edge in getattr(edges_results, "edges", []) or []:
                    fact = getattr(edge, "fact", "")
                    valid_at = getattr(edge, "valid_at", "")
                    if fact:
                        facts.append(f"{fact} (event_time: {valid_at})")
                
                entities = []
                for node in getattr(nodes_results, "nodes", []) or []:
                    name = getattr(node, "name", "")
                    summary = getattr(node, "summary", "")
                    if name and summary:
                        entities.append(f"{name}: {summary}")
                
                # Combine all memories
                all_memories = facts + entities
                
                # Format context for LLM using Zep template
                context = self._format_search_context(facts, entities)
                
                duration_ms = (time.time() - start_time) * 1000
                
                return SearchResult(
                    question_id=kwargs.get("question_id", ""),
                    query=query,
                    retrieved_memories=all_memories,
                    context=context,
                    search_duration_ms=duration_ms,
                    metadata={
                        "graph_id": graph_id,
                        "facts_count": len(facts),
                        "entities_count": len(entities),
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
        facts: List[str],
        entities: List[str]
    ) -> str:
        """
        Format retrieved facts and entities into context string for LLM.
        
        Uses Zep's recommended template format with FACTS and ENTITIES sections.
        
        Args:
            facts: List of fact strings with event times
            entities: List of entity strings (name: summary)
            
        Returns:
            Formatted context string
        """
        facts_text = "\n".join(f"  - {f}" for f in facts) if facts else "  (no facts found)"
        entities_text = "\n".join(f"  - {e}" for e in entities) if entities else "  (no entities found)"
        
        return f"""FACTS and ENTITIES represent relevant context to the current conversation.

# These are the most relevant facts for the conversation along with the datetime of the event that the fact refers to.
# If a fact mentions something happening a week ago, then the datetime will be the date time of last week and not the datetime
# of when the fact was stated.
# Timestamps in memories represent the actual time the event occurred, not the time the event was mentioned in a message.
    
<FACTS>
{facts_text}
</FACTS>

# These are the most relevant entities
# ENTITY_NAME: entity summary
<ENTITIES>
{entities_text}
</ENTITIES>"""


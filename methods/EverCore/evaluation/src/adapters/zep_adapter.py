"""
Zep Adapter - adapt Zep online API for evaluation framework.
Reference: https://www.getzep.com/

Key features:
- Knowledge graph-based memory with facts (edges) and entities (nodes)
- Dual search: searches both nodes (entities) and edges (facts)
- Temporal awareness with created_at timestamps
- Simple design: one conversation → one graph, speaker info in content

Version: Zep v3+ API
Migration from v2:
- group.add(group_id) → graph.create(graph_id)
- graph.add(..., group_id=...) → graph.add(..., graph_id=...)
- graph.search(..., group_id=...) → graph.search(..., graph_id=...)

Design:
- graph_id = conversation_id (simple 1:1 mapping)
- Speaker info embedded in content: "Jane: Hello, I work at Acme"
- Zep automatically extracts entities and facts from conversations
"""

import asyncio
import json
from datetime import timezone
from pathlib import Path
from typing import Any, Dict, List

from rich.console import Console

from evaluation.src.adapters.online_base import OnlineAPIAdapter
from evaluation.src.adapters.registry import register_adapter
from evaluation.src.core.data_models import Conversation, SearchResult


@register_adapter("zep")
class ZepAdapter(OnlineAPIAdapter):
    """
    Zep online API adapter (Simplified design following reference implementation).

    Key simplifications:
    - One conversation = One graph (no per-speaker graphs)
    - Speaker info in message content (Zep handles entity extraction)
    - Single search per query (no dual-perspective merging)

    Config example:
    ```yaml
    adapter: "zep"
    api_key: "${ZEP_API_KEY}"
    search:
      top_k: 20
      reranker_nodes: "rrf"
      reranker_edges: "cross_encoder"
    ```
    """

    def __init__(self, config: dict, output_dir: Path = None):
        super().__init__(config, output_dir)

        # Import Zep async client
        try:
            from zep_cloud.client import AsyncZep
        except ImportError:
            raise ImportError(
                "Zep client not installed. " "Please install: pip install zep-cloud"
            )

        # Initialize Zep async client
        api_key = config.get("api_key", "")
        if not api_key:
            raise ValueError("Zep API key is required. Set 'api_key' in config.")

        self.client = AsyncZep(api_key=api_key)
        self.max_retries = config.get("max_retries", 5)
        self.poll_interval = config.get(
            "poll_interval", 5
        )  # Poll interval for checking episode processed status (seconds)
        self.add_message_interval = config.get(
            "add_message_interval", 0.2
        )  # Interval between sending messages (seconds)
        self.console = Console()

        # Search configuration
        search_config = config.get("search", {})
        self.reranker_nodes = search_config.get("reranker_nodes", "rrf")
        self.reranker_edges = search_config.get("reranker_edges", "cross_encoder")

        print(f"   Reranker (Nodes): {self.reranker_nodes}")
        print(f"   Reranker (Edges): {self.reranker_edges}")
        print(f"   Processing Mode: Serial (add → wait processed → next)")
        print(f"   Add Message Interval: {self.add_message_interval}s")
        print(f"   Poll Interval: {self.poll_interval}s")

    async def _add_user_messages(
        self, conv: Conversation, messages: List[Dict[str, Any]], speaker: str, **kwargs
    ) -> Any:
        """
        Add messages to Zep graph with concurrent control.

        Simplified: All messages go to the same graph (conversation-level),
        with speaker info embedded in content.

        Args:
            conv: Original conversation object
            messages: Formatted message list
            speaker: "speaker_a" or "speaker_b" (ignored in this simple design)
            **kwargs: Extra parameters

        Returns:
            None
        """
        # Simple: graph_id = conversation_id
        graph_id = conv.conversation_id

        # Skip if this is speaker_b (we already added all messages with speaker_a)
        # This avoids duplicate additions in dual-perspective mode
        if speaker == "speaker_b":
            return None

        # Ensure graph exists before adding messages
        try:
            await self.client.graph.create(graph_id=graph_id)
        except Exception:
            # Graph already exists, which is fine
            pass

        # Log info
        is_fake_timestamp = (
            conv.messages[0].metadata.get("is_fake_timestamp", False)
            if conv.messages
            else False
        )

        self.console.print(
            f"   📤 Adding to graph {graph_id}: {len(conv.messages)} messages (serial processing)",
            style="dim",
        )
        if is_fake_timestamp:
            self.console.print(f"   ⚠️  Using fake timestamp", style="yellow")

        # Serial processing: add message → wait for processed → next message
        # Simple and reliable - ensures strict ordering
        for i, msg in enumerate(conv.messages):
            # Get timestamp in UTC format
            timestamp = None
            if msg.timestamp:
                # If timestamp doesn't have timezone info, assume UTC
                if msg.timestamp.tzinfo is None:
                    timestamp_utc = msg.timestamp.replace(tzinfo=timezone.utc)
                else:
                    # Convert to UTC if it has other timezone
                    timestamp_utc = msg.timestamp.astimezone(timezone.utc)
                timestamp = timestamp_utc.isoformat()

            # Include speaker name in content (Zep will extract entities from this)
            data = f"{msg.sender_name}: {msg.content}"

            # Add message with retry mechanism
            episode = None
            for attempt in range(self.max_retries):
                try:
                    episode = await self.client.graph.add(
                        data=data,
                        type="message",
                        created_at=timestamp,
                        graph_id=graph_id,
                    )
                    break
                except Exception as e:
                    if attempt < self.max_retries - 1:
                        self.console.print(
                            f"   ⚠️  Message {i+1}/{len(conv.messages)} Add Retry {attempt + 1}/{self.max_retries}: {e}",
                            style="yellow",
                        )
                        await asyncio.sleep(2**attempt)
                    else:
                        self.console.print(
                            f"   ❌ Message {i+1}/{len(conv.messages)} Failed after {self.max_retries} retries: {e}",
                            style="red",
                        )
                        raise e

            # Wait for episode to be processed
            if episode and hasattr(episode, 'uuid_'):
                poll_count = 0
                while True:
                    try:
                        retrieved_episode = await self.client.graph.episode.get(
                            uuid_=episode.uuid_
                        )
                        if (
                            hasattr(retrieved_episode, 'processed')
                            and retrieved_episode.processed
                        ):
                            self.console.print(
                                f"   ✅ Message {i+1}/{len(conv.messages)} processed (polled {poll_count} times)",
                                style="dim green",
                            )
                            break
                        poll_count += 1
                        await asyncio.sleep(self.poll_interval)
                    except Exception as e:
                        self.console.print(
                            f"   ⚠️  Message {i+1}/{len(conv.messages)} Poll error: {e}",
                            style="yellow",
                        )
                        await asyncio.sleep(self.poll_interval)

        self.console.print(
            f"   🎉 All {len(conv.messages)} messages processed!", style="bold green"
        )

        return None

    async def _search_single_user(
        self, query: str, conversation_id: str, user_id: str, top_k: int, **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Search memories (simplified: single graph per conversation).

        Performs dual search (nodes + edges) on a single graph.

        Args:
            query: Query text
            conversation_id: Conversation ID (used as graph_id)
            user_id: User ID (ignored, we use conversation_id as graph_id)
            top_k: Number of results to retrieve (per search type)
            **kwargs: Additional parameters

        Returns:
            List of search results combining nodes and edges
        """
        # Simple: graph_id = conversation_id
        graph_id = conversation_id

        try:
            # Dual search (nodes and edges)
            search_results = await asyncio.gather(
                self.client.graph.search(
                    query=query,
                    graph_id=graph_id,
                    scope='nodes',
                    reranker=self.reranker_nodes,
                    limit=top_k,
                ),
                self.client.graph.search(
                    query=query,
                    graph_id=graph_id,
                    scope='edges',
                    reranker=self.reranker_edges,
                    limit=top_k,
                ),
            )

            nodes = (
                search_results[0].nodes if hasattr(search_results[0], 'nodes') else []
            )
            edges = (
                search_results[1].edges if hasattr(search_results[1], 'edges') else []
            )

            # Debug output
            self.console.print(f"\n[DEBUG] Zep Search Results:", style="yellow")
            self.console.print(f"  Query: {query}", style="dim")
            self.console.print(f"  Graph ID: {graph_id}", style="dim")
            self.console.print(
                f"  Nodes: {len(nodes)}, Edges: {len(edges)}", style="dim"
            )

        except Exception as e:
            self.console.print(f"❌ Zep search error: {e}", style="red")
            return []

        # Convert to standard format
        results = []

        # Add edges (facts) first
        for edge in edges:
            fact = getattr(edge, 'fact', '')
            valid_at = getattr(edge, 'valid_at', '')
            score = getattr(edge, 'score', 0.0)

            results.append(
                {
                    "content": f"FACT (event_time: {valid_at}): {fact}",
                    "score": score,
                    "user_id": graph_id,
                    "metadata": {
                        "type": "fact",
                        "fact": fact,
                        "valid_at": valid_at,
                        "graph_id": graph_id,
                    },
                }
            )

        # Add nodes (entities)
        for node in nodes:
            name = getattr(node, 'name', '')
            summary = getattr(node, 'summary', '')
            score = getattr(node, 'score', 0.0)

            results.append(
                {
                    "content": f"ENTITY ({name}): {summary}",
                    "score": score,
                    "user_id": graph_id,
                    "metadata": {
                        "type": "entity",
                        "name": name,
                        "summary": summary,
                        "graph_id": graph_id,
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
        Build SearchResult (simplified: no dual-perspective merging needed).

        Args:
            query: Query text
            conversation_id: Conversation ID
            results: Search results from _search_single_user
            user_id: User ID (actually graph_id)
            top_k: Number of results requested
            **kwargs: Additional parameters

        Returns:
            SearchResult with formatted_context
        """
        # Separate facts and entities
        facts = [r for r in results if r["metadata"]["type"] == "fact"]
        entities = [r for r in results if r["metadata"]["type"] == "entity"]

        # Format facts and entities
        facts_text = "\n".join(
            [
                f"  - {r['metadata']['fact']} (event_time: {r['metadata']['valid_at']})"
                for r in facts
            ]
        )
        entities_text = "\n".join(
            [
                f"  - {r['metadata']['name']}: {r['metadata']['summary']}"
                for r in entities
            ]
        )

        if not facts_text:
            facts_text = "  (No facts found)"
        if not entities_text:
            entities_text = "  (No entities found)"

        # Build formatted context using Zep's template from prompts.yaml
        zep_template = (
            self._prompts.get("online_api", {}).get("templates", {}).get("zep", "")
        )
        formatted_context = zep_template.format(
            facts=facts_text, entities=entities_text
        )

        return SearchResult(
            query=query,
            conversation_id=conversation_id,
            results=results,
            retrieval_metadata={
                "system": "zep",
                "top_k": top_k,
                "dual_perspective": False,
                "user_ids": [conversation_id],
                "graph_id": conversation_id,
                "formatted_context": formatted_context,
                "facts_count": len(facts),
                "entities_count": len(entities),
            },
        )

    async def search(
        self, query: str, conversation_id: str, index: Any, **kwargs
    ) -> SearchResult:
        """
        Retrieve relevant memories (simplified: no dual-perspective logic).

        Since Zep uses one graph per conversation (speaker info in content),
        we directly search the conversation graph without dual-perspective handling.

        Args:
            query: Query text
            conversation_id: Conversation ID (used as graph_id)
            index: Index metadata (not used)
            **kwargs: Optional parameters (top_k, etc.)

        Returns:
            SearchResult with standard format
        """
        # Get top_k from kwargs or config
        default_top_k = self.config.get("search", {}).get("top_k", 10)
        top_k = kwargs.get("top_k", default_top_k)

        # Direct search (no dual-perspective logic)
        results = await self._search_single_user(
            query=query,
            conversation_id=conversation_id,
            user_id=conversation_id,  # user_id = conversation_id in simplified design
            top_k=top_k,
            **kwargs,
        )

        # Build result (no dual-perspective merging)
        return self._build_single_search_result(
            query=query,
            conversation_id=conversation_id,
            results=results,
            user_id=conversation_id,
            top_k=top_k,
            **kwargs,
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
        Build dual-perspective search result (NOT USED by Zep).

        This method is required by the base class but not used by Zep.
        Zep uses simplified single-graph design without dual-perspective.

        Raises:
            NotImplementedError: Always raises as Zep doesn't use dual-perspective
        """
        raise NotImplementedError(
            "Zep adapter uses simplified single-graph design. "
            "Use search() method instead, which calls _build_single_search_result()."
        )

    def _get_answer_prompt(self) -> str:
        """
        Return answer prompt for Zep.

        Loads from prompts.yaml (answer_prompt_zep).
        """
        # Load from prompts.yaml
        return (
            self._prompts.get("online_api", {})
            .get("default", {})
            .get("answer_prompt_zep", "")
        )

    def get_system_info(self) -> Dict[str, Any]:
        """Return system info."""
        return {
            "name": "Zep",
            "type": "online_api",
            "description": "Zep - Knowledge Graph Architecture for Agent Memory",
            "adapter": "ZepAdapter",
        }

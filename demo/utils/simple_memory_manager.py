"""Simple Memory Manager - Simplified Memory Manager (HTTP API Version)

Encapsulates all HTTP API call details and provides the simplest interface.
"""

import re
import asyncio
import httpx
from typing import List, Dict, Any
from common_utils.datetime_utils import (
    get_now_with_timezone,
    get_timezone,
    to_iso_format,
)


def extract_event_time_from_memory(mem: Dict[str, Any]) -> str:
    """Extract actual event time from memory data

    Extraction priority:
    1. Date in 'subject' field (parentheses format, e.g., "(2025-08-26)")
    2. Date in 'subject' field (Chinese format, e.g., "2025年8月26日")
    3. Date in 'episode' content (Chinese or ISO format)
    4. Return "N/A" if extraction fails (do not show storage time)

    Args:
        mem: Memory dictionary containing subject, episode, etc.

    Returns:
        Date string in YYYY-MM-DD format, or "N/A"
    """
    subject = mem.get("subject", "")
    episode = mem.get("episode", "")

    # 1. Extract from subject: Match ISO date format inside parentheses (YYYY-MM-DD)
    if subject:
        match = re.search(r'\((\d{4}-\d{2}-\d{2})\)', subject)
        if match:
            return match.group(1)

        # 2. Extract from subject: Match Chinese date format "YYYY年MM月DD日"
        match = re.search(r'(\d{4})年(\d{1,2})月(\d{1,2})日', subject)
        if match:
            year, month, day = match.groups()
            return f"{year}-{month.zfill(2)}-{day.zfill(2)}"

    # 3. Extract from episode (search entire content, no character limit)
    if episode:
        # Match "于YYYY年MM月DD日" or "在YYYY年MM月DD日"
        match = re.search(r'[于在](\d{4})年(\d{1,2})月(\d{1,2})日', episode)
        if match:
            year, month, day = match.groups()
            return f"{year}-{month.zfill(2)}-{day.zfill(2)}"

        # Match ISO format "YYYY-MM-DD"
        match = re.search(r'(\d{4})-(\d{2})-(\d{2})', episode)
        if match:
            return match.group(0)

        # Match other Chinese date formats (without "at" prefix)
        match = re.search(r'(\d{4})年(\d{1,2})月(\d{1,2})日', episode)
        if match:
            year, month, day = match.groups()
            return f"{year}-{month.zfill(2)}-{day.zfill(2)}"

    # 4. Failed to extract event time, return N/A
    return "N/A"


class SimpleMemoryManager:
    """Super Simple Memory Manager

    Uses HTTP API, no need to worry about internal implementation.

    Usage:
        memory = SimpleMemoryManager()
        await memory.store("I love playing soccer")
        results = await memory.search("What sports does the user like?")
    """

    def __init__(
        self,
        base_url: str = "http://localhost:1995",
        group_id: str = "default_group",
        scene: str = "assistant",
    ):
        """Initialize the manager

        Args:
            base_url: API server address (default: localhost:1995)
            group_id: Group ID (default: default_group)
            scene: Scene type (default: "assistant", options: "assistant" or "companion")
        """
        self.base_url = base_url
        self.group_id = group_id
        self.group_name = "Simple Demo Group"
        self.scene = scene
        self.memorize_url = f"{base_url}/api/v1/memories"
        self.retrieve_url = f"{base_url}/api/v1/memories/search"
        self.conversation_meta_url = f"{base_url}/api/v1/memories/conversation-meta"
        self._message_counter = 0
        self._conversation_meta_saved = (
            False  # Flag to indicate if conversation-meta is saved
        )

    async def store(self, content: str, sender: str = "User") -> bool:
        """Store a message

        Args:
            content: Message content
            sender: Sender name (default: "User")

        Returns:
            Success status
        """
        # ========== Save conversation-meta first when storing for the first time ==========
        if not self._conversation_meta_saved:
            await self._save_conversation_meta()

        # Generate unique message ID
        self._message_counter += 1
        now = (
            get_now_with_timezone()
        )  # Use project's unified time utility (with timezone)
        message_id = f"msg_{self._message_counter}_{int(now.timestamp() * 1000)}"

        # Build message data (completely consistent with test_v1api_search.py format)
        message_data = {
            "message_id": message_id,
            "create_time": to_iso_format(
                now
            ),  # Use project's unified time formatting (with timezone)
            "sender": sender,
            "sender_name": sender,  # Consistent with JSON data format
            "type": "text",  # Message type
            "content": content,
            "group_id": self.group_id,
            "group_name": self.group_name,
            "scene": self.scene,  # Use configured scene
        }

        try:
            async with httpx.AsyncClient(timeout=500.0) as client:
                response = await client.post(self.memorize_url, json=message_data)
                response.raise_for_status()
                result = response.json()

                if result.get("status") == "ok":
                    count = result.get("result", {}).get("count", 0)
                    if count > 0:
                        print(
                            f"  ✅ Stored: {content[:40]}... (Extracted {count} memories)"
                        )
                    else:
                        print(
                            f"  📝 Recorded: {content[:40]}... (Waiting for more context to extract memories)"
                        )
                    return True
                else:
                    print(f"  ❌ Storage failed: {result.get('message')}")
                    return False

        except httpx.ConnectError:
            print(f"  ❌ Cannot connect to API server ({self.base_url})")
            print(
                f"     Please start first: uv run python src/run.py"
            )
            return False
        except Exception as e:
            print(f"  ❌ Storage failed: {e}")
            return False

    async def _save_conversation_meta(self) -> bool:
        """
        Save conversation metadata (called when storing the first message)

        Returns:
            Success status
        """
        if self._conversation_meta_saved:
            return True

        # Build conversation-meta request data
        now = get_now_with_timezone()
        conversation_meta_request = {
            "version": "1.0.0",
            "scene": self.scene,
            "scene_desc": {},
            "name": self.group_name,
            "description": f"Simple Demo - {self.scene} scene",
            "group_id": self.group_id,
            "created_at": to_iso_format(now),
            "default_timezone": get_timezone().key,
            "user_details": {
                "User": {"full_name": "Demo User", "role": "user", "extra": {}},
                "Assistant": {
                    "full_name": "AI Assistant",
                    "role": "assistant",
                    "extra": {},
                },
            },
            "tags": ["demo", self.scene],
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.conversation_meta_url, json=conversation_meta_request
                )
                response.raise_for_status()
                result = response.json()

                if result.get("status") == "ok":
                    self._conversation_meta_saved = True
                    print(
                        f"  ℹ️  Initialized conversation metadata (Scene: {self.scene})"
                    )
                    return True
                else:
                    print(
                        f"  ⚠️  Failed to save conversation metadata: {result.get('message')}"
                    )
                    # Mark as saved even if failed to avoid retrying repeatedly
                    self._conversation_meta_saved = True
                    return False

        except httpx.ConnectError:
            print(f"  ⚠️  Cannot connect to API server for conversation metadata")
            # Mark as saved even if failed to avoid retrying repeatedly
            self._conversation_meta_saved = True
            return False
        except Exception as e:
            print(f"  ⚠️  Failed to save conversation metadata: {e}")
            # Mark as saved even if failed to avoid retrying repeatedly
            self._conversation_meta_saved = True
            return False

    async def search(
        self, query: str, top_k: int = 3, mode: str = "rrf", show_details: bool = True
    ) -> List[Dict[str, Any]]:
        """Search memories

        Args:
            query: Query text
            top_k: Number of results to return (default: 3)
            mode: Retrieval mode (default: "rrf")
                - "keyword": Keyword retrieval (BM25)
                - "vector": Vector retrieval
                - "hybrid": Keyword + Vector + Rerank
                - "rrf": Keyword + Vector + Reciprocal Rank Fusion (recommended)
                - "agentic": LLM-guided multi-round retrieval
            show_details: Whether to show detailed information (default: True)

        Returns:
            List of memories
        """
        payload = {
            "query": query,
            "top_k": top_k,
            "memory_types": "episodic_memory",
            "retrieve_method": mode,
            "group_id": self.group_id,
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(self.retrieve_url, params=payload)
                response.raise_for_status()
                result = response.json()

                if result.get("status") == "ok":
                    # memories is grouped: [{"group_id": [Memory, ...]}, ...]
                    raw_memories = result.get("result", {}).get("memories", [])
                    metadata = result.get("result", {}).get("metadata", {})
                    latency = metadata.get("total_latency_ms", 0)
                    
                    # Flatten grouped memories to flat list
                    memories = []
                    for group_dict in raw_memories:
                        for group_id, mem_list in group_dict.items():
                            memories.extend(mem_list)

                    if show_details:
                        print(
                            f"  🔍 Found {len(memories)} memories (took {latency:.2f}ms)"
                        )
                        self._print_memories(memories)

                    return memories
                else:
                    print(f"  ❌ Search failed: {result.get('message')}")
                    return []

        except httpx.ConnectError:
            print(f"  ❌ Cannot connect to API server ({self.base_url})")
            return []
        except Exception as e:
            print(f"  ❌ Search failed: {e}")
            return []

    def _print_memories(self, memories: List[Dict[str, Any]]):
        """Print memory details (internal method)"""
        if not memories:
            print("     💡 Tip: No related memories found")
            print("         Possible reasons:")
            print(
                "         - Too little conversation input, system hasn't generated memories yet"
            )
            print(
                "           (This simple demo only demonstrates retrieval, not full memory generation)"
            )
            return

        for i, mem in enumerate(memories, 1):
            score = mem.get('score', 0)
            # Extract actual event time (not storage time)
            event_time = extract_event_time_from_memory(mem)
            subject = mem.get('subject', '')
            summary = mem.get('summary', '')
            episode = mem.get('episode', '')

            print(f"\n     [{i}] Relevance: {score:.4f} | Time: {event_time}")
            if subject:
                print(f"         Subject: {subject}")
            if summary:
                print(f"         Summary: {summary[:60]}...")
            if episode:
                print(f"         Details: {episode[:80]}...")

    async def wait_for_index(self, seconds: int = 10):
        """Wait for index building

        Args:
            seconds: Wait time in seconds (default: 10)
        """
        print("  💡 Tip: Memory extraction requires sufficient context")
        print(
            "     - Short conversations may only record messages, not generate memories immediately"
        )
        print(
            "     - Multi-turn conversations with specific information are easier to extract memories from"
        )
        print(
            "     - System extracts memories at conversation boundaries (topic changes, time gaps)"
        )
        print(f"  ⏳ Waiting {seconds} seconds to ensure data is written...")
        await asyncio.sleep(seconds)
        print(f"  ✅ Index building completed")

    def print_separator(self, text: str = ""):
        """Print separator line"""
        if text:
            print(f"\n{'='*60}")
            print(f"{text}")
            print('=' * 60)
        else:
            print('-' * 60)

    def print_summary(self):
        """Print usage summary and tips"""
        print("\n" + "=" * 60)
        print("✅ Demo completed!")
        print("=" * 60)
        print("\n📚 About Memory Extraction:")
        print(
            "   The memory system uses intelligent extraction strategy, not recording all conversations:"
        )
        print(
            "   - ✅ Will extract: Conversations with specific info, opinions, preferences, events"
        )
        print("   - ❌ Won't extract: Too brief, low-information small talk")
        print(
            "   - 🎯 Best practice: Multi-turn conversations, rich context, specific details"
        )

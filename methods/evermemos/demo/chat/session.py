"""Conversation Session Management

Manages conversation sessions for a single group, providing memory retrieval and LLM chat functionality.
"""

import json
import httpx
from typing import List, Dict, Any, Optional, Tuple
from datetime import timedelta
from pathlib import Path

from demo.config import ChatModeConfig, LLMConfig, ScenarioType
from demo.utils import query_memcells_by_group_and_time
from demo.ui import I18nTexts
from memory_layer.llm.llm_provider import LLMProvider
from common_utils.datetime_utils import get_now_with_timezone, to_iso_format
from api_specs.memory_types import ProfileMemory


class ChatSession:
    """Conversation Session Manager"""

    def __init__(
        self,
        group_id: str,
        config: ChatModeConfig,
        llm_config: LLMConfig,
        scenario_type: ScenarioType,
        retrieval_mode: str,  # "keyword" / "vector" / "hybrid" / "rrf" / "agentic"
        data_source: str,  # "episode" / "atomic_fact"
        texts: I18nTexts,
        user_id: str = "user_001",  # User ID for profile fetch
    ):
        """Initialize conversation session

        Args:
            group_id: Group ID
            config: Chat mode configuration
            llm_config: LLM configuration
            scenario_type: Scenario type
            retrieval_mode: Retrieval mode (keyword/vector/hybrid/rrf/agentic)
            data_source: Data source (episode/atomic_fact)
            texts: I18nTexts object
            user_id: User ID for fetching profile
        """
        self.group_id = group_id
        self.user_id = user_id
        self.config = config
        self.llm_config = llm_config
        self.scenario_type = scenario_type
        self.retrieval_mode = retrieval_mode
        self.data_source = data_source
        self.texts = texts

        # Session State
        self.conversation_history: List[Tuple[str, str]] = []
        self.memcell_count: int = 0

        # Services
        self.llm_provider: Optional[LLMProvider] = None

        # API Configuration
        self.api_base_url = config.api_base_url
        self.retrieve_url = f"{self.api_base_url}/api/v1/memories/search"

        # Last Retrieval Metadata
        self.last_retrieval_metadata: Optional[Dict[str, Any]] = None

    async def initialize(self) -> bool:
        """Initialize session

        Returns:
            Whether initialization was successful
        """
        try:
            display_name = (
                "team_chat"
                if self.group_id == "AI产品群"  # skip-i18n-check
                else self.group_id
            )
            print(
                f"\n[{self.texts.get('loading_label')}] {self.texts.get('loading_group_data', name=display_name)}"
            )

            # Check API Server Health
            await self._check_api_server()

            # Count MemCells
            now = get_now_with_timezone()
            start_date = now - timedelta(days=self.config.time_range_days)
            memcells = await query_memcells_by_group_and_time(
                self.group_id, start_date, now
            )
            self.memcell_count = len(memcells)
            print(
                f"[{self.texts.get('loading_label')}] {self.texts.get('loading_memories_success', count=self.memcell_count)} ✅"
            )

            # Load Conversation History
            loaded_history_count = await self.load_conversation_history()
            if loaded_history_count > 0:
                print(
                    f"[{self.texts.get('loading_label')}] {self.texts.get('loading_history_success', count=loaded_history_count)} ✅"
                )
            else:
                print(
                    f"[{self.texts.get('loading_label')}] {self.texts.get('loading_history_new')} ✅"
                )

            # Create LLM Provider
            self.llm_provider = LLMProvider(
                self.llm_config.provider,
                model=self.llm_config.model,
                api_key=self.llm_config.api_key,
                base_url=self.llm_config.base_url,
                temperature=self.llm_config.temperature,
                max_tokens=self.llm_config.max_tokens,
            )

            print(
                f"\n[{self.texts.get('hint_label')}] {self.texts.get('loading_help_hint')}\n"
            )
            return True

        except Exception as e:
            print(
                f"\n[{self.texts.get('error_label')}] {self.texts.get('session_init_error', error=str(e))}"
            )
            import traceback

            traceback.print_exc()
            return False

    async def _check_api_server(self) -> None:
        """Check if API server is running

        Raises:
            ConnectionError: If server is not running
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                # Try accessing health check endpoint or any endpoint
                response = await client.get(f"{self.api_base_url}/docs")
                if response.status_code >= 500:
                    raise ConnectionError("API Server returned error")
        except (httpx.ConnectError, httpx.TimeoutException, ConnectionError) as e:
            error_msg = (
                f"\n❌ Cannot connect to API server: {self.api_base_url}\n\n"
                f"Please start V1 API server first:\n"
                f"  uv run python src/run.py\n\n"
                f"Then run the chat application in another terminal.\n"
            )
            raise ConnectionError(error_msg) from e

    async def load_conversation_history(self) -> int:
        """Load conversation history from file

        Returns:
            Number of loaded conversation turns
        """
        try:
            display_name = (
                "team_chat"
                if self.group_id == "AI产品群"  # skip-i18n-check
                else self.group_id
            )
            history_files = sorted(
                self.config.chat_history_dir.glob(f"{display_name}_*.json"),
                reverse=True,
            )

            if not history_files:
                return 0

            latest_file = history_files[0]
            with latest_file.open("r", encoding="utf-8") as fp:
                data = json.load(fp)

            history = data.get("conversation_history", [])
            self.conversation_history = [
                (item["user_input"], item["assistant_response"])
                for item in history[-self.config.conversation_history_size :]
            ]

            return len(self.conversation_history)

        except Exception as e:
            print(
                f"[{self.texts.get('warning_label')}] {self.texts.get('loading_history_new')}: {e}"
            )
            return 0

    async def save_conversation_history(self) -> None:
        """Save conversation history to file"""
        try:
            display_name = (
                "team_chat"
                if self.group_id == "AI产品群"  # skip-i18n-check
                else self.group_id
            )
            timestamp = get_now_with_timezone().strftime("%Y-%m-%d_%H-%M")
            filename = f"{display_name}_{timestamp}.json"
            filepath = self.config.chat_history_dir / filename

            data = {
                "group_id": self.group_id,
                "last_updated": get_now_with_timezone().isoformat(),
                "conversation_history": [
                    {
                        "timestamp": get_now_with_timezone().isoformat(),
                        "user_input": user_q,
                        "assistant_response": assistant_a,
                    }
                    for user_q, assistant_a in self.conversation_history
                ],
            }

            with filepath.open("w", encoding="utf-8") as fp:
                json.dump(data, fp, ensure_ascii=False, indent=2)

            print(f"[{self.texts.get('save_label')}] {filename} ✅")

        except Exception as e:
            print(f"[{self.texts.get('error_label')}] {e}")

    async def retrieve_memories(self, query: str) -> Dict[str, List[Dict[str, Any]]]:
        """Retrieve memories (episodes, foresights, profile) in parallel."""
        import asyncio

        tasks = [
            self._search(query, memory_types=["episodic_memory"]),
            self._search(query, memory_types=["foresight"]),
            self._fetch_profile(),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_memories = {"episodes": [], "foresights": [], "profiles": []}

        for i, (key, res) in enumerate(
            zip(["episodes", "foresights", "profiles"], results)
        ):
            if isinstance(res, Exception):
                print(f"[Warning] {key}: {res}")
            elif key == "profiles":
                all_memories[key] = res
            else:
                all_memories[key] = self._flatten_result(res)

        # Metadata
        latency = sum(
            float(self._get_metadata(r).get("total_latency_ms", 0) or 0)
            for r in results[:2]
            if not isinstance(r, Exception)
        )
        self.last_retrieval_metadata = {
            "retrieval_mode": self.retrieval_mode,
            "total_latency_ms": latency,
            "episodes_count": len(all_memories["episodes"]),
            "foresights_count": len(all_memories["foresights"]),
            "profiles_count": len(all_memories["profiles"]),
        }
        return all_memories

    # ==================== Unified Search API (aligned with test_v1api_search.py) ====================

    async def _search(
        self,
        query: str,
        memory_types: List[str] = None,
        method: str = None,
        top_k: int = None,
        user_id: str = None,
        group_id: str = None,
        timeout: float = 120.0,
    ) -> Dict[str, Any]:
        """Unified search API call (same as test_v1api_search.test_search_memories)."""
        params = {
            "query": query,
            "method": method or self.retrieval_mode,
            "top_k": top_k or self.config.top_k_memories,
        }
        if user_id:
            params["user_id"] = user_id
        if group_id or self.group_id:
            params["group_id"] = group_id or self.group_id
        if memory_types:
            params["memory_types"] = ",".join(memory_types)

        async with httpx.AsyncClient(timeout=timeout, verify=False) as client:
            response = await client.get(self.retrieve_url, params=params)
            response.raise_for_status()
            return response.json()

    async def _fetch_profile(self) -> List[Dict[str, Any]]:
        """Fetch profile via GET /api/v1/memories."""
        url = f"{self.api_base_url}/api/v1/memories"
        params = {"user_id": self.user_id, "memory_type": "profile", "limit": 10}

        async with httpx.AsyncClient(timeout=30.0, verify=False) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

        if data.get("status") != "ok":
            raise RuntimeError(f"API Error: {data.get('message')}")

        memories = data.get("result", {}).get("memories", []) or []
        # For demo: generate readable_profile locally (moved from fetch_mem_service.py)
        for mem in memories:
            profile_data = mem.get("profile_data") or {}
            if (
                "readable_profile" not in profile_data
                and "explicit_info" in profile_data
            ):

                profile_data["readable_profile"] = ProfileMemory.from_dict(
                    profile_data
                ).to_readable_profile()
                mem["profile_data"] = profile_data
        return memories

    def _get_metadata(self, resp: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from API response."""
        if not resp or not isinstance(resp, dict):
            return {}
        result = resp.get("result") if isinstance(resp.get("result"), dict) else resp
        return (result or {}).get("metadata", {}) or {}

    def _flatten_result(self, resp: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Flatten grouped search result to flat list."""
        if not resp or not isinstance(resp, dict):
            return []

        result = resp.get("result") if isinstance(resp.get("result"), dict) else resp
        if not result:
            return []

        memories = result.get("memories", []) or []
        scores = result.get("scores", []) or []

        # Already flat list?
        if memories and isinstance(memories[0], dict):
            if not any(isinstance(v, list) for v in memories[0].values()):
                return list(memories)

        # Grouped: [{gid: [mem...]}, ...] + [{gid: [score...]}, ...]
        score_map = {}
        for s in scores:
            if isinstance(s, dict):
                for gid, slist in s.items():
                    if isinstance(slist, list):
                        score_map[gid] = slist

        flat = []
        for grp in memories:
            if not isinstance(grp, dict):
                continue
            for gid, mlist in grp.items():
                if not isinstance(mlist, list):
                    continue
                gscores = score_map.get(gid, [])
                for i, m in enumerate(mlist):
                    if isinstance(m, dict):
                        item = dict(m)
                        if "score" not in item and i < len(gscores):
                            item["score"] = gscores[i]
                        flat.append(item)
        return flat

    def build_prompt(
        self, user_query: str, memories: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, str]]:
        """Build Prompt

        Args:
            user_query: User query
            memories: Dict with "episodes", "foresights", "profiles"

        Returns:
            List of Chat Messages
        """
        messages = []

        # System Message
        lang_key = "zh" if self.texts.language == "zh" else "en"
        system_content = self.texts.get(f"prompt_system_role_{lang_key}")
        messages.append({"role": "system", "content": system_content})

        # Build memory context
        memory_sections: List[str] = []

        # 1) Profile (no numbering)
        profiles = memories.get("profiles") or []
        first_profile = profiles[0] if profiles else None
        if isinstance(first_profile, dict):
            profile_text = (first_profile.get("profile_data", {}) or {}).get(
                "readable_profile"
            )
            if profile_text:
                memory_sections.append(f"【User Profile】\n{profile_text}")

        # 2) Foresights (no numbering)
        foresights = memories.get("foresights", [])
        if foresights:
            foresight_lines: List[str] = []
            for f in foresights[: self.config.top_k_memories]:
                if not isinstance(f, dict):
                    continue
                content = f.get("foresight") or f.get("summary")
                if content:
                    foresight_lines.append(f"  - {content}")
            if foresight_lines:
                memory_sections.append("【Foresights】\n" + "\n".join(foresight_lines))

        # 3) Episodes (numbered, aligned with UI)
        episodes = memories.get("episodes", [])
        if episodes:
            episode_lines: List[str] = []
            for i, mem in enumerate(episodes[: self.config.top_k_memories], start=1):
                if not isinstance(mem, dict):
                    continue
                raw_timestamp = mem.get("timestamp", "")
                iso_timestamp = to_iso_format(raw_timestamp)
                timestamp = iso_timestamp[:10] if iso_timestamp else ""
                content = mem.get("summary") or mem.get("episode") or mem.get("subject")
                if content:
                    episode_lines.append(f"  [{i}] ({timestamp}) {content}")
            if episode_lines:
                memory_sections.append(
                    "【Related Memories】\n" + "\n".join(episode_lines)
                )

        # Add all memory sections as one system message
        if memory_sections:
            messages.append({"role": "system", "content": "\n\n".join(memory_sections)})
        # Conversation History
        for user_q, assistant_a in self.conversation_history[
            -self.config.conversation_history_size :
        ]:
            messages.append({"role": "user", "content": user_q})
            messages.append({"role": "assistant", "content": assistant_a})

        # Current Question
        messages.append({"role": "user", "content": user_query})
        return messages

    async def chat(self, user_input: str) -> str:
        """Core Chat Logic

        Args:
            user_input: User input

        Returns:
            Assistant response
        """
        from .ui import ChatUI

        # Retrieve Memories
        memories = await self.retrieve_memories(user_input)

        # Show Retrieval Results
        if self.config.show_retrieved_memories and memories:
            # Combine all memory types for display (episodes have numbers)
            all_memories = memories.get("episodes", [])[:5]
            ChatUI.print_retrieved_memories(
                all_memories,
                texts=self.texts,
                retrieval_metadata=self.last_retrieval_metadata,
            )

        # Build Prompt
        messages = self.build_prompt(user_input, memories)

        # Show Generation Progress
        ChatUI.print_generating_indicator(self.texts)

        # Call LLM
        try:
            if hasattr(self.llm_provider, 'provider') and hasattr(
                self.llm_provider.provider, 'chat_with_messages'
            ):
                raw_response = await self.llm_provider.provider.chat_with_messages(
                    messages
                )
            else:
                prompt_parts = []
                for msg in messages:
                    role = msg["role"]
                    content = msg["content"]
                    if role == "system":
                        prompt_parts.append(f"System: {content}")
                    elif role == "user":
                        prompt_parts.append(f"User: {content}")
                    elif role == "assistant":
                        prompt_parts.append(f"Assistant: {content}")

                prompt = "\n\n".join(prompt_parts)
                raw_response = await self.llm_provider.generate(prompt)

            raw_response = raw_response.strip()

            # Clear Generation Progress
            ChatUI.print_generation_complete(self.texts)

            assistant_response = raw_response

        except Exception as e:
            ChatUI.clear_progress_indicator()
            error_msg = f"[{self.texts.get('error_label')}] {self.texts.get('chat_llm_error', error=str(e))}"
            print(f"\n{error_msg}")
            import traceback

            traceback.print_exc()
            return error_msg

        # Update Conversation History
        self.conversation_history.append((user_input, assistant_response))

        if len(self.conversation_history) > self.config.conversation_history_size:
            self.conversation_history = self.conversation_history[
                -self.config.conversation_history_size :
            ]

        return assistant_response

    def clear_history(self) -> None:
        """Clear conversation history"""
        from .ui import ChatUI

        count = len(self.conversation_history)
        self.conversation_history = []
        ChatUI.print_info(self.texts.get("cmd_clear_done", count=count), self.texts)

    async def reload_data(self) -> None:
        """Reload memory data"""
        from .ui import ChatUI
        from common_utils.cli_ui import CLIUI

        display_name = (
            "team_chat"
            if self.group_id == "AI产品群"  # skip-i18n-check
            else self.group_id
        )

        ui = CLIUI()
        print()
        ui.note(self.texts.get("cmd_reload_refreshing", name=display_name), icon="🔄")

        # Recount MemCells
        now = get_now_with_timezone()
        start_date = now - timedelta(days=self.config.time_range_days)
        memcells = await query_memcells_by_group_and_time(
            self.group_id, start_date, now
        )
        self.memcell_count = len(memcells)

        print()
        ui.success(
            f"✓ {self.texts.get('cmd_reload_complete', users=0, memories=self.memcell_count)}"
        )
        print()

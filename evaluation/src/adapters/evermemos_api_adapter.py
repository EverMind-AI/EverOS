"""
EverMemOS HTTP Memory API adapter (evaluation side).

This adapter talks to EverMemOS server endpoints:
- POST   /api/v1/memories         (ingest single message)
- GET    /api/v1/memories/search  (retrieve memories)

Note:
- This file was missing in the current workspace; registry.py still references it as
  `evaluation.src.adapters.evermemos_api_adapter`.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp

from common_utils.datetime_utils import to_iso_format
from evaluation.src.adapters.online_base import OnlineAPIAdapter
from evaluation.src.adapters.registry import register_adapter
from evaluation.src.core.data_models import Conversation, SearchResult


@register_adapter("evermemos_api")
class EverMemOSAPIAdapter(OnlineAPIAdapter):
    """
    Adapter for EverMemOS Memory API.

    Design:
    - Ingest each conversation once (do NOT duplicate per-speaker perspectives).
    - Retrieval is controlled by system config: search.scope = "personal" | "group".
    """

    def __init__(self, config: dict, output_dir: Path = None):
        super().__init__(config, output_dir)

        self.base_url = str(config.get("base_url", "")).rstrip("/")
        self.api_key = str(config.get("api_key", "") or "")
        self.sync_mode = bool(config.get("sync_mode", False))
        self.max_retries = int(config.get("max_retries", 3))
        self.timeout_seconds = float(config.get("timeout_seconds", 60))
        self.request_interval = float(config.get("request_interval", 0.0))

        self._session: Optional[aiohttp.ClientSession] = None

        self._memories_url = self._normalize_memories_url(self.base_url)
        self._search_url = self._memories_url.rstrip("/") + "/search"

        print(f"   Memory API: {self._memories_url}")

    # --- override add() to support clean_groups ---
    async def add(
        self, conversations: List[Conversation], **kwargs: Any
    ) -> Dict[str, Any]:
        """Override to support clean_groups config before ingestion."""
        if self.config.get("clean_groups"):
            from evaluation.src.utils.cleaner import clear_group_data

            print("\n🧹 clean_groups enabled, clearing data for involved groups...")
            for conv in conversations:
                await clear_group_data(conv.conversation_id, verbose=True)
            print()
        return await super().add(conversations, **kwargs)

    # --- lifecycle ---
    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session and not self._session.closed:
            return self._session
        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    # --- helpers ---
    @staticmethod
    def _normalize_memories_url(base_url: str) -> str:
        url = (base_url or "").rstrip("/")
        if url.endswith("/api/v1/memories"):
            return url
        return url + "/api/v1/memories"

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def _request_json_with_retry(
        self, method: str, url: str, **kwargs: Any
    ) -> Dict[str, Any]:
        session = await self._get_session()
        req = getattr(session, method)

        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                async with req(url, **kwargs) as resp:
                    text = await resp.text()
                    if resp.status >= 400:
                        raise RuntimeError(
                            f"{method.upper()} {url} -> {resp.status}: {text[:800]}"
                        )
                    if not text:
                        return {}
                    try:
                        return json.loads(text)
                    except json.JSONDecodeError:
                        # Some gateways may return wrong content-type; still parse as JSON.
                        return await resp.json(content_type=None)
            except Exception as e:  # noqa: BLE001
                last_exc = e
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(min(2**attempt, 8))
                    continue
                raise
        raise last_exc or RuntimeError("request failed")

    @staticmethod
    def _speaker_to_user_id(conversation_id: str, speaker_name: str) -> str:
        # Align with evaluation loader speaker_id style: "{speaker_lower}_{conv_id}"
        return f"{speaker_name.lower().replace(' ', '_')}_{conversation_id}"

    # --- overrides to avoid per-speaker duplication on ingest/search ---
    def _need_dual_perspective(self, speaker_a: str, speaker_b: str) -> bool:
        # EverMemOS Memory API stores group chat stream; do not split perspectives.
        return False

    def _conversation_to_messages(
        self,
        conversation: Conversation,
        format_type: str = "basic",
        perspective: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        del format_type, perspective
        conv_id = conversation.conversation_id

        out: List[Dict[str, Any]] = []
        for idx, msg in enumerate(conversation.messages):
            if not msg.timestamp:
                continue

            message_id = (
                msg.metadata.get("message_id")
                or msg.metadata.get("dia_id")
                or f"{conv_id}_{idx}"
            )

            out.append(
                {
                    "group_id": conv_id,
                    "group_name": conv_id,
                    "message_id": str(message_id),
                    "create_time": to_iso_format(msg.timestamp),
                    "sender": msg.speaker_id or self._speaker_to_user_id(conv_id, msg.speaker_name),
                    "sender_name": msg.speaker_name,
                    "content": msg.content,
                    "refer_list": msg.metadata.get("refer_list") or [],
                }
            )
        return out

    def _get_answer_prompt(self) -> str:
        """Use EverMemOS CoT answer prompt (same as evermemos adapter)."""
        from evaluation.src.adapters.evermemos.prompts.answer_prompts import ANSWER_PROMPT
        return ANSWER_PROMPT

    # --- required abstract methods (OnlineAPIAdapter hooks) ---
    async def _add_user_messages(
        self,
        conv: Conversation,
        messages: List[Dict[str, Any]],
        speaker: str,
        **kwargs: Any,
    ) -> Any:
        del speaker
        if not self._memories_url:
            raise ValueError("base_url is empty; set system config 'base_url'")

        progress = kwargs.get("progress")
        task_id = kwargs.get("task_id")

        params = {"sync_mode": "true"} if self.sync_mode else None
        headers = self._headers()

        # Preserve ordering: send sequentially.
        for payload in messages:
            await self._request_json_with_retry(
                "post", self._memories_url, json=payload, params=params, headers=headers
            )
            if progress is not None and task_id is not None:
                progress.update(task_id, advance=1)
            if self.request_interval > 0:
                await asyncio.sleep(self.request_interval)

        return None

    async def _search_single_user(
        self, query: str, conversation_id: str, user_id: str, top_k: int, **kwargs: Any
    ) -> List[Dict[str, Any]]:
        del kwargs
        if not self._search_url:
            raise ValueError("base_url is empty; set system config 'base_url'")

        search_cfg = self.config.get("search", {}) or {}
        scope = str(search_cfg.get("scope", "personal"))
        retrieve_method = str(
            search_cfg.get("retrieve_method") or search_cfg.get("mode") or "keyword"
        )
        memory_types = search_cfg.get("memory_types", []) or []

        params: Dict[str, Any] = {
            "query": query,
            "retrieve_method": retrieve_method,
            "top_k": int(top_k),
        }

        if memory_types:
            if isinstance(memory_types, str):
                params["memory_types"] = memory_types
            else:
                params["memory_types"] = ",".join([str(x) for x in memory_types])

        if scope == "group":
            params["group_id"] = conversation_id
            params["user_id"] = ""  # Empty string to filter duplicate memories
        else:
            params["user_id"] = user_id

        headers = self._headers()
        data = await self._request_json_with_retry(
            "get", self._search_url, params=params, headers=headers
        )

        result = (data or {}).get("result") or {}
        memories = result.get("memories") or []
        scores = result.get("scores") or []

        mem_groups: Dict[str, List[Dict[str, Any]]] = {}
        for obj in memories:
            if isinstance(obj, dict):
                for gid, mem_list in obj.items():
                    if mem_list:
                        mem_groups.setdefault(str(gid), []).extend(mem_list)

        score_groups: Dict[str, List[float]] = {}
        for obj in scores:
            if isinstance(obj, dict):
                for gid, score_list in obj.items():
                    if score_list:
                        score_groups.setdefault(str(gid), []).extend(score_list)

        results_out: List[Dict[str, Any]] = []
        for gid, mem_list in mem_groups.items():
            score_list = score_groups.get(gid, [])
            for i, mem in enumerate(mem_list):
                sc = score_list[i] if i < len(score_list) else 0.0
                ts = (mem or {}).get("timestamp") or ""
                episode = (mem or {}).get("episode") or (mem or {}).get("summary") or ""
                content = f"{ts}: {episode}".strip()
                results_out.append(
                    {
                        "content": content,
                        "score": float(sc) if sc is not None else 0.0,
                        "user_id": (mem or {}).get("user_id") or user_id,
                        "metadata": {"group_id": gid, "raw": mem},
                    }
                )

        results_out.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return results_out[: int(top_k)]

    def _build_single_search_result(
        self,
        query: str,
        conversation_id: str,
        results: List[Dict[str, Any]],
        user_id: str,
        top_k: int,
        **kwargs: Any,
    ) -> SearchResult:
        del kwargs
        search_cfg = self.config.get("search", {}) or {}
        retrieve_method = str(
            search_cfg.get("retrieve_method") or search_cfg.get("mode") or "keyword"
        )
        system_name = str(self.config.get("name") or "evermemos_api")

        retrieval_metadata = {
            "system": system_name,
            "top_k": int(top_k),
            "retrieve_method": retrieve_method,
            "memory_types": ["episodic_memory"],
            "user_id": "",
            "group_id": conversation_id,
        }

        return SearchResult(
            query=query,
            conversation_id=conversation_id,
            results=results[: int(top_k)],
            retrieval_metadata=retrieval_metadata,
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
        **kwargs: Any,
    ) -> SearchResult:
        # Not used (we force single perspective), but keep minimal implementation to satisfy ABC.
        del all_results, results_a, results_b, speaker_a, speaker_b, speaker_b_user_id, kwargs
        return self._build_single_search_result(
            query=query,
            conversation_id=conversation_id,
            results=[],
            user_id=speaker_a_user_id,
            top_k=top_k,
        )



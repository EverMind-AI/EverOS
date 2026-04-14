"""
OpenClaw adapter for the EverMemOS evaluation pipeline.

Wraps OpenClaw memory lifecycle (ingest / flush / index / search / get) via a
Node bridge, exposes the BaseAdapter surface, and emits session-level
retrieval traces + lifecycle diagnostics alongside the shared answer prompt.

Task 4 adds add() + build_lazy_index() with per-conversation sandbox
isolation. ingest + flush calls are currently thin placeholders so the
control flow is exercisable without a real OpenClaw runtime; Task 8 wires
them to the native bridge.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, List, Optional

from evaluation.src.adapters.base import BaseAdapter
from evaluation.src.adapters.openclaw_manifest import (
    build_session_manifest,
    project_message_id_to_session_id,
)
from evaluation.src.adapters.openclaw_runtime import arun_bridge, build_sandbox_paths
from evaluation.src.adapters.registry import register_adapter
from evaluation.src.core.data_models import Conversation, SearchResult


logger = logging.getLogger(__name__)


_RUN_ID_LATEST_FILE = "LATEST"
_ARTIFACT_ROOT = "artifacts/openclaw"


_DEFAULT_ANSWER_PROMPT = (
    "You are a helpful assistant answering a question about a conversation.\n"
    "Use the memory snippets in CONTEXT to answer concisely (<=6 words when possible).\n"
    "If the context does not contain the answer, respond with \"No relevant information.\".\n\n"
    "# CONTEXT\n{context}\n\n# QUESTION\n{question}\n\n# ANSWER"
)


@register_adapter("openclaw")
class OpenClawAdapter(BaseAdapter):
    def __init__(self, config: dict, output_dir: Any = None):
        super().__init__(config)
        self.output_dir = output_dir
        self._prepared: bool = False
        self._run_id: Optional[str] = None
        self._openclaw_cfg: dict = dict(config.get("openclaw") or {})
        search_cfg = config.get("search", {}) or {}
        self.max_inflight_queries_per_conversation: int = int(
            search_cfg.get("max_inflight_queries_per_conversation", 1)
        )
        self._conversation_semaphores: dict[str, asyncio.Semaphore] = {}
        self._llm_provider = None  # lazy
        self._shared_prompt_template: Optional[str] = None

    # ----------------------------------------------------------------- prepare
    async def prepare(
        self,
        conversations: List[Conversation],
        output_dir: Any = None,
        checkpoint_manager: Any = None,
        **kwargs,
    ) -> None:
        """Idempotent initialization.

        Pipeline currently doesn't call prepare() explicitly, so add() calls it
        internally. When a future pipeline wires prepare() in, this flag keeps
        initialization from running twice.
        """
        if self._prepared:
            return
        self._prepared = True
        self._prepared_conversation_ids = [c.conversation_id for c in conversations]
        logger.debug(
            "openclaw adapter prepared for %d conversations",
            len(self._prepared_conversation_ids),
        )

    # --------------------------------------------------------------------- add
    async def add(
        self,
        conversations: List[Conversation],
        output_dir: Any = None,
        checkpoint_manager: Any = None,
        **kwargs,
    ) -> dict:
        if not self._prepared:
            await self.prepare(
                conversations=conversations,
                output_dir=output_dir,
                checkpoint_manager=checkpoint_manager,
                **kwargs,
            )

        root_dir = self._resolve_run_root(output_dir or self.output_dir)
        run_id = root_dir.name
        conversations_map: dict[str, dict] = {}

        for conv in conversations:
            sandbox = self._prepare_conversation_sandbox(root_dir, conv)
            t0 = time.perf_counter()
            try:
                await self._ingest_conversation(sandbox, conv)
                await self._flush_and_settle_if_needed(sandbox)
            except Exception as err:
                sandbox["run_status"] = "failed"
                self._write_handle(sandbox, add_summary={"error": str(err)})
                logger.exception("openclaw ingest failed for %s", conv.conversation_id)
                raise
            add_latency_ms = (time.perf_counter() - t0) * 1000.0
            sandbox["run_status"] = "ready"
            sandbox["visibility_state"] = "settled"
            self._write_handle(
                sandbox,
                add_summary={
                    "conversation_id": conv.conversation_id,
                    "add_latency_ms": add_latency_ms,
                },
            )
            conversations_map[conv.conversation_id] = sandbox

        return {
            "type": "openclaw_sandboxes",
            "run_id": run_id,
            "root_dir": str(root_dir),
            "conversations": conversations_map,
        }

    # --------------------------------------------------------- build_lazy_index
    def build_lazy_index(
        self, conversations: List[Conversation], output_dir: Any
    ) -> dict:
        root_dir = self._locate_existing_run_root(Path(output_dir))
        handles: dict[str, dict] = {}
        for conv in conversations:
            handle_path = root_dir / "conversations" / conv.conversation_id / "handle.json"
            if not handle_path.exists():
                continue
            handle = json.loads(handle_path.read_text())
            if (
                handle.get("run_status") != "ready"
                or handle.get("visibility_state") != "settled"
            ):
                continue
            handles[conv.conversation_id] = handle
        return {
            "type": "openclaw_sandboxes",
            "run_id": root_dir.name,
            "root_dir": str(root_dir),
            "conversations": handles,
        }

    # ---------------------------------------------------------------- search
    async def search(
        self, query: str, conversation_id: str, index: Any, **kwargs
    ) -> SearchResult:
        sandbox = index["conversations"][conversation_id]
        backend_mode = sandbox.get("backend_mode", self._openclaw_cfg.get("backend_mode", "hybrid"))
        retrieval_route = sandbox.get(
            "retrieval_route", self._openclaw_cfg.get("retrieval_route", "search_then_get")
        )
        top_k = int(self.config.get("search", {}).get("top_k", 30))

        semaphore = self._conversation_semaphores.setdefault(
            conversation_id,
            asyncio.Semaphore(self.max_inflight_queries_per_conversation),
        )

        scheduler_start = time.perf_counter()
        async with semaphore:
            scheduler_wait_ms = (time.perf_counter() - scheduler_start) * 1000.0
            retrieval_start = time.perf_counter()

            search_payload = {
                "command": "search",
                "config_path": sandbox.get("resolved_config_path", ""),
                "workspace_dir": sandbox.get("workspace_dir", ""),
                "state_dir": sandbox.get("native_store_dir", ""),
                "query": query,
                "top_k": top_k,
            }
            bridge_response = await arun_bridge(self._bridge_script_path(), search_payload)
            hits = list(bridge_response.get("hits") or [])

            if retrieval_route == "search_then_get":
                hits = await self._enrich_with_get(sandbox, hits)

            results = []
            context_parts = []
            for rank, hit in enumerate(hits, start=1):
                hit_meta = dict(hit.get("metadata") or {})
                source_sessions = hit_meta.get("source_sessions") or self._derive_source_sessions(hit)
                hit_meta["source_sessions"] = source_sessions
                snippet = hit.get("snippet", "")
                results.append(
                    {
                        "content": snippet,
                        "score": float(hit.get("score", 0.0)),
                        "metadata": {
                            **hit_meta,
                            "artifact_locator": hit.get("artifact_locator"),
                        },
                    }
                )
                if snippet:
                    context_parts.append(f"{rank}. {snippet}")

            retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000.0
            retrieval_metadata = {
                "system": "openclaw",
                "top_k": top_k,
                "backend_mode": backend_mode,
                "retrieval_route": retrieval_route,
                "retrieval_latency_ms": retrieval_latency_ms,
                "scheduler_wait_ms": scheduler_wait_ms,
                "formatted_context": "\n\n".join(context_parts),
                "conversation_id": conversation_id,
            }
            return SearchResult(
                query=query,
                conversation_id=conversation_id,
                results=results,
                retrieval_metadata=retrieval_metadata,
            )

    async def _enrich_with_get(self, sandbox: dict, hits: list[dict]) -> list[dict]:
        """For search_then_get routes, fetch narrower content per artifact.

        We avoid mutating the hit in-place so upstream bridge tracing stays
        intact.
        """
        enriched = []
        for hit in hits:
            locator = hit.get("artifact_locator")
            if not locator:
                enriched.append(hit)
                continue
            get_payload = {
                "command": "get",
                "config_path": sandbox.get("resolved_config_path", ""),
                "workspace_dir": sandbox.get("workspace_dir", ""),
                "state_dir": sandbox.get("native_store_dir", ""),
                "artifact_locator": locator,
            }
            try:
                resp = await arun_bridge(self._bridge_script_path(), get_payload)
            except Exception as err:  # noqa: BLE001
                logger.warning("get failed for artifact %s: %s", locator, err)
                enriched.append(hit)
                continue
            snippet = resp.get("snippet") or hit.get("snippet", "")
            new_hit = dict(hit)
            new_hit["snippet"] = snippet
            enriched.append(new_hit)
        return enriched

    @staticmethod
    def _derive_source_sessions(hit: dict) -> list[str]:
        """Best-effort source_sessions derivation when the bridge didn't set it.

        Falls back to empty list rather than raising - retrieval metrics are
        the only consumer and they treat missing sessions as a zero-recall hit.
        """
        locator = hit.get("artifact_locator") or {}
        path_rel = locator.get("path_rel") or ""
        raw_session_ids = hit.get("metadata", {}).get("source_message_ids") or []
        out: list[str] = []
        for mid in raw_session_ids:
            try:
                out.append(project_message_id_to_session_id(mid))
            except ValueError:
                continue
        if out:
            return sorted(set(out))
        # No message ids - leave empty; path_rel alone is not enough to project
        _ = path_rel
        return []

    # ---------------------------------------------------------------- answer
    async def answer(self, query: str, context: str, **kwargs) -> str:
        prompt = self._shared_answer_prompt().format(context=context, question=query)
        return await self._generate_answer(prompt)

    async def _generate_answer(self, prompt: str) -> str:
        provider = self._get_llm_provider()
        result = await provider.generate(prompt=prompt, temperature=0)
        if "FINAL ANSWER:" in result:
            parts = result.split("FINAL ANSWER:")
            result = parts[1].strip() if len(parts) > 1 else result.strip()
        return result.strip()

    def _get_llm_provider(self):
        if self._llm_provider is not None:
            return self._llm_provider
        from memory_layer.llm.llm_provider import LLMProvider

        llm_cfg = self.config.get("llm", {}) or {}
        self._llm_provider = LLMProvider(
            provider_type=llm_cfg.get("provider", "openai"),
            model=llm_cfg.get("model", "gpt-4o-mini"),
            api_key=llm_cfg.get("api_key", ""),
            base_url=llm_cfg.get("base_url", "https://api.openai.com/v1"),
            temperature=llm_cfg.get("temperature", 0.0),
            max_tokens=llm_cfg.get("max_tokens", 1024),
        )
        return self._llm_provider

    def _shared_answer_prompt(self) -> str:
        """Return the answer prompt shared with other adapters.

        Uses the mem0-compatible prompt from prompts.yaml when available so
        openclaw answers are judged by the same yardstick as mem0/memos, and
        falls back to a concise in-process template if prompts.yaml is absent
        (e.g. in minimal test environments).
        """
        if self._shared_prompt_template is not None:
            return self._shared_prompt_template
        try:
            from evaluation.src.utils.config import load_yaml

            prompts_path = Path(__file__).parent.parent.parent / "config" / "prompts.yaml"
            prompts = load_yaml(str(prompts_path))
            self._shared_prompt_template = prompts["online_api"]["default"]["answer_prompt_mem0"]
        except Exception:
            self._shared_prompt_template = _DEFAULT_ANSWER_PROMPT
        return self._shared_prompt_template

    # ----------------------------------------------------------- system info
    def get_system_info(self) -> dict:
        return {"name": "OpenClaw", "config": self.config}

    def _bridge_script_path(self) -> Path:
        return Path(__file__).parent.parent.parent / "scripts" / "openclaw_eval_bridge.mjs"

    # ===================================================== internal helpers
    def _resolve_run_root(self, output_dir: Any) -> Path:
        if output_dir is None:
            raise ValueError("output_dir is required to resolve openclaw sandbox root")
        if self._run_id is None:
            self._run_id = time.strftime("run-%Y%m%dT%H%M%S")
        root = Path(output_dir) / _ARTIFACT_ROOT / self._run_id
        root.mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / _ARTIFACT_ROOT / _RUN_ID_LATEST_FILE).write_text(self._run_id)
        return root

    def _locate_existing_run_root(self, output_dir: Path) -> Path:
        latest_file = output_dir / _ARTIFACT_ROOT / _RUN_ID_LATEST_FILE
        if latest_file.exists():
            run_id = latest_file.read_text().strip()
            root = output_dir / _ARTIFACT_ROOT / run_id
            if root.exists():
                return root
        # Fallback: newest run directory by mtime
        parent = output_dir / _ARTIFACT_ROOT
        if not parent.exists():
            raise FileNotFoundError(f"no openclaw artifacts under {parent}")
        runs = [p for p in parent.iterdir() if p.is_dir()]
        if not runs:
            raise FileNotFoundError(f"no openclaw runs under {parent}")
        runs.sort(key=lambda p: p.stat().st_mtime)
        return runs[-1]

    def _prepare_conversation_sandbox(
        self, root_dir: Path, conv: Conversation
    ) -> dict:
        run_id = root_dir.name
        output_dir = root_dir.parent.parent.parent  # strip artifacts/openclaw/<run>
        paths = build_sandbox_paths(output_dir, run_id, conv.conversation_id)

        base = Path(paths["base_dir"])
        base.mkdir(parents=True, exist_ok=True)
        Path(paths["native_store_dir"]).mkdir(parents=True, exist_ok=True)
        Path(paths["metrics_dir"]).mkdir(parents=True, exist_ok=True)
        Path(paths["events_path"]).touch(exist_ok=True)

        manifest = build_session_manifest(
            conv, dataset_name=self.config.get("dataset_name", "")
        )
        manifest_path = base / "session_manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

        resolved_config_path = base / "openclaw.resolved.json"
        resolved_config_path.write_text(
            json.dumps(self._openclaw_cfg, ensure_ascii=False, indent=2)
        )

        handle: dict = {
            "conversation_id": conv.conversation_id,
            "workspace_dir": paths["workspace_dir"],
            "native_store_dir": paths["native_store_dir"],
            "resolved_config_path": str(resolved_config_path),
            "session_manifest_path": str(manifest_path),
            "prov_units_path": str(base / "prov_units.jsonl"),
            "artifact_bindings_path": str(base / "artifact_bindings.jsonl"),
            "events_path": paths["events_path"],
            "metrics_dir": paths["metrics_dir"],
            "backend_mode": self._openclaw_cfg.get("backend_mode", "hybrid"),
            "retrieval_route": self._openclaw_cfg.get(
                "retrieval_route", "search_then_get"
            ),
            "visibility_mode": self._openclaw_cfg.get("visibility_mode", "settled"),
            "visibility_state": "prepared",
            "run_status": "pending",
            "last_flush_epoch": 0,
            "last_index_epoch": 0,
            "retrieval_eval_supported": True,
        }
        return handle

    def _write_handle(self, handle: dict, add_summary: Optional[dict] = None) -> None:
        base = Path(handle["workspace_dir"])
        handle_path = base / "handle.json"
        handle_path.write_text(json.dumps(handle, ensure_ascii=False, indent=2))

        if add_summary is not None:
            (Path(handle["metrics_dir"]) / "add_summary.json").write_text(
                json.dumps(add_summary, ensure_ascii=False, indent=2)
            )

    async def _ingest_conversation(self, sandbox: dict, conv: Conversation) -> None:
        """Ingest raw transcript into the OpenClaw sandbox.

        Placeholder - Task 8 wires this to the native ``index`` bridge command
        once OpenClaw CLI is available. Tests monkeypatch this method.
        """
        sandbox["visibility_state"] = "ingested"

    async def _flush_and_settle_if_needed(self, sandbox: dict) -> None:
        """Run flush + wait for index settle if visibility_mode == 'settled'.

        Placeholder - Task 8 wires this to the ``flush`` / ``status`` bridge
        commands. Tests monkeypatch this method.
        """
        sandbox["visibility_state"] = "indexed"

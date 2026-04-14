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
from evaluation.src.adapters.openclaw_ingestion import (
    session_id_from_path,
    write_session_files,
)
from evaluation.src.adapters.openclaw_manifest import (
    build_session_manifest,
    project_message_id_to_session_id,
)
from evaluation.src.adapters.openclaw_resolved_config import (
    build_openclaw_resolved_config,
)
from evaluation.src.adapters.openclaw_runtime import (
    arun_bridge,
    build_sandbox_paths,
    isolated_env_for_sandbox,
)
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
        # repo_path resolution: yaml > env (env is still honored as a
        # fallback so smoke tests that only set OPENCLAW_REPO_PATH keep
        # working). P0-2 makes the bridge honor the payload value
        # unconditionally so the yaml setting is no longer cosmetic.
        self._openclaw_repo_path: str = (
            (self._openclaw_cfg.get("repo_path") or "").strip()
        )

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
                # _flush_and_settle_if_needed is authoritative for
                # visibility_state. It raises if visibility_mode=='settled'
                # and the OpenClaw status check does not confirm settled,
                # so we never persist a handle that claims 'settled' when
                # the backend disagrees.
                await self._flush_and_settle_if_needed(sandbox)
                self._assert_visibility_contract(sandbox)
            except Exception as err:
                sandbox["run_status"] = "failed"
                self._write_handle(sandbox, add_summary={"error": str(err)})
                logger.exception("openclaw ingest failed for %s", conv.conversation_id)
                raise
            add_latency_ms = (time.perf_counter() - t0) * 1000.0
            sandbox["run_status"] = "ready"
            self._write_handle(
                sandbox,
                add_summary={
                    "conversation_id": conv.conversation_id,
                    "add_latency_ms": add_latency_ms,
                    "visibility_state": sandbox.get("visibility_state"),
                    "visibility_mode": sandbox.get("visibility_mode"),
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
            if handle.get("run_status") != "ready":
                continue
            # visibility_mode decides what visibility_state is acceptable:
            #   settled mode   -> only 'settled' (strict)
            #   eventual mode  -> 'indexed' or 'settled' (search re-syncs)
            mode = handle.get("visibility_mode")
            state = handle.get("visibility_state")
            if mode == "settled":
                if state != "settled":
                    continue
            else:
                if state not in ("indexed", "settled"):
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
                **self._bridge_base_payload(sandbox),
                "command": "search",
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
                **self._bridge_base_payload(sandbox),
                "command": "get",
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

        Preference order:
          1. metadata.source_message_ids (projected one by one)
          2. artifact_locator.path_rel (matches session-SX-*.md layout)
        Falls back to empty list rather than raising - retrieval metrics are
        the only consumer and they treat missing sessions as a zero-recall hit.
        """
        raw_message_ids = hit.get("metadata", {}).get("source_message_ids") or []
        out: list[str] = []
        for mid in raw_message_ids:
            try:
                out.append(project_message_id_to_session_id(mid))
            except ValueError:
                continue
        if out:
            return sorted(set(out))

        locator = hit.get("artifact_locator") or {}
        sid = session_id_from_path(locator.get("path_rel") or "")
        return [sid] if sid else []

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

    def _bridge_base_payload(self, sandbox: dict) -> dict:
        """Fields every BridgeCommand needs: where OpenClaw lives, where the
        sandbox lives, and which config to read. repo_path comes from the
        yaml (preferred) so the config surface is authoritative; bridge
        still falls back to the env var for developer convenience.
        """
        return {
            "repo_path": self._openclaw_repo_path,
            "config_path": sandbox.get("resolved_config_path", ""),
            "workspace_dir": sandbox.get("workspace_dir", ""),
            "state_dir": sandbox.get("native_store_dir", ""),
            "home_dir": sandbox.get("home_dir", ""),
            "cwd_dir": sandbox.get("cwd_dir", ""),
        }

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

        # Create the full sandbox skeleton up-front so ingest + bridge can
        # just write files without mkdir guards.
        base = Path(paths["base_dir"])
        base.mkdir(parents=True, exist_ok=True)
        Path(paths["memory_dir"]).mkdir(parents=True, exist_ok=True)
        Path(paths["native_store_dir"]).mkdir(parents=True, exist_ok=True)
        (Path(paths["native_store_dir"]) / "memory").mkdir(parents=True, exist_ok=True)
        Path(paths["home_dir"]).mkdir(parents=True, exist_ok=True)
        Path(paths["cwd_dir"]).mkdir(parents=True, exist_ok=True)
        Path(paths["metrics_dir"]).mkdir(parents=True, exist_ok=True)
        Path(paths["events_path"]).touch(exist_ok=True)

        manifest = build_session_manifest(
            conv, dataset_name=self.config.get("dataset_name", "")
        )
        manifest_path = base / "session_manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

        # Write the OpenClaw-schema config the CLI will read via
        # OPENCLAW_CONFIG_PATH.
        backend_mode = self._openclaw_cfg.get("backend_mode", "hybrid")
        flush_mode = self._openclaw_cfg.get("flush_mode", "shared_llm")
        resolved = build_openclaw_resolved_config(
            workspace_dir=paths["workspace_dir"],
            native_store_dir=paths["native_store_dir"],
            backend_mode=backend_mode,
            flush_mode=flush_mode,
            embedding=self._openclaw_cfg.get("embedding"),
        )
        resolved_config_path = Path(paths["config_path"])
        resolved_config_path.write_text(
            json.dumps(resolved, ensure_ascii=False, indent=2)
        )

        handle: dict = {
            "conversation_id": conv.conversation_id,
            "workspace_dir": paths["workspace_dir"],
            "memory_dir": paths["memory_dir"],
            "native_store_dir": paths["native_store_dir"],
            "home_dir": paths["home_dir"],
            "cwd_dir": paths["cwd_dir"],
            "resolved_config_path": str(resolved_config_path),
            "session_manifest_path": str(manifest_path),
            "prov_units_path": str(base / "prov_units.jsonl"),
            "artifact_bindings_path": str(base / "artifact_bindings.jsonl"),
            "events_path": paths["events_path"],
            "metrics_dir": paths["metrics_dir"],
            "backend_mode": backend_mode,
            "retrieval_route": self._openclaw_cfg.get(
                "retrieval_route", "search_then_get"
            ),
            "visibility_mode": self._openclaw_cfg.get("visibility_mode", "settled"),
            "flush_mode": flush_mode,
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
        """Render each session as markdown and ask OpenClaw to build its FTS/vector index.

        flush_mode selects between:
          * ``disabled``: raw transcript dumped to memory/session-*.md
          * ``native``  : LLM-driven selective retention (OpenClaw's
                         production memoryFlush behaviour approximated)

        The index step is always the real ``openclaw memory index --force``
        via the bridge - that is the point of faithful ingest. A bridge
        failure raises and propagates so the surrounding add() marks
        run_status=failed rather than silently producing an empty sandbox.
        """
        flush_mode = sandbox.get("flush_mode", "shared_llm")
        llm_generate = self._make_flush_generate() if flush_mode == "shared_llm" else None

        flush_plan: Optional[dict] = None
        if flush_mode == "shared_llm":
            flush_plan = await self._fetch_native_flush_plan(sandbox)
            sandbox["flush_plan_native"] = bool(flush_plan and flush_plan.get("native"))
            self._append_events(
                sandbox,
                [
                    {
                        "event": "flush_plan_resolved",
                        "native": bool(flush_plan and flush_plan.get("native")),
                        "relative_path": (flush_plan or {}).get("relative_path"),
                        "soft_threshold_tokens": (flush_plan or {}).get(
                            "soft_threshold_tokens"
                        ),
                    }
                ],
            )

        rows = await write_session_files(
            conversation=conv,
            memory_dir=Path(sandbox["memory_dir"]),
            flush_mode=flush_mode,
            llm_generate=llm_generate,
            flush_plan=flush_plan,
            honor_silent_token=bool(self._openclaw_cfg.get("honor_silent_token", False)),
        )
        self._append_events(sandbox, [{"event": "session_ingested", **r} for r in rows])

        # Drive the real OpenClaw index build so search has something to hit.
        index_resp = await arun_bridge(
            self._bridge_script_path(),
            {
                **self._bridge_base_payload(sandbox),
                "command": "index",
            },
            timeout=self._index_timeout(),
        )
        sandbox["last_index_epoch"] = int(index_resp.get("index_epoch") or 0)
        sandbox["visibility_state"] = "ingested"
        self._append_events(
            sandbox,
            [{"event": "index_complete", "index_epoch": sandbox["last_index_epoch"]}],
        )

    async def _flush_and_settle_if_needed(self, sandbox: dict) -> None:
        """Transition visibility_state to its final value per visibility_mode.

        Post-conditions:
          * visibility_mode == 'settled': visibility_state becomes 'settled'
            **only** when OpenClaw's ``memory status`` returns settled=true.
            Otherwise this method raises so add() fails fast rather than
            persisting a handle that lies about being queryable.
          * visibility_mode == 'eventual': we do not wait; visibility_state
            stays at 'indexed' and the caller accepts that search() may
            trigger a background re-sync via memorySearch.sync.onSearch.
        """
        if sandbox.get("visibility_mode") != "settled":
            sandbox["visibility_state"] = "indexed"
            return

        status_resp = await arun_bridge(
            self._bridge_script_path(),
            {**self._bridge_base_payload(sandbox), "command": "status"},
            timeout=self._status_timeout(),
        )
        sandbox["last_flush_epoch"] = int(status_resp.get("flush_epoch") or 0)
        settled = status_resp.get("settled") is True
        self._append_events(
            sandbox,
            [
                {
                    "event": "status_checked",
                    "settled": settled,
                    "flush_epoch": sandbox["last_flush_epoch"],
                }
            ],
        )
        if not settled:
            raise RuntimeError(
                "openclaw status reported not settled for "
                f"{sandbox.get('conversation_id')!r}: {status_resp!r}"
            )
        sandbox["visibility_state"] = "settled"

    def _assert_visibility_contract(self, sandbox: dict) -> None:
        """Guard the plan's settled-mode guarantee at the add() boundary."""
        if sandbox.get("visibility_mode") == "settled":
            vs = sandbox.get("visibility_state")
            if vs != "settled":
                raise RuntimeError(
                    f"settled mode requires visibility_state=='settled' but got {vs!r}"
                )

    # -- helpers -----------------------------------------------------------

    def _make_flush_generate(self):
        """Return a coroutine callable that hits our LLM provider with
        (system_prompt, user_prompt) and returns plain text.

        Includes retry-with-backoff because Sophnet (and OpenAI-compat
        endpoints in general) returns transient 500s under load. Failing
        an entire LoCoMo run because one out of several thousand flush
        calls hit a 500 is wasteful; cap the retries so a real outage
        still surfaces.
        """
        max_retries = int(self._openclaw_cfg.get("flush_max_retries", 4))
        base_delay = float(self._openclaw_cfg.get("flush_retry_base_seconds", 2.0))

        async def _call(system_prompt: str, user_prompt: str) -> str:
            provider = self._get_llm_provider()
            prompt = f"{system_prompt}\n\n{user_prompt}"
            last_err: Optional[Exception] = None
            for attempt in range(max_retries):
                try:
                    result = await provider.generate(prompt=prompt, temperature=0)
                    return result.strip() if isinstance(result, str) else ""
                except Exception as err:  # noqa: BLE001
                    last_err = err
                    if attempt == max_retries - 1:
                        break
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        "flush LLM call failed (attempt %d/%d): %s; retrying in %.1fs",
                        attempt + 1, max_retries, err, delay,
                    )
                    await asyncio.sleep(delay)
            raise RuntimeError(
                f"flush LLM exhausted {max_retries} retries: {last_err}"
            )

        return _call

    def _append_events(self, sandbox: dict, events: list[dict]) -> None:
        path = Path(sandbox.get("events_path", ""))
        if not path:
            return
        try:
            with path.open("a", encoding="utf-8") as fp:
                for event in events:
                    fp.write(json.dumps(event, ensure_ascii=False) + "\n")
        except Exception as err:  # noqa: BLE001
            logger.warning("failed to append events to %s: %s", path, err)

    def _index_timeout(self) -> float:
        return float(self._openclaw_cfg.get("index_timeout_seconds", 600.0))

    def _status_timeout(self) -> float:
        return float(self._openclaw_cfg.get("status_timeout_seconds", 60.0))

    async def _fetch_native_flush_plan(self, sandbox: dict) -> Optional[dict]:
        """Call the bridge's build_flush_plan to get OpenClaw's own flush
        plan (system_prompt / prompt / silent_token).

        NOTE: the bridge payload deliberately OMITS config_path for this
        command. Our runtime openclaw.json sets memoryFlush.enabled=false
        (so OpenClaw doesn't re-flush during search), but OpenClaw's
        ``buildMemoryFlushPlan`` returns null when that flag is off. We
        want the prompt text, not the runtime behaviour, so we ask the
        bridge to build the plan from OpenClaw defaults instead.

        Non-fatal: if the bridge cannot produce a plan (stub mode, missing
        dist, upstream error) we return None and the caller falls back to
        the in-process template. The event log records which branch we
        took.
        """
        base = self._bridge_base_payload(sandbox)
        plan_payload = {
            **base,
            "config_path": "",  # force OpenClaw defaults, not our runtime cfg
            "command": "build_flush_plan",
        }
        try:
            resp = await arun_bridge(
                self._bridge_script_path(),
                plan_payload,
                timeout=self._status_timeout(),
            )
        except Exception as err:  # noqa: BLE001
            logger.warning("build_flush_plan failed, falling back: %s", err)
            return None
        if not resp.get("ok") or resp.get("disabled") or not resp.get("system_prompt"):
            return None
        return {
            "native": bool(resp.get("native")),
            "system_prompt": resp["system_prompt"],
            "prompt": resp["prompt"],
            "silent_token": resp.get("silent_token") or "NO_REPLY",
            "relative_path": resp.get("relative_path"),
            "soft_threshold_tokens": resp.get("soft_threshold_tokens"),
        }

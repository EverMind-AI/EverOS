"""EverOS LoCoMo Benchmark Runner — typed pipeline with JSONL I/O.

Per-conversation pipeline: ADD -> wait_ready -> SEARCH -> ANSWER -> JUDGE.
Multiple conversations run in parallel via ThreadPoolExecutor.

Usage:
    python benchmarks/run.py --run-name baseline-v1
    python benchmarks/run.py --run-name baseline-v1 --smoke
    python benchmarks/run.py --run-name baseline-v1 --stages search answer judge
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import hashlib
import json
import os
import platform
import re
import shutil
import signal
import socket
import sqlite3
import statistics
import subprocess
import sys
import textwrap
import threading
import time
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple

import adapters
import openai
import requests
from adapters._profile import with_profile_block
from dotenv import load_dotenv

# Both this file's own directory and the repo root go on sys.path. The repo root is
# for `python -m benchmarks.run`; the file's own directory is what makes a standalone
# copy of this tree work -- each dataset under Evaluation/ carries its own complete
# copy, with no `benchmarks` package above it, so a package-qualified import of its
# sibling would fail there while looking fine here.
_here = str(Path(__file__).resolve().parent)
_repo_root = str(Path(__file__).resolve().parent.parent)
for _p in (_here, _repo_root):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import (  # noqa: E402
    AnswerResult,
    BenchmarkConfig,
    JudgeResult,
    RunSpec,
    SearchResult,
    ServingSpec,
    unresolved,
)
from tqdm import tqdm as _tqdm  # noqa: E402

_BAR_WIDTH = 16
_IS_TTY = sys.stdout.isatty()
_FILL_BG = "\033[44m" if _IS_TTY else ""
_EMPTY_BG = "\033[48;5;237m" if _IS_TTY else ""
_RESET = "\033[0m" if _IS_TTY else ""


class _ColorBarTqdm(_tqdm):
    """tqdm subclass with fixed fill + background colors."""

    @staticmethod
    def format_meter(  # type: ignore[override]
        n,
        total,
        elapsed,
        ncols=None,
        prefix="",
        ascii=False,
        unit="it",
        unit_scale=False,
        rate=None,
        bar_format=None,
        postfix=None,
        unit_divisor=1000,
        initial=0,
        colour=None,
        **extra_kwargs,
    ) -> str:
        if bar_format and bar_format == "{desc}":
            return prefix

        frac = n / total if total else 0
        filled = int(_BAR_WIDTH * frac)
        empty = _BAR_WIDTH - filled
        bar = f"{_FILL_BG}{' ' * filled}{_EMPTY_BG}{' ' * empty}{_RESET}"

        pct = f"{frac * 100:3.0f}%"

        elapsed_str = _tqdm.format_interval(elapsed)
        rate_val = n / elapsed if elapsed and n else 0
        remaining = (total - n) / rate_val if rate_val and total else 0
        remaining_str = _tqdm.format_interval(remaining) if total else "?"

        return f"{prefix} {pct} {bar} {n}/{total} [{elapsed_str}<{remaining_str}]"


# =============================================================================
# Inline prompts (originally from everosos-opensource evaluation/)
# =============================================================================


# =============================================================================
# Category labels
# =============================================================================


def _category_label(config: BenchmarkConfig, cat_key: object) -> str:
    """This benchmark's name for a category id.

    The map lives in the adapter, next to the loader that assigns the ids. A second copy
    here drifted from it and mislabelled every per-category line in the report.
    """
    names = adapters.get(config.adapter or "locomo").categories()
    key = str(cat_key)
    return names.get(key, names.get(cat_key, key))


# =============================================================================
# Minimal HTTP client for everos (single-tenant, no auth headers)
# =============================================================================


class EverosClient:
    """Minimal HTTP client for everos's /api/v1/memory/* endpoints.

    The default timeout is an hour, not the usual tens of seconds: `/memory/flush`
    returns only after the server has extracted that whole session, and a dense
    conversation can keep it busy far past ten minutes. At 600s SubtleMemory aborted
    mid-run with `Read timed out` after search had already completed 262 questions --
    the server was working fine, the client simply gave up on it.
    """

    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 3600):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def post(self, path: str, data: dict[str, Any]) -> tuple[int, dict]:
        full_url = f"{self.base_url}{path}"
        try:
            resp = requests.post(
                full_url,
                json=data,
                headers={"Content-Type": "application/json"},
                timeout=(10, self.timeout),
            )
        except requests.RequestException as e:
            # A transport error is reported as a negative status so the caller can retry
            # it like a 5xx. Letting it propagate would end the stage on one blip, and
            # over a multi-hour ingest a blip is routine.
            return -1, {"error": str(e)}
        try:
            return resp.status_code, resp.json()
        except ValueError:
            return resp.status_code, {}


def print_section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print(f"{'=' * 72}")


# =============================================================================
# LLM client pool -- round-robin across multiple API keys with 429 failover
# =============================================================================


def _split_keys(s: str) -> list[str]:
    """Split a comma-separated key string into a list of stripped non-empty keys."""
    return [k.strip() for k in s.split(",") if k.strip()]


class _PoolCompletions:
    def __init__(self, pool: LLMClientPool):
        self._pool = pool

    def create(self, **kwargs: Any) -> Any:
        return self._pool._create_with_failover(**kwargs)


class _PoolChat:
    def __init__(self, pool: LLMClientPool):
        self.completions = _PoolCompletions(pool)


class LLMClientPool:
    """Round-robin pool of openai.OpenAI clients with RateLimitError failover.

    Duck-types openai.OpenAI: callers may use ``pool.chat.completions.create(...)``
    transparently. On RateLimitError, the next key in the pool is tried; after
    all keys are exhausted, the last error is re-raised. Other errors propagate
    immediately (they're not "this key is throttled" signals).

    When ``base_url`` points to OpenRouter, the pool injects
    ``extra_body={"provider": {"only": [...], "allow_fallbacks": ...}}`` for any model
    whose vendor appears in ``providers`` (see BenchmarkConfig.providers), so the
    serving provider -- and with it the quantisation -- is fixed. A vendor absent from
    the mapping is left to OpenRouter's own routing. A vendor that declares more than
    one provider is declaring alternatives, which only mean something if the router may
    fall back between them, so ``allow_fallbacks`` follows the count.

    Env overrides, all inert unless set:

    * ``OPENROUTER_PROVIDER_ONLY`` (comma-separated, or ``any`` to disable pinning) --
      replaces the mapping for every model in the run.
    * ``OPENROUTER_PROVIDER_JSON`` -- per-model escape hatch for an open-weights model
      published under a vendor namespace that vendor does not serve:
      ``openai/gpt-oss-120b`` is served by DeepInfra and Novita, not by openai, so the
      vendor rule pins it to a provider without the model and every call 404s. Flat form
      ``{"only":[...],"allow_fallbacks":true}`` applies to every model; map form
      ``{"gpt-oss":{"only":["deepinfra"]}}`` matches by substring, first match wins, and
      anything unmatched keeps the vendor rule.
    * ``OPENROUTER_ALLOW_FALLBACKS=1`` -- keep the pins, just permit fallbacks.
    * ``LLM_EXTRA_BODY`` -- JSON merged into the extra body of every answer and judge
      request, e.g. ``{"chat_template_kwargs":{"enable_thinking":false}}`` to suppress a
      reasoning model's think phase. Without it such a model thinks on every call, which
      changes the answers, the token counts and the latency.
    """

    def __init__(
        self,
        api_keys: list[str],
        base_url: str,
        providers: Mapping[str, str] | None = None,
        **kwargs: Any,
    ):
        if not api_keys:
            raise ValueError("LLMClientPool: at least one API key required")
        self._providers = dict(providers or {})
        self._clients = [
            openai.OpenAI(api_key=k, base_url=base_url, **kwargs) for k in api_keys
        ]
        self._idx = 0
        self._lock = threading.Lock()
        self.key_count = len(self._clients)
        self.chat = _PoolChat(self)
        self._provider_constraint = self._resolve_provider_constraint(base_url)
        try:
            self._extra_body_default = json.loads(os.getenv("LLM_EXTRA_BODY") or "{}")
        except json.JSONDecodeError:
            # Never fail a run on a malformed override; an unset default is the
            # pre-existing behaviour.
            self._extra_body_default = {}

    @staticmethod
    def _resolve_provider_constraint(base_url: str) -> dict[str, Any] | str | None:
        """Resolve the OpenRouter ``provider`` block: a fixed one, ``auto``, or None."""
        if "openrouter" not in (base_url or "").lower():
            return None
        raw = os.getenv("OPENROUTER_PROVIDER_ONLY")
        if raw is None:
            return "auto"  # unset: derive per request from the declared vendor mapping
        raw = raw.strip()
        if not raw or raw.lower() == "any":
            # Set but blank is how an operator turns pinning off, and it is not the same
            # as leaving it unset.
            return None
        only = [p.strip() for p in raw.split(",") if p.strip()]
        return {"only": only, "allow_fallbacks": False}

    @staticmethod
    def _model_override(model: str) -> dict[str, Any] | None:
        """The OPENROUTER_PROVIDER_JSON escape hatch, or None if it does not apply."""
        raw = os.getenv("OPENROUTER_PROVIDER_JSON", "").strip()
        if not raw:
            return None
        try:
            ov = json.loads(raw)
        except json.JSONDecodeError:
            # Degrade to the vendor rule rather than failing the run.
            return None
        if not isinstance(ov, dict):
            return None
        if "only" in ov or "allow_fallbacks" in ov:
            return ov  # flat form: every model in the run
        low = (model or "").lower()
        for key, block in ov.items():
            if key.lower() in low and isinstance(block, dict):
                return block  # map form: first substring match wins
        return None

    @staticmethod
    def _vendor_of(model: str) -> str:
        """The vendor a model id belongs to: by prefix, else by name, else unknown.

        An id with no vendor prefix still has to be placed, because a bare name is how
        a self-hosted endpoint spells its model. A bare name matching none of the known
        vendors returns "" and so goes unpinned -- pinning it to a default vendor is how
        a bare qwen id ends up routed to a provider that does not serve it.
        """
        if "/" in model:
            return model.split("/", 1)[0]
        low = model.lower()
        if low.startswith("gpt"):
            return "openai"
        if "claude" in low:
            return "anthropic"
        if "gemini" in low:
            return "google"
        return ""

    def _constraint_for(self, model: str) -> dict[str, Any] | None:
        """The provider block for one request, from the override then the mapping."""
        override = self._model_override(model)
        if override is not None:
            return override
        vendor = self._vendor_of(model)
        # The allow-list is per vendor, because a pin only means anything against the
        # providers serving that vendor's model: ["openai"] 404s a deepseek model
        # outright. A vendor with no entry is left unpinned rather than guessed at.
        only = [
            s.strip() for s in self._providers.get(vendor, "").split(",") if s.strip()
        ]
        if not only:
            return None
        block = {"only": only, "allow_fallbacks": len(only) > 1}
        if os.getenv("OPENROUTER_ALLOW_FALLBACKS", "").strip() in ("1", "true", "yes"):
            block = {**block, "allow_fallbacks": True}
        return block

    def _next_client(self) -> openai.OpenAI:
        with self._lock:
            c = self._clients[self._idx]
            self._idx = (self._idx + 1) % len(self._clients)
            return c

    def _create_with_failover(self, **kwargs: Any) -> Any:
        constraint = self._provider_constraint
        if constraint == "auto":
            constraint = self._constraint_for(str(kwargs.get("model") or ""))
        if constraint is not None or self._extra_body_default:
            extra = dict(kwargs.get("extra_body") or {})
            if constraint is not None:
                extra.setdefault("provider", constraint)
            for key, value in self._extra_body_default.items():
                extra.setdefault(key, value)
            kwargs["extra_body"] = extra
        last_err: Exception | None = None
        for _ in range(len(self._clients)):
            client = self._next_client()
            try:
                return client.chat.completions.create(**kwargs)
            except openai.RateLimitError as e:
                last_err = e
                _tqdm.write(
                    f"  [warn] RateLimitError, rotating key "
                    f"({_ + 1}/{len(self._clients)})"
                )
                continue
        assert last_err is not None
        raise last_err


# =============================================================================
# Helpers
# =============================================================================


def _quiesce_servers(base_urls: Sequence[str] | str) -> None:
    """Freeze every server's md -> LanceDB projection before the read stages.

    Drains the projection queue, then stops cascade. Both halves matter. Without
    the drain, retrieval reads an index that is missing whatever had not been
    projected yet and scores the gap as a retrieval miss. Without the stop,
    cascade keeps compacting and pruning underneath the reader -- and prune's 60s
    retention window is shorter than one full-text-decider search (56-72s
    measured), so it reclaims files the search still holds.

    A server that reports no cascade (503) is already read-only; that is the
    expected answer when :func:`server_env_for` disabled the subsystem at startup,
    so it is logged and skipped rather than treated as a failure. Anything else IS a
    failure: proceeding would run the read stages against a live projection,
    which is the exact configuration that lost 225 of 493 questions.
    """
    urls = [base_urls] if isinstance(base_urls, str) else list(base_urls)
    for url in urls:
        client = EverosClient(url)
        status, resp = client.post("/api/v1/cascade/quiesce", {})
        if status == 503:
            print(f"  [quiesce] {url}: cascade not running (already read-only)")
            continue
        if status != 200:
            raise RuntimeError(
                f"cascade quiesce failed on {url}: status={status} detail={resp}"
            )
        data = resp.get("data", resp) if isinstance(resp, dict) else {}
        left = data.get("pending_after")
        note = f"  [quiesce] {url}: drained {data.get('drained')}"
        if left:
            # Not fatal -- the run is still worth having -- but it means the index
            # is an incomplete projection, so say so where it will be read.
            note += f"  WARNING {left} still pending (index is incomplete)"
        print(note)


def server_env_for(
    stages: Sequence[str], retrieval_env: Mapping[str, str]
) -> dict[str, str]:
    """The environment the servers this run starts should get.

    A run with no ADD stage never writes markdown, so both background subsystems are
    pure overhead on it -- and both actively hurt. Cascade's periodic scan re-enqueues
    the whole store, which starved search on a dense one; OME holds an exclusive
    per-store lock, which stops a second server from sharing one pre-built store root,
    the shape a parallel-lane evaluation needs. Only set when ADD is absent, so an
    ingesting run still extracts.
    """
    env = {str(k): str(v) for k, v in retrieval_env.items()}
    if "add" not in stages:
        env["EVEROS_DISABLE_CASCADE"] = "1"
        env["EVEROS_DISABLE_OME"] = "1"
    return env


_BIND_FAILURE_MARKERS = (
    "address already in use",
    "errno 98",
    "errno 48",
    "cannot assign requested address",
)


def _looks_like_bind_failure(log_tail: str) -> bool:
    """Whether a dead server's log says it lost the port, not something else.

    Only a bind failure justifies retrying at another port block. An empty tail
    counts as a bind failure: a child killed before it wrote anything is exactly
    what losing the race looks like, and treating it as a hard error would turn a
    recoverable collision into a failed run.
    """
    tail = log_tail.strip().lower()
    if not tail or tail == "(no server log)":
        return True
    return any(m in tail for m in _BIND_FAILURE_MARKERS)


class _PortRaceLostError(RuntimeError):
    """A child exited before readiness, which on this path means it lost a bind race."""


class _ServerFleet:
    """Servers started by the run, torn down with it.

    Sharded ADD needs one server per shard, each on its own root, because the index
    queue lock is exclusive. Making the caller spell that out -- a shell loop over
    `everos server start`, then a matching list of --base-url and --everos-root -- leaks
    an implementation detail into the interface and is easy to get subtly wrong (a
    mismatched pair silently polls the wrong store). `--servers N` is the whole knob.
    """

    def __init__(
        self,
        n: int,
        base_root: Path,
        first_port: int,
        backbone_model: str = "",
        backbone_base_url: str = "",
        backbone_api_key: str = "",
        decider_model: str = "",
        decider_base_url: str = "",
        decider_api_key: str = "",
        extra_env: dict[str, str] | None = None,
        trace_dir: Path | None = None,
    ) -> None:
        self.decider_model = decider_model
        self.decider_base_url = decider_base_url
        self.decider_api_key = decider_api_key
        self.backbone_model = backbone_model
        self.backbone_base_url = backbone_base_url
        self.backbone_api_key = backbone_api_key
        self.extra_env = dict(extra_env or {})
        self.trace_dir = trace_dir
        self.procs: list[subprocess.Popen] = []
        self.urls: list[str] = []
        self.roots: list[str] = []
        self.logs: list[Path] = []
        self.n = n
        self._base_root = base_root
        self._first_port = first_port

    @staticmethod
    @staticmethod
    def _free_port(start: int) -> tuple[int, socket.socket]:
        """A port plus the socket holding it, which the caller closes just before spawn.

        Probing with `connect_ex` and returning the number is a race: two runs starting
        at the same moment both saw the port free, both spawned, and one child died at
        bind while a DIFFERENT run's server answered /health on that port. The losing
        run then searched another dataset's store and got HTTP 200 with zero episodes
        for every question -- a complete-looking 0.0%. Binding here means a concurrent
        probe finds the port taken, so the window closes.
        """
        for port in range(start, start + 400):
            s = socket.socket()
            try:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("127.0.0.1", port))
                s.listen(1)
            except OSError:
                s.close()
                continue
            return port, s
        raise RuntimeError(f"no free port in {start}..{start + 400}")

    def start(self, attempts: int = 4) -> None:
        """Start the fleet, moving to a different port block if a child loses a race.

        The base port is derived from the run name so a relaunch reuses its own ports
        instead of colliding with a lane still running. Holding the port until spawn
        narrows the race but cannot close it: there is still a gap between releasing the
        socket and the child binding it, and two runs launched in the same instant both
        pass through it. So a child that dies at bind is not a failure -- it is a lost
        race, and the fleet moves to another block. Only after every attempt is it an
        error, because by then the cause is not contention.
        """
        base = self._first_port
        for attempt in range(attempts):
            self._first_port = (
                base + attempt * 149
            )  # prime stride: no overlap with n<149
            try:
                self._start_once()
                return
            except _PortRaceLostError as e:
                _tqdm.write(
                    f"  [warn] port block {self._first_port} lost to a concurrent run "
                    f"({e}); retrying at {base + (attempt + 1) * 149}"
                )
                self.stop()
                self.procs.clear()
                self.urls.clear()
                self.roots.clear()
        raise RuntimeError(
            f"could not claim a free port block after {attempts} attempts from {base}"
        )

    def _start_once(self) -> None:
        exe = Path(sys.executable).parent / "everos"
        port = self._first_port
        for i in range(self.n):
            root = (
                self._base_root
                if self.n == 1
                else self._base_root.parent / f"{self._base_root.name}_s{i}"
            )
            root.mkdir(parents=True, exist_ok=True)
            if not (root / "everos.toml").exists():
                subprocess.run(
                    [str(exe), "init", "--root", str(root)],
                    check=True,
                    capture_output=True,
                )
            port, _held = self._free_port(port)
            # Released immediately before the child binds it. Every other run's probe
            # has seen it occupied up to this point.
            _held.close()
            log = root / "server.log"
            srv_env = {**os.environ, "EVEROS_ROOT": str(root)}
            # The retrieval trace is what every bad-case attribution reads, and a run
            # without it cannot be attributed afterwards -- which is why the reference
            # launchers all export it. It takes a path, so it cannot live in a config
            # file as a flag; it is derived per server here instead. An operator who
            # sets it themselves keeps their own value.
            if self.trace_dir is not None and not os.environ.get(
                "EVEROS_LLMMR_TRACE_DUMP"
            ):
                self.trace_dir.mkdir(parents=True, exist_ok=True)
                srv_env["EVEROS_LLMMR_TRACE_DUMP"] = str(
                    self.trace_dir / f"trace_port{port}.jsonl"
                )
            # Profile extraction runs in OME, concurrently across owners, so it gets
            # its own file: interleaved into the retrieval trace it would be
            # unreadable. Same reason the retrieval file is per-server -- one writer.
            # Same two guards as above: no trace_dir means the caller wants no dump,
            # and an operator-set env keeps their own value.
            if self.trace_dir is not None and not os.environ.get(
                "EVEROS_PROFILE_TRACE_DUMP"
            ):
                self.trace_dir.mkdir(parents=True, exist_ok=True)
                srv_env["EVEROS_PROFILE_TRACE_DUMP"] = str(
                    self.trace_dir / f"profile_port{port}.jsonl"
                )
            if self.backbone_model:
                srv_env["EVEROS_LLM__MODEL"] = self.backbone_model
            if self.backbone_base_url:
                srv_env["EVEROS_LLM__BASE_URL"] = self.backbone_base_url
            # EVEROS_DECIDER__* is a separate section: extraction keeps [llm], while the
            # multi-round retrieval decider runs this model instead.
            if self.decider_model:
                srv_env["EVEROS_DECIDER__MODEL"] = self.decider_model
            if self.decider_base_url:
                srv_env["EVEROS_DECIDER__BASE_URL"] = self.decider_base_url
            if self.decider_api_key:
                srv_env["EVEROS_DECIDER__API_KEY"] = self.decider_api_key
            if self.backbone_api_key:
                srv_env["EVEROS_LLM__API_KEY"] = self.backbone_api_key
            srv_env.update({k: str(v) for k, v in self.extra_env.items()})
            self.procs.append(
                subprocess.Popen(
                    [str(exe), "server", "start", "--port", str(port)],
                    env=srv_env,
                    # The child writes to this for its whole life, so the handle
                    # cannot be closed when this call returns.
                    stdout=open(log, "w"),  # noqa: SIM115
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            )
            self.urls.append(f"http://127.0.0.1:{port}")
            self.roots.append(str(root))
            self.logs.append(log)
            port += 1
        self._await_ready()

    def _log_tail(self, url: str, lines: int = 12) -> str:
        """Last few lines the dead child wrote, for the failure message."""
        try:
            log = self.logs[self.urls.index(url)]
            return "\n".join(log.read_text(errors="replace").splitlines()[-lines:])
        except (ValueError, IndexError, OSError):
            return "(no server log)"

    def _await_ready(self, timeout_s: float = 180.0) -> None:
        import urllib.error
        import urllib.request

        deadline = time.time() + timeout_s
        pending = set(self.urls)
        while pending and time.time() < deadline:
            # A child that has exited cannot be the thing answering, so its port now
            # belongs to somebody else. `/health` carries no store identity, so this is
            # the check that tells "my server is up" from "some server is up".
            dead = [
                (u, pr.returncode)
                for u, pr in zip(self.urls, self.procs, strict=False)
                if pr.poll() is not None
            ]
            if dead:
                # A dead child is only a lost port race if it died FAILING TO BIND.
                # Anything else (a schema guard, a bad provider, a missing dep) is the
                # server's own error, and retrying at another port block just repeats
                # it -- burning every attempt and then reporting "could not claim a free
                # port block", which is a lie that hides the real cause in a log nobody
                # was told to read. Measured: a `user_profile` schema drift cost three
                # debugging rounds because of exactly that.
                tails = {u: self._log_tail(u) for u, _ in dead}
                if not any(_looks_like_bind_failure(t) for t in tails.values()):
                    detail = "\n".join(
                        f"  {u} exit={code}\n"
                        + textwrap.indent(tails.get(u, ""), "    ")
                        for u, code in dead
                    )
                    raise RuntimeError(
                        "EverOS server(s) exited during startup for their own reason, "
                        f"not a port conflict:\n{detail}"
                    )
                # Distinguished from a real failure so `start` can move to another port
                # block instead of failing the run. Either way it never proceeds against
                # whichever store happens to answer on that port.
                raise _PortRaceLostError(
                    f"server process(es) exited before becoming ready: {dead}"
                )
            for url in sorted(pending):
                try:
                    with urllib.request.urlopen(f"{url}/health", timeout=5) as r:
                        if r.status == 200:
                            pending.discard(url)
                except (urllib.error.URLError, OSError, TimeoutError):
                    pass
            if pending:
                time.sleep(2.0)
        if pending:
            self.stop()
            raise RuntimeError(
                f"server(s) did not become ready within {timeout_s:.0f}s: "
                f"{sorted(pending)}"
                " -- see server.log under each root"
            )
        print(f"  Started {self.n} server(s): {', '.join(self.urls)}", flush=True)

    def stop(self) -> None:
        for proc in self.procs:
            if proc.poll() is None:
                proc.terminate()
        for proc in self.procs:
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()


def _parse_conv_spec(spec: list[str] | None, total: int) -> list[int]:
    """Expand `--conv` into indices: `all`, `0-499`, plain numbers, or a mix."""
    if not spec:
        return list(range(total))
    out: list[int] = []
    for tok in spec:
        tok = tok.strip()
        if tok.lower() == "all":
            out.extend(range(total))
        elif "-" in tok.lstrip("-"):
            lo, _, hi = tok.partition("-")
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(tok))
    # An index the dataset does not have must not reach the loader: it either raises
    # deep in a stage or, worse, silently shrinks the denominator the accuracy is
    # computed over.
    if not out:
        raise SystemExit(
            f"--conv {' '.join(spec)} selects no conversations. A descending range "
            f"like `5-2` yields nothing, and the run would finish with an empty "
            f"report and exit 0, indistinguishable from a completed run."
        )
    bad = sorted({c for c in out if not 0 <= c < total})
    if bad:
        raise SystemExit(
            f"--conv names conversation(s) this benchmark does not have: {bad} "
            f"(it has {total}, so 0-{total - 1})"
        )
    # De-duplicate while keeping the given order; a conversation listed twice would be
    # ingested twice.
    seen: set[int] = set()
    return [c for c in out if not (c in seen or seen.add(c))]


def _prompt(config: BenchmarkConfig, name: str) -> str:
    """This benchmark's prompt. Each adapter carries its own -- nothing to select."""
    return getattr(adapters.get(config.adapter or "locomo"), name)


def _stratified_sample(qa_list: list[dict], *, n: int = 10) -> list[dict]:
    """Pick up to *n* QA items evenly across all categories present.

    Round-robins across categories so each gets roughly ``n / num_cats``
    items. Preserves original order within each category.
    """
    by_cat: dict[int, list[dict]] = {}
    for qa in qa_list:
        cat = qa.get("category")
        if cat is not None:
            by_cat.setdefault(cat, []).append(qa)

    selected: list[dict] = []
    while len(selected) < n:
        picked_any = False
        for cat in sorted(by_cat):
            if len(selected) >= n:
                break
            if by_cat[cat]:
                selected.append(by_cat[cat].pop(0))
                picked_any = True
        if not picked_any:
            break

    # Restore original order
    order = {id(qa): i for i, qa in enumerate(qa_list)}
    selected.sort(key=lambda q: order[id(q)])
    return selected


def _check_failures(raw: list) -> None:
    """Raise if any element in *raw* is an exception from ``_parallel_map``.

    A DELIBERATE divergence from the reference, which turns such an exception into an
    `[ERROR: ...]` row and grades it WRONG. That is right for an API failure and wrong
    for a bug: it converts a broken stage into a low accuracy number nobody
    investigates.

    Every API and model failure is now handled inside the item that hit it -- retrieval
    records `search_error`, the answer stage records `[ERROR]`/`[ANSWER_EMPTY]`, the
    judge grades WRONG -- and all of them stay in the denominator. So an exception
    arriving here is either a defect in this harness or a `BudgetExhaustedError`, and
    both should stop the run rather than become a number: retrying a spent account
    cannot help, and every question after the credit ran out would be graded wrong. A
    missing required field on AnswerResult was caught this way within seconds of being
    written.
    """
    errors = [(i, item) for i, item in enumerate(raw) if isinstance(item, Exception)]
    if not errors:
        return
    if len(errors) == 1:
        raise errors[0][1]
    msg = f"{len(errors)} failures:\n" + "\n".join(f"  [{i}] {e}" for i, e in errors)
    raise RuntimeError(msg) from errors[0][1]


def _parallel_map(
    items: list,
    worker,
    *,
    concurrency: int,
    pbar: _tqdm | None = None,
    on_result=None,
) -> list:
    """Run ``worker(i, item)`` over *items* concurrently; preserve input order.

    Updates *pbar* on each completion if provided. Falls back to serial
    execution when *concurrency* <= 1.
    """
    results: list = [None] * len(items)

    if concurrency <= 1:
        for i, item in enumerate(items):
            results[i] = worker(i, item)
            if on_result is not None:
                on_result(results[i])
            if pbar:
                pbar.update(1)
        return results

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        future_to_idx: dict[concurrent.futures.Future, int] = {
            pool.submit(worker, i, item): i for i, item in enumerate(items)
        }
        for fut in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[fut]
            try:
                results[idx] = fut.result()
                if on_result is not None:
                    # persist the moment it lands; a later failure cannot undo it
                    on_result(results[idx])
            except Exception as exc:
                results[idx] = exc
            if pbar:
                pbar.update(1)

    return results


# =============================================================================
# Wait for cascade + OME drain
# =============================================================================


class _OmeOutcome(NamedTuple):
    """What the readiness wait observed, so ADD can record it in the marker."""

    total: int
    failed: int


def _poll_cascade(db_path: Path, conv_pattern: str) -> tuple[int, int]:
    """Return (total, pending) for cascade md_change_state rows.

    A database that does not exist yet reads as an empty queue, matching `_poll_ome`.
    The two are polled in the same loop, and one of them raising while the other returns
    zeros made the wait fail on a store whose first flush had not landed.
    """
    if not db_path.exists():
        return 0, 0
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    total, pending = conn.execute(
        "SELECT COUNT(*), "
        "SUM(CASE WHEN status IN ('pending','processing') THEN 1 ELSE 0 END) "
        "FROM md_change_state WHERE md_path LIKE ?",
        (conv_pattern,),
    ).fetchone()
    conn.close()
    # SUM over no rows is NULL, which is not a count.
    return total or 0, pending or 0


def _poll_ome(
    ome_db_path: Path,
    since: str,
    ome_filter: str,
    ome_params: tuple[str, ...],
) -> tuple[int, int, int]:
    """Return (total, pending, failed) for OME run_record rows."""
    if not ome_db_path.exists():
        return 0, 0, 0
    conn = sqlite3.connect(f"file:{ome_db_path}?mode=ro", uri=True)
    rows = conn.execute(
        "SELECT status, COUNT(*) FROM run_record "
        f"WHERE started_at >= ? AND {ome_filter} "
        "GROUP BY status",
        (since, *ome_params),
    ).fetchall()
    conn.close()
    total, pending = 0, 0
    for status, count in rows:
        total += count
        if status == "running":
            pending += count
    # A retry is a NEW run_record row for the same event_id, not an update of the old
    # one, so counting rows with status='failed' counts attempts, not losses. On LoCoMo
    # that read as 25 failures when all 22 affected events had in fact succeeded on
    # retry -- the store was complete. Only an event with no successful run for its
    # strategy is genuinely lost.
    conn = sqlite3.connect(f"file:{ome_db_path}?mode=ro", uri=True)
    failed = conn.execute(
        "SELECT COUNT(*) FROM ("
        "  SELECT DISTINCT f.event_id, f.strategy_name FROM run_record f"
        "  WHERE f.started_at >= ? "
        f" AND {ome_filter} "
        "   AND f.status IN ('failed', 'dead_letter', 'crashed')"
        # Only count an attempt as lost once its retry budget is spent. A retry is
        # queued behind everything else in flight, so on a large ingest it lands 25-40
        # MINUTES after the failure (measured on LoCoMo: 1.1s when the queue is empty,
        # 2465s when it is not). Counting a fresh attempt=0 failure as a loss reports
        # data as missing while its retry is still sitting in the queue.
        "   AND f.attempt >= f.max_retries_snapshot"
        "   AND NOT EXISTS ("
        "     SELECT 1 FROM run_record s"
        "     WHERE s.event_id = f.event_id"
        "       AND s.strategy_name = f.strategy_name"
        "       AND s.status = 'success')"
        ")",
        (since, *ome_params),
    ).fetchone()[0]
    conn.close()
    return total, pending, failed


def _wait_ready(
    everos_root: str,
    conv_index: int,
    project_id: str,
    timeout_s: int,
    app_id: str = "locomo_benchmark",
    owner_id: str = "",
    ome_failure_tolerance: float = 0.01,
    owner_pattern: str = "",
    session_pattern: str = "",
    session_ids: Sequence[str] = (),
    poll_interval_s: float = 3.0,
    *,
    since: str = "",
    pbar: _tqdm | None = None,
) -> _OmeOutcome:
    """Wait until cascade queue AND OME jobs finish for a conversation."""
    root = Path(everos_root).expanduser()
    db_path = root / ".index" / "sqlite" / "system.db"
    ome_db_path = root / ".index" / "sqlite" / "ome.db"
    conv_pattern = f"%/{owner_id}/%" if owner_id else f"%_conv{conv_index}/%"

    if not db_path.exists():
        raise RuntimeError(
            f"Cascade DB not found at {db_path} — "
            f"is --everos-root ({everos_root}) correct? "
            f"It must match the server's --root."
        )

    # app_id and the owner / session patterns are parameters, not literals. They used to
    # be hardcoded to LoCoMo's naming ('locomo_benchmark', '%_conv{i}',
    # 'locomo_conv{i}_%'), which made this readiness check unusable for any other
    # dataset: the filter simply never matched, so instead of failing it waited out the
    # full timeout on every conversation. Defaults keep LoCoMo behaviour byte-for-byte.
    ome_filter = (
        "json_extract(event_payload, '$.app_id') = ? "
        "AND json_extract(event_payload, '$.project_id') = ? "
        "AND ("
        "  json_extract(event_payload, '$.owner_id') LIKE ? "
        "  OR json_extract(event_payload, '$.session_id') LIKE ?"
        ")"
    )
    ome_params: tuple[str, ...] = (
        app_id,
        project_id,
        owner_pattern or f"%_conv{conv_index}",
        session_pattern or f"locomo_conv{conv_index}_%",
    )
    if session_ids:
        placeholders = ",".join("?" * len(session_ids))
        ome_filter = (
            "json_extract(event_payload, '$.app_id') = ? "
            "AND json_extract(event_payload, '$.project_id') = ? "
            f"AND json_extract(event_payload, '$.session_id') IN ({placeholders})"
        )
        ome_params = (app_id, project_id, *session_ids)

    deadline = time.time() + timeout_s
    stable_count = 0
    cascade_pending = 0
    ome_pending = 0

    while time.time() < deadline:
        cascade_total, cascade_pending = _poll_cascade(db_path, conv_pattern)
        ome_total, ome_pending, ome_failed = _poll_ome(
            ome_db_path, since, ome_filter, ome_params
        )

        # A hard `ome_failed > 0` abort cannot work here. Every observed failure is the
        # backbone emitting malformed JSON (truncated objects, raw control characters),
        # which is a property of the model, not a fault to recover from: re-running the
        # conversation just draws a different sample and fails somewhere else, so an
        # abort-and-retry loop never converges. Tolerate a small fraction, and surface
        # it -- silently discarding failures would be the worse error.
        _ome_seen = max(ome_total, 1)
        if ome_failed > 0 and ome_failed / _ome_seen > ome_failure_tolerance:
            raise RuntimeError(
                f"{ome_failed}/{ome_total} OME task(s) failed for conv {conv_index} "
                f"({ome_failed / _ome_seen:.1%} > "
                f"{ome_failure_tolerance:.1%} tolerance) — data is incomplete, aborting"
            )

        if pbar is not None:
            _ct = cascade_total or 0
            _cp = cascade_pending or 0
            _ot = ome_total or 0
            _op = ome_pending or 0
            done = (_ct - _cp) + (_ot - _op)
            total = _ct + _ot
            if total > 0:
                pbar.total = total
                pbar.n = done
                pbar.refresh()

        if (cascade_pending or 0) == 0 and (ome_pending or 0) == 0:
            stable_count += 1
            if stable_count >= 2:
                if pbar is not None and pbar.total and pbar.total > 0:
                    pbar.n = pbar.total
                    pbar.refresh()
                if ome_failed > 0:
                    _tqdm.write(
                        f"  [warn] conv {conv_index}: {ome_failed}/{ome_total} "
                        f"extraction task(s) failed (malformed LLM JSON), within the "
                        f"{ome_failure_tolerance:.1%} tolerance — those facts are "
                        f"missing"
                    )
                return _OmeOutcome(total=ome_total, failed=ome_failed)
        else:
            stable_count = 0

        time.sleep(poll_interval_s)

    raise RuntimeError(
        f"Timeout after {timeout_s}s waiting for conv {conv_index} "
        f"(cascade_pending={cascade_pending}, ome_running={ome_pending}) "
        f"— increase cascade_timeout in config.toml"
    )


# =============================================================================
# Data loading -- preserve LoCoMo session_N structure for per-session flushing
# =============================================================================


def _parse_session_timestamp(ts_str: str) -> int:
    """Parse LoCoMo timestamp string to epoch milliseconds.

    Format examples: "1:56 pm on 8 May, 2023", "12:09 am on 13 September, 2023".

    LoCoMo's raw timestamps carry no timezone, so we pin them to UTC --
    matching ``everalgo/benchmarks/datasets/locomo/loader.py:_parse_timestamp``.
    Without an explicit tz, ``naive_dt.timestamp()`` would shift epochs by
    the OS's local-vs-UTC offset, so the same dataset would produce
    different absolute timestamps on different machines.
    """
    dt = datetime.strptime(ts_str.strip(), "%I:%M %p on %d %B, %Y")
    return int(dt.replace(tzinfo=UTC).timestamp() * 1000)


def load_conversation_via_adapter(
    benchmark: str, data_path: str, conv_index: int
) -> tuple[list[dict], list[dict], str, str]:
    """Sessions + questions for one unit, in the shape the ADD stage expects.

    LoCoMo keeps its original loader (speaker_a/speaker_b drive both the owner id and
    the answer prompt's framing). Everything else goes through the adapter, which is
    what the ADD stage was missing: it read unit["conversation"] directly, so
    LongMemEval, EverMemBench and SubtleMemory each died with KeyError: 'conversation'
    the moment ingestion started -- three full runs that burned two hours producing
    nothing.
    """
    if benchmark == "locomo":
        sessions, qa_list, spk_a, spk_b = load_conversation(data_path, conv_index)
        return sessions, qa_list, spk_a, spk_b, f"{spk_a.lower()}_conv{conv_index}"

    ad = adapters.get(benchmark)
    units = ad.load_units(data_path)
    if conv_index < 0 or conv_index >= len(units):
        raise ValueError(f"conv_index {conv_index} out of range (have {len(units)})")
    unit = units[conv_index]
    sessions = ad.sessions_of(unit)
    owner = ad.owner_of(unit, "speaker_a")
    # Resolve gold here: this is the loader the
    # SEARCH stage uses, and without it every result row was written with an empty
    # gold_sessions -- the IR and core metrics then had nothing to score against.
    qa_list = [
        {**qa, "gold_sessions": sorted(ad.gold_of(unit, qa))}
        for qa in (unit.get("qa") or [])
    ]
    # These two are LoCoMo's notion of "who is talking". A benchmark whose reference
    # renders a different pair says so; the rest have a single owner, so the owner id
    # stands in for both and the context template degrades to it.
    speaker_a, speaker_b = getattr(ad, "speakers_of", lambda _u: (owner, owner))(unit)
    # The owner is returned rather than inferred from `speaker_a`, because on
    # LongMemEval the two are different things and conflating them cost every question.
    # Its `speakers_of` returns `user_<question_id>` / `assistant_<question_id>` to
    # reproduce the reference's answer-prompt header, while the store's owner is
    # `longmemeval_0`. `run_conversation` used `speaker_a` as the owner, so every search
    # filtered on an owner with zero rows: HTTP 200, zero episodes, 0.1s, no error
    # anywhere -- 0.0% for the whole run. LoCoMo escaped it through its own branch and
    # the other two because their owner and speaker happen to be the same string.
    return sessions, qa_list, speaker_a, speaker_b, owner


def load_conversation(
    data_path: str, conv_index: int
) -> tuple[list[dict], list[dict], str, str]:
    """Load a LoCoMo conversation, preserving session_N boundaries.

    Returns (sessions, qa_list, speaker_a, speaker_b) where ``sessions`` is
    a list of {session_idx, messages} ordered by session_idx. Each message
    carries dia_id / speaker / text / timestamp_ms. QA list excludes
    category 5 (adversarial).
    """
    with open(data_path, encoding="utf-8") as f:
        dataset = json.load(f)

    if conv_index < 0 or conv_index >= len(dataset):
        raise ValueError(
            f"conv_index {conv_index} out of range "
            f"(dataset has {len(dataset)} conversations, valid: 0..{len(dataset) - 1})"
        )

    conv = dataset[conv_index]
    conversation = conv["conversation"]
    speaker_a = conversation["speaker_a"]
    speaker_b = conversation["speaker_b"]

    sessions: list[dict] = []
    session_idx = 1
    while True:
        session_key = f"session_{session_idx}"
        dt_key = f"session_{session_idx}_date_time"
        if dt_key not in conversation:
            break
        if session_key in conversation:
            ts_str = conversation[dt_key]
            base_ts_ms = _parse_session_timestamp(ts_str)
            session_msgs = conversation[session_key]
            if isinstance(session_msgs, list):
                msgs: list[dict] = []
                for i, msg in enumerate(session_msgs):
                    if not msg.get("text"):
                        continue  # skip image-only messages
                    msgs.append(
                        {
                            "dia_id": msg["dia_id"],
                            "speaker": msg["speaker"],
                            "text": msg["text"],
                            "timestamp_ms": base_ts_ms + i * 30000,
                        }
                    )
                if msgs:
                    sessions.append({"session_idx": session_idx, "messages": msgs})
        session_idx += 1

    qa_list = [q for q in conv.get("qa", []) if q.get("category") != 5]
    # Same reason as the adapter path: SEARCH writes gold_sessions straight from the qa
    # dict, so gold has to be resolved by the loader or the metrics have nothing to
    # read. LoCoMo cites evidence as ``D<session>:<turn>`` and the store names sessions
    # ``locomo_conv<i>_s<session>``.
    qa_list = [
        {
            **q,
            "gold_sessions": sorted(
                {
                    f"locomo_conv{conv_index}_s{int(m.group(1))}"
                    for ev in (q.get("evidence") or [])
                    if (m := re.match(r"D(\d+):", str(ev)))
                }
            ),
        }
        for q in qa_list
    ]
    return sessions, qa_list, speaker_a, speaker_b


# =============================================================================
# JSONL I/O helpers
# =============================================================================


def _write_jsonl(path: Path, items: list) -> None:
    """Write a list of Pydantic models (or dicts) to a JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            if hasattr(item, "model_dump_json"):
                f.write(item.model_dump_json())
            else:
                f.write(json.dumps(item, ensure_ascii=False, default=str))
            f.write("\n")


def _append_jsonl(path: Path, item) -> None:
    """Append one finished item and flush.

    Stages used to write their whole result list once, at the end. A crash at question
    190/199 therefore threw away all 190 -- and at full-benchmark scale (5,962 questions
    across four datasets) something does crash: a server dies, an endpoint stalls past
    the wall clock, a lane is killed. Appending per item costs one fsync per question
    and turns every such failure into a resumable one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        if hasattr(item, "model_dump_json"):
            f.write(item.model_dump_json())
        else:
            f.write(json.dumps(item, ensure_ascii=False, default=str))
        f.write("\n")
        f.flush()


def _done_keys(path: Path, key: str = "index") -> set[str]:
    """Keys already present in a partial artifact, for skipping on resume.

    Keyed on ``index``, not on the question text. LoCoMo asks the same question more
    than once inside a conversation -- conv7 has 191 questions but only 180 distinct
    strings -- so a text key silently marked the duplicates as finished. Five answers
    lost to a full disk were reported as "180 done, 0 to go" and the conversation was
    scored as complete on 186 of its 191 questions.

    Reads defensively: a run killed mid-write leaves a truncated final line, and that
    line must be ignored rather than aborting the resume.
    """
    done: set[str] = set()
    if not path.exists():
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue  # truncated tail from a hard kill
            v = rec.get(key)
            if v is not None:
                done.add(str(v))
    return done


def _read_jsonl(path: Path, model_cls: type, *, strict: bool = True) -> list:
    """Read a JSONL file into a list of Pydantic model instances.

    With ``strict=False`` an unparseable line is dropped instead of raising. A process
    killed mid-write leaves a torn final line, and `_done_keys` was already written to
    tolerate exactly that -- so the work was counted as done while every later read of
    the same file raised, which locked the stage permanently: the retry did the
    remaining questions, appended them, and still died in `_finalize_jsonl`. Dropping
    the torn line is safe because its question is then absent from the done set and
    simply gets redone.
    """
    results = []
    dropped = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                results.append(model_cls.model_validate_json(line))
            except ValueError:
                if strict:
                    raise
                dropped += 1
    if dropped:
        print(
            f"  [resume] {path.name}: dropped {dropped} unparseable line(s) from an "
            f"interrupted write; their questions will be redone",
            flush=True,
        )
    return results


def _finalize_jsonl(path: Path, model_cls: type) -> list:
    """Read a stage artifact back in question order, rewriting it in that order.

    Rows are appended as each worker finishes, which is what makes an interrupted stage
    resumable, but it leaves the file in completion order -- so two identical runs
    produce different files and cannot be diffed against each other. Sorting once at the
    end keeps both properties.
    """
    # Tolerant, and it rewrites the file, so the torn line is gone afterwards.
    results = sorted(_read_jsonl(path, model_cls, strict=False), key=lambda r: r.index)
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(r.model_dump_json() + "\n")
    return results


# =============================================================================
# Add phase -- one everos session_id per LoCoMo session, flush after each
# =============================================================================


def _post_with_retry(
    client: EverosClient,
    path: str,
    payload: dict,
    *,
    what: str,
    attempts: int = 5,
    base_delay: float = 2.0,
    tolerate: bool = False,
) -> dict | None:
    """POST, retrying server-side 5xx with exponential backoff.

    ADD had no retry at all -- a single `assert status == 200` abandoned the whole
    conversation on the first bad response. The 500s it hits are transient overload
    (starlette surfaces a CancelledError from an over-subscribed extraction path as
    TimeoutError), so 18% of conversations were being thrown away for something a retry
    clears. 4xx is not retried: that is a malformed request, and repeating it only
    wastes the budget.
    """
    last = None
    for attempt in range(attempts):
        status, resp = client.post(path, payload)
        if status == 200:
            return resp
        last = (status, resp)
        # A negative status is a transport error, which is as retryable as a 5xx. Only a
        # 4xx is final: the request is malformed and repeating it just burns budget.
        if 0 <= status < 500:
            break
        if attempt < attempts - 1:
            wait = min(base_delay * (2**attempt), 30.0)
            _tqdm.write(
                f"  [warn] {what} -> {status}, retry "
                f"{attempt + 1}/{attempts - 1} in {wait:.0f}s"
            )
            time.sleep(wait)
    if tolerate:
        # The caller counts and skips. Used by ADD, where the reference keeps going so a
        # partially-extractable store still produces an honest number plus a failure
        # count.
        return None
    raise AssertionError(
        f"{what} failed after {attempts} attempts: status={last[0]} resp={last[1]}"
    )


def run_add_phase(
    client: EverosClient,
    sessions: list[dict],
    conv_index: int,
    owner_id: str,
    batch_size: int,
    *,
    app_id: str,
    project_id: str,
    pbar: _tqdm | None = None,
    sender_id_of: Callable[[int, str], str] | None = None,
) -> dict[str, int]:
    """Send each session to its own everos session_id and flush.

    ``sender_id_of`` is the benchmark's message-ownership rule. Without one every
    message is filed under the queried owner, which is what the single-owner benchmarks
    want.

    A batch or flush that keeps failing is counted and skipped rather than raised,
    matching the reference: an extractor that returns unparseable JSON for some sessions
    should still yield a number, with the failure count reported beside it. Returns
    those counts.
    """
    failed_add_batches = 0
    failed_flush_sessions = 0
    for sess in sessions:
        # The store's session id has to match what the adapter's gold_of() resolves to.
        # Hardcoding locomo_conv{i}_s{n} silently mislabels every other benchmark's
        # sessions -- the ingest succeeds and every gold lookup then misses.
        session_id = (
            sess.get("session_id") or f"locomo_conv{conv_index}_s{sess['session_idx']}"
        )
        api_messages: list[dict] = [
            {
                "sender_id": (
                    sender_id_of(conv_index, msg["speaker"])
                    if sender_id_of is not None
                    else owner_id
                ),
                "sender_name": msg["speaker"],
                "role": "user",
                "timestamp": msg["timestamp_ms"],
                "content": [{"type": "text", "text": msg["text"]}],
            }
            for msg in sess["messages"]
        ]

        batches = [
            api_messages[i : i + batch_size]
            for i in range(0, len(api_messages), batch_size)
        ]
        for idx, batch in enumerate(batches):
            payload = {
                "session_id": session_id,
                "app_id": app_id,
                "project_id": project_id,
                "messages": batch,
            }
            if (
                _post_with_retry(
                    client,
                    "/api/v1/memory/add",
                    payload,
                    what=f"Add (session_id={session_id}, batch {idx + 1})",
                    tolerate=True,
                )
                is None
            ):
                failed_add_batches += 1
                _tqdm.write(
                    f"  [add-skipped] session_id={session_id} batch {idx + 1} -- "
                    f"unparseable extraction or persistent 5xx; continuing"
                )
        if (
            _post_with_retry(
                client,
                "/api/v1/memory/flush",
                {"session_id": session_id, "app_id": app_id, "project_id": project_id},
                what=f"Flush (session_id={session_id})",
                tolerate=True,
            )
            is None
        ):
            failed_flush_sessions += 1
            _tqdm.write(
                f"  [flush-skipped] session_id={session_id} -- continuing to the next"
            )
        if pbar:
            pbar.update(len(sess["messages"]))

    if failed_add_batches or failed_flush_sessions:
        _tqdm.write(
            f"  EXTRACTION FAILURES: {failed_add_batches} add batch(es), "
            f"{failed_flush_sessions} flush session(s) skipped"
        )
    return {
        "failed_add_batches": failed_add_batches,
        "failed_flush_sessions": failed_flush_sessions,
    }


# =============================================================================
# Search phase -- single-owner partition
# =============================================================================


# Five, matching the reference's `_INGEST_MAX_RETRIES` that its search goes through
# (test_locomo.py:807). At three, a search that would have succeeded on the fourth
# attempt becomes [SEARCH_FAILED] and is graded wrong -- a one-directional loss.
_SEARCH_RETRIES = 5


def qa_meta_keys_for(config: BenchmarkConfig) -> tuple[str, ...]:
    """Which QA fields this benchmark's judge reads, if any.

    Both spellings are accepted because the same concept was named twice: the adapters
    declaring `JUDGE_META_KEYS` were silently returning nothing, which emptied every
    field SubtleMemory's relation-aware judge renders.
    """
    ad = adapters.get(config.adapter or "locomo")
    keys = getattr(ad, "QA_META_KEYS", None) or getattr(ad, "JUDGE_META_KEYS", ())
    return tuple(keys)


def _search_one(
    i: int,
    qa: dict,
    *,
    client: EverosClient,
    method: str,
    top_k: int,
    owner_id: str,
    app_id: str,
    project_id: str,
    qa_meta_keys: Sequence[str] = (),
    include_profile: bool = False,
    profile_subject: str | None = None,
) -> SearchResult:
    """Search a single QA question with retry on server errors."""
    question = qa["question"]
    payload: dict = {
        "query": question,
        "method": method,
        "top_k": top_k,
        "user_id": owner_id,
    }
    # Omit the partition keys entirely when unset. Sending "" is NOT the same as leaving
    # them out: an empty string still becomes a filter and matches nothing, which is why
    # a search-only run kept returning 0 episodes in 0.03s even after the values were
    # blanked. Measured on caroline_conv0: 11 episodes with the keys absent, 0 with
    # app_id="default_app"/project_id="default_project", 0 with either set to "".
    if app_id:
        payload["app_id"] = app_id
    if project_id:
        payload["project_id"] = project_id
    # Off by default on the server (dto.py: include_profile = False), so a benchmark
    # that grades persona questions has to ask. Only the adapter declaring it does.
    if include_profile:
        payload["include_profile"] = True
        # A group owner holds one profile per participant, so the question has to say
        # which one. Left unset the server returns every profile under the owner --
        # 38 of them on this benchmark's topic 01, which is the composite-profile
        # failure again, only longer. An owner who is its own subject has a single
        # profile and ignores this.
        if profile_subject:
            payload["profile_subject"] = profile_subject
    resp: dict = {}
    search_time = 0.0
    for attempt in range(_SEARCH_RETRIES):
        t0 = time.perf_counter()
        status, resp = client.post("/api/v1/memory/search", payload)
        search_time = time.perf_counter() - t0

        if status == 200:
            break
        error_detail = resp.get("detail", resp) if isinstance(resp, dict) else resp
        last_err = RuntimeError(
            f"Search failed for question {i}: status={status} detail={error_detail}"
        )
        # Same rule as ingest: retry 5xx and transport errors, give up on 4xx.
        if (0 <= status < 500) or attempt >= _SEARCH_RETRIES - 1:
            # Keep the question with the error recorded. The answer stage renders
            # [SEARCH_FAILED] and the judge grades it WRONG, so it stays in the
            # denominator exactly as the reference has it.
            _tqdm.write(f"  [search-failed] question {i}: {last_err}")
            return SearchResult(
                index=i,
                question_id=str(qa.get("question_id", "")),
                question=question,
                golden_answer=str(qa["answer"]),
                category=qa.get("category"),
                question_date=str(qa.get("question_date", "")),
                evidence=qa.get("evidence", []),
                gold_sessions=list(qa.get("gold_sessions") or []),
                qa_meta={k: qa[k] for k in qa_meta_keys if k in qa},
                episodes=[],
                profiles=[],
                search_time_s=round(search_time, 4),
                method=method,
                search_error=str(error_detail)[:500],
            )
        wait = min(2.0 * (2**attempt), 30.0)
        _tqdm.write(
            f"  [warn] search retry {attempt + 1}/{_SEARCH_RETRIES} "
            f"(question {i}): status={status}, backoff {wait:.0f}s"
        )
        time.sleep(wait)

    data = resp.get("data", {})
    episodes = data.get("episodes", [])
    profiles = data.get("profiles", [])
    return SearchResult(
        index=i,
        question_id=str(qa.get("question_id", "")),
        question=question,
        golden_answer=str(qa["answer"]),
        category=qa.get("category"),
        question_date=str(qa.get("question_date", "")),
        evidence=qa.get("evidence", []),
        gold_sessions=list(qa.get("gold_sessions") or []),
        qa_meta={k: qa[k] for k in qa_meta_keys if k in qa},
        episodes=episodes,
        profiles=profiles,
        search_time_s=round(search_time, 4),
        method=method,
    )


def run_search_phase(
    client: EverosClient,
    qa_list: list[dict],
    owner_id: str,
    method: str,
    top_k: int,
    app_id: str,
    project_id: str,
    conv_dir: Path,
    config: BenchmarkConfig,
    *,
    method_label: str,
    pbar: _tqdm | None = None,
    qa_meta_keys: Sequence[str] = (),
) -> list[SearchResult]:
    """Search for each QA question and write results to JSONL."""
    # Resolved once: both hooks are module attributes, and looking them up per question
    # would re-enter the adapter registry 2400 times for a constant.
    _ad = adapters.get(config.adapter or "locomo")
    # Config wins when it says anything; otherwise the adapter's own declaration.
    _include_profile = (
        config.include_profile
        if config.include_profile is not None
        else getattr(_ad, "INCLUDE_PROFILE", False)
    )
    # A group owner holds one profile per participant; an adapter that knows how to read
    # the asker out of a question says so with this hook. Absent, the owner is its own
    # subject and the server's single profile needs no naming.
    _subject_of = getattr(_ad, "profile_subject_of", None)

    def _worker(_pos: int, item: tuple[int, dict]) -> SearchResult:
        i, qa = item
        return _search_one(
            i,
            qa,
            client=client,
            method=method,
            top_k=top_k,
            owner_id=owner_id,
            app_id=app_id,
            project_id=project_id,
            qa_meta_keys=qa_meta_keys,
            include_profile=_include_profile,
            profile_subject=_subject_of(qa) if _subject_of else None,
        )

    out_path = conv_dir / f"search_{method_label}.jsonl"
    # Resume: questions already in the artifact are skipped, and every new result is
    # appended the moment it lands. Writing the whole list once at the end meant a crash
    # at question 190/199 discarded all 190.
    _done = _done_keys(out_path)
    _todo = [(i, q) for i, q in enumerate(qa_list) if str(i) not in _done]
    if _done:
        print(f"  [resume] search: {len(_done)} done, {len(_todo)} to go", flush=True)
        if pbar is not None:
            pbar.update(len(_done))

    raw = _parallel_map(
        _todo,
        _worker,
        concurrency=config.search_concurrency,
        pbar=pbar,
        on_result=lambda r: _append_jsonl(out_path, r),
    )

    _check_failures(raw)
    # The file, not memory, is the source of truth -- it already holds whatever an
    # earlier attempt completed.
    return _finalize_jsonl(out_path, SearchResult)


# =============================================================================
# Answer phase
# =============================================================================


def _build_context(
    episodes: list[dict],
    profiles: list[dict],
    speaker_a: str,
    speaker_b: str,
    config: BenchmarkConfig,
) -> str:
    """Build context string from search results.

    Matches the benchmark's context format: each episode renders as
    ``{subject}: {episode_text}\\n---`` with double-newline separators.
    Profile memories are intentionally omitted (benchmark doesn't use them).
    """
    _own = getattr(adapters.get(config.adapter or "locomo"), "build_context", None)
    if _own is not None:
        return _own(episodes, profiles)
    episode_lines = [
        f"{ep.get('subject', 'N/A')}: "
        f"{ep.get('episode') or ep.get('summary') or ep.get('content') or 'N/A'}\n---"
        for ep in episodes
    ]
    rendered = _prompt(config, "CONTEXT_TEMPLATE").format(
        speaker_a=speaker_a,
        speaker_b=speaker_b,
        episodes="\n\n".join(episode_lines),
    )
    # `profiles` used to stop here. The server fetched them whenever `include_profile`
    # was set and this function dropped them, so the flag changed nothing on locomo and
    # longmemeval -- and the two "profile on vs off" runs compared a configuration
    # against itself. With no profile fetched this returns `rendered` untouched, so the
    # published prompts stay byte-identical to their reference harnesses.
    return with_profile_block(rendered, profiles)


_BUDGET_STOP = threading.Event()
"""Set once a call reports a spent account, so workers stop asking for more."""


class BudgetExhaustedError(RuntimeError):
    """The account ran out of credit, so nothing after this point can be graded.

    Without this a 402 is just another exception: the answer stage burns its retries,
    records `[ERROR: ...]`, the judge grades it wrong, and every remaining question goes
    the same way -- producing a complete-looking report whose number is the fraction of
    questions asked before the money ran out. The reference stops the run instead
    (reanswer_dec_precise_hybrid.py:120-121).
    """


def _is_budget_error(message: str) -> bool:
    """Whether an API error says the account cannot pay for more calls.

    Ported from reanswer_dec_precise_hybrid.py:92-95. Matching on the message is what
    the reference does; the providers do not agree on a machine-readable code.
    """
    m = (message or "").lower()
    return (
        "402" in m
        or "insufficient" in m
        or ("exceeded" in m and "credit" in m)
        or "payment required" in m
    )


def _extract_final_answer(text: str) -> str:
    """Extract the final answer using a 3-marker priority chain.

    Matches the benchmark's extraction logic:
      1. ``## STEP 7: FINAL ANSWER`` (prompt STEP 7 section header)
      2. ``FINAL ANSWER:`` (colon-suffixed)
      3. ``FINAL ANSWER`` (bare -- leading colon stripped if present)

    Each marker uses ``rsplit`` to take the LAST occurrence (handles marker
    appearing in reasoning prose before the actual answer).
    """
    result = text.strip()
    for marker in ("## STEP 7: FINAL ANSWER", "FINAL ANSWER:", "FINAL ANSWER"):
        if marker in result:
            answer = result.rsplit(marker, 1)[1].strip()
            # Bare "FINAL ANSWER" may have a leading ":" -- strip it
            if marker == "FINAL ANSWER" and answer.startswith(":"):
                answer = answer[1:].strip()
            return answer
    return result


def _answer_one(
    i: int,
    sr: SearchResult,
    *,
    speaker_a: str,
    speaker_b: str,
    llm_client: LLMClientPool,
    llm_model: str,
    config: BenchmarkConfig,
) -> AnswerResult:
    """Generate an answer for a single search result; safe to run in a thread.

    Retries up to config.answer_max_retries times. LLM parameters (temperature,
    max_tokens, timeout) come from config so they can be tuned without touching
    the code.
    """
    _ad_first = adapters.get(config.adapter or "locomo")
    if sr.search_error and not getattr(_ad_first, "ANSWER_ON_SEARCH_ERROR", False):
        # Nothing was retrieved, so there is nothing to answer from. Three of the four
        # references record this marker and grade it WRONG; the one that asks anyway
        # says so with ANSWER_ON_SEARCH_ERROR.
        return AnswerResult(
            question_id=sr.question_id,
            index=sr.index,
            question=sr.question,
            golden_answer=sr.golden_answer,
            category=sr.category,
            qa_meta=sr.qa_meta,
            generated_answer="[SEARCH_FAILED]",
            answer_time_s=0.0,
            answer_attempts=0,
            answer_tokens=0,
            search_error=sr.search_error,
        )

    _ad_early = adapters.get(config.adapter or "locomo")
    _empty_marker = getattr(_ad_early, "EMPTY_RETRIEVAL_MARKER", "")
    if _empty_marker and not sr.episodes:
        # A search that succeeded and returned nothing. Whether that is answerable is
        # the benchmark's call, not this runner's: LoCoMo has no such branch, and
        # SubtleMemory and EverMemBench both render a "(no memories retrieved)" context
        # and answer anyway. LongMemEval's harness returns [NO_CONTEXT] with
        # is_correct=False and never calls the model
        # (reanswer_dec_precise_hybrid.py:173-176) -- which matters because its
        # unanswerable `_abs` questions are graded CORRECT for saying nothing is
        # mentioned, so answering from an empty context would score them right for the
        # wrong reason.
        return AnswerResult(
            question_id=sr.question_id,
            index=sr.index,
            question=sr.question,
            golden_answer=sr.golden_answer,
            category=sr.category,
            qa_meta=sr.qa_meta,
            generated_answer=_empty_marker,
            answer_time_s=0.0,
            answer_attempts=0,
            answer_tokens=0,
        )

    _err_context = getattr(_ad_first, "SEARCH_ERROR_CONTEXT", "")
    if sr.search_error and _err_context:
        # A benchmark that answers a failed search may still render it differently from
        # an empty one, and this one does.
        context = _err_context
    else:
        context = _build_context(sr.episodes, sr.profiles, speaker_a, speaker_b, config)
    # LongMemEval's official run_generation.py prepends the question's own date so that
    # temporal questions ("how many weeks ago", "as of now") anchor to when they were
    # asked rather than to the memories' timestamps. Benchmarks without a question date
    # render an empty line, leaving the prompt byte-identical to the version that omits
    # it.
    _date = getattr(sr, "question_date", "") or ""
    _ad = adapters.get(config.adapter or "locomo")
    # A benchmark may grade several question kinds with different prompts (EverMemBench
    # renders multiple-choice options), so it chooses its own template and extra fields.
    _tpl_name = getattr(_ad, "answer_prompt_of", lambda _m: "ANSWER_PROMPT")(
        sr.qa_meta or {}
    )
    _extra = getattr(_ad, "answer_fields", lambda _m: {})(sr.qa_meta or {})
    # A benchmark may need to look at the context before it can pick a prompt.
    # `answer_prompt_of` cannot: it sees only the question's metadata, so it can route
    # on question kind but not on what the retrieved memories actually say. The case
    # that needs this is conflict handling -- whether two memories answer the same
    # thing incompatibly is a property of the context, and the reference's best
    # SubtleMemory arm (71.75% vs 66.56%) routes on exactly that. The hook is given a
    # `call` so its extra stage is billed and counted here rather than opening its own
    # client, and a failure inside it falls back to the default prompt instead of
    # losing the question.
    _route = getattr(_ad, "answer_route", None)
    _route_stages = 0
    if _route is not None:

        def _route_call(prompt_text: str, max_tokens: int) -> str:
            nonlocal _route_stages
            _route_stages += 1
            r = llm_client.chat.completions.create(
                model=llm_model,
                messages=[{"role": "user", "content": prompt_text}],
                temperature=config.answer_temperature,
                max_tokens=max_tokens,
                timeout=config.answer_timeout,
            )
            _u = getattr(r, "usage", None)
            if _u is not None:
                nonlocal _route_prompt_tokens, _route_completion_tokens
                _route_prompt_tokens += getattr(_u, "prompt_tokens", 0) or 0
                _route_completion_tokens += getattr(_u, "completion_tokens", 0) or 0
            return r.choices[0].message.content or ""

        _route_prompt_tokens = 0
        _route_completion_tokens = 0
        try:
            _tpl_name, _routed_extra = _route(
                context=context,
                question=sr.question,
                qa_meta=sr.qa_meta or {},
                call=_route_call,
            )
            _extra = {**_extra, **_routed_extra}
        except BudgetExhaustedError:
            raise
        except Exception as e:
            _tqdm.write(
                f"  [warn] answer_route failed on q{sr.index}: {e}; "
                f"using default prompt"
            )
            _tpl_name = getattr(_ad, "answer_prompt_of", lambda _m: "ANSWER_PROMPT")(
                sr.qa_meta or {}
            )
    else:
        _route_prompt_tokens = 0
        _route_completion_tokens = 0
    prompt = _prompt(config, _tpl_name).format(
        context=context,
        current_date_line=f"Current Date: {_date}\n" if _date else "",
        question=sr.question,
        **_extra,
    )

    t0 = time.perf_counter()
    raw_answer = ""
    generated_answer = ""
    attempts_used = 0
    total_tokens = 0
    # Per-attempt and cumulative, kept apart: the first pair is the paper's Average
    # Tokens (the context this answer was produced from), the second is what the run
    # paid.
    prompt_tokens = 0
    completion_tokens = 0
    prompt_tokens_total = 0
    completion_tokens_total = 0
    for attempt in range(config.answer_max_retries):
        attempts_used = attempt + 1
        try:
            r = llm_client.chat.completions.create(
                model=llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=config.answer_temperature,
                max_tokens=config.answer_max_tokens,
                timeout=config.answer_timeout,
            )
            raw_answer = r.choices[0].message.content or ""
            _usage = getattr(r, "usage", None)
            if _usage is not None:
                total_tokens += _usage.total_tokens or 0
                prompt_tokens = getattr(_usage, "prompt_tokens", 0) or 0
                completion_tokens = getattr(_usage, "completion_tokens", 0) or 0
                prompt_tokens_total += prompt_tokens
                completion_tokens_total += completion_tokens
        except Exception as e:
            if _is_budget_error(str(e)):
                # Retrying cannot help and every later question would go the same way,
                # producing a complete-looking report over the questions asked before
                # the money ran out.
                raise BudgetExhaustedError(str(e)) from e
            cause = f" <- {e.__cause__}" if e.__cause__ else ""
            if attempt < config.answer_max_retries - 1:
                wait = 1.0 * (2**attempt)
                _tqdm.write(
                    f"  [warn] answer retry {attempt + 1}/{config.answer_max_retries} "
                    f"(question {sr.index}): {e}{cause}, backoff {wait:.0f}s"
                )
                time.sleep(wait)
                continue
            # Keep the question: a marker answer is graded WRONG, which is what the
            # reference does. Raising here would remove it from the denominator.
            _ANSWER_FAILURES.append(f"q{sr.index}: {e}{cause}")
            _tqdm.write(
                f"  [answer-failed] q{sr.index} after {config.answer_max_retries} "
                f"retries, recording marker: {e}{cause}"
            )
            raw_answer = f"[ERROR: {e}]"

        # Only the benchmarks whose answer prompt asks for a FINAL ANSWER section get
        # their replies split on it. EverMemBench and SubtleMemory ask for the bare
        # answer, and splitting those truncates any reply containing the phrase.
        generated_answer = (
            _extract_final_answer(raw_answer)
            if getattr(_ad, "EXTRACT_FINAL_ANSWER", True)
            else raw_answer.strip()
        )
        if generated_answer.strip():
            break
        if attempt < config.answer_max_retries - 1:
            wait = 1.0 * (2**attempt)
            _tqdm.write(
                f"  [warn] answer empty, retry "
                f"{attempt + 1}/{config.answer_max_retries} "
                f"(q{sr.index}), backoff {wait:.0f}s"
            )
            time.sleep(wait)

    if not generated_answer.strip():
        # The reference records a marker and lets the judge grade it WRONG, so the
        # question stays in the denominator.
        _ANSWER_FAILURES.append(f"q{sr.index}: empty after retries")
        generated_answer = "[ANSWER_EMPTY]"

    answer_time = time.perf_counter() - t0
    # Fold the router's stage into the cumulative totals only. `prompt_tokens` stays
    # the answering call's own context, which is the number the reference reports as
    # Average Tokens.
    prompt_tokens_total += _route_prompt_tokens
    completion_tokens_total += _route_completion_tokens
    total_tokens += _route_prompt_tokens + _route_completion_tokens
    return AnswerResult(
        question_id=sr.question_id,
        index=sr.index,
        question=sr.question,
        golden_answer=sr.golden_answer,
        category=sr.category,
        qa_meta=sr.qa_meta,
        # Carried even on the path that answered anyway, or a run degraded by retrieval
        # failures reads as clean -- which is the promise the field's own docstring
        # makes.
        search_error=sr.search_error,
        generated_answer=generated_answer,
        answer_time_s=round(answer_time, 4),
        answer_attempts=attempts_used,
        answer_tokens=total_tokens,
        answer_prompt_tokens=prompt_tokens,
        answer_completion_tokens=completion_tokens,
        answer_prompt_tokens_total=prompt_tokens_total,
        answer_completion_tokens_total=completion_tokens_total,
    )


def _guard_stale_stage_file(path: Path, rows: list, config: BenchmarkConfig) -> None:
    """Refuse to resume from a file written before this benchmark carried its metadata.

    A benchmark whose grading reads `qa_meta` -- EverMemBench splits multiple-choice
    from open-ended on it, SubtleMemory renders its relation fields from it -- grades a
    row with an empty `qa_meta` as though it were the other kind, silently and with no
    error. Resuming onto such a file therefore reproduces the very protocol bug the
    metadata was added to fix, on exactly the questions nobody re-runs.
    """
    keys = qa_meta_keys_for(config)
    if not keys or not rows:
        return
    empty = sum(1 for r in rows if not getattr(r, "qa_meta", None))
    if empty:
        raise SystemExit(
            f"{path} has {empty}/{len(rows)} rows with no qa_meta, so it predates the "
            f"metadata this benchmark grades on ({', '.join(keys)}). Resuming would "
            f"grade those rows by the wrong rule. Delete the run directory and start "
            f"it again, or point --run-name at a new one."
        )


def run_answer_phase(
    search_path: Path,
    speaker_a: str,
    speaker_b: str,
    llm_client: LLMClientPool,
    config: BenchmarkConfig,
    conv_dir: Path,
    *,
    method_label: str,
    pbar: _tqdm | None = None,
) -> list[AnswerResult]:
    """Read search JSONL, generate answers, write answer JSONL."""
    search_results = _read_jsonl(search_path, SearchResult, strict=False)
    _guard_stale_stage_file(search_path, search_results, config)

    def _worker(_pos: int, sr: SearchResult) -> AnswerResult:
        return _answer_one(
            sr.index,
            sr,
            speaker_a=speaker_a,
            speaker_b=speaker_b,
            llm_client=llm_client,
            llm_model=config.answer_model,
            config=config,
        )

    out_path = conv_dir / f"answer_{method_label}.jsonl"
    # Resume + incremental write, same reason as the search stage: a stage that only
    # persists on completion throws away everything it finished when it is interrupted.
    _done = _done_keys(out_path)
    _todo = [r for r in search_results if str(getattr(r, "index", None)) not in _done]
    if _done:
        print(f"  [resume] answer: {len(_done)} done, {len(_todo)} to go", flush=True)
        if pbar is not None:
            pbar.update(len(_done))

    raw = _parallel_map(
        _todo,
        _worker,
        concurrency=config.eval_concurrency,
        pbar=pbar,
        on_result=lambda r: _append_jsonl(out_path, r),
    )

    _check_failures(raw)
    return _finalize_jsonl(out_path, AnswerResult)


# =============================================================================
# Evaluate phase -- LLM-as-Judge
# =============================================================================


# Questions whose judge never returned a usable label. They are graded WRONG (matching
# the reference) and listed in the report, so a run degraded by API failures is not read
# as a clean result.
_JUDGE_FAILURES: list[str] = []
_ANSWER_FAILURES: list[str] = []


def _extract_json(content: str) -> str | None:
    """Robustly extract JSON from LLM response."""
    m = re.search(r"```(?:json)?\s*(\{[^`]*\})\s*```", content, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r'\{[^{}]*"label"\s*:\s*"[^"]*"[^{}]*\}', content)
    if m:
        return m.group(0)
    return content.strip()


def _judge_single(
    llm_client: LLMClientPool,
    llm_model: str,
    question: str,
    golden_answer: str,
    generated_answer: str,
    config: BenchmarkConfig,
    clause: str = "",
    qa_meta: dict | None = None,
) -> tuple[bool | None, int]:
    """Judge a single answer. Returns (is_correct, tokens_used).

    `is_correct` is None when no verdict could be read after every retry. What that
    means for the score is the benchmark's call, not this function's.

    Retries up to config.judge_max_retries times on any error (API failures,
    JSON parse errors, missing label) with exponential backoff. Raises on
    exhaustion — benchmark results are unusable with missing judgments.
    """
    _ad = adapters.get(config.adapter or "locomo")
    _extra = getattr(_ad, "judge_fields", lambda _m, _g: {})(
        qa_meta or {}, generated_answer
    )
    user_prompt = _prompt(config, "JUDGE_USER_PROMPT").format(
        question=question,
        golden_answer=golden_answer,
        generated_answer=generated_answer,
        **_extra,
    )
    # The benchmark's own grading rule for this question, if it has one. Prepended so it
    # takes precedence over the general instructions below it.
    if clause:
        user_prompt = f"{clause}\n\n{user_prompt}"
    last_err: Exception | None = None
    for attempt in range(config.judge_max_retries):
        try:
            r = llm_client.chat.completions.create(
                model=llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": _prompt(config, "JUDGE_SYSTEM_PROMPT"),
                    },
                    {"role": "user", "content": user_prompt},
                ],
                temperature=config.judge_temperature,
                timeout=config.judge_timeout,
            )
            tokens = 0
            if hasattr(r, "usage") and r.usage is not None:
                tokens = r.usage.total_tokens or 0

            content = r.choices[0].message.content or ""
            # A benchmark whose harness parses the verdict its own way supplies that
            # parser; its tolerances are part of the protocol.
            _parse = getattr(_ad, "parse_judge_label", None)
            if _parse is not None:
                _label = _parse(content)
                if _label is None:
                    # The parser could not read a verdict. Ask again, as the reference
                    # does; a benchmark whose protocol says an odd reply IS wrong
                    # returns "WRONG" here instead and never reaches this branch.
                    raise ValueError("judge reply carried no readable verdict")
                return _label == "CORRECT", tokens
            json_str = _extract_json(content)
            if not json_str:
                raise ValueError("Empty JSON from judge response")
            result = json.loads(json_str)
            label = result.get("label", "").strip().upper()
            if label not in ("CORRECT", "WRONG"):
                raise ValueError(f"Unknown judge label: {label!r}")
            return label == "CORRECT", tokens
        except Exception as e:
            if _is_budget_error(str(e)):
                raise BudgetExhaustedError(str(e)) from e
            last_err = e
            cause = f" <- {e.__cause__}" if e.__cause__ else ""
            if attempt < config.judge_max_retries - 1:
                wait = 0.5 * (2**attempt)
                _tqdm.write(
                    f"  [warn] judge retry {attempt + 1}/{config.judge_max_retries}: "
                    f"{e}{cause}, backoff {wait:.1f}s"
                )
                time.sleep(wait)
                continue
    # Grade WRONG rather than raise: the reference keeps the row so the denominator
    # stays the question count. Dropping the question instead removes it from the
    # denominator, which silently RAISES the reported accuracy. Counted for the report
    # so the failure is still visible.
    _JUDGE_FAILURES.append(f"q={question[:80]!r}: {last_err}")
    _tqdm.write(
        f"  [judge-failed] after {config.judge_max_retries} retries: {last_err}"
    )
    return None, 0


def _evaluate_one(
    i: int,
    ar: AnswerResult,
    *,
    llm_client: LLMClientPool,
    llm_model: str,
    judge_runs: int,
    config: BenchmarkConfig,
) -> JudgeResult:
    """Evaluate a single answer result with majority-vote judging."""
    if ar.search_error and not getattr(
        adapters.get(config.adapter or "locomo"), "ANSWER_ON_SEARCH_ERROR", False
    ):
        # Graded WRONG without a judge call, matching the reference.
        return JudgeResult(
            question_id=ar.question_id,
            index=ar.index,
            question=ar.question,
            golden_answer=ar.golden_answer,
            generated_answer=ar.generated_answer,
            category=ar.category,
            qa_meta=ar.qa_meta,
            is_correct=False,
            judgments=[],
            judge_tokens=0,
            search_error=ar.search_error,
        )

    # Both are invariant across judge runs. The deterministic verdict especially: it is
    # a property of the answer, so recomputing it per run would just repeat the same
    # result.
    _ad = adapters.get(config.adapter or "locomo")
    _empty_marker = getattr(_ad, "EMPTY_RETRIEVAL_MARKER", "")
    if _empty_marker and ar.generated_answer == _empty_marker:
        # Same rule as a failed search: graded WRONG with no judge call, and kept in the
        # denominator.
        return JudgeResult(
            question_id=ar.question_id,
            index=ar.index,
            question=ar.question,
            golden_answer=ar.golden_answer,
            generated_answer=ar.generated_answer,
            category=ar.category,
            qa_meta=ar.qa_meta,
            is_correct=False,
            judgments=[],
            judge_tokens=0,
        )
    # No judge selection step: ask the benchmark for this question's grading rule and
    # apply it. A benchmark with no rule returns "" and gets the plain judge.
    _clause = getattr(_ad, "leniency_clause", lambda _q: "")(
        {
            "question_id": ar.question_id,
            "category": ar.category,
            **(ar.qa_meta or {}),
        }
    )
    # Some benchmarks grade an exact match against their reference lists without an LLM
    # call at all; that bypass is part of their official protocol, not an optimisation,
    # so it has to run before the judge.
    _fixed = getattr(_ad, "deterministic_verdict", lambda _m, _g: None)(
        ar.qa_meta or {}, ar.generated_answer
    )

    judgments: list[bool] = []
    total_tokens = 0
    unreadable = False
    for _ in range(judge_runs):
        if _fixed is not None:
            is_correct, tokens = _fixed, 0
        else:
            is_correct, tokens = _judge_single(
                llm_client,
                llm_model,
                ar.question,
                ar.golden_answer,
                ar.generated_answer,
                config=config,
                clause=_clause,
                qa_meta=ar.qa_meta,
            )
        if is_correct is None:
            unreadable = True
            is_correct = False
        judgments.append(is_correct)
        total_tokens += tokens

    # A row the judge never answered is wrong here unless the benchmark says such a row
    # leaves the denominator instead.
    _excluded = unreadable and bool(getattr(_ad, "JUDGE_FAILURE_EXCLUDES_ROW", False))
    correct = sum(judgments) > judge_runs / 2
    return JudgeResult(
        question_id=ar.question_id,
        index=ar.index,
        question=ar.question,
        golden_answer=ar.golden_answer,
        generated_answer=ar.generated_answer,
        category=ar.category,
        qa_meta=ar.qa_meta,
        # Kept on every path, not only the short-circuit one: a benchmark that answers a
        # failed search still has to show the failure in its graded rows.
        search_error=ar.search_error,
        is_correct=correct,
        judge_failed=_excluded,
        judgments=judgments,
        judge_tokens=total_tokens,
    )


def run_evaluate_phase(
    answer_path: Path,
    judge_client: LLMClientPool,
    config: BenchmarkConfig,
    judge_runs: int,
    conv_dir: Path,
    *,
    method_label: str,
    pbar: _tqdm | None = None,
) -> list[JudgeResult]:
    """Read answer JSONL, judge answers, write judge JSONL."""
    answer_results = _read_jsonl(answer_path, AnswerResult, strict=False)
    _guard_stale_stage_file(answer_path, answer_results, config)

    def _worker(_pos: int, ar: AnswerResult) -> JudgeResult:
        return _evaluate_one(
            ar.index,
            ar,
            llm_client=judge_client,
            llm_model=config.judge_model,
            judge_runs=judge_runs,
            config=config,
        )

    out_path = conv_dir / f"judge_{method_label}.jsonl"
    # Resume + incremental write, same reason as the search stage: a stage that only
    # persists on completion throws away everything it finished when it is interrupted.
    _done = _done_keys(out_path)
    _todo = [r for r in answer_results if str(getattr(r, "index", None)) not in _done]
    if _done:
        print(f"  [resume] judge: {len(_done)} done, {len(_todo)} to go", flush=True)
        if pbar is not None:
            pbar.update(len(_done))

    raw = _parallel_map(
        _todo,
        _worker,
        concurrency=config.eval_concurrency,
        pbar=pbar,
        on_result=lambda r: _append_jsonl(out_path, r),
    )

    _check_failures(raw)
    return _finalize_jsonl(out_path, JudgeResult)


# =============================================================================
# Reporting
# =============================================================================


def _pct(n: int, total: int) -> str:
    if total == 0:
        return "N/A"
    return f"{n / total * 100:.1f}%"


def _collect_method_summary(
    method: str,
    output_dir: Path,
    conversations: list[int],
    config: BenchmarkConfig,
) -> dict[str, Any] | None:
    """Load per-conv results and compute accuracy/latency stats for one method."""
    all_search: list[SearchResult] = []
    all_answer: list[AnswerResult] = []
    all_judge: list[JudgeResult] = []
    conv_accuracy: dict[int, dict[str, int]] = {}

    for conv_idx in conversations:
        conv_dir = output_dir / f"conv{conv_idx}"
        search_p = conv_dir / f"search_{method}.jsonl"
        answer_p = conv_dir / f"answer_{method}.jsonl"
        judge_p = conv_dir / f"judge_{method}.jsonl"

        if search_p.exists():
            all_search.extend(_read_jsonl(search_p, SearchResult, strict=False))
        if answer_p.exists():
            all_answer.extend(_read_jsonl(answer_p, AnswerResult, strict=False))
        if judge_p.exists():
            conv_judges = _read_jsonl(judge_p, JudgeResult, strict=False)
            all_judge.extend(conv_judges)
            # A row the benchmark's own protocol excludes leaves both halves of the
            # ratio. Only LongMemEval sets this, and the count is reported below,
            # because excluding rows can only ever raise the number.
            scored = [r for r in conv_judges if not r.judge_failed]
            c = sum(1 for r in scored if r.is_correct)
            conv_accuracy[conv_idx] = {"correct": c, "total": len(scored)}
        else:
            print(f"  [report] skip conv{conv_idx}/{method} -- no judge JSONL")

    if not all_judge:
        return None

    # Rows the benchmark's protocol excludes leave both halves of every ratio, and are
    # counted separately so an exclusion can never pass as a clean run.
    excluded = [r for r in all_judge if r.judge_failed]
    scored = [r for r in all_judge if not r.judge_failed]
    total = len(scored)
    correct = sum(1 for r in scored if r.is_correct)
    if excluded:
        print(
            f"  [report] {len(excluded)} question(s) left the denominator: the judge "
            f"returned no readable verdict and this benchmark excludes such rows"
        )

    cat_stats: dict[int | str, dict[str, int]] = {}
    for r in scored:
        cat = r.category
        if cat is None:
            continue
        if cat not in cat_stats:
            cat_stats[cat] = {"correct": 0, "total": 0}
        cat_stats[cat]["total"] += 1
        if r.is_correct:
            cat_stats[cat]["correct"] += 1

    actual_judge_runs = max(
        (len(r.judgments) for r in scored if r.judgments), default=config.judge_runs
    )

    per_run_correct = [0] * actual_judge_runs
    per_run_total = 0
    for r in scored:
        if len(r.judgments) >= actual_judge_runs:
            per_run_total += 1
            for ri in range(actual_judge_runs):
                if r.judgments[ri]:
                    per_run_correct[ri] += 1
    per_run_accuracies = (
        [c / per_run_total for c in per_run_correct] if per_run_total > 0 else []
    )

    mean_accuracy = (
        round(statistics.mean(per_run_accuracies), 4) if per_run_accuracies else 0
    )
    overall_accuracy = correct / total if total else 0
    all_candidates = [*per_run_accuracies, mean_accuracy, overall_accuracy]
    max_accuracy = round(max(all_candidates), 4) if all_candidates else 0

    search_times = [r.search_time_s for r in all_search]
    # Retrieval health, which nothing reported. A LongMemEval smoke run scored 0.0%
    # because `llm_multiround` returned HTTP 200 with an empty result for every question
    # -- and the report showed only an accuracy, so it read exactly like a memory system
    # that answered everything wrong. `empty` is the one that matters: a failed search
    # is at least an error somewhere, an empty successful one leaves no trace at all.
    search_failed = sum(1 for r in all_search if r.search_error)
    search_empty = sum(1 for r in all_search if not r.search_error and not r.episodes)
    answer_times = [r.answer_time_s for r in all_answer]
    answer_tokens = sum(r.answer_tokens for r in all_answer)
    # Average Tokens: the context each answer was produced from, averaged over
    # questions. Retries are excluded on purpose -- retrying does not change how much
    # context retrieval put in front of the model, which is what the metric is about.
    prompt_tokens = [
        r.answer_prompt_tokens for r in all_answer if r.answer_prompt_tokens
    ]
    avg_prompt_tokens = round(statistics.mean(prompt_tokens), 1) if prompt_tokens else 0
    answer_retries = sum(
        r.answer_attempts - 1 for r in all_answer if r.answer_attempts > 1
    )
    # Tokens are what the run spent, so every row counts. Unanimity is a rate over the
    # rows that were scored, or it can exceed 100%.
    judge_tokens = sum(r.judge_tokens for r in all_judge)
    judge_agreements = sum(
        1
        for r in scored
        if r.judgments and all(j == r.judgments[0] for j in r.judgments)
    )

    return {
        "method": method,
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total else 0,
        "mean_accuracy": mean_accuracy,
        "max_accuracy": max_accuracy,
        "per_run_accuracies": [round(a, 4) for a in per_run_accuracies],
        "category_stats": {
            str(k): {"correct": v["correct"], "total": v["total"]}
            for k, v in sorted(cat_stats.items(), key=lambda kv: str(kv[0]))
        },
        "per_conversation": {
            str(k): {"correct": v["correct"], "total": v["total"]}
            for k, v in sorted(conv_accuracy.items())
        },
        # Which conversations the accuracy above is actually over. A conversation whose
        # extraction failed past tolerance never produces a judge file, so its questions
        # are absent from both halves of the ratio -- and a number over a smaller set of
        # conversations must not read as one over the whole list.
        "scored_conversations": sorted(conv_accuracy),
        "requested_conversations": sorted(conversations),
        "judge_excluded": len(excluded),
        "search": {
            "count": len(all_search),
            "failed": search_failed,
            "empty": search_empty,
            "avg_latency_s": round(statistics.mean(search_times), 3)
            if search_times
            else 0,
            "p50_latency_s": round(statistics.median(search_times), 3)
            if search_times
            else 0,
            "max_latency_s": round(max(search_times), 3) if search_times else 0,
        },
        "answer": {
            "count": len(all_answer),
            "avg_latency_s": round(statistics.mean(answer_times), 3)
            if answer_times
            else 0,
            "total_tokens": answer_tokens,
            "avg_prompt_tokens": avg_prompt_tokens,
            "prompt_tokens_total": sum(
                r.answer_prompt_tokens_total for r in all_answer
            ),
            "completion_tokens_total": sum(
                r.answer_completion_tokens_total for r in all_answer
            ),
            "retries": answer_retries,
        },
        "judge": {
            "count": len(scored),
            "excluded": len(excluded),
            "total_tokens": judge_tokens,
            "judge_runs": actual_judge_runs,
            "unanimous": judge_agreements,
            "unanimous_rate": round(judge_agreements / total, 3) if total else 0,
        },
    }


def _write_report_txt(
    txt_path: Path,
    all_summaries: dict[str, dict[str, Any]],
    conversations: list[int],
    config: BenchmarkConfig,
    run_spec: dict[str, Any],
    duration_str: str,
) -> None:
    """Write the human-readable report.txt."""
    generated = datetime.now(UTC)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 64 + "\n")
        f.write("  EverOS LoCoMo Benchmark Report\n")
        f.write("=" * 64 + "\n\n")

        f.write("Run Info\n")
        f.write(f"  Run name:       {run_spec.get('run_name', 'N/A')}\n")
        f.write(f"  Generated:      {generated.isoformat()}\n")
        if duration_str:
            f.write(f"  Duration:       {duration_str}\n")
        f.write(f"  Git hash:       {run_spec.get('git_hash', 'N/A')}\n")
        f.write(f"  EverOS version: {run_spec.get('everos_version', 'N/A')}\n")
        f.write(f"  Python:         {run_spec.get('python_version', 'N/A')}\n")
        f.write(f"  Conversations:  {conversations}\n")
        _missing = sorted(
            set(conversations)
            - {
                int(c)
                for s in all_summaries.values()
                for c in s.get("scored_conversations", [])
            }
        )
        if _missing:
            f.write(
                f"  INCOMPLETE:     {len(_missing)} conversation(s) produced no graded "
                f"rows and are absent from every number below: {_missing}\n"
            )
        _dropped = sum(s.get("judge_excluded", 0) for s in all_summaries.values())
        if _dropped:
            f.write(
                f"  EXCLUDED:       {_dropped} question(s) left the denominator (no "
                f"readable judge verdict)\n"
            )
        f.write(f"  Stages:         {run_spec.get('stages', 'N/A')}\n\n")

        f.write("Configuration\n")
        f.write(f"  Answer model:   {config.answer_model}\n")
        f.write(f"  Judge model:    {config.judge_model}\n")
        f.write(f"  Judge runs:     {config.judge_runs} (config)\n")
        f.write(f"  Top-k:          {config.top_k}\n")
        f.write(f"  Eval owner:     {config.eval_owner}\n\n")

        for method, s in all_summaries.items():
            f.write("-" * 64 + "\n")
            f.write(f"  Method: {method}\n")
            f.write("-" * 64 + "\n\n")

            jr = s["judge"]["judge_runs"]
            # Majority first and alone on the headline: it is the figure every reference
            # reports. best-of-N is diagnostic only and never the result.
            f.write(
                f"  Accuracy:         {_pct(s['correct'], s['total'])} "
                f"({s['correct']}/{s['total']}, majority of {jr})\n"
            )
            f.write(
                f"  Mean accuracy:    {s['mean_accuracy'] * 100:.1f}% "
                f"(avg across {jr} judge runs)\n\n"
            )

            f.write("  Per category:\n")
            # LoCoMo numbers its categories; every other benchmark names them. The int()
            # here crashed the whole report on 'single-session-user' AFTER search,
            # answer and judge had all completed -- the numbers were computed and then
            # thrown away at the last formatting step.
            for cat_key, cs in sorted(
                s["category_stats"].items(), key=lambda kv: str(kv[0])
            ):
                label = _category_label(config, cat_key)
                f.write(
                    f"    {cat_key}. {label:<14s} "
                    f"{_pct(cs['correct'], cs['total']):>6s} "
                    f"({cs['correct']}/{cs['total']})\n"
                )

            f.write("\n  Per conversation:\n")
            for conv_key, cv in sorted(s["per_conversation"].items()):
                f.write(
                    f"    conv{conv_key:<4s} "
                    f"{_pct(cv['correct'], cv['total']):>6s} "
                    f"({cv['correct']}/{cv['total']})\n"
                )

            ss = s["search"]
            f.write(
                f"\n  Search: {ss['count']} queries, "
                f"avg {ss['avg_latency_s']}s, "
                f"p50 {ss['p50_latency_s']}s, "
                f"max {ss['max_latency_s']}s"
            )
            if ss.get("failed") or ss.get("empty"):
                f.write(
                    f"\n  RETRIEVAL:      {ss.get('failed', 0)} failed, "
                    f"{ss.get('empty', 0)} returned nothing -- an accuracy over these "
                    f"is not a measure of the memory's content"
                )
            f.write("\n")

            ans = s["answer"]
            f.write(
                f"  Answer: {ans['count']} questions, "
                f"avg {ans['avg_latency_s']}s, "
                f"{ans['total_tokens']:,} tokens"
            )
            if ans["retries"]:
                f.write(f", {ans['retries']} retries")
            f.write("\n")

            js = s["judge"]
            # The count, not the rate times the count: `unanimous_rate` is rounded to
            # three places, and reconstructing from it prints the wrong number for 756
            # of the (agree, total) pairs up to 60 -- 4 of 7 printed as 3, 1 of 3 as 0.
            unan = _pct(int(js.get("unanimous", 0)), js["count"])
            f.write(
                f"  Judge:  {js['count']} questions"
                f" × {js['judge_runs']} runs, "
                f"{js['total_tokens']:,} tokens, "
                f"unanimous {unan}\n"
            )

            total_tokens = ans["total_tokens"] + js["total_tokens"]
            # Average Tokens first: it is the context-efficiency figure, and the raw
            # total mixes in retries and the judge.
            f.write(f"\n  Average tokens: {ans['avg_prompt_tokens']:,} per question\n")
            f.write(f"  Total tokens:   {total_tokens:,}\n\n")


def _print_terminal_summary(
    all_summaries: dict[str, dict[str, Any]],
    output_dir: Path,
    duration_str: str,
    config: BenchmarkConfig,
) -> None:
    """Print condensed results to the terminal."""
    for method, s in all_summaries.items():
        print(f"\n{'=' * 64}")
        print(f"  Method: {method}")
        jr = s["judge"]["judge_runs"]
        maj = _pct(s["correct"], s["total"])
        print(f"  Accuracy:{maj:>7s} ({s['correct']}/{s['total']}, majority of {jr})")
        if jr > 1:
            print(f"  Mean:    {s['mean_accuracy'] * 100:.1f}% (avg of {jr} runs)")
        # Rows an API failure forced to WRONG. They stay in the denominator, matching
        # the reference, so the count has to be visible or the run reads as clean.
        if _ANSWER_FAILURES or _JUDGE_FAILURES:
            print(
                f"  DEGRADED: {len(_ANSWER_FAILURES)} answer + "
                f"{len(_JUDGE_FAILURES)} judge failure(s) graded WRONG"
            )
        for cat_key, cs in sorted(
            s["category_stats"].items(), key=lambda kv: str(kv[0])
        ):
            label = _category_label(config, cat_key)
            acc = _pct(cs["correct"], cs["total"])
            n, t = cs["correct"], cs["total"]
            print(f"    {cat_key}. {label:<14s} {acc:>6s} ({n}/{t})")
        ss, ans, js = s["search"], s["answer"], s["judge"]
        total_tokens = ans["total_tokens"] + js["total_tokens"]
        print(f"  Search: avg {ss['avg_latency_s']}s, p50 {ss['p50_latency_s']}s")
        if ss.get("failed") or ss.get("empty"):
            print(
                f"  RETRIEVAL: {ss.get('failed', 0)} failed, "
                f"{ss.get('empty', 0)} empty of {ss['count']}"
            )
        a_tok = ans["total_tokens"]
        j_tok = js["total_tokens"]
        print(f"  Avg tokens: {ans['avg_prompt_tokens']:,} per question (context)")
        print(f"  Tokens: {total_tokens:,} (answer {a_tok:,} + judge {j_tok:,})")
    if duration_str:
        print(f"  Duration: {duration_str}")
    print(f"{'=' * 64}")
    print(f"\n  Reports: {output_dir / 'report.json'}, {output_dir / 'report.txt'}")


def aggregate_report(
    output_dir: Path,
    conversations: list[int],
    config: BenchmarkConfig,
) -> None:
    """Aggregate search/answer/judge results and write report files."""
    all_summaries: dict[str, dict[str, Any]] = {}
    for method in config.parsed_methods:
        summary = _collect_method_summary(method, output_dir, conversations, config)
        if summary is not None:
            all_summaries[method] = summary

    report_path = output_dir / "report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)

    run_spec_path = output_dir / "run_spec.json"
    run_spec: dict[str, Any] = {}
    if run_spec_path.exists():
        with open(run_spec_path, encoding="utf-8") as f:
            run_spec = json.load(f)

    duration_str = ""
    started_str = run_spec.get("started_at", "")
    if started_str:
        started = datetime.fromisoformat(started_str)
        duration = datetime.now(UTC) - started
        hours, rem = divmod(int(duration.total_seconds()), 3600)
        mins, secs = divmod(rem, 60)
        duration_str = f"{hours}h {mins}m {secs}s"

    _write_report_txt(
        output_dir / "report.txt",
        all_summaries,
        conversations,
        config,
        run_spec,
        duration_str,
    )
    _print_terminal_summary(all_summaries, output_dir, duration_str, config)


# =============================================================================
# Run spec
# =============================================================================


def _get_everos_version() -> str:
    """Return the installed everos package version, or 'unknown'."""
    try:
        from importlib.metadata import version as _pkg_version

        return _pkg_version("everos")
    except Exception:
        return "unknown"


_PROVENANCE_PACKAGES = (
    # The everalgo family first: these ARE the algorithms. user-memory / agent-memory do
    # extraction (they decide what the store contains), rank does retrieval fusion.
    "everalgo-user-memory",
    "everalgo-agent-memory",
    "everalgo-rank",
    "everalgo-core",
    "everalgo-clustering",
    "everalgo-parser",
    "everalgo-boundary",
    "everos",
    "lancedb",
    "pyarrow",
    "openai",
    "numpy",
)


def _results_root(args: argparse.Namespace, config: BenchmarkConfig) -> str:
    """Where runs are written: CLI, then config, then a repo-relative default.

    A config value still holding ``${BENCH_EVAL_ROOT}`` is treated as unset rather than
    used verbatim. The shipped configs name the root as a variable so nothing in them is
    specific to one machine, and a user who has not set it should get a working default,
    not a directory literally called `${BENCH_EVAL_ROOT}` -- which is what the previous
    version would have created, silently, on the first run out of a fresh clone.
    """
    if args.results_root:
        return str(args.results_root)
    if config.results_root and not unresolved(config.results_root):
        return config.results_root
    if config.results_root and not _results_root._warned:
        # Said once. Three call sites resolve this, and printing per call put the same
        # line in the banner three times, which reads like three different problems.
        _results_root._warned = True
        print(
            f"results  BENCH_EVAL_ROOT is unset, so {config.results_root} cannot be "
            f"resolved; writing to benchmarks/results instead"
        )
    return "benchmarks/results"


_results_root._warned = False  # type: ignore[attr-defined]


def _serving_from_env(config: BenchmarkConfig) -> list[ServingSpec]:
    """Snapshot which endpoint served each role, read from the environment.

    Only what this process can actually observe: the answer / judge endpoints it calls
    itself, and the EverOS-side providers it was told about via ``EVEROS_*``. The
    retrieval server's own launch argv is not visible from here -- that is why the
    record keeps ``endpoint`` and ``model`` for remote roles and leaves ``launch_argv``
    empty rather than guessing.
    """
    roles: list[ServingSpec] = []
    roles.append(
        ServingSpec(
            role="answer",
            model=config.answer_model,
            endpoint=os.environ.get("ANSWER_BASE_URL", ""),
            extra={
                "temperature": config.answer_temperature,
                "max_tokens": config.answer_max_tokens,
            },
        )
    )
    roles.append(
        ServingSpec(
            role="judge",
            model=config.judge_model,
            endpoint=os.environ.get("JUDGE_BASE_URL", ""),
            extra={"runs": config.judge_runs, "temperature": config.judge_temperature},
        )
    )
    # "backbone+decider", not "decider": EverOS has a single [llm] setting, so the same
    # model extracted every memory in the store AND made every retrieval decision.
    # Labelling it "decider" alone hid the extraction half, and which backbone built a
    # store is not recoverable from the store later. Config first, environment only as a
    # fallback. Reading the environment alone recorded whatever `benchmarks/.env`
    # happened to hold, while the servers were started with the config's value -- so a
    # run that really used gpt-4.1-mini was filed as deepseek. The point of this record
    # is that someone else can reproduce the number from it.
    for role, prefix, cfg_model, cfg_url in (
        ("backbone", "EVEROS_LLM__", config.backbone_model, config.backbone_base_url),
        ("decider", "EVEROS_DECIDER__", config.decider_model, config.decider_base_url),
    ):
        model = cfg_model or os.environ.get(f"{prefix}MODEL", "")
        endpoint = cfg_url or os.environ.get(f"{prefix}BASE_URL", "")
        if not model and not endpoint:
            continue
        extra = {
            k[len(prefix) :].lower(): v
            for k, v in os.environ.items()
            if k.startswith(prefix)
            and not k.endswith("API_KEY")
            and k not in (f"{prefix}MODEL", f"{prefix}BASE_URL")
        }
        for k in ("EVEROS_LLMMR_DECIDER_FULL_TEXT", "EVEROS_LLMMR_MAX_ROUNDS"):
            v = config.retrieval_env.get(k) or os.environ.get(k)
            if v is not None:
                extra[k.lower()] = v
        roles.append(
            ServingSpec(role=role, model=model, endpoint=endpoint, extra=extra)
        )

    for role, prefix in (
        ("embedding", "EVEROS_EMBEDDING__"),
        ("reranker", "EVEROS_RERANK__"),
    ):
        model = os.environ.get(f"{prefix}MODEL", "")
        endpoint = os.environ.get(f"{prefix}BASE_URL", "")
        if not model and not endpoint:
            continue
        extra = {
            k[len(prefix) :].lower(): v
            for k, v in os.environ.items()
            if k.startswith(prefix)
            and not k.endswith("API_KEY")
            and k not in (f"{prefix}MODEL", f"{prefix}BASE_URL")
        }
        # These two decide whether a served decider behaves like the one in production;
        # omitting them from a launch script silently changes what was measured.
        for k in ("EVEROS_LLMMR_DECIDER_FULL_TEXT", "EVEROS_LLMMR_MAX_ROUNDS"):
            if role == "decider" and k in os.environ:
                extra[k.lower()] = os.environ[k]
        roles.append(
            ServingSpec(
                role=role,
                model=model,
                endpoint=endpoint,
                local="127.0.0.1" in endpoint or "localhost" in endpoint,
                extra=extra,
            )
        )
    return roles


def _decider_extra(config: BenchmarkConfig) -> dict[str, Any]:
    """``EVEROS_DECIDER__EXTRA`` as request kwargs, from the config or the environment.

    The launch scripts export it and the config's ``retrieval_env`` also carries it, and
    the servers read whichever is set -- so the probe has to look in both, or it vouches
    for a request nobody makes.
    """
    raw = config.retrieval_env.get("EVEROS_DECIDER__EXTRA") or os.environ.get(
        "EVEROS_DECIDER__EXTRA", ""
    )
    if not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Worth saying out loud: the servers ignore it too, so the run would quietly
        # become a thinking-on run.
        print(
            f"  WARNING: EVEROS_DECIDER__EXTRA is not valid JSON, ignoring: {raw[:60]}"
        )
        return {}
    return parsed if isinstance(parsed, dict) else {}


class DeciderUnreachableError(RuntimeError):
    """The configured decider does not answer, so a multi-round run cannot be graded.

    Raised instead of letting the run proceed because the degradation is invisible from
    the outside. When every decider call fails, `llm_multiround` falls back to a fixed
    top-3 core (llm_multiround.py `_DECIDER_FALLBACK_CORE`), stops after one round, and
    the harness records a complete, plausible number that describes the fallback
    rather than the model under test -- and several runs can measure the same degraded
    path before anyone thinks to read a trace.
    """


def _assert_decider_answers(config: BenchmarkConfig) -> None:
    """One real call to the configured decider, asserting it returns content.

    Checked by content and not by `/v1/models` or `/health`: a gateway answers both for
    models it will refuse to serve, and the failure this exists to catch is exactly that
    -- a valid endpoint plus a model name it does not host. A reachability check would
    have passed while every completion 404'd.
    """
    if not (config.decider_model and config.decider_base_url):
        return  # no separate decider: the backbone decides, and its own probe covers it
    key = config.decider_api_key or "EMPTY"
    client = openai.OpenAI(
        api_key=key, base_url=config.decider_base_url, timeout=60.0, max_retries=0
    )
    where = f"{config.decider_model} @ {config.decider_base_url}"
    # The same EVEROS_DECIDER__EXTRA the servers are launched with. Without it the probe
    # is not the request the decider makes: on a qwen checkpoint served with
    # --reasoning-parser, a probe missing enable_thinking=false spends its whole budget
    # in reasoning_content and returns empty content -- so the probe fails a decider
    # that would have worked. A check that is not the real call is worse than none.
    extra = _decider_extra(config)
    try:
        resp = client.chat.completions.create(
            model=config.decider_model,
            messages=[{"role": "user", "content": "Reply with the single word: ready"}],
            # No max_tokens, because the decider does not send one either
            # (`RoundDecider.__call__` passes messages and nothing else). A cap here is
            # what made this probe reject a working setup: a thinking model spends a
            # small budget entirely on reasoning tokens and returns empty content, so
            # the probe failed while the real call -- uncapped, bounded only by the
            # timeout -- succeeds. Every difference between this request and the real
            # one is a way for the check to be wrong about what it vouches for.
            temperature=0.0,
            **extra,
        )
    except Exception as err:
        raise DeciderUnreachableError(f"decider {where} did not answer: {err}") from err
    msg = resp.choices[0].message if resp.choices else None
    content = (getattr(msg, "content", None) or "") if msg else ""
    if not content.strip():
        # Separate "said nothing" from "thought instead of answering": the second has a
        # specific fix, and the decider's own parser treats both as a failed attempt.
        thought = getattr(msg, "reasoning_content", None) if msg else None
        why = (
            "answered in reasoning_content only -- thinking is still on"
            if thought
            else "returned an empty completion"
        )
        raise DeciderUnreachableError(
            f"decider {where} produced no usable content: {why}. Set "
            'EVEROS_DECIDER__EXTRA=\'{"extra_body": {"chat_template_kwargs": '
            '{"enable_thinking": false}}}\''
        )
    print(f"  Decider LLM: {where} -> {content.strip()[:40]!r}")


def _collect_packages() -> dict[str, str]:
    """Installed versions of the packages that can move the numbers.

    Recorded per run because ``everos_version`` alone is ambiguous: the shipped stores
    were extracted with everalgo-user-memory 0.3.1 while this repository pins 0.4.0, and
    those two produce different stores from the same input.
    """
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _pkg_version

    out: dict[str, str] = {}
    for name in _PROVENANCE_PACKAGES:
        try:
            out[name] = _pkg_version(name)
        except PackageNotFoundError:
            continue
    return out


def _write_run_spec(
    output_dir: Path,
    run_name: str,
    config: BenchmarkConfig,
    conversations: list[int],
    stages: list[str],
    benchmark: str = "locomo",
    store_root: str = "",
    serving: list[ServingSpec] | None = None,
) -> None:
    """Write reproducibility snapshot to run_spec.json."""
    serving = list(serving or [])
    git_hash = "unknown"
    try:  # noqa: SIM105
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    spec = RunSpec(
        run_name=run_name,
        config=config.model_dump(),
        conversations=conversations,
        stages=stages,
        git_hash=git_hash,
        python_version=platform.python_version(),
        everos_version=_get_everos_version(),
        started_at=datetime.now(UTC).isoformat(),
        benchmark=benchmark,
        store_root=store_root,
        packages=_collect_packages(),
        serving=serving,
    )
    (output_dir / "run_spec.json").write_text(
        spec.model_dump_json(indent=2), encoding="utf-8"
    )


# =============================================================================
# Per-conversation orchestrator
# =============================================================================


def _assert_owner_in_store(
    root: str, owner_id: str, app_id: str, project_id: str, conv_index: int
) -> None:
    """Fail if the store serving this conversation holds nothing for its owner.

    A search against a store that does not contain the owner returns an empty list with
    HTTP 200. Every question then scores wrong, the run finishes, and `exit 0` says it
    worked -- there is no error anywhere to notice.

    That is easy to walk into whenever pre-built stores are reused. Passing ONE
    `--everos-root` replicates it across every server (`_roots * len(_urls)`), so a
    benchmark whose conversations live in separate shards reads all of them from the
    first shard. Measured on EverMemBench: conv0 retrieved 10/10 and conv1 retrieved
    0/10, and the run reported 40% instead of failing.

    Checked against the markdown, not by searching: the markdown is the source of truth,
    an empty search result is exactly the symptom being diagnosed, and this runs once
    per conversation rather than once per question.
    """
    base = Path(root).expanduser() / f"{app_id}_app" / f"{project_id}_project" / "users"
    if not base.is_dir():
        # Some layouts partition differently, and ADD has not run yet on a fresh store.
        # Absence of the directory is not evidence of a missing owner.
        return
    owner_dir = base / owner_id
    if owner_dir.is_dir() and next(owner_dir.rglob("*.md"), None) is not None:
        return
    present = sorted(d.name for d in base.iterdir() if d.is_dir())[:6]
    raise SystemExit(
        f"conv{conv_index}: store {root} holds no markdown for owner {owner_id!r}.\n"
        f"  It holds: {', '.join(present) or '(nothing)'}"
        f"{' ...' if len(present) == 6 else ''}\n"
        f"  Searching it returns an empty list for every question, and the run then\n"
        f"  finishes with exit 0 reporting a score built from nothing. If this\n"
        f"  benchmark's conversations live in separate shards, pass one\n"
        f"  --everos-root per shard."
    )


def run_conversation(
    conv_index: int,
    *,
    args: argparse.Namespace,
    config: BenchmarkConfig,
    stages: list[str],
    answer_client: LLMClientPool,
    judge_client: LLMClientPool,
    data_path: str,
    position: int | None = None,
    output_dir: Path,
) -> _tqdm:
    """Run the full pipeline for a single conversation."""
    conv_dir = output_dir / f"conv{conv_index}"
    # ADD is the one stage that cannot be replayed cheaply -- it re-extracts every
    # session through the LLM -- and it was also the one stage with no resume: any
    # interruption re-ingested from scratch AND wiped the conversation's
    # search/answer/judge output on the way in. A marker written only after the cascade
    # queue has actually drained makes ADD skippable, so a restart resumes at the first
    # conversation that never finished.
    add_marker = conv_dir / "add.done"
    add_needed = "add" in stages and not add_marker.exists()
    # A marker is only trustworthy if the store it describes still holds that owner's
    # memories. Results directories get copied and reused, so a marker can arrive from a
    # different run and point at a store that was rebuilt, moved, or never written --
    # exactly what happened once: markers from an earlier run were mirrored in, ADD was
    # skipped for all four conversations, and an empty store scored 14.7% with 85% of
    # searches returning nothing.
    if not add_needed and "add" in stages:
        # Derive from args, not the enclosing `_roots`: this runs inside a nested
        # function reachable before that local is assigned, which raised
        # UnboundLocalError and killed both conversations after ADD had finished.
        _rl = (
            args.everos_root
            if isinstance(args.everos_root, list)
            else [args.everos_root]
        )
        _root = Path(_rl[conv_index % len(_rl)]).expanduser()
        _owner = json.loads(add_marker.read_text()).get("owner_id", "")
        # A validation check must never abort the run. This block twice referenced
        # locals that are only assigned further down the enclosing function, and each
        # time the run died AFTER ADD had completed. `_owner_dir` was also dead code --
        # the glob below answers "does this owner have memories" without the partition
        # dirs.
        try:
            _stale = bool(_owner) and not any(_root.glob(f"**/{_owner}/**/*.md"))
        except Exception as _e:
            print(
                f"  [resume] conv{conv_index}: cannot validate marker ({_e}); "
                f"trusting it",
                flush=True,
            )
            _stale = False
        if _stale:
            print(
                f"  [resume] conv{conv_index}: marker present but store has "
                f"no memories for {_owner} — re-running ADD",
                flush=True,
            )
            add_marker.unlink()
            add_needed = True
    if add_needed and conv_dir.exists():
        shutil.rmtree(conv_dir)
    conv_dir.mkdir(parents=True, exist_ok=True)

    sessions, qa_list, spk_a, spk_b, adapter_owner = load_conversation_via_adapter(
        config.adapter or "locomo", data_path, conv_index
    )
    _adapter = adapters.get(config.adapter or "locomo")

    judge_runs = config.judge_runs
    # `--smoke` is 50 messages; `--max-messages` is the same trim at a caller-chosen
    # size. A cost probe needs the SECOND one: 50 messages produces only INIT profile
    # calls, and INIT is the cheap path -- the dominant term (re-sending a matured
    # profile on every UPDATE) does not exist until the profile has grown, so a smoke
    # run measures a cost the real run never pays.
    msg_cap = args.max_messages or (50 if args.smoke else 0)
    if msg_cap:
        trimmed: list[dict] = []
        msg_count = 0
        for sess in sessions:
            if msg_count >= msg_cap:
                break
            remaining = msg_cap - msg_count
            if len(sess["messages"]) <= remaining:
                trimmed.append(sess)
                msg_count += len(sess["messages"])
            else:
                trimmed.append({**sess, "messages": sess["messages"][:remaining]})
                msg_count += remaining
        sessions = trimmed
    if args.smoke:
        qa_list = _stratified_sample(qa_list, n=10)
        judge_runs = 1

    # app_id / project_id partition the store, so they must match how the store was
    # BUILT, not how this run is named. Hardcoding app_id="locomo_benchmark" and
    # project_id=<run_name> silently returns zero episodes against any store built under
    # different keys -- e.g. every shared store here lives in
    # default_app/default_project, so a search-only run scored 0/152 with the answer
    # model politely reporting that the memories contain nothing. Overridable from the
    # CLI; defaults keep a fresh add-then-search run behaving exactly as before. Empty
    # means "do not filter". The store's LanceDB rows carry no app/project labels --
    # those are directory levels, not row columns -- so sending them adds a WHERE clause
    # that matches nothing: hybrid returned 11 episodes with the fields omitted and 0
    # with app_id=default_app/project_id=default_project, and a full 10-conversation run
    # scored 0.58% because every search came back empty in 0.03s.  The ADD stage still
    # needs them (that is where the directories come from), so the historical defaults
    # are kept for a run that builds its own store. Config first, CLI flag only as an
    # override. A reproduction command that needs extra flags the documented user
    # command does not have is exactly where mistakes hide -- passing the directory name
    # `default_app` instead of the stored `default` returned zero episodes silently and
    # cost three full runs.
    app_id = args.app_id or config.app_id
    project_id = args.project_id or config.project_id
    _bench = config.adapter or "locomo"
    if _bench == "locomo":
        _speaker = spk_a if config.eval_owner == "speaker_a" else spk_b
        owner_id = f"{_speaker.lower()}_conv{conv_index}"
    elif adapter_owner:
        # From the adapter, not from the speaker pair: see
        # load_conversation_via_adapter.
        owner_id = adapter_owner
    else:
        # The adapter already returned the owner; re-wrapping it in LoCoMo's
        # "{speaker}_conv{i}" pattern produced owners like `assistant_conv0` where gold
        # expects `longmemeval_0` / `persona_0`. The ingest succeeds and every later
        # gold lookup misses -- silently, with a 0% score at the end and nothing in the
        # logs.
        owner_id = spk_a
    # c % N, the same mapping the historical fleet used, so a lane that dies can be
    # resumed with the identical assignment by re-running just its conversations.
    _urls = args.base_url if isinstance(args.base_url, list) else [args.base_url]
    _roots = (
        args.everos_root if isinstance(args.everos_root, list) else [args.everos_root]
    )
    if len(_roots) != 1 and len(_roots) != len(_urls):
        raise SystemExit(
            f"--everos-root takes either 1 path or exactly one per --base-url "
            f"(got {len(_roots)} roots for {len(_urls)} servers)"
        )
    if len(_roots) == 1:
        _roots = _roots * len(_urls)
    if len(_urls) > 1 and "add" in stages and conv_index == 0:
        # ADD writes, and a server takes an exclusive lock on its store's index queue
        # (.index/sqlite/ome.db.lock), so two servers cannot ingest into one root -- the
        # second refuses to start with BlockingIOError. Sharded ingestion therefore
        # means The read-only stages have no such constraint and may share a single
        # root.
        print(
            "  NOTE: sharded ADD needs one --everos-root per --base-url (the index "
            "queue lock is exclusive). Conversation c is both written and read through "
            "server c %% N, so the shards need no combining afterwards.",
            flush=True,
        )
    client = EverosClient(base_url=_urls[conv_index % len(_urls)])

    methods = config.parsed_methods
    label = f"conv{conv_index}"

    total_stages = len(stages)
    stage_num = 0

    pbar = _ColorBarTqdm(
        total=0,
        desc=f"{label:<6s} init",
        unit="it",
        dynamic_ncols=True,
        position=position,
        leave=False,
    )

    def _stage(name: str, total: int, suffix: str = "") -> None:
        nonlocal stage_num
        stage_num += 1
        pbar.reset(total=total)
        tag = f"{name} {suffix}".rstrip()
        pbar.set_description_str(f"{label:<6s} {stage_num}/{total_stages} {tag:<15s}")

    if "add" in stages and not add_needed:
        print(
            f"  [resume] conv{conv_index}: add already complete, skipping", flush=True
        )

    if add_needed:
        add_started = datetime.now(UTC).isoformat()
        _stage("add", sum(len(s["messages"]) for s in sessions), "sending")
        _add_stats = run_add_phase(
            client,
            sessions,
            conv_index,
            owner_id,
            config.batch_size,
            app_id=app_id,
            project_id=project_id,
            pbar=pbar,
            sender_id_of=getattr(_adapter, "sender_id_of", None),
        )
        pbar.reset(total=0)
        pbar.set_description_str(
            f"{label:<6s} {stage_num}/{total_stages} {'add processing':<15s}"
        )
        _ome = _wait_ready(
            _roots[conv_index % len(_roots)],
            conv_index,
            project_id,
            config.cascade_timeout,
            app_id=app_id,
            owner_id=owner_id,
            session_ids=[str(s["session_id"]) for s in sessions if s.get("session_id")],
            since=add_started,
            pbar=pbar,
        )
        # Only now: _wait_ready returning is what proves the extraction actually landed.
        # Writing the marker right after run_add_phase would mark a conversation done
        # while its cascade queue was still draining.
        add_marker.write_text(
            json.dumps(
                {
                    "owner_id": owner_id,
                    "sessions": len(sessions),
                    # Extraction losses are recorded, not swallowed: a conversation can
                    # be "complete" and still be missing the facts from a few malformed-
                    # JSON responses, and that has to be auditable after the fact.
                    "ome_total": _ome.total,
                    "ome_failed": _ome.failed,
                    # Batches and flushes the extractor could not produce parseable
                    # output for. The published extractor comparison reports these
                    # beside accuracy.
                    **_add_stats,
                    "session_ids": [
                        str(s["session_id"]) for s in sessions if s.get("session_id")
                    ],
                    "finished_at": datetime.now(UTC).isoformat(),
                }
            ),
            encoding="utf-8",
        )

    for method in methods:
        if "search" in stages:
            # Before the first question: a store without this owner scores every
            # question wrong and still exits 0.
            _assert_owner_in_store(
                str(_roots[conv_index % len(_roots)]),
                owner_id,
                app_id,
                project_id,
                conv_index,
            )
            _stage("search", len(qa_list))
            run_search_phase(
                client,
                qa_list,
                owner_id,
                method,
                config.top_k,
                app_id,
                project_id,
                conv_dir,
                config,
                method_label=method,
                pbar=pbar,
                qa_meta_keys=qa_meta_keys_for(config),
            )

        if "answer" in stages:
            search_path = conv_dir / f"search_{method}.jsonl"
            if not search_path.exists():
                raise FileNotFoundError(
                    f"Missing {search_path} -- run 'search' stage first"
                )
            _stage("answer", len(_read_jsonl(search_path, SearchResult, strict=False)))
            run_answer_phase(
                search_path,
                spk_a,
                spk_b,
                answer_client,
                config,
                conv_dir,
                method_label=method,
                pbar=pbar,
            )

        if "judge" in stages:
            answer_path = conv_dir / f"answer_{method}.jsonl"
            if not answer_path.exists():
                raise FileNotFoundError(
                    f"Missing {answer_path} -- run 'answer' stage first"
                )
            _stage("judge", len(_read_jsonl(answer_path, AnswerResult)))
            run_evaluate_phase(
                answer_path,
                judge_client,
                config,
                judge_runs,
                conv_dir,
                method_label=method,
                pbar=pbar,
            )

    pbar.bar_format = "{desc}"
    pbar.set_description_str(f"{label:<6s} {total_stages}/{total_stages} done")
    pbar.refresh()
    return pbar


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> tuple[argparse.Namespace, BenchmarkConfig]:
    """Parse CLI args and load benchmark config."""
    p = argparse.ArgumentParser(
        prog="benchmarks/run.py",
        description="EverOS LoCoMo Benchmark Runner",
    )
    p.add_argument(
        "--run-name",
        default="",
        help="Directory the results are written to. Defaults to the benchmark name; "
        "give "
        "it only to keep several runs of the same benchmark apart.",
    )
    p.add_argument(
        "--conv",
        type=str,
        nargs="+",
        default=None,
        help="Conversations to run: indices, ranges like 0-499, or `all` for every "
        "conversation the config declares. Defaults to `all`. LongMemEval has 500 "
        "units, "
        "so spelling them out is not a usable interface.",
    )
    p.add_argument(
        "--stages",
        nargs="+",
        default=["add", "search", "answer", "judge"],
        choices=["add", "search", "answer", "judge"],
        help="Pipeline stages to run (default: all)",
    )
    p.add_argument(
        "--config",
        default="",
        help="Override the config file. Normally unnecessary: the benchmark name "
        "selects "
        "benchmarks/configs/<name>.toml.",
    )
    p.add_argument(
        "--base-url",
        default="http://localhost:8000",
        nargs="+",
        help="EverOS server address(es). Several may be given: conversation c is "
        "served "
        ""
        "by address c %% len(addresses). Retrieval is read-only, so N servers can "
        "share "
        ""
        ""
        "one store -- only ADD needs a root per server, because it holds an exclusive "
        "lock on the index queue (.index/sqlite/ome.db.lock). Splitting the read "
        "stages "
        ""
        ""
        "across servers is the cheapest speedup available: the client already runs "
        "conversations_concurrency x search_concurrency requests at once, and a single "
        "server serialises them behind its own LLM calls.",
    )
    p.add_argument(
        "--everos-root",
        default=[],
        nargs="+",
        help="EverOS --root path(s). Give one per --base-url when sharding ADD across "
        "servers: ADD holds an exclusive lock on <root>/.index/sqlite/ome.db, so two "
        "servers cannot ingest into the same root. Conversation c is served by "
        "base_url[c %% n] and polled at everos_root[c %% n], keeping the two aligned. "
        "SEARCH is read-only and happily shares one root.",
    )
    p.add_argument(
        "--data-path",
        default="",
        help="Override the config's data_path. Datasets are not vendored here, so each "
        "benchmark's TOML carries its own location.",
    )
    p.add_argument(
        "benchmark_name",
        nargs="?",
        default="",
        help="Which benchmark to run: locomo | longmemeval | evermembench | "
        "subtlememory. "
        "Names benchmarks/configs/<name>.toml, and is the run name unless --run-name "
        "says "
        "otherwise. One identifier instead of a name plus a config path.",
    )
    p.add_argument(
        "--decider-model",
        default="",
        help="Model for the multi-round retrieval decider, when it differs from the "
        "backbone. Every published run used a different model for each.",
    )
    p.add_argument(
        "--decider-base-url",
        default="",
        help="Endpoint serving --decider-model. Required whenever the decider is not "
        "on the endpoint the config names: without it the new model name is sent to "
        "the OLD endpoint, which 404s on every call. That is silent -- the decider "
        "falls back to a fixed top-3 core and the run still reports a number -- and it "
        "That is silent: the decider falls back to a fixed top-ranked core and the "
        "run still reports a complete result.",
    )
    p.add_argument(
        "--decider-api-key",
        default="",
        help="Key for --decider-base-url. A self-hosted vLLM accepts any non-empty "
        "value; pass one so the client does not fall back to the config's key for a "
        "different provider.",
    )
    p.add_argument(
        "--backbone-base-url",
        default="",
        help="Endpoint for the backbone. Point it at a local vLLM/SGLang server "
        "(http://127.0.0.1:8000/v1) to run extraction and the retrieval decider on-box "
        "instead of a hosted API.",
    )
    p.add_argument(
        "--backbone-model",
        default="",
        help="Model EverOS runs when this harness starts the servers (--servers). One "
        "setting covers both jobs -- memory extraction during ADD and the decider "
        "inside "
        "multi-round retrieval -- because EverOS has a single [llm] section. Ignored "
        "when "
        "attaching to servers with --base-url: their backbone was fixed at launch.",
    )
    p.add_argument(
        "--methods",
        default="",
        help="Override the config's search methods (comma-separated): keyword | vector "
        "| "
        "hybrid | agentic | llm_multiround. Use hybrid when the variable under test is "
        "the "
        "extraction backbone rather than the decider -- hybrid makes no LLM calls, so "
        "the "
        "store is the only thing being measured.",
    )
    p.add_argument(
        "--answer-model",
        default="",
        help="Override the config's answer_model. Include the vendor prefix when the "
        "endpoint is OpenRouter (`openai/gpt-4.1-mini`, not `gpt-4.1-mini`) -- the "
        "prefix "
        "is what selects the serving provider.",
    )
    p.add_argument(
        "--judge-model",
        default="",
        help="Override the config's judge_model. Same prefix rule as --answer-model.",
    )
    p.add_argument(
        "--first-port",
        type=int,
        default=0,
        help="Lowest port to try when --servers starts a fleet; occupied ports are "
        "skipped, so concurrent runs do not collide.",
    )
    p.add_argument(
        "--servers",
        type=int,
        default=-1,
        help="Start this many EverOS servers for the run and shut them down "
        "afterwards. "
        ""
        "Defaults to the benchmark's own `servers` setting; 0 disables the fleet. "
        "Each gets its own root derived from --everos-root (`<root>_s0`, `<root>_s1`, "
        "...) "
        "because ADD takes an exclusive lock on a store's index queue. This is the "
        "normal "
        "way to run: --base-url is only for attaching to servers you manage yourself.",
    )
    p.add_argument(
        "--results-root",
        default="",
        help="Where per-run result directories are written. Worth pointing off the "
        "repository when the filesystem holding it is near capacity: a full disk "
        "surfaces as [Errno 28] on every answer write, which is retried and then "
        "recorded as a failed question rather than as an outage.",
    )
    p.add_argument(
        "--benchmark",
        default="",
        help="Which dataset adapter drives the run. The adapter owns the four things "
        "that "
        "differ between benchmarks: how questions load, what owner they live under, "
        "how "
        ""
        ""
        "gold translates into store session ids, and which judge grades them.",
    )
    p.add_argument(
        "--app-id",
        default="",
        help="Store partition to query. Must match how the store was BUILT; the shared "
        "stores use 'default_app'. Empty keeps the historical 'locomo_benchmark'.",
    )
    p.add_argument(
        "--project-id",
        default="",
        help="Project partition to query. Empty defaults to --run-name, which is only "
        "correct when this run also built the store.",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke: 2 convs, 50 msgs, 10 QA, judge_runs=1",
    )
    p.add_argument(
        "--max-messages",
        type=int,
        default=0,
        help=(
            "Ingest at most N messages per conversation (0 = no cap). Same trim as "
            "--smoke but at a chosen size, and without touching the QA sample: a cost "
            "probe needs enough messages for profiles to MATURE, because the expensive "
            "path is UPDATE re-sending a grown profile, not INIT."
        ),
    )
    p.add_argument(
        "--reuse-store",
        action="store_true",
        help="Allow ADD into a store that already holds memories without this run's "
        "add.done markers. Off by default because re-adding duplicates memories.",
    )
    p.add_argument(
        "--list-servers",
        action="store_true",
        help="List running EverOS servers with their store, port, version and owning "
        "run.py, then exit. Use before attaching to one or debugging a fleet that will "
        "not start.",
    )
    p.add_argument(
        "--reap-servers",
        action="store_true",
        help="Stop every EverOS server no run.py owns, then exit. Servers with an "
        "owner "
        ""
        "are left alone.",
    )

    args = p.parse_args()
    if args.list_servers or args.reap_servers:
        (_reap_servers if args.reap_servers else _print_servers)()
        raise SystemExit(0)
    # The available set comes from the config directory, so it cannot drift from what is
    # actually runnable the way a hardcoded list can.
    _cfg_dir = Path(__file__).parent / "configs"
    _available = sorted(f.stem for f in _cfg_dir.glob("*.toml") if f.stem != "default")
    if not args.benchmark_name and not args.config:
        p.error(
            "give a benchmark name, e.g. `run.py locomo` "
            f"(available: {', '.join(_available)})"
        )
    if (
        args.benchmark_name
        and not args.config
        and args.benchmark_name not in _available
    ):
        # Without this a typo reached `BenchmarkConfig.from_toml` and came back as a raw
        # FileNotFoundError naming a .toml path -- the missing-name case listed the
        # benchmarks, the wrong-name case did not, and the wrong-name case is the one an
        # operator hits.
        p.error(
            f"unknown benchmark {args.benchmark_name!r} "
            f"(available: {', '.join(_available)}); pass --config to point at a file "
            f"outside benchmarks/configs/"
        )
    args.run_name = args.run_name or args.benchmark_name

    config = BenchmarkConfig.from_toml(args.config or args.benchmark_name)

    # The store lives with the results that came out of it, unless told otherwise.
    #
    # It used to be mandatory to say where: launch.sh passed
    # `--everos-root <somewhere outside the results tree>` "for easy cleanup", and every
    # consequence of that was bad. The store sat on a different filesystem from its
    # results (a near-full filesystem surfaces as ENOSPC on every write and takes the
    # run with it), the only record of which store produced which numbers was a path
    # buried in run_spec.json, and when one was deleted the results it explained became
    # unreproducible with nothing in the directory saying so.
    #
    # So: ADD defaults the root to <results>/<run>/store, and a later stage-only
    # run given no root looks there first. An explicit --everos-root still wins,
    # which is what lets a run reuse a canonical store it did not build (the
    # a run that scores an existing store does).
    _res_root = Path(_results_root(args, config)) / args.run_name
    _default_store = _res_root / "store"
    if not args.everos_root:
        if "add" in args.stages:
            args.everos_root = [str(_default_store)]
            print(f"store    {_default_store} (default: lives with the results)")
        elif _default_store.exists():
            args.everos_root = [str(_default_store)]
            print(f"store    reusing {_default_store}")
        else:
            p.error(
                f"--everos-root is required for a run without ADD when "
                f"{_default_store} does not exist. That directory is where ADD would "
                f"have put the store; pass the path of the store to reuse."
            )

    # Mirrors everos.memory.search.dto.SearchMethod. llm_multiround was missing, so the
    # harness rejected the very method the multi-round results were produced with -- a
    # whitelist in the runner, not a limitation of the server: /api/v1 and /api/v2 mount
    # the same search route and it accepts the full SearchRequest including `method`.
    supported = ("keyword", "vector", "hybrid", "agentic", "llm_multiround")
    bad = [m for m in config.parsed_methods if m not in supported]
    if bad:
        p.error(f"unsupported method(s) in config.toml: {bad}; supported: {supported}")

    # Resolve --conv against the config: `all` and bare ranges both need the declared
    # conversation count, which only the config knows.
    args.conv = _parse_conv_spec(args.conv, config.conversations)
    if args.smoke:
        args.conv = [0, 1]

    # Model overrides exist so a re-run under a different backbone does not need a whole
    # duplicate TOML: the four `config.repro_*.toml` files differed from their originals
    # in exactly one field, `answer_model`.
    _over = {}
    if args.answer_model:
        _over["answer_model"] = args.answer_model
    if args.judge_model:
        _over["judge_model"] = args.judge_model
    # Folded into the config rather than threaded to the fleet as a separate argument,
    # because two sources of truth make a run unreadable after the fact: the fleet would
    # take the CLI value while `run_spec.json` kept recording the config's, so every run
    # is filed under a decider it may not have used.
    if args.decider_model:
        _over["decider_model"] = args.decider_model
    if args.decider_base_url:
        _over["decider_base_url"] = args.decider_base_url
    if args.decider_api_key:
        _over["decider_api_key"] = args.decider_api_key
    if _over:
        config = config.model_copy(update=_over)

    # An unresolved `${...}` in the decider means the variable was never set, so treat
    # it as "no separate decider" -- the decider then runs [llm]'s model, which is the
    # single-model setup that works with no extra configuration.
    #
    # Both are cleared together even if only one is unresolved. A model name pointed at
    # the wrong endpoint 404s on every call and the retrieval loop falls back to a fixed
    # core while still reporting a complete result -- plausible numbers for an
    # experiment that never ran.
    if unresolved(config.decider_model) or unresolved(config.decider_base_url):
        _named = config.decider_model if not unresolved(config.decider_model) else ""
        print(
            "decider  no endpoint for "
            + (f"{_named}" if _named else "the configured decider")
            + " (set BENCH_DECIDER_BASE_URL); falling back to the [llm] model.\n"
            "         ⚠ This is NOT the published configuration -- the published "
            "numbers were\n         produced with a separate decider, so this run "
            "is not comparable to them."
        )
        config = config.model_copy(
            update={"decider_model": "", "decider_base_url": "", "decider_api_key": ""}
        )

    if not args.data_path:
        args.data_path = config.data_path
    if not getattr(args, "benchmark", ""):
        args.benchmark = config.adapter
    return args, config


# =============================================================================
# Main
# =============================================================================


def _iter_servers() -> list[tuple[int, str, str]]:
    """Every running EverOS server as (pid, port, store root), read from /proc."""
    out: list[tuple[int, str, str]] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmd = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode()
            if (
                "everos server start" not in cmd
                and "everos-embench server start" not in cmd
            ):
                continue
            port = ""
            parts = cmd.split()
            if "--port" in parts:
                port = parts[parts.index("--port") + 1]
            root = ""
            env = (entry / "environ").read_bytes().decode(errors="replace")
            for line in env.split("\0"):
                if line.startswith("EVEROS_ROOT="):
                    root = line.split("=", 1)[1]
                    break
        except (OSError, ValueError, IndexError, UnicodeDecodeError):
            continue
        out.append((int(entry.name), port, root))
    return sorted(out, key=lambda r: r[1])


def _server_owner(pid: int) -> int | None:
    """The run.py that spawned this server, found by walking the parent chain.

    Ownership is an ancestry question, not a path-matching one: a fleet started without
    an explicit --everos-root has no store path on any command line, so matching paths
    would read every such live server as an orphan.
    """
    seen = 0
    while pid > 1 and seen < 40:
        seen += 1
        try:
            cmd = (
                Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
            )
            if "benchmarks/run.py" in cmd:
                return pid
            status = Path(f"/proc/{pid}/status").read_text()
        except OSError:
            return None
        nxt = [ln for ln in status.splitlines() if ln.startswith("PPid:")]
        if not nxt:
            return None
        pid = int(nxt[0].split()[1])
    return None


def _server_version(port: str) -> str:
    if not port:
        return "?"
    # Loopback must not go through an HTTP proxy: no_proxy matches by host, so a CIDR
    # entry does not exempt it, and the request would be routed out and time out.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(f"http://127.0.0.1:{port}/health", timeout=3) as resp:
            return str(json.loads(resp.read()).get("version") or "?")
    except Exception:
        return "none"


def _print_servers() -> None:
    """One place to answer 'who serves this store, on what port, at what version'.

    A store admits exactly ONE server -- the index-queue lock is exclusive and taken at
    startup whatever stages will run -- so a second run against a store still held by a
    previous fleet cannot start.
    """
    print(f"  {'PID':<9}{'PORT':<7}{'VERSION':<10}{'OWNER':<9}{'STORE'}")
    for pid, port, root in _iter_servers():
        owner = _server_owner(pid)
        print(
            f"  {pid:<9}{port or '?':<7}{_server_version(port):<10}"
            f"{(str(owner) if owner else 'ORPHAN'):<9}{root or '?'}"
        )


def _reap_servers() -> None:
    """Stop servers no run.py owns. Anything still owned is left strictly alone."""
    for pid, port, root in _iter_servers():
        owner = _server_owner(pid)
        if owner is not None:
            print(f"  keep   :{port}  owned by run.py {owner}")
            continue
        try:
            os.kill(pid, signal.SIGTERM)
            print(f"  reaped :{port}  {root}")
        except OSError as e:
            print(f"  failed :{port}  {e}")


def _guard_stale_store(args, config: BenchmarkConfig) -> None:
    """Refuse to ADD into a store that already holds memories this run did not write.

    ADD is not idempotent: re-sending a conversation into a store that already contains
    its memories duplicates them. A results directory with no add.done markers next to a
    non-empty store is exactly that case -- the store outlived the results it was built
    with -- and resuming silently would produce a number nobody can interpret.
    """
    base = Path(
        args.everos_root[0] if isinstance(args.everos_root, list) else args.everos_root
    ).expanduser()
    # `_ServerFleet` uses the base root itself when it starts exactly one server and
    # only appends `_s<i>` for a fleet, so deriving the suffix unconditionally made this
    # guard inspect directories that do not exist -- it passed on a populated store
    # whenever `--servers 1`. And `--servers` is clamped to the conversation count, so
    # resuming a single conversation both landed on a different root and switched the
    # guard off.
    roots = (
        [base]
        if args.servers == 1
        else [base.parent / f"{base.name}_s{i}" for i in range(args.servers)]
        if args.servers > 0
        else [base]
    )
    populated = [
        r for r in roots if r.is_dir() and next(r.rglob("*.md"), None) is not None
    ]
    if not populated:
        return
    results = Path(_results_root(args, config)) / args.run_name
    if next(results.glob("conv*/add.done"), None) is not None:
        return  # markers present: an ordinary resume, and ADD will skip what is done
    if getattr(args, "reuse_store", False):
        print(f"  [store] reusing {len(populated)} populated store(s) as asked")
        return
    print(
        "\nRefusing to start: these stores already hold memories, but "
        f"{results} has no add.done marker, so ADD would duplicate them."
    )
    for r in populated:
        print(f"  {r}")
    print("\nDelete the stores for a clean run, or pass --reuse-store to add on top.")
    raise SystemExit(2)


def main() -> None:
    """Entry point: orchestrate all conversations."""
    args, config = parse_args()
    _explicit_base_url = any(
        a == "--base-url" or a.startswith("--base-url=") for a in sys.argv[1:]
    )

    # Before the fleet: the servers inherit this process's environment, and their
    # backbone/embedding credentials live in that file.
    load_dotenv(Path(__file__).parent / ".env")

    if args.methods:
        config = config.model_copy(update={"methods": args.methods})
    if args.servers < 0:
        # An explicit --base-url means the caller is pointing at servers they manage,
        # with stores that already hold data. Starting a fleet then would silently
        # replace those addresses with fresh, EMPTY stores: the run completes, every
        # search returns zero episodes, and the score is whatever the answer model
        # guesses -- measured at 21.55% on SubtleMemory before this was caught. Never
        # auto-start when told where to go.
        args.servers = 0 if _explicit_base_url else config.servers
    # A fleet larger than the conversation count leaves servers with no work and still
    # pays their startup; cap it rather than silently wasting them.
    if args.servers > len(args.conv):
        print(
            f"  --servers {args.servers} exceeds {len(args.conv)} conversation(s); "
            f"using {len(args.conv)}",
            flush=True,
        )
        args.servers = len(args.conv)

    # A fleet of N derives its roots as <root>_s0 .. <root>_s{N-1} because the index
    # queue lock is exclusive; N == 1 uses the given root as-is. So pointing --everos-
    # root
    # at a single prebuilt store while asking for more than one server reads that
    # store's
    # NAME and then serves N siblings of it -- directories that are empty, or worse,
    # already present and empty (`everos_store_s0` sits next to `everos_store` in every
    # shared store here). The run then completes with zero episodes on every question
    # and
    # exit 0: measured on a smoke run against a store holding 711 episodes, 20/20
    # questions came back empty in 1.15s each.
    #
    # A populated base root with unpopulated shards is unambiguous -- the caller means
    # "read this store" -- so serve it with one server and say so.
    if args.servers > 1 and args.everos_root:
        _base = Path(
            args.everos_root[0]
            if isinstance(args.everos_root, list)
            else args.everos_root
        ).expanduser()
        _shards = [_base.parent / f"{_base.name}_s{i}" for i in range(args.servers)]
        _base_has_md = _base.is_dir() and next(_base.rglob("*.md"), None) is not None
        _shards_have_md = any(
            r.is_dir() and next(r.rglob("*.md"), None) is not None for r in _shards
        )
        if _base_has_md and not _shards_have_md:
            print(
                f"  --servers {args.servers} would serve {_base.name}_s0.."
                f"{_base.name}_s{args.servers - 1}, which hold no markdown, while "
                f"{_base.name} does; using 1 server on it",
                flush=True,
            )
            args.servers = 1

    # Derive both from the run name when unset, so several benchmarks can run at once
    # from bare commands. A shared default root made four concurrent lanes ingest into
    # ONE store, and a shared first port made them race for the same free port while
    # probing.
    if not args.everos_root:
        args.everos_root = [str(Path("~/.everos_bench").expanduser() / args.run_name)]
    if not args.first_port:
        # Stable per run name rather than random: a relaunch reuses its own ports
        # instead of colliding with a lane that is still up.
        args.first_port = (
            9400
            + (int(hashlib.sha256(args.run_name.encode()).hexdigest(), 16) % 40) * 10
        )

    if "add" in args.stages:
        _guard_stale_store(args, config)

    fleet: _ServerFleet | None = None
    if args.servers > 0:
        base = Path(
            args.everos_root[0]
            if isinstance(args.everos_root, list)
            else args.everos_root
        ).expanduser()
        backbone = args.backbone_model or config.backbone_model
        fleet = _ServerFleet(
            args.servers,
            base,
            first_port=args.first_port,
            backbone_model=backbone,
            backbone_base_url=args.backbone_base_url or config.backbone_base_url,
            backbone_api_key=config.backbone_api_key,
            decider_model=config.decider_model,
            decider_base_url=config.decider_base_url,
            decider_api_key=config.decider_api_key,
            extra_env=server_env_for(args.stages, config.retrieval_env),
            # None disables both dumps inside the fleet; see BenchmarkConfig.trace.
            trace_dir=(
                Path(_results_root(args, config)) / args.run_name / "traces"
                if config.trace
                else None
            ),
        )
        fleet.start()
        args.base_url = fleet.urls
        args.everos_root = fleet.roots
    try:
        # `finally` alone only covers a clean exit or an exception. A SIGTERM -- which
        # is how a lane gets stopped from the outside -- kills the process without
        # unwinding, so the fleet survived its owner and kept the exclusive index-queue
        # lock. The next run against that store then failed with "server(s) did not
        # become ready", having spent 180s waiting for a port it could never bind. Turn
        # the signal into an exception so the same teardown path runs.
        def _term(signum, _frame):
            raise KeyboardInterrupt(f"signal {signum}")

        for _sig in (signal.SIGTERM, signal.SIGINT):
            # Not the main thread, or a platform that will not allow it.
            with contextlib.suppress(ValueError, OSError):
                signal.signal(_sig, _term)
        _main_inner(args, config)
    finally:
        if fleet is not None:
            fleet.stop()


def _main_inner(args, config) -> None:

    load_dotenv(Path(__file__).parent / ".env")

    answer_api_keys = _split_keys(os.getenv("ANSWER_API_KEY", ""))
    answer_base_url = os.getenv("ANSWER_BASE_URL", "https://api.openai.com/v1")
    judge_api_keys = _split_keys(os.getenv("JUDGE_API_KEY", ""))
    judge_base_url = os.getenv("JUDGE_BASE_URL", "https://api.openai.com/v1")

    if not answer_api_keys:
        print("ERROR: ANSWER_API_KEY not set in benchmarks/.env")
        sys.exit(1)
    if not judge_api_keys:
        print("ERROR: JUDGE_API_KEY not set in benchmarks/.env")
        sys.exit(1)

    answer_client = LLMClientPool(
        answer_api_keys,
        base_url=answer_base_url,
        providers=config.providers,
        timeout=60,
        max_retries=1,
    )
    if answer_base_url == judge_base_url and answer_api_keys == judge_api_keys:
        judge_client = answer_client
    else:
        judge_client = LLMClientPool(
            judge_api_keys,
            base_url=judge_base_url,
            providers=config.providers,
            timeout=60,
            max_retries=1,
        )

    output_dir = Path(_results_root(args, config)) / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_run_spec(
        output_dir,
        args.run_name,
        config,
        args.conv,
        args.stages,
        benchmark=getattr(args, "benchmark", "locomo"),
        store_root=str(getattr(args, "everos_root", "") or ""),
        serving=_serving_from_env(config),
    )

    print(
        f"  Answer LLM: {config.answer_model} @ {answer_base_url}"
        f" ({answer_client.key_count} keys)"
    )
    print(
        f"  Judge  LLM: {config.judge_model} @ {judge_base_url}"
        f" ({judge_client.key_count} keys)"
    )
    # Before any question is asked: a decider that cannot answer degrades silently into
    # a fixed top-3 core, and the run still produces a full report.
    if "search" in args.stages:
        _assert_decider_answers(config)
    print(f"  Search mode: {config.methods}")
    print(f"  Conversations: {args.conv}")
    print(f"  Stages: {args.stages}")
    print(f"  Output: {output_dir}")

    conv_positions = {ci: pos for pos, ci in enumerate(args.conv)}

    conv_errors: dict[int, str] = {}
    conv_pbars: dict[int, _tqdm] = {}

    def _run_conv(conv_index: int, stages: list[str] | None = None) -> bool:
        try:
            pbar = run_conversation(
                conv_index,
                args=args,
                config=config,
                stages=list(stages if stages is not None else args.stages),
                answer_client=answer_client,
                judge_client=judge_client,
                data_path=args.data_path,
                output_dir=output_dir,
                position=conv_positions[conv_index],
            )
            conv_pbars[conv_index] = pbar
            return True
        except BudgetExhaustedError:
            # Not a conversation failure: nothing after this can be graded, so it has to
            # leave this handler rather than be recorded as one conv among many.
            # Recorded once so the pool's other workers stop starting new questions.
            _BUDGET_STOP.set()
            raise
        except Exception as e:
            import traceback

            tb = traceback.format_exc()
            conv_dir = output_dir / f"conv{conv_index}"
            conv_dir.mkdir(parents=True, exist_ok=True)
            (conv_dir / "error.log").write_text(tb, encoding="utf-8")
            conv_errors[conv_index] = str(e)
            _tqdm.write(f"  conv{conv_index} FAILED: {e}")
            return False

    # Ingest a server's conversations, freeze that server's projection, then read
    # them. Walking all four stages per conversation put a 56-72s retrieval (a
    # full-text decider) next to cascade's projection maintenance, and prune's 60s
    # retention window reclaimed files the in-flight search still held:
    # `LanceError(IO): Object at location .../_indices/<uuid>/tokens.lance not
    # found` -- one HTTP 500 per search, an empty context handed to the answer
    # model, a scored zero. Measured: 225 of 493 questions on the first full LoCoMo
    # run.
    #
    # The grouping is PER SERVER, not global. Conversation c is served by
    # base_url[c % n] against everos_root[c % n], and ADD needs one root per server
    # (it holds an exclusive lock on that root's OME queue), so server j's prune
    # physically cannot touch server i's store. Grouping globally would serialise
    # the whole fleet's ingest against its retrieval for no reason; per server, a
    # server that has finished its own share starts reading while its neighbours
    # are still ingesting, and with one conversation per server -- LoCoMo's shape --
    # nothing is serialised at all.
    _urls = args.base_url if isinstance(args.base_url, list) else [args.base_url]
    _n_srv = max(len(_urls), 1)
    _groups: dict[int, list[int]] = {}
    for _ci in args.conv:
        _groups.setdefault(_ci % _n_srv, []).append(_ci)

    _read_stages = [s for s in args.stages if s != "add"]
    _does_add = "add" in args.stages
    # Split the configured width across the fleet instead of handing every server
    # the whole budget, which would multiply it by the server count.
    _width = max(1, config.conversations_concurrency // _n_srv)

    budget_stopped = False
    results: dict[int, bool] = {}
    _res_lock = threading.Lock()

    def _run_pass(convs: list[int], stages: list[str]) -> None:
        nonlocal budget_stopped
        with ThreadPoolExecutor(max_workers=_width) as pool:
            futures = {pool.submit(_run_conv, ci, stages): ci for ci in convs}
            for f, ci in futures.items():
                try:
                    ok = f.result()
                    # A conversation that failed its ingest must not be read: its
                    # store is short whatever the failure dropped, and searching it
                    # would score the gap as a retrieval miss.
                    with _res_lock:
                        results[ci] = ok and results.get(ci, True)
                except BudgetExhaustedError as e:
                    # Every remaining conversation would fail the same way on its
                    # first answer call, and a report scored from the questions the
                    # account could afford is worse than no report. Cancel what has
                    # not started.
                    with _res_lock:
                        budget_stopped = True
                        results[ci] = False
                        conv_errors[ci] = f"budget exhausted: {e}"
                    for other in futures:
                        other.cancel()
                    _tqdm.write(f"  BUDGET EXHAUSTED on conv{ci}: {e}")

    def _run_server_group(srv_idx: int, convs: list[int]) -> None:
        """One server's whole pipeline: ingest its share, freeze it, then read it."""
        if _does_add:
            _run_pass(convs, ["add"])
            if not budget_stopped:
                _quiesce_servers([_urls[srv_idx]])
        if _read_stages and not budget_stopped:
            _run_pass(convs, _read_stages)

    with ThreadPoolExecutor(max_workers=_n_srv) as fleet_pool:
        group_futures = [
            fleet_pool.submit(_run_server_group, si, cs) for si, cs in _groups.items()
        ]
        for gf in group_futures:
            try:
                gf.result()
            except BudgetExhaustedError:
                budget_stopped = True

    for pbar in conv_pbars.values():
        pbar.leave = True
        pbar.close()

    failed = [ci for ci, ok in results.items() if not ok]
    if failed:
        print(f"\n{len(failed)} conversation(s) failed: {failed}")
        for ci in failed:
            print(f"  see {output_dir}/conv{ci}/error.log")

    if budget_stopped:
        # No report. The rows that exist are the questions the account could pay for,
        # and a number over those is indistinguishable from a number over all of them.
        print(
            "\nSTOPPED: the account ran out of credit. No report was written -- the "
            "rows on disk cover only the questions paid for, and scoring those would "
            "read as a complete run. Top up and resume: the finished stages are kept."
        )
        sys.exit(2)

    # Aggregate
    if "judge" in args.stages:
        aggregate_report(output_dir, args.conv, config)

    if failed:
        # Exit non-zero. A run whose every answer call 404'd still printed "Done" and
        # left a report scored from whatever partial conversations survived --
        # indistinguishable from a clean run unless somebody opened error.log. A wrong
        # number that looks finished is worse than no number at all.
        print(
            f"\nFAILED: {len(failed)}/{len(results)} conversation(s) did not complete."
        )
        sys.exit(1)

    print(f"\nDone. Results: {output_dir}")


if __name__ == "__main__":
    main()

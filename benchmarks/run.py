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
import json
import os
import platform
import re
import shutil
import sqlite3
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import openai
import requests
from dotenv import load_dotenv

# Ensure the repo root is on sys.path when run as a script
# (python benchmarks/run.py) rather than as a module (python -m benchmarks.run).
_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from tqdm import tqdm as _tqdm  # noqa: E402

from benchmarks.config import (  # noqa: E402
    AnswerResult,
    BenchmarkConfig,
    JudgeResult,
    RunSpec,
    SearchResult,
)

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

ANSWER_PROMPT = """
You are an intelligent memory assistant tasked with retrieving accurate information from episodic memories.

# CONTEXT:
You have access to episodic memories from conversations between two speakers. These memories contain
timestamped information that may be relevant to answering the question.

# INSTRUCTIONS:
Your goal is to synthesize information from all relevant memories to provide a comprehensive and accurate answer.
You MUST follow a structured Chain-of-Thought process to ensure no details are missed.
Actively look for connections between people, places, and events to build a complete picture. Synthesize information from different memories to answer the user's question.
It is CRITICAL that you move beyond simple fact extraction and perform logical inference. When the evidence strongly suggests a connection, you must state that connection. Do not dismiss reasonable inferences as "speculation." Your task is to provide the most complete answer supported by the available evidence.

# CRITICAL REQUIREMENTS:
1. NEVER omit specific names - use "Amy's colleague Rob" not "a colleague"
2. ALWAYS include exact numbers, amounts, prices, percentages, dates, times
3. PRESERVE frequencies exactly - "every Tuesday and Thursday" not "twice a week"
4. MAINTAIN all proper nouns and entities as they appear
5. EXPLICITLY state confidence levels for inferences (High/Medium/Low)

# RESPONSE FORMAT (You MUST follow this structure):

## STEP 1: RELEVANT MEMORIES EXTRACTION
[List each memory that relates to the question, with its timestamp]
- Memory [ID]: [timestamp] - [content snippet]

## STEP 2: KEY INFORMATION IDENTIFICATION
[Extract ALL specific details from the memories]
- Names mentioned: [list all person names, place names, company names]
- Numbers/Quantities: [list all amounts, prices, percentages]
- Dates/Times: [list all temporal information]
- Frequencies: [list any recurring patterns]
- Other entities: [list brands, products, etc.]

## STEP 3: CROSS-MEMORY LINKING & INFERENCE
[Identify entities that appear in multiple memories and link related information. Make reasonable inferences when entities are strongly connected.]
- Shared entities: [list people, places, events mentioned across different memories]
- Connections found: [e.g., "Memory 1 mentions A moved from hometown -> Memory 2 mentions A's hometown is LA -> Therefore A moved from LA"]
- Inferences: [Connect the dots. Label confidence: (Confidence: High/Medium/Low)]

## STEP 4: TIME REFERENCE CALCULATION
[If applicable, convert relative time references using the timestamps]
- Original reference: [e.g., "last year" from May 2022]
- Calculation: [Show logic]
- Actual time: [e.g., "2021"]

## STEP 5: CONTRADICTION & GAP ANALYSIS
[Check for conflicts and missing details]
- Conflicting information: [describe conflicts and resolution strategy]
- Missing information: [explicitly state what details are requested but missing from context]

## STEP 6: DETAIL VERIFICATION CHECKLIST
- [ ] All person names included?
- [ ] All locations included?
- [ ] All numbers exact?
- [ ] All frequencies specific?
- [ ] All dates/times precise?
- [ ] All proper nouns preserved?

## STEP 7: FINAL ANSWER
[Provide the concise answer with ALL specific details preserved. Do not include the internal checklist in this section, just the final synthesized answer.]

---

{context}

Question: {question}

Now, follow the Chain-of-Thought process above to answer the question:
"""

JUDGE_SYSTEM_PROMPT = "You are an expert grader that determines if answers to questions match a gold standard answer"

JUDGE_USER_PROMPT = """Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
    (1) a question (posed by one user to another user),
    (2) a 'gold' (ground truth) answer,
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it's time for the real question:
Question: {question}
Gold answer: {golden_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label".
"""

# =============================================================================
# Category labels
# =============================================================================

CATEGORY_NAMES: dict[int, str] = {
    1: "single-hop",
    2: "multi-hop",
    3: "open-domain",
    4: "temporal",
}

# =============================================================================
# Minimal HTTP client for everos (single-tenant, no auth headers)
# =============================================================================


class EverosClient:
    """Minimal HTTP client for everos's /api/v1/memory/* endpoints."""

    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 600):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def post(self, path: str, data: dict[str, Any]) -> tuple[int, dict]:
        full_url = f"{self.base_url}{path}"
        resp = requests.post(
            full_url,
            json=data,
            headers={"Content-Type": "application/json"},
            timeout=(10, self.timeout),
        )
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
    ``extra_body={"provider": {"only": [...]}}`` on every request so the
    downstream provider is fixed. OpenRouter otherwise routes freely across
    providers (OpenAI, Azure, Fireworks, ...), which on a 1.5k-question batch
    eventually lands on a region-restricted Azure instance and 403s every
    later request. The allow-list defaults to ``["openai"]`` and can be
    overridden via the ``OPENROUTER_PROVIDER_ONLY`` env var (comma-separated,
    e.g. ``openai,fireworks``).
    """

    def __init__(self, api_keys: list[str], base_url: str, **kwargs: Any):
        if not api_keys:
            raise ValueError("LLMClientPool: at least one API key required")
        self._clients = [
            openai.OpenAI(api_key=k, base_url=base_url, **kwargs) for k in api_keys
        ]
        self._idx = 0
        self._lock = threading.Lock()
        self.key_count = len(self._clients)
        self.chat = _PoolChat(self)
        self._provider_constraint = self._resolve_provider_constraint(base_url)

    @staticmethod
    def _resolve_provider_constraint(base_url: str) -> dict[str, Any] | None:
        """Resolve the OpenRouter ``provider`` extra-body block (or None)."""
        if "openrouter" not in (base_url or "").lower():
            return None
        raw = os.getenv("OPENROUTER_PROVIDER_ONLY", "openai").strip()
        if not raw or raw.lower() == "any":
            return None
        only = [p.strip() for p in raw.split(",") if p.strip()]
        return {"only": only, "allow_fallbacks": False}

    def _next_client(self) -> openai.OpenAI:
        with self._lock:
            c = self._clients[self._idx]
            self._idx = (self._idx + 1) % len(self._clients)
            return c

    def _create_with_failover(self, **kwargs: Any) -> Any:
        if self._provider_constraint is not None:
            extra = dict(kwargs.get("extra_body") or {})
            extra.setdefault("provider", self._provider_constraint)
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
    """Raise if any element in *raw* is an exception from ``_parallel_map``."""
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
) -> list:
    """Run ``worker(i, item)`` over *items* concurrently; preserve input order.

    Updates *pbar* on each completion if provided. Falls back to serial
    execution when *concurrency* <= 1.
    """
    results: list = [None] * len(items)

    if concurrency <= 1:
        for i, item in enumerate(items):
            results[i] = worker(i, item)
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
            except Exception as exc:  # noqa: BLE001
                results[idx] = exc
            if pbar:
                pbar.update(1)

    return results


# =============================================================================
# Wait for cascade + OME drain
# =============================================================================


def _poll_cascade(db_path: Path, conv_pattern: str) -> tuple[int, int]:
    """Return (total, pending) for cascade md_change_state rows."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    total, pending = conn.execute(
        "SELECT COUNT(*), "
        "SUM(CASE WHEN status IN ('pending','processing') THEN 1 ELSE 0 END) "
        "FROM md_change_state WHERE md_path LIKE ?",
        (conv_pattern,),
    ).fetchone()
    conn.close()
    return total, pending


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
    total, pending, failed = 0, 0, 0
    for status, count in rows:
        total += count
        if status == "running":
            pending += count
        elif status in ("failed", "dead_letter", "crashed"):
            failed += count
    return total, pending, failed


def _wait_ready(
    everos_root: str,
    conv_index: int,
    project_id: str,
    timeout_s: int,
    poll_interval_s: float = 3.0,
    *,
    since: str = "",
    pbar: _tqdm | None = None,
) -> None:
    """Wait until cascade queue AND OME jobs finish for a conversation."""
    root = Path(everos_root).expanduser()
    db_path = root / ".index" / "sqlite" / "system.db"
    ome_db_path = root / ".index" / "sqlite" / "ome.db"
    conv_pattern = f"%/{project_id}/%_conv{conv_index}/%"

    if not db_path.exists():
        raise RuntimeError(
            f"Cascade DB not found at {db_path} — "
            f"is --everos-root ({everos_root}) correct? "
            f"It must match the server's --root."
        )

    ome_filter = (
        "json_extract(event_payload, '$.app_id') = 'locomo_benchmark' "
        "AND json_extract(event_payload, '$.project_id') = ? "
        "AND ("
        "  json_extract(event_payload, '$.owner_id') LIKE ? "
        "  OR json_extract(event_payload, '$.session_id') LIKE ?"
        ")"
    )
    ome_params = (project_id, f"%_conv{conv_index}", f"locomo_conv{conv_index}_%")

    deadline = time.time() + timeout_s
    stable_count = 0
    cascade_pending = 0
    ome_pending = 0

    while time.time() < deadline:
        cascade_total, cascade_pending = _poll_cascade(db_path, conv_pattern)
        ome_total, ome_pending, ome_failed = _poll_ome(
            ome_db_path, since, ome_filter, ome_params
        )

        if ome_failed > 0:
            raise RuntimeError(
                f"{ome_failed} OME task(s) failed for conv {conv_index} "
                "— data is incomplete, aborting"
            )

        if pbar is not None:
            done = (cascade_total - cascade_pending) + (ome_total - ome_pending)
            total = cascade_total + ome_total
            if total > 0:
                pbar.total = total
                pbar.n = done
                pbar.refresh()

        if cascade_pending == 0 and ome_pending == 0:
            stable_count += 1
            if stable_count >= 2:
                if pbar is not None and pbar.total and pbar.total > 0:
                    pbar.n = pbar.total
                    pbar.refresh()
                return
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


def _read_jsonl(path: Path, model_cls: type) -> list:
    """Read a JSONL file into a list of Pydantic model instances."""
    results = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(model_cls.model_validate_json(line))
    return results


# =============================================================================
# Add phase -- one everos session_id per LoCoMo session, flush after each
# =============================================================================


def run_add_phase(
    client: EverosClient,
    sessions: list[dict],
    conv_index: int,
    batch_size: int,
    *,
    app_id: str,
    project_id: str,
    pbar: _tqdm | None = None,
) -> None:
    """Send each LoCoMo session to its own everos session_id and flush."""
    for sess in sessions:
        session_id = f"locomo_conv{conv_index}_s{sess['session_idx']}"
        api_messages: list[dict] = [
            {
                "sender_id": f"{msg['speaker'].lower()}_conv{conv_index}",
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
            status, resp = client.post("/api/v1/memory/add", payload)
            assert status == 200, (
                f"Add (session_id={session_id}, batch {idx + 1}) failed: "
                f"status={status} resp={resp}"
            )
        flush_status, flush_resp = client.post(
            "/api/v1/memory/flush",
            {"session_id": session_id, "app_id": app_id, "project_id": project_id},
        )
        assert flush_status == 200, (
            f"Flush (session_id={session_id}) failed: "
            f"status={flush_status} resp={flush_resp}"
        )
        if pbar:
            pbar.update(len(sess["messages"]))


# =============================================================================
# Search phase -- single-owner partition
# =============================================================================


_SEARCH_RETRIES = 3


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
) -> SearchResult:
    """Search a single QA question with retry on server errors."""
    question = qa["question"]
    payload: dict = {
        "query": question,
        "method": method,
        "top_k": top_k,
        "user_id": owner_id,
        "app_id": app_id,
        "project_id": project_id,
    }
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
        if status < 500 or attempt >= _SEARCH_RETRIES - 1:
            raise last_err
        wait = 2.0 * (2**attempt)
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
        question=question,
        golden_answer=str(qa["answer"]),
        category=qa.get("category"),
        evidence=qa.get("evidence", []),
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
) -> list[SearchResult]:
    """Search for each QA question and write results to JSONL."""

    def _worker(i: int, qa: dict) -> SearchResult:
        return _search_one(
            i,
            qa,
            client=client,
            method=method,
            top_k=top_k,
            owner_id=owner_id,
            app_id=app_id,
            project_id=project_id,
        )

    raw = _parallel_map(
        qa_list,
        _worker,
        concurrency=config.search_concurrency,
        pbar=pbar,
    )

    _check_failures(raw)
    results: list[SearchResult] = raw  # type: ignore[assignment]

    out_path = conv_dir / f"search_{method_label}.jsonl"
    _write_jsonl(out_path, results)
    return results


# =============================================================================
# Answer phase
# =============================================================================


_CONTEXT_TEMPLATE = """Episodes memories for conversation between {speaker_a} and {speaker_b}:

    {episodes}
"""


def _build_context(
    episodes: list[dict], profiles: list[dict], speaker_a: str, speaker_b: str
) -> str:
    """Build context string from search results.

    Matches the benchmark's context format: each episode renders as
    ``{subject}: {episode_text}\\n---`` with double-newline separators.
    Profile memories are intentionally omitted (benchmark doesn't use them).
    """
    episode_lines = [
        f"{ep.get('subject', 'N/A')}: "
        f"{ep.get('episode') or ep.get('summary') or ep.get('content') or 'N/A'}\n---"
        for ep in episodes
    ]
    return _CONTEXT_TEMPLATE.format(
        speaker_a=speaker_a,
        speaker_b=speaker_b,
        episodes="\n\n".join(episode_lines),
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
    context = _build_context(sr.episodes, sr.profiles, speaker_a, speaker_b)
    prompt = ANSWER_PROMPT.format(context=context, question=sr.question)

    t0 = time.perf_counter()
    raw_answer = ""
    generated_answer = ""
    attempts_used = 0
    total_tokens = 0
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
            if hasattr(r, "usage") and r.usage is not None:
                total_tokens += r.usage.total_tokens or 0
        except Exception as e:
            cause = f" <- {e.__cause__}" if e.__cause__ else ""
            if attempt < config.answer_max_retries - 1:
                wait = 1.0 * (2**attempt)
                _tqdm.write(
                    f"  [warn] answer retry {attempt + 1}/{config.answer_max_retries} "
                    f"(question {sr.index}): {e}{cause}, backoff {wait:.0f}s"
                )
                time.sleep(wait)
                continue
            raise RuntimeError(
                f"Answer failed after {config.answer_max_retries} retries "
                f"(question {sr.index}): {e}{cause}"
            ) from e

        generated_answer = _extract_final_answer(raw_answer)
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
        raise RuntimeError(
            f"Answer empty after {config.answer_max_retries} retries "
            f"(question {sr.index})"
        )

    answer_time = time.perf_counter() - t0
    return AnswerResult(
        index=sr.index,
        question=sr.question,
        golden_answer=sr.golden_answer,
        category=sr.category,
        generated_answer=generated_answer,
        answer_time_s=round(answer_time, 4),
        answer_attempts=attempts_used,
        answer_tokens=total_tokens,
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
    search_results = _read_jsonl(search_path, SearchResult)

    def _worker(i: int, sr: SearchResult) -> AnswerResult:
        return _answer_one(
            i,
            sr,
            speaker_a=speaker_a,
            speaker_b=speaker_b,
            llm_client=llm_client,
            llm_model=config.answer_model,
            config=config,
        )

    raw = _parallel_map(
        search_results,
        _worker,
        concurrency=config.eval_concurrency,
        pbar=pbar,
    )

    _check_failures(raw)
    results: list[AnswerResult] = raw  # type: ignore[assignment]

    out_path = conv_dir / f"answer_{method_label}.jsonl"
    _write_jsonl(out_path, results)
    return results


# =============================================================================
# Evaluate phase -- LLM-as-Judge
# =============================================================================


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
) -> tuple[bool, int]:
    """Judge a single answer. Returns (is_correct, tokens_used).

    Retries up to config.judge_max_retries times on any error (API failures,
    JSON parse errors, missing label) with exponential backoff. Raises on
    exhaustion — benchmark results are unusable with missing judgments.
    """
    user_prompt = JUDGE_USER_PROMPT.format(
        question=question,
        golden_answer=golden_answer,
        generated_answer=generated_answer,
    )
    last_err: Exception | None = None
    for attempt in range(config.judge_max_retries):
        try:
            r = llm_client.chat.completions.create(
                model=llm_model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=config.judge_temperature,
                timeout=config.judge_timeout,
            )
            tokens = 0
            if hasattr(r, "usage") and r.usage is not None:
                tokens = r.usage.total_tokens or 0

            content = r.choices[0].message.content or ""
            json_str = _extract_json(content)
            if not json_str:
                raise ValueError("Empty JSON from judge response")
            result = json.loads(json_str)
            label = result.get("label", "").strip().upper()
            if label not in ("CORRECT", "WRONG"):
                raise ValueError(f"Unknown judge label: {label!r}")
            return label == "CORRECT", tokens
        except Exception as e:  # noqa: BLE001
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
    raise RuntimeError(
        f"Judge failed after {config.judge_max_retries} retries "
        f"(question: {question!r}, golden: {golden_answer!r})"
    ) from last_err


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
    judgments: list[bool] = []
    total_tokens = 0
    for _ in range(judge_runs):
        is_correct, tokens = _judge_single(
            llm_client,
            llm_model,
            ar.question,
            ar.golden_answer,
            ar.generated_answer,
            config=config,
        )
        judgments.append(is_correct)
        total_tokens += tokens

    correct = sum(judgments) > judge_runs / 2
    return JudgeResult(
        index=ar.index,
        question=ar.question,
        golden_answer=ar.golden_answer,
        generated_answer=ar.generated_answer,
        category=ar.category,
        is_correct=correct,
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
    answer_results = _read_jsonl(answer_path, AnswerResult)

    def _worker(i: int, ar: AnswerResult) -> JudgeResult:
        return _evaluate_one(
            i,
            ar,
            llm_client=judge_client,
            llm_model=config.judge_model,
            judge_runs=judge_runs,
            config=config,
        )

    raw = _parallel_map(
        answer_results,
        _worker,
        concurrency=config.eval_concurrency,
        pbar=pbar,
    )

    _check_failures(raw)
    results: list[JudgeResult] = raw  # type: ignore[assignment]

    out_path = conv_dir / f"judge_{method_label}.jsonl"
    _write_jsonl(out_path, results)
    return results


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
            all_search.extend(_read_jsonl(search_p, SearchResult))
        if answer_p.exists():
            all_answer.extend(_read_jsonl(answer_p, AnswerResult))
        if judge_p.exists():
            conv_judges = _read_jsonl(judge_p, JudgeResult)
            all_judge.extend(conv_judges)
            c = sum(1 for r in conv_judges if r.is_correct)
            conv_accuracy[conv_idx] = {"correct": c, "total": len(conv_judges)}
        else:
            print(f"  [report] skip conv{conv_idx}/{method} -- no judge JSONL")

    if not all_judge:
        return None

    total = len(all_judge)
    correct = sum(1 for r in all_judge if r.is_correct)

    cat_stats: dict[int, dict[str, int]] = {}
    for r in all_judge:
        cat = r.category
        if cat is None:
            continue
        if cat not in cat_stats:
            cat_stats[cat] = {"correct": 0, "total": 0}
        cat_stats[cat]["total"] += 1
        if r.is_correct:
            cat_stats[cat]["correct"] += 1

    actual_judge_runs = max(
        (len(r.judgments) for r in all_judge if r.judgments), default=config.judge_runs
    )

    per_run_correct = [0] * actual_judge_runs
    per_run_total = 0
    for r in all_judge:
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
    all_candidates = per_run_accuracies + [mean_accuracy, overall_accuracy]
    max_accuracy = round(max(all_candidates), 4) if all_candidates else 0

    search_times = [r.search_time_s for r in all_search]
    answer_times = [r.answer_time_s for r in all_answer]
    answer_tokens = sum(r.answer_tokens for r in all_answer)
    answer_retries = sum(
        r.answer_attempts - 1 for r in all_answer if r.answer_attempts > 1
    )
    judge_tokens = sum(r.judge_tokens for r in all_judge)
    judge_agreements = sum(
        1
        for r in all_judge
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
            for k, v in sorted(cat_stats.items())
        },
        "per_conversation": {
            str(k): {"correct": v["correct"], "total": v["total"]}
            for k, v in sorted(conv_accuracy.items())
        },
        "search": {
            "count": len(all_search),
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
            "retries": answer_retries,
        },
        "judge": {
            "count": len(all_judge),
            "total_tokens": judge_tokens,
            "judge_runs": actual_judge_runs,
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
            f.write(
                f"  Max accuracy:     {s['max_accuracy'] * 100:.1f}% "
                f"(best of {jr} judge runs / mean / majority)\n"
            )
            f.write(
                f"  Majority:         {_pct(s['correct'], s['total'])} "
                f"({s['correct']}/{s['total']})\n"
            )
            f.write(
                f"  Mean accuracy:    {s['mean_accuracy'] * 100:.1f}% "
                f"(avg across {jr} judge runs)\n\n"
            )

            f.write("  Per category:\n")
            for cat_key, cs in sorted(s["category_stats"].items()):
                cat_int = int(cat_key)
                label = CATEGORY_NAMES.get(cat_int, f"cat-{cat_key}")
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
                f"max {ss['max_latency_s']}s\n"
            )

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
            unan = _pct(
                int(js["unanimous_rate"] * js["count"]), js["count"]
            )
            f.write(
                f"  Judge:  {js['count']} questions"
                f" × {js['judge_runs']} runs, "
                f"{js['total_tokens']:,} tokens, "
                f"unanimous {unan}\n"
            )

            total_tokens = ans["total_tokens"] + js["total_tokens"]
            f.write(f"\n  Total tokens: {total_tokens:,}\n\n")


def _print_terminal_summary(
    all_summaries: dict[str, dict[str, Any]],
    output_dir: Path,
    duration_str: str,
) -> None:
    """Print condensed results to the terminal."""
    for method, s in all_summaries.items():
        print(f"\n{'=' * 64}")
        print(f"  Method: {method}")
        jr = s["judge"]["judge_runs"]
        max_pct = f"{s['max_accuracy'] * 100:.1f}%"
        mean_pct = f"{s['mean_accuracy'] * 100:.1f}%"
        maj = _pct(s["correct"], s["total"])
        print(f"  Max:     {max_pct} (best of {jr} runs)")
        print(f"  Majority:{maj:>6s} ({s['correct']}/{s['total']})")
        print(f"  Mean:    {mean_pct} (avg of {jr} runs)")
        for cat_key, cs in sorted(s["category_stats"].items()):
            cat_int = int(cat_key)
            label = CATEGORY_NAMES.get(cat_int, f"cat-{cat_key}")
            acc = _pct(cs["correct"], cs["total"])
            n, t = cs["correct"], cs["total"]
            print(f"    {cat_key}. {label:<14s} {acc:>6s} ({n}/{t})")
        ss, ans, js = s["search"], s["answer"], s["judge"]
        total_tokens = ans["total_tokens"] + js["total_tokens"]
        print(f"  Search: avg {ss['avg_latency_s']}s, p50 {ss['p50_latency_s']}s")
        a_tok = ans["total_tokens"]
        j_tok = js["total_tokens"]
        print(
            f"  Tokens: {total_tokens:,} "
            f"(answer {a_tok:,} + judge {j_tok:,})"
        )
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
    _print_terminal_summary(all_summaries, output_dir, duration_str)


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


def _write_run_spec(
    output_dir: Path,
    run_name: str,
    config: BenchmarkConfig,
    conversations: list[int],
    stages: list[str],
) -> None:
    """Write reproducibility snapshot to run_spec.json."""
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
    )
    (output_dir / "run_spec.json").write_text(
        spec.model_dump_json(indent=2), encoding="utf-8"
    )


# =============================================================================
# Per-conversation orchestrator
# =============================================================================


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
    if "add" in stages and conv_dir.exists():
        shutil.rmtree(conv_dir)
    conv_dir.mkdir(parents=True, exist_ok=True)

    sessions, qa_list, spk_a, spk_b = load_conversation(data_path, conv_index)

    judge_runs = config.judge_runs
    if args.smoke:
        trimmed: list[dict] = []
        msg_count = 0
        for sess in sessions:
            if msg_count >= 50:
                break
            remaining = 50 - msg_count
            if len(sess["messages"]) <= remaining:
                trimmed.append(sess)
                msg_count += len(sess["messages"])
            else:
                trimmed.append({**sess, "messages": sess["messages"][:remaining]})
                msg_count += remaining
        sessions = trimmed
        qa_list = _stratified_sample(qa_list, n=10)
        judge_runs = 1

    app_id = "locomo_benchmark"
    project_id = args.run_name
    _speaker = spk_a if config.eval_owner == "speaker_a" else spk_b
    owner_id = f"{_speaker.lower()}_conv{conv_index}"
    client = EverosClient(base_url=args.base_url)

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

    if "add" in stages:
        add_started = datetime.now(UTC).isoformat()
        _stage("add", sum(len(s["messages"]) for s in sessions), "sending")
        run_add_phase(
            client,
            sessions,
            conv_index,
            config.batch_size,
            app_id=app_id,
            project_id=project_id,
            pbar=pbar,
        )
        pbar.reset(total=0)
        pbar.set_description_str(
            f"{label:<6s} {stage_num}/{total_stages} {'add processing':<15s}"
        )
        _wait_ready(
            args.everos_root,
            conv_index,
            project_id,
            config.cascade_timeout,
            since=add_started,
            pbar=pbar,
        )

    for method in methods:
        if "search" in stages:
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
            )

        if "answer" in stages:
            search_path = conv_dir / f"search_{method}.jsonl"
            if not search_path.exists():
                raise FileNotFoundError(
                    f"Missing {search_path} -- run 'search' stage first"
                )
            _stage("answer", len(_read_jsonl(search_path, SearchResult)))
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
        required=True,
        help="Run name used as project_id for cross-run isolation.",
    )
    p.add_argument(
        "--conv",
        type=int,
        nargs="+",
        default=list(range(10)),
        help="Conversation indices (default: 0..9)",
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
        default="config",
        help="Config TOML name without extension (default: config)",
    )
    p.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="EverOS server address",
    )
    p.add_argument(
        "--everos-root",
        default=str(Path("~/.everos").expanduser()),
        help="EverOS --root path for cascade polling",
    )
    p.add_argument(
        "--data-path",
        default="data/locomo10.json",
        help="Path to LoCoMo dataset",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke: 2 convs, 50 msgs, 10 QA, judge_runs=1",
    )

    args = p.parse_args()
    config = BenchmarkConfig.from_toml(args.config)

    supported = ("keyword", "vector", "hybrid", "agentic")
    bad = [m for m in config.parsed_methods if m not in supported]
    if bad:
        p.error(f"unsupported method(s) in config.toml: {bad}; supported: {supported}")

    if args.smoke:
        args.conv = [0, 1]

    return args, config


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Entry point: orchestrate all conversations."""
    args, config = parse_args()

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
        timeout=60,
        max_retries=1,
    )
    if answer_base_url == judge_base_url and answer_api_keys == judge_api_keys:
        judge_client = answer_client
    else:
        judge_client = LLMClientPool(
            judge_api_keys,
            base_url=judge_base_url,
            timeout=60,
            max_retries=1,
        )

    output_dir = Path("benchmarks/results") / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_run_spec(output_dir, args.run_name, config, args.conv, args.stages)

    print(
        f"  Answer LLM: {config.answer_model} @ {answer_base_url}"
        f" ({answer_client.key_count} keys)"
    )
    print(
        f"  Judge  LLM: {config.judge_model} @ {judge_base_url}"
        f" ({judge_client.key_count} keys)"
    )
    print(f"  Search mode: {config.methods}")
    print(f"  Conversations: {args.conv}")
    print(f"  Stages: {args.stages}")
    print(f"  Output: {output_dir}")

    conv_positions = {ci: pos for pos, ci in enumerate(args.conv)}

    conv_errors: dict[int, str] = {}
    conv_pbars: dict[int, _tqdm] = {}

    def _run_conv(conv_index: int) -> bool:
        try:
            pbar = run_conversation(
                conv_index,
                args=args,
                config=config,
                stages=args.stages,
                answer_client=answer_client,
                judge_client=judge_client,
                data_path=args.data_path,
                output_dir=output_dir,
                position=conv_positions[conv_index],
            )
            conv_pbars[conv_index] = pbar
            return True
        except Exception as e:
            import traceback

            tb = traceback.format_exc()
            conv_dir = output_dir / f"conv{conv_index}"
            conv_dir.mkdir(parents=True, exist_ok=True)
            (conv_dir / "error.log").write_text(tb, encoding="utf-8")
            conv_errors[conv_index] = str(e)
            _tqdm.write(f"  conv{conv_index} FAILED: {e}")
            return False

    with ThreadPoolExecutor(max_workers=config.conversations_concurrency) as pool:
        futures = {pool.submit(_run_conv, ci): ci for ci in args.conv}
        results = {ci: f.result() for f, ci in futures.items()}

    for pbar in conv_pbars.values():
        pbar.leave = True
        pbar.close()

    failed = [ci for ci, ok in results.items() if not ok]
    if failed:
        print(f"\n{len(failed)} conversation(s) failed: {failed}")
        for ci in failed:
            print(f"  see {output_dir}/conv{ci}/error.log")

    # Aggregate
    if "judge" in args.stages:
        aggregate_report(output_dir, args.conv, config)

    print(f"\nDone. Results: {output_dir}")


if __name__ == "__main__":
    main()

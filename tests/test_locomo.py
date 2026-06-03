"""
EverOS E2E Benchmark — LoCoMo conversation evaluation.

Self-contained script that exercises the full everos HTTP pipeline:
  Add (per LoCoMo session) -> Flush -> Search -> Answer (LLM) -> Evaluate (LLM Judge)

LoCoMo ↔ everos mapping (Plan C: single-owner evaluation):
- LoCoMo conversation N         → owner_id = speaker_a.lower() (or speaker_b
                                  via --eval-owner). Query a single speaker's
                                  partition; everos's pipeline fan-outs the
                                  same episode narrative to every user sender
                                  (see user_memory.py:95-117), so single-owner
                                  retrieval is informationally equivalent to
                                  multi-owner merged.
- LoCoMo session_N (sitting)    → everos session_id = f"locomo_conv{N}_s{idx}"
                                  with a /memory/flush after each session.
- Each LoCoMo message           → MessageItemDTO with real sender_id
                                  (speaker.lower()), preserving speaker
                                  attribution in the storage / extraction.

Usage:
  python tests/test_locomo.py --methods hybrid --quiet
  python tests/test_locomo.py --methods keyword,hybrid,agentic --output results.json
  python tests/test_locomo.py --skip-add --methods hybrid  # reuse loaded data
"""

import argparse
import concurrent.futures
import json
import os
import re
import statistics
import sys
import threading
import time
from datetime import UTC, datetime
from typing import Any

import openai
import requests
from dotenv import load_dotenv

try:
    from tqdm import tqdm as _tqdm
except ImportError:  # progress bar is a nice-to-have, never a hard dep
    _tqdm = None


def _progress(iterable, *, desc: str, total: int, quiet: bool):
    """Wrap a loop iterable in a tqdm bar, but only in quiet mode.

    Verbose mode already prints a per-item line, so a bar there would just
    fight those prints. Quiet mode prints nothing per item, which is exactly
    when the user is left staring at a frozen screen — so that is where the
    bar earns its keep. Falls back to the bare iterable if tqdm is missing.
    """
    if not quiet or _tqdm is None:
        return iterable
    return _tqdm(iterable, desc=desc, total=total, unit="item", dynamic_ncols=True)


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

# RESPONSE FORMAT (You MUST follow this structure):

## STEP 1: RELEVANT MEMORIES EXTRACTION
[List each memory that relates to the question, with its timestamp]
- Memory 1: [timestamp] - [content]
- Memory 2: [timestamp] - [content]
...

## STEP 2: KEY INFORMATION IDENTIFICATION
[Extract ALL specific details from the memories]
- Names mentioned: [list all person names, place names, company names]
- Numbers/Quantities: [list all amounts, prices, percentages]
- Dates/Times: [list all temporal information]
- Frequencies: [list any recurring patterns]
- Other entities: [list brands, products, etc.]

## STEP 3: CROSS-MEMORY LINKING
[Identify entities that appear in multiple memories and link related information. Make reasonable inferences when entities are strongly connected.]
- Shared entities: [list people, places, events mentioned across different memories]
- Connections found: [e.g., "Memory 1 mentions A moved from hometown → Memory 2 mentions A's hometown is LA → Therefore A moved from LA"]
- Inferred facts: [list any facts that require combining information from multiple memories]

## STEP 4: TIME REFERENCE CALCULATION
[If applicable, convert relative time references]
- Original reference: [e.g., "last year" from May 2022]
- Calculated actual time: [e.g., "2021"]

## STEP 5: CONTRADICTION CHECK
[If multiple memories contain different information]
- Conflicting information: [describe]
- Resolution: [explain which is most recent/reliable]

## STEP 6: DETAIL VERIFICATION CHECKLIST
- [ ] All person names included: [list them]
- [ ] All locations included: [list them]
- [ ] All numbers exact: [list them]
- [ ] All frequencies specific: [list them]
- [ ] All dates/times precise: [list them]
- [ ] All proper nouns preserved: [list them]

## STEP 7: ANSWER FORMULATION
[Explain how you're combining the information to answer the question]

## FINAL ANSWER:
[Provide the concise answer with ALL specific details preserved]

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

    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 300):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def post(
        self, path: str, data: dict[str, Any], quiet: bool = False
    ) -> tuple[int, dict]:
        full_url = f"{self.base_url}{path}"
        if not quiet:
            print(f"\n📍 URL: POST {full_url}")
            print(f"📤 Request Data:\n{json.dumps(data, indent=2, ensure_ascii=False)}")
        try:
            resp = requests.post(
                full_url,
                json=data,
                headers={"Content-Type": "application/json"},
                timeout=(10, self.timeout),
            )
        except requests.RequestException as e:
            if not quiet:
                print(f"📥 Request Error: {e}")
            return -1, {"error": str(e)}
        if not quiet:
            print(f"\n📥 Response Status Code: {resp.status_code}")
        try:
            body = resp.json()
            if not quiet:
                print(
                    f"📥 Response Data:\n{json.dumps(body, indent=2, ensure_ascii=False)}"
                )
            return resp.status_code, body
        except Exception:
            if not quiet:
                print(f"📥 Raw Response: {resp.text[:500]}")
            return resp.status_code, {}


def print_section(title: str):
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print(f"{'=' * 72}")


# =============================================================================
# LLM client pool — round-robin across multiple API keys with 429 failover
# =============================================================================


def _split_keys(s: str) -> list[str]:
    """Split a comma-separated key string into a list of stripped non-empty keys."""
    return [k.strip() for k in s.split(",") if k.strip()]


class _PoolCompletions:
    def __init__(self, pool: "LLMClientPool"):
        self._pool = pool

    def create(self, **kwargs: Any) -> Any:
        return self._pool._create_with_failover(**kwargs)


class _PoolChat:
    def __init__(self, pool: "LLMClientPool"):
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
                continue
        assert last_err is not None
        raise last_err


def _parallel_map(
    items: list,
    worker,
    *,
    desc: str,
    total: int,
    quiet: bool,
    concurrency: int,
) -> list:
    """Run ``worker(i, item)`` over *items* concurrently; preserve input order.

    Quiet mode drives a tqdm progress bar via ``as_completed``; verbose mode
    lets workers stay silent to avoid interleaved output.  Falls back to serial
    execution when *concurrency* <= 1.

    Worker exceptions are caught per-item: the exception object is stored in
    ``results[i]`` and re-raised by callers as needed.  This prevents one bad
    LLM call from aborting the entire batch.

    Args:
        items: Input list to process.
        worker: Callable ``(i: int, item: Any) -> Any``.
        desc: Label shown in the tqdm bar.
        total: Expected number of items (for the bar).
        quiet: When True and tqdm is available, show a progress bar.
        concurrency: Thread-pool size; <= 1 means serial.

    Returns:
        List of worker results in the same order as *items*.
    """
    results: list = [None] * len(items)

    if concurrency <= 1:
        for i, item in enumerate(items):
            results[i] = worker(i, item)
        return results

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        future_to_idx: dict[concurrent.futures.Future, int] = {
            pool.submit(worker, i, item): i for i, item in enumerate(items)
        }
        if quiet and _tqdm is not None:
            completed_iter = _tqdm(
                concurrent.futures.as_completed(future_to_idx),
                total=total,
                desc=desc,
                unit="item",
                dynamic_ncols=True,
            )
        else:
            completed_iter = concurrent.futures.as_completed(future_to_idx)

        for fut in completed_iter:
            idx = future_to_idx[fut]
            try:
                results[idx] = fut.result()
            except Exception as exc:  # noqa: BLE001
                results[idx] = exc

    return results


def _wait_cascade_drain(
    corpus_path: str, max_wait_s: int, poll_interval_s: float = 3.0
) -> None:
    """Poll everos cascade queue in <corpus>/.index/sqlite/system.db.

    Returns as soon as ``md_change_state`` has no row in
    ``status IN ('pending', 'processing')``, or after ``max_wait_s``
    seconds, whichever comes first. Falls back to a fixed sleep if the
    sqlite file is missing (e.g. corpus_path wrong / server not yet
    written its system.db).
    """
    import sqlite3
    from pathlib import Path

    db_path = Path(corpus_path).expanduser() / ".index" / "sqlite" / "system.db"
    if not db_path.exists():
        print(
            f"  [warn] cascade queue db not found at {db_path}; "
            f"falling back to fixed {max_wait_s}s sleep"
        )
        time.sleep(max_wait_s)
        return

    print(
        f"  Polling cascade queue at {db_path} (max {max_wait_s}s, "
        f"interval {poll_interval_s}s)..."
    )
    deadline = time.time() + max_wait_s
    last_pending = -1
    while time.time() < deadline:
        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            n = conn.execute(
                "SELECT COUNT(*) FROM md_change_state "
                "WHERE status IN ('pending', 'processing')"
            ).fetchone()[0]
            conn.close()
        except sqlite3.Error as e:
            print(f"  [warn] sqlite read failed: {e}; assume drained")
            return
        if n == 0:
            elapsed = max_wait_s - (deadline - time.time())
            print(f"  cascade drained after {elapsed:.1f}s")
            return
        if n != last_pending:
            print(f"  cascade pending = {n}")
            last_pending = n
        time.sleep(poll_interval_s)
    print(
        f"  [warn] cascade still has {last_pending} pending after "
        f"{max_wait_s}s; proceeding anyway"
    )


# =============================================================================
# Data loading — preserve LoCoMo session_N structure for per-session flushing
# =============================================================================


def _parse_session_timestamp(ts_str: str) -> int:
    """Parse LoCoMo timestamp string to epoch milliseconds.

    Format examples: "1:56 pm on 8 May, 2023", "12:09 am on 13 September, 2023".

    LoCoMo's raw timestamps carry no timezone, so we pin them to UTC —
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

    Returns (sessions, qa_list, speaker_a, speaker_b) where `sessions` is
    a list of {session_idx, messages} ordered by session_idx. Each message
    carries dia_id / speaker / text / timestamp_ms. QA list excludes
    category 5 (adversarial).
    """
    with open(data_path, encoding="utf-8") as f:
        dataset = json.load(f)

    if conv_index >= len(dataset):
        raise ValueError(
            f"conv_index {conv_index} out of range (dataset has {len(dataset)} conversations)"
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
# Add phase — one everos session_id per LoCoMo session, flush after each
# =============================================================================


def run_add_phase(
    client: EverosClient,
    sessions: list[dict],
    speaker_a: str,
    speaker_b: str,
    conv_index: int,
    batch_size: int,
    quiet: bool = False,
) -> dict[str, Any]:
    """Send each LoCoMo session to its own everos session_id and flush."""
    print_section("Add Phase")
    total_msgs = sum(len(s["messages"]) for s in sessions)
    print(
        f"  LoCoMo sessions: {len(sessions)} | Messages: {total_msgs} | "
        f"Batch size: {batch_size} | Speakers: {speaker_a} & {speaker_b}"
    )

    t0 = time.perf_counter()
    total_batches = 0

    for sess in _progress(sessions, desc="Add+Flush", total=len(sessions), quiet=quiet):
        session_id = f"locomo_conv{conv_index}_s{sess['session_idx']}"
        api_messages: list[dict] = [
            {
                # Append `_conv{N}` so the same speaker name across conversations
                # (e.g. "John" appears in conv_2, conv_4, conv_6) does NOT collide
                # on a shared owner_id partition. Without the suffix, repeated
                # benchmark runs cross-pollute each other's memory store.
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
            payload = {"session_id": session_id, "messages": batch}
            status, resp = client.post("/api/v1/memory/add", payload, quiet=quiet)
            if not quiet:
                print(
                    f"  Session {sess['session_idx']} batch {idx + 1}/{len(batches)}: "
                    f"{len(batch)} msgs -> status {status}"
                )
            assert status == 200, (
                f"Add (session_id={session_id}, batch {idx + 1}) failed: "
                f"status={status} resp={resp}"
            )
        total_batches += len(batches)

        flush_status, flush_resp = client.post(
            "/api/v1/memory/flush", {"session_id": session_id}, quiet=quiet
        )
        assert flush_status == 200, (
            f"Flush (session_id={session_id}) failed: "
            f"status={flush_status} resp={flush_resp}"
        )

    add_time = time.perf_counter() - t0
    result = {
        "total_messages": total_msgs,
        "session_count": len(sessions),
        "batch_count": total_batches,
        "batch_size": batch_size,
        "add_time_seconds": round(add_time, 2),
    }
    print(
        f"  Done: {total_msgs} msgs across {len(sessions)} sessions "
        f"({total_batches} batches), {add_time:.2f}s incl. flushes"
    )
    return result


# =============================================================================
# Search phase — single-owner partition (Plan C)
# =============================================================================


def _search_one(
    i: int,
    qa: dict,
    *,
    client: EverosClient,
    method: str,
    top_k: int,
    owner_id: str,
) -> dict:
    """Search a single QA question; safe to run in a thread.

    The per-request timeout is handled by ``EverosClient.post`` via the
    underlying ``requests`` ``timeout=(10, self.timeout)`` kwarg (self.timeout
    defaults to 300 s), so no extra thread-based wrapping is needed here.
    """
    question = qa["question"]
    payload: dict = {
        "query": question,
        "method": method,
        "top_k": top_k,
        "user_id": owner_id,
    }
    t0 = time.perf_counter()
    try:
        status, resp = client.post("/api/v1/memory/search", payload, quiet=True)
    except Exception as e:
        status, resp = -1, {"error": str(e)}
    search_time = time.perf_counter() - t0

    if status != 200:
        error_detail = resp.get("detail", resp) if isinstance(resp, dict) else resp
        return {
            "index": i,
            "question": question,
            "golden_answer": qa["answer"],
            "category": qa.get("category"),
            "evidence": qa.get("evidence", []),
            "episodes": [],
            "profiles": [],
            "search_time_s": round(search_time, 4),
            "search_error": error_detail,
            "_search_status": status,
        }

    data = resp.get("data", {})
    episodes = data.get("episodes", [])
    profiles = data.get("profiles", [])
    return {
        "index": i,
        "question": question,
        "golden_answer": qa["answer"],
        "category": qa.get("category"),
        "evidence": qa.get("evidence", []),
        "episodes": episodes,
        "profiles": profiles,
        "search_time_s": round(search_time, 4),
    }


def run_search_phase(
    client: EverosClient,
    qa_list: list[dict],
    owner_id: str,
    method: str,
    top_k: int,
    quiet: bool = False,
    concurrency: int = 10,
) -> list[dict]:
    """Search for each QA question against a single owner_id partition (parallel).

    Vector retrieval strategy (``episode`` vs ``maxsim_atomic``) is selected
    on the server side via ``EVEROS_SEARCH__VECTOR_STRATEGY`` — this driver
    just hits the public ``/api/v1/memory/search`` endpoint and reports
    what the server returned.
    """
    print_section(f"Search Phase (method={method}, top_k={top_k}, owner_id={owner_id})")

    def _worker(i: int, qa: dict) -> dict:
        return _search_one(
            i,
            qa,
            client=client,
            method=method,
            top_k=top_k,
            owner_id=owner_id,
        )

    raw = _parallel_map(
        qa_list,
        _worker,
        desc="Search",
        total=len(qa_list),
        quiet=quiet,
        concurrency=concurrency,
    )

    # Unwrap: _parallel_map stores exceptions as values; surface them as error dicts.
    results: list[dict] = []
    for item in raw:
        if isinstance(item, Exception):
            results.append(
                {
                    "episodes": [],
                    "profiles": [],
                    "search_time_s": 0,
                    "search_error": str(item),
                }
            )
        else:
            results.append(item)

    errors = [r for r in results if r.get("search_error")]
    search_times = [r["search_time_s"] for r in results if not r.get("search_error")]
    success_count = len(results) - len(errors)
    summary_parts = [f"Done: {success_count}/{len(results)} succeeded"]
    if search_times:
        summary_parts.append(f"avg={statistics.mean(search_times):.3f}s")
    if errors:
        summary_parts.append(f"{len(errors)} FAILED")
        for err in errors:
            print(
                f"    ERROR Q{err.get('index', '?')}: "
                f"status={err.get('_search_status', 'exc')} | "
                f"{str(err.get('question', ''))[:60]}"
            )
    print(f"  {', '.join(summary_parts)}")

    # Strip internal bookkeeping key before returning.
    for r in results:
        r.pop("_search_status", None)
    return results


# =============================================================================
# Answer phase
# =============================================================================


def _build_context(
    episodes: list[dict], profiles: list[dict], speaker_a: str, speaker_b: str
) -> str:
    """Build context string from search results."""
    lines = [
        f"Episodes memories for conversation between {speaker_a} and {speaker_b}:\n"
    ]
    for idx, ep in enumerate(episodes, 1):
        subject = ep.get("subject", "")
        body = ep.get("episode") or ep.get("summary") or ep.get("content") or ""
        prefix = f"{subject}: " if subject else ""
        lines.append(f"{idx}. {prefix}{body}")

    if profiles:
        lines.append("\nProfile memories:")
        for idx, p in enumerate(profiles, 1):
            content = p.get("content") or p.get("summary") or ""
            lines.append(f"  {idx}. {content}")

    return "\n".join(lines)


def _extract_final_answer(text: str) -> str:
    """Extract text after 'FINAL ANSWER:' marker."""
    marker = "FINAL ANSWER:"
    idx = text.upper().rfind(marker.upper())
    if idx != -1:
        answer = text[idx + len(marker) :].strip()
        answer = re.sub(r"^#+\s*", "", answer).strip()
        return answer
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if line:
            return line
    return text.strip()


def _answer_one(
    i: int,
    sr: dict,
    *,
    speaker_a: str,
    speaker_b: str,
    llm_client: LLMClientPool,
    llm_model: str,
) -> dict:
    """Generate an answer for a single search result; safe to run in a thread.

    Retry up to 3× with rising temperature when the response parses to an
    empty FINAL ANSWER:.  gpt-4.1-mini occasionally finishes the STEP 7
    reasoning, emits the marker, then stops without the body — at temperature=0
    the truncation is deterministic, so retries bump temperature to break the
    same sampling path.

    The openai ``timeout=300`` kwarg is a per-request socket deadline passed
    directly to the underlying HTTP client, which is safe to use from a thread
    pool (no extra nesting needed).
    """
    if sr.get("search_error"):
        return {**sr, "generated_answer": "[SEARCH_FAILED]", "answer_time_s": 0}

    context = _build_context(sr["episodes"], sr["profiles"], speaker_a, speaker_b)
    prompt = ANSWER_PROMPT.format(context=context, question=sr["question"])

    t0 = time.perf_counter()
    raw_answer = ""
    generated_answer = ""
    last_error: str | None = None
    attempts_used = 0
    for attempt, temp in enumerate((0.0, 0.3, 0.6)):
        attempts_used = attempt + 1
        try:
            r = llm_client.chat.completions.create(
                model=llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temp,
                timeout=300,
            )
            raw_answer = r.choices[0].message.content or ""
        except Exception as e:
            last_error = f"[ERROR: {e}]"
            raw_answer = last_error
            continue

        generated_answer = _extract_final_answer(raw_answer)
        if generated_answer.strip():
            break

    if not generated_answer.strip() and last_error:
        generated_answer = last_error

    answer_time = time.perf_counter() - t0
    return {
        **sr,
        "generated_answer": generated_answer,
        "answer_time_s": round(answer_time, 4),
        "answer_attempts": attempts_used,
    }


def run_answer_phase(
    search_results: list[dict],
    speaker_a: str,
    speaker_b: str,
    llm_client: LLMClientPool,
    llm_model: str,
    quiet: bool = False,
    concurrency: int = 8,
) -> list[dict]:
    """Generate answers using LLM for each search result (parallel)."""
    print_section("Answer Phase")

    def _worker(i: int, sr: dict) -> dict:
        return _answer_one(
            i,
            sr,
            speaker_a=speaker_a,
            speaker_b=speaker_b,
            llm_client=llm_client,
            llm_model=llm_model,
        )

    raw = _parallel_map(
        search_results,
        _worker,
        desc="Answer",
        total=len(search_results),
        quiet=quiet,
        concurrency=concurrency,
    )
    # Unwrap: _parallel_map stores exceptions as values; surface them as error strings.
    results: list[dict] = []
    for item in raw:
        if isinstance(item, Exception):
            results.append({"generated_answer": f"[ERROR: {item}]", "answer_time_s": 0})
        else:
            results.append(item)

    answer_times = [
        r["answer_time_s"] for r in results if r.get("answer_time_s", 0) > 0
    ]
    avg_str = f", avg={statistics.mean(answer_times):.2f}s" if answer_times else ""
    skipped = sum(1 for r in results if r.get("search_error"))
    skip_str = f", {skipped} skipped (search failed)" if skipped else ""
    print(f"  Done: {len(results)} answers{avg_str}{skip_str}")
    return results


# =============================================================================
# Evaluate phase — LLM-as-Judge
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
) -> bool:
    """Judge a single answer. Returns True if CORRECT.

    Uses ``timeout=300`` passed directly to the openai HTTP client so this
    function is safe to call from a thread pool without further nesting.
    """
    user_prompt = JUDGE_USER_PROMPT.format(
        question=question,
        golden_answer=golden_answer,
        generated_answer=generated_answer,
    )
    try:
        r = llm_client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            timeout=300,
        )
        content = r.choices[0].message.content or ""
        json_str = _extract_json(content)
        if not json_str:
            return False
        result = json.loads(json_str)
        return result.get("label", "").strip().upper() == "CORRECT"
    except Exception as e:
        print(f"    Judge error: {e}")
        return False


def _evaluate_one(
    i: int,
    ar: dict,
    *,
    llm_client: LLMClientPool,
    llm_model: str,
    judge_runs: int,
) -> dict:
    """Evaluate a single answer result with majority-vote judging.

    Safe to run from a thread pool; exception handling mirrors the original
    serial loop so a single LLM failure degrades gracefully.
    """
    if ar.get("search_error"):
        return {**ar, "judgments": [], "is_correct": False}

    judgments: list[bool] = []
    for _ in range(judge_runs):
        judgments.append(
            _judge_single(
                llm_client,
                llm_model,
                ar["question"],
                ar["golden_answer"],
                ar["generated_answer"],
            )
        )

    correct = sum(judgments) > judge_runs / 2
    return {**ar, "judgments": judgments, "is_correct": correct}


def run_evaluate_phase(
    answer_results: list[dict],
    llm_client: LLMClientPool,
    llm_model: str,
    judge_runs: int = 1,
    quiet: bool = False,
    concurrency: int = 8,
) -> list[dict]:
    """Evaluate answers using LLM judge (parallel)."""
    print_section(f"Evaluate Phase (judge_runs={judge_runs})")

    def _worker(i: int, ar: dict) -> dict:
        return _evaluate_one(
            i,
            ar,
            llm_client=llm_client,
            llm_model=llm_model,
            judge_runs=judge_runs,
        )

    raw = _parallel_map(
        answer_results,
        _worker,
        desc="Evaluate",
        total=len(answer_results),
        quiet=quiet,
        concurrency=concurrency,
    )
    results: list[dict] = []
    for item in raw:
        if isinstance(item, Exception):
            results.append({"judgments": [], "is_correct": False})
        else:
            results.append(item)

    correct_count = sum(1 for r in results if r["is_correct"])
    print(
        f"  Done: {correct_count}/{len(results)} correct ({_pct(correct_count, len(results))})"
    )
    return results


# =============================================================================
# Reporting
# =============================================================================


def _pct(n: int, total: int) -> str:
    if total == 0:
        return "N/A"
    return f"{n / total * 100:.1f}%"


def _percentiles(values: list[float]) -> tuple[float, float, float]:
    """Return P50, P90, P99."""
    if not values:
        return 0.0, 0.0, 0.0
    s = sorted(values)
    n = len(s)

    def _p(pct: float) -> float:
        idx = int(pct / 100 * (n - 1))
        return s[min(idx, n - 1)]

    return _p(50), _p(90), _p(99)


def print_report(
    eval_results: list[dict],
    method: str,
    add_result: dict | None,
    conv_label: str = "",
    top_k: int = 10,
) -> dict[str, Any]:
    """Print formatted report and return summary dict."""
    total = len(eval_results)
    correct = sum(1 for r in eval_results if r["is_correct"])
    search_errors = sum(1 for r in eval_results if r.get("search_error"))

    cat_stats: dict[int, dict[str, int]] = {}
    for r in eval_results:
        cat = r.get("category")
        if cat is None:
            continue
        if cat not in cat_stats:
            cat_stats[cat] = {"correct": 0, "total": 0}
        cat_stats[cat]["total"] += 1
        if r["is_correct"]:
            cat_stats[cat]["correct"] += 1

    search_times = [
        r["search_time_s"] for r in eval_results if not r.get("search_error")
    ]
    answer_times = [r["answer_time_s"] for r in eval_results if r["answer_time_s"] > 0]
    s_p50, s_p90, s_p99 = _percentiles(search_times)
    a_p50, a_p90, a_p99 = _percentiles(answer_times)

    today = datetime.now().strftime("%Y-%m-%d")

    print(f"\n{'=' * 64}")
    print("  EverOS E2E Benchmark Report")
    if conv_label:
        print(f"  Conversation: {conv_label}")
    print(
        f"  Messages: {add_result['total_messages'] if add_result else 'N/A (--skip-add)'} | Questions: {total}"
    )
    print(f"  Search Method: {method} | top_k: {top_k}")
    print(f"  Date: {today}")
    print(f"{'=' * 64}")

    if add_result:
        print("\nADD PHASE")
        print(f"  Total messages:      {add_result['total_messages']}")
        print(f"  LoCoMo sessions:     {add_result['session_count']}")
        print(
            f"  Batch size:          {add_result['batch_size']} ({add_result['batch_count']} requests)"
        )
        print(f"  Add + flush time:    {add_result['add_time_seconds']}s")

    print("\nSEARCH PHASE")
    if search_times:
        print(f"  Avg search time:     {statistics.mean(search_times):.3f}s")
        print(f"  P50 / P90 / P99:     {s_p50:.3f}s / {s_p90:.3f}s / {s_p99:.3f}s")
    else:
        print("  No successful searches")

    print("\nANSWER PHASE")
    if answer_times:
        print(f"  Avg answer time:     {statistics.mean(answer_times):.2f}s")
        print(f"  P50 / P90 / P99:     {a_p50:.2f}s / {a_p90:.2f}s / {a_p99:.2f}s")
    else:
        print("  No successful answers")

    if search_errors:
        print(f"\nSEARCH ERRORS:         {search_errors}/{total}")

    print("\nACCURACY")
    print(f"  Overall:             {_pct(correct, total)} ({correct}/{total})")
    for cat in sorted(cat_stats.keys()):
        s = cat_stats[cat]
        label = CATEGORY_NAMES.get(cat, f"cat-{cat}")
        print(
            f"  Category {cat} ({label}): {_pct(s['correct'], s['total'])} ({s['correct']}/{s['total']})"
        )

    print(f"\n{'=' * 64}")

    return {
        "method": method,
        "total": total,
        "correct": correct,
        "search_errors": search_errors,
        "accuracy": correct / total if total else 0,
        "category_stats": {
            str(k): {"correct": v["correct"], "total": v["total"]}
            for k, v in cat_stats.items()
        },
        "avg_search_s": round(statistics.mean(search_times), 4) if search_times else 0,
        "avg_answer_s": round(statistics.mean(answer_times), 4) if answer_times else 0,
    }


def print_comparison(all_summaries: dict[str, dict[str, Any]]):
    """Print a comparison table across methods."""
    print(f"\n{'=' * 64}")
    print("  METHOD COMPARISON")
    print(f"{'=' * 64}")

    all_cats = set()
    for s in all_summaries.values():
        all_cats.update(s.get("category_stats", {}).keys())
    cats_sorted = sorted(all_cats)

    cat_headers = [f"Cat {c}" for c in cats_sorted]
    header = (
        f"  {'Method':<10} | {'Overall':>8} | "
        + " | ".join(f"{ch:>8}" for ch in cat_headers)
        + f" | {'Avg Search':>10} | {'Avg Answer':>10}"
    )
    print(header)
    print(f"  {'-' * (len(header) - 2)}")

    for method, summary in all_summaries.items():
        overall = f"{summary['accuracy'] * 100:.1f}%"
        cat_strs = []
        for c in cats_sorted:
            cs = summary.get("category_stats", {}).get(c, {})
            if cs.get("total", 0) > 0:
                cat_strs.append(f"{cs['correct'] / cs['total'] * 100:.1f}%")
            else:
                cat_strs.append("N/A")
        cat_part = " | ".join(f"{s:>8}" for s in cat_strs)
        print(
            f"  {method:<10} | {overall:>8} | {cat_part} | "
            f"{summary['avg_search_s']:>9.3f}s | {summary['avg_answer_s']:>9.2f}s"
        )

    print()


# =============================================================================
# Checkpoint helpers
# =============================================================================


def _summarize_episode(ep: dict) -> str:
    """One-line summary of an episode for checkpoint display."""
    subject = ep.get("subject", "")
    body = ep.get("episode") or ep.get("summary") or ep.get("content") or ""
    if len(body) > 280:
        body = body[:117] + "..."
    return f"{subject}: {body}" if subject else body


def _compact_search_result(r: dict) -> dict:
    """Compact a search result for checkpoint: strip bulky fields."""
    entry = {
        "index": r["index"],
        "question": r["question"],
        "golden_answer": r["golden_answer"],
        "category": r.get("category"),
        "search_time_s": r["search_time_s"],
        "episode_count": len(r.get("episodes", [])),
        "profile_count": len(r.get("profiles", [])),
        "episodes_preview": [
            _summarize_episode(ep) for ep in r.get("episodes", [])[:5]
        ],
    }
    if r.get("search_error"):
        entry["search_error"] = str(r["search_error"])
    return entry


def _compact_answer_result(r: dict) -> dict:
    entry = _compact_search_result(r)
    entry["generated_answer"] = r.get("generated_answer", "")
    entry["answer_time_s"] = r.get("answer_time_s", 0)
    return entry


def _compact_eval_result(r: dict) -> dict:
    entry = _compact_answer_result(r)
    entry["is_correct"] = r.get("is_correct", False)
    entry["judgments"] = r.get("judgments", [])
    return entry


def _build_checkpoint(phase: str, results: list[dict]) -> dict:
    """Build a user-friendly checkpoint dict with summary + details."""
    if phase == "search":
        compact_fn = _compact_search_result
    elif phase == "answer":
        compact_fn = _compact_answer_result
    else:
        compact_fn = _compact_eval_result

    details = [compact_fn(r) for r in results]
    error_count = sum(1 for r in results if r.get("search_error"))
    summary: dict[str, Any] = {"total": len(results), "errors": error_count}

    if phase == "search":
        times = [r["search_time_s"] for r in results if not r.get("search_error")]
        summary["avg_search_time_s"] = (
            round(statistics.mean(times), 4) if times else None
        )
    elif phase == "answer":
        times = [r["answer_time_s"] for r in results if r.get("answer_time_s", 0) > 0]
        summary["avg_answer_time_s"] = (
            round(statistics.mean(times), 4) if times else None
        )
    elif phase == "eval":
        correct = sum(1 for r in results if r.get("is_correct"))
        summary["correct"] = correct
        summary["accuracy"] = (
            f"{correct / len(results) * 100:.1f}%" if results else "N/A"
        )
        cat_stats: dict[int, dict[str, int]] = {}
        for r in results:
            cat = r.get("category")
            if cat is None:
                continue
            if cat not in cat_stats:
                cat_stats[cat] = {"correct": 0, "total": 0}
            cat_stats[cat]["total"] += 1
            if r.get("is_correct"):
                cat_stats[cat]["correct"] += 1
        summary["by_category"] = {
            f"cat{k} ({CATEGORY_NAMES.get(k, '?')})": f"{v['correct']}/{v['total']}"
            for k, v in sorted(cat_stats.items())
        }

    return {"summary": summary, "details": details}


def _save_checkpoint(
    checkpoint_dir: str, filename: str, phase: str, results: list[dict]
) -> str:
    """Save user-friendly checkpoint with summary + compact details."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, filename)
    checkpoint = _build_checkpoint(phase, results)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2, ensure_ascii=False, default=str)
    print(f"  [checkpoint] {path}")
    return path


# =============================================================================
# CLI
# =============================================================================

_SUPPORTED_METHODS = ("keyword", "vector", "hybrid", "agentic")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="EverOS E2E Benchmark (LoCoMo)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--base-url", default="http://localhost:8000", help="everos API base URL"
    )
    p.add_argument(
        "--data-path", default="data/locomo10.json", help="Path to LoCoMo dataset"
    )
    p.add_argument(
        "--conv-index", type=int, default=0, help="Conversation index in dataset"
    )
    p.add_argument(
        "--methods",
        default="hybrid",
        help=f"Comma-separated search methods ({'/'.join(_SUPPORTED_METHODS)})",
    )
    p.add_argument(
        "--top-k", type=int, default=10, help="Number of results to retrieve (1..100)"
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Messages per add request (1..500; everos caps at 500)",
    )
    p.add_argument(
        "--post-flush-wait",
        type=int,
        default=180,
        help="Seconds to wait after final flush for async indexing "
        "(OME chain: extract_atomic_facts → extract_foresight → "
        "extract_user_profile, plus cascade md → LanceDB sync). "
        "When --corpus-path is provided, this becomes a MAX wait + "
        "the script polls the cascade queue and returns as soon as "
        "pending==0. Otherwise it is a fixed sleep.",
    )
    p.add_argument(
        "--corpus-path",
        default=None,
        help="Path to the everos memory root (e.g. ~/.everos-locomo-X). "
        "When provided, post-flush wait switches from fixed sleep to "
        "polling <corpus>/.index/sqlite/system.db for md_change_state "
        "pending==0, capped at --post-flush-wait seconds.",
    )
    p.add_argument(
        "--judge-runs",
        type=int,
        default=3,
        help="LLM judge evaluation runs per question (majority vote, default: 3)",
    )
    p.add_argument(
        "--eval-owner",
        default="speaker_a",
        choices=["speaker_a", "speaker_b"],
        help="Which speaker's memory partition to query (Plan C: single-owner eval)",
    )
    p.add_argument(
        "--skip-add", action="store_true", help="Skip add phase (reuse existing data)"
    )
    p.add_argument(
        "--quiet", action="store_true", help="Suppress per-request verbose output"
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=20,
        help=(
            "Number of concurrent requests in Answer and Evaluate phases "
            "(default: 20). With a single API key, lower to 4 to avoid LLM rate "
            "limits. Search has its own knob: --search-concurrency."
        ),
    )
    p.add_argument(
        "--search-concurrency",
        type=int,
        default=10,
        help=(
            "Number of concurrent /search requests (default: 10). Separate from "
            "--concurrency because the bottleneck is LanceDB file descriptors "
            "(each BM25 query opens all active FTS index parts), not LLM rate. "
            "Raise to 20+ only if FD headroom is verified."
        ),
    )
    p.add_argument(
        "--output", type=str, default=None, help="Write full results to JSON file"
    )
    p.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory for intermediate checkpoint files (auto-saved after each phase)",
    )
    # Answer / Judge LLM split. Resolution order for each field:
    #   CLI flag > ANSWER_* / JUDGE_* env > LLM_* env > built-in default
    p.add_argument(
        "--answer-model",
        default=None,
        help="Model for Answer phase (env: ANSWER_MODEL, falls back to LLM_MODEL)",
    )
    p.add_argument(
        "--answer-base-url",
        default=None,
        help="Base URL for Answer phase (env: ANSWER_BASE_URL, falls back to LLM_BASE_URL)",
    )
    p.add_argument(
        "--answer-api-key",
        default=None,
        help="API key for Answer phase (env: ANSWER_API_KEY, falls back to LLM_API_KEY)",
    )
    p.add_argument(
        "--judge-model",
        default=None,
        help="Model for Judge phase (env: JUDGE_MODEL, falls back to LLM_MODEL)",
    )
    p.add_argument(
        "--judge-base-url",
        default=None,
        help="Base URL for Judge phase (env: JUDGE_BASE_URL, falls back to LLM_BASE_URL)",
    )
    p.add_argument(
        "--judge-api-key",
        default=None,
        help="API key for Judge phase (env: JUDGE_API_KEY, falls back to LLM_API_KEY)",
    )

    args = p.parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    bad = [m for m in methods if m not in _SUPPORTED_METHODS]
    if bad:
        p.error(f"unsupported method(s): {bad}; supported: {_SUPPORTED_METHODS}")
    args._methods = methods
    return args


# =============================================================================
# Main
# =============================================================================


def main():
    args = parse_args()

    load_dotenv()

    # Resolution: CLI flag > ANSWER_*/JUDGE_* env > LLM_* env > default.
    # Empty strings from getenv fall through via `or`.
    answer_model = (
        args.answer_model
        or os.getenv("ANSWER_MODEL")
        or os.getenv("LLM_MODEL")
        or "gpt-4o-mini"
    )
    answer_base_url = (
        args.answer_base_url
        or os.getenv("ANSWER_BASE_URL")
        or os.getenv("LLM_BASE_URL")
        or "https://api.openai.com/v1"
    )
    # API keys are comma-separated lists; the LLMClientPool round-robins across
    # them and fails over to the next on RateLimitError.
    answer_api_keys = _split_keys(
        args.answer_api_key
        or os.getenv("ANSWER_API_KEY")
        or os.getenv("LLM_API_KEY")
        or ""
    )

    judge_model = (
        args.judge_model
        or os.getenv("JUDGE_MODEL")
        or os.getenv("LLM_MODEL")
        or "gpt-4o-mini"
    )
    judge_base_url = (
        args.judge_base_url
        or os.getenv("JUDGE_BASE_URL")
        or os.getenv("LLM_BASE_URL")
        or "https://api.openai.com/v1"
    )
    judge_api_keys = _split_keys(
        args.judge_api_key
        or os.getenv("JUDGE_API_KEY")
        or os.getenv("LLM_API_KEY")
        or ""
    )

    if not answer_api_keys:
        print(
            "ERROR: no API key for Answer phase. "
            "Set --answer-api-key, ANSWER_API_KEY, or LLM_API_KEY "
            "(comma-separated for key rotation)."
        )
        sys.exit(1)
    if not judge_api_keys:
        print(
            "ERROR: no API key for Judge phase. "
            "Set --judge-api-key, JUDGE_API_KEY, or LLM_API_KEY "
            "(comma-separated for key rotation)."
        )
        sys.exit(1)

    answer_client = LLMClientPool(
        answer_api_keys, base_url=answer_base_url, timeout=60, max_retries=1
    )
    # Reuse the same pool when endpoint + keys match (the common case).
    if answer_base_url == judge_base_url and answer_api_keys == judge_api_keys:
        judge_client = answer_client
    else:
        judge_client = LLMClientPool(
            judge_api_keys, base_url=judge_base_url, timeout=60, max_retries=1
        )

    print(
        f"  Answer LLM: {answer_model} @ {answer_base_url}"
        f" ({answer_client.key_count} keys)"
    )
    print(
        f"  Judge  LLM: {judge_model} @ {judge_base_url}"
        f" ({judge_client.key_count} keys)"
    )

    # 1. Load data (preserve LoCoMo session boundaries)
    print_section("Loading Data")
    sessions, qa_list, spk_a, spk_b = load_conversation(args.data_path, args.conv_index)
    conv_label = f"conv_{args.conv_index} ({spk_a} & {spk_b})"
    total_msgs = sum(len(s["messages"]) for s in sessions)
    print(
        f"  Conversation: {conv_label}\n"
        f"  LoCoMo sessions: {len(sessions)} | Messages: {total_msgs} | "
        f"QA pairs: {len(qa_list)} (excl. category 5)"
    )

    # 2. Init client + pick search owner_id
    client = EverosClient(base_url=args.base_url)
    # Mirror the `_conv{N}` suffix used in run_add_phase so search hits the
    # right partition.
    _speaker = spk_a if args.eval_owner == "speaker_a" else spk_b
    owner_id = f"{_speaker.lower()}_conv{args.conv_index}"
    print(f"  Eval owner: {args.eval_owner} -> owner_id='{owner_id}'")

    # 3. Setup checkpoint dir
    checkpoint_dir = args.checkpoint_dir
    if checkpoint_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_dir = os.path.join(
            "benchmark_checkpoints", f"run_{ts}_conv{args.conv_index}"
        )

    # 4. Add phase
    add_result = None
    if not args.skip_add:
        add_result = run_add_phase(
            client,
            sessions,
            spk_a,
            spk_b,
            args.conv_index,
            args.batch_size,
            quiet=args.quiet,
        )
        if args.post_flush_wait > 0:
            if args.corpus_path:
                _wait_cascade_drain(args.corpus_path, args.post_flush_wait)
            else:
                print(f"  Waiting {args.post_flush_wait}s for async indexing...")
                time.sleep(args.post_flush_wait)
    else:
        print_section("Add Phase (SKIPPED)")
        print(f"  --skip-add: reusing existing data for owner_id='{owner_id}'")

    # 5. For each method: Search -> Answer -> Evaluate
    all_method_results: dict[str, list[dict]] = {}
    all_summaries: dict[str, dict[str, Any]] = {}

    for method in args._methods:
        search_results = run_search_phase(
            client,
            qa_list,
            owner_id,
            method,
            args.top_k,
            quiet=args.quiet,
            concurrency=args.search_concurrency,
        )
        _save_checkpoint(
            checkpoint_dir, f"{method}_1_search.json", "search", search_results
        )

        answer_results = run_answer_phase(
            search_results,
            spk_a,
            spk_b,
            answer_client,
            answer_model,
            quiet=args.quiet,
            concurrency=args.concurrency,
        )
        _save_checkpoint(
            checkpoint_dir, f"{method}_2_answer.json", "answer", answer_results
        )

        eval_results = run_evaluate_phase(
            answer_results,
            judge_client,
            judge_model,
            args.judge_runs,
            quiet=args.quiet,
            concurrency=args.concurrency,
        )
        _save_checkpoint(checkpoint_dir, f"{method}_3_eval.json", "eval", eval_results)

        all_method_results[method] = eval_results
        summary = print_report(eval_results, method, add_result, conv_label, args.top_k)
        all_summaries[method] = summary

    # 6. Comparison table
    if len(all_summaries) > 1:
        print_comparison(all_summaries)

    # 7. Optional JSON export
    if args.output:
        export = {
            "conv_index": args.conv_index,
            "conv_label": conv_label,
            "eval_owner": args.eval_owner,
            "owner_id": owner_id,
            "add_result": add_result,
            "methods": {},
        }
        for method, results in all_method_results.items():
            serializable = []
            for r in results:
                entry = {
                    "index": r["index"],
                    "question": r["question"],
                    "golden_answer": r["golden_answer"],
                    "generated_answer": r["generated_answer"],
                    "category": r["category"],
                    "is_correct": r["is_correct"],
                    "judgments": r["judgments"],
                    "search_time_s": r["search_time_s"],
                    "answer_time_s": r["answer_time_s"],
                    "episode_count": len(r.get("episodes", [])),
                }
                if r.get("search_error"):
                    entry["search_error"] = str(r["search_error"])
                serializable.append(entry)
            export["methods"][method] = {
                "summary": all_summaries[method],
                "details": serializable,
            }

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(export, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()

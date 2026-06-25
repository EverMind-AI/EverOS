"""Reflection E2E test -- validates the full Reflection pipeline with real
LLM, real embedder, and LoCoMo conversation data.

Usage:
  python tests/test_reflection_e2e.py                    # run all TCs
  python tests/test_reflection_e2e.py --tc 1,2,14        # run selected TCs
  python tests/test_reflection_e2e.py --verbose           # verbose output
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

# Path setup for sibling-module import (test_locomo lives in the same dir).
# load_dotenv() is deferred to main() to avoid module-level side effects.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_locomo import (  # noqa: E402
    ANSWER_PROMPT,
    JUDGE_SYSTEM_PROMPT,
    JUDGE_USER_PROMPT,
    EverosClient,
    LLMClientPool,
    _build_context,
    _extract_final_answer,
    _extract_json,
    _parse_session_timestamp,
    print_section,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — session indices per storyline and golden data
# ---------------------------------------------------------------------------

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "locomo10.json"

ADOPTION_INIT_SESSIONS = [2, 8, 13, 17]
ADOPTION_UPDATE_SESSIONS = [19]
LGBTQ_INIT_SESSIONS = [1, 3, 5, 12]
LGBTQ_UPDATE_SESSIONS = [14]
PET_INIT_SESSIONS = [1, 5, 12, 24]
PET_UPDATE_SESSIONS = [27, 28]
HEALTH_INIT_SESSIONS = [2, 4, 8, 10, 13, 14]
HEALTH_UPDATE_SESSIONS = [16, 20]

QUERIES = {
    "adoption": "What steps has Caroline taken toward adoption?",
    "lgbtq": "How has Caroline dealt with discrimination?",
    "pet": "How many pets does Andrew have and what are their names?",
    "health": "How has Sam's diet and health journey been going?",
}

GOLDEN_FACTS = {
    "adoption": [
        "research",
        "adoption council",
        "applied",
        "mentor",
        "interview",
    ],
    "lgbtq": [
        "support group",
        "school",
        "pride",
        "discriminat",
        "apolog",
    ],
    "pet": [
        "no pet",
        "toby",
        "buddy",
        "scout",
    ],
    "health": [
        "doctor",
        "diet",
        "before and after",
        "snack",
        "gastritis",
        "weight watchers",
        "struggl",
    ],
}

# ---------------------------------------------------------------------------
# Data parsing
# ---------------------------------------------------------------------------


def load_locomo() -> list[dict[str, Any]]:
    """Load the LoCoMo dataset from the project data directory."""
    with open(DATA_PATH) as f:
        return json.load(f)


def parse_sessions(
    conv: dict[str, Any],
    session_indices: list[int],
    conv_index: int,
) -> list[dict[str, Any]]:
    """Parse LoCoMo sessions into the everos /add message format.

    Returns a list of dicts, each with ``session_idx``, ``session_id``,
    and ``messages`` (ready for the ``/api/v1/memory/add`` payload).
    """
    raw = conv["conversation"]
    results: list[dict[str, Any]] = []
    for idx in session_indices:
        key = f"session_{idx}"
        if key not in raw:
            raise ValueError(f"session {key} not found in conv {conv_index}")
        date_key = f"{key}_date_time"
        base_ts = _parse_session_timestamp(raw.get(date_key, ""))
        session_id = f"refl_conv{conv_index}_s{idx}"
        messages: list[dict[str, Any]] = []
        for i, dia in enumerate(raw[key]):
            messages.append(
                {
                    "sender_id": f"{dia['speaker'].lower()}_conv{conv_index}",
                    "sender_name": dia["speaker"],
                    "role": "user",
                    "timestamp": base_ts + i * 30,
                    "content": [{"type": "text", "text": dia["text"]}],
                }
            )
        results.append(
            {
                "session_idx": idx,
                "session_id": session_id,
                "messages": messages,
            }
        )
    return results


# ---------------------------------------------------------------------------
# Infrastructure helpers
# ---------------------------------------------------------------------------


_SYSTEM_DB = DATA_PATH.parent.parent / ".everos" / ".index" / "sqlite" / "system.db"


def print_episode_locations(
    owner_id: str,
    episodes: list[dict[str, Any]],
) -> None:
    """Print md paths for human review of merged vs source episodes."""
    merged = [e for e in episodes if e.get("session_id") is None]
    original = [e for e in episodes if e.get("session_id") is not None]
    print(f"\n  episode locations ({owner_id}):")
    if merged:
        for ep in merged:
            print(f"    [MERGED] {ep.get('id', '?')}")
    if original:
        for ep in original[:3]:
            print(f"    [source] {ep.get('id', '?')} session={ep.get('session_id')}")
        if len(original) > 3:
            print(f"    ... and {len(original) - 3} more sources")
    root = str(DATA_PATH.parent.parent / ".everos")
    print(f"    md root: {root}")


def _owner_id(speaker: str, conv_index: int) -> str:
    """Build the canonical owner_id for a speaker in a conversation."""
    return f"{speaker.lower()}_conv{conv_index}"


def count_reflection_reports(owner_id: str) -> int:
    """Query SQLite directly to count reflection reports for an owner."""
    import sqlite3

    conn = sqlite3.connect(str(_SYSTEM_DB))
    try:
        cur = conn.execute(
            "SELECT count(*) FROM reflection_report WHERE owner_id = ?",
            (owner_id,),
        )
        return cur.fetchone()[0]
    finally:
        conn.close()


def count_deprecated_episodes(owner_id: str) -> int:
    """Query LanceDB via search with a special filter is not possible from
    outside the server. Instead check reflection_report source_count as proxy."""
    import sqlite3

    conn = sqlite3.connect(str(_SYSTEM_DB))
    try:
        cur = conn.execute(
            "SELECT coalesce(sum(source_count), 0) "
            "FROM reflection_report WHERE owner_id = ?",
            (owner_id,),
        )
        return cur.fetchone()[0]
    finally:
        conn.close()


def add_and_flush(
    client: EverosClient,
    sessions: list[dict[str, Any]],
    *,
    quiet: bool = True,
) -> None:
    """Ingest sessions: /add all messages first, then /flush each session."""
    for sess in sessions:
        payload = {"session_id": sess["session_id"], "messages": sess["messages"]}
        status, _ = client.post("/api/v1/memory/add", payload, quiet=quiet)
        assert status == 200, f"add failed for {sess['session_id']}: {status}"

    for sess in sessions:
        status, _ = client.post(
            "/api/v1/memory/flush",
            {"session_id": sess["session_id"]},
            quiet=quiet,
        )
        assert status == 200, f"flush failed for {sess['session_id']}: {status}"


def wait_pipeline(seconds: int = 180) -> None:
    """Wait for cascade + OME pipeline to settle after flush."""
    print(f"  waiting {seconds}s for pipeline to settle...")
    time.sleep(seconds)  # tz-noqa — wall-clock delay, not a datetime
    print("  pipeline wait done")


def trigger_reflection(
    client: EverosClient,
    *,
    timeout: float = 120.0,
) -> None:
    """Trigger Reflection via HTTP endpoint on the running server."""
    print("  triggering reflection via HTTP...")
    status, resp = client.post(
        "/api/v1/ome/trigger",
        {"name": "reflect_episodes", "timeout": timeout, "force": True},
        quiet=True,
    )
    result_status = resp.get("status", "unknown") if isinstance(resp, dict) else "error"
    print(f"  trigger response: status={result_status}")
    if status != 200 or result_status != "ok":
        raise RuntimeError(f"reflection trigger failed: HTTP {status}, {resp}")


def search_episodes(
    client: EverosClient,
    query: str,
    owner_id: str,
    *,
    method: str = "hybrid",
    top_k: int = 10,
) -> dict[str, Any]:
    """Run a memory search and return the ``data`` payload."""
    payload = {
        "query": query,
        "method": method,
        "top_k": top_k,
        "user_id": owner_id,
    }
    status, resp = client.post("/api/v1/memory/search", payload, quiet=True)
    assert status == 200, f"search failed: {status}"
    return resp.get("data", {})


def answer_and_judge(
    query: str,
    search_data: dict[str, Any],
    golden_answer: str,
    *,
    speaker_a: str,
    speaker_b: str,
    llm_client: LLMClientPool,
    llm_model: str,
) -> dict[str, Any]:
    """Generate an answer from search results and judge correctness.

    Returns a dict with ``answer``, ``judge_score`` (0 or 1), and
    ``episodes_count``.
    """
    context = _build_context(
        search_data.get("episodes", []),
        search_data.get("profiles", []),
        speaker_a,
        speaker_b,
    )
    prompt = ANSWER_PROMPT.format(context=context, question=query)
    try:
        resp = llm_client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        answer = _extract_final_answer(resp.choices[0].message.content or "")
    except Exception as e:
        answer = f"[error: {e}]"

    try:
        judge_resp = llm_client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": JUDGE_USER_PROMPT.format(
                        question=query,
                        golden_answer=golden_answer,
                        generated_answer=answer,
                    ),
                },
            ],
            temperature=0.0,
        )
        judge_text = judge_resp.choices[0].message.content or ""
        raw_json = _extract_json(judge_text)
        if raw_json:
            parsed = json.loads(raw_json)
            is_correct = parsed.get("label", "").upper() == "CORRECT"
        else:
            is_correct = False
    except Exception:
        logger.warning("judge evaluation failed", exc_info=True)
        is_correct = False

    return {
        "answer": answer,
        "judge_score": 1 if is_correct else 0,
        "episodes_count": len(search_data.get("episodes", [])),
    }


def compute_fact_coverage(text: str, facts: list[str]) -> float:
    """Compute fraction of golden facts found (case-insensitive substring)."""
    text_lower = text.lower()
    hits = sum(1 for f in facts if f.lower() in text_lower)
    return hits / len(facts) if facts else 0.0


# ---------------------------------------------------------------------------
# TCResult — lightweight per-test-case assertion tracker
# ---------------------------------------------------------------------------


class TCResult:
    """Accumulate pass/fail checks for a single test case."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.passed: list[str] = []
        self.failed: list[str] = []

    def check(self, condition: bool, description: str) -> None:
        (self.passed if condition else self.failed).append(description)

    @property
    def ok(self) -> bool:
        return len(self.failed) == 0

    def print_summary(self) -> None:
        status = "PASS" if self.ok else "FAIL"
        print(f"\n  {self.name}: {status}")
        for p in self.passed:
            print(f"    [ok] {p}")
        for f in self.failed:
            print(f"    [FAIL] {f}")


# ---------------------------------------------------------------------------
# Test cases (TC1-TC8) — INIT + UPDATE per storyline
# ---------------------------------------------------------------------------


def tc1_adoption_init(client: EverosClient) -> TCResult:
    tc = TCResult("TC1: Adoption INIT")
    print_section("TC1: Adoption INIT (conv0, sessions 2,8,13,17)")
    owner = _owner_id("caroline", 0)
    data = load_locomo()
    sessions = parse_sessions(data[0], ADOPTION_INIT_SESSIONS, 0)
    add_and_flush(client, sessions)
    wait_pipeline()
    trigger_reflection(client)
    # Positive: reflection report was written (deprecation completed)
    reports = count_reflection_reports(owner)
    tc.check(reports >= 1, f"reflection report created ({reports} found)")
    dep_count = count_deprecated_episodes(owner)
    tc.check(dep_count >= 1, f"source episodes deprecated ({dep_count} source_count)")
    # Search: merged episode visible, deprecated filtered out
    result = search_episodes(client, QUERIES["adoption"], owner)
    episodes = result.get("episodes", [])
    tc.check(len(episodes) > 0, "search returns episodes")
    merged = [e for e in episodes if e.get("session_id") is None]
    tc.check(len(merged) >= 1, "merged episode exists (session_id=None)")
    print_episode_locations(owner, episodes)
    tc.print_summary()
    return tc


def tc2_adoption_update(client: EverosClient) -> TCResult:
    tc = TCResult("TC2: Adoption UPDATE")
    print_section("TC2: Adoption UPDATE (conv0, session 19)")
    owner = _owner_id("caroline", 0)
    reports_before = count_reflection_reports(owner)
    data = load_locomo()
    sessions = parse_sessions(data[0], ADOPTION_UPDATE_SESSIONS, 0)
    add_and_flush(client, sessions)
    wait_pipeline()
    trigger_reflection(client)
    reports_after = count_reflection_reports(owner)
    tc.check(
        reports_after > reports_before,
        f"report count up ({reports_before}->{reports_after})",
    )
    result = search_episodes(client, QUERIES["adoption"], owner)
    merged = [e for e in result.get("episodes", []) if e.get("session_id") is None]
    tc.check(len(merged) >= 1, "merged episode exists after update")
    print_episode_locations(owner, result.get("episodes", []))
    tc.print_summary()
    return tc


def tc3_lgbtq_init(client: EverosClient) -> TCResult:
    tc = TCResult("TC3: LGBTQ+Conflict INIT")
    print_section("TC3: LGBTQ+Conflict INIT (conv0, sessions 1,3,5,12)")
    owner = _owner_id("caroline", 0)
    data = load_locomo()
    sessions = parse_sessions(data[0], LGBTQ_INIT_SESSIONS, 0)
    add_and_flush(client, sessions)
    wait_pipeline()
    trigger_reflection(client)
    reports = count_reflection_reports(owner)
    tc.check(reports >= 1, f"reflection report(s) exist ({reports})")
    result = search_episodes(client, QUERIES["lgbtq"], owner)
    merged = [e for e in result.get("episodes", []) if e.get("session_id") is None]
    tc.check(len(merged) >= 1, "merged episode exists")
    print_episode_locations(owner, result.get("episodes", []))
    tc.print_summary()
    return tc


def tc4_lgbtq_update(client: EverosClient) -> TCResult:
    tc = TCResult("TC4: LGBTQ+Conflict UPDATE")
    print_section("TC4: LGBTQ+Conflict UPDATE (conv0, session 14)")
    owner = _owner_id("caroline", 0)
    reports_before = count_reflection_reports(owner)
    data = load_locomo()
    sessions = parse_sessions(data[0], LGBTQ_UPDATE_SESSIONS, 0)
    add_and_flush(client, sessions)
    wait_pipeline()
    trigger_reflection(client)
    reports_after = count_reflection_reports(owner)
    tc.check(
        reports_after > reports_before,
        f"report count up ({reports_before}->{reports_after})",
    )
    result = search_episodes(client, QUERIES["lgbtq"], owner)
    merged = [e for e in result.get("episodes", []) if e.get("session_id") is None]
    tc.check(len(merged) >= 1, "merged episode exists after update")
    print_episode_locations(owner, result.get("episodes", []))
    tc.print_summary()
    return tc


def tc5_pet_init(client: EverosClient) -> TCResult:
    tc = TCResult("TC5: Pet Count INIT")
    print_section("TC5: Pet Count INIT (conv5, sessions 1,5,12,24)")
    owner = _owner_id("andrew", 5)
    data = load_locomo()
    sessions = parse_sessions(data[5], PET_INIT_SESSIONS, 5)
    add_and_flush(client, sessions)
    wait_pipeline()
    trigger_reflection(client)
    reports = count_reflection_reports(owner)
    tc.check(reports >= 1, f"reflection report created ({reports})")
    result = search_episodes(client, QUERIES["pet"], owner)
    merged = [e for e in result.get("episodes", []) if e.get("session_id") is None]
    tc.check(len(merged) >= 1, "merged episode exists")
    print_episode_locations(owner, result.get("episodes", []))
    tc.print_summary()
    return tc


def tc6_pet_update(client: EverosClient) -> TCResult:
    tc = TCResult("TC6: Pet Count UPDATE")
    print_section("TC6: Pet Count UPDATE (conv5, sessions 27,28)")
    owner = _owner_id("andrew", 5)
    reports_before = count_reflection_reports(owner)
    data = load_locomo()
    sessions = parse_sessions(data[5], PET_UPDATE_SESSIONS, 5)
    add_and_flush(client, sessions)
    wait_pipeline()
    trigger_reflection(client)
    reports_after = count_reflection_reports(owner)
    tc.check(
        reports_after > reports_before,
        f"report count up ({reports_before}->{reports_after})",
    )
    result = search_episodes(client, QUERIES["pet"], owner)
    merged = [e for e in result.get("episodes", []) if e.get("session_id") is None]
    tc.check(len(merged) >= 1, "merged episode exists after update")
    print_episode_locations(owner, result.get("episodes", []))
    tc.print_summary()
    return tc


def tc7_health_init(client: EverosClient) -> TCResult:
    tc = TCResult("TC7: Health Relapse INIT")
    print_section("TC7: Health INIT (conv8, sessions 2,4,8,10,13,14)")
    owner = _owner_id("sam", 8)
    data = load_locomo()
    sessions = parse_sessions(data[8], HEALTH_INIT_SESSIONS, 8)
    add_and_flush(client, sessions)
    wait_pipeline()
    trigger_reflection(client)
    reports = count_reflection_reports(owner)
    tc.check(reports >= 1, f"reflection report created ({reports})")
    result = search_episodes(client, QUERIES["health"], owner)
    merged = [e for e in result.get("episodes", []) if e.get("session_id") is None]
    tc.check(len(merged) >= 1, "merged episode exists")
    print_episode_locations(owner, result.get("episodes", []))
    tc.print_summary()
    return tc


def tc8_health_update(client: EverosClient) -> TCResult:
    tc = TCResult("TC8: Health Relapse UPDATE")
    print_section("TC8: Health UPDATE (conv8, sessions 16,20)")
    owner = _owner_id("sam", 8)
    reports_before = count_reflection_reports(owner)
    data = load_locomo()
    sessions = parse_sessions(data[8], HEALTH_UPDATE_SESSIONS, 8)
    add_and_flush(client, sessions)
    wait_pipeline()
    trigger_reflection(client)
    reports_after = count_reflection_reports(owner)
    tc.check(
        reports_after > reports_before,
        f"report count up ({reports_before}->{reports_after})",
    )
    result = search_episodes(client, QUERIES["health"], owner)
    merged = [e for e in result.get("episodes", []) if e.get("session_id") is None]
    tc.check(len(merged) >= 1, "merged episode exists after update")
    print_episode_locations(owner, result.get("episodes", []))
    tc.print_summary()
    return tc


# ---------------------------------------------------------------------------
# Test cases (TC9, TC11-TC14) — cross-cutting validation
# ---------------------------------------------------------------------------


def tc9_search_visibility(client: EverosClient) -> TCResult:
    tc = TCResult("TC9: Search Visibility")
    print_section("TC9: Search Visibility")
    checks = [
        (QUERIES["adoption"], _owner_id("caroline", 0)),
        (QUERIES["lgbtq"], _owner_id("caroline", 0)),
        (QUERIES["pet"], _owner_id("andrew", 5)),
        (QUERIES["health"], _owner_id("sam", 8)),
    ]
    for query, owner in checks:
        # Positive: deprecation actually happened
        reports = count_reflection_reports(owner)
        tc.check(reports >= 1, f"reports exist for {owner} ({reports})")
        # Search: merged visible, deprecated filtered
        data = search_episodes(client, query, owner)
        episodes = data.get("episodes", [])
        merged = [e for e in episodes if e.get("session_id") is None]
        tc.check(len(merged) >= 1, f"merged present for '{query[:40]}...'")
    tc.print_summary()
    return tc


def tc11_idempotency(client: EverosClient) -> TCResult:
    tc = TCResult("TC11: Idempotency")
    print_section("TC11: Idempotency")
    owner = _owner_id("caroline", 0)
    before = search_episodes(client, QUERIES["adoption"], owner)
    merged_before = [
        e for e in before.get("episodes", []) if e.get("session_id") is None
    ]
    count_before = len(merged_before)
    trigger_reflection(client)
    after = search_episodes(client, QUERIES["adoption"], owner)
    merged_after = [e for e in after.get("episodes", []) if e.get("session_id") is None]
    tc.check(
        len(merged_after) == count_before,
        f"merged count unchanged ({count_before} -> {len(merged_after)})",
    )
    if merged_before and merged_after:
        tc.check(
            merged_before[0].get("id") == merged_after[0].get("id"),
            "same merged episode ID (no duplicate)",
        )
    tc.print_summary()
    return tc


def tc12_atomic_facts(client: EverosClient) -> TCResult:
    tc = TCResult("TC12: Atomic Facts Re-extraction")
    print_section("TC12: Atomic Facts Re-extraction")
    owner = _owner_id("caroline", 0)
    result = search_episodes(client, QUERIES["adoption"], owner)
    merged = [e for e in result.get("episodes", []) if e.get("session_id") is None]
    tc.check(len(merged) >= 1, "merged episode exists")
    if merged:
        facts = merged[0].get("atomic_facts", [])
        tc.check(len(facts) > 0, f"merged has atomic facts ({len(facts)} found)")
    tc.print_summary()
    return tc


def tc13_topic_isolation(client: EverosClient) -> TCResult:
    tc = TCResult("TC13: Cross-topic Isolation")
    print_section("TC13: Cross-topic Isolation")
    owner = _owner_id("caroline", 0)
    adoption = search_episodes(client, QUERIES["adoption"], owner)
    lgbtq = search_episodes(client, QUERIES["lgbtq"], owner)
    a_merged = [e for e in adoption.get("episodes", []) if e.get("session_id") is None]
    l_merged = [e for e in lgbtq.get("episodes", []) if e.get("session_id") is None]
    tc.check(len(a_merged) >= 1, "adoption has merged episode")
    tc.check(len(l_merged) >= 1, "lgbtq has merged episode")
    if a_merged and l_merged:
        tc.check(
            a_merged[0].get("id") != l_merged[0].get("id"),
            "different merged episode IDs",
        )
        a_text = a_merged[0].get("episode", "").lower()
        l_text = l_merged[0].get("episode", "").lower()
        tc.check(
            "discriminat" not in a_text and "hike" not in a_text,
            "adoption text has no discrimination content",
        )
        tc.check(
            "agenc" not in l_text and "adoption council" not in l_text,
            "lgbtq text has no adoption process content",
        )
    tc.print_summary()
    return tc


def tc14_answer_judge(
    client: EverosClient,
    llm_client: LLMClientPool,
    llm_model: str,
) -> TCResult:
    tc = TCResult("TC14: Answer+Judge Quality")
    print_section("TC14: Answer+Judge Quality Comparison")

    golden_answers = {
        "adoption": (
            "Caroline researched adoption agencies, attended an adoption council "
            "meeting, applied to multiple agencies, contacted her mentor for advice, "
            "and passed the adoption agency interviews."
        ),
        "lgbtq": (
            "Caroline dealt with discrimination by attending LGBTQ support groups, "
            "speaking at her school, participating in a Pride parade. When she "
            "encountered discrimination on a hike from religious conservatives, "
            "she later wrote an apology letter to reconcile."
        ),
        "pet": (
            "Andrew has three dogs: Toby, Buddy, and Scout. He initially had no "
            "pets, then adopted Toby, followed by Buddy from a shelter, and most "
            "recently Scout."
        ),
        "health": (
            "Sam's journey has been non-linear. After a doctor warned about his "
            "weight, he started dieting with good results. But he relapsed by "
            "buying unhealthy snacks, then had a gastritis emergency. He recovered "
            "to become a Weight Watchers coach, but later struggled again."
        ),
    }
    owner_map = {
        "adoption": (_owner_id("caroline", 0), "Caroline", "Melanie"),
        "lgbtq": (_owner_id("caroline", 0), "Caroline", "Melanie"),
        "pet": (_owner_id("andrew", 5), "Audrey", "Andrew"),
        "health": (_owner_id("sam", 8), "Evan", "Sam"),
    }

    total_score = 0
    for topic, query in QUERIES.items():
        owner, speaker_a, speaker_b = owner_map[topic]
        data = search_episodes(client, query, owner)
        result = answer_and_judge(
            query,
            data,
            golden_answers[topic],
            speaker_a=speaker_a,
            speaker_b=speaker_b,
            llm_client=llm_client,
            llm_model=llm_model,
        )
        score = result["judge_score"]
        total_score += score
        merged = [e for e in data.get("episodes", []) if e.get("session_id") is None]
        merged_text = merged[0].get("episode", "") if merged else ""
        coverage = compute_fact_coverage(merged_text, GOLDEN_FACTS[topic])
        status = "CORRECT" if score else "WRONG"
        print(f"  {topic}: {status} | fact_coverage={coverage:.0%}")
        print(f"    answer: {result['answer'][:120]}...")
        tc.check(score == 1, f"{topic} answered correctly")

    print(f"\n  Overall: {total_score}/{len(QUERIES)}")
    tc.print_summary()
    return tc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

TC_REGISTRY: dict[int, tuple[str, Any]] = {
    1: ("Adoption INIT", lambda c, **kw: tc1_adoption_init(c)),
    2: ("Adoption UPDATE", lambda c, **kw: tc2_adoption_update(c)),
    3: ("LGBTQ INIT", lambda c, **kw: tc3_lgbtq_init(c)),
    4: ("LGBTQ UPDATE", lambda c, **kw: tc4_lgbtq_update(c)),
    5: ("Pet INIT", lambda c, **kw: tc5_pet_init(c)),
    6: ("Pet UPDATE", lambda c, **kw: tc6_pet_update(c)),
    7: ("Health INIT", lambda c, **kw: tc7_health_init(c)),
    8: ("Health UPDATE", lambda c, **kw: tc8_health_update(c)),
    9: ("Search Visibility", lambda c, **kw: tc9_search_visibility(c)),
    # TC10 removed: was a duplicate of TC9 visibility checks.
    11: ("Idempotency", lambda c, **kw: tc11_idempotency(c)),
    12: ("Atomic Facts", lambda c, **kw: tc12_atomic_facts(c)),
    13: ("Topic Isolation", lambda c, **kw: tc13_topic_isolation(c)),
    14: (
        "Answer+Judge",
        lambda c, **kw: tc14_answer_judge(c, kw["llm_client"], kw["llm_model"]),
    ),
}


def main() -> None:
    import os

    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(description="Reflection E2E Test")
    parser.add_argument(
        "--tc",
        type=str,
        default=None,
        help="Comma-separated TC numbers (e.g. '1,2,14'). Default: all.",
    )
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    tc_ids = (
        [int(x.strip()) for x in args.tc.split(",")]
        if args.tc
        else sorted(TC_REGISTRY.keys())
    )

    client = EverosClient(base_url=args.base_url)
    llm_model = args.llm_model or os.getenv("EVEROS_LLM__MODEL", "openai/gpt-4.1-mini")
    api_key = os.getenv("EVEROS_LLM__API_KEY", "")
    base_url = os.getenv("EVEROS_LLM__BASE_URL", "https://openrouter.ai/api/v1")
    llm_client = LLMClientPool(api_keys=[api_key], base_url=base_url)

    print_section("Reflection E2E Test")
    print(f"  TCs: {tc_ids}")
    print(f"  Server: {args.base_url}")
    print(f"  LLM: {llm_model}")

    results: list[TCResult] = []
    for tc_id in tc_ids:
        if tc_id not in TC_REGISTRY:
            print(f"  WARNING: TC{tc_id} not found, skipping")
            continue
        name, func = TC_REGISTRY[tc_id]
        try:
            r = func(client, llm_client=llm_client, llm_model=llm_model)
            results.append(r)
        except Exception as e:
            print(f"\n  TC{tc_id} ({name}) CRASHED: {e}")
            tc = TCResult(f"TC{tc_id}: {name}")
            tc.check(False, f"crashed: {e}")
            results.append(tc)

    print_section("SUMMARY")
    passed = sum(1 for r in results if r.ok)
    for r in results:
        status = "PASS" if r.ok else "FAIL"
        checks = f"{len(r.passed)}/{len(r.passed) + len(r.failed)}"
        print(f"  {status} {r.name} ({checks} checks)")
    print(f"\n  Total: {passed}/{len(results)} TCs passed")
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()

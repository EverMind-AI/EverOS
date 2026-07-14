"""End-to-end demo: EverOS memory operations traced into Langfuse.

Replays one realistic memory lifecycle — ingest -> extraction -> recall (with
an updated fact winning over a stale one) -> agent-skill recall -> reflection —
through the instrumentation in everos_langfuse.py.

Two modes, same code path:
  * offline (default)  — spans land in ./spans.jsonl for offline inspection
  * live               — set LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY /
                         LANGFUSE_HOST and the exact same spans + recall
                         scores also flow into your Langfuse project.

The MockEverOSTransport returns responses in the exact envelope/shape of the
EverOS HTTP API v1 (see EverOS docs/api.md); swap in HTTPTransport to run
against a real `pip install everos` server — the instrumentation is identical.
"""

from __future__ import annotations

import time
import uuid

from everos_langfuse import HTTPTransport, InstrumentedEverOS, force_flush, init_tracing

TS = int(time.time() * 1000)
DAY = "20260702"


def _envelope(data: dict, detail: dict | None = None) -> dict:
    resp = {"request_id": uuid.uuid4().hex, "data": data}
    if detail:
        resp["_detail"] = detail  # server-side facts the spans describe
    return resp


class MockEverOSTransport:
    """Faithful mock of the EverOS HTTP API v1 (response envelope + field
    shapes from docs/api.md), so the demo runs without provider keys."""

    def __init__(self):
        self.buffer: list[dict] = []

    def __call__(self, path: str, payload: dict) -> dict:
        if path == "/api/v1/memory/add":
            self.buffer.extend(payload["messages"])
            time.sleep(0.012)
            return _envelope({"message_count": len(payload["messages"]),
                              "status": "accumulated"})

        if path == "/api/v1/memory/flush":
            buffered, self.buffer = self.buffer, []
            time.sleep(0.01)
            return _envelope(
                {"status": "extracted"},
                detail={
                    "model": "gpt-4.1-mini",
                    "buffered_messages": [m["content"] for m in buffered],
                    "memory_cell": {
                        "episode_id": "alice_ep_%s_001" % DAY,
                        "subject": "Alice's routines and recent move",
                        "summary": ("Alice climbs in Yosemite every spring, bikes to "
                                    "work, and recently moved from SOMA to Oakland; "
                                    "her go-to coffee used to be Blue Bottle in SOMA."),
                        "atomic_facts": [
                            "Alice climbs in Yosemite every spring.",
                            "Alice bikes to work most days.",
                            "Alice moved from SOMA to Oakland in June 2026.",
                            "Alice's favorite coffee shop was Blue Bottle in SOMA.",
                        ],
                    },
                    "usage": {"input": 642, "output": 187},
                    "md_files": ["memory/alice/episodic/2026-07-02-alice-routines.md"],
                    "rows_indexed": 5,
                    "index_lag_ms": 512,
                    "extract_s": 0.42,
                },
            )

        if path == "/api/v1/memory/search":
            q = payload["query"].lower()
            if "live" in q:  # conflict-resolution showcase: fresh fact outranks stale
                ranked = [
                    {"id": "alice_af_%s_003" % DAY,
                     "content": "Alice moved from SOMA to Oakland in June 2026.",
                     "score": 0.81},
                    {"id": "alice_af_%s_004" % DAY,
                     "content": "Alice's favorite coffee shop was Blue Bottle in SOMA.",
                     "score": 0.34},
                ]
            elif "sport" in q or "outdoor" in q:
                ranked = [
                    {"id": "alice_af_%s_001" % DAY,
                     "content": "Alice climbs in Yosemite every spring.", "score": 0.86},
                    {"id": "alice_af_%s_002" % DAY,
                     "content": "Alice bikes to work most days.", "score": 0.72},
                ]
            elif payload.get("agent_id"):  # agent track: cases + skills
                ranked = [
                    {"id": "raven_case_%s_007" % DAY,
                     "content": "Case: flaky LanceDB test fixed by pinning fsync "
                                "before rename and retrying open with backoff.",
                     "score": 0.74},
                    {"id": "raven_skill_retry_backoff",
                     "content": "Skill: wrap flaky IO in retry-with-backoff; verify "
                                "with 3 consecutive green runs.",
                     "score": 0.69},
                ]
            else:  # deliberate miss: query about something never stored
                ranked = [
                    {"id": "alice_af_%s_002" % DAY,
                     "content": "Alice bikes to work most days.", "score": 0.31},
                ]

            time.sleep(0.01)
            if payload.get("agent_id"):
                data = {"episodes": [], "profiles": [],
                        "agent_cases": [r for r in ranked if "case" in r["id"]],
                        "agent_skills": [r for r in ranked if "skill" in r["id"]],
                        "unprocessed_messages": []}
            else:
                data = {"episodes": [{
                            "id": "alice_ep_%s_001" % DAY,
                            "user_id": payload.get("user_id"),
                            "session_id": "sess-cafe-chat-001",
                            "summary": "Alice's routines and recent move",
                            "score": ranked[0]["score"],
                            "atomic_facts": ranked,
                        }],
                        "profiles": [], "agent_cases": [], "agent_skills": [],
                        "unprocessed_messages": []}
            return _envelope(data, detail={
                "embed_model": "Qwen/Qwen3-Embedding-4B", "embed_tokens": 11,
                "rerank_model": "Qwen/Qwen3-Reranker-4B",
                "candidates": 24, "ranked": ranked,
                "embed_s": 0.028, "recall_s": 0.019, "rerank_s": 0.047,
            })

        if path == "/api/v1/ome/trigger":
            time.sleep(0.01)
            return _envelope(
                {"status": "ok", "name": payload["name"]},
                detail={
                    "model": "gpt-4.1-mini",
                    "episodes_in": ["alice_ep_%s_001" % DAY],
                    "consolidated": {
                        "profile_update": "home_location: SOMA -> Oakland (2026-06)",
                        "episodes_merged": 1,
                    },
                    "usage": {"input": 918, "output": 141},
                    "reflect_s": 0.31,
                },
            )

        raise ValueError(f"unknown path {path}")


def main() -> None:
    live = init_tracing(service_name="everos", spans_jsonl="spans.jsonl")
    print(f"[demo] tracing initialised — live Langfuse export: {live}")

    import os
    if os.getenv("EVEROS_BASE_URL"):
        transport = HTTPTransport(os.environ["EVEROS_BASE_URL"])
        print(f"[demo] using real EverOS server at {os.environ['EVEROS_BASE_URL']}")
    else:
        transport = MockEverOSTransport()
        print("[demo] using MockEverOSTransport (EverOS HTTP API v1 shapes)")

    # public_traces=True: demo data is synthetic (fictional "Alice"), so the
    # resulting traces are safe to share as public Langfuse trace URLs.
    ev = InstrumentedEverOS(transport, public_traces=True)
    session, user = "sess-cafe-chat-001", "alice"

    # -- 1. write path: ingest a conversation ------------------------------
    ev.add(session, [
        {"sender_id": user, "role": "user", "timestamp": TS,
         "content": "I love climbing in Yosemite every spring."},
        {"sender_id": user, "role": "user", "timestamp": TS + 10,
         "content": "My favorite coffee shop is Blue Bottle in SOMA."},
        {"sender_id": user, "role": "user", "timestamp": TS + 20,
         "content": "I bike to work most days."},
    ], user_id=user)
    ev.add(session, [
        {"sender_id": user, "role": "user", "timestamp": TS + 30,
         "content": "Oh — actually I moved from SOMA to Oakland last month."},
    ], user_id=user)

    # -- 2. boundary/flush: LLM extraction -> markdown -> index ------------
    ev.flush(session, user_id=user)

    # -- 3. read path: recall with quality scores ---------------------------
    r1 = ev.search("What outdoor sports does Alice do?", user_id=user,
                   session_id=session)
    r2 = ev.search("Where does Alice live now?", user_id=user, session_id=session)
    r3 = ev.search("What are Alice's favorite books?", user_id=user,
                   session_id=session)  # deliberate low-quality recall
    # agent-memory track (cases / skills) — the Raven angle
    r4 = ev.search("How did we fix the flaky LanceDB test last time?",
                   agent_id="raven-dev-agent", session_id="raven-run-042")

    # -- 4. self-evolution: offline reflection ------------------------------
    ev.trigger_ome("reflect_episodes", user_id=user, session_id=session)

    force_flush()
    time.sleep(0.5)

    print("\n[demo] traces emitted:")
    for label, r in [("recall: sports", r1), ("recall: moved city", r2),
                     ("recall: miss (books)", r3), ("recall: agent skill", r4)]:
        print(f"  - {label:24s} trace_id={r['_trace_id']} "
              f"scores_pushed={r['_scores_pushed']}")
    print("\n[demo] spans also written to spans.jsonl (offline copy)")
    if live:
        print("[demo] open your Langfuse project -> Traces; "
              "scores 'recall_top_score' / 'recall_hit' attached to searches.")


if __name__ == "__main__":
    main()

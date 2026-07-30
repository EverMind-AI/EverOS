"""EverOS x Langfuse demo — native OpenTelemetry tracing.

EverOS emits OTel spans for its own memory operations when ``[observability]``
is enabled; this script contains **no instrumentation code**. It drives a
running server through a memory lifecycle worth looking at in Langfuse.

Eleven short conversations spread over ten weeks, each on its own topic, so
recall has to pick the right memory out of a populated store rather than
returning the only thing in it. Two revisit the same subject days apart (an
October trip moves from Lisbon to Porto), close enough that geometry
clustering groups them, which gives reflection something to consolidate. One
query asks about something never discussed, so a miss looks like a miss.

Prereqs (see README.md):
  1. pip install "everos[otel]"
  2. configure [observability] in everos.toml with your Langfuse keys
  3. everos server start        # defaults to http://127.0.0.1:8000

Then: python demo.py
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8000"
USER = "alice"
DAY_MS = 86_400_000

# Extraction and the SQLite -> LanceDB index sync run asynchronously, so how
# long a memory takes to become searchable depends on the LLM behind it.
INDEX_TIMEOUT_SECONDS = 300.0
# Deliberately slack: every probe is itself a traced search, and polling hard
# would bury the five real questions under a wall of readiness checks.
INDEX_POLL_SECONDS = 10.0
CONSOLIDATION_TIMEOUT_SECONDS = 180.0

# ── the conversations ────────────────────────────────────────────────────
# ``days_ago`` only spaces the timestamps out; every session is ingested now.
SESSIONS: list[dict] = [
    {
        "id": "everos-demo-trip-booked",
        "days_ago": 70,
        "messages": [
            {
                "role": "user",
                "content": (
                    "We booked the October trip: Lisbon, a week, flying out on the "
                    "12th."
                ),
            },
            {
                "role": "assistant",
                "content": "Noted, a week in Lisbon in October departing on the 12th.",
            },
        ],
    },
    {
        "id": "everos-demo-dentist",
        "days_ago": 63,
        "messages": [
            {
                "role": "user",
                "content": (
                    "The dentist put a crown on my lower left molar today. Check-up in "
                    "six months."
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "Recorded the crown on your lower left molar, with a check-up due "
                    "in six months."
                ),
            },
        ],
    },
    {
        "id": "everos-demo-trip-changed",
        "days_ago": 65,
        "messages": [
            {
                "role": "user",
                "content": (
                    "October changed. We cancelled Lisbon and booked Porto instead, my "
                    "sister moved the wedding there."
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "Updated, the October trip is Porto now rather than Lisbon, "
                    "because the wedding moved."
                ),
            },
        ],
    },
    {
        "id": "everos-demo-cello",
        "days_ago": 49,
        "messages": [
            {
                "role": "user",
                "content": (
                    "I started cello lessons. My teacher is Marta, Thursdays at 19:00."
                ),
            },
            {
                "role": "assistant",
                "content": "Cello with Marta on Thursdays at 19:00, noted.",
            },
        ],
    },
    {
        "id": "everos-demo-cat-allergy",
        "days_ago": 42,
        "messages": [
            {
                "role": "user",
                "content": (
                    "The vet says Mochi is allergic to chicken. We switched her to the "
                    "salmon food."
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "Mochi has a chicken allergy and is now on salmon food. I will "
                    "keep chicken out of any suggestions for her."
                ),
            },
        ],
    },
    {
        "id": "everos-demo-bike",
        "days_ago": 35,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Radhaus replaced the rear derailleur on my bike. They said to "
                    "swap the chain at 3000 km."
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "New rear derailleur from Radhaus, and the chain is due for a swap "
                    "at 3000 km."
                ),
            },
        ],
    },
    {
        "id": "everos-demo-sourdough",
        "days_ago": 28,
        "messages": [
            {
                "role": "user",
                "content": (
                    "My sourdough came out flat three bakes running. The starter "
                    "doubles by noon, so that is not it."
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "The kitchen drops to about 16C overnight and the dough sat on the "
                    "counter until morning. Bulk proofing wants 24 to 26C, so the rise "
                    "stalled and the loaf spread instead of holding shape."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Proofed it in the oven with just the light on and it worked. Best "
                    "crumb yet."
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "The oven light holds about 24C, which fixed it. Worth keeping: "
                    "judge proofing by the dough, not the clock, because the same "
                    "recipe needs different hours at different temperatures."
                ),
            },
        ],
    },
    {
        "id": "everos-demo-spare-keys",
        "days_ago": 21,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Tomas next door has our spare keys now. He waters the plants when "
                    "we travel."
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "Tomas next door holds the spare keys and waters the plants while "
                    "you are away."
                ),
            },
        ],
    },
    {
        "id": "everos-demo-physio",
        "days_ago": 14,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Started physio for my right shoulder. Band work twice a day, and "
                    "no overhead presses until they clear me."
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "Physio for the right shoulder: band exercises twice daily, and "
                    "overhead presses are off the table until you are cleared."
                ),
            },
        ],
    },
    {
        "id": "everos-demo-food-rules",
        "days_ago": 8,
        "messages": [
            {
                "role": "user",
                "content": (
                    "When you plan meals for me, remember I am vegetarian and I "
                    "cannot stand mushrooms. No fish either."
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "Recorded your food rules for meal planning: vegetarian, no "
                    "fish, and no mushrooms in anything."
                ),
            },
        ],
    },
    {
        "id": "everos-demo-morning-routine",
        "days_ago": 5,
        "messages": [
            {
                "role": "user",
                "content": (
                    "I run before work every morning, so breakfast ends up late, "
                    "usually around ten."
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "Noted: you run before work each morning and eat breakfast "
                    "late, around ten."
                ),
            },
        ],
    },
]

# ── the queries ──────────────────────────────────────────────────────────
# KEYWORD is deliberately absent: its top score is raw BM25, on a different
# scale from the calibrated methods, so showing the three side by side invites
# a comparison that means nothing. It still runs as the readiness probe.
QUERIES: list[dict] = [
    {
        "label": "history",
        "query": (
            "what happened with the October trip we booked, did the destination "
            "change after the wedding moved"
        ),
        "note": "the plan, its revision, and whatever reflection made of them",
    },
    {
        "label": "constraint",
        "query": (
            "what did the vet say about Mochi's allergy and which food did we "
            "switch her to"
        ),
        "note": "one specific memory out of eleven conversations",
    },
    {
        "label": "how-to",
        "query": (
            "why did my sourdough loaves keep coming out flat and what fixed "
            "the overnight proofing"
        ),
        "note": "the diagnosis and the fix, not just a stated fact",
    },
    {
        "label": "profile",
        "query": (
            "which foods should you leave out when you plan my meals, I am vegetarian"
        ),
        "note": "include_profile also returns the distilled profile",
        "include_profile": True,
    },
    {
        "label": "miss",
        "query": "what did the accountant say about our tax return this year",
        "note": "never discussed — candidates come back, the score says no",
    },
]

METHODS = ("hybrid", "agentic")


def _post(path: str, body: dict) -> dict:
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return json.load(resp)


def _search(spec: dict, method: str, *, top_k: int = 5) -> dict:
    """Search one query across everything its owner remembers.

    No session filter: the point is to make recall choose between memories
    from different conversations.
    """
    body: dict = {
        "user_id": USER,
        "query": spec["query"],
        "method": method,
        "top_k": top_k,
    }
    if spec.get("include_profile"):
        body["include_profile"] = True
    return _post("/api/v2/memory/search", body)


def _wire_messages(session: dict, base_ts: int) -> list[dict]:
    """Expand a session's (role, content) pairs into API message items."""
    return [
        {
            "message_id": f"{session['id']}-m{index}",
            "role": message["role"],
            "content": message["content"],
            "timestamp": base_ts + index * 60_000,
            "sender_id": USER if message["role"] == "user" else "assistant",
        }
        for index, message in enumerate(session["messages"], start=1)
    ]


def _ingest(session: dict, now_ms: int) -> None:
    base_ts = now_ms - session["days_ago"] * DAY_MS
    messages = _wire_messages(session, base_ts)
    add = _post(
        "/api/v2/memory/add", {"session_id": session["id"], "messages": messages}
    )
    flush = _post("/api/v2/memory/flush", {"session_id": session["id"], "messages": []})
    print(
        f"  {session['id']:<30} {len(messages):>2} msgs  "
        f"add={add['data'].get('status')} flush={flush['data'].get('status')}"
    )


def _session_is_searchable(session: dict) -> bool:
    """Keyword-probe one session with its own opening line.

    Querying the session's own words guarantees the lexical overlap BM25
    needs, so an empty result means "not indexed yet" rather than "no match".
    """
    opening = next(m["content"] for m in session["messages"] if m["role"] == "user")
    body = {
        "user_id": USER,
        "query": opening,
        "method": "keyword",
        "top_k": 1,
        "filters": {"session_id": session["id"]},
    }
    return bool(_post("/api/v2/memory/search", body)["data"].get("episodes"))


def _wait_for_index() -> list[str]:
    """Poll until every ingested session is searchable; return any laggards.

    Waiting on one session is not enough: extraction and the SQLite ->
    LanceDB sync run per session and finish out of order, so querying too
    early makes recall choose from a partial store and the scores read
    lower than the memory deserves.
    """
    deadline = time.monotonic() + INDEX_TIMEOUT_SECONDS
    laggards: list[str] = []
    # One session at a time, in ingest order. Probing every pending session on
    # every round would work too, but each probe is itself a traced search, and
    # a hundred readiness probes would bury the five real queries in Langfuse.
    # Extraction broadly follows ingest order, so by the time session N answers
    # its predecessors already have.
    for session in SESSIONS:
        while not _session_is_searchable(session):
            if time.monotonic() > deadline:
                laggards.append(session["id"])
                break
            time.sleep(INDEX_POLL_SECONDS)
    return laggards


def _reflect() -> str:
    """Run episode reflection now instead of waiting for its weekly cron.

    Consolidation is what merges a cluster of related memories and deprecates
    what they superseded, so a demo that never triggers it never shows the
    part of EverOS that improves memory over time. ``reflect_episodes`` is
    scheduled ``0 2 * * 1``, hence the manual nudge.
    """
    body = {"name": "reflect_episodes", "force": True, "timeout": 300.0}
    return str(_post("/api/v2/ome/trigger", body)["status"])


def _wait_for_consolidation() -> bool:
    """Poll until the consolidated memory has replaced what it superseded.

    ``/ome/trigger`` returns once the OME engine is idle, but the merge reaches
    LanceDB through the cascade, and deprecating the old episodes is a separate
    write from indexing the merged one. Querying in between sees neither, and
    scores lower than the memory deserves. So wait for both edges: the first
    trip session going unsearchable, then the merged memory answering for it.
    """
    superseded, survivor = SESSIONS[0], SESSIONS[2]
    opening = next(m["content"] for m in survivor["messages"] if m["role"] == "user")
    deadline = time.monotonic() + CONSOLIDATION_TIMEOUT_SECONDS

    while time.monotonic() < deadline:
        if not _session_is_searchable(superseded):
            break
        time.sleep(INDEX_POLL_SECONDS)
    else:
        return False

    # The originals are gone; wait for the merged episode to answer in their
    # place. No session filter: the merge is its own entry, not either source.
    while time.monotonic() < deadline:
        found = _post(
            "/api/v2/memory/search",
            {"user_id": USER, "query": opening, "method": "keyword", "top_k": 1},
        )["data"].get("episodes")
        if found:
            return True
        time.sleep(INDEX_POLL_SECONDS)
    return False


def _describe(data: dict) -> str:
    """What a search returned per memory kind, plus its best score.

    The score matters more than the count: recall returns candidates up to
    ``top_k`` whether or not they are relevant, so a query about something
    never discussed still comes back with episodes. The top score is what
    says they do not answer it.
    """
    parts = [
        f"{len(items)} {kind.replace('_', ' ')}"
        for kind in ("episodes", "profiles", "agent_cases", "agent_skills")
        if (items := data.get(kind) or [])
    ]
    scored = [
        item.get("score")
        for kind in ("episodes", "agent_cases", "agent_skills")
        for item in data.get(kind) or []
        if item.get("score") is not None
    ]
    summary = ", ".join(parts) or "nothing"
    return f"{summary:<34} top_score={max(scored):.3f}" if scored else summary


def main() -> None:
    now_ms = int(time.time() * 1000)

    print(f"ingesting {len(SESSIONS)} sessions ...")
    for session in SESSIONS:
        _ingest(session, now_ms)

    print("\nwaiting for async extraction + index sync ...")
    if pending := _wait_for_index():
        print(
            f"  still not searchable after {INDEX_TIMEOUT_SECONDS:.0f}s: "
            f"{', '.join(pending)}; searching anyway so you can still see "
            "the traces"
        )
    else:
        print(f"  all {len(SESSIONS)} sessions searchable")

    print("\nrunning reflection (normally a weekly cron) ...")
    print(f"  reflect_episodes -> {_reflect()}")
    if _wait_for_consolidation():
        print(f"  {SESSIONS[0]['id']} superseded and no longer searchable")
    else:
        print("  nothing was consolidated; the originals are both still live")

    print()
    for spec in QUERIES:
        print(f"{spec['label']}: {spec['query']}")
        print(f"  ({spec['note']})")
        for method in spec.get("methods", METHODS):
            try:
                data = _search(spec, method)["data"]
            except urllib.error.HTTPError as exc:
                # Embedding and rerank are soft dependencies: with neither
                # configured a server serves KEYWORD only, and HYBRID /
                # AGENTIC answer 422 CAPABILITY_UNAVAILABLE. Report it and
                # carry on so the other queries still have something to show.
                print(f"  {method:<8} HTTP {exc.code}: {exc.reason}")
                continue
            print(f"  {method:<8} {_describe(data)}")
        print()

    print(
        "Open Langfuse -> Tracing. Traces are grouped by session; the search "
        "traces carry recall-quality scores, and the flush traces carry the "
        "LLM token usage Langfuse turns into cost."
    )


if __name__ == "__main__":
    main()

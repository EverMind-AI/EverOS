"""End-to-end report generator: fresh corpus → ingest → retrieve → markdown report.

Run with::

    PYTHONPATH=src python tests/integration/search/_run_full_report.py

Writes a fresh ``~/.everos-report-corpus/`` memory_root, runs a small
synthetic 16-message conversation between two new users (``u_diana`` +
``u_ethan``) through ``/add`` + ``/flush``, waits for cascade drain, then
runs a curated set of search probes and dumps a structured markdown
report to ``tests/integration/search/SEARCH_REPORT.md``.

Not a pytest test — pure investigative script, real LLM, real embedder.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from pathlib import Path

import httpx
from dotenv import load_dotenv

# Load .env BEFORE any everos import so settings are correct.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(_PROJECT_ROOT / ".env", override=False)


# ── Corpus location ────────────────────────────────────────────────────


CORPUS_ROOT = Path.home() / ".everos-report-corpus"
REPORT_PATH = _PROJECT_ROOT / "tests/integration/search/SEARCH_REPORT.md"
SESSION_ID = "report_session_diana_ethan"


# ── Synthetic conversation (16 msgs, 2 batches) ────────────────────────


CONVERSATION = [
    # Batch 1 — introducing hobbies
    [
        {
            "sender_id": "u_diana",
            "role": "user",
            "timestamp": 1778414400000,
            "content": "Hey Ethan! Just got back from a 3-day hike in Yosemite. "
            "My new Sony A7 camera is amazing for landscape shots.",
        },
        {
            "sender_id": "u_ethan",
            "role": "user",
            "timestamp": 1778407260000,
            "content": "Wow that sounds intense! I'd never survive without my "
            "espresso. How's the Rust programming learning going?",
        },
        {
            "sender_id": "u_diana",
            "role": "user",
            "timestamp": 1778407320000,
            "content": "Slow but steady. Working through the official book. "
            "The borrow checker still trips me up.",
        },
        {
            "sender_id": "u_ethan",
            "role": "user",
            "timestamp": 1778407380000,
            "content": "I'm marathon training — up to 15 miles long runs now. "
            "Plus I joined a jazz quartet on weekends.",
        },
        {
            "sender_id": "u_diana",
            "role": "user",
            "timestamp": 1778407440000,
            "content": "That's awesome! Saxophone again?",
        },
        {
            "sender_id": "u_ethan",
            "role": "user",
            "timestamp": 1778407500000,
            "content": "Yeah, alto sax. We're playing at the Blue Note next month.",
        },
        {
            "sender_id": "u_diana",
            "role": "user",
            "timestamp": 1778407560000,
            "content": "I'll come watch! Speaking of trips, want to do "
            "that Iceland thing this summer?",
        },
        {
            "sender_id": "u_ethan",
            "role": "user",
            "timestamp": 1778407620000,
            "content": "100% yes. I've been researching ring road photography spots.",
        },
    ],
    # Batch 2 — Iceland trip planning
    [
        {
            "sender_id": "u_diana",
            "role": "user",
            "timestamp": 1778410800000,
            "content": "I want to see the Northern Lights and shoot some "
            "volcanic landscapes.",
        },
        {
            "sender_id": "u_ethan",
            "role": "user",
            "timestamp": 1778410860000,
            "content": "We should rent a 4x4. The F-roads are insane I hear.",
        },
        {
            "sender_id": "u_diana",
            "role": "user",
            "timestamp": 1778410920000,
            "content": "And I want to try Icelandic lamb stew. You cook, right?",
        },
        {
            "sender_id": "u_ethan",
            "role": "user",
            "timestamp": 1778410980000,
            "content": (
                "Yeah, I'll bring my Dutch oven. Maybe a cast iron pan for fish."
            ),
        },
        {
            "sender_id": "u_diana",
            "role": "user",
            "timestamp": 1778411040000,
            "content": "Perfect. Mid-July works for me — I have a Rust conference "
            "in late August.",
        },
        {
            "sender_id": "u_ethan",
            "role": "user",
            "timestamp": 1778411100000,
            "content": "July it is. I have the Boston Marathon qualifier in October "
            "so I can't go after.",
        },
        {
            "sender_id": "u_diana",
            "role": "user",
            "timestamp": 1778411160000,
            "content": "Let's book flights this weekend?",
        },
        {
            "sender_id": "u_ethan",
            "role": "user",
            "timestamp": 1778411220000,
            "content": "Deal. Also bringing my Olympus E-M1 for the landscapes.",
        },
    ],
]


# ── Probe set ───────────────────────────────────────────────────────────


PROBES: list[dict] = [
    # Owner-specific topical: should recall the right owner's episodes.
    {
        "section": "Owner-specific topical (diana)",
        "owner": "u_diana",
        "query": "hiking",
        "method": "hybrid",
        "expect": "diana's Yosemite episode",
    },
    {
        "section": "Owner-specific topical (diana)",
        "owner": "u_diana",
        "query": "Rust programming",
        "method": "hybrid",
        "expect": "diana's Rust learning facts",
    },
    {
        "section": "Owner-specific topical (diana)",
        "owner": "u_diana",
        "query": "photography",
        "method": "hybrid",
        "expect": "diana's camera (Sony A7) facts",
    },
    {
        "section": "Owner-specific topical (ethan)",
        "owner": "u_ethan",
        "query": "jazz",
        "method": "hybrid",
        "expect": "ethan's jazz quartet / sax facts",
    },
    {
        "section": "Owner-specific topical (ethan)",
        "owner": "u_ethan",
        "query": "marathon training",
        "method": "hybrid",
        "expect": "ethan's marathon facts",
    },
    {
        "section": "Owner-specific topical (ethan)",
        "owner": "u_ethan",
        "query": "cooking",
        "method": "hybrid",
        "expect": "ethan's Dutch oven / lamb stew facts",
    },
    # Shared topic — both should recall their own perspective.
    {
        "section": "Shared topic (Iceland)",
        "owner": "u_diana",
        "query": "Iceland trip",
        "method": "hybrid",
        "expect": "diana's planning episode",
    },
    {
        "section": "Shared topic (Iceland)",
        "owner": "u_ethan",
        "query": "Iceland trip",
        "method": "hybrid",
        "expect": "ethan's planning episode",
    },
    # Method comparison on the same query.
    {
        "section": "Method comparison (diana + 'Rust')",
        "owner": "u_diana",
        "query": "Rust",
        "method": "keyword",
        "expect": "BM25 single token",
    },
    {
        "section": "Method comparison (diana + 'Rust')",
        "owner": "u_diana",
        "query": "Rust",
        "method": "vector",
        "expect": "cosine ANN",
    },
    {
        "section": "Method comparison (diana + 'Rust')",
        "owner": "u_diana",
        "query": "Rust",
        "method": "hybrid",
        "expect": "fusion of BM25 + vector",
    },
    # Owner partition: diana searching for ethan's exclusive topic.
    {
        "section": "Owner partition",
        "owner": "u_diana",
        "query": "jazz quartet",
        "method": "hybrid",
        "expect": "should NOT leak ethan's content",
    },
    {
        "section": "Owner partition",
        "owner": "u_ethan",
        "query": "Rust programming",
        "method": "hybrid",
        "expect": "should NOT leak diana's content",
    },
    # Phrase + bigram.
    {
        "section": "Phrase queries",
        "owner": "u_diana",
        "query": "Northern Lights",
        "method": "keyword",
        "expect": "diana's Iceland aurora plans",
    },
    {
        "section": "Phrase queries",
        "owner": "u_ethan",
        "query": "Boston Marathon",
        "method": "keyword",
        "expect": "ethan's qualifier date",
    },
    # include_profile.
    {
        "section": "Profile attach",
        "owner": "u_diana",
        "query": "anything",
        "method": "hybrid",
        "include_profile": True,
        "expect": "should return diana's profile object",
    },
    # Unknown owner.
    {
        "section": "Unknown owner",
        "owner": "u_ghost_does_not_exist",
        "query": "hiking",
        "method": "hybrid",
        "expect": "empty arrays, status 200",
    },
    # Non-existent term.
    {
        "section": "Non-existent term",
        "owner": "u_diana",
        "query": "quantum blockchain pizza",
        "method": "keyword",
        "expect": "0 hits, status 200",
    },
]


# ── Pipeline runners ───────────────────────────────────────────────────


async def ingest(client: httpx.AsyncClient) -> dict:
    """POST /add for each batch, then /flush. Return summary."""
    summary: dict = {"batches": [], "flush_status": None}
    for i, batch in enumerate(CONVERSATION):
        resp = await client.post(
            "/api/v1/memory/add",
            json={"session_id": SESSION_ID, "messages": batch},
            timeout=600.0,
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        summary["batches"].append(
            {
                "idx": i,
                "msg_count": len(batch),
                "status": data["status"],
                "returned_count": data["message_count"],
            }
        )
    resp = await client.post(
        "/api/v1/memory/flush",
        json={"session_id": SESSION_ID},
        timeout=600.0,
    )
    resp.raise_for_status()
    summary["flush_status"] = resp.json()["data"]["status"]
    return summary


async def wait_cascade(
    *,
    expected_md_paths: int = 8,
    stable_checks: int = 5,
    deadline_seconds: float = 600.0,
) -> dict:
    """Block until cascade is *stably* done across all expected md kinds.

    A plain ``pending == 0`` check is racy: OME async strategies
    (extract_foresight / extract_user_profile) emit md writes
    asynchronously after ``/flush`` returns, and there's a window
    where the cascade queue is momentarily empty before OME's writes
    arrive. We require two stronger conditions:

    1. At least ``expected_md_paths`` rows exist in ``md_change_state``
       (one per expected (owner × kind) — episodes + atomic_facts +
       foresights + user_profile, per owner). This guards against
       returning before OME has emitted *anything*.
    2. ``pending == 0`` stays true for ``stable_checks`` consecutive
       polls (separated by 1s sleep). This guards against a transient
       empty queue while a strategy is still mid-write.
    """
    from everos.infra.persistence.sqlite import md_change_state_repo

    consecutive_zero = 0
    async with asyncio.timeout(deadline_seconds):
        while True:
            sm = await md_change_state_repo.queue_summary()
            total_rows = (
                sm.pending + sm.done + sm.failed_retryable + sm.failed_permanent
            )
            if sm.pending == 0 and total_rows >= expected_md_paths:
                consecutive_zero += 1
                if consecutive_zero >= stable_checks:
                    return {
                        "done": sm.done,
                        "failed_retryable": sm.failed_retryable,
                        "failed_permanent": sm.failed_permanent,
                        "max_lsn": sm.max_lsn,
                        "last_processed_lsn": sm.last_processed_lsn,
                    }
            else:
                consecutive_zero = 0
            await asyncio.sleep(1.0)


async def inspect_artifacts(memory_root: Path) -> dict:
    """Read md files + LanceDB counts after cascade drain."""
    from everos.infra.persistence.lancedb import (
        atomic_fact_repo,
        dispose_connection,
        episode_repo,
        foresight_repo,
        get_connection,
        user_profile_repo,
        verify_business_schemas,
    )

    await get_connection()
    await verify_business_schemas()
    counts = {
        "episode_rows": await episode_repo.count(),
        "atomic_fact_rows": await atomic_fact_repo.count(),
        "foresight_rows": await foresight_repo.count(),
        "user_profile_rows": await user_profile_repo.count(),
    }
    await dispose_connection()

    md_files: list[str] = []
    users_dir = memory_root / "default_app" / "default_project" / "users"
    if users_dir.is_dir():
        for f in sorted(users_dir.rglob("*.md")):
            md_files.append(str(f.relative_to(memory_root)))
    counts["md_files"] = md_files
    return counts


async def run_probes(client: httpx.AsyncClient) -> list[dict]:
    """Execute every probe in :data:`PROBES`; return captured rows."""
    rows: list[dict] = []
    for p in PROBES:
        payload: dict = {
            "owner_id": p["owner"],
            "owner_type": "user",
            "query": p["query"],
            "method": p["method"],
            "top_k": 5,
        }
        if p.get("include_profile"):
            payload["include_profile"] = True
        resp = await client.post("/api/v1/memory/search", json=payload, timeout=120.0)
        body = resp.json()
        data = body.get("data", {})
        rows.append(
            {
                "section": p["section"],
                "expect": p["expect"],
                "request": payload,
                "status": resp.status_code,
                "episodes": [
                    {
                        "id": e["id"],
                        "owner_id": e["owner_id"],
                        "score": round(float(e["score"]), 3),
                        "summary": (e.get("summary") or "")[:150],
                        "atomic_facts_count": len(e.get("atomic_facts", [])),
                    }
                    for e in data.get("episodes", [])
                ],
                "profiles": [
                    {
                        "owner_id": p_.get("owner_id"),
                        "score": p_.get("score"),
                        "summary_excerpt": str(p_.get("profile_data", {}))[:200],
                    }
                    for p_ in data.get("profiles", [])
                ],
            }
        )
    return rows


# ── Markdown report renderer ───────────────────────────────────────────


def render_report(
    *,
    memory_root: Path,
    ingest_summary: dict,
    cascade_summary: dict,
    artifacts: dict,
    probes: list[dict],
) -> str:
    lines: list[str] = []
    lines.append("# Search E2E Report — fresh corpus (u_diana + u_ethan)\n")
    lines.append(
        "Generated by [`_run_full_report.py`](_run_full_report.py). "
        "Two synthetic users with distinct hobbies feed a 16-message "
        "conversation through the full pipeline; the report below "
        "captures ingest stats, cascade drain numbers, on-disk "
        "artifacts, and the response of every curated search probe.\n"
    )

    # ── Section: Setup ────────────────────────────────────────────────
    lines.append("## 1. Setup\n")
    lines.append(f"- **Memory root**: `{memory_root}`\n")
    lines.append(f"- **Session id**: `{SESSION_ID}`\n")
    lines.append(
        "- **Users**: `u_diana` (hiking / Rust / photography), "
        "`u_ethan` (jazz / marathon / cooking)\n"
    )
    lines.append(
        f"- **Batches**: {len(CONVERSATION)} "
        f"({sum(len(b) for b in CONVERSATION)} messages total)\n"
    )

    # ── Section: Ingest stats ─────────────────────────────────────────
    lines.append("\n## 2. Ingest (`/add` × N + `/flush`)\n")
    lines.append("| batch | msg_count | status |\n")
    lines.append("|---|---|---|\n")
    for b in ingest_summary["batches"]:
        lines.append(f"| {b['idx']} | {b['msg_count']} | `{b['status']}` |\n")
    lines.append(f"\n**Flush status**: `{ingest_summary['flush_status']}`\n")

    # ── Section: Cascade drain ────────────────────────────────────────
    lines.append("\n## 3. Cascade drain (md → LanceDB sync)\n")
    lines.append("```\n")
    lines.append(json.dumps(cascade_summary, indent=2) + "\n")
    lines.append("```\n")

    # ── Section: Artifacts ────────────────────────────────────────────
    lines.append("\n## 4. On-disk artifacts\n")
    lines.append("### LanceDB row counts\n\n")
    lines.append("| table | rows |\n")
    lines.append("|---|---|\n")
    for k in (
        "episode_rows",
        "atomic_fact_rows",
        "foresight_rows",
        "user_profile_rows",
    ):
        lines.append(f"| {k.replace('_rows', '')} | {artifacts[k]} |\n")
    lines.append("\n### Markdown files\n\n")
    for f in artifacts["md_files"]:
        lines.append(f"- `{f}`\n")

    # ── Section: Probes ───────────────────────────────────────────────
    lines.append("\n## 5. Retrieval probes\n")
    lines.append(
        "Every row below is one POST to `/api/v1/memory/search`. "
        "`expected` is what the test designer expects to see; "
        "actual results are captured verbatim.\n"
    )
    current_section = None
    for row in probes:
        if row["section"] != current_section:
            lines.append(f"\n### {row['section']}\n")
            current_section = row["section"]
        req = row["request"]
        lines.append(
            f"\n#### `{req['query']}`  (method=`{req['method']}`, "
            f"owner=`{req['owner_id']}`)\n"
        )
        lines.append(f"\n- **Expected**: {row['expect']}\n")
        lines.append(f"- **Status**: {row['status']}\n")
        lines.append(f"- **Episodes returned**: {len(row['episodes'])}\n")
        if row["episodes"]:
            lines.append("\n| rank | score | owner | atomic_facts | summary |\n")
            lines.append("|---|---|---|---|---|\n")
            for i, ep in enumerate(row["episodes"], 1):
                summary = ep["summary"].replace("|", "\\|")
                lines.append(
                    f"| {i} | {ep['score']} | `{ep['owner_id']}` | "
                    f"{ep['atomic_facts_count']} | {summary} |\n"
                )
        else:
            lines.append("\n_(no episodes)_\n")
        if row["profiles"]:
            lines.append(
                "\n**Profile attached**: "
                f"`{row['profiles'][0]['owner_id']}` "
                f"(excerpt: {row['profiles'][0]['summary_excerpt']!r})\n"
            )

    # ── Section: Pass/Fail summary ────────────────────────────────────
    lines.append("\n## 6. Pass / Fail summary\n")
    pf = _grade(probes)
    lines.append("| # | section | query | result |\n")
    lines.append("|---|---|---|---|\n")
    for r in pf:
        lines.append(
            f"| {r['idx']} | {r['section']} | `{r['query']}` | {r['verdict']} |\n"
        )
    passed = sum(1 for r in pf if r["verdict"].startswith("✅"))
    lines.append(f"\n**Total: {passed}/{len(pf)} passed.**\n")

    return "".join(lines)


def _grade(probes: list[dict]) -> list[dict]:
    """Apply soft heuristic pass/fail to each probe based on its 'expect'."""
    graded: list[dict] = []
    for i, row in enumerate(probes, 1):
        req = row["request"]
        expect = row["expect"].lower()
        verdict = "—"
        if "should not leak" in expect:
            leaked = any(ep["owner_id"] != req["owner_id"] for ep in row["episodes"])
            verdict = "❌ leaked" if leaked else "✅ no leak"
        elif "empty arrays" in expect or "0 hits" in expect:
            verdict = "✅" if not row["episodes"] else f"❌ got {len(row['episodes'])}"
        elif "profile" in expect:
            verdict = "✅" if row["profiles"] else "❌ no profile"
        elif row["episodes"]:
            top_owner = row["episodes"][0]["owner_id"]
            verdict = (
                "✅" if top_owner == req["owner_id"] else f"❌ wrong owner: {top_owner}"
            )
        else:
            verdict = "❌ no hits"
        graded.append(
            {
                "idx": i,
                "section": row["section"],
                "query": req["query"],
                "verdict": verdict,
            }
        )
    return graded


# ── Main ────────────────────────────────────────────────────────────────


async def main() -> None:
    # Reset corpus to a known empty state.
    if CORPUS_ROOT.exists():
        shutil.rmtree(CORPUS_ROOT)
    CORPUS_ROOT.mkdir(parents=True)
    os.environ["EVEROS_ROOT"] = str(CORPUS_ROOT)

    # Reset cached singletons so they pick up the new env.
    from everos.config import load_settings

    load_settings.cache_clear()

    print(f"[1/6] fresh corpus at {CORPUS_ROOT}")

    from everos.entrypoints.api.app import create_app

    app = create_app()
    transport = httpx.ASGITransport(app=app)

    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://test") as client,
    ):
        print("[2/6] ingesting via /add + /flush ...")
        ingest_summary = await ingest(client)
        print(f"      batches={ingest_summary['batches']}")

        print("[3/6] waiting for cascade drain ...")
        cascade_summary = await wait_cascade()
        print(f"      drained: {cascade_summary}")

        print("[4/6] inspecting on-disk artifacts ...")
        artifacts = await inspect_artifacts(CORPUS_ROOT)
        print(
            "      lancedb: {k: v for k,v in artifacts.items() if k.endswith('_rows')}"
        )

        print(f"[5/6] running {len(PROBES)} search probes ...")
        probes = await run_probes(client)

    print("[6/6] rendering report ...")
    md = render_report(
        memory_root=CORPUS_ROOT,
        ingest_summary=ingest_summary,
        cascade_summary=cascade_summary,
        artifacts=artifacts,
        probes=probes,
    )
    REPORT_PATH.write_text(md, encoding="utf-8")
    print(f"      → {REPORT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())

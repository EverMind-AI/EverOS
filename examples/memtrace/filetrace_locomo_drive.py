"""Drive a running, memtrace-enabled EverOS server through part of the first
LoCoMo conversation (Caroline / Melanie), reconstructed from the MemTraceBench
mem0 split (``MemTraceBench/mem0/locomo_locomo-1.json``, whose message nodes
carry content/name/role/timestamp for all 419 messages across 19 session
dates).

Sequence:

1. add + flush the first ``--sessions`` sessions (default 5)
2. wait for the memory-root markdown tree to settle
3. run a few annotated queries via /memory/search (default: the first three
   annotations, attributed to the user the question is about)
4. with ``--qa``: answer each query with an LLM (``EVEROS_QA_BASE_URL`` /
   ``EVEROS_QA_API_KEY`` / ``EVEROS_QA_MODEL``), judge it against the bench
   annotations, and write ``qa_results.json`` — feed that file to
   ``inject_qa.py`` to stitch the answer/evaluation chain onto the graphs
5. one final light search so the file tracker observes any late background
   (OME) writes

Stdlib only — run against a server started with, e.g.::

    EVEROS_MEMTRACE_ENABLED=1 everos server start --root <root> --port 8016

    python examples/memtrace/filetrace_locomo_drive.py \
        --base http://127.0.0.1:8016 \
        --graph ../MemTrace/MemTraceBench/mem0/locomo_locomo-1.json \
        --memory-root <root>
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
import urllib.request
from datetime import UTC, datetime

_DEFAULT_GRAPH = str(
    pathlib.Path(__file__).resolve().parents[3]
    / "MemTrace/MemTraceBench/mem0/locomo_locomo-1.json"
)

# (query, user_id) — the first annotated MemTraceBench queries for conv-1.
_DEFAULT_QUERIES = [
    ("What career path has Caroline decided to persue?", "caroline"),
    ("When did Caroline go to the adoption meeting?", "caroline"),
    ("When did Melanie go camping in July?", "melanie"),
]

_META_SUFFIX = "\nBelow is this message's metadata:"


def _load_sessions(graph_path: str) -> list[tuple[str, list[dict]]]:
    """Rebuild (date, messages) sessions from the bench graph's message nodes."""
    with open(graph_path, encoding="utf-8") as f:
        graph_data = json.load(f)
    nodes = graph_data["data"]["nodes"]
    by_date: dict[str, list[dict]] = {}
    for node in nodes:
        if node.get("category") != "message":
            continue
        value = json.loads(node["value"])
        text = value["content"].split(_META_SUFFIX, 1)[0]
        ts = datetime.fromisoformat(value["timestamp"]).replace(tzinfo=UTC)
        by_date.setdefault(value["timestamp"][:10], []).append(
            {
                "role": "user",
                "sender_id": value["name"].lower(),
                "timestamp": int(ts.timestamp() * 1000),
                "content": text,
            }
        )
    sessions = sorted(by_date.items())
    return [
        (f"locomo1-session-{i:02d}", sorted(msgs, key=lambda m: m["timestamp"]))
        for i, (_, msgs) in enumerate(sessions, start=1)
    ]


def _post(
    base: str, path: str, body: dict, timeout: float = 600.0
) -> tuple[dict, dict]:
    req = urllib.request.Request(
        base + path,
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp), dict(resp.headers)


def _header(headers: dict, name: str) -> str:
    """Case-insensitive header lookup (uvicorn may lowercase header names)."""
    for key, value in headers.items():
        if key.lower() == name.lower():
            return value
    return ""


def _normalize_query(text: str) -> str:
    """Normalize a query for annotation matching (bench quotes its queries)."""
    return text.strip().strip("'\"").strip()


def _load_annotations(graph_path: str) -> dict[str, list[str]]:
    """Map query text -> golden answers from the bench graph annotations."""
    with open(graph_path, encoding="utf-8") as f:
        graph_data = json.load(f)
    golden: dict[str, list[str]] = {}
    for annotation in graph_data.get("data", {}).get("annotations") or []:
        query = annotation.get("query")
        answers = annotation.get("golden_answers")
        if query and answers:
            golden.setdefault(_normalize_query(query), list(answers))
    return golden


def _chat(prompt: str, *, timeout: float = 300.0) -> str:
    """One OpenAI-compatible chat completion for the QA / judge calls.

    Endpoint credentials come from the environment (``EVEROS_QA_BASE_URL`` /
    ``EVEROS_QA_API_KEY`` / ``EVEROS_QA_MODEL``); nothing is hardcoded.
    """
    base_url = os.environ["EVEROS_QA_BASE_URL"].rstrip("/")
    req = urllib.request.Request(
        base_url + "/chat/completions",
        data=json.dumps(
            {
                "model": os.environ["EVEROS_QA_MODEL"],
                "messages": [{"role": "user", "content": prompt}],
            }
        ).encode(),
        headers={
            "content-type": "application/json",
            "authorization": f"Bearer {os.environ['EVEROS_QA_API_KEY']}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.load(resp)
    return payload["choices"][0]["message"]["content"]


_ANSWER_TEMPLATE = (
    "Answer the question using only the retrieved memory context below.\n\n"
    "Memory context:\n{context}\n\nQuestion: {query}\nAnswer:"
)

_JUDGE_TEMPLATE = (
    "Question: {query}\nReference answers: {golden}\nPredicted answer: {answer}\n"
    "Does the predicted answer match any reference answer? The first line must be "
    "'VERDICT: 1' (correct) or 'VERDICT: 0' (incorrect), then one sentence why."
)


def _qa_record(
    query: str,
    user_id: str,
    request_id: str,
    data: dict,
    golden: dict[str, list[str]],
) -> dict:
    """Answer one searched query with an LLM and judge it (for injection)."""
    episodes = data.get("episodes") or []
    context = (
        "\n".join(
            f"- {ep.get('episode') or ep.get('subject') or ep.get('id')}"
            for ep in episodes
        )
        or "(no memories retrieved)"
    )
    prompt = _ANSWER_TEMPLATE.format(context=context, query=query)
    answer = _chat(prompt)
    record = {
        "user_id": user_id,
        "request_id": request_id,
        "query": query,
        "prompt": prompt,
        "answer_context": context,
        "answer": answer,
        "golden_answers": golden.get(_normalize_query(query), []),
    }
    if record["golden_answers"]:
        verdict = _chat(
            _JUDGE_TEMPLATE.format(
                query=query,
                golden="; ".join(record["golden_answers"]),
                answer=answer,
            )
        )
        record["judge_response"] = verdict
        record["judge_score"] = 1.0 if "VERDICT: 1" in verdict else 0.0
    return record


def _wait_settled(
    directory: pathlib.Path,
    pattern: str,
    *,
    settle_checks: int = 5,
    timeout: float = 1800.0,
    label: str = "",
) -> None:
    """Wait until the files matching `pattern` stop changing."""
    deadline = time.time() + timeout
    last, stable = None, 0
    while time.time() < deadline:
        state = (
            sorted(
                (str(p), p.stat().st_mtime_ns, p.stat().st_size)
                for p in directory.rglob(pattern)
            )
            if directory.exists()
            else []
        )
        if state == last:
            stable += 1
            if state and stable >= settle_checks:
                print(
                    f"[settle] {label or pattern}: stable at {len(state)} files",
                    flush=True,
                )
                return
        else:
            stable = 0
        last = state
        time.sleep(2)
    print(f"[settle] {label or pattern}: timeout", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", default=os.environ.get("EVEROS_BASE", "http://127.0.0.1:8016")
    )
    parser.add_argument(
        "--graph", default=_DEFAULT_GRAPH, help="MemTraceBench conversation graph JSON"
    )
    parser.add_argument(
        "--sessions", type=int, default=5, help="Number of leading sessions to replay"
    )
    parser.add_argument(
        "--memory-root", default="", help="EverOS memory root (for settle waiting)"
    )
    parser.add_argument(
        "--queries", default="", help="Optional JSON file with [[query, user_id], ...]"
    )
    parser.add_argument(
        "--qa",
        action="store_true",
        help="Answer each query with an LLM and judge it against "
        "the bench annotations (needs EVEROS_QA_BASE_URL / "
        "EVEROS_QA_API_KEY / EVEROS_QA_MODEL)",
    )
    parser.add_argument(
        "--qa-out",
        default="qa_results.json",
        help="Where to write the QA records for inject_qa.py",
    )
    args = parser.parse_args()

    if args.qa:
        missing = [
            var
            for var in ("EVEROS_QA_BASE_URL", "EVEROS_QA_API_KEY", "EVEROS_QA_MODEL")
            if not os.environ.get(var)
        ]
        if missing:
            raise SystemExit(f"--qa requires env vars: {', '.join(missing)}")

    sessions = _load_sessions(args.graph)[: args.sessions]
    total = sum(len(msgs) for _, msgs in sessions)
    print(
        f"driving {args.base}: {len(sessions)} sessions, {total} messages", flush=True
    )

    for session_id, messages in sessions:
        r, _ = _post(
            args.base,
            "/api/v2/memory/add",
            {"session_id": session_id, "messages": messages},
        )
        print(
            f"[add]   {session_id} ({len(messages)} msgs) -> {r['data'].get('status')}",
            flush=True,
        )
        r, _ = _post(
            args.base,
            "/api/v2/memory/flush",
            {"session_id": session_id, "messages": []},
        )
        print(f"[flush] {session_id} -> {r['data'].get('status')}", flush=True)

    if args.memory_root:
        root = pathlib.Path(args.memory_root)
        _wait_settled(root, "*.md", label="memory markdown")
    else:
        time.sleep(30)

    queries = _DEFAULT_QUERIES
    if args.queries:
        with open(args.queries, encoding="utf-8") as f:
            queries = [tuple(item) for item in json.load(f)]

    golden = _load_annotations(args.graph) if args.qa else {}
    qa_records = []
    for query, user_id in queries:
        r, headers = _post(
            args.base,
            "/api/v2/memory/search",
            {"user_id": user_id, "query": query, "method": "hybrid", "top_k": 5},
        )
        data = r.get("data", {})
        episodes = data.get("episodes") or []
        print(f"[search] {user_id} {query!r} -> episodes={len(episodes)}", flush=True)
        for ep in episodes[:5]:
            print(
                f"  hit id={ep.get('id')} score={ep.get('score', 0):.3f} "
                f"subject={str(ep.get('subject'))[:60]}",
                flush=True,
            )
        if args.qa:
            request_id = _header(headers, "X-Request-Id")
            record = _qa_record(query, user_id, request_id, data, golden)
            qa_records.append(record)
            print(
                f"  [qa] request_id={request_id} "
                f"judge_score={record.get('judge_score')}",
                flush=True,
            )

    if args.qa:
        out = pathlib.Path(args.qa_out)
        out.write_text(
            json.dumps({"records": qa_records}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[qa] wrote {len(qa_records)} record(s) -> {out}", flush=True)

    # Final light request: lets the file tracker observe late background writes.
    if args.memory_root:
        _wait_settled(
            pathlib.Path(args.memory_root),
            "*.md",
            label="memory markdown (post-search)",
        )
    _post(
        args.base,
        "/api/v2/memory/search",
        {
            "user_id": queries[0][1],
            "query": "final sync",
            "method": "keyword",
            "top_k": 1,
        },
    )
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Minimal EverOS x Langfuse demo — native OpenTelemetry tracing.

EverOS emits OTel spans for its own memory operations when ``[observability]``
is enabled; this script contains **no instrumentation code**. It just drives a
running server (add -> flush -> search) so the traces the server produces show
up in your Langfuse project.

Prereqs (see README.md):
  1. pip install "everos[otel]"
  2. configure [observability] in everos.toml with your Langfuse keys
  3. everos server start        # defaults to http://127.0.0.1:8000

Then: python demo.py
"""

from __future__ import annotations

import json
import time
import urllib.request

BASE = "http://127.0.0.1:8000"
SESSION = "langfuse_demo"
USER = "alice"


def _post(path: str, body: dict) -> dict:
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return json.load(resp)


def main() -> None:
    ts = int(time.time() * 1000)

    add = _post(
        "/api/v2/memory/add",
        {
            "session_id": SESSION,
            "messages": [
                {
                    "message_id": "m1",
                    "role": "user",
                    "content": "Moved our vector store to LanceDB to fix index bloat.",
                    "timestamp": ts,
                    "sender_id": USER,
                },
                {
                    "message_id": "m2",
                    "role": "assistant",
                    "content": "Noted — LanceDB with compaction keeps it compact.",
                    "timestamp": ts + 1000,
                    "sender_id": "assistant",
                },
            ],
        },
    )
    print("add   ->", add["data"])

    flush = _post("/api/v2/memory/flush", {"session_id": SESSION, "messages": []})
    print("flush ->", flush["data"])

    print("waiting for async index sync ...")
    time.sleep(10)

    for method in ("keyword", "hybrid", "agentic"):
        resp = _post(
            "/api/v2/memory/search",
            {
                "user_id": USER,
                "query": "which vector database did we move to and why",
                "method": method,
                "top_k": 5,
                "filters": {"session_id": SESSION},
            },
        )
        hits = len(resp["data"].get("episodes", []))
        print(f"search[{method}] -> {hits} hit(s)")

    print(
        f"\nOpen Langfuse -> Tracing and filter session.id = {SESSION} "
        "to see the traces (add / flush / search, with token usage and "
        "recall-quality scores)."
    )


if __name__ == "__main__":
    main()

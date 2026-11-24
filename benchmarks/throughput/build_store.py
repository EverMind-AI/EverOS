"""Build the throughput benchmark's store: real conversations, real extraction.

usage: python build_store.py <base_url> <sources_dir> [--plan PLAN] [--workers N]

Ingests conversations into a running EverOS server (configured with real LLM and
embedding providers) under app ``perf`` / project ``real100k``:

  locomo:<k>   LoCoMo10 conversation k; each of its two speakers is a user
               (``locomo<k>_<name>``)
  lme:<i>      LongMemEval-S question i's haystack sessions, one user (``lme_<i>``)

<sources_dir> holds ``locomo10.json`` and ``longmemeval_s_cleaned.json``
(``bash benchmarks/download_data.sh locomo`` / ``longmemeval``). A LongMemEval
session shared by several questions goes to the first of them in plan order, so
no conversation is stored twice. Each session is one or more synchronous
``/memory/add`` calls of at most 50 messages, then ``/memory/flush``. After the
last session, waits until the store stops growing: facts and profiles are
extracted in the background, so it waits for five minutes in a row of an idle
cascade and unchanged index row counts.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import re
import time
from pathlib import Path

import httpx

WHERE = {"app_id": "perf", "project_id": "real100k"}


def lme_ms(date: str) -> int:  # '2023/05/20 (Sat) 02:21'
    d = dt.datetime.strptime(re.sub(r" \(\w+\)", "", date), "%Y/%m/%d %H:%M")
    return int(d.replace(tzinfo=dt.UTC).timestamp() * 1000)


def locomo_ms(date: str) -> int:  # '1:56 pm on 8 May, 2023'
    d = dt.datetime.strptime(date, "%I:%M %p on %d %B, %Y")
    return int(d.replace(tzinfo=dt.UTC).timestamp() * 1000)


def sessions(plan: str, sources: Path) -> list[tuple[str, list[dict]]]:
    """(session_id, messages) for every unit of the plan, in plan order."""
    lme = json.loads((sources / "longmemeval_s_cleaned.json").read_text("utf-8"))
    locomo = json.loads((sources / "locomo10.json").read_text("utf-8"))
    units = []
    for part in plan.split(","):
        kind, span = part.split(":")
        lo, _, hi = span.partition("-")
        units += [(kind, i) for i in range(int(lo), int(hi or lo) + 1)]
    out, seen = [], set()
    for kind, i in units:
        if kind == "lme":
            q, owner = lme[i], f"lme_{i}"
            for sid, sess, date in zip(
                q["haystack_session_ids"],
                q["haystack_sessions"],
                q["haystack_dates"],
                strict=True,
            ):
                if sid in seen:
                    continue
                seen.add(sid)
                t0 = lme_ms(date)
                msgs = [
                    {
                        "sender_id": owner if m["role"] == "user" else "assistant",
                        "role": m["role"],
                        "content": m["content"],
                        "timestamp": t0 + j * 30_000,
                    }
                    for j, m in enumerate(sess)
                    if m.get("content")
                ]
                out.append((f"{owner}_{len(out)}", msgs))
        else:
            conv, k = locomo[i]["conversation"], 1
            while f"session_{k}" in conv:
                t0 = locomo_ms(conv[f"session_{k}_date_time"])
                msgs = [
                    {
                        "sender_id": f"locomo{i}_{m['speaker'].lower()}",
                        "sender_name": m["speaker"],
                        "role": "user",
                        "content": m["text"],
                        "timestamp": t0 + j * 30_000,
                    }
                    for j, m in enumerate(conv[f"session_{k}"])
                    if m.get("text")
                ]
                out.append((f"locomo{i}_s{k}", msgs))
                k += 1
    return [(sid, msgs) for sid, msgs in out if msgs]


async def index_state(client: httpx.AsyncClient) -> tuple[int, dict[str, int]]:
    """The cascade's pending count and the row count of each index table."""
    cascade = (await client.get("/health")).json().get("cascade") or {}
    return cascade.get("pending", -1), cascade.get("index_rows", {})


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("sources", type=Path)
    ap.add_argument("--plan", default="locomo:0-9,lme:0-111")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    todo = sessions(a.plan, a.sources)
    queue: asyncio.Queue[tuple[str, list[dict]]] = asyncio.Queue()
    for item in todo:
        queue.put_nowait(item)
    started, failed = time.monotonic(), []
    async with httpx.AsyncClient(base_url=a.base, timeout=900) as client:

        async def worker() -> None:
            while not queue.empty():
                sid, msgs = queue.get_nowait()
                codes = []
                for i in range(0, len(msgs), 50):
                    body = {"session_id": sid, "messages": msgs[i : i + 50], **WHERE}
                    r = await client.post("/api/v2/memory/add", json=body)
                    codes.append(r.status_code)
                r = await client.post(
                    "/api/v2/memory/flush", json={"session_id": sid, **WHERE}
                )
                codes.append(r.status_code)
                if any(c != 200 for c in codes):
                    failed.append((sid, codes))
                done = len(todo) - queue.qsize()
                print(f"{done}/{len(todo)} {sid} {codes}", flush=True)

        await asyncio.gather(*(worker() for _ in range(a.workers)))
        print(f"ingested in {time.monotonic() - started:.0f} s, failed {failed}")
        calm, last = 0, None
        while calm < 5:
            await asyncio.sleep(60)
            pending, rows = await index_state(client)
            calm = calm + 1 if pending == 0 and rows == last else 0
            last = rows
            print(f"pending {pending} rows {rows}", flush=True)
    print(f"settled after {time.monotonic() - started:.0f} s")


if __name__ == "__main__":
    asyncio.run(main())

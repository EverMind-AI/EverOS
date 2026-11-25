"""Build the throughput benchmark's store: real conversations, real extraction.

usage: python build_store.py <base_url> <sources_dir> --root <everos_root>
                              [--plan PLAN] [--workers N]

Ingests conversations into a running EverOS server (configured with real LLM and
embedding providers) under app ``perf`` / project ``real100k``:

  locomo:<k>   LoCoMo10 conversation k; each of its two speakers is a user
               (``locomo<k>_<name>``)
  lme:<i>      LongMemEval-S question i's haystack sessions, one user (``lme_<i>``)

<sources_dir> holds ``locomo10.json`` and ``longmemeval_s_cleaned.json``
(``bash benchmarks/download_data.sh locomo``; the cleaned LongMemEval-S as in
benchmarks/throughput.md, Reproduce step 1). A LongMemEval
session shared by several questions goes to the first of them in plan order, so
no conversation is stored twice. Each session is one or more synchronous
``/memory/add`` calls of at most 50 messages, then ``/memory/flush``; a call
that fails (not 200, or no answer) is sent again, up to ``--attempts`` times. A
session still failing then is left unfinished, and once every worker is done the
build exits 1 listing them: a store missing sessions is not the documented one.
On EverOS 1.8.0 or later re-running it is safe: a resent message is not stored
twice. Otherwise it waits until the store stops growing: facts and profiles
are extracted in the background, and no HTTP signal tells when that work is
done, so it waits for five minutes in a row of an idle cascade, an empty
memorize queue and an unchanged markdown tree under --root (the server's
EVEROS_ROOT; extraction ends by writing markdown, which the cascade indexes).
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import os
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


async def post(
    client: httpx.AsyncClient, path: str, body: dict, attempts: int, wait: float
) -> int | str:
    """POST until a 200, at most ``attempts`` times; the last status or error."""
    status: int | str = "not sent"
    for attempt in range(attempts):
        if attempt:
            await asyncio.sleep(wait * attempt)
        try:
            status = (await client.post(path, json=body)).status_code
        except httpx.HTTPError as e:
            status = f"error:{type(e).__name__}"
        if status == 200:
            break
    return status


def markdown_state(root: Path) -> tuple[int, int, int]:
    """Count, total size and newest mtime of the markdown files under ``root``."""
    count = size = newest = 0
    for top, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d != ".index"]
        for name in files:
            if name.endswith(".md"):
                st = os.stat(os.path.join(top, name))
                count, size, newest = (
                    count + 1,
                    size + st.st_size,
                    max(newest, st.st_mtime_ns),
                )
    return count, size, newest


async def background_work(client: httpx.AsyncClient) -> tuple[int, int]:
    """Changes the cascade has yet to index, and batches the memorize queue holds."""
    health = (await client.get("/health")).json()
    cascade = health.get("cascade") or {}
    queue = health.get("memorize_queue") or {}
    return cascade.get("pending", -1), queue.get("pending", 0)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("sources", type=Path)
    ap.add_argument("--root", type=Path, required=True, help="the server's EVEROS_ROOT")
    ap.add_argument("--plan", default="locomo:0-9,lme:0-111")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--retry-wait", type=float, default=5.0)
    a = ap.parse_args()
    todo = sessions(a.plan, a.sources)
    queue: asyncio.Queue[tuple[str, list[dict]]] = asyncio.Queue()
    for item in todo:
        queue.put_nowait(item)
    started, failed, finished = time.monotonic(), [], 0
    async with httpx.AsyncClient(base_url=a.base, timeout=900) as client:

        async def worker() -> None:
            nonlocal finished
            while not queue.empty():
                sid, msgs = queue.get_nowait()
                calls = [
                    (
                        "/api/v2/memory/add",
                        {"session_id": sid, "messages": msgs[i : i + 50]},
                    )
                    for i in range(0, len(msgs), 50)
                ] + [("/api/v2/memory/flush", {"session_id": sid})]
                codes: list[int | str] = []
                for path, body in calls:
                    codes.append(
                        await post(client, path, body | WHERE, a.attempts, a.retry_wait)
                    )
                    if codes[-1] != 200:
                        failed.append((sid, codes))
                        break
                finished += 1
                print(f"{finished}/{len(todo)} {sid} {codes}", flush=True)

        await asyncio.gather(*(worker() for _ in range(a.workers)))
        print(f"ingested in {time.monotonic() - started:.0f} s", flush=True)
        if failed:
            for sid, codes in failed:
                print(f"FAILED {sid} {codes}", flush=True)
            raise SystemExit(f"{len(failed)} sessions failed; the store is incomplete")
        calm, last = 0, None
        while calm < 5:
            await asyncio.sleep(60)
            pending, queued = await background_work(client)
            md = markdown_state(a.root)
            calm = calm + 1 if pending == 0 and queued == 0 and md == last else 0
            last = md
            print(f"cascade {pending} queue {queued} markdown {md[:2]}", flush=True)
    print(f"settled after {time.monotonic() - started:.0f} s")


if __name__ == "__main__":
    asyncio.run(main())

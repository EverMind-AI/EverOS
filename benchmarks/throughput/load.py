"""Open-loop search load at one fixed rate, and whether the server kept up.

usage: python load.py --base URL --method hybrid --rate 8 --seconds 60
                      --queries queries.jsonl --out DIR/hybrid_8 [--pid SERVER_PID]
                      [--senders N]

Request i goes out at start + i/rate whether or not earlier ones have returned,
so a server that cannot keep up shows as climbing latency, not as a lower rate.
Each request is POST /api/v2/memory/search for a (user_id, query) line drawn from
--queries. Writes <out>.jsonl (one line per request) and <out>.summary.json.

Kept up = every request returned 200, throughput reached 90% of the rate, and
latency did not climb during the step: the median of the last third of requests
stays under twice the first third's (or within 0.5 s of it). Throughput counts
requests both sent and completed while requests were still being sent, so a tail
after the last send cannot hide a backlog, and a server slow from the first
request on fails on throughput even when its latency stays flat.

With --pid (Linux), /proc is read once a second while requests go out: the
server's CPU in cores (process and children) and its peak resident memory.

One Python process cannot keep its send schedule much past 140 requests/s on the
benchmark's client cores. --senders N splits the rate over N processes, each with
its own query order (seed + k); the step is judged on all their requests merged in
send order.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import httpx

_TICK = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100


def pct(values: list[float], p: float) -> float | None:
    s = sorted(values)
    return s[min(len(s) - 1, round(p / 100 * (len(s) - 1)))] if s else None


def _tree(pid: int) -> list[int]:
    """``pid`` and its descendants, from /proc."""
    children: dict[int, list[int]] = {}
    for entry in os.listdir("/proc"):
        if entry.isdigit():
            try:
                stat = Path(f"/proc/{entry}/stat").read_text()
            except OSError:
                continue
            ppid = int(stat.rsplit(")", 1)[1].split()[1])
            children.setdefault(ppid, []).append(int(entry))
    out, todo = [], [pid]
    while todo:
        p = todo.pop()
        out.append(p)
        todo += children.get(p, [])
    return out


def _usage(pid: int) -> tuple[float, float]:
    """CPU seconds used so far and resident MB, summed over the process tree."""
    cpu, rss_kb = 0.0, 0
    for p in _tree(pid):
        try:
            fields = Path(f"/proc/{p}/stat").read_text().rsplit(")", 1)[1].split()
            status = Path(f"/proc/{p}/status").read_text()
        except OSError:
            continue
        cpu += (int(fields[11]) + int(fields[12])) / _TICK  # utime + stime
        rss_kb += next(
            (int(x.split()[1]) for x in status.splitlines() if x.startswith("VmRSS:")),
            0,
        )
    return cpu, rss_kb / 1024


async def sample(pid: int, stop: asyncio.Event, out: dict) -> None:
    cpu0, rss = await asyncio.to_thread(_usage, pid)
    t0, peak = time.monotonic(), rss
    while not stop.is_set():
        await asyncio.sleep(1)
        peak = max(peak, (await asyncio.to_thread(_usage, pid))[1])
    cpu1, rss = await asyncio.to_thread(_usage, pid)
    out["cpu_cores"] = round((cpu1 - cpu0) / (time.monotonic() - t0), 2)
    out["rss_mb_max"] = round(max(peak, rss))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--rate", type=float, required=True)
    ap.add_argument("--seconds", type=float, required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--app", default="perf")
    ap.add_argument("--project", default="real100k")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--pid", type=int)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--senders", type=int, default=1)
    a = ap.parse_args()
    lines = Path(a.queries).read_text(encoding="utf-8").splitlines()
    queries = [json.loads(line) for line in lines]
    if a.senders == 1:
        rows, res = asyncio.run(send(a, queries))
    else:
        jobs = [(a, queries, k) for k in range(a.senders)]
        with ProcessPoolExecutor(a.senders) as pool:
            parts = list(pool.map(sender, jobs))
        rows = [row for part, _ in parts for row in part]
        res = parts[0][1]
    rows.sort(key=lambda r: r["sent"])
    Path(f"{a.out}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    summary = judge(a, rows) | res
    Path(f"{a.out}.summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


def sender(job: tuple[argparse.Namespace, list[dict], int]) -> tuple[list[dict], dict]:
    """Sender k of --senders: its share of the rate, its own seed; k 0 samples --pid."""
    a, queries, k = job
    share = argparse.Namespace(**vars(a))
    share.rate, share.seed = a.rate / a.senders, a.seed + k
    share.pid = a.pid if k == 0 else None
    rows, res = asyncio.run(send(share, queries))
    return [r | {"sender": k} for r in rows], res


async def send(a: argparse.Namespace, queries: list[dict]) -> tuple[list[dict], dict]:
    """Send the step's requests on schedule; the rows and the resource sample."""
    rng = random.Random(a.seed)
    rows: list[dict] = []
    res: dict = {}
    limits = httpx.Limits(max_connections=1000, max_keepalive_connections=1000)
    client = httpx.AsyncClient(base_url=a.base, timeout=a.timeout, limits=limits)
    async with client as c:

        async def one(i: int) -> None:
            q = rng.choice(queries)
            body = {
                "user_id": q["user_id"],
                "app_id": a.app,
                "project_id": a.project,
                "query": q["query"],
                "method": a.method,
                "top_k": a.top_k,
            }
            row: dict = {"i": i, "sent": time.time()}
            t0 = time.perf_counter()
            try:
                r = await c.post("/api/v2/memory/search", json=body)
                row["status"] = r.status_code
                if r.status_code != 200:
                    row["body"] = r.text[:160]
            except httpx.HTTPError as e:
                row["status"] = f"error:{type(e).__name__}"
            row["s"] = round(time.perf_counter() - t0, 4)
            rows.append(row)

        stop = asyncio.Event()
        sampler = asyncio.create_task(sample(a.pid, stop, res)) if a.pid else None
        n, start, tasks = int(a.rate * a.seconds), time.perf_counter(), []
        for i in range(n):
            delay = start + i / a.rate - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            tasks.append(asyncio.create_task(one(i)))
        stop.set()
        if sampler:
            await sampler
        await asyncio.gather(*tasks)
    return rows, res


def judge(a: argparse.Namespace, rows: list[dict]) -> dict:
    ok = [r for r in rows if r["status"] == 200]
    t0 = rows[0]["sent"]
    span = rows[-1]["sent"] - t0
    third = len(rows) // 3
    early = pct([r["s"] for r in rows[:third] if r["status"] == 200], 50) or 0.0
    late = pct([r["s"] for r in rows[-third:] if r["status"] == 200], 50) or 0.0
    lat = [r["s"] for r in ok]
    throughput = sum(r["sent"] + r["s"] <= t0 + span for r in ok) / span
    return {
        "method": a.method,
        "rate": a.rate,
        "sent": len(rows),
        "ok": len(ok),
        "throughput": round(throughput, 1),
        "p50_ms": round(1000 * (pct(lat, 50) or 0)),
        "p95_ms": round(1000 * (pct(lat, 95) or 0)),
        "keeps_up": len(ok) == len(rows)
        and throughput >= 0.9 * a.rate
        and late <= max(2 * early, early + 0.5),
    }


if __name__ == "__main__":
    main()

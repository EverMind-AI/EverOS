"""Tables for benchmarks/throughput.md from sweep.sh output.

usage: python summarize.py <results_dir>

Reads <results_dir>/<server>/<round>/<method>/<method>_<rate>.summary.json,
one sweep.sh run per (server, round, method). Prints two Markdown tables:

* sustained rate: per method and server, the highest rate that every round
  of that method and server kept up at, with the worst round's p50 / p95
  latency, CPU and peak memory at that rate, and the highest rate any single
  round kept up at. ">=" marks a lower bound: no round fell behind at any rate
  tried, or the rate above it failed only where the load generator stalled
  (the server used under 0.3 cores while requests went out: load.py could not
  send faster, see benchmarks/throughput.md);
* every step: each round's result at each rate.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

root = Path(sys.argv[1])
steps: dict[tuple[str, str], dict[float, dict[str, dict]]] = defaultdict(
    lambda: defaultdict(dict)
)
for f in sorted(root.glob("*/*/*/*.summary.json")):
    server, rnd, method = f.parts[-4:-1]
    s = json.loads(f.read_text())
    steps[(method, server)][s["rate"]][rnd] = s

_STALLED_CORES = 0.3


def cell(rows: list[dict], key: str, unit: str = "") -> str:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return f"{max(vals):g}{unit}" if vals else "-"


print(
    "| Method | Server | Sustained QPS | p50 | p95 | CPU (cores) | Peak memory "
    "| Best round |"
)
print("|---|---|---|---|---|---|---|---|")
for (method, server), per in sorted(steps.items()):
    rounds = {rnd for by in per.values() for rnd in by}
    held = [
        rate
        for rate, by in sorted(per.items())
        if len(by) == len(rounds) and all(s["keeps_up"] for s in by.values())
    ]
    kept = [rate for rate, by in per.items() if any(s["keeps_up"] for s in by.values())]
    best = f"{max(kept):g}" if kept else "-"
    if not held:
        print(f"| {method} | {server} | below {min(per):g} | | | | | {best} |")
        continue
    top = max(held)
    above = [by for rate, by in sorted(per.items()) if rate > top]
    bound = not above or all(
        s["keeps_up"] or (s.get("cpu_cores") or 0) < _STALLED_CORES
        for s in above[0].values()
    )
    rows = list(per[top].values())
    print(
        f"| {method} | {server} | {'>=' if bound else ''}{top:g} "
        f"| {cell(rows, 'p50_ms', ' ms')} | {cell(rows, 'p95_ms', ' ms')} "
        f"| {cell(rows, 'cpu_cores')} | {cell(rows, 'rss_mb_max', ' MB')} | {best} |"
    )

print()
print(
    "| Method | Server | QPS | Round | OK | Done/s | p50 | p95 | CPU | Memory "
    "| Kept up |"
)
print("|---|---|---|---|---|---|---|---|---|---|---|")
for (method, server), per in sorted(steps.items()):
    for rate, by in sorted(per.items()):
        for rnd, s in sorted(by.items()):
            print(
                f"| {method} | {server} | {rate:g} | {rnd} | {s['ok']}/{s['sent']} "
                f"| {s['throughput']:g} | {s['p50_ms']} ms | {s['p95_ms']} ms "
                f"| {s.get('cpu_cores', '-')} | {s.get('rss_mb_max', '-')} MB "
                f"| {'yes' if s['keeps_up'] else 'no'} |"
            )

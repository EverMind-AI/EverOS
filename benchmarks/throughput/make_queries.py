"""The benchmark's queries: the datasets' own questions, each sent as its user.

usage: python make_queries.py <store> <sources_dir> <out.jsonl>

One JSON line per (user_id, query): ``lme_<i>`` asks LongMemEval question i;
``locomo<k>_<speaker>`` asks each LoCoMo10 question of conversation k that
names that speaker (a question naming neither goes to both). Only users present
in <store> (an EverOS root built by build_store.py) are used.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

store, sources, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
users = {p.name for p in (store / "perf" / "real100k" / "users").iterdir()}
lme = json.loads((sources / "longmemeval_s_cleaned.json").read_text("utf-8"))
locomo = json.loads((sources / "locomo10.json").read_text("utf-8"))
rows = [
    {"user_id": u, "query": lme[int(u[4:])]["question"]}
    for u in sorted(users)
    if u.startswith("lme_")
]
for k, conv in enumerate(locomo):
    speakers = [conv["conversation"]["speaker_a"], conv["conversation"]["speaker_b"]]
    for qa in conv["qa"]:
        q = qa.get("question") or ""
        named = [s for s in speakers if s.lower() in q.lower()] or speakers
        rows += [
            {"user_id": f"locomo{k}_{s.lower()}", "query": q}
            for s in named
            if q and f"locomo{k}_{s.lower()}" in users
        ]
out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8")
print(f"{len(rows)} queries for {len({r['user_id'] for r in rows})} users -> {out}")

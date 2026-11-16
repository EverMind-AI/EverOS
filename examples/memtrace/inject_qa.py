"""Inject LoCoMo QA results into exported EverOS MemTrace graphs.

The file-tracking tracer records everything up to the assembled
``memory_context``; the QA answer stage runs out-of-process (the drive
script), so this post-hoc step stitches ``prompt`` / ``prediction`` /
``golden_answers`` / ``judge_response`` onto the recorded query nodes with
the native MemTraceBench vocabulary (see ``everos.infra.memtrace.qa``).

Run against a stopped server, after ``filetrace_locomo_drive.py
--qa`` has written ``qa_results.json``::

    python examples/memtrace/inject_qa.py \
        --graph-dir <memory-root>/memtrace \
        --qa-results qa_results.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph-dir", required=True, help="Directory holding the exported graph JSONs"
    )
    parser.add_argument(
        "--qa-results",
        required=True,
        help="qa_results.json written by filetrace_locomo_drive.py --qa",
    )
    args = parser.parse_args()

    from everos.infra.memtrace.qa import inject_qa_dir

    payload = json.loads(pathlib.Path(args.qa_results).read_text(encoding="utf-8"))
    records = payload.get("records", payload) if isinstance(payload, dict) else payload
    injected = inject_qa_dir(pathlib.Path(args.graph_dir), records)
    for name, count in injected.items():
        print(f"[inject] {name}: {count} record(s)", flush=True)
    print(
        f"injected {sum(injected.values())} record(s) into {len(injected)} graph(s)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

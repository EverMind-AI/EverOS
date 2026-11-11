"""One-shot dumper: extract a search-test seed from a corpus snapshot.

Reads the LanceDB tables under
``/tmp/everos_corpus_v2/.index/lancedb/`` (the snapshot produced by
``tests/e2e/test_add_flush_user_pipeline_e2e.py`` with ``EVEROS_KEEP_CORPUS_TO``
set), samples a small representative slice, and emits JSON fixtures
under ``tests/fixtures/search_seed/``.

Sampling rules:

- **episode**: the first 8 per owner (= 16 total), so the filter tests
  have several sessions' worth of rows to slice.
- **user_profile**: 1 per owner (= 2 total).

Run::

    python tests/fixtures/_dump_search_seed.py

Re-run any time the corpus changes; output JSON is committed to
git so other contributors don't need the corpus locally.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import lancedb

CORPUS = Path("/tmp/everos_corpus_v2/.index/lancedb")
OUT_DIR = Path(__file__).parent / "search_seed"
ALL_OWNERS = ("caroline", "melanie")
EPISODES_PER_OWNER = 8


def _serialise(row: dict[str, Any]) -> dict[str, Any]:
    """Make a LanceDB row dict JSON-safe (numpy → list, datetime → ISO)."""
    out: dict[str, Any] = {}
    for k, v in row.items():
        if v is None:
            out[k] = None
        elif hasattr(v, "tolist"):  # numpy ndarray (vector)
            out[k] = v.tolist()
        elif isinstance(v, datetime):
            out[k] = v.isoformat()
        else:
            out[k] = v
    return out


def _read(db: lancedb.DBConnection, table: str) -> list[dict[str, Any]]:
    if f"{table}.lance" not in {p.name for p in CORPUS.iterdir()}:
        raise FileNotFoundError(f"corpus table missing: {table}")
    return db.open_table(table).to_arrow().to_pylist()


def main() -> None:
    if not CORPUS.exists():
        print(f"corpus not found: {CORPUS}", file=sys.stderr)
        print("hint: run the add+flush pipeline first with", file=sys.stderr)
        print("      EVEROS_KEEP_CORPUS_TO=/tmp/everos_corpus_v2", file=sys.stderr)
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(CORPUS))

    # 1) episodes — the first few per owner.
    eps_all = _read(db, "episode")
    eps: list[dict[str, Any]] = []
    for owner in ALL_OWNERS:
        eps.extend([r for r in eps_all if r["owner_id"] == owner][:EPISODES_PER_OWNER])

    # 2) user_profile — 1 per owner.
    ups_all = _read(db, "user_profile")
    ups = [r for r in ups_all if r["owner_id"] in ALL_OWNERS]

    written: list[tuple[str, int, int]] = []
    for name, rows in (("episode", eps), ("user_profile", ups)):
        serialised = [_serialise(r) for r in rows]
        out = OUT_DIR / f"{name}.json"
        out.write_text(json.dumps(serialised, indent=2, default=str))
        written.append((name, len(serialised), out.stat().st_size))

    for name, count, size in written:
        print(f"  {name:14s}: {count:3d} rows  ({size // 1024} KB)")


if __name__ == "__main__":
    main()

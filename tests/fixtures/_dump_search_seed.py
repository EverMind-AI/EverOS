"""One-shot dumper: extract a search-test seed from a corpus snapshot.

Reads the LanceDB tables under
``/tmp/everos_corpus_v2/.index/lancedb/`` (the snapshot produced by
``tests/e2e/test_add_flush_user_pipeline_e2e.py`` with ``EVEROS_KEEP_CORPUS_TO``
set), samples a small representative slice, and emits JSON fixtures
under ``tests/fixtures/search_seed/``.

Sampling rules:

- **episode**: first 8 rows per owner (caroline + melanie). Captures
  the parent_id (= memcell_id) set so downstream tables can be
  bridge-consistent.
- **atomic_fact**: every row whose ``parent_id`` is in the episode-
  parent set above, capped at 50 to keep the seed compact. This
  guarantees hierarchical-eviction testing can verify "facts sharing a
  memcell with the matched episode get embedded".
- **foresight**: 5 per owner. Archived for future use; current
  ``/search`` does not query foresight, so the seed only exists so
  downstream tests can opt in without re-cutting the corpus.
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

    # 1) episodes — first 8 per owner.
    eps_all = _read(db, "episode")
    eps: list[dict[str, Any]] = []
    parent_memcells: set[str] = set()
    for owner in ALL_OWNERS:
        owned = [r for r in eps_all if r["owner_id"] == owner][:8]
        eps.extend(owned)
        for r in owned:
            parent_memcells.add(r["parent_id"])

    # 2) atomic_facts — every fact whose parent_id is in the episode
    #    parent set, capped to keep the seed compact (and so hierarchical
    #    ``facts_for_episodes`` has a useful but bounded pool to
    #    bucket back into episodes).
    afs_all = _read(db, "atomic_fact")
    # Atomic facts fan out per-owner (a single fact about a memcell that
    # mentions two users gets two rows, one for each owner) — sampling
    # naively can leave one owner with zero facts. Take per-owner caps
    # so both caroline and melanie have facts whose parent_id matches
    # their own episodes' parent_id (memcell bridge).
    afs: list[dict[str, Any]] = []
    for owner in ALL_OWNERS:
        afs.extend(
            [
                r
                for r in afs_all
                if r["owner_id"] == owner and r["parent_id"] in parent_memcells
            ][:10]
        )

    # 3) foresights — 5 per owner, archived for future use.
    fss_all = _read(db, "foresight")
    fss: list[dict[str, Any]] = []
    for owner in ALL_OWNERS:
        fss.extend([r for r in fss_all if r["owner_id"] == owner][:5])

    # 4) user_profile — 1 per owner.
    ups_all = _read(db, "user_profile")
    ups = [r for r in ups_all if r["owner_id"] in ALL_OWNERS]

    written: list[tuple[str, int, int]] = []
    for name, rows in (
        ("episode", eps),
        ("atomic_fact", afs),
        ("foresight", fss),
        ("user_profile", ups),
    ):
        serialised = [_serialise(r) for r in rows]
        out = OUT_DIR / f"{name}.json"
        out.write_text(json.dumps(serialised, indent=2, default=str))
        written.append((name, len(serialised), out.stat().st_size))

    for name, count, size in written:
        print(f"  {name:14s}: {count:3d} rows  ({size // 1024} KB)")
    print(f"  parent_memcells captured: {len(parent_memcells)}")


if __name__ == "__main__":
    main()

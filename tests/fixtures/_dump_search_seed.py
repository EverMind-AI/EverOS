"""One-shot dumper: extract a search-test seed from a corpus snapshot.

Reads the LanceDB tables under
``/tmp/everos_corpus_v2/.index/lancedb/`` (the snapshot produced by
``tests/e2e/test_add_flush_user_pipeline_e2e.py`` with ``EVEROS_KEEP_CORPUS_TO``
set), samples a small representative slice, and emits JSON fixtures
under ``tests/fixtures/search_seed/``.

Sampling rules:

- **episode + atomic_fact**: sampled together for fact↔episode
  coherence. Facts link to their host episode via
  ``atomic_fact.parent_id == episode.entry_id`` (parent_type="episode";
  see ``memory/strategies/extract_atomic_facts.py``). Episodes that host
  facts are picked first (up to 8/owner) so hierarchical-eviction and the
  fact-first paths (vector MaxSim, agentic) have a non-trivial,
  multi-episode corpus; facts are then kept iff their host episode made
  the cut (≤15/owner), guaranteeing every kept fact bridges back.
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

    # 1) episodes + 2) atomic_facts — sampled together so the slice is
    #    fact↔episode coherent AND rich (facts spread across several
    #    episodes, not piled on one).
    #
    # Current extraction links a fact to its host episode via
    # ``atomic_fact.parent_id == episode.entry_id`` (parent_type="episode";
    # see ``memory/strategies/extract_atomic_facts.py``). The fact-first
    # search paths (vector MaxSim, agentic) fetch episodes by that
    # entry_id, so the seed must preserve that linkage. Sample the
    # episodes that actually HAVE facts first (up to 8/owner), so the
    # agentic LLM-sufficiency step and hierarchical eviction have a
    # non-trivial, multi-episode corpus to work with; only then fall back
    # to fact-less episodes to top up owner coverage.
    eps_all = _read(db, "episode")
    afs_all = _read(db, "atomic_fact")

    eps: list[dict[str, Any]] = []
    afs: list[dict[str, Any]] = []
    for owner in ALL_OWNERS:
        owner_eps = [r for r in eps_all if r["owner_id"] == owner]
        owner_facts = [r for r in afs_all if r["owner_id"] == owner]
        facts_by_ep: dict[str, list[dict[str, Any]]] = {}
        for f in owner_facts:
            facts_by_ep.setdefault(f["parent_id"], []).append(f)
        # Episodes that host facts come first (richest bridges), then the
        # rest — capped at 8 to keep the seed compact.
        with_facts = [e for e in owner_eps if e["entry_id"] in facts_by_ep]
        without_facts = [e for e in owner_eps if e["entry_id"] not in facts_by_ep]
        chosen = (with_facts + without_facts)[:8]
        eps.extend(chosen)
        # Spread facts across episodes (<=3 per host) so several episodes are
        # bridged, not one — richer corpus for agentic + hierarchical
        # eviction. Every kept fact bridges back via its host entry_id.
        owner_afs: list[dict[str, Any]] = []
        for e in chosen:
            owner_afs.extend(facts_by_ep.get(e["entry_id"], [])[:3])
        afs.extend(owner_afs[:15])

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
    bridged = len({f["parent_id"] for f in afs})
    print(f"  fact-bridged episodes: {bridged}")


if __name__ == "__main__":
    main()

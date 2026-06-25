#!/usr/bin/env python
"""Check md ↔ LanceDB consistency for an everos corpus.

Three checks per kind:
  1. id set equality              — md entry ids == LanceDB row entry_ids
  2. content_sha256 equality      — every shared id matches on both sides
  3. id monotonicity (md-only)    — within each daily-log md, the numeric
                                    counter at the end of entry.id ascends
                                    from 1 with no gap and no dupe

Two modes:
  --mode lifespan (default)   Full strict check through the everos app
                              lifespan stack (sqlite + lance + cascade +
                              ome). Safe ONLY on an idle corpus (no live
                              server writing). Covers every kind in
                              KIND_REGISTRY.
  --mode readonly             Bypass the lifespan stack, open LanceDB with
                              a fresh read connection, read md directly.
                              Safe even on an active corpus, but only
                              covers the three daily-log kinds (episode /
                              atomic_fact / foresight).

Examples:
  scripts/check_consistency.py ~/.everos-locomo-all-kv-fast
  scripts/check_consistency.py ~/.everos-locomo-all-kv-fast --mode readonly
  scripts/check_consistency.py ~/.everos-locomo-all-kv-fast --owners joanna,nate
"""
# This script must mutate sys.path before importing everos/tests, and
# uses synchronous pathlib because it's a one-shot CLI, not server code.
# ruff: noqa: E402, ASYNC240

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import os
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")


# ── shared: id counter parsing ──────────────────────────────────────────

_ID_NUM_RE = re.compile(r"_(\d+)$")


def _entry_counter(entry_id: str) -> int | None:
    m = _ID_NUM_RE.search(entry_id)
    return int(m.group(1)) if m else None


@dataclasses.dataclass
class MonotonicityReport:
    path: str
    total: int
    not_sorted: bool
    starts_at_1: bool
    gaps: list[int]
    dupes: list[int]
    bad_format: list[str]

    @property
    def ok(self) -> bool:
        return self.total == 0 or (
            not self.not_sorted
            and self.starts_at_1
            and not self.gaps
            and not self.dupes
            and not self.bad_format
        )


async def _scan_monotonicity(corpus: Path) -> list[MonotonicityReport]:
    """Walk all daily-log md files; report id-counter monotonicity per file."""
    from everos.core.persistence import MarkdownReader

    daily_dirs = ("/episodes/", "/.atomic_facts/", "/.foresights/", "/.agent_cases/")
    reports: list[MonotonicityReport] = []
    for md in sorted(corpus.rglob("*.md")):
        rel = md.relative_to(corpus).as_posix()
        if not (rel.startswith("users/") or rel.startswith("agents/")):
            continue
        if not any(d in "/" + rel for d in daily_dirs):
            continue
        parsed = await MarkdownReader.read(md)
        counters: list[int] = []
        bad_format: list[str] = []
        for entry in parsed.entries:
            c = _entry_counter(entry.id)
            if c is None:
                bad_format.append(entry.id)
            else:
                counters.append(c)
        not_sorted = counters != sorted(counters)
        starts_at_1 = bool(counters) and min(counters) == 1
        gaps: list[int] = []
        dupes: list[int] = []
        if counters:
            seen = set(counters)
            for i in range(1, max(counters) + 1):
                if i not in seen:
                    gaps.append(i)
            cc = Counter(counters)
            dupes = sorted(v for v, n in cc.items() if n > 1)
        reports.append(
            MonotonicityReport(
                path=rel,
                total=len(parsed.entries),
                not_sorted=not_sorted,
                starts_at_1=starts_at_1 if parsed.entries else True,
                gaps=gaps,
                dupes=dupes,
                bad_format=bad_format,
            )
        )
    return reports


def _print_monotonicity(reports: list[MonotonicityReport]) -> int:
    issues = sum(1 for r in reports if not r.ok)
    if issues == 0:
        print(
            f"  all {len(reports)} daily-log md files have strictly ascending"
            " ids from 1"
        )
        return 0
    print(f"  ⚠ {issues}/{len(reports)} md files have id-counter issues:")
    for r in reports:
        if r.ok:
            continue
        problems = []
        if r.not_sorted:
            problems.append("not-sorted")
        if not r.starts_at_1 and r.total:
            problems.append("not-from-1")
        if r.gaps:
            preview = r.gaps[:5]
            problems.append(f"gaps={preview}{'...' if len(r.gaps) > 5 else ''}")
        if r.dupes:
            problems.append(f"dupes={r.dupes}")
        if r.bad_format:
            problems.append(f"bad-format×{len(r.bad_format)}")
        print(f"    {r.path}: total={r.total} {' '.join(problems)}")
    return issues


# ── mode: lifespan ──────────────────────────────────────────────────────


async def run_lifespan_mode(corpus: Path) -> int:
    """Full strict check via app lifespan; covers every kind in KIND_REGISTRY."""
    os.environ["EVEROS_ROOT"] = str(corpus)
    from everos.config import load_settings

    load_settings.cache_clear()

    from everos.entrypoints.api.app import create_app
    from tests._consistency_assertions import assert_md_lance_strict_consistent

    app = create_app()
    rc = 0
    async with app.router.lifespan_context(app):
        # 1+2. id set + sha
        print("─── md ↔ LanceDB strict consistency ───")
        try:
            stats = await assert_md_lance_strict_consistent(corpus)
            print("  PASS")
        except AssertionError as e:
            print(f"  DRIFT:\n{e}")
            rc = 1
            stats = None

        if stats is not None:
            print()
            print(
                f"  {'kind':<15s} {'md_files':>10s}"
                f" {'md_entries':>12s} {'lance_rows':>12s}"
            )
            print("  " + "─" * 53)
            for kind, s in stats.items():
                print(
                    f"  {kind:<15s} {s.md_file_count:>10d}"
                    f" {s.md_entry_count:>12d} {s.lance_row_count:>12d}"
                )

        # 3. id monotonicity
        print()
        print("─── id monotonicity ───")
        reports = await _scan_monotonicity(corpus)
        if _print_monotonicity(reports) > 0:
            rc = max(rc, 2)
    return rc


# ── mode: readonly ──────────────────────────────────────────────────────


async def run_readonly_mode(corpus: Path, owners_filter: list[str] | None) -> int:
    """Direct LanceDB read + md read; no lifespan / cascade / ome started.

    Covers the three daily-log kinds; agent_case + user_profile + agent_skill
    are NOT checked in this mode (use --mode lifespan on an idle corpus
    snapshot for full coverage).
    """
    import lancedb

    from everos.core.persistence import MarkdownReader
    from everos.memory.cascade.handlers.atomic_fact import AtomicFactHandler
    from everos.memory.cascade.handlers.episode import EpisodeHandler
    from everos.memory.cascade.handlers.foresight import ForesightHandler
    from tests._consistency_assertions import _daily_log_sha_for_entry

    db = lancedb.connect(str(corpus / ".index" / "lancedb"))

    kinds = [
        ("episode", "episodes", "episode-", EpisodeHandler),
        ("atomic_fact", ".atomic_facts", "atomic_fact-", AtomicFactHandler),
        ("foresight", ".foresights", "foresight-", ForesightHandler),
    ]

    # Pick owners
    if owners_filter:
        owners = owners_filter
    else:
        owners = (
            sorted(p.name for p in (corpus / "users").iterdir() if p.is_dir())
            if (corpus / "users").exists()
            else []
        )

    print("─── md ↔ LanceDB consistency (readonly) ───")
    rc = 0
    for table_name, dir_name, prefix, handler_cls in kinds:
        try:
            table = db.open_table(table_name)
        except FileNotFoundError:
            print(f"  {table_name}: table not in lancedb (skip)")
            continue
        for owner in owners:
            md_dir = corpus / "users" / owner / dir_name
            if not md_dir.exists():
                continue
            md_files = sorted(md_dir.glob(f"{prefix}*.md"))
            md_sha_total: dict[str, str] = {}
            for md in md_files:
                parsed = await MarkdownReader.read(md)
                for entry in parsed.entries:
                    md_sha_total[entry.id] = _daily_log_sha_for_entry(
                        handler_cls, entry.as_structured()
                    )
            arr = (
                table.search().where(f"owner_id = '{owner}'").limit(100_000).to_arrow()
            )
            lance_sha = dict(
                zip(
                    arr["entry_id"].to_pylist(),
                    arr["content_sha256"].to_pylist(),
                    strict=True,
                )
            )
            only_md = sorted(set(md_sha_total) - set(lance_sha))
            only_lance = sorted(set(lance_sha) - set(md_sha_total))
            sha_mismatch = sorted(
                k
                for k in set(md_sha_total) & set(lance_sha)
                if md_sha_total[k] != lance_sha[k]
            )
            ok = not (only_md or only_lance or sha_mismatch)
            status = "OK" if ok else "DRIFT"
            if not ok:
                rc = 1
            print(
                f"  {table_name:<12s} owner={owner:<12s}"
                f" md={len(md_sha_total):5d} lance={len(lance_sha):5d}"
                f"  {status}"
            )
            if only_md:
                print(f"    only_in_md (first 5):    {only_md[:5]}")
            if only_lance:
                print(f"    only_in_lance (first 5): {only_lance[:5]}")
            if sha_mismatch:
                print(f"    sha_mismatch (first 5):  {sha_mismatch[:5]}")

    # id monotonicity (md-only, owner-filtered if provided)
    print()
    print("─── id monotonicity ───")
    reports = await _scan_monotonicity(corpus)
    if owners_filter:
        owner_paths = tuple(f"users/{o}/" for o in owners_filter)
        reports = [r for r in reports if any(r.path.startswith(p) for p in owner_paths)]
    if _print_monotonicity(reports) > 0:
        rc = max(rc, 2)
    return rc


# ── main ────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("corpus", help="memory root (e.g. ~/.everos-locomo-all-kv-fast)")
    p.add_argument(
        "--mode",
        choices=("lifespan", "readonly"),
        default="lifespan",
        help="lifespan = full strict check (idle corpus only); "
        "readonly = direct lance read (safe on active corpus)",
    )
    p.add_argument(
        "--owners",
        help="comma-separated owner filter (readonly mode only)",
    )
    return p.parse_args()


async def main() -> int:
    args = _parse_args()
    corpus = Path(args.corpus).expanduser().resolve()
    if not corpus.exists():
        print(f"ERROR: corpus does not exist: {corpus}")
        return 1
    owners = (
        [o.strip() for o in args.owners.split(",") if o.strip()]
        if args.owners
        else None
    )
    print(f"corpus: {corpus}")
    print(f"mode:   {args.mode}")
    if owners:
        print(f"owners: {owners}")
    print()
    if args.mode == "lifespan":
        return await run_lifespan_mode(corpus)
    return await run_readonly_mode(corpus, owners)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

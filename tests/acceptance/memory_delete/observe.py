"""Reading each fact from the layer that owns it."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx

from everos.infra.persistence.markdown import MarkdownReader


def md_entries(root: Path) -> list[dict[str, Any]]:
    """Every daily-log entry under the memory root, with where it lives."""
    out = []
    for md in sorted(root.rglob("*.md")):
        rel = md.relative_to(root)
        if rel.parts[0] in (".index", ".tmp") or md.name in ("user.md", "SKILL.md"):
            continue
        parsed = MarkdownReader.parse(md.read_text(encoding="utf-8"))
        for entry in parsed.entries:
            inline = dict(entry.as_structured().inline)
            out.append(
                {
                    "path": rel.as_posix(),
                    "kind": md.parent.name,
                    "owner": md.parent.parent.name,
                    "entry_id": entry.id,
                    "session_id": inline.get("session_id"),
                    "parent_id": inline.get("parent_id"),
                    "parent_type": inline.get("parent_type"),
                }
            )
    return out


def frontmatter(root: Path, rel: str) -> dict[str, Any]:
    return dict(
        MarkdownReader.parse((root / rel).read_text(encoding="utf-8")).frontmatter
    )


def md_tree_hash(root: Path) -> str:
    """Content hash of the memory tree (md files only; not .index / .tmp)."""
    h = hashlib.sha256()
    for f in sorted(root.rglob("*.md")):
        rel = f.relative_to(root)
        if rel.parts[0] in (".index", ".tmp"):
            continue
        h.update(rel.as_posix().encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def system_db(root: Path) -> Path:
    return root / ".index" / "sqlite" / "system.db"


def rows(db: Path, sql: str, *args: Any) -> list[tuple[Any, ...]]:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return list(con.execute(sql, args))
    finally:
        con.close()


_TABLES = (
    "memcell",
    "unprocessed_buffer",
    "conversation_status",
    "memorize_inbox",
    "cluster",
    "cluster_member",
    "reflection_report",
)


def logical_dump(root: Path) -> dict[str, list[tuple[Any, ...]]]:
    """The deletion-relevant tables, row by row, in a stable order."""
    db = system_db(root)
    dump: dict[str, list[tuple[Any, ...]]] = {}
    for table in _TABLES:
        try:
            dump[table] = sorted(
                map(tuple, rows(db, f"SELECT * FROM {table}")), key=repr
            )
        except sqlite3.OperationalError:
            dump[table] = []
    return dump


def nothing_deleted_snapshot(root: Path) -> tuple[str, str]:
    """md tree hash + logical table dump (spec: the "nothing deleted" check)."""
    return md_tree_hash(root), json.dumps(logical_dump(root), default=str)


def lance_ids(root: Path, table: str) -> set[str]:
    import lancedb

    db = lancedb.connect(str(root / ".index" / "lancedb"))
    try:
        opened = db.open_table(table)
    except (ValueError, FileNotFoundError):
        return set()
    return set(opened.to_arrow().column("id").to_pylist())


def health(base: str) -> dict[str, Any]:
    return httpx.get(f"{base}/health", timeout=10).json()


def wait_converged(base: str, *, timeout: float = 240) -> None:
    """``cascade.pending`` reads 0 twice in a row."""
    deadline = time.monotonic() + timeout
    zeros = 0
    while time.monotonic() < deadline:
        pending = health(base).get("cascade", {}).get("pending")
        zeros = zeros + 1 if pending == 0 else 0
        if zeros >= 2:
            return
        time.sleep(1)
    raise TimeoutError("cascade did not converge")


def metrics(base: str) -> str:
    return httpx.get(f"{base}/metrics", timeout=10).text


def metric_value(text: str, name: str, **labels: str) -> float | None:
    want = ",".join(f'{k}="{v}"' for k, v in labels.items())
    for line in text.splitlines():
        if line.startswith(f"{name}{{{want}}}") or (
            not labels and line.startswith(f"{name} ")
        ):
            return float(line.rsplit(" ", 1)[1])
    return None


def wait_idle(base: str, root: Path, *, timeout: float = 240) -> None:
    """Index converged and no OME run in flight, twice in a row."""
    deadline = time.monotonic() + timeout
    calm = 0
    ome = root / ".index" / "sqlite" / "ome.db"
    while time.monotonic() < deadline:
        cascade = health(base).get("cascade")
        pending = cascade.get("pending") if cascade else 0  # null: cascade off
        running = (
            rows(ome, "SELECT COUNT(*) FROM run_record WHERE status='running'")[0][0]
            if ome.exists()
            else 0
        )
        calm = calm + 1 if pending == 0 and running == 0 else 0
        if calm >= 2:
            return
        time.sleep(1)
    raise TimeoutError("server never went idle")


_SAMPLE = re.compile(r"^(?P<name>\w+)\{(?P<labels>[^}]*)\}\s+(?P<value>\S+)$")


def _samples(
    text: str, name: str, want: dict[str, str]
) -> Iterator[tuple[dict[str, str], float]]:
    for line in text.splitlines():
        m = _SAMPLE.match(line)
        if m and m["name"] == name:
            labels = dict(re.findall(r'(\w+)="([^"]*)"', m["labels"]))
            if all(labels.get(k) == v for k, v in want.items()):
                yield labels, float(m["value"])


def histogram(text: str, name: str, **want: str) -> dict[float, float]:
    """``le`` -> cumulative count of histogram ``name`` with labels ``want``."""
    return {float(lb["le"]): v for lb, v in _samples(text, f"{name}_bucket", want)}


def sample(text: str, name: str, **want: str) -> float:
    """The first sample of ``name`` with labels ``want`` (0 if none)."""
    return next((v for _, v in _samples(text, name, want)), 0.0)


def bucket_edge(cum: dict[float, float], q: float) -> float:
    """The smallest bucket edge holding a ``q`` share of the samples."""
    total = cum.get(math.inf, 0.0)
    return min((le for le, n in cum.items() if n >= q * total), default=math.inf)

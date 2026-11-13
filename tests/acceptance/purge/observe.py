"""Reading one scope's facts from each layer that owns them.

``scope_counts`` / ``scope_dump`` cover the Markdown entries, the system.db
tables a purge clears, the index tables (filtered by ``app_id`` /
``project_id``), ``md_change_state`` and ``memorize_inbox``. ``ome.db`` is
not in them: ``run_record`` has no scope column, so OME rows are recorded by
``event_id`` before a purge and compared by ``event_id`` after it.
"""

from __future__ import annotations

import hashlib
import sqlite3
from collections import Counter
from collections.abc import Collection
from pathlib import Path
from typing import Any

from everos.infra.persistence.markdown import MarkdownReader
from everos.infra.persistence.memory_root import app_dir_name, project_dir_name
from tests.acceptance.memory_delete.observe import (  # noqa: F401 - re-exported
    health,
    nothing_deleted_snapshot,
    rows,
    system_db,
    wait_converged,
    wait_idle,
)

SCOPED_TABLES = (
    "memcell",
    "unprocessed_buffer",
    "conversation_status",
    "cluster",
    "reflection_report",
    "knowledge_documents",
    "knowledge_topics",
)
"""system.db tables with ``app_id`` / ``project_id`` columns a purge clears."""
SQLITE_TABLES = (*SCOPED_TABLES, "cluster_member")
INDEX_TABLES = (
    "episode",
    "atomic_fact",
    "foresight",
    "user_profile",
    "agent_case",
    "agent_skill",
    "knowledge_topic",
)
MD_KINDS = (
    "episode",
    "atomic_fact",
    "foresight",
    "agent_case",
    "user_profile",
    "agent_skill",
    "knowledge_document",
)


def scope_dir(root: Path, app: str, project: str) -> Path:
    return root / app_dir_name(app) / project_dir_name(project)


def _md_kind(rel: Path) -> str | None:
    """Which kind a file under a scope dir holds (``None``: not counted)."""
    if rel.name == ".taxonomy.md" or rel.suffix != ".md":
        return None
    if rel.parts[0] == "knowledge":  # a document is its index.md
        return "knowledge_document" if rel.name == "index.md" else None
    if rel.parts[0] not in ("users", "agents"):
        return None  # a file of the user's own in the project directory
    if rel.name == "user.md":
        return "user_profile"
    if rel.name == "SKILL.md":
        return "agent_skill"
    # A daily log: users|agents/<owner>/<dir>/<kind>-<YYYY-MM-DD>.md
    kind = rel.stem.rsplit("-", 3)[0]
    return kind if kind in MD_KINDS else None


def md_counts(root: Path, app: str, project: str) -> Counter[str]:
    """Entries per kind (daily logs count entries, so a today-only counter
    shell counts 0); a profile, skill or knowledge file counts 1."""
    base = scope_dir(root, app, project)
    counts: Counter[str] = Counter({f"md.{k}": 0 for k in MD_KINDS})
    if not base.is_dir():
        return counts
    for f in sorted(base.rglob("*.md")):
        kind = _md_kind(f.relative_to(base))
        if kind is None:
            continue
        if kind in ("knowledge_document", "user_profile", "agent_skill"):
            counts[f"md.{kind}"] += 1
            continue
        parsed = MarkdownReader.parse(f.read_text(encoding="utf-8"))
        counts[f"md.{kind}"] += len(parsed.entries)
    return counts


def md_files(root: Path, app: str, project: str) -> list[tuple[str, str]]:
    """``(path relative to root, sha256)`` of every file under the scope dir."""
    base = scope_dir(root, app, project)
    if not base.is_dir():
        return []
    return [
        (f.relative_to(root).as_posix(), hashlib.sha256(f.read_bytes()).hexdigest())
        for f in sorted(base.rglob("*"))
        if f.is_file()
    ]


def _sqlite_where(table: str) -> str:
    if table == "cluster_member":
        return (
            "cluster_id IN (SELECT cluster_id FROM cluster"
            " WHERE app_id = ? AND project_id = ?)"
        )
    return "app_id = ? AND project_id = ?"


def _safe_rows(db: Path, sql: str, *args: Any) -> list[tuple[Any, ...]]:
    if not db.exists():
        return []
    try:
        return rows(db, sql, *args)
    except sqlite3.OperationalError as exc:  # table not created yet
        if "no such table" in str(exc):
            return []
        raise


def _prefix(app: str, project: str) -> str:
    return f"{app_dir_name(app)}/{project_dir_name(project)}/"


def index_ids(root: Path, table: str, app: str, project: str) -> set[str]:
    import lancedb

    db = lancedb.connect(str(root / ".index" / "lancedb"))
    try:
        opened = db.open_table(table)
    except (ValueError, FileNotFoundError):
        return set()
    data = opened.to_arrow().select(["id", "app_id", "project_id"]).to_pylist()
    return {r["id"] for r in data if r["app_id"] == app and r["project_id"] == project}


def scope_dump(root: Path, app: str, project: str) -> dict[str, list[Any]]:
    """Every fact of one scope, row by row, in a stable order."""
    db = system_db(root)
    dump: dict[str, list[Any]] = {}
    for table in SQLITE_TABLES:
        dump[f"sqlite.{table}"] = sorted(
            map(
                tuple,
                _safe_rows(
                    db,
                    f"SELECT * FROM {table} WHERE {_sqlite_where(table)}",
                    app,
                    project,
                ),
            ),
            key=repr,
        )
    dump["sqlite.memorize_inbox"] = sorted(
        map(
            tuple,
            _safe_rows(
                db,
                "SELECT * FROM memorize_inbox WHERE app_id = ? AND project_id = ?",
                app,
                project,
            ),
        ),
        key=repr,
    )
    dump["sqlite.md_change_state"] = sorted(
        map(
            tuple,
            _safe_rows(
                db,
                "SELECT * FROM md_change_state WHERE substr(md_path, 1, length(?)) = ?",
                _prefix(app, project),
                _prefix(app, project),
            ),
        ),
        key=repr,
    )
    for table in INDEX_TABLES:
        dump[f"index.{table}"] = sorted(index_ids(root, table, app, project))
    dump["md.files"] = md_files(root, app, project)
    return dump


def scope_counts(root: Path, app: str, project: str) -> dict[str, int]:
    """Row / entry counts of one scope in every layer (H4)."""
    counts: dict[str, int] = dict(md_counts(root, app, project))
    dump = scope_dump(root, app, project)
    for key, value in dump.items():
        if key != "md.files":
            counts[key] = len(value)
    return counts


def run_rows_by_event(root: Path, event_ids: Collection[str]) -> dict[str, list[tuple]]:
    """``ome.db`` ``run_record`` rows per ``event_id``, oldest first."""
    db = root / ".index" / "sqlite" / "ome.db"
    out: dict[str, list[tuple]] = {}
    for event_id in event_ids:
        out[event_id] = _safe_rows(
            db,
            "SELECT run_id, strategy_name, status, attempt, started_at,"
            " finished_at, error, event_topic, event_payload, event_id"
            " FROM run_record WHERE event_id = ? ORDER BY started_at, run_id",
            event_id,
        )
    return out


def release(fp_dir: Path, name: str) -> None:
    """Let a ``name=pause`` failpoint go on."""
    fp_dir.mkdir(parents=True, exist_ok=True)
    (fp_dir / f"{name}.release").write_text("go", encoding="utf-8")


def _counter_shell(path: Path) -> bool:
    """A daily-log file a purge put back with only its id counter."""
    if not path.is_file():
        return False
    try:
        return MarkdownReader.parse(path.read_text(encoding="utf-8")).entries == []
    except (OSError, ValueError):
        return False


def ome_s_payloads(
    root: Path, app: str, project: str, before: Any | None = None
) -> list[tuple[str, str]]:
    """``(strategy, status)`` of ``run_record`` rows still carrying the
    content of an event of ``(app, project)`` (older than ``before``)."""
    import json

    from everos.core.utils.datetime import from_iso_format

    db = root / ".index" / "sqlite" / "ome.db"
    out = []
    for strategy, status, payload in _safe_rows(
        db, "SELECT strategy_name, status, event_payload FROM run_record"
    ):
        try:
            data = json.loads(payload)
        except (TypeError, ValueError):
            continue
        if not isinstance(data, dict) or (
            data.get("app_id"),
            data.get("project_id"),
        ) != (
            app,
            project,
        ):
            continue
        if (
            before is not None
            and "ts" in data
            and from_iso_format(data["ts"]) >= before
        ):
            continue
        out.append((strategy, status))
    return out


def s_leftovers(
    root: Path, app: str, project: str, before: Any | None = None
) -> dict[str, Any]:
    """What of ``(app, project)`` survived a purge, by the A37 yardstick.

    Markdown entries, the system.db tables a purge clears, the index tables,
    ``md_change_state`` (rows of today's counter-only files excepted) and
    ``ome.db`` rows still carrying an event's content (older than
    ``before``, when given). Empty means the scope is cleared.
    """
    counts = scope_counts(root, app, project)
    left: dict[str, Any] = {
        k: n
        for k, n in counts.items()
        if n
        and (
            k.startswith(("md.", "index."))
            or k.removeprefix("sqlite.") in SQLITE_TABLES
        )
    }
    ledger = [
        row[2]
        for row in scope_dump(root, app, project)["sqlite.md_change_state"]
        if not _counter_shell(root / row[2])
    ]
    if ledger:
        left["sqlite.md_change_state"] = ledger
    payloads = ome_s_payloads(root, app, project, before)
    if payloads:
        left["ome.run_record_payload"] = payloads
    return left

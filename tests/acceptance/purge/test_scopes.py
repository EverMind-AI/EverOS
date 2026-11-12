"""T1.1-T4.1: project, app and space purges clear exactly their range (A1-A4)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from everos.core.utils.datetime import from_iso_format
from everos.infra.persistence.markdown import MarkdownReader

from .journey import finish, inbox_rows, purge
from .observe import (
    INDEX_TABLES,
    SQLITE_TABLES,
    md_counts,
    rows,
    s_leftovers,
    scope_counts,
    scope_dir,
    scope_dump,
    system_db,
    wait_idle,
)
from .seed import WORLD

pytestmark = pytest.mark.acceptance


def _zero_except_ledger(counts: dict[str, int]) -> dict[str, int]:
    """Everything a purge clears, ``md_change_state`` aside (it re-registers
    today's counter files); must be all zero."""
    return {
        k: n for k, n in counts.items() if n and k not in ("sqlite.md_change_state",)
    }


async def _entries(path: Path) -> int:
    return len((await MarkdownReader.read(path)).entries)


def _ledger_is_counter_files_only(root: Path, app: str, project: str) -> None:
    """Every ledger row left for the scope is today's empty counter file."""
    prefix = f"{scope_dir(root, app, project).relative_to(root).as_posix()}/"
    paths = [
        r[0]
        for r in rows(
            system_db(root),
            "SELECT md_path FROM md_change_state WHERE substr(md_path,1,length(?))=?",
            prefix,
            prefix,
        )
    ]
    for rel in paths:
        file = root / rel
        assert file.is_file(), f"ledger row for a missing file: {rel}"
        assert MarkdownReader.parse(file.read_text("utf-8")).entries == [], rel


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scope_of(body: dict[str, Any]) -> list[tuple[str, str]]:
    if body["scope"] == "space":
        return list(WORLD)
    return [
        (a, p)
        for a, p in WORLD
        if a == body["app_id"] and body.get("project_id") in (None, p)
    ]


def _purge_and_check(world: Any, body: dict[str, Any]) -> dict[str, Any]:
    srv, ev = world.srv, world.ev
    root = srv.root
    purged = _scope_of(body)
    controls = [s for s in WORLD if s not in purged]
    ev.json("before_counts", {f"{a}/{p}": scope_counts(root, a, p) for a, p in WORLD})
    before_dumps = {s: scope_dump(root, *s) for s in controls}
    history = {s: inbox_rows(root, *s) for s in purged}
    assert all(history.values()), "P0: the purged range has inbox history"
    job_id, _ = purge(srv, **body)
    job = finish(srv, job_id)
    wait_idle(srv.base, root, timeout=300)
    after = {f"{a}/{p}": scope_counts(root, a, p) for a, p in WORLD}
    ev.json("after_counts", after)
    ev.json("job", job)
    assert job["status"] == "succeeded", job
    at = from_iso_format(job["created_at"])
    for app, project in purged:
        assert _zero_except_ledger(after[f"{app}/{project}"]) == {}
        left = s_leftovers(root, app, project, before=at)  # A37, ome.db included
        assert left == {}, (app, project, left)
        assert inbox_rows(root, app, project) == []  # D17: history gone too
        _ledger_is_counter_files_only(root, app, project)
    for scope in controls:
        assert scope_dump(root, *scope) == before_dumps[scope], scope
    return job


def test_t1_1_project_purge_clears_only_the_project(make_world: Any) -> None:
    world = make_world("T1.1")
    _purge_and_check(world, {"scope": "project", "app_id": "a1", "project_id": "p1"})


def test_t2_1_app_purge_clears_every_project_of_the_app(make_world: Any) -> None:
    world = make_world("T2.1")
    _purge_and_check(world, {"scope": "app", "app_id": "a1"})
    app_dir = world.srv.root / "a1"
    for f in sorted(p for p in app_dir.rglob("*") if p.is_file()):
        if f.name == ".taxonomy.md":
            continue
        assert f.suffix == ".md", f
        assert MarkdownReader.parse(f.read_text("utf-8")).entries == [], f


def test_t3_1_space_purge_keeps_config_and_user_files(make_world: Any) -> None:
    world = make_world("T3.1")
    root = world.srv.root
    (root / "notes.txt").write_text("mine", encoding="utf-8")
    (root / "a1" / "p1" / "README.txt").write_text("project notes", encoding="utf-8")
    kept = [
        root / "everos.toml",
        root / "ome.toml",
        root / "notes.txt",
        root / "a1" / "p1" / "README.txt",
        *sorted(root.glob("*/*/knowledge/.taxonomy.md")),
    ]
    assert len(kept) == 4 + len(WORLD)
    hashes = {str(p.relative_to(root)): _sha(p) for p in kept}
    job = _purge_and_check(world, {"scope": "space"})
    after = {str(p.relative_to(root)): _sha(p) for p in kept}
    world.ev.json("kept_sha256", {"before": hashes, "after": after})
    assert after == hashes
    jobs = rows(system_db(root), "SELECT job_id, status FROM purge_job")
    assert jobs == [(job["job_id"], "succeeded")]


def test_t4_1_summary_matches_what_was_deleted(make_world: Any) -> None:
    world = make_world("T4.1")
    srv, root = world.srv, world.srv.root
    assert srv.call("GET", "/health", version="").json()["cascade"]["pending"] == 0
    md = md_counts(root, "a1", "p1")
    own = scope_counts(root, "a1", "p1")
    expected = {
        "markdown_total": sum(md.values()),
        "sqlite_total": sum(own[f"sqlite.{t}"] for t in SQLITE_TABLES),
        "lancedb_total": sum(own[f"index.{t}"] for t in INDEX_TABLES),
    }
    job_id, request = purge(srv, scope="project", app_id="a1", project_id="p1")
    summary = request["summary"]
    job = finish(srv, job_id)
    deleted = {s["store"]: s["deleted"] for s in job["steps"]}
    world.ev.json("three_way", {"own": expected, "summary": summary, "steps": deleted})
    assert {k: summary[k] for k in expected} == expected
    assert summary["complete"] is True
    assert {
        "markdown_total": deleted["markdown"],
        "sqlite_total": deleted["sqlite"],
        "lancedb_total": deleted["lancedb"],
    } == expected
    again = srv.purge(scope="project", app_id="a1", project_id="p1").json()["data"]
    world.ev.json("summary_after", again["summary"])
    assert again["summary"]["markdown_total"] == 0
    assert again["summary"]["sqlite_total"] == 0
    assert again["summary"]["lancedb_total"] == 0
    assert again["summary"]["complete"] is True

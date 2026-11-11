"""Markdown side of a purge: move out, keep today's counters, remove."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from everos.core.purge_marks import MARKS, PURGING, PurgeScope
from everos.core.utils.datetime import today_with_timezone
from everos.infra.cascade.registry import KIND_REGISTRY
from everos.infra.persistence import MemoryRoot
from everos.infra.persistence.markdown import (
    MarkdownOwnerDirStore,
    MarkdownReader,
    MarkdownScopePurger,
)
from everos.infra.persistence.markdown._writers import EpisodeWriter

S = PurgeScope("project", "a1", "p1", "a1", "p1")


@pytest.fixture(autouse=True)
def _purging() -> Iterator[None]:
    MARKS.reset()
    token = PURGING.set(True)
    yield
    PURGING.reset(token)
    MARKS.reset()


@pytest.fixture
def root(tmp_path: Path) -> MemoryRoot:
    root = MemoryRoot(tmp_path)
    root.ensure()
    return root


def _purger(root: MemoryRoot) -> MarkdownScopePurger:
    return MarkdownScopePurger(root, MarkdownOwnerDirStore(root))


async def _episodes(root: MemoryRoot, app: str, project: str, n: int) -> list[str]:
    writer = EpisodeWriter(root)
    ids = []
    for i in range(n):
        eid = await writer.append_entry(
            "u1",
            inline={"owner_id": "u1", "parent_type": "memcell", "parent_id": "mc"},
            sections={"Subject": "s", "Summary": "s", "Content": f"episode {i}"},
            date=today_with_timezone(),
            app_id=app,
            project_id=project,
        )
        ids.append(eid.format())
    return ids


def _today_file(root: MemoryRoot, app: str = "a1", project: str = "p1") -> Path:
    return EpisodeWriter(root).path_for(
        "u1", today_with_timezone(), app_id=app, project_id=project
    )


def _knowledge(root: MemoryRoot, app: str = "a1", project: str = "p1") -> Path:
    base = root.knowledge_dir(app, project)
    (base / "Tech" / "Guide_d_abc").mkdir(parents=True)
    (base / "Tech" / "Guide_d_abc" / "index.md").write_text("---\n---\n")
    (base / ".taxonomy.md").write_text("---\nkind: knowledge_taxonomy\n---\n")
    return base


async def _purge(root: MemoryRoot, scope: PurgeScope, job: str = "j1") -> int:
    purger = _purger(root)
    await purger.restore_counters(scope, job)
    moved = await purger.move_out(scope, job)
    await purger.restore_counters(scope, job)
    await purger.empty(scope, job)
    return moved


async def test_move_out_layout(root: MemoryRoot) -> None:
    await _episodes(root, "a1", "p1", 2)
    _knowledge(root)
    moved = await _purger(root).move_out(S, "j1")
    assert moved == 3  # two episodes + one document
    trash = root.tmp_dir / "purge-j1" / "a1" / "p1"
    rel = _today_file(root).relative_to(root.root / "a1" / "p1")
    assert (trash / rel).is_file()
    assert (trash / "knowledge" / "Tech" / "Guide_d_abc" / "index.md").is_file()
    assert not (root.root / "a1" / "p1" / "users").exists()
    assert (root.root / "a1" / "p1" / "knowledge" / ".taxonomy.md").is_file()


async def test_trash_not_matched_by_scanner_glob(root: MemoryRoot) -> None:
    await _episodes(root, "a1", "p1", 1)
    _knowledge(root)
    await _purger(root).move_out(S, "j1")
    trash = root.tmp_dir / "purge-j1"
    seen = {p for spec in KIND_REGISTRY for p in root.root.glob(spec.path_glob())}
    assert not {p for p in seen if trash in p.parents}


async def test_counters_survive_purge(root: MemoryRoot) -> None:
    before = await _episodes(root, "a1", "p1", 5)
    assert await _purge(root, S) == 5
    live = _today_file(root)
    parsed = await MarkdownReader.read(live)
    assert parsed.entries == [] and parsed.frontmatter["entry_count"] == 5
    after = await _episodes(root, "a1", "p1", 1)
    assert not set(after) & set(before)
    assert not (root.tmp_dir / "purge-j1").exists()


async def test_counters_restored_from_delete_trash(root: MemoryRoot) -> None:
    before = await _episodes(root, "a1", "p1", 4)
    owner_dir = root.users_dir("a1", "p1") / "u1"
    trash = root.tmp_dir / "delete-x"  # a /delete stopped after its rename
    trash.mkdir()
    (trash / "scope.json").write_text(
        json.dumps(
            {"app_id": "a1", "project_id": "p1", "owner_type": "user", "owner_id": "u1"}
        )
    )
    owner_dir.rename(trash / "owner")
    await _purge(root, S)
    assert not trash.exists()
    after = await _episodes(root, "a1", "p1", 1)
    assert not set(after) & set(before)


async def test_taxonomy_and_user_files_kept(root: MemoryRoot) -> None:
    await _episodes(root, "a1", "p1", 1)
    knowledge = _knowledge(root)
    notes = root.root / "a1" / "p1" / "notes.txt"
    notes.write_text("mine")
    await _purge(root, S)
    assert notes.read_text() == "mine"
    assert (knowledge / ".taxonomy.md").is_file()
    assert sorted(p.name for p in knowledge.iterdir()) == [".taxonomy.md"]


async def test_other_scopes_untouched(root: MemoryRoot) -> None:
    for app, project in (("a1", "p1"), ("a1", "p2"), ("a2", "p1")):
        await _episodes(root, app, project, 2)
    await _purge(root, S)
    for app, project in (("a1", "p2"), ("a2", "p1")):
        entries = (await MarkdownReader.read(_today_file(root, app, project))).entries
        assert len(entries) == 2


@pytest.mark.parametrize("app", [".tmp", ".index"])
async def test_system_named_scope_touches_no_system_dir(
    root: MemoryRoot, app: str
) -> None:
    await _episodes(root, "a1", "p1", 1)
    delete_trash = root.tmp_dir / "delete-x" / "owner"
    delete_trash.mkdir(parents=True)
    (root.index_dir / "keep").write_text("index")
    (root.tmp_dir / "scratch").mkdir()  # an empty dir a purge must not prune
    before = sorted(p.relative_to(root.root) for p in root.root.rglob("*"))
    await _purge(root, PurgeScope("app", app, None, app))
    assert sorted(p.relative_to(root.root) for p in root.root.rglob("*")) == before


async def test_space_skips_system_dirs_and_files(root: MemoryRoot) -> None:
    await _episodes(root, "a1", "p1", 1)
    await _episodes(root, ".foo", "p1", 1)  # a dot-named app is still an app
    (root.root / "ome.toml").write_text("x")
    (root.index_dir / "keep").write_text("index")
    assert await _purge(root, PurgeScope("space")) == 2
    assert (root.root / "ome.toml").read_text() == "x"
    assert (root.index_dir / "keep").read_text() == "index"


async def test_sweep_trash_skip(root: MemoryRoot) -> None:
    for owner, project in (("u1", "p1"), ("u2", "p2")):
        trash = root.tmp_dir / f"delete-{owner}"
        (trash / "owner").mkdir(parents=True)
        (trash / "scope.json").write_text(
            json.dumps(
                {
                    "app_id": "a1",
                    "project_id": project,
                    "owner_type": "user",
                    "owner_id": owner,
                }
            )
        )
    swept = await MarkdownOwnerDirStore(root).sweep_trash(
        skip=lambda scope: S.covers(scope.app_id, scope.project_id)
    )
    assert swept == 1
    assert (root.tmp_dir / "delete-u1").is_dir()
    assert not (root.tmp_dir / "delete-u2").exists()


async def test_sweep_inactive_keeps_the_active_job(root: MemoryRoot) -> None:
    for job in ("old", "live"):
        (root.tmp_dir / f"purge-{job}" / "a1").mkdir(parents=True)
    assert await _purger(root).sweep_inactive("live") == 1
    assert sorted(p.name for p in root.tmp_dir.iterdir()) == ["purge-live"]

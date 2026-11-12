"""The scanner walks the registered kinds only, and leaves retired kinds' rows.

EverOS 2.0 has no agent or knowledge memory, but a store an earlier build
wrote still holds their markdown, and its ``md_change_state`` ledger still
holds their rows. What these pin:

- :func:`_collect_scan_inputs` walks every kind of :data:`KIND_REGISTRY` --
  an episode and a profile both surface -- and nothing else: an older store's
  agent case, agent skill and knowledge files are never stat'd or emitted.
- :meth:`CascadeScanner.scan_once` reconciles only the ledger rows of
  registered kinds. A retired kind's row names a file the walk never visits,
  so reconciling it would emit ``deleted``, the re-stat would find the file
  and queue it again, and the worker would fail it on every pass; the row is
  left as it is. A registered kind's vanished file is still a deletion.
"""

from __future__ import annotations

from pathlib import Path

from everos.infra.cascade.scanner import CascadeScanner, _collect_scan_inputs
from tests.fakes.storage import Storage

_SCOPE = ("default_app", "default_project")


def _write(root: Path, *parts: str) -> str:
    """Create a plausible md file under the memory root; its relative path."""
    path = root.joinpath(*_SCOPE, *parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("ok")
    return path.relative_to(root).as_posix()


def _retired_files(root: Path) -> list[str]:
    """What an older store holds of the kinds EverOS 2.0 removed."""
    return [
        _write(root, "agents", "a1", ".cases", "agent_case-2026-01-01.md"),
        _write(root, "agents", "a1", "skills", "skill_greet", "SKILL.md"),
        _write(root, "knowledge", "Sports", "doc_1", "index.md"),
    ]


def test_collect_scan_inputs_walks_every_registered_kind_only(
    tmp_path: Path,
) -> None:
    episode = _write(tmp_path, "users", "u1", "episodes", "episode-2026-01-01.md")
    profile = _write(tmp_path, "users", "u1", "user.md")
    retired = _retired_files(tmp_path)

    inputs = {i.md_path: i.kind for i in _collect_scan_inputs(tmp_path)}

    assert inputs == {episode: "episode", profile: "user_profile"}
    assert not set(retired) & set(inputs)


async def test_scan_once_leaves_a_retired_kinds_row_as_it_is(
    storage: Storage,
) -> None:
    await storage.create_schema()
    root = storage.root.root
    ledger = storage.sqlite.md_change_state
    episode = _write(root, "users", "u1", "episodes", "episode-2026-01-01.md")
    gone = "default_app/default_project/users/u1/episodes/episode-2025-12-31.md"
    skill = _retired_files(root)[1]
    for path, kind in ((skill, "agent_skill"), (gone, "episode")):
        await ledger.upsert(path, kind=kind, change_type="added", mtime=1.0)
        await ledger.mark_done(path)
    before = await ledger.read_one(skill)

    decisions = await CascadeScanner(storage.root, ledger).scan_once()

    assert {d.md_path: d.change_type for d in decisions} == {
        episode: "added",
        gone: "deleted",
    }
    after = await ledger.read_one(skill)
    assert before is not None and after is not None
    assert (after.status, after.change_type, after.lsn) == (
        before.status,
        before.change_type,
        before.lsn,
    )

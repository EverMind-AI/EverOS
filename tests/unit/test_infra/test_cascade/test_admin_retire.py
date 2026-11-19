"""Startup removal of what a retired kind (foresight) left in a store."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from everos.infra.cascade.admin import CascadeAdmin


class _Index:
    def __init__(self, tables: set[str]) -> None:
        self.tables = tables
        self.asked: list[list[str]] = []

    async def drop_retired_tables(self, names: Sequence[str]) -> list[str]:
        self.asked.append(list(names))
        dropped = [n for n in names if n in self.tables]
        self.tables -= set(dropped)
        return dropped


def _admin(index: _Index) -> CascadeAdmin:
    none: Any = None
    return CascadeAdmin(database=none, ledger=none, index=index, provenance=none)  # type: ignore[arg-type]


def _leftover(root: Path) -> list[Path]:
    return list(root.glob("*/*/users/*/.foresights"))


def _all_exist(paths: list[Path]) -> bool:
    return all(p.exists() for p in paths)


def _write(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


async def test_removes_retired_logs_and_table_and_keeps_the_rest(
    tmp_path: Path,
) -> None:
    users = tmp_path / "default_app" / "default_project" / "users"
    _write(users / "u1" / ".foresights" / "foresight-2026-11-19.md")
    _write(users / "u2" / ".foresights" / "foresight-2026-11-18.md")
    keep = [
        _write(users / "u1" / "episodes" / "episode-2026-11-19.md"),
        _write(users / "u1" / ".atomic_facts" / "atomic_fact-2026-11-19.md"),
        _write(users / "u1" / "user.md"),
    ]
    index = _Index({"episode", "foresight"})

    result = await _admin(index).retire_removed_kinds(tmp_path)

    assert result == {"tables": ["foresight"], "folders": {"foresight": 2}}
    assert _leftover(tmp_path) == []
    assert _all_exist(keep)
    assert index.tables == {"episode"}


async def test_a_second_start_removes_nothing(tmp_path: Path) -> None:
    _write(tmp_path / "a" / "p" / "users" / "u1" / ".foresights" / "f.md")
    index = _Index({"foresight"})
    admin = _admin(index)
    await admin.retire_removed_kinds(tmp_path)

    assert await admin.retire_removed_kinds(tmp_path) == {"tables": [], "folders": {}}
    assert index.asked == [["foresight"], ["foresight"]]

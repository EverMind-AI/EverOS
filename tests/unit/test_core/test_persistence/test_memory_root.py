"""Unit tests for MemoryRoot path manager."""

from __future__ import annotations

from pathlib import Path

import pytest

from everos.core.persistence import MemoryRoot


def test_default_returns_home_everos(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("EVEROS_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    from everos.config import load_settings

    load_settings.cache_clear()
    mr = MemoryRoot.default()
    assert mr.root == (Path.home() / ".everos").resolve()


def test_default_from_everos_root_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("EVEROS_ROOT", str(tmp_path / "custom"))
    mr = MemoryRoot.default()
    assert mr.root == (tmp_path / "custom").resolve()


def test_default_explicit_root(tmp_path: Path) -> None:
    mr = MemoryRoot.default(explicit_root=str(tmp_path / "explicit"))
    assert mr.root == (tmp_path / "explicit").resolve()


def test_accepts_str_path(tmp_path: Path) -> None:
    mr = MemoryRoot(str(tmp_path))
    assert mr.root == tmp_path.resolve()


def test_accepts_pathlib_path(tmp_path: Path) -> None:
    mr = MemoryRoot(tmp_path)
    assert mr.root == tmp_path.resolve()


def test_user_visible_dirs_default_scope(tmp_path: Path) -> None:
    mr = MemoryRoot(tmp_path)
    # Omitting app/project resolves to the default space; "default" lands as
    # the reserved ``default_app`` / ``default_project`` directory names.
    base = mr.root / "default_app" / "default_project"
    assert mr.agents_dir() == base / "agents"
    assert mr.users_dir() == base / "users"
    assert mr.knowledge_dir() == base / "knowledge"


def test_user_visible_dirs_named_scope(tmp_path: Path) -> None:
    mr = MemoryRoot(tmp_path)
    # A non-default app/project maps to itself (no ``default_*`` rewrite).
    base = mr.root / "claude_code" / "oss"
    assert mr.agents_dir("claude_code", "oss") == base / "agents"
    assert mr.users_dir("claude_code", "oss") == base / "users"
    assert mr.knowledge_dir("claude_code", "oss") == base / "knowledge"


def test_dotfile_paths(tmp_path: Path) -> None:
    mr = MemoryRoot(tmp_path)
    assert mr.index_dir == tmp_path / ".index"
    assert mr.lancedb_dir == tmp_path / ".index" / "lancedb"
    assert mr.sqlite_dir == tmp_path / ".index" / "sqlite"
    assert mr.system_db == tmp_path / ".index" / "sqlite" / "system.db"
    assert mr.lock_file == tmp_path / ".lock"
    assert mr.tmp_dir == tmp_path / ".tmp"


def test_ensure_creates_required_dirs(tmp_path: Path) -> None:
    mr = MemoryRoot(tmp_path / "fresh")
    mr.ensure()
    assert mr.root.is_dir()
    assert mr.index_dir.is_dir()
    assert mr.sqlite_dir.is_dir()
    assert mr.lancedb_dir.is_dir()
    assert mr.tmp_dir.is_dir()
    # User-visible dirs are NOT pre-created.
    assert not mr.agents_dir().exists()
    assert not mr.users_dir().exists()
    assert not mr.knowledge_dir().exists()


def test_ensure_is_idempotent(tmp_path: Path) -> None:
    mr = MemoryRoot(tmp_path)
    mr.ensure()
    mr.ensure()  # second call must not fail
    assert mr.tmp_dir.is_dir()


def test_ensure_does_not_create_ome_toml(tmp_path: Path) -> None:
    """ome.toml creation moved to ``everos init``; ensure() only makes dirs."""
    mr = MemoryRoot(tmp_path / "fresh")
    mr.ensure()
    assert not mr.ome_config.exists()


def test_frozen_dataclass_hashable(tmp_path: Path) -> None:
    a = MemoryRoot(tmp_path)
    b = MemoryRoot(tmp_path)
    assert a == b
    assert hash(a) == hash(b)
    assert {a, b} == {a}  # set deduplication works


def test_user_expansion(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    mr = MemoryRoot("~/custom")
    assert mr.root == (tmp_path / "custom").resolve()

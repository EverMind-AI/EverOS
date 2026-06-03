"""Unit tests for MemoryRoot path manager."""

from __future__ import annotations

from pathlib import Path

import pytest

from everos.core.persistence import MemoryRoot


def test_default_returns_home_everos(monkeypatch: pytest.MonkeyPatch) -> None:
    # Isolate from any ambient EVEROS_MEMORY__ROOT (e.g. the session-scoped
    # search-corpus fixture sets it for the whole run); the autouse
    # _reset_settings_cache fixture clears the load_settings cache, so the
    # delenv takes effect for this assertion of the hard-coded default.
    monkeypatch.delenv("EVEROS_MEMORY__ROOT", raising=False)
    mr = MemoryRoot.default()
    assert mr.root == (Path.home() / ".everos").resolve()


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


def test_ensure_materializes_ome_config_template(tmp_path: Path) -> None:
    """First ensure() drops a real ``ome.toml`` users can edit.

    Without this, ``pip install everos && everos server start`` produced
    a warning (``config_reload_failed: No such file``) because the OME
    config reloader had no file to point at. The template ships under
    ``src/everos/config/default_ome.toml`` and is byte-copied on first run.
    """
    mr = MemoryRoot(tmp_path)
    mr.ensure()
    assert mr.ome_config.is_file()
    # Content is the shipped template verbatim — protects against a future
    # diff that silently changes what users see on first run.
    template = Path(__file__).resolve().parents[4] / (
        "src/everos/config/default_ome.toml"
    )
    assert mr.ome_config.read_bytes() == template.read_bytes()


def test_ensure_preserves_user_edited_ome_config(tmp_path: Path) -> None:
    """Second ensure() must not overwrite user edits.

    The template materialisation is an existence check, not a content
    sync — once the user has tweaked their overrides the file is theirs.
    """
    mr = MemoryRoot(tmp_path)
    mr.ensure()
    custom = b"# user-edited\n[strategies.extract_foresight]\nenabled = false\n"
    mr.ome_config.write_bytes(custom)
    mr.ensure()
    assert mr.ome_config.read_bytes() == custom


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

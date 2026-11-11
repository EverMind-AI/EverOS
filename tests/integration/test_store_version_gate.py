"""The store-version gate, through the real server start and the real CLI.

A root shaped like an EverOS 1.x store (LanceDB table schema v2, an
``atomic_fact`` table, a system DB without any 2.0 table) is refused by both
before anything in it changes; a fresh root is stamped ``2`` and starts.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pytest
from typer.testing import CliRunner

from everos.core.errors import StoreVersionError
from everos.entrypoints.cli.commands import cascade as cascade_mod
from tests.fakes import make_providers


def _one_x_root(root: Path) -> None:
    lance = root / ".index" / "lancedb"
    lance.mkdir(parents=True)
    (lance / ".table_schema_version").write_text("2")
    for table in ("episode", "atomic_fact", "user_profile"):
        (lance / f"{table}.lance").mkdir()
        (lance / f"{table}.lance" / "data.lance").write_bytes(b"1.x rows")
    sqlite_dir = root / ".index" / "sqlite"
    sqlite_dir.mkdir(parents=True)
    connection = sqlite3.connect(sqlite_dir / "system.db")
    connection.execute("CREATE TABLE md_change_state (id INTEGER PRIMARY KEY)")
    connection.execute("CREATE TABLE memcell (id INTEGER PRIMARY KEY)")
    connection.commit()
    connection.close()
    users = root / "default_app" / "default_project" / "users" / "u1" / "episodes"
    users.mkdir(parents=True)
    (users / "episode-2026-11-08.md").write_text("---\nschema_version: 1\n---\n")
    (root / "everos.toml").write_text("[llm]\n")


def _snapshot(root: Path) -> dict[str, str]:
    """Every file with its hash; lock files (taken before any check) aside."""
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file() and not p.name.endswith(".lock") and p.name != ".lock"
    }


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("EVEROS_ROOT", str(tmp_path))
    monkeypatch.setenv("EVEROS_LLM__API_KEY", "unused")
    monkeypatch.setenv("EVEROS_LLM__BASE_URL", "http://127.0.0.1:9/v1")
    from everos.config import load_settings

    load_settings.cache_clear()
    return tmp_path


async def test_the_server_refuses_a_1x_store_and_changes_nothing(root: Path) -> None:
    from everos.config import load_settings
    from everos.service import build_application

    _one_x_root(root)
    before = _snapshot(root)
    application = build_application(load_settings(), providers=make_providers())

    with pytest.raises(StoreVersionError, match=r"EverOS 1\.x store"):
        await application.start()

    assert _snapshot(root) == before
    assert not (root / ".index" / "store_version").exists()


def test_the_cli_refuses_a_1x_store_with_exit_code_3(root: Path) -> None:
    _one_x_root(root)
    before = _snapshot(root)

    result = CliRunner().invoke(cascade_mod.app, ["status"])

    assert result.exit_code == 3, result.output
    assert "EverOS 1.x store" in result.output
    assert _snapshot(root) == before


async def test_a_fresh_root_is_stamped_and_starts(root: Path) -> None:
    from everos.config import load_settings
    from everos.service import build_application

    (root / "ome.toml").write_text("# test\n")
    application = build_application(load_settings(), providers=make_providers())
    try:
        await application.start()
    finally:
        await application.aclose()

    assert (root / ".index" / "store_version").read_text() == "2\n"

"""The store-version gate: EverOS 2 opens only the stores EverOS 2 built.

The roots here are shaped after real ones (read from the 11-08 and 11-11
evaluation stores without opening them): an EverOS 1.x store has LanceDB
table schema v2, ``atomic_fact`` / ``foresight`` tables and no picture
tables; a 2.0 store built on 11-08 is still at v2 but has ``image_memory``;
one built on 11-10 is at v3, one on 11-11 at v5.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pytest

from everos.core.errors import StoreVersionError
from everos.infra.persistence import STORE_VERSION, MemoryRoot, StoreVersion


def _snapshot(root: Path) -> dict[str, str]:
    """Every file under ``root`` with its content hash."""
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def _store(
    root: Path,
    *,
    schema: int | None = 2,
    tables: tuple[str, ...] = ("episode", "user_profile", "atomic_fact"),
    system_tables: tuple[str, ...] = ("memcell", "md_change_state"),
    memory: bool = True,
) -> MemoryRoot:
    """A root with memory, an index and a system DB, as a version left it."""
    lance = root / ".index" / "lancedb"
    lance.mkdir(parents=True)
    if schema is not None:
        (lance / ".table_schema_version").write_text(str(schema))
    for table in tables:
        (lance / f"{table}.lance").mkdir()
        (lance / f"{table}.lance" / "data.lance").write_bytes(b"rows")
    sqlite_dir = root / ".index" / "sqlite"
    sqlite_dir.mkdir(parents=True)
    connection = sqlite3.connect(sqlite_dir / "system.db")
    for table in system_tables:
        connection.execute(f"CREATE TABLE {table} (id INTEGER PRIMARY KEY)")
    connection.commit()
    connection.close()
    if memory:
        users = root / "default_app" / "default_project" / "users" / "u1"
        users.mkdir(parents=True)
        (users / "user.md").write_text("---\nschema_version: 1\n---\n")
    return MemoryRoot(root)


def test_a_new_root_is_stamped(tmp_path: Path) -> None:
    (tmp_path / "everos.toml").write_text("[llm]\n")
    (tmp_path / ".index" / "sqlite").mkdir(parents=True)  # what start creates
    gate = StoreVersion(MemoryRoot(tmp_path))

    gate.check()

    assert gate.marker.read_text() == f"{STORE_VERSION}\n"
    assert STORE_VERSION == 2


def test_a_stamped_root_opens_unchanged(tmp_path: Path) -> None:
    root = _store(tmp_path)
    (tmp_path / ".index" / "store_version").write_text("2\n")
    before = _snapshot(tmp_path)

    StoreVersion(root).check()

    assert _snapshot(tmp_path) == before


def test_a_1x_store_is_refused_and_left_as_it_was(tmp_path: Path) -> None:
    root = _store(tmp_path)  # schema v2, atomic_fact, no picture table
    before = _snapshot(tmp_path)

    with pytest.raises(StoreVersionError) as refused:
        StoreVersion(root).check()

    message = str(refused.value)
    assert "EverOS 1.x store" in message
    assert "index table schema v2" in message
    assert "not upgraded in place" in message
    assert _snapshot(tmp_path) == before
    assert not (tmp_path / ".index" / "store_version").exists()


@pytest.mark.parametrize(
    ("marker", "words"),
    [
        ("1", "holds an EverOS 1 store"),
        ("3", "Run it with the newer EverOS"),
        ("two", "not a store version"),
    ],
)
def test_another_marker_is_refused(tmp_path: Path, marker: str, words: str) -> None:
    root = _store(tmp_path, schema=5, tables=("episode", "image_memory"))
    (tmp_path / ".index" / "store_version").write_text(marker)
    before = _snapshot(tmp_path)

    with pytest.raises(StoreVersionError, match=words):
        StoreVersion(root).check()

    assert _snapshot(tmp_path) == before


@pytest.mark.parametrize(
    ("shape", "evidence"),
    [
        # Built by the current 2.0 code (11-11 and later): table schema v5.
        (
            {"schema": 5, "tables": ("episode", "image_memory")},
            "index table schema v5",
        ),
        # Built on 11-10 (visual vectors): v3.
        ({"schema": 3, "tables": ("episode",)}, "index table schema v3"),
        # Built on 11-08: still v2, but the picture table exists.
        (
            {"schema": 2, "tables": ("episode", "atomic_fact", "image_memory")},
            "index table image_memory.lance",
        ),
        # A Milvus-backed 2.0 store: no LanceDB, but 2.0's system tables.
        (
            {"schema": None, "tables": (), "system_tables": ("memcell", "group_meta")},
            "system table group_meta",
        ),
    ],
)
def test_a_store_from_before_the_gate_is_adopted_when_2_0_wrote_it(
    tmp_path: Path, shape: dict, evidence: str
) -> None:
    root = _store(tmp_path, **shape)
    gate = StoreVersion(root)

    assert evidence in gate._evidence_of_2_0()
    gate.check()

    assert gate.marker.read_text() == "2\n"
    gate.check()  # and now opens by its marker


def test_markdown_without_store_state_is_a_new_store(tmp_path: Path) -> None:
    """Hand-written markdown, or a root whose .index was wiped: markdown is the
    source of truth and carries no store state, so this build indexes it."""
    users = tmp_path / "default_app" / "default_project" / "users" / "u1"
    users.mkdir(parents=True)
    (users / "user.md").write_text("profile")

    StoreVersion(MemoryRoot(tmp_path)).check()

    assert (tmp_path / ".index" / "store_version").read_text() == "2\n"


def test_a_1x_system_db_alone_is_still_refused(tmp_path: Path) -> None:
    """Wiping only the LanceDB index leaves 1.x state in the system DB."""
    root = _store(tmp_path, schema=None, tables=())

    with pytest.raises(StoreVersionError, match=r"EverOS 1\.x store"):
        StoreVersion(root).check()

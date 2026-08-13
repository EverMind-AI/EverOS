"""Fail-closed storage-identity generation marker contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from everos.core.persistence import MemoryRoot
from everos.core.persistence.lancedb.row_id import STORAGE_ID_GENERATION
from everos.infra.persistence.lancedb.storage_identity import (
    StorageIdentityMigrationRequiredError,
    ensure_storage_identity_ready,
    mark_storage_identity_ready,
    mark_storage_identity_rebuilding,
    marker_path,
    read_storage_identity_state,
)


def test_fresh_empty_root_initializes_current_ready_marker(tmp_path: Path) -> None:
    root = MemoryRoot(tmp_path)

    ensure_storage_identity_ready(root)

    state = read_storage_identity_state(root)
    assert state is not None
    assert state.state == "READY"
    assert state.generation == STORAGE_ID_GENERATION == 2


def test_existing_source_without_marker_fails_closed(tmp_path: Path) -> None:
    root = MemoryRoot(tmp_path)
    source = tmp_path / "default_app/default_project/users/u1/episodes/day.md"
    source.parent.mkdir(parents=True)
    source.write_text("# existing memory\n", encoding="utf-8")

    with pytest.raises(StorageIdentityMigrationRequiredError, match="missing"):
        ensure_storage_identity_ready(root)

    assert not marker_path(root).exists()


def test_existing_lancedb_artifact_without_marker_fails_closed(tmp_path: Path) -> None:
    root = MemoryRoot(tmp_path)
    artifact = root.lancedb_dir / "episode.lance/data/legacy.lance"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"legacy")

    with pytest.raises(StorageIdentityMigrationRequiredError, match="missing"):
        ensure_storage_identity_ready(root)


@pytest.mark.parametrize(
    "payload",
    [
        {"generation": True, "state": "READY"},
        {"generation": 2.0, "state": "READY"},
        {"generation": 2, "state": "ready"},
        {"generation": 2, "state": "READY", "extra": "ignored?"},
        ["READY", 2],
    ],
)
def test_marker_json_types_and_shape_are_strict(
    tmp_path: Path, payload: object
) -> None:
    root = MemoryRoot(tmp_path)
    path = marker_path(root)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StorageIdentityMigrationRequiredError):
        read_storage_identity_state(root)


def test_rebuilding_and_old_generation_both_block_startup(tmp_path: Path) -> None:
    root = MemoryRoot(tmp_path)
    mark_storage_identity_rebuilding(root)

    with pytest.raises(StorageIdentityMigrationRequiredError, match="REBUILDING"):
        ensure_storage_identity_ready(root)

    marker_path(root).write_text(
        json.dumps({"generation": 1, "state": "READY"}), encoding="utf-8"
    )
    with pytest.raises(StorageIdentityMigrationRequiredError, match="required=2"):
        ensure_storage_identity_ready(root)


def test_ready_replaces_rebuilding_atomically_at_same_path(tmp_path: Path) -> None:
    root = MemoryRoot(tmp_path)
    mark_storage_identity_rebuilding(root)
    assert read_storage_identity_state(root).state == "REBUILDING"  # type: ignore[union-attr]

    mark_storage_identity_ready(root)

    state = read_storage_identity_state(root)
    assert state is not None
    assert state.state == "READY"
    assert list(root.lancedb_dir.glob(f".{marker_path(root).name}.*")) == []

"""Fail-closed lifecycle marker for LanceDB storage-key generations.

The shape of ``row.id`` is value-level state, so LanceDB's schema checker
cannot distinguish a legacy index from one rebuilt with the current storage
identity.  This marker binds the on-disk projection to the implementation that
created it and prevents a partially rebuilt index from being served.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from everos.core.persistence import MemoryRoot
from everos.core.persistence.lancedb.row_id import STORAGE_ID_GENERATION

_MARKER_NAME = ".storage_identity.json"


class StorageIdentityMigrationRequiredError(RuntimeError):
    """Raised when an index is not proven ready for this key generation."""


@dataclass(frozen=True)
class StorageIdentityState:
    generation: int
    state: Literal["READY", "REBUILDING"]


def marker_path(memory_root: MemoryRoot) -> Path:
    return memory_root.lancedb_dir / _MARKER_NAME


def read_storage_identity_state(
    memory_root: MemoryRoot,
) -> StorageIdentityState | None:
    """Read and strictly validate the marker, returning ``None`` if absent."""
    path = marker_path(memory_root)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StorageIdentityMigrationRequiredError(
            f"Storage identity marker {path} is unreadable or malformed. "
            "Run `everos cascade rebuild` with the server stopped."
        ) from exc
    if type(payload) is not dict or set(payload) != {"generation", "state"}:
        raise StorageIdentityMigrationRequiredError(
            f"Storage identity marker {path} has an unknown JSON shape. "
            "Run `everos cascade rebuild` with the server stopped."
        )
    generation = payload["generation"]
    state = payload["state"]
    # ``bool`` is an ``int`` subclass in Python. Accepting ``true`` as
    # generation 1/2 would turn malformed JSON into a valid migration gate.
    if (
        type(generation) is not int
        or type(state) is not str
        or state not in {"READY", "REBUILDING"}
    ):
        raise StorageIdentityMigrationRequiredError(
            f"Storage identity marker {path} has invalid fields. "
            "Run `everos cascade rebuild` with the server stopped."
        )
    return StorageIdentityState(generation=generation, state=state)


def mark_storage_identity_rebuilding(memory_root: MemoryRoot) -> None:
    """Invalidate any prior ready marker before destructive rebuild work."""
    _write_state(
        memory_root,
        StorageIdentityState(
            generation=STORAGE_ID_GENERATION,
            state="REBUILDING",
        ),
    )


def mark_storage_identity_ready(memory_root: MemoryRoot) -> None:
    """Atomically publish readiness after every rebuild gate has passed."""
    _write_state(
        memory_root,
        StorageIdentityState(
            generation=STORAGE_ID_GENERATION,
            state="READY",
        ),
    )


def ensure_storage_identity_ready(memory_root: MemoryRoot) -> None:
    """Require READY(current generation), initializing only an empty projection.

    A missing marker is safe only when there is no source markdown and no
    existing LanceDB artifact. This prevents a legacy projection, or a source
    tree awaiting its first generation-2 rebuild, from being mistaken for an
    empty projection. Retained SQLite state is outside this marker's scope.
    """
    state = read_storage_identity_state(memory_root)
    if state is None:
        if _projection_is_provably_empty(memory_root):
            mark_storage_identity_ready(memory_root)
            return
        raise StorageIdentityMigrationRequiredError(
            "Storage identity marker is missing for an existing memory root. "
            "Run `everos cascade rebuild` with the server stopped."
        )
    if state.state != "READY" or state.generation != STORAGE_ID_GENERATION:
        raise StorageIdentityMigrationRequiredError(
            "LanceDB storage identity is not ready for this EverOS build "
            f"(state={state.state!r}, generation={state.generation}, "
            f"required={STORAGE_ID_GENERATION}). Run `everos cascade rebuild` "
            "with the server stopped."
        )


def _projection_is_provably_empty(memory_root: MemoryRoot) -> bool:
    root = memory_root.root
    # Any markdown makes a marker-less root non-empty. App identifiers now
    # reject the system-managed ``.index`` / ``.tmp`` namespaces, but a
    # legacy or manually created source file there must still fail closed
    # rather than be mistaken for a fresh installation.
    if root.exists() and any(root.rglob("*.md")):
        return False

    lancedb_dir = memory_root.lancedb_dir
    if not lancedb_dir.exists():
        return True
    return not any(path.is_file() for path in lancedb_dir.rglob("*"))


def _write_state(memory_root: MemoryRoot, state: StorageIdentityState) -> None:
    path = marker_path(memory_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        {"generation": state.generation, "state": state.state},
        sort_keys=True,
        separators=(",", ":"),
    )
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            fd = -1
            stream.write(payload)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        if fd >= 0:
            os.close(fd)
        temp_path.unlink(missing_ok=True)


__all__ = [
    "StorageIdentityMigrationRequiredError",
    "StorageIdentityState",
    "ensure_storage_identity_ready",
    "mark_storage_identity_ready",
    "mark_storage_identity_rebuilding",
    "marker_path",
    "read_storage_identity_state",
]

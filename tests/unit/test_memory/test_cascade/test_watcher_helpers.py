"""Unit tests for the pure helpers in :mod:`everos.memory.cascade.watcher`.

The :class:`CascadeWatcher` itself needs a running event loop + real
filesystem to test end-to-end (see ``tests/integration/``). The pure
helpers can be exercised in isolation.
"""

from __future__ import annotations

from pathlib import Path

from everos.memory.cascade.watcher import _relative_to_root, _safe_mtime


def test_relative_to_root_within(tmp_path: Path) -> None:
    target = tmp_path / "users" / "u1" / "x.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("x")
    assert _relative_to_root(tmp_path, str(target)) == "users/u1/x.md"


def test_relative_to_root_outside(tmp_path: Path) -> None:
    """A path outside the memory root returns ``None``."""
    outside = tmp_path.parent / "completely-different" / "y.md"
    assert _relative_to_root(tmp_path, str(outside)) is None


def test_safe_mtime_missing_path_returns_zero(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist.md"
    assert _safe_mtime(str(missing)) == 0.0


def test_safe_mtime_existing_path_returns_positive(tmp_path: Path) -> None:
    f = tmp_path / "f.md"
    f.write_text("ok")
    assert _safe_mtime(str(f)) > 0

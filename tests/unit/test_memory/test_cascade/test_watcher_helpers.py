"""Unit tests for the pure helpers in :mod:`everos.memory.cascade.watcher`.

The :class:`CascadeWatcher` itself needs a running event loop + real
filesystem to test end-to-end (see ``tests/integration/``). The pure
helpers can be exercised in isolation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

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


def test_relative_to_root_within_does_not_touch_the_filesystem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An in-root path is relativised textually — no ``resolve()``.

    ``resolve()`` is a filesystem round trip per component (through
    Defender on Windows) and the watcher calls this on every event.
    """

    def _boom(self: Path, strict: bool = False) -> Path:
        raise AssertionError("resolve() must not run for an in-root path")

    monkeypatch.setattr(Path, "resolve", _boom)
    target = tmp_path / "users" / "u1" / "x.md"  # need not exist
    assert _relative_to_root(tmp_path, str(target)) == "users/u1/x.md"


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need a privilege")
def test_relative_to_root_via_symlink_still_resolves(
    tmp_path: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """A path that reaches the root through an external symlink still maps."""
    root = tmp_path.resolve()
    (root / "users" / "u1").mkdir(parents=True)
    (root / "users" / "u1" / "x.md").write_text("x", encoding="utf-8")
    link = tmp_path_factory.mktemp("elsewhere") / "link_to_root"
    link.symlink_to(root, target_is_directory=True)
    assert (
        _relative_to_root(root, str(link / "users" / "u1" / "x.md")) == "users/u1/x.md"
    )


def test_relative_to_root_normalises_dotdot_segments(tmp_path: Path) -> None:
    """A ``..`` segment takes the resolving path so the key stays canonical."""
    root = tmp_path.resolve()
    (root / "users" / "u1").mkdir(parents=True)
    raw = str(root / "users" / "u2" / ".." / "u1" / "x.md")
    assert _relative_to_root(root, raw) == "users/u1/x.md"

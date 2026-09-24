"""The package-init hook that lets compiled extensions find the MSVC runtime.

The hook runs at ``import everos``; these tests call the factored function
with a fake prefix and a recorded ``os.add_dll_directory`` so they hold on
every platform. What only Windows can prove -- that greenlet then actually
loads on a machine without the redistributable -- was proven by hand on a
stock Windows 11 Enterprise box; see the commit that introduced this.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import everos


def _recording_add(seen: list[str]):  # type: ignore[no-untyped-def]
    def add(d: str) -> object:
        seen.append(d)
        return object()

    return add


def test_registers_prefix_and_scripts_when_the_runtime_dll_is_there(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "msvcp140.dll").write_bytes(b"")
    (tmp_path / "Scripts").mkdir()
    (tmp_path / "Scripts" / "msvcp140.dll").write_bytes(b"")
    seen: list[str] = []
    monkeypatch.setattr(os, "add_dll_directory", _recording_add(seen), raising=False)

    got = everos._register_runtime_dll_dirs(
        str(tmp_path), user_base=str(tmp_path / "ub")
    )

    assert got == seen == [str(tmp_path), str(tmp_path / "Scripts")]


def test_registers_the_per_user_install_dir_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``pip install`` without write access to site-packages lands the wheel's
    data files under ``site.getuserbase()``, not ``sys.prefix``."""
    prefix = tmp_path / "prefix"
    prefix.mkdir()
    user_base = tmp_path / "AppData" / "Python"
    (user_base / "Scripts").mkdir(parents=True)
    (user_base / "msvcp140.dll").write_bytes(b"")
    (user_base / "Scripts" / "msvcp140.dll").write_bytes(b"")
    seen: list[str] = []
    monkeypatch.setattr(os, "add_dll_directory", _recording_add(seen), raising=False)

    got = everos._register_runtime_dll_dirs(str(prefix), user_base=str(user_base))

    assert got == seen == [str(user_base), str(user_base / "Scripts")]


def test_skips_directories_without_the_dll(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Registering a dir with no runtime in it would only widen the search
    path for nothing; the dll is the signal that msvc-runtime is present."""
    (tmp_path / "Scripts").mkdir()
    seen: list[str] = []
    monkeypatch.setattr(os, "add_dll_directory", _recording_add(seen), raising=False)

    assert (
        everos._register_runtime_dll_dirs(str(tmp_path), user_base=str(tmp_path)) == []
    )
    assert seen == []


def test_is_a_no_op_where_the_os_has_no_add_dll_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """POSIX: the dll may even be there (a shared checkout), still nothing."""
    (tmp_path / "msvcp140.dll").write_bytes(b"")
    monkeypatch.delattr(os, "add_dll_directory", raising=False)

    assert (
        everos._register_runtime_dll_dirs(str(tmp_path), user_base=str(tmp_path)) == []
    )

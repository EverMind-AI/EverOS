"""Unit tests for :class:`CascadeScanner` lifecycle + ``_collect_scan_inputs``.

The reconcile-against-state flow is integration territory; this file
covers the no-real-DB-needed pieces: idempotent start/stop and the
sync-thread walker's resilience to broken files.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from everos.core.persistence import MemoryRoot
from everos.memory.cascade import scanner as scanner_module
from everos.memory.cascade.scanner import CascadeScanner, _collect_scan_inputs


async def test_double_start_is_idempotent(tmp_path: Path) -> None:
    mr = MemoryRoot(tmp_path)
    mr.ensure()
    scanner = CascadeScanner(mr, scan_interval_seconds=60.0)
    await scanner.start()
    first_task = scanner._task
    await scanner.start()  # second start: no-op
    assert scanner._task is first_task
    await scanner.stop()


async def test_stop_before_start_is_noop(tmp_path: Path) -> None:
    mr = MemoryRoot(tmp_path)
    mr.ensure()
    scanner = CascadeScanner(mr, scan_interval_seconds=60.0)
    await scanner.stop()  # must not raise


async def test_double_stop_is_idempotent(tmp_path: Path) -> None:
    mr = MemoryRoot(tmp_path)
    mr.ensure()
    scanner = CascadeScanner(mr, scan_interval_seconds=60.0)
    await scanner.start()
    await scanner.stop()
    await scanner.stop()  # second stop: no-op


def test_collect_scan_inputs_skips_dangling_symlinks(tmp_path: Path) -> None:
    """A symlink whose target was deleted yields ``stat`` OSError → skipped."""
    mr = MemoryRoot(tmp_path)
    mr.ensure()
    # Build a real .md under a registered kind path (with the <app>/<project>
    # scope prefix the glob requires), then add a broken symlink next to it to
    # exercise the OSError branch.
    user_dir = (
        tmp_path / "default_app" / "default_project" / "users" / "u1" / "episodes"
    )
    user_dir.mkdir(parents=True, exist_ok=True)
    real = user_dir / "episode-2026-01-01.md"
    real.write_text("ok")
    broken = user_dir / "episode-2026-01-02.md"
    target = tmp_path / "deleted-target"
    target.write_text("temp")
    broken.symlink_to(target)
    target.unlink()  # Now ``broken`` is a dangling symlink.

    inputs = _collect_scan_inputs(tmp_path)
    paths = {i.md_path for i in inputs}
    assert real.relative_to(tmp_path).as_posix() in paths
    # Dangling symlink was silently skipped.
    assert broken.relative_to(tmp_path).as_posix() not in paths


def test_collect_scan_inputs_raises_on_transient_oserror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Non-ENOENT stat errors (EMFILE / EACCES / EIO) must propagate.

    Regression guard for the 2026-05-28 incident where FD exhaustion
    during a scan made every healthy md look "deleted" to reconcile().
    The fix in ``_collect_scan_inputs`` swallows only ``FileNotFoundError``
    and re-raises any other ``OSError`` so the reconciler never sees a
    partial scan.
    """
    user_dir = (
        tmp_path / "default_app" / "default_project" / "users" / "u1" / "episodes"
    )
    user_dir.mkdir(parents=True, exist_ok=True)
    real = user_dir / "episode-2026-01-01.md"
    real.write_text("ok")

    real_stat = Path.stat

    def boom_stat(self: Path, *args, **kwargs):  # type: ignore[no-untyped-def]
        # Only fail on the .md file — let glob / directory walks succeed.
        if self.suffix == ".md":
            raise OSError(24, "Too many open files")
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", boom_stat)

    with pytest.raises(OSError) as exc_info:
        _collect_scan_inputs(tmp_path)
    # errno 24 = EMFILE on every POSIX system we care about.
    assert exc_info.value.errno == 24


async def test_run_loop_swallows_scan_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure in ``scan_once`` is logged but the loop keeps going."""
    mr = MemoryRoot(tmp_path)
    mr.ensure()
    scanner = CascadeScanner(mr, scan_interval_seconds=0.05)

    call_count = {"n": 0}
    second_scan = asyncio.Event()
    logged_errors: list[str] = []

    async def fake_scan() -> list:  # type: ignore[type-arg]
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("simulated scanner failure")
        second_scan.set()
        return []

    def fake_exception(_event: str, *, error: str) -> None:
        logged_errors.append(error)

    monkeypatch.setattr(scanner, "scan_once", fake_scan)
    monkeypatch.setattr(scanner_module.logger, "exception", fake_exception)
    await scanner.start()
    try:
        await asyncio.wait_for(second_scan.wait(), timeout=1.0)
    finally:
        await scanner.stop()

    assert logged_errors == ["simulated scanner failure"]
    assert call_count["n"] >= 2  # second call ran despite first throwing

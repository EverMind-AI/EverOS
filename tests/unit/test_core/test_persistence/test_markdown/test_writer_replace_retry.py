"""The staging-to-target swap must outlast a Windows sharing violation.

On Windows, ``os.replace`` onto a file another process has open raises
``PermissionError``; on POSIX it never does. These tests fake ``os.replace``
so the behaviour is pinned on every platform: a transient refusal is retried
and the write lands, a persistent one propagates after the budget with the
staging file cleaned up, and any other error is not retried at all.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from everos.core.persistence import MemoryRoot
from everos.core.persistence.markdown import writer as writer_mod
from everos.core.persistence.markdown.writer import MarkdownWriter


class _Refusing:
    """Stand-in for a Windows loader that refuses the first ``n`` replaces."""

    def __init__(self, refuse: int, error: type[OSError] = PermissionError) -> None:
        self.refuse = refuse
        self.error = error
        self.calls = 0
        self._real = os.replace

    def __call__(self, src: str | Path, dst: str | Path) -> None:
        self.calls += 1
        if self.calls <= self.refuse:
            raise self.error(5, "The process cannot access the file", str(dst))
        self._real(src, dst)


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    slept: list[float] = []
    monkeypatch.setattr(time, "sleep", slept.append)
    return slept


async def test_transient_refusal_is_retried_and_the_write_lands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_sleep: list[float]
) -> None:
    refusing = _Refusing(refuse=2)
    monkeypatch.setattr(os, "replace", refusing)
    target = tmp_path / "users" / "u1" / "note.md"

    await MarkdownWriter(MemoryRoot(tmp_path)).write(target, "survived")

    assert target.read_text(encoding="utf-8") == "survived"
    assert refusing.calls == 3
    assert no_sleep == [0.02, 0.04], "exponential backoff between the two refusals"
    assert not list(target.parent.glob(".*.tmp.*")), "no staging file left behind"


async def test_persistent_refusal_propagates_after_the_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_sleep: list[float]
) -> None:
    refusing = _Refusing(refuse=10**6)
    monkeypatch.setattr(os, "replace", refusing)
    target = tmp_path / "users" / "u1" / "note.md"

    with pytest.raises(PermissionError):
        await MarkdownWriter(MemoryRoot(tmp_path)).write(target, "never lands")

    assert refusing.calls == writer_mod._REPLACE_ATTEMPTS
    assert len(no_sleep) == writer_mod._REPLACE_ATTEMPTS - 1
    assert not target.exists()
    assert not list(target.parent.glob(".*.tmp.*")), "staging file cleaned on failure"


async def test_other_errors_are_not_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_sleep: list[float]
) -> None:
    """Retrying would only delay a real failure (disk full, bad path)."""
    refusing = _Refusing(refuse=10**6, error=FileNotFoundError)
    monkeypatch.setattr(os, "replace", refusing)
    target = tmp_path / "users" / "u1" / "note.md"

    with pytest.raises(FileNotFoundError):
        await MarkdownWriter(MemoryRoot(tmp_path)).write(target, "x")

    assert refusing.calls == 1
    assert no_sleep == []

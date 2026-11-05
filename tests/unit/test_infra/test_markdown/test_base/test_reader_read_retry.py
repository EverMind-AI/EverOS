"""Reading a markdown file must outlast a Windows sharing violation.

On Windows, ``open`` on a file that another thread or process is replacing
with ``os.replace`` can raise ``PermissionError`` (errno 13) for a moment; on
POSIX it never does. Before this, the writer retried its replace but every
reader failed at once, so a session deletion raised ``DeletionIncompleteError``
and a cascade projection failed permanently. These tests fake the read so the
behaviour is pinned on every platform, and check that no markdown read
bypasses the retrying entry point.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import pytest

from everos.infra.persistence.markdown import MarkdownReader
from everos.infra.persistence.markdown._base import reader as reader_mod

_SRC = Path(reader_mod.__file__).resolve().parents[4]  # .../src/everos


class _Refusing:
    """Stand-in for a Windows open that refuses the first ``n`` reads of one file."""

    def __init__(self, target: Path, refuse: int, error: type[OSError]) -> None:
        self.target = target
        self.refuse = refuse
        self.error = error
        self.calls = 0
        self._real = Path.read_text

    def __call__(self, path: Path, *args: object, **kwargs: object) -> str:
        if Path(path) != self.target:
            return self._real(path, *args, **kwargs)  # type: ignore[arg-type]
        self.calls += 1
        if self.calls <= self.refuse:
            raise self.error(13, "Permission denied", str(path))
        return self._real(path, *args, **kwargs)  # type: ignore[arg-type]

    def as_method(self) -> object:
        """A plain function, so ``Path`` binds it like the real method."""

        def read_text(path: Path, *args: object, **kwargs: object) -> str:
            return self(path, *args, **kwargs)

        return read_text


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    slept: list[float] = []
    monkeypatch.setattr(time, "sleep", slept.append)
    return slept


def _md(tmp_path: Path) -> Path:
    target = tmp_path / "episode-2026-11-05.md"
    target.write_text("---\nentry_count: 1\n---\nbody\n", encoding="utf-8")
    return target


async def test_transient_refusal_is_retried_and_the_read_lands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_sleep: list[float]
) -> None:
    target = _md(tmp_path)
    refusing = _Refusing(target, refuse=2, error=PermissionError)
    monkeypatch.setattr(Path, "read_text", refusing.as_method())

    parsed = await MarkdownReader.read(target)

    assert parsed.frontmatter == {"entry_count": 1}
    assert refusing.calls == 3
    assert no_sleep == [0.02, 0.04], "exponential backoff between the two refusals"


async def test_persistent_refusal_propagates_after_the_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_sleep: list[float]
) -> None:
    target = _md(tmp_path)
    refusing = _Refusing(target, refuse=10**6, error=PermissionError)
    monkeypatch.setattr(Path, "read_text", refusing.as_method())

    with pytest.raises(PermissionError):
        await MarkdownReader.read_text(target)

    assert refusing.calls == reader_mod._READ_ATTEMPTS
    assert len(no_sleep) == reader_mod._READ_ATTEMPTS - 1


async def test_a_missing_file_is_not_retried(
    tmp_path: Path, no_sleep: list[float]
) -> None:
    """Callers branch on ``FileNotFoundError``; waiting would only delay it."""
    with pytest.raises(FileNotFoundError):
        await MarkdownReader.read_text(tmp_path / "absent.md")
    assert no_sleep == []


# Reads of files the markdown writers replace must go through
# ``MarkdownReader.read_text``; a raw ``read_text`` here is the bug this file
# pins. The one exception is a deletion's trash manifest, written and read by
# the same deletion and never replaced underneath it.
_ALLOWED = {
    ("infra/persistence/markdown/_base/reader.py", "path.read_text("),
    ("infra/persistence/markdown/owner_dir.py", "(trash / _MANIFEST).read_text("),
}
_RAW_READ = re.compile(r"[\w)\]]\.read_text\(")


def test_no_markdown_read_bypasses_the_retrying_reader() -> None:
    offenders = []
    for folder in ("infra/persistence/markdown", "infra/cascade/_projections"):
        for path in sorted((_SRC / folder).rglob("*.py")):
            rel = path.relative_to(_SRC).as_posix()
            lines = path.read_text(encoding="utf-8").splitlines()
            for number, line in enumerate(lines, 1):
                if "MarkdownReader.read_text(" in line or not _RAW_READ.search(line):
                    continue
                if any(rel == f and snippet in line for f, snippet in _ALLOWED):
                    continue
                offenders.append(f"{rel}:{number}: {line.strip()}")
    assert offenders == [], "read through MarkdownReader.read_text:\n" + "\n".join(
        offenders
    )

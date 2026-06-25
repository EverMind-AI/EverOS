"""Block deprecated product names in tracked repository text."""

from __future__ import annotations

import re
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

DEPRECATED_NAME_RE = re.compile(r"\bever[\s_-]*core\b", flags=re.IGNORECASE)
SKIP_SUFFIXES = frozenset(
    {
        ".avif",
        ".bmp",
        ".gif",
        ".heic",
        ".heif",
        ".icns",
        ".ico",
        ".jpeg",
        ".jpg",
        ".mov",
        ".mp4",
        ".png",
        ".webp",
    }
)


@dataclass(frozen=True)
class Violation:
    path: str
    line_number: int
    line: str


def find_violations(files: Iterable[tuple[str, str]]) -> list[Violation]:
    violations: list[Violation] = []
    for path, text in files:
        for line_number, line in enumerate(text.splitlines(), start=1):
            if DEPRECATED_NAME_RE.search(line):
                violations.append(
                    Violation(path=path, line_number=line_number, line=line.strip())
                )
    return violations


def _tracked_paths() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        check=True,
        stdout=subprocess.PIPE,
        text=False,
    )
    return [Path(raw.decode("utf-8")) for raw in result.stdout.split(b"\0") if raw]


def _tracked_text_files() -> Iterable[tuple[str, str]]:
    for path in _tracked_paths():
        if not path.exists():
            continue
        if path.suffix.lower() in SKIP_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        yield path.as_posix(), text


def main() -> int:
    violations = find_violations(_tracked_text_files())
    if not violations:
        print("Deprecated-name check passed.")
        return 0

    print(
        "Deprecated-name check failed.\n"
        "Use EverOS or EverMind Cloud. Do not use deprecated product naming.\n"
    )
    for violation in violations:
        print(f"- {violation.path}:{violation.line_number}: {violation.line}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

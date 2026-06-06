"""Block committed image/video files and asset-style directories."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Iterable

BLOCKED_DIR_NAMES = frozenset(
    {
        "asset",
        "assets",
        "image",
        "images",
        "img",
        "media",
        "video",
        "videos",
    }
)
IMAGE_EXTENSIONS = frozenset(
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
        ".png",
        ".svg",
        ".tif",
        ".tiff",
        ".webp",
    }
)
VIDEO_EXTENSIONS = frozenset(
    {
        ".avi",
        ".flv",
        ".m4v",
        ".mkv",
        ".mov",
        ".mp4",
        ".mpeg",
        ".mpg",
        ".webm",
        ".wmv",
    }
)


@dataclass(frozen=True)
class Violation:
    path: str
    reason: str


def _normalise_path(path: str) -> PurePosixPath:
    return PurePosixPath(path.replace("\\", "/"))


def _violation_reason(path: str) -> str | None:
    posix_path = _normalise_path(path)
    lower_parts = tuple(part.lower() for part in posix_path.parts)
    if any(part in BLOCKED_DIR_NAMES for part in lower_parts):
        return "asset/media directory"

    suffix = posix_path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return "image file"
    if suffix in VIDEO_EXTENSIONS:
        return "video file"
    return None


def find_violations(paths: Iterable[str]) -> list[Violation]:
    violations: list[Violation] = []
    for path in paths:
        reason = _violation_reason(path)
        if reason is not None:
            violations.append(Violation(path=path, reason=reason))
    return violations


def _tracked_paths() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        check=True,
        stdout=subprocess.PIPE,
        text=False,
    )
    return [
        raw.decode("utf-8")
        for raw in result.stdout.split(b"\0")
        if raw
    ]


def main() -> int:
    violations = find_violations(_tracked_paths())
    if not violations:
        print("Repository asset/media check passed.")
        return 0

    print(
        "Repository asset/media check failed.\n"
        "Do not commit images, videos, or asset/media directories. "
        "Host visual media externally, in release artifacts, or another "
        "approved storage location, then link to it from docs.\n"
    )
    for violation in violations:
        print(f"- {violation.path}: {violation.reason}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

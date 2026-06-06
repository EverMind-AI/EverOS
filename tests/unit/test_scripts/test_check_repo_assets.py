"""Self-tests for ``scripts/check_repo_assets.py``."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CHECKER_PATH = _REPO_ROOT / "scripts" / "check_repo_assets.py"


def _load_checker():
    assert _CHECKER_PATH.exists(), "repo asset checker should exist"
    spec = importlib.util.spec_from_file_location("_repo_asset_checker", _CHECKER_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_clean_source_and_docs_paths_are_allowed() -> None:
    checker = _load_checker()

    violations = checker.find_violations(
        [
            "README.md",
            "docs/engineering.md",
            "src/everos/__init__.py",
            "use-cases/claude-code-plugin/dashboard/dashboard.html",
        ]
    )

    assert violations == []


def test_image_extensions_are_blocked() -> None:
    checker = _load_checker()

    violations = checker.find_violations(["docs/banner.png", "icons/logo.svg"])

    assert [violation.path for violation in violations] == [
        "docs/banner.png",
        "icons/logo.svg",
    ]
    assert {violation.reason for violation in violations} == {"image file"}


def test_video_extensions_are_blocked() -> None:
    checker = _load_checker()

    violations = checker.find_violations(["demo/launch.mp4", "docs/clip.webm"])

    assert [violation.path for violation in violations] == [
        "demo/launch.mp4",
        "docs/clip.webm",
    ]
    assert {violation.reason for violation in violations} == {"video file"}


def test_asset_and_media_directories_are_blocked() -> None:
    checker = _load_checker()

    violations = checker.find_violations(
        [
            "assets/banner.txt",
            "docs/images/diagram.txt",
            "use-cases/example/media/story.md",
            "use-cases/example/videos/walkthrough.md",
        ]
    )

    assert [violation.path for violation in violations] == [
        "assets/banner.txt",
        "docs/images/diagram.txt",
        "use-cases/example/media/story.md",
        "use-cases/example/videos/walkthrough.md",
    ]
    assert {violation.reason for violation in violations} == {"asset/media directory"}

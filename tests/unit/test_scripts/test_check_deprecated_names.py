"""Self-tests for ``scripts/check_deprecated_names.py``."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CHECKER_PATH = _REPO_ROOT / "scripts" / "check_deprecated_names.py"


def _load_checker():
    assert _CHECKER_PATH.exists(), "deprecated-name checker should exist"
    spec = importlib.util.spec_from_file_location(
        "_deprecated_name_checker", _CHECKER_PATH
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_clean_text_is_allowed() -> None:
    checker = _load_checker()

    violations = checker.find_violations(
        [("README.md", "EverOS is the public project name.\n")]
    )

    assert violations == []


def test_deprecated_name_variants_are_blocked() -> None:
    checker = _load_checker()
    compact_name = "Ever" + "Core"
    spaced_name = "ever" + " core"
    hyphenated_name = "ever" + "-core"

    violations = checker.find_violations(
        [
            ("README.md", f"{compact_name} should not appear.\n"),
            ("docs/example.md", f"{spaced_name} should not appear.\n"),
            ("src/example.py", f"{hyphenated_name} should not appear.\n"),
        ]
    )

    assert [(violation.path, violation.line_number) for violation in violations] == [
        ("README.md", 1),
        ("docs/example.md", 1),
        ("src/example.py", 1),
    ]


def test_real_repo_has_no_deprecated_names() -> None:
    checker = _load_checker()

    assert checker.main() == 0

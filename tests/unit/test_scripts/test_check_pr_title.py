"""Self-tests for ``scripts/check_pr_title.py``."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CHECKER_PATH = _REPO_ROOT / "scripts" / "check_pr_title.py"


def _load_checker():
    assert _CHECKER_PATH.exists(), "PR title checker should exist"
    spec = importlib.util.spec_from_file_location("_pr_title_checker", _CHECKER_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_conventional_pr_title_is_allowed() -> None:
    checker = _load_checker()

    assert checker.validate_title("docs(readme): polish launch highlights") == []


def test_bracketed_codex_pr_title_is_blocked() -> None:
    checker = _load_checker()

    failures = checker.validate_title("[codex] simplify README launch highlights")

    assert len(failures) == 1
    assert "invalid PR title" in failures[0]
    assert "expected: <type>[(scope)][!]: <description>" in failures[0]


def test_long_pr_title_is_blocked() -> None:
    checker = _load_checker()
    title = (
        "docs(readme): polish launch highlights and banner with a title that is "
        "too long"
    )

    failures = checker.validate_title(title)

    assert len(failures) == 1
    assert "max is 72" in failures[0]


def test_main_reads_pr_title_environment(monkeypatch) -> None:
    checker = _load_checker()
    monkeypatch.setenv("PR_TITLE", "docs(readme): polish launch highlights")

    assert checker.main([]) == 0

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
    assert "max is 64" in failures[0]


def test_title_that_only_fits_before_the_squash_suffix_is_blocked() -> None:
    """The v1.4.2 regression: a legal title that squashes into an illegal one.

    ``fix(lancedb): cap both caches, give queries their own embed budget`` is
    66 characters and passed this check. GitHub appended `` (#496)`` on squash
    and put a 73-character subject on ``main``, which then failed
    ``commit messages`` on every open PR until each one rebased past it.
    """
    checker = _load_checker()
    title = "fix(lancedb): cap both caches, give queries their own embed budget"
    assert len(title) == 66

    assert checker.validate_title(title, pr_number=None) != []

    failures = checker.validate_title(title, pr_number=496)

    assert len(failures) == 1
    assert "max is 65" in failures[0]
    assert '" (#496)"' in failures[0]


def test_title_that_still_fits_with_the_suffix_is_allowed() -> None:
    checker = _load_checker()
    title = "fix(lancedb): cap both caches, give queries their own budget"
    assert len(title) + len(" (#496)") <= checker.MAX_TITLE_LENGTH

    assert checker.validate_title(title, pr_number=496) == []


def test_longer_pr_numbers_cost_more_budget() -> None:
    """The reserve is the real suffix, not a guess."""
    checker = _load_checker()
    title = "fix(lancedb): cap both caches, give queries their own embe budget"
    assert len(title) == 65

    assert checker.validate_title(title, pr_number=496) == []
    assert checker.validate_title(title, pr_number=4096) != []


def test_main_reads_pr_title_environment(monkeypatch) -> None:
    checker = _load_checker()
    monkeypatch.setenv("PR_TITLE", "docs(readme): polish launch highlights")

    assert checker.main([]) == 0


def test_main_reads_pr_number_environment(monkeypatch) -> None:
    """CI passes the number; without it the check falls back to a reserve."""
    checker = _load_checker()
    monkeypatch.setenv(
        "PR_TITLE", "fix(lancedb): cap both caches, give queries their own budget"
    )
    monkeypatch.setenv("PR_NUMBER", "496")
    assert checker.main([]) == 0

    monkeypatch.setenv("PR_NUMBER", "496000000")
    assert checker.main([]) == 1

"""Self-tests for ``scripts/check_github_contributor_docs.py``."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CHECKER_PATH = _REPO_ROOT / "scripts" / "check_github_contributor_docs.py"


def _load_checker():
    assert _CHECKER_PATH.exists(), "GitHub contributor-doc checker should exist"
    spec = importlib.util.spec_from_file_location(
        "_github_contributor_docs_checker", _CHECKER_PATH
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_clean_github_workflow_is_allowed() -> None:
    checker = _load_checker()

    violations = checker.find_violations(
        [
            (
                Path("CLAUDE.md"),
                "Create scoped branches from main and open pull requests.\n",
            )
        ]
    )

    assert violations == []


def test_legacy_branch_model_terms_are_blocked() -> None:
    checker = _load_checker()
    stable_branch = "mas" + "ter"
    integration_branch = "de" + "v"
    legacy_review = "merge " + "request"
    branch_model = "Git" + "Flow"
    emergency_branch = "hot" + "fix"

    violations = checker.find_violations(
        [
            (
                Path("CLAUDE.md"),
                f"{stable_branch} is stable and {integration_branch} is integration.\n",
            ),
            (Path("CONTRIBUTING.md"), f"Open an internal {legacy_review}.\n"),
            (
                Path(".claude/skills/pr/SKILL.md"),
                f"{branch_model} {emergency_branch} path.\n",
            ),
        ]
    )

    assert [(v.path.as_posix(), v.label) for v in violations] == [
        ("CLAUDE.md", "legacy stable branch reference"),
        ("CLAUDE.md", "legacy integration branch reference"),
        ("CONTRIBUTING.md", "legacy review wording"),
        (".claude/skills/pr/SKILL.md", "legacy branch model"),
        (".claude/skills/pr/SKILL.md", "legacy emergency branch model"),
    ]


def test_real_repo_has_no_legacy_contributor_doc_residue() -> None:
    checker = _load_checker()

    assert checker.main() == 0

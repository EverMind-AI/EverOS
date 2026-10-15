"""Self-tests for ``scripts/check_commit_messages.py``.

The range the checker picks is the part worth pinning. On ``pull_request``
GitHub checks out ``refs/pull/N/merge`` -- the PR head merged into the
*current* base tip -- while ``pull_request.base.sha`` is the base tip from
when the PR was last synced. Anything the base branch gained in between sits
inside ``base.sha..HEAD``, so a bad subject merged to ``main`` fails every
open PR until each one rebases.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CHECKER_PATH = _REPO_ROOT / "scripts" / "check_commit_messages.py"

_BAD_SUBJECT = (
    "fix(lancedb): cap both caches, give queries their own embed budget (#496)"
)


def _load_checker():
    assert _CHECKER_PATH.exists(), "commit message checker should exist"
    spec = importlib.util.spec_from_file_location("_commit_msg_checker", _CHECKER_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def _commit(repo: Path, subject: str) -> str:
    """Commit on a file of its own, so branches merge without conflicts."""
    name = f"{abs(hash(subject)):x}.txt"
    (repo / name).write_text(subject, encoding="utf-8")
    _git(repo, "add", name)
    _git(repo, "commit", "-q", "-m", subject)
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def merge_ref_repo(tmp_path: Path) -> tuple[Path, str]:
    """A repo shaped like the ``refs/pull/N/merge`` checkout.

    Returns the repo and the base SHA GitHub would report, i.e. the base tip
    as it was when the PR was opened -- *before* ``main`` moved on.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")

    pr_base_sha = _commit(repo, "chore: base")

    _git(repo, "checkout", "-q", "-b", "pr")
    _commit(repo, "fix(cascade): a well-formed subject")

    # main moves on, carrying a subject that violates the policy.
    _git(repo, "checkout", "-q", "main")
    _commit(repo, _BAD_SUBJECT)

    # What CI actually checks out: PR head merged into the current base tip.
    _git(repo, "merge", "-q", "--no-ff", "-m", "Merge pull request #1", "pr")
    return repo, pr_base_sha


def test_base_branch_commits_do_not_fail_the_pr(
    merge_ref_repo: tuple[Path, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bad subject already on the base branch is not this PR's problem."""
    repo, pr_base_sha = merge_ref_repo
    checker = _load_checker()
    monkeypatch.chdir(repo)
    monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
    monkeypatch.setenv("GITHUB_PR_BASE_SHA", pr_base_sha)

    assert checker._validate(checker._default_range()) == []


def test_bad_subject_on_the_pr_still_fails(
    merge_ref_repo: tuple[Path, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Narrowing the range must not stop the check from catching anything."""
    repo, pr_base_sha = merge_ref_repo
    _git(repo, "checkout", "-q", "pr")
    _commit(repo, _BAD_SUBJECT)
    _git(repo, "checkout", "-q", "main")
    _git(repo, "merge", "-q", "--no-ff", "-m", "Merge pull request #1", "pr")

    checker = _load_checker()
    monkeypatch.chdir(repo)
    monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
    monkeypatch.setenv("GITHUB_PR_BASE_SHA", pr_base_sha)

    failures = checker._validate(checker._default_range())

    assert len(failures) == 1
    assert "subject is 73 chars" in failures[0]


def test_non_merge_checkout_falls_back_to_the_reported_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A workflow checking out the PR head directly still gets a range."""
    repo = tmp_path / "flat"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    pr_base_sha = _commit(repo, "chore: base")
    _commit(repo, _BAD_SUBJECT)

    checker = _load_checker()
    monkeypatch.chdir(repo)
    monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
    monkeypatch.setenv("GITHUB_PR_BASE_SHA", pr_base_sha)

    failures = checker._validate(checker._default_range())

    assert len(failures) == 1
    assert "subject is 73 chars" in failures[0]

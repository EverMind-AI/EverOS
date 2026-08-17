"""Self-tests for ``scripts/check_file_sizes.py``.

Pins three things: the ceiling's boundary semantics, that the gate's scope is
the change under review rather than the whole tree, and that the script and
the ``check-added-large-files`` pre-commit hook share one number. Drift
between the last two would recreate the failure this gate exists to prevent —
a ceiling that only ever runs on a developer's machine.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CHECKER_PATH = _REPO_ROOT / "scripts" / "check_file_sizes.py"
_PRE_COMMIT_CONFIG = _REPO_ROOT / ".pre-commit-config.yaml"


def _load_checker():
    assert _CHECKER_PATH.exists(), "file size checker should exist"
    spec = importlib.util.spec_from_file_location("_file_size_checker", _CHECKER_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _write(root: Path, name: str, size_bytes: int) -> str:
    target = root / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"x" * size_bytes)
    return name


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A real git repo with one commit on ``main`` as the diff base."""
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "test")
    _write(tmp_path, "existing-huge.bin", 9 * 1024)
    _write(tmp_path, "README.md", 16)
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "base")
    return tmp_path


def test_files_within_the_ceiling_are_allowed(tmp_path: Path) -> None:
    checker = _load_checker()
    paths = [
        _write(tmp_path, "README.md", 1024),
        _write(tmp_path, "tests/fixtures/seed.json", 4 * 1024),
    ]

    violations = checker.find_violations(paths, root=tmp_path, max_kb=8)

    assert violations == []


def test_oversized_file_is_flagged_with_its_size(tmp_path: Path) -> None:
    checker = _load_checker()
    paths = [
        _write(tmp_path, "small.json", 1024),
        _write(tmp_path, "examples/recorded_trace.json", 9 * 1024),
    ]

    violations = checker.find_violations(paths, root=tmp_path, max_kb=8)

    assert [violation.path for violation in violations] == [
        "examples/recorded_trace.json"
    ]
    assert violations[0].size_bytes == 9 * 1024


def test_ceiling_is_inclusive_at_the_exact_limit(tmp_path: Path) -> None:
    checker = _load_checker()
    exact = _write(tmp_path, "exact.bin", 8 * 1024)
    over = _write(tmp_path, "over.bin", 8 * 1024 + 1)

    violations = checker.find_violations([exact, over], root=tmp_path, max_kb=8)

    assert [violation.path for violation in violations] == ["over.bin"]


def test_missing_and_symlink_paths_are_skipped(tmp_path: Path) -> None:
    checker = _load_checker()
    real = _write(tmp_path, "real.bin", 9 * 1024)
    link = tmp_path / "link.bin"
    link.symlink_to(tmp_path / "real.bin")

    violations = checker.find_violations(
        ["deleted-from-worktree.bin", "link.bin", real],
        root=tmp_path,
        max_kb=8,
    )

    assert [violation.path for violation in violations] == ["real.bin"]


def test_scope_excludes_files_the_change_did_not_touch(repo: Path) -> None:
    checker = _load_checker()
    _git(repo, "checkout", "-q", "-b", "feature")
    _write(repo, "added-small.json", 32)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "small addition")

    paths = checker.changed_paths(repo, "main")

    assert paths == ["added-small.json"]
    assert "existing-huge.bin" not in paths


def test_scope_includes_a_grown_existing_file(repo: Path) -> None:
    checker = _load_checker()
    _git(repo, "checkout", "-q", "-b", "feature")
    _write(repo, "README.md", 9 * 1024)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "grow readme")

    violations = checker.find_violations(
        checker.changed_paths(repo, "main"), root=repo, max_kb=8
    )

    assert [violation.path for violation in violations] == ["README.md"]


def test_scope_includes_uncommitted_working_tree_edits(repo: Path) -> None:
    checker = _load_checker()
    _git(repo, "checkout", "-q", "-b", "feature")
    _write(repo, "staged-only.bin", 9 * 1024)
    _git(repo, "add", "-A")

    paths = checker.changed_paths(repo, "main")

    assert "staged-only.bin" in paths


def test_scope_includes_untracked_files(repo: Path) -> None:
    """`git diff` cannot see a new file before it is staged; the gate must."""
    checker = _load_checker()
    _git(repo, "checkout", "-q", "-b", "feature")
    _write(repo, "never-staged.bin", 9 * 1024)

    violations = checker.find_violations(
        checker.changed_paths(repo, "main"), root=repo, max_kb=8
    )

    assert [violation.path for violation in violations] == ["never-staged.bin"]


def test_scope_respects_gitignore(repo: Path) -> None:
    checker = _load_checker()
    _git(repo, "checkout", "-q", "-b", "feature")
    _write(repo, ".gitignore", 0)
    (repo / ".gitignore").write_text("scratch/\n", encoding="utf-8")
    _write(repo, "scratch/huge.bin", 9 * 1024)

    paths = checker.changed_paths(repo, "main")

    assert "scratch/huge.bin" not in paths


def test_deletions_are_not_reported(repo: Path) -> None:
    checker = _load_checker()
    _git(repo, "checkout", "-q", "-b", "feature")
    _git(repo, "rm", "-q", "existing-huge.bin")
    _git(repo, "commit", "-qm", "drop the big one")

    paths = checker.changed_paths(repo, "main")

    assert paths == []


def test_unresolvable_base_is_a_hard_failure(repo: Path) -> None:
    checker = _load_checker()

    with pytest.raises(checker.BaseRefError):
        checker.changed_paths(repo, "origin/does-not-exist")


def test_exempt_directory_may_exceed_the_ceiling(tmp_path: Path) -> None:
    checker = _load_checker()
    exempt = _write(tmp_path, "tests/fixtures/search_seed/episode.json", 9 * 1024)
    nearby = _write(tmp_path, "tests/fixtures/other_seed.json", 9 * 1024)

    violations = checker.find_violations([exempt, nearby], root=tmp_path, max_kb=8)

    assert [violation.path for violation in violations] == [
        "tests/fixtures/other_seed.json"
    ], "the exemption must cover exactly its directory, not siblings"


def test_exemption_list_is_pinned() -> None:
    """Every entry is a place the ceiling stops protecting — keep it visible."""
    checker = _load_checker()

    assert checker.EXEMPT_PREFIXES == ("tests/fixtures/search_seed/",)


def test_exempt_prefixes_point_at_real_directories() -> None:
    checker = _load_checker()

    for prefix in checker.EXEMPT_PREFIXES:
        assert (_REPO_ROOT / prefix).is_dir(), (
            f"exempt prefix {prefix!r} does not exist; drop it rather than "
            "leaving a hole for a path that may come back"
        )


def test_ceiling_matches_the_pre_commit_hook() -> None:
    checker = _load_checker()
    config = _PRE_COMMIT_CONFIG.read_text(encoding="utf-8")

    hook_limits = re.findall(r"--maxkb=(\d+)", config)

    assert hook_limits == [str(checker.MAX_KB)], (
        "scripts/check_file_sizes.py and the check-added-large-files hook must "
        "share one ceiling; update both in the same commit."
    )


def test_github_pull_request_base_is_preferred(monkeypatch: pytest.MonkeyPatch) -> None:
    checker = _load_checker()

    monkeypatch.setenv("GITHUB_BASE_REF", "release/1.3")
    assert checker.default_base_ref() == "origin/release/1.3"

    monkeypatch.setenv("GITHUB_BASE_REF", "")
    assert checker.default_base_ref() == checker.DEFAULT_BASE

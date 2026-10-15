"""Validate commit subjects against the EverOS Conventional Commits policy."""

from __future__ import annotations

import os
import re
import subprocess
import sys

ZERO_SHA = "0" * 40
ALLOWED_TYPES = (
    "feat",
    "fix",
    "refactor",
    "test",
    "docs",
    "style",
    "perf",
    "chore",
    "build",
    "ci",
    "revert",
)
TITLE_RE = re.compile(rf"^({'|'.join(ALLOWED_TYPES)})(\([A-Za-z0-9._/-]+\))?(!)?: .+")
MAX_TITLE_LENGTH = 72


def _run_git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def _default_range() -> str:
    event_name = os.getenv("GITHUB_EVENT_NAME", "")
    before = os.getenv("GITHUB_EVENT_BEFORE", "") or os.getenv(
        "GITHUB_EVENT_BEFORE_SHA", ""
    )
    after = os.getenv("GITHUB_SHA", "HEAD")
    pr_base = os.getenv("GITHUB_PR_BASE_SHA", "")

    if event_name.startswith("pull_request") and pr_base:
        return _pull_request_range(pr_base)

    if before and before != ZERO_SHA:
        return f"{before}..{after}"

    try:
        _run_git(["rev-parse", "--verify", f"{after}^"])
    except subprocess.CalledProcessError:
        return after
    return f"{after}^..{after}"


def _pull_request_range(pr_base: str) -> str:
    """The commits this PR adds, and nothing the base branch already had.

    ``actions/checkout`` resolves a ``pull_request`` event to
    ``refs/pull/N/merge``: the PR head merged into the *current* base tip.
    ``pull_request.base.sha`` is a different commit -- the base tip from when
    the PR was last synced. Everything the base branch gained in between
    therefore falls inside ``base.sha..HEAD``, so one bad subject merged to
    ``main`` fails every open PR until each of them rebases, blaming authors
    for a commit they did not write.

    The merge commit already names both sides, so ask it: first parent is the
    base tip that was merged in, second is the PR head. A checkout that is not
    a merge (a workflow taking ``head.sha`` directly) has one parent, and
    there the reported base is the best answer available.
    """
    parents = _run_git(["rev-list", "--parents", "-n", "1", "HEAD"]).split()[1:]
    if len(parents) == 2:
        base_tip, pr_head = parents
        return f"{base_tip}..{pr_head}"
    return f"{pr_base}..HEAD"


def _commit_rows(commit_range: str) -> list[tuple[str, str, str]]:
    output = _run_git(
        [
            "log",
            "--format=%H%x00%s%x00%P",
            commit_range,
        ]
    )
    if not output:
        return []

    rows = []
    for line in output.splitlines():
        commit, subject, parents = line.split("\x00", 2)
        rows.append((commit, subject, parents))
    return rows


def _is_exempt(subject: str, parents: str) -> bool:
    if len(parents.split()) > 1:
        return True
    return subject.startswith(("Revert ", "fixup!", "squash!"))


def _validate(commit_range: str) -> list[str]:
    failures: list[str] = []
    for commit, subject, parents in _commit_rows(commit_range):
        if _is_exempt(subject, parents):
            continue

        short = commit[:12]
        if len(subject) > MAX_TITLE_LENGTH:
            failures.append(
                f"{short}: subject is {len(subject)} chars; "
                f"max is {MAX_TITLE_LENGTH}: {subject}"
            )
            continue

        if not TITLE_RE.match(subject):
            allowed = ", ".join(ALLOWED_TYPES)
            failures.append(
                f"{short}: invalid subject: {subject}\n"
                f"  expected: <type>[(scope)][!]: <description>\n"
                f"  allowed types: {allowed}"
            )

    return failures


def main() -> int:
    commit_range = sys.argv[1] if len(sys.argv) > 1 else _default_range()
    failures = _validate(commit_range)
    if failures:
        print(f"Commit message check failed for range {commit_range}:")
        print("\n".join(failures))
        return 1

    print(f"Commit messages follow Conventional Commits for range {commit_range}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Validate pull request titles against the EverOS Conventional Commits policy."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

_SCRIPT_DIR = Path(__file__).resolve().parent


def _load_commit_policy() -> ModuleType:
    policy_path = _SCRIPT_DIR / "check_commit_messages.py"
    spec = importlib.util.spec_from_file_location("_commit_message_policy", policy_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load commit policy from {policy_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_POLICY = _load_commit_policy()
ALLOWED_TYPES = _POLICY.ALLOWED_TYPES
MAX_TITLE_LENGTH = _POLICY.MAX_TITLE_LENGTH
TITLE_RE = _POLICY.TITLE_RE

# Room for the longest suffix a squash merge is likely to append, used when the
# PR number is not known (a local ``make check-pr-title``). CI passes the real
# number and gets an exact budget.
FALLBACK_SUFFIX_LENGTH = len(" (#0000)")


def squash_suffix(pr_number: str | int | None) -> str | None:
    """What a squash merge appends to the title, if the PR number is known.

    This repository merges by squash, and GitHub builds that commit's subject
    as ``<PR title> (#<number>)``. The subject that lands on ``main`` is
    therefore longer than the title the author typed -- which is the only
    thing this check used to measure. v1.4.2 shipped a 73-char subject that
    way, from a title that was a perfectly legal 66, and every open PR then
    failed ``commit messages`` for a commit none of them wrote.
    """
    if pr_number is None or not str(pr_number).strip():
        return None
    return f" (#{str(pr_number).strip()})"


def validate_title(title: str, *, pr_number: str | int | None = None) -> list[str]:
    title = title.strip()
    if not title:
        return ["missing PR title"]

    suffix = squash_suffix(pr_number)
    reserved = len(suffix) if suffix is not None else FALLBACK_SUFFIX_LENGTH
    budget = MAX_TITLE_LENGTH - reserved

    if len(title) > budget:
        appended = f'"{suffix}"' if suffix is not None else f"{reserved} chars"
        return [
            f"PR title is {len(title)} chars; max is {budget}: {title}\n"
            f"  a squash merge appends {appended}, and the commit subject it "
            f"produces may not exceed {MAX_TITLE_LENGTH}"
        ]

    if not TITLE_RE.match(title):
        allowed = ", ".join(ALLOWED_TYPES)
        return [
            f"invalid PR title: {title}\n"
            "  expected: <type>[(scope)][!]: <description>\n"
            f"  allowed types: {allowed}"
        ]

    return []


def _title_from_args_or_env(argv: list[str]) -> str:
    if argv:
        return " ".join(argv)
    return os.getenv("PR_TITLE", "")


def main(argv: list[str] | None = None) -> int:
    title = _title_from_args_or_env(sys.argv[1:] if argv is None else argv)
    failures = validate_title(title, pr_number=os.getenv("PR_NUMBER"))
    if failures:
        print("Pull request title check failed:")
        print("\n".join(failures))
        return 1

    print("Pull request title follows Conventional Commits.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

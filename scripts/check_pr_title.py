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


def validate_title(title: str) -> list[str]:
    title = title.strip()
    if not title:
        return ["missing PR title"]

    if len(title) > MAX_TITLE_LENGTH:
        return [f"PR title is {len(title)} chars; max is {MAX_TITLE_LENGTH}: {title}"]

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
    failures = validate_title(title)
    if failures:
        print("Pull request title check failed:")
        print("\n".join(failures))
        return 1

    print("Pull request title follows Conventional Commits.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

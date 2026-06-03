#!/usr/bin/env python3
"""Scan tracked text files for CJK characters outside the language-policy allowlist.

Replaces the grep-based reference command that used to live in
``.claude/rules/language-policy.md``. That command silently produced false
negatives on this repo: the ``grep -zZv`` + ``xargs -0`` NUL pipeline
mis-parsed the path list and reported "clean" even when violations existed.

Exit code 0 = clean, 1 = violations found (paths + line numbers printed).

Usage:
    python scripts/check_cjk.py            # scan all tracked files
    python scripts/check_cjk.py a.py b.md  # scan specific files (pre-commit)
    python scripts/check_cjk.py --quiet    # per-file counts only
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess

# CJK / fullwidth code points: CJK symbols & ideographs, Hangul syllables,
# and halfwidth/fullwidth forms. Kept as escapes so this file stays ASCII.
_CJK = re.compile("[\\u3000-\\u9fff\\uac00-\\ud7af\\uff00-\\uffef]")


def _is_allowlisted(path: str) -> bool:
    """Return True if CJK is permitted in this path (see language-policy.md)."""
    name = os.path.basename(path)
    # 1. Tests: fixtures, sample inputs, and CJK-behavior assertions.
    if path.startswith("tests/"):
        return True
    # 2. Tokenizer NLP resources (stopword lists, segmentation examples).
    if path.startswith("src/everos/component/tokenizer/"):
        return True
    # 3. Locale-suffixed sample data, e.g. data/solo_chat_zh.json.
    if re.match(r"data/.*_(zh|ja|ko)\.", path):
        return True
    # 4. Translated doc mirrors, e.g. README.zh.md.
    if re.search(r"\.(zh|ja|ko)\.md$", path):
        return True
    # 5. Filenames explicitly marked with a CJK/locale token.
    return bool(re.search(r"(^|[._-])(cjk|zh|ja|ko)([._-]|$)", name))


def _tracked_files() -> list[str]:
    out = subprocess.check_output(["git", "ls-files"], text=True)
    return out.splitlines()


def main() -> int:
    parser = argparse.ArgumentParser(description="CJK language-policy scanner.")
    parser.add_argument("files", nargs="*", help="files to scan (default: all tracked)")
    parser.add_argument("--quiet", action="store_true", help="per-file counts only")
    args = parser.parse_args()

    paths = args.files or _tracked_files()
    violations: list[tuple[str, int, str]] = []
    for path in paths:
        if _is_allowlisted(path):
            continue
        try:
            with open(path, encoding="utf-8") as fh:
                lines = fh.readlines()
        except (UnicodeDecodeError, FileNotFoundError, IsADirectoryError):
            continue  # binary / missing / directory: nothing to scan
        for i, line in enumerate(lines, start=1):
            if _CJK.search(line):
                violations.append((path, i, line.strip()))

    if not violations:
        print("CJK language-policy: clean")
        return 0

    by_file: dict[str, int] = {}
    for path, _lineno, _text in violations:
        by_file[path] = by_file.get(path, 0) + 1

    print(f"CJK language-policy: {len(violations)} hit(s) in {len(by_file)} file(s)\n")
    if args.quiet:
        for path, count in sorted(by_file.items(), key=lambda kv: -kv[1]):
            print(f"  {count:4d}  {path}")
    else:
        for path, lineno, text in violations:
            print(f"  {path}:{lineno}: {text[:100]}")
    print("\nAllowed CJK locations are defined in .claude/rules/language-policy.md")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

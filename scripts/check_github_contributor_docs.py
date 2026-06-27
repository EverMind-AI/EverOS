"""Block legacy/internal branch-model residue in contributor guidance docs."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

MONITORED_FILES = (
    Path("CLAUDE.md"),
    Path("CONTRIBUTING.md"),
    Path("docs/engineering.md"),
    Path(".github/PULL_REQUEST_TEMPLATE.md"),
    Path(".claude/skills/commit/SKILL.md"),
    Path(".claude/skills/new-branch/SKILL.md"),
    Path(".claude/skills/pr/SKILL.md"),
)

BRANCH_WORKFLOW_FILES = frozenset(
    {
        Path("CLAUDE.md"),
        Path(".claude/skills/commit/SKILL.md"),
        Path(".claude/skills/new-branch/SKILL.md"),
        Path(".claude/skills/pr/SKILL.md"),
    }
)


@dataclass(frozen=True)
class Rule:
    label: str
    pattern: re.Pattern[str]
    paths: frozenset[Path] | None = None


@dataclass(frozen=True)
class Violation:
    path: Path
    line_number: int
    label: str
    text: str


_LEGACY_HOSTING = "git" + "lab"
_LEGACY_STABLE_BRANCH = "mas" + "ter"
_LEGACY_INTEGRATION_BRANCH = "de" + "v"
_LEGACY_EMERGENCY_BRANCH = "hot" + "fix"
_LEGACY_BRANCH_MODEL = "git" + "flow"

RULES = (
    Rule(
        "legacy hosting reference",
        re.compile(rf"\b{_LEGACY_HOSTING}\b", re.IGNORECASE),
    ),
    Rule(
        "legacy review wording",
        re.compile(r"\bmerge\s+request\b", re.IGNORECASE),
    ),
    Rule(
        "legacy branch model",
        re.compile(rf"\b{_LEGACY_BRANCH_MODEL}\b", re.IGNORECASE),
    ),
    Rule(
        "legacy stable branch reference",
        re.compile(rf"\b{_LEGACY_STABLE_BRANCH}\b", re.IGNORECASE),
    ),
    Rule(
        "legacy integration branch reference",
        re.compile(
            rf"(?<![A-Za-z]){_LEGACY_INTEGRATION_BRANCH}(?![A-Za-z])",
            re.IGNORECASE,
        ),
        paths=BRANCH_WORKFLOW_FILES,
    ),
    Rule(
        "legacy emergency branch model",
        re.compile(rf"\b{_LEGACY_EMERGENCY_BRANCH}\b", re.IGNORECASE),
        paths=BRANCH_WORKFLOW_FILES,
    ),
)


def find_violations(
    file_texts: Iterable[tuple[Path, str]],
) -> list[Violation]:
    violations: list[Violation] = []
    for path, text in file_texts:
        for line_number, line in enumerate(text.splitlines(), start=1):
            for rule in RULES:
                if rule.paths is not None and path not in rule.paths:
                    continue
                if rule.pattern.search(line):
                    violations.append(
                        Violation(
                            path=path,
                            line_number=line_number,
                            label=rule.label,
                            text=line.strip(),
                        )
                    )
    return violations


def _read_monitored_files() -> list[tuple[Path, str]]:
    return [
        (path, path.read_text(encoding="utf-8"))
        for path in MONITORED_FILES
        if path.exists()
    ]


def main() -> int:
    violations = find_violations(_read_monitored_files())
    if not violations:
        print("GitHub contributor-doc check passed.")
        return 0

    print(
        "GitHub contributor-doc check failed.\n"
        "These files should describe the public GitHub workflow: protected "
        "`main`, scoped branches, and pull requests back to `main`.\n"
    )
    for violation in violations:
        print(
            f"- {violation.path}:{violation.line_number}: "
            f"{violation.label}: {violation.text}"
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

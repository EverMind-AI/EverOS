"""EverOS carries its algorithms in ``everos.algo``; nothing reaches for everalgo."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
_IMPORT = re.compile(r"^\s*(?:from|import)\s+everalgo\b", re.MULTILINE)


def test_no_module_imports_everalgo() -> None:
    offenders = sorted(
        str(path.relative_to(_ROOT))
        for folder in ("src", "tests", "benchmarks", "scripts")
        for path in (_ROOT / folder).rglob("*.py")
        if _IMPORT.search(path.read_text(encoding="utf-8"))
    )
    assert offenders == []


def test_pyproject_declares_no_everalgo_dependency() -> None:
    project = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    declared = [*project["dependencies"]]
    for extra in project.get("optional-dependencies", {}).values():
        declared.extend(extra)
    assert [d for d in declared if d.lower().startswith("everalgo")] == []

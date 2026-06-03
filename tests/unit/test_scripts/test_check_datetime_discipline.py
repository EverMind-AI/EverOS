"""Self-tests for ``scripts/check_datetime_discipline.py``.

The scanner is a hard CI gate, so it must catch every forbidden pattern
listed in :doc:`.claude/rules/datetime-handling` AND must not false-
positive on legitimate code (comments, docstrings, allowlisted files,
``# tz-noqa`` exemptions).

This module monkey-patches the scanner's roots to point at a per-test
``tmp_path`` tree so we can synthesise tiny .py files with exactly the
shape we want to assert on.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCANNER_PATH = _REPO_ROOT / "scripts" / "check_datetime_discipline.py"


def _load_scanner_with(root: Path, allowlist: set[Path] | None = None):
    """Load the scanner module fresh and rewire its ``_ROOT`` / ``_SRC``.

    Returns the loaded module so tests can call ``main()`` or
    ``_scan_file(path)`` directly.
    """
    spec = importlib.util.spec_from_file_location(
        f"_scanner_under_test_{root.name}", _SCANNER_PATH
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod._ROOT = root
    mod._SRC = root / "src"
    mod._ALLOWLIST = allowlist or set()
    return mod


def _make_src(tmp_path: Path, files: dict[str, str]) -> Path:
    """Build a tmp ``src/`` tree from ``relative_path → contents``."""
    src = tmp_path / "src"
    for rel, body in files.items():
        target = src / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")
    return src


# ── Each forbidden pattern is caught ────────────────────────────────────


@pytest.mark.parametrize(
    "line,expected_substring",
    [
        ("x = datetime.now()", "datetime.now()"),
        ("x = datetime.utcnow()", "datetime.utcnow()"),
        ("x = datetime.today()", "datetime.today()"),
        ("x = dt.datetime.now()", "dt.datetime.now()"),
        ("x = dt.datetime.utcnow()", "dt.datetime.utcnow()"),
        ("x = _dt.datetime.now()", "_dt.datetime.now()"),
        ("x = time.time()", "time.time()"),
        ("x = time.time_ns()", "time.time"),
        ("x = d.astimezone(UTC)", ".astimezone"),
        ("x = d.replace(tzinfo=UTC)", ".replace(tzinfo="),
    ],
)
def test_scanner_catches_forbidden_pattern(
    tmp_path: Path, line: str, expected_substring: str
) -> None:
    """Each pattern listed in the rule must produce a violation."""
    body = "import datetime\nimport time\n" + line + "\n"
    _make_src(tmp_path, {"everos/sample.py": body})
    scanner = _load_scanner_with(tmp_path)
    rc = scanner.main()
    assert rc == 1, f"scanner should flag {line!r}"


# ── Allowlist and # tz-noqa exemptions ─────────────────────────────────


def test_scanner_respects_file_allowlist(tmp_path: Path) -> None:
    """Files on the allowlist may legitimately use forbidden patterns."""
    sample_path = tmp_path / "src" / "everos" / "datetime.py"
    _make_src(
        tmp_path,
        {"everos/datetime.py": "x = datetime.now()\n"},
    )
    scanner = _load_scanner_with(tmp_path, allowlist={sample_path})
    rc = scanner.main()
    assert rc == 0, "allowlisted file should not be flagged"


def test_scanner_respects_noqa_tz_marker(tmp_path: Path) -> None:
    """A trailing ``# tz-noqa`` exempts a single line from the gate."""
    _make_src(
        tmp_path,
        {
            "everos/sample.py": (
                "import datetime\n"
                "x = datetime.now()  # tz-noqa -- documented exception\n"
            )
        },
    )
    scanner = _load_scanner_with(tmp_path)
    rc = scanner.main()
    assert rc == 0, "# tz-noqa should suppress the violation"


# ── False-positive guards: comments and docstrings ──────────────────────


def test_scanner_ignores_pure_comment_line(tmp_path: Path) -> None:
    """A pattern inside a pure ``#`` comment is not a violation."""
    _make_src(
        tmp_path,
        {
            "everos/sample.py": (
                "import datetime\n# Don't use datetime.now() — see rules.\n"
            )
        },
    )
    scanner = _load_scanner_with(tmp_path)
    assert scanner.main() == 0


def test_scanner_ignores_triple_quoted_docstring(tmp_path: Path) -> None:
    """Patterns inside a triple-quoted docstring are not violations."""
    body = (
        '"""Module docstring — datetime.now() and .astimezone(UTC) appear here.\n'
        "Multiple lines of explanation with replace(tzinfo=UTC).\n"
        '"""\n'
        "x = 1\n"
    )
    _make_src(tmp_path, {"everos/sample.py": body})
    scanner = _load_scanner_with(tmp_path)
    assert scanner.main() == 0


def test_scanner_ignores_inline_trailing_comment(tmp_path: Path) -> None:
    """A pattern inside an inline ``#`` trailing comment is not a violation."""
    _make_src(
        tmp_path,
        {
            "everos/sample.py": (
                "import datetime\nx = 1  # used to be datetime.now() before Q2\n"
            )
        },
    )
    scanner = _load_scanner_with(tmp_path)
    assert scanner.main() == 0


def test_scanner_clean_on_typical_use_of_helper(tmp_path: Path) -> None:
    """Code that goes through ``component.utils.datetime`` is clean."""
    body = (
        "from everos.component.utils.datetime import (\n"
        "    get_utc_now, get_now_with_timezone, to_display_tz,\n"
        ")\n"
        "ts = get_utc_now()\n"
        "display = to_display_tz(ts)\n"
        "wall = get_now_with_timezone()\n"
    )
    _make_src(tmp_path, {"everos/sample.py": body})
    scanner = _load_scanner_with(tmp_path)
    assert scanner.main() == 0


# ── The real scanner against the real tree is also clean ────────────────


def test_real_repo_passes_discipline_gate() -> None:
    """Pin the live state: ``make ci`` should pass the datetime gate today.

    A regression here means someone introduced a forbidden datetime
    pattern outside the allowlist. Fix the offending line (rules doc
    points at the helpers) instead of weakening the gate.
    """
    spec = importlib.util.spec_from_file_location("_real_scanner", _SCANNER_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main() == 0

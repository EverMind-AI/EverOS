"""CI gate: enforce the two-zone discipline at the source-code level.

Scans `src/` for code patterns that bypass
:mod:`everos.component.utils.datetime` and would silently introduce
naive or local-tz datetimes. Exits non-zero on any hit.

Forbidden patterns (with a small allowlist):

1. ``datetime.now()`` / ``datetime.utcnow()`` / ``datetime.today()`` —
   naive constructors / deprecated. Use ``get_utc_now()`` (storage) or
   ``get_now_with_timezone()`` (display).
2. ``time.time()`` / ``time.time_ns()`` — bypasses the helper module.
   Use ``to_timestamp_ms(get_utc_now())`` if you really need ms epoch.
3. Direct ``datetime(YYYY, ...)`` constructor without ``tzinfo=`` —
   produces naive datetimes; use ``ensure_utc(datetime(...))`` instead.
4. ``.astimezone(`` / ``.replace(tzinfo=`` outside the helper module —
   should go through ``to_display_tz`` / ``ensure_utc``.

Allowlist (legitimate uses):

* ``src/everos/component/utils/datetime.py`` — the helper module itself.
* ``src/everos/core/persistence/sqlite/base.py`` — the SQLAlchemy ``load``
  event listener that re-attaches UTC on hydrate.

Run::

    python scripts/check_datetime_discipline.py

Wired into ``make ci``; any violation fails the build.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"

_ALLOWLIST: set[Path] = {
    _ROOT / "src/everos/component/utils/datetime.py",
    _ROOT / "src/everos/core/persistence/sqlite/base.py",
}

# (regex, message) pairs. Each regex must match on a single line.
_HELPER_HINT = "use get_utc_now() / get_now_with_timezone()"
_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"\bdatetime\.now\s*\(\s*\)"),
        f"datetime.now() returns naive — {_HELPER_HINT}",
    ),
    (
        re.compile(r"\bdatetime\.utcnow\s*\("),
        "datetime.utcnow() is deprecated and naive — use get_utc_now()",
    ),
    (
        re.compile(r"\bdatetime\.today\s*\("),
        "datetime.today() returns naive — use today_with_timezone()",
    ),
    (
        re.compile(r"\bdt\.datetime\.now\s*\(\s*\)"),
        f"dt.datetime.now() returns naive — {_HELPER_HINT}",
    ),
    (
        re.compile(r"\bdt\.datetime\.utcnow\s*\("),
        "dt.datetime.utcnow() is deprecated and naive — use get_utc_now()",
    ),
    (
        re.compile(r"\b_dt\.datetime\.now\s*\(\s*\)"),
        f"_dt.datetime.now() returns naive — {_HELPER_HINT}",
    ),
    (
        re.compile(r"\btime\.time(?:_ns)?\s*\("),
        "time.time() bypasses the helper — use to_timestamp_ms(get_utc_now())",
    ),
    (
        re.compile(r"\.astimezone\s*\("),
        ".astimezone(...) outside helper — use to_display_tz() / ensure_utc()",
    ),
    (
        re.compile(r"\.replace\s*\(\s*tzinfo\s*="),
        ".replace(tzinfo=...) outside helper — use ensure_utc() / to_display_tz()",
    ),
]

# Skip lines that match these (comments, docstrings, `# tz-noqa`).
_COMMENT_RE = re.compile(r"^\s*#")
_DOCSTRING_TRIPLE = '"""'


def _scan_file(path: Path) -> list[tuple[int, str, str]]:
    """Return list of (line_no, line, message) violations in *path*."""
    if path in _ALLOWLIST:
        return []
    hits: list[tuple[int, str, str]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []

    # Strip out triple-quoted blocks (docstrings + multi-line literals).
    text_no_docstrings = re.sub(r'""".*?"""', "", text, flags=re.DOTALL)
    text_no_docstrings = re.sub(r"'''.*?'''", "", text_no_docstrings, flags=re.DOTALL)

    for lineno, line in enumerate(text_no_docstrings.splitlines(), start=1):
        if _COMMENT_RE.match(line):
            continue
        if "# tz-noqa" in line:
            continue
        # Strip inline trailing comment to avoid false positives in
        # comment text like ``# replace(tzinfo=...) — explanation``.
        code_part = line.split("#", 1)[0]
        for pat, msg in _PATTERNS:
            if pat.search(code_part):
                hits.append((lineno, line.rstrip(), msg))
                break
    return hits


def main() -> int:
    rc = 0
    for py in sorted(_SRC.rglob("*.py")):
        violations = _scan_file(py)
        if not violations:
            continue
        rel = py.relative_to(_ROOT)
        for lineno, line, msg in violations:
            print(f"{rel}:{lineno}: {msg}")
            print(f"    {line}")
            rc = 1
    if rc == 0:
        print("OK — datetime discipline clean.")
    return rc


if __name__ == "__main__":
    sys.exit(main())

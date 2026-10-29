"""CI gate: no module-level singletons in ``src/``.

The application bootstrap builds every provider, adapter and connection and
injects it; nothing reaches for a process-wide instance. Three shapes give a
singleton away, and each is a hit:

1. ``_x: X | None = None`` at module level -- the slot of a lazy singleton.
2. ``def get_x() -> X:`` / ``def _get_x() -> X:`` at module level, taking no
   arguments -- its accessor.
3. ``x_repo = ...`` at module level -- a repository built at import time.

Allowlist (facade functions, not singletons):

* ``core/utils/datetime.py`` ``get_utc_now`` / ``get_now_with_timezone``
  -- read the clock; there is no instance behind them.
* ``core/context/request.py`` ``get_request_id`` -- reads a ``ContextVar``,
  which is per task, not per process.
* ``core/observability/metrics/registry.py`` ``get_metrics_registry`` -- the
  metrics facade over the registry ``configure_metrics`` settles once; it is
  configured, never built lazily.

Run::

    python scripts/check_singletons.py

Wired into ``make lint``; any hit fails the build.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src" / "everos"

_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"^(?P<name>_[a-z_]+): .*\| None = None$"),
        "lazy singleton slot",
    ),
    (
        re.compile(r"^def (?P<name>_?get_[a-z_]+)\(\) ->"),
        "singleton accessor",
    ),
    (
        re.compile(r"^(?P<name>[a-z_]+_repo) = "),
        "module-level repository instance",
    ),
]

_FACADES: dict[str, set[str]] = {
    "core/utils/datetime.py": {"get_utc_now", "get_now_with_timezone"},
    "core/context/request.py": {"get_request_id"},
    "core/observability/metrics/registry.py": {"get_metrics_registry"},
}


def _allowed(rel: str, name: str) -> bool:
    return name in _FACADES.get(rel, set())


def _scan(path: Path) -> list[tuple[int, str, str]]:
    """Return ``(line_no, line, kind)`` for each hit in ``path``."""
    rel = path.relative_to(_SRC).as_posix()
    hits: list[tuple[int, str, str]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        for pattern, kind in _PATTERNS:
            match = pattern.match(line)
            if match and not _allowed(rel, match["name"]):
                hits.append((lineno, line, kind))
    return hits


def main() -> int:
    rc = 0
    for path in sorted(_SRC.rglob("*.py")):
        for lineno, line, kind in _scan(path):
            print(f"{path.relative_to(_ROOT)}:{lineno}: {kind}")
            print(f"    {line}")
            rc = 1
    if rc == 0:
        print("OK — no module-level singletons.")
    return rc


if __name__ == "__main__":
    sys.exit(main())

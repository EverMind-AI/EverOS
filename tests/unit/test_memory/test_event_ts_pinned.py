"""``Event.ts`` is the origin of an event's input, read only by the purge.

The purge drops a scoped event older than the latest purge of its scope, and
``ts`` defaults to the time the system took the input in
(``core.purge_marks.ORIGIN``). That is safe only while nothing else reads
``ts`` as "when this event was made" and nothing builds an event with an
explicit ``ts=`` (spec C15). A new reader or writer fails here; decide
whether it means the origin, then list it.
"""

from __future__ import annotations

import ast
from pathlib import Path

import everos

SRC = Path(everos.__file__).parent

ALLOWED: dict[str, int] = {
    "service/purge/_gate.py": 1,  # getattr(event, "ts"): verdict and bind
    "service/purge/_runner.py": 1,  # a stored event's "ts": the wait
    "memory/search/filters.py": 3,  # a filter field kind named "ts", not Event.ts
}
"""Reviewed uses per file (path under src/everos -> count)."""


def _uses() -> dict[str, int]:
    """``.ts``, ``ts=`` and the string ``"ts"`` (``getattr``, a dict key,
    ``model_copy(update=...)``) per file."""
    found: dict[str, int] = {}
    for path in SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        rel = path.relative_to(SRC).as_posix()
        for node in ast.walk(tree):
            if (
                (isinstance(node, ast.Attribute) and node.attr == "ts")
                or (isinstance(node, ast.keyword) and node.arg == "ts")
                or (isinstance(node, ast.Constant) and node.value == "ts")
            ):
                found[rel] = found.get(rel, 0) + 1
    return found


def test_event_ts_has_no_unreviewed_readers_or_writers() -> None:
    assert _uses() == ALLOWED

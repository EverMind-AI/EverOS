"""Test-only crash points: die at a named step, as ``kill -9`` would there.

Some crash windows of a deletion are two adjacent system calls apart, and a
random kill never lands in them. A real-server test starts the process with
``EVEROS_TEST_FAILPOINTS`` set to a comma-separated list of names; the code
calls :func:`failpoint` at each named step, and a listed name ends the
process at once with status 137 (no cleanup runs, like a hard kill). The
variable is read once, at import. Unset -- always, outside such a test --
nothing happens.
"""

from __future__ import annotations

__all__ = ["failpoint"]

import os

_ARMED = frozenset(
    name.strip()
    for name in os.environ.get("EVEROS_TEST_FAILPOINTS", "").split(",")
    if name.strip()
)


def failpoint(name: str) -> None:
    """Exit the process with status 137 if ``name`` is armed."""
    if name in _ARMED:
        os._exit(137)

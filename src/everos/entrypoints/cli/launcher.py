"""Launch the CLI with a bounded glibc arena count before native allocations.

glibc reads allocator environment settings at process startup. Re-exec the
interpreter once when supplying our default; setting os.environ in an already
running interpreter would not configure its allocator. Importing EverOS as a
library never invokes this launcher or changes the host's allocation policy.
"""

from __future__ import annotations

import os
import sys


def _default_arena_environment() -> dict[str, str] | None:
    if sys.platform != "linux" or "MALLOC_ARENA_MAX" in os.environ:
        return None
    # A supplied value, including 0 (glibc's automatic policy), belongs to
    # the operator. Preserve unrelated tunables verbatim as well.
    tunables = os.environ.get("GLIBC_TUNABLES", "").split(":")
    if any(item.partition("=")[0] == "glibc.malloc.arena_max" for item in tunables):
        return None
    try:
        glibc = os.confstr("CS_GNU_LIBC_VERSION")
    except (AttributeError, OSError, ValueError):
        return None
    if not glibc:
        return None
    return {**os.environ, "MALLOC_ARENA_MAX": "2"}


def main() -> None:
    """Run the CLI, respecting explicit allocator settings and Python flags."""
    if environment := _default_arena_environment():
        # orig_argv retains -m, -X, -W and the console-script path. The
        # inherited setting prevents a second exec and reaches reload workers.
        os.execve(
            sys.executable,
            [sys.executable, *sys.orig_argv[1:]],
            environment,
        )

    from everos.entrypoints.cli.main import app

    app()

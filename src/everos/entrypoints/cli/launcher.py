"""Launch the CLI with a memory-friendly allocator before native allocations.

glibc reads allocator environment settings at process startup. Re-exec the
interpreter once when supplying our default; setting os.environ in an already
running interpreter would not configure its allocator. Importing EverOS as a
library never invokes this launcher or changes the host's allocation policy.

Where jemalloc is installed (``libjemalloc.so.2``, Debian/Ubuntu package
``libjemalloc2``) the re-exec preloads it with :data:`JEMALLOC_CONF`; otherwise
glibc runs with two arenas. Measured on a 4-core, 8 GB server (2026-11):
jemalloc doubled keyword search throughput (161 vs 80 QPS) and halved hybrid
search latency at 5 QPS (2.4 vs 4.3 s) while the last hour of an 8-hour
write-and-search soak averaged 1125 MB against glibc's 1221 MB.
"""

from __future__ import annotations

import ctypes.util
import os
import sys

JEMALLOC_CONF = "oversize_threshold:0,dirty_decay_ms:1000,narenas:4"
"""``oversize_threshold:0`` keeps large blocks in jemalloc's arenas instead
of returning each to the kernel on free (per-search page faults fell from
221k to 39k); ``dirty_decay_ms:1000`` returns idle pages after a second;
``narenas:4`` instead of four per core keeps the long-run footprint at or
under glibc's."""


def _allocator_environment() -> dict[str, str]:
    """``MALLOC_ARENA_MAX=2`` plus jemalloc where it is installed and the
    operator chose no allocator of their own. The arena cap stays set beside
    jemalloc: a preload that fails to load leaves glibc, still capped."""
    environment = {"MALLOC_ARENA_MAX": "2"}
    if "LD_PRELOAD" in os.environ or "MALLOC_CONF" in os.environ:
        return environment
    library = ctypes.util.find_library("jemalloc")
    if library:
        environment.update(LD_PRELOAD=library, MALLOC_CONF=JEMALLOC_CONF)
    return environment


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
    return {**os.environ, **_allocator_environment()}


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

"""Paths a markdown writer touched while a caller was collecting them.

Memorize writes the source markdown and returns before cascade has
indexed it. The collector is how that call learns which files to ask
the projection about, so a failed index write is not reported as a
clean extraction.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

_written: ContextVar[set[str] | None] = ContextVar("everos_md_writes", default=None)


@contextmanager
def collect_md_writes() -> Iterator[set[str]]:
    """Yield the set of memory-root-relative paths written in this scope."""
    bucket: set[str] = set()
    token = _written.set(bucket)
    try:
        yield bucket
    finally:
        _written.reset(token)


def note_md_write(root: Path, target: Path) -> None:
    """Record ``target`` when a :func:`collect_md_writes` scope is active.

    Paths are POSIX and relative to ``root``, the same key the cascade
    ledger uses. A write outside a collection, or outside ``root``, is
    ignored.
    """
    bucket = _written.get()
    if bucket is None:
        return
    try:
        bucket.add(target.resolve().relative_to(root).as_posix())
    except ValueError:
        return

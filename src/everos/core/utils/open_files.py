"""The process's open-file soft limit.

Linux defaults it to 1024 and macOS to 256, while the hard limit (the
ceiling a process may raise its own soft limit to, without privileges) is
usually far higher. A burst of searches reached 981 open descriptors in a
2026-11 soak; the LanceDB query cap bounds that, and this raise is the
margin on top. It stops at :data:`TARGET_SOFT_LIMIT` rather than the hard
limit so that a descriptor leak still surfaces.
"""

from __future__ import annotations

__all__ = [
    "TARGET_SOFT_LIMIT",
    "WARN_BELOW",
    "low_limit_warning",
    "raise_open_file_limit",
    "raised_soft_limit",
]

TARGET_SOFT_LIMIT = 8192
WARN_BELOW = 4096
"""Below this the query cap alone (32 queries x a table's data files) can
still reach the limit in a burst, so the operator is told."""


def raised_soft_limit(soft: int, hard: int, target: int = TARGET_SOFT_LIMIT) -> int:
    """The soft limit to run with: ``target`` capped at ``hard``, never lower
    than ``soft`` (an operator's higher setting is kept)."""
    return max(soft, min(target, hard))


def low_limit_warning(limits: tuple[int, int] | None) -> str | None:
    """A warning for the operator when the soft limit stayed below
    :data:`WARN_BELOW` (a hard limit too low to raise it), else ``None``."""
    if limits is None or limits[1] >= WARN_BELOW:
        return None
    return (
        f"open-file soft limit is {limits[1]}, below {WARN_BELOW}: a burst of "
        "searches can fail with 'Too many open files'. Raise the limit for this "
        "process: `ulimit -n`, systemd LimitNOFILE= or docker --ulimit nofile="
    )


def raise_open_file_limit(target: int = TARGET_SOFT_LIMIT) -> tuple[int, int] | None:
    """Raise this process's open-file soft limit toward ``target``.

    Returns:
        ``(before, after)``; equal when nothing changed or the platform
        refused. ``None`` where there is no such limit (Windows).
    """
    try:
        import resource
    except ImportError:
        return None
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    wanted = raised_soft_limit(soft, hard, target)
    if wanted == soft:
        return soft, soft
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (wanted, hard))
    except (ValueError, OSError):
        return soft, soft
    return soft, wanted

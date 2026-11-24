"""The process's open-file soft limit.

Linux defaults it to 1024 and macOS to 256, while the hard limit (the
ceiling a process may raise its own soft limit to, without privileges) is
usually far higher. Concurrent searches spiked to 6,000 open descriptors in
a 2026-11 soak (each one opening a table's data files) and once hit 8192.
A LanceDB search waits for headroom under the limit (:func:`open_file_usage`,
read by the query slot), and this raise is the margin on top. It stops at
:data:`TARGET_SOFT_LIMIT` rather than the hard limit so that a descriptor
leak still surfaces.
"""

from __future__ import annotations

import os

__all__ = [
    "TARGET_SOFT_LIMIT",
    "WARN_BELOW",
    "low_limit_warning",
    "open_file_usage",
    "raise_open_file_limit",
    "raised_soft_limit",
]

TARGET_SOFT_LIMIT = 32768
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
    # An unlimited hard limit caps nothing. RLIM_INFINITY is -1 on Linux, where
    # min(target, hard) would otherwise keep the soft limit where it is.
    cap = target if hard == resource.RLIM_INFINITY else hard
    wanted = raised_soft_limit(soft, cap, target)
    while wanted > soft:
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (wanted, hard))
        except (ValueError, OSError):
            # macOS refuses a soft limit above kern.maxfilesperproc although
            # the hard limit reads "unlimited": step down instead of staying
            # at 256.
            wanted = max(soft, wanted // 2)
        else:
            return soft, wanted
    return soft, soft


def open_file_usage() -> tuple[int, int] | None:
    """``(open descriptors, soft limit)`` of this process.

    ``None`` where either cannot be read (Windows) or the soft limit is
    unlimited. Lists ``/proc/self/fd`` (Linux) or ``/dev/fd`` (macOS); on
    Linux that took 0.05 ms at 150 descriptors and 2.2 ms at 6,000 (2026-11).
    """
    try:
        import resource
    except ImportError:
        return None
    soft = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
    if soft == resource.RLIM_INFINITY:
        return None
    for directory in ("/proc/self/fd", "/dev/fd"):
        try:
            return len(os.listdir(directory)), soft
        except OSError:
            continue
    return None

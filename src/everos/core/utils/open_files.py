"""The process's open-file soft limit.

Linux defaults it to 1024 and macOS to 256, while the hard limit (the
ceiling a process may raise its own soft limit to, without privileges) is
usually far higher. A burst of searches reached 981 open descriptors in a
2026-11 soak; the LanceDB query cap bounds that, and this raise is the
margin on top. It stops at :data:`TARGET_SOFT_LIMIT` rather than the hard
limit so that a descriptor leak still surfaces.
"""

from __future__ import annotations

__all__ = ["TARGET_SOFT_LIMIT", "raise_open_file_limit", "raised_soft_limit"]

TARGET_SOFT_LIMIT = 8192


def raised_soft_limit(soft: int, hard: int, target: int = TARGET_SOFT_LIMIT) -> int:
    """The soft limit to run with: ``target`` capped at ``hard``, never lower
    than ``soft`` (an operator's higher setting is kept)."""
    return max(soft, min(target, hard))


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

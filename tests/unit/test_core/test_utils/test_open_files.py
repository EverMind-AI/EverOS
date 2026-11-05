"""The server raises its open-file soft limit, moderately.

Linux defaults the soft limit to 1024 and macOS to 256; a burst of searches
reached 981 open descriptors in a 2026-11 soak. The raise is a margin, not
the fix (the LanceDB query cap is), so it stops at 8192 instead of the hard
limit: a leak should still surface.
"""

from __future__ import annotations

import sys

import pytest

from everos.core.utils.open_files import (
    TARGET_SOFT_LIMIT,
    raise_open_file_limit,
    raised_soft_limit,
)


@pytest.mark.parametrize(
    ("soft", "hard", "expected"),
    [
        (1024, 1_048_576, TARGET_SOFT_LIMIT),  # Linux default: raised to the target
        (256, 2**63 - 1, TARGET_SOFT_LIMIT),  # macOS: hard is "unlimited"
        (1024, 4096, 4096),  # never past the hard limit
        (10_000, 1_048_576, 10_000),  # an operator's higher limit is kept
        (TARGET_SOFT_LIMIT, 1_048_576, TARGET_SOFT_LIMIT),
    ],
)
def test_the_new_soft_limit(soft: int, hard: int, expected: int) -> None:
    assert raised_soft_limit(soft, hard) == expected


@pytest.mark.skipif(sys.platform == "win32", reason="no RLIMIT_NOFILE on Windows")
def test_raising_takes_effect_and_never_touches_the_hard_limit() -> None:
    import resource

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(512, soft), hard))
    try:
        before, after = raise_open_file_limit() or (0, 0)

        assert before == min(512, soft)
        assert resource.getrlimit(resource.RLIMIT_NOFILE) == (after, hard)
        assert after == raised_soft_limit(before, hard)
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))

"""What memorize asks the index projection, without depending on cascade.

The bootstrap supplies the object. When it is absent, a write keeps the
status the markdown step already earned.
"""

from __future__ import annotations

from collections.abc import Collection
from typing import Protocol


class IndexVerdict(Protocol):
    """How the files a call wrote finished in the projection ledger."""

    observed: bool
    failed_permanent: int
    failed_retryable: int
    pending: int


class IndexProjection(Protocol):
    """Settle the paths a memorize call just wrote."""

    async def await_written(self, paths: Collection[str]) -> IndexVerdict:
        """How ``paths`` finished, or unobserved when cascade is not running."""

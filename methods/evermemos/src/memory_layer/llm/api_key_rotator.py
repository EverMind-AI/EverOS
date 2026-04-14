"""API key round-robin rotator.

Supports multiple keys rotating in turn to spread rate-limit pressure
across OpenRouter and other providers. Behaves identically to a single
key when only one key is supplied.
"""

import itertools
from collections.abc import Sequence
from typing import ClassVar

from core.observation.logger import get_logger

logger = get_logger(__name__)


class ApiKeyRotator:
    """API Key rotator for round-robin selection.

    Spreads rate-limit pressure across multiple keys. Behaves identically
    to a single key when only one key is supplied. Process-level shared
    instance available via ``get_or_create``.

    Args:
        keys: One or more API keys for rotation.

    Note:
        Relies on asyncio single-threaded event loop; not thread-safe.
    """

    _shared: ClassVar["ApiKeyRotator | None"] = None

    def __init__(self, keys: Sequence[str]) -> None:
        if not keys:
            raise ValueError("At least one API key is required")
        if len(keys) != len(set(keys)):
            logger.warning(
                "ApiKeyRotator: duplicate keys detected, rotation may be uneven"
            )
        self._keys: tuple[str, ...] = tuple(keys)
        self._cycle = itertools.cycle(range(len(self._keys)))

    def get_rotation(self) -> tuple[str, ...]:
        """Return all keys starting from the current cycle position.

        Advances the global cycle by one position for load distribution.
        The returned tuple is a per-request local snapshot:

        - ``rotation[0]`` is the key for the first attempt (equivalent to
          a single ``get_next()`` call).
        - ``rotation[1:]`` are the retry keys, starting from the one after
          the first-attempt key, guaranteeing each key is tried before any
          key is reused.

        Concurrent requests each advance the cycle independently, so their
        first-attempt keys are naturally staggered.
        """
        start_idx = next(self._cycle)
        if self.size > 1:
            logger.debug(
                "ApiKeyRotator: selected key index %d/%d", start_idx + 1, self.size
            )
        return tuple(self._keys[(start_idx + i) % self.size] for i in range(self.size))

    @property
    def size(self) -> int:
        """Number of API keys in the rotation pool."""
        return len(self._keys)

    @classmethod
    def get_or_create(cls, raw: str) -> "ApiKeyRotator":
        """Get the process-level shared instance, creating it on first call.

        Subsequent calls return the existing instance; ``raw`` is only used
        for the initial creation and is ignored afterward.
        """
        if cls._shared is None:
            keys = [k.strip() for k in raw.split(",") if k.strip()]
            cls._shared = cls(keys)
        else:
            new_keys = tuple(k.strip() for k in raw.split(",") if k.strip())
            if new_keys != cls._shared._keys:
                logger.warning(
                    "ApiKeyRotator: get_or_create called with different keys, "
                    "returning existing instance (keys are locked at first creation)"
                )
        return cls._shared

    def __repr__(self) -> str:
        return f"ApiKeyRotator(size={self.size})"

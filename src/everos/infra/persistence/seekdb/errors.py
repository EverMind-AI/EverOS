"""Typed failures raised at the SeekDB adapter boundary."""

from __future__ import annotations

from everos.core.errors import ConfigurationError


class SeekdbConfigurationError(ConfigurationError):
    """SeekDB settings or optional runtime dependencies are invalid."""


class SeekdbSchemaMismatchError(RuntimeError):
    """A physical SeekDB table no longer matches its logical schema."""


class SeekdbValueLimitError(ValueError):
    """A record cannot be represented by the declared SeekDB schema."""


class SeekdbIntegrityError(RuntimeError):
    """A SeekDB write violates a uniqueness or integrity constraint."""


class SeekdbOperationalError(RuntimeError):
    """SeekDB rejected or could not execute an adapter operation."""


__all__ = [
    "SeekdbConfigurationError",
    "SeekdbIntegrityError",
    "SeekdbOperationalError",
    "SeekdbSchemaMismatchError",
    "SeekdbValueLimitError",
]

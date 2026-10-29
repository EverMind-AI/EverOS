"""Fixed stand-ins for the configuration readers and capabilities bootstrap injects."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from everalgo.types import RawFile

from everos.config.settings import DeciderSettings


def static_tuning(
    settings: DeciderSettings | None = None,
) -> Callable[[], DeciderSettings]:
    """A tuning reader that always returns ``settings`` (defaults when omitted)."""
    fixed = settings if settings is not None else DeciderSettings()
    return lambda: fixed


def static_trace_path(path: str | None = None) -> Callable[[], str | None]:
    """A trace-path reader that always returns ``path`` (no trace when omitted)."""
    return lambda: path


async def unused_raw_file_builder(item: dict[str, Any]) -> RawFile:
    """Stands in for file hydration the code under test must not reach."""
    raise NotImplementedError(f"raw_file_builder is not faked (item={item!r})")

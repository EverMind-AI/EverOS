"""Dataset adapters, resolved by name.

``run.py`` never imports a specific adapter; it asks for one by ``--benchmark``. Adding
a benchmark means adding a module here and one line to the registry -- no change to the
pipeline.
"""

from __future__ import annotations

from types import ModuleType

from . import evermembench, locomo, longmemeval, subtlememory

_REGISTRY: dict[str, ModuleType] = {
    "locomo": locomo,
    "longmemeval": longmemeval,
    "evermembench": evermembench,
    "subtlememory": subtlememory,
}


def get(name: str) -> ModuleType:
    """Return the adapter module for ``name``, or raise with the valid choices."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown benchmark {name!r}; choices: {sorted(_REGISTRY)}"
        ) from None


def names() -> list[str]:
    return sorted(_REGISTRY)


__all__ = ["get", "names"]

"""Import shim for the Hermes-agnostic plugin submodules.

The plugin's ``integrations/hermes/__init__.py`` wires the Hermes
``MemoryProvider`` ABC and imports Hermes-only symbols (``agent``,
``hermes_constants``, ``tools``, ``utils``) that are not installable from
this repo. The logic modules under test (``_client`` / ``_config`` /
``_formatting`` / ``_setup`` / ``_types`` / ``_constants``) are
Hermes-agnostic; once their parent package is registered they import
cleanly with no stubbing of Hermes symbols.

We register a synthetic ``hermes`` package in ``sys.modules`` (with
``__path__`` pointing at the real plugin directory) so importing
``hermes._client`` etc. initialises only that submodule — the Hermes
shell in ``__init__.py`` is never executed.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType

_PLUGIN_DIR = Path(__file__).resolve().parents[4] / "integrations" / "hermes"

_SUBMODULES = (
    "_constants",
    "_types",
    "_client",
    "_config",
    "_formatting",
    "_setup",
)


def _install_synthetic_hermes_package() -> None:
    if "hermes" in sys.modules:
        return
    pkg = ModuleType("hermes")
    pkg.__path__ = [str(_PLUGIN_DIR)]  # type: ignore[attr-defined]
    pkg.__package__ = "hermes"
    sys.modules["hermes"] = pkg


def _ensure_loaded() -> None:
    _install_synthetic_hermes_package()
    for name in _SUBMODULES:
        importlib.import_module(f"hermes.{name}")


_ensure_loaded()

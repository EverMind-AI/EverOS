"""
Project-level pytest conftest.

Ensures ``src/`` is on ``sys.path`` the same way ``make test`` does via
``PYTHONPATH=src``. The project's internal imports (``common_utils``,
``memory_layer``, ``core``, ``agentic_layer`` etc.) live under ``src/`` and
expect to be imported as top-level modules.

Without this file, tests that pull in any of those modules fail with
``ModuleNotFoundError`` unless the developer remembers to prefix pytest
with the env var.
"""
from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).parent / "src"
if _SRC.is_dir():
    src_str = str(_SRC)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)

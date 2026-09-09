#!/usr/bin/env python3
"""Add hooklib/ to sys.path for hook scripts."""

from __future__ import annotations

import sys
from pathlib import Path

_HOOKS_DIR = Path(__file__).resolve().parent
_ROOT = _HOOKS_DIR
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

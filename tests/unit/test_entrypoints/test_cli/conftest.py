"""Shared helpers for the CLI unit tests.

``_hermetic_env`` was copied into four modules before this file existed. Each
of them runs the CLI in a subprocess, and each needs the same thing: an
environment with no ambient ``EVEROS_*`` settings, so a developer machine
exporting a root or a key cannot decide the outcome.
"""

from __future__ import annotations

import os


def hermetic_env() -> dict[str, str]:
    """Subprocess environment with every ambient EverOS setting stripped."""
    env = {
        key: value for key, value in os.environ.items() if not key.startswith("EVEROS_")
    }
    env["EVEROS_LLM__API_KEY"] = ""
    env["EVEROS_EMBEDDING__API_KEY"] = ""
    env["EVEROS_RERANK__API_KEY"] = ""
    return env

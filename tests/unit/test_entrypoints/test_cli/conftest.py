"""Shared helpers for the CLI unit tests.

Every module that runs the CLI in a subprocess needs the same thing: an
environment with no ambient ``EVEROS_*`` settings, so a developer machine
exporting a root or a key cannot decide the outcome. In-process commands get a
placeholder LLM key, because building the application requires one.
"""

from __future__ import annotations

import os

import pytest


def hermetic_env() -> dict[str, str]:
    """Subprocess environment with every ambient EverOS setting stripped.

    The extraction LLM is required to build the application, so it gets a
    placeholder key (building a client never calls it; ``base_url`` comes
    from the bundled defaults); the optional providers stay unconfigured.
    """
    env = {
        key: value for key, value in os.environ.items() if not key.startswith("EVEROS_")
    }
    env["EVEROS_LLM__API_KEY"] = "test-key"
    env["EVEROS_EMBEDDING__API_KEY"] = ""
    env["EVEROS_RERANK__API_KEY"] = ""
    return env


@pytest.fixture(autouse=True)
def _placeholder_llm_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """In-process CLI commands build the application, which needs an LLM.

    A placeholder key satisfies the build without a network call, whatever
    the host carries; a test about the missing LLM blanks it again.
    """
    monkeypatch.setenv("EVEROS_LLM__API_KEY", "test-key")

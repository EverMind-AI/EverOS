"""User-facing error message helpers for missing provider configuration.

Guidance points users at the ``<root>/everos.toml`` file, not at
environment variables. Env vars still work as an override mechanism
(see ``pydantic-settings`` precedence in ``config/settings.py``) but
are not surfaced in onboarding-facing text.
"""

from __future__ import annotations

from everos.core.persistence import MemoryRoot


def missing_config_error(field_label: str, toml_section: str) -> str:
    """Return a uniform error message for a missing config field.

    Args:
        field_label: Human-readable label (e.g. ``"LLM api_key"``).
        toml_section: TOML section name without brackets (e.g. ``"llm"``).

    Returns:
        A single-line message including the resolved memory-root path
        and a hint to run ``everos init``. Never mentions env vars.
    """
    root = MemoryRoot.resolve().root
    return (
        f"{field_label} is not configured. "
        f"Edit {root}/everos.toml (run `everos init` to scaffold), "
        f"section [{toml_section}]."
    )

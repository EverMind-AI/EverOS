"""Config processing capability.

YAML loader for category-organised config trees (PromptSlot templates,
etc.). Distinct from :mod:`everos.config` (configuration *data* + Settings
schema, which uses TOML for the Pydantic Settings file) — this subpackage
holds *capability* (how to load), the other holds *data* (what to load).

External usage:
    from everos.component.config import YamlConfigLoader
"""

from .loader import YamlConfigLoader as YamlConfigLoader

__all__ = ["YamlConfigLoader"]

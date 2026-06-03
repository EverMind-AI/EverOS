"""Prompt slot loader — wraps :class:`YamlConfigLoader` for prompt templates.

Slot file shape::

    # config/prompt_slots/<name>.yaml
    enabled: false
    template: ""

When ``enabled`` is ``True`` and ``template`` is a non-empty string, the
loader returns it as-is; otherwise it returns ``None``. The pipeline
forwards ``None`` directly to algo, where the bundled default prompt is
used (zero override cost).

Three-layer overlay (defaults → ``~/.everos/prompt_slots/`` → runtime
override) is reserved for a future milestone; this version only resolves
the bundled defaults under ``src/everos/config/prompt_slots/``.
"""

from __future__ import annotations

from pathlib import Path

from everos.component.config import YamlConfigLoader

_CATEGORY = "prompt_slots"


class PromptLoader:
    """Read prompt template strings from ``config/prompt_slots/<name>.yaml``.

    Returns ``None`` when the slot is disabled or the template is empty.
    """

    def __init__(self, config_root: Path) -> None:
        self._loader = YamlConfigLoader(
            root=config_root,
            categories={_CATEGORY: None},
        )

    def load(self, name: str) -> str | None:
        """Return the override prompt for ``name``, or ``None`` to use algo default.

        ``None`` is returned when any of the following holds:

        - the slot file is missing,
        - the slot file has ``enabled: false`` (or no ``enabled`` key),
        - the ``template`` field is missing or an empty string.
        """
        try:
            slot = self._loader.find(_CATEGORY, name)
        except FileNotFoundError:
            return None
        if not slot.get("enabled", False):
            return None
        template = slot.get("template")
        if not isinstance(template, str) or not template.strip():
            return None
        return template

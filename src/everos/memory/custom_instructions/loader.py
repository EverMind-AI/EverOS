"""Load the instructions a host appends to an everalgo operator's own rules.

File shape (``config/custom_instructions/<operator>.yaml``)::

    enabled: false
    instructions: ""

When ``enabled`` is true and ``instructions`` is a non-blank string the
loader returns it; otherwise ``None``. The text is passed to the operator as
``custom_instructions=``: everalgo appends it after the bundled rules, so it
can add a requirement but never replace a rule or the output shape.
"""

from __future__ import annotations

__all__ = ["CustomInstructions", "Operator"]

from pathlib import Path
from typing import Literal

from everos.component.yaml_loader import YamlConfigLoader

type Operator = Literal[
    "boundary_detection", "episode_extract", "profile_extract", "reflect"
]

_CATEGORY = "custom_instructions"


class CustomInstructions:
    """Per-operator custom instructions from ``<config_root>/custom_instructions/``.

    Args:
        config_root: Directory holding the ``custom_instructions`` folder.
    """

    def __init__(self, config_root: Path) -> None:
        self._loader = YamlConfigLoader(root=config_root, categories={_CATEGORY: None})

    def load(self, operator: Operator) -> str | None:
        """Return the operator's instructions, or ``None`` when there are none.

        ``None`` when the file is missing, ``enabled`` is not true, or
        ``instructions`` is missing or blank.
        """
        try:
            entry = self._loader.find(_CATEGORY, operator)
        except FileNotFoundError:
            return None
        if not entry.get("enabled", False):
            return None
        instructions = entry.get("instructions")
        if not isinstance(instructions, str) or not instructions.strip():
            return None
        return instructions

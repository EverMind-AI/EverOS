"""Custom instructions appended to everalgo operators' bundled rules.

External usage:
    from everos.memory.custom_instructions import CustomInstructions

Bundled files live under ``src/everos/config/custom_instructions/``, one
per operator; all ship disabled.
"""

from __future__ import annotations

from .loader import CustomInstructions as CustomInstructions
from .loader import Operator as Operator

__all__ = ["CustomInstructions", "Operator"]

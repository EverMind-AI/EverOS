"""Prompt strings for the multimodal parser, organised by language.

Prompts live as module-level string constants under
``prompts/{en,zh}/<operator>.py``. The parser operators import from ``prompts.en``
by default; the bundled rules change only by rebinding a constant at startup.
"""

from __future__ import annotations

from everos.component.parser.backend.prompts import en, zh

__all__ = ["en", "zh"]

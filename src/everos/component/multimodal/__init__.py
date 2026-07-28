"""Multimodal LLM capability — optional vision/audio support for parsing.

Public surface:

- :class:`MultimodalLLMCapability` — soft-dependency wrapper around an
  optional multimodal LLMClient (``available`` / ``require``; no soft-degrade
  accessor — multimodal parsing is entirely optional).
- :func:`get_multimodal_llm_capability` — process-wide lazy singleton accessor
  for :class:`MultimodalLLMCapability`.

External usage::

    from everos.component.multimodal import get_multimodal_llm_capability
    cap = get_multimodal_llm_capability()
    if cap.available:
        client = cap.require()
"""

from __future__ import annotations

from .accessor import get_multimodal_llm_capability as get_multimodal_llm_capability
from .capability import MultimodalLLMCapability as MultimodalLLMCapability

__all__ = [
    "MultimodalLLMCapability",
    "get_multimodal_llm_capability",
]

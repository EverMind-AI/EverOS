"""PromptSlot — prompt template loading.

External usage:
    from everos.memory.prompt_slots import PromptLoader

Three-layer overlay (defaults → ``~/.everos/prompt_slots/`` → runtime
override) is reserved for a future milestone; this version only resolves
the bundled defaults under ``src/everos/config/prompt_slots/``.
"""

from .loader import PromptLoader as PromptLoader

__all__ = ["PromptLoader"]

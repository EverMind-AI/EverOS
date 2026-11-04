"""English prompts for parser operators."""

from __future__ import annotations

from everos.component.parser.backend.prompts.en.audio import PROMPT_FOR_AUDIO
from everos.component.parser.backend.prompts.en.document import (
    HTML_USER_PROMPT,
    PROMPT_FOR_FILE,
    PROMPT_FOR_HTML,
)
from everos.component.parser.backend.prompts.en.image import (
    MERGE_USER_PROMPT,
    PROMPT_FOR_MERGE,
    PROMPT_FOR_PICTURE,
)

__all__ = [
    "HTML_USER_PROMPT",
    "MERGE_USER_PROMPT",
    "PROMPT_FOR_AUDIO",
    "PROMPT_FOR_FILE",
    "PROMPT_FOR_HTML",
    "PROMPT_FOR_MERGE",
    "PROMPT_FOR_PICTURE",
]

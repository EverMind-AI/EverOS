"""General-purpose tools any algorithm package may use, one kind per module.

Prompt strings are not tools: they live next to their algorithm, at ``<subpkg>/prompts/{en,zh}/<name>.py``.
"""

from __future__ import annotations

__all__ = [
    "InvalidInputError",
    "append_custom_instructions",
    "ascii_width",
    "count_tokens",
    "decode_bytes",
    "encode",
    "format_utc_date",
    "format_utc_time",
    "json_default",
    "paragraphs",
    "parse_utc_date",
    "render_chat_line",
    "render_prompt",
    "speaker",
    "tagged",
    "truncate_text",
]

import logging

from everos.algo.utils.chat_lines import render_chat_line, speaker
from everos.algo.utils.errors import InvalidInputError
from everos.algo.utils.json import json_default
from everos.algo.utils.markup import append_custom_instructions, paragraphs, tagged
from everos.algo.utils.templates import render_prompt
from everos.algo.utils.text import ascii_width, decode_bytes
from everos.algo.utils.timestamps import (
    format_utc_date,
    format_utc_time,
    parse_utc_date,
)
from everos.algo.utils.tokens import count_tokens, encode, truncate_text

# Library-style logging: silent until the application configures handlers.
logging.getLogger(__name__).addHandler(logging.NullHandler())

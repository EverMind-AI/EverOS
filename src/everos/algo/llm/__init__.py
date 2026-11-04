"""LLM facade: the ``LLMClient`` contract, its request types, request builders for implementations, reply parsing."""

from __future__ import annotations

__all__ = [
    "ContentPart",
    "ImageUrlInner",
    "ImageUrlPart",
    "LLMClient",
    "LLMMessage",
    "LLMOutputError",
    "StrictOutputModel",
    "TextPart",
    "describe_errors",
    "image_url_part_from_bytes",
    "parse_reply",
    "to_anthropic_response_format",
    "to_gemini_response_format",
    "to_ollama_response_format",
    "to_openai_response_format",
]  # fmt: skip

import logging

from everos.algo.llm.errors import LLMOutputError
from everos.algo.llm.output import StrictOutputModel, describe_errors, parse_reply
from everos.algo.llm.protocols import LLMClient
from everos.algo.llm.schema import (
    to_anthropic_response_format,
    to_gemini_response_format,
    to_ollama_response_format,
    to_openai_response_format,
)
from everos.algo.llm.types import (
    ContentPart,
    ImageUrlInner,
    ImageUrlPart,
    LLMMessage,
    TextPart,
    image_url_part_from_bytes,
)

# Library-style logging: silent until the application configures handlers.
_logger = logging.getLogger(__name__)
_logger.addHandler(logging.NullHandler())

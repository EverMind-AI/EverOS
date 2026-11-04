"""The one error the ``LLMClient`` contract defines."""

from __future__ import annotations

__all__ = ["LLMOutputError"]


class LLMOutputError(Exception):
    """An operator rejected the model's reply: it failed to parse or failed a business check."""

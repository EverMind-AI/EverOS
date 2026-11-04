"""The reply model every parser prompt asks for, and the one LLM callsite every parser call goes through.

Each parser output is free text (an OCR, a transcription, a merged OCR text). The contract has no free-text tier, so
every prompt asks for the ``{"text": "..."}`` object instead. One call, one
validation, no retry: the host decides (§4, §5).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from everos.algo.llm import LLMMessage, StrictOutputModel, parse_reply

if TYPE_CHECKING:
    from everos.algo.llm import ContentPart, LLMClient


class _TextReply(StrictOutputModel):
    """The ``{"text": "..."}`` skeleton every parser prompt's ``<reply_format>`` shows."""

    text: str


async def _call_llm_for_text(
    llm: LLMClient,
    *,
    system: str,
    user: str | list[ContentPart],
    operator: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """Send the rules as the system message and the input as the user message; return the reply's ``text``.

    Args:
        llm: The client to call.
        system: The prompt's rules, ending with its ``<reply_format>``.
        user: The input: the tagged text, or the file's content parts.
        operator: Names the call in a parse error, e.g. ``"parser pdf"``.
        temperature: Passed through; None leaves it to the host.
        max_tokens: Passed through; None leaves it to the host.

    Returns:
        The reply's ``text``, verbatim; empty when the model returned an empty string.

    Raises:
        LLMOutputError: If the reply is not a valid ``{"text": "..."}`` object.
    """
    messages = [
        LLMMessage(role="system", content=system),
        LLMMessage(role="user", content=user),
    ]
    reply = await llm.achat(
        messages,
        response_format=_TextReply,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return parse_reply(_TextReply, reply, operator=operator).text

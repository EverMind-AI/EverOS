"""Boundary requests with the window's count after the window, so they share a cache.

Every add re-sends the session's whole uncut buffer to the boundary model, and the
buffer only grows between cuts, so consecutive requests of one session repeat their
first records verbatim. everalgo-boundary (0.4.1, and 0.4.2 alike) renders the user
message as ``<messages count="N">``, the numbered records, ``</messages>``: the
count, the one part that changes, comes first, so a provider's prefix cache can
match nothing past the system prompt. Measured 2026-11-11 on a 10K-word audio
transcript (25 adds, two sessions, DeepSeek): consecutive boundary calls shared
only the 1,664-token system prompt, and boundary detection was 21-43% of the
extraction cost.

:class:`CountAfterMessages` sends the same request with the count moved behind the
records -- ``<messages>``, the records, ``</messages>``, ``<count>N</count>`` -- so
the previous call's records are a prefix of the next call's. Nothing else changes:
the records, their numbering and the system message are byte-identical. A message in
any other shape is sent unchanged, so an everalgo release that renders the request
differently loses the reorder rather than breaking the call.
"""

from __future__ import annotations

__all__ = ["CountAfterMessages", "count_after_messages"]

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from everalgo.llm import LLMClient, LLMMessage
    from pydantic import BaseModel

_OPENING = re.compile(r'<messages count="(\d+)">\n')
_CLOSING = "\n</messages>"


def count_after_messages(content: str) -> str:
    """``content`` with a leading ``<messages count="N">`` block's count moved after it.

    The records are escaped by everalgo (``<`` is ``&lt;``), so ``</messages>``
    can only be the block's own closing tag. Anything that is not exactly one such
    block comes back unchanged.
    """
    opening = _OPENING.match(content)
    if opening is None or not content.endswith(_CLOSING):
        return content
    records = content[opening.end() : -len(_CLOSING)]
    return f"<messages>\n{records}\n</messages>\n<count>{opening.group(1)}</count>"


class CountAfterMessages:
    """An ``LLMClient`` sending boundary requests with the count after the window.

    Args:
        llm: The client every request goes to.
    """

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def achat(
        self,
        messages: list[LLMMessage],
        *,
        response_format: type[BaseModel],
        temperature: float | None = None,
        max_tokens: int | None = None,
        **extra: Any,
    ) -> str:
        """Forward the request, its user message reordered (see the module)."""
        reordered = [
            message.model_copy(update={"content": count_after_messages(content)})
            if message.role == "user" and isinstance(content := message.content, str)
            else message
            for message in messages
        ]
        return await self._llm.achat(
            reordered,
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_tokens,
            **extra,
        )

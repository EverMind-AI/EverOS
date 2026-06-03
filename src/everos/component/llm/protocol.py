"""LLM client protocol re-export.

The structural contract every everos LLM provider satisfies is the same
:class:`everalgo.llm.LLMClient` Protocol — everos providers must be
pass-through-compatible with the everalgo extractors that accept an
``llm=`` parameter. Re-exporting the type here keeps the import path
stable (``everos.component.llm``) even if the everalgo namespace
shifts later.

The :class:`ChatMessage` / :class:`ChatResponse` / :class:`Usage`
shapes are likewise re-exported so callers can build / inspect chat
payloads without reaching into the everalgo package directly.
"""

from __future__ import annotations

from everalgo.llm import (
    ChatMessage as ChatMessage,
)
from everalgo.llm import (
    ChatResponse as ChatResponse,
)
from everalgo.llm import (
    LLMClient as LLMClient,
)
from everalgo.llm import (
    LLMError as LLMError,
)
from everalgo.llm import (
    Usage as Usage,
)

__all__ = [
    "ChatMessage",
    "ChatResponse",
    "LLMClient",
    "LLMError",
    "Usage",
]

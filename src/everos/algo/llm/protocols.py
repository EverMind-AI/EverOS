"""The ``LLMClient`` protocol: what operators call and hosts implement."""

from __future__ import annotations

__all__ = ["LLMClient"]

from typing import Any, Protocol, runtime_checkable

# Imported at runtime, not under TYPE_CHECKING: the public contract keeps its annotations resolvable by
# typing.get_type_hints, for hosts that introspect it.
from pydantic import BaseModel

from everos.algo.llm.types import LLMMessage


@runtime_checkable
class LLMClient(Protocol):
    """Async chat client an operator is given; any class with a matching ``achat`` conforms (PEP 544)."""

    async def achat(
        self,
        messages: list[LLMMessage],
        *,
        response_format: type[BaseModel],
        temperature: float | None = None,
        max_tokens: int | None = None,
        **extra: Any,
    ) -> str:
        """Send one chat request and return the model's reply text.

        ``everos.component.llm.openai_provider`` is the reference implementation.

        Args:
            messages: The request; every content part, non-text ones included, must reach the upstream.
            response_format: Pydantic model the operator validates the reply against with ``model_validate_json``.
                How the implementation gets the model to produce it is its own choice.
            temperature: Sent exactly as given. If ``None``, the implementation sends its own default, or leaves the
                parameter out when it has none.
            max_tokens: Same as ``temperature``; no request, retries included, may raise it.
            **extra: Provider-specific parameters, passed through.

        Returns:
            The reply text, expected to pass ``response_format.model_validate_json``.

        Implementations MUST (operators cannot detect a breach):
            - never let a configured default override an explicit value;
            - raise instead of silently dropping a content part they cannot deliver;
            - let ``asyncio.CancelledError`` propagate;
            - raise when the upstream reports a refusal, even if the content is valid JSON;
            - retry or raise when the upstream reports truncation or an abnormal stop.

        Raises:
            Exception: Implementation-defined; operators let it propagate unchanged.
        """
        ...

"""
Token Usage Collector Interface

Request-level token usage collector for LLM/embedding/rerank calls.
Opensource provides a Noop implementation; Enterprise overrides via addon bean.
"""

from abc import ABC, abstractmethod
from typing import Optional

from core.di.decorators import component


class TokenUsageCollector(ABC):
    """
    Request-level token usage collector interface.

    Called by LLM/embedding/rerank call sites after each invocation to accumulate
    token usage. Enterprise reads totals at request end for MCU billing.

    Lifecycle:
    - reset(): middleware at request start (before asyncio.create_task)
    - add(): each LLM/embedding/rerank call site after response
    - get_totals(): enterprise on_request_complete for MCU calculation
    - reset(): middleware at request end (cleanup)
    """

    @abstractmethod
    def add(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        *,
        call_type: str = "llm",
        request_id: Optional[str] = None,
    ) -> None:
        """
        Accumulate one invocation's token usage into the current request context.

        Multiple calls within a single request are summed (not overwritten).

        Args:
            model: Model identifier (e.g. "gpt-4", "gemini-2.5-flash")
            input_tokens: Prompt / input token count
            output_tokens: Completion / output token count
            call_type: "llm" | "embedding" | "rerank"
            request_id: Optional request ID for diagnostics
        """
        ...

    @abstractmethod
    def get_totals(self) -> dict:
        """
        Return accumulated totals for the current request.

        Returns:
            {
                "input_tokens": int,
                "output_tokens": int,
                "embedding_calls": int,
                "call_count": int,
            }
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset accumulated values (called at request start and end)."""
        ...


@component(name="token_usage_collector")
class NoopTokenUsageCollector(TokenUsageCollector):
    """Noop implementation — opensource does not need MCU billing."""

    def add(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        *,
        call_type: str = "llm",
        request_id: Optional[str] = None,
    ) -> None:
        pass

    def get_totals(self) -> dict:
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "embedding_calls": 0,
            "call_count": 0,
        }

    def reset(self) -> None:
        pass

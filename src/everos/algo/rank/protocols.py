"""Retrieval protocols and decision types — caller-implementable boundaries.

``RetrieveFn`` / ``RerankFn`` are caller-supplied callables: caller binds storage
/ index / model client inside; algo only sees the abstract ``(query, k) -> list[Candidate]``
surface. See the package README for the algo-vs-caller responsibility split.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from everos.algo.types import Candidate

__all__ = ["AgenticDecision", "RerankFn", "RetrieveFn"]

RetrieveFn = Callable[[str, int], Awaitable[list[Candidate]]]
"""Retrieve top-k candidates for a query. Caller binds storage / index / client inside."""

RerankFn = Callable[[str, list[Candidate]], Awaitable[list[Candidate]]]
"""Rerank candidates for a query, returning a (possibly truncated) reordered list."""


class AgenticDecision(BaseModel):
    """Decisions emitted by ``aagentic_retrieve`` that caller cannot reconstruct externally.

    Caller-side metrics (token / latency / result counts / per-stage traces) are
    caller's responsibility — wrap your LLMClient / RetrieveFn for accounting,
    or use ``time.monotonic()`` and ``len(results)`` directly.
    """

    model_config = ConfigDict(frozen=True)

    is_multi_round: bool
    is_sufficient: bool | None = None
    reasoning: str | None = None
    missing_info: list[str] = Field(default_factory=list)
    key_information_found: list[str] = Field(default_factory=list)
    refined_queries: list[str] = Field(default_factory=list)
    query_strategy: Literal["multi_query", "refined_query"] | None = None

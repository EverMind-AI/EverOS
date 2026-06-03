"""LLM lifespan provider — eagerly resolves the LLM singleton at startup.

The framework's core value (memory extraction) is meaningless without
an LLM, so misconfiguration must surface as a startup failure instead
of N silent skips per request downstream. Ordered before the storage
stack so we fail before paying to bring sqlite / lancedb / cascade up.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI

from everos.component.llm import get_llm_client
from everos.core.lifespan import LifespanProvider
from everos.core.observability.logging import get_logger

logger = get_logger(__name__)


class LLMLifespanProvider(LifespanProvider):
    """Resolve the LLM client at startup; raise if credentials are missing."""

    def __init__(self, order: int = 8) -> None:
        super().__init__(name="llm", order=order)

    async def startup(self, app: FastAPI) -> Any:
        client = get_llm_client()
        logger.info("llm_lifespan_ready")
        return client

    async def shutdown(self, app: FastAPI) -> None:
        # The client is stateless (algo facade over openai.AsyncOpenAI);
        # nothing to tear down.
        return None

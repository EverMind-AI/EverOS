"""Boundary-tokenizer lifespan provider.

Prewarms the tiktoken encoding used by everalgo boundary detection so
the first ``/api/v1/memory/add`` request does not block on an on-demand
download of ``o200k_base``.
"""

from __future__ import annotations

from typing import Any

import tiktoken
from fastapi import FastAPI

from everos.core.lifespan import LifespanProvider
from everos.core.observability.logging import get_logger

logger = get_logger(__name__)

_BOUNDARY_ENCODING_NAME = "o200k_base"


def _warm_boundary_tokenizer() -> tiktoken.Encoding:
    """Resolve the boundary detector's shared tiktoken encoding."""

    try:
        return tiktoken.get_encoding(_BOUNDARY_ENCODING_NAME)
    except Exception as exc:  # pragma: no cover - exercised via provider tests
        raise RuntimeError(
            "failed to prewarm the boundary tokenizer encoding "
            f"{_BOUNDARY_ENCODING_NAME!r}; start the server once with network "
            "access so tiktoken can cache it before serving /api/v1/memory/add"
        ) from exc


class BoundaryTokenizerLifespanProvider(LifespanProvider):
    """Prewarm the boundary tokenizer at startup; fail before serving traffic."""

    def __init__(self, order: int = 9) -> None:
        super().__init__(name="boundary_tokenizer", order=order)

    async def startup(self, app: FastAPI) -> Any:
        encoding = _warm_boundary_tokenizer()
        logger.info(
            "boundary_tokenizer_lifespan_ready",
            encoding=_BOUNDARY_ENCODING_NAME,
        )
        return encoding

    async def shutdown(self, app: FastAPI) -> None:
        # tiktoken keeps a process-local cache; nothing to tear down.
        return None

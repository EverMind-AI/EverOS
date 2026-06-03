"""Cascade lifespan provider — starts/stops :class:`CascadeOrchestrator`.

Ordered after SqliteLifespan + LanceDBLifespan: the orchestrator
depends on both stores being ready before its watcher / scanner /
worker tasks can take the first row.

Construction reads the live :class:`Settings` to build the embedding +
tokenizer providers. If either is misconfigured the lifespan fails
fast — the daemon would be useless without them anyway.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI

from everos.component.embedding import build_embedding_provider
from everos.component.tokenizer import build_tokenizer
from everos.config import load_settings
from everos.core.lifespan import LifespanProvider
from everos.core.observability.logging import get_logger
from everos.core.persistence import MemoryRoot
from everos.memory.cascade import CascadeOrchestrator

logger = get_logger(__name__)


class CascadeLifespanProvider(LifespanProvider):
    """Manage the cascade subsystem for the app lifecycle."""

    def __init__(self, order: int = 12) -> None:
        super().__init__(name="cascade", order=order)
        self._orchestrator: CascadeOrchestrator | None = None

    async def startup(self, app: FastAPI) -> Any:
        settings = load_settings()
        memory_root = MemoryRoot.default()
        memory_root.ensure()

        embedder = build_embedding_provider(settings.embedding)
        tokenizer = build_tokenizer()
        self._orchestrator = CascadeOrchestrator(
            memory_root=memory_root,
            embedder=embedder,
            tokenizer=tokenizer,
        )
        await self._orchestrator.start()
        logger.info("cascade_lifespan_ready")
        return self._orchestrator

    async def shutdown(self, app: FastAPI) -> None:
        if self._orchestrator is not None:
            await self._orchestrator.stop()
            self._orchestrator = None

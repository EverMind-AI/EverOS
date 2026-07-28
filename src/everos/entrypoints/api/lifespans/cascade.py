"""Cascade lifespan provider — starts/stops :class:`CascadeOrchestrator`.

Ordered after SqliteLifespan + LanceDBLifespan: the orchestrator
depends on both stores being ready before its watcher / scanner /
worker tasks can take the first row.

Construction reads the live :class:`Settings` to build the tokenizer
provider, which fails fast if misconfigured. Embedding is a soft
dependency: startup warms the process-wide
:class:`~everos.component.embedding.EmbeddingCapability` singleton via
:func:`get_embedding_capability`, which never raises — the daemon
runs in keyword-only mode when embedding is unavailable.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI

from everos.component.embedding import get_embedding_capability
from everos.component.tokenizer import build_tokenizer
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
        memory_root = MemoryRoot.resolve()
        memory_root.ensure()

        tokenizer = build_tokenizer()

        capability = get_embedding_capability()
        if capability.available:
            logger.info("cascade_startup_embed_available")
        else:
            logger.info(
                "cascade_startup_embed_unavailable",
                reason="embedding not configured; keyword-only mode",
            )

        self._orchestrator = CascadeOrchestrator(
            memory_root=memory_root,
            tokenizer=tokenizer,
        )
        await self._orchestrator.start()
        logger.info("cascade_lifespan_ready")
        return self._orchestrator

    async def shutdown(self, app: FastAPI) -> None:
        if self._orchestrator is not None:
            await self._orchestrator.stop()
            self._orchestrator = None

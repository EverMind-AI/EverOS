"""Lifespan composition factory.

Builds a FastAPI lifespan context manager from an explicit list of
LifespanProvider instances.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager

from fastapi import FastAPI

from everos.core.observability.logging import get_logger

from .base import LifespanProvider

logger = get_logger(__name__)


def build_lifespan(
    providers: Sequence[LifespanProvider],
) -> Callable[[FastAPI], AsyncIterator[None]]:
    """Compose providers into a FastAPI lifespan context manager.

    Providers are run in ``order`` ascending on startup and reverse on
    shutdown. A non-None return value from ``startup`` is stored under
    ``app.state.lifespan_data[provider.name]``.
    """
    sorted_providers = sorted(providers, key=lambda p: p.order)

    @asynccontextmanager
    async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
        lifespan_data: dict[str, object] = {}
        try:
            for provider in sorted_providers:
                logger.info(
                    "lifespan_provider_startup",
                    name=provider.name,
                    order=provider.order,
                )
                result = await provider.startup(app)
                if result is not None:
                    lifespan_data[provider.name] = result
            app.state.lifespan_data = lifespan_data
            yield
        finally:
            for provider in reversed(sorted_providers):
                try:
                    logger.info("lifespan_provider_shutdown", name=provider.name)
                    await provider.shutdown(app)
                except Exception:
                    logger.exception(
                        "lifespan_provider_shutdown_failed", name=provider.name
                    )

    return _lifespan

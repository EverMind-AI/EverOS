"""Lifespan provider abstract base.

A LifespanProvider is one unit of startup / shutdown work invoked by the
FastAPI lifespan factory. Providers are registered explicitly (no DI
auto-discovery) and executed in ``order`` ascending on startup, reverse
on shutdown.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from fastapi import FastAPI


class LifespanProvider(ABC):
    """One unit of startup / shutdown work."""

    def __init__(self, name: str, order: int = 0) -> None:
        self.name = name
        self.order = order

    @abstractmethod
    async def startup(self, app: FastAPI) -> Any:
        """Startup hook; return value is stored on ``app.state.lifespan_data[name]``."""

    @abstractmethod
    async def shutdown(self, app: FastAPI) -> None:
        """Shutdown hook; called in reverse order during application teardown."""

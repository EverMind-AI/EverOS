"""
FastAPI lifespan interface definition

Simple lifespan management interface that supports ordering and name field definition
"""

from abc import ABC, abstractmethod
from fastapi import FastAPI
from typing import Any

from core.observation.logger import get_logger

logger = get_logger(__name__)


class LifespanProvider(ABC):
    """Lifespan provider interface"""

    def __init__(self, name: str, order: int = 0):
        """
        Initialize lifespan provider

        Args:
            name (str): Provider name
            order (int): Execution order, smaller numbers execute first
        """
        self.name = name
        self.order = order

    @abstractmethod
    async def startup(self, app: FastAPI) -> Any:
        """Startup logic"""
        ...

    @abstractmethod
    async def shutdown(self, app: FastAPI) -> None:
        """Shutdown logic"""
        ...

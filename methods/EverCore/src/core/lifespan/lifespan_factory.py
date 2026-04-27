"""
Lifecycle factory

Provides factory methods for dynamically obtaining and creating lifecycles.
"""

import os
from typing import List
from abc import abstractmethod, ABC
from core.di.utils import get_beans_by_type, get_bean
from core.di.decorators import component
from .lifespan_interface import LifespanProvider
from core.observation.logger import get_logger
from contextlib import asynccontextmanager
from fastapi import FastAPI

logger = get_logger(__name__)


class AppReadyListener(ABC):
    """
    Application ready listener protocol

    Components implementing this protocol will be called after all lifespan providers have started.
    This is a decoupled hook mechanism that automatically discovers and invokes all listeners via the DI container.

    Usage:
        1. Create a class implementing the on_app_ready() method
        2. Register it into the DI container using the @component decorator
        3. Lifespan will automatically discover and invoke it

    Example:
        >>> from core.di.decorators import component
        >>> from core.lifespan.lifespan_factory import AppReadyListener
        >>>
        >>> @component(name="my_app_ready_listener")
        >>> class MyAppReadyListener(AppReadyListener):
        ...     def on_app_ready(self) -> None:
        ...         print("Application is ready, executing my logic")
    """

    @abstractmethod
    def on_app_ready(self) -> None:
        """Called when the application is ready"""
        ...


def create_lifespan_with_providers(providers: list[LifespanProvider]):
    """
    Create a lifecycle manager containing multiple providers

    Args:
        providers (list[LifespanProvider]): List of lifecycle providers

    Returns:
        callable: FastAPI lifecycle context manager
    """
    # Sort by order
    sorted_providers = sorted(providers, key=lambda x: x.order)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """FastAPI lifecycle context manager"""
        lifespan_data = {}

        try:
            # Start all providers
            for provider in sorted_providers:
                logger.info(
                    "Starting lifecycle provider: %s (order=%d)",
                    provider.name,
                    provider.order,
                )
                result = await provider.startup(app)
                if result is not None:
                    lifespan_data[provider.name] = result
                logger.info("Lifecycle provider started: %s", provider.name)

            # Store data in app.state for easy access
            app.state.lifespan_data = lifespan_data

            # Get all application ready listeners via DI and invoke them (decoupled design)
            listeners = get_beans_by_type(AppReadyListener)
            for listener in listeners:
                try:
                    listener.on_app_ready()
                except Exception as e:
                    logger.error(
                        "Application ready listener execution failed: %s - %s",
                        type(listener).__name__,
                        e,
                    )

            yield  # During application runtime

        finally:
            # Shut down all providers in reverse order
            for provider in reversed(sorted_providers):
                try:
                    logger.info("Shutting down lifecycle provider: %s", provider.name)
                    await provider.shutdown(app)
                    logger.info(
                        "Lifecycle provider shutdown completed: %s", provider.name
                    )
                except Exception as e:
                    logger.error(
                        "Failed to shut down lifecycle provider: %s - %s",
                        provider.name,
                        str(e),
                    )

    return lifespan


@component(name="lifespan_factory")
class LifespanFactory:
    """Lifecycle factory"""

    def create_auto_lifespan(self):
        """
        Automatically create a lifecycle containing all registered providers.

        Supports disabling specific providers via the LIFESPAN_DISABLED environment variable.
        Example: LIFESPAN_DISABLED=milvus,elasticsearch

        Returns:
            callable: FastAPI lifecycle context manager
        """
        providers = get_beans_by_type(LifespanProvider)

        # Filter out disabled providers
        disabled_raw = os.getenv("LIFESPAN_DISABLED", "").strip()
        if disabled_raw:
            disabled_names = {n.strip() for n in disabled_raw.split(",") if n.strip()}
            before_count = len(providers)
            providers = [p for p in providers if p.name not in disabled_names]
            skipped = before_count - len(providers)
            if skipped:
                logger.info(
                    "LIFESPAN_DISABLED=%s — skipped %d provider(s)",
                    disabled_raw,
                    skipped,
                )

        # Sort by order
        sorted_providers = sorted(providers, key=lambda x: x.order)
        return create_lifespan_with_providers(sorted_providers)

    def create_lifespan_with_names(self, provider_names: List[str]):
        """
        Create a lifecycle based on provider names

        Args:
            provider_names (List[str]): List of provider names

        Returns:
            callable: FastAPI lifecycle context manager
        """
        providers = []
        for name in provider_names:
            provider = get_bean(name)
            if isinstance(provider, LifespanProvider):
                providers.append(provider)

        # Sort by order
        sorted_providers = sorted(providers, key=lambda x: x.order)
        return create_lifespan_with_providers(sorted_providers)

    def create_lifespan_with_orders(self, orders: List[int]):
        """
        Create a lifecycle based on order values

        Args:
            orders (List[int]): List of order values

        Returns:
            callable: FastAPI lifecycle context manager
        """
        all_providers = get_beans_by_type(LifespanProvider)
        filtered_providers = [p for p in all_providers if p.order in orders]

        # Sort by order
        sorted_providers = sorted(filtered_providers, key=lambda x: x.order)
        return create_lifespan_with_providers(sorted_providers)

    def list_available_providers(self) -> List[LifespanProvider]:
        """
        List all available lifecycle providers

        Returns:
            List[LifespanProvider]: List of providers (sorted by order)
        """
        providers = get_beans_by_type(LifespanProvider)
        return sorted(providers, key=lambda x: x.order)

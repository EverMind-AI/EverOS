"""
Business lifecycle provider implementation
"""

from fastapi import FastAPI
from typing import Dict, Any

from core.observation.logger import get_logger
from core.di.utils import get_bean_by_type, get_beans_by_type, get_bean
from core.di.decorators import component
from core.interface.controller.base_controller import BaseController
from core.capability.app_capability import ApplicationCapability
from core.component.llm.tokenizer.tokenizer_factory import TokenizerFactory
from .lifespan_interface import LifespanProvider

logger = get_logger(__name__)


@component(name="business_lifespan_provider")
class BusinessLifespanProvider(LifespanProvider):
    """Business lifecycle provider"""

    def __init__(self, name: str = "business", order: int = 20):
        """
        Initialize business lifecycle provider

        Args:
            name (str): Provider name
            order (int): Execution order, business logic usually starts after database
        """
        super().__init__(name, order)

    async def startup(self, app: FastAPI) -> Dict[str, Any]:
        """
        Start business logic

        Args:
            app (FastAPI): FastAPI application instance

        Returns:
            Dict[str, Any]: Business initialization information
        """
        logger.info("Initializing business logic...")

        # 0. Preload tokenizers to avoid blocking requests
        tokenizer_factory: TokenizerFactory = get_bean_by_type(TokenizerFactory)
        tokenizer_factory.load_default_encodings()

        # 1. Create business graph structure
        graphs = self._register_graphs(app)

        # 2. Register controllers
        controllers = self._register_controllers(app)

        # 3. Register capabilities
        capabilities = self._register_capabilities(app)

        logger.info("Business application initialization completed")

        return {
            'graphs': graphs,
            'controllers': controllers,
            'capabilities': capabilities,
        }

    async def shutdown(self, app: FastAPI) -> None:
        """
        Shutdown business logic

        Args:
            app (FastAPI): FastAPI application instance
        """
        logger.info("Shutting down business logic...")

        # Clean up business-related attributes in app.state
        if hasattr(app.state, 'graphs'):
            delattr(app.state, 'graphs')

        logger.info("Business application shutdown completed")

    def _register_controllers(self, app: FastAPI) -> list:
        """Register all controllers"""
        all_controllers = get_beans_by_type(BaseController)
        for controller in all_controllers:
            controller.register_to_app(app)
        logger.info(
            "Controller registration completed, %d controllers registered",
            len(all_controllers),
        )
        return all_controllers

    def _register_capabilities(self, app: FastAPI) -> list:
        """Register all application capabilities"""
        capability_beans = get_beans_by_type(ApplicationCapability)
        for capability in capability_beans:
            capability.enable(app)
        logger.info(
            "Application capability registration completed, %d capabilities registered",
            len(capability_beans),
        )
        return capability_beans

    def _create_graphs(self, checkpointer=None) -> dict:
        """Create all business graph structures"""
        logger.info("Creating business graph structures...")
        graphs = {}
        # Business graph structures can be created based on specific requirements here
        logger.info("Business graph structures created, %d graphs created", len(graphs))
        return graphs

    def _register_graphs(self, app: FastAPI) -> dict:
        """Register all graph structures to FastAPI application"""
        checkpointer = getattr(app.state, 'checkpointer', None)
        if not checkpointer:
            logger.warning("Checkpointer not found, skipping graph structure creation")
            return {}

        graphs = self._create_graphs(checkpointer)
        app.state.graphs = graphs
        logger.info("Graph structures registered, %d graphs registered", len(graphs))
        return graphs

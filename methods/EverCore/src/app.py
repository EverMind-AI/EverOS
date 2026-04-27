"""
Application module

Contains business-specific logic such as controller registration, graph structure creation, capability loading, etc.
"""

from fastapi import FastAPI
from core.di.utils import get_beans_by_type, get_bean_by_type
from core.capability.app_capability import ApplicationCapability
from core.observation.logger import get_logger
from core.interface.controller.base_controller import BaseController
from core.middleware.user_context_middleware import UserContextMiddleware
from core.middleware.app_logic_middleware import AppLogicMiddleware
from core.middleware.prometheus_middleware import PrometheusMiddleware
from fastapi.middleware import Middleware

from base_app import create_base_app
from core.lifespan.lifespan_factory import LifespanFactory


# Recommended usage: obtain logger once at the module top level, then use directly (high performance)
logger = get_logger(__name__)


def register_controllers(fastapi_app: FastAPI):
    """
    Register all controllers to the FastAPI application.

    Args:
        fastapi_app (FastAPI): FastAPI application instance
    """
    all_controllers = get_beans_by_type(BaseController)
    for controller in all_controllers:
        controller.register_to_app(fastapi_app)
    logger.info(
        "Controller registration completed, %d controllers registered",
        len(all_controllers),
    )


def create_graphs(checkpointer):
    """
    Create all business graph structures.

    Args:
        checkpointer: Checkpointer

    Returns:
        dict: Dictionary containing all graph structures
    """
    logger.info("Creating business graph structures...")

    graphs = {}

    logger.info("Business graph structures created, %d graphs created", len(graphs))
    return graphs


def register_capabilities(fastapi_app: FastAPI):
    """
    Register all application capabilities.

    Args:
        fastapi_app (FastAPI): FastAPI application instance
    """
    capability_beans = get_beans_by_type(ApplicationCapability)
    for capability in capability_beans:
        capability.enable(fastapi_app)
    logger.info(
        "Application capabilities registered, %d capabilities registered",
        len(capability_beans),
    )


def register_graphs(fastapi_app: FastAPI):
    """
    Register all graph structures to the FastAPI application.

    Args:
        fastapi_app (FastAPI): FastAPI application instance
    """
    checkpointer = fastapi_app.state.checkpointer
    graphs = create_graphs(checkpointer)
    fastapi_app.state.graphs = graphs
    logger.info("Graph structures registered, %d graphs registered", len(graphs))


# Note: create_business_lifespan is now imported from core.lifespan
# The original registration functions are kept here for use by new business components


def create_business_app(
    cors_origins=None,
    cors_allow_credentials=True,
    cors_allow_methods=None,
    cors_allow_headers=None,
):
    """
    Create a complete application with business logic.

    Args:
        cors_origins (list[str], optional): List of allowed CORS origins
        cors_allow_credentials (bool): Whether to allow credentials
        cors_allow_methods (list[str], optional): Allowed HTTP methods
        cors_allow_headers (list[str], optional): Allowed HTTP headers

    Returns:
        FastAPI: Configured FastAPI application instance
    """
    # Use DI to get lifespan factory, automatically creating a lifespan with all providers
    lifespan_factory = get_bean_by_type(LifespanFactory)
    combined_lifespan = lifespan_factory.create_auto_lifespan()

    # Create base app with combined lifespan manager
    fastapi_app = create_base_app(
        cors_origins=cors_origins,
        cors_allow_credentials=cors_allow_credentials,
        cors_allow_methods=cors_allow_methods,
        cors_allow_headers=cors_allow_headers,
        lifespan_context=combined_lifespan,
    )

    # Add business-related middleware
    fastapi_app.user_middleware.append(Middleware(AppLogicMiddleware))
    # Not directly interfacing with users
    # fastapi_app.user_middleware.append(Middleware(UserContextMiddleware))
    
    # Add Prometheus HTTP metrics middleware
    fastapi_app.user_middleware.append(Middleware(PrometheusMiddleware))

    return fastapi_app


# Create default business application instance
app = create_business_app()

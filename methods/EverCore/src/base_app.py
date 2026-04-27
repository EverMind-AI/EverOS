"""
Base application module

Contains business-agnostic FastAPI base configurations such as CORS, middleware, lifecycle management, etc.
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware

from core.observation.logger import get_logger
from core.middleware.database_session_middleware import DatabaseSessionMiddleware
from core.middleware.global_exception_handler import global_exception_handler
from core.middleware.profile_middleware import ProfileMiddleware
from core.di.utils import get_bean_by_type
from core.component.database_connection_provider import DatabaseConnectionProvider

from core.lifespan.lifespan_factory import LifespanFactory

# Recommended usage: obtain logger once at the module top, then use directly (high performance)
logger = get_logger(__name__)


def create_base_app(
    cors_origins=None,
    cors_allow_credentials=True,
    cors_allow_methods=None,
    cors_allow_headers=None,
    lifespan_context=None,
):
    """
    Create a base FastAPI application

    Args:
        cors_origins (list[str], optional): List of allowed CORS origins, default is ["*"]
        cors_allow_credentials (bool): Whether to allow credentials, default is True
        cors_allow_methods (list[str], optional): Allowed HTTP methods, default is ["*"]
        cors_allow_headers (list[str], optional): Allowed HTTP headers, default is ["*"]
        lifespan_context (callable, optional): Lifecycle context manager, default uses database lifecycle

    Returns:
        FastAPI: Configured FastAPI application instance
    """
    # Use the provided lifespan context or default database lifecycle
    if lifespan_context is None:
        lifespan_factory = get_bean_by_type(LifespanFactory)
        lifespan_context = lifespan_factory.create_lifespan_with_names(
            ["database_lifespan_provider"]
        )

    # Control docs display based on environment variable
    # Only enable docs in development environment (ENV=dev)
    env = os.environ.get('ENV', 'prod').upper()
    enable_docs = env == 'DEV'

    # Create FastAPI application
    app = FastAPI(
        lifespan=lifespan_context,
        docs_url="/docs" if enable_docs else None,
        redoc_url="/redoc" if enable_docs else None,
        openapi_url="/openapi.json" if enable_docs else None,
    )

    if enable_docs:
        logger.info("FastAPI documentation enabled (ENV=%s)", env)
    else:
        logger.info("FastAPI documentation disabled (ENV=%s)", env)

    # Set default CORS values
    if cors_origins is None:
        cors_origins = ["*"]
    if cors_allow_methods is None:
        cors_allow_methods = ["*"]
    if cors_allow_headers is None:
        cors_allow_headers = ["*"]

    # Add HTTP exception handler, otherwise HTTPException won't be handled by global_exception_handler
    app.add_exception_handler(HTTPException, global_exception_handler)

    # Convert FastAPI/Pydantic 422 validation errors to our ErrorApiResponse format
    app.add_exception_handler(RequestValidationError, global_exception_handler)

    # Add global exception handler
    # Acts as a fallback outside middleware
    app.add_exception_handler(Exception, global_exception_handler)

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=cors_allow_credentials,
        allow_methods=cors_allow_methods,
        allow_headers=cors_allow_headers,
    )

    # Add basic middleware
    # The order of middleware matters: the earlier added, the later executed
    # app.add_middleware(DatabaseSessionMiddleware)

    # Add performance profiling middleware (add last, executes first)
    app.add_middleware(ProfileMiddleware)

    # Mount lifespan management methods to app instance
    _mount_lifespan_methods(app)

    return app


def _mount_lifespan_methods(app: FastAPI):
    """
    Mount lifespan management methods to the FastAPI application instance

    After mounting, you can directly use:
    - app.start_lifespan(): Start lifespan
    - app.exit_lifespan(): Exit lifespan

    Args:
        app (FastAPI): FastAPI application instance
    """
    # Store reference to lifespan manager
    app.lifespan_manager = None

    async def start_lifespan():
        """Start the application's lifespan context manager"""
        if app.lifespan_manager is not None:
            logger.warning("Lifespan already started, no need to start again")
            return app.lifespan_manager

        # Get lifespan context manager
        lifespan_context = app.router.lifespan_context

        if lifespan_context:
            # Create context manager instance
            lifespan_manager = lifespan_context(app)

            # Manually enter context (equivalent to starting)
            await lifespan_manager.__aenter__()

            # Store manager reference
            app.lifespan_manager = lifespan_manager

            logger.info("Application Lifespan startup completed")
            return lifespan_manager
        else:
            logger.warning("This application has no lifespan configured")
            return None

    async def exit_lifespan():
        """Exit the application's lifespan context manager"""
        if app.lifespan_manager is None:
            logger.warning("Lifespan not started or already exited")
            return

        try:
            # Manually exit context
            await app.lifespan_manager.__aexit__(None, None, None)
            logger.info("Application Lifespan exit completed")
        except (AttributeError, RuntimeError) as e:
            logger.error("Error occurred when exiting Lifespan: %s", str(e))
        finally:
            # Clean up reference
            app.lifespan_manager = None

    # Mount methods to app instance
    app.start_lifespan = start_lifespan
    app.exit_lifespan = exit_lifespan


async def manually_start_lifespan(app: FastAPI):
    """
    Manually start the lifespan context manager of a FastAPI application

    Note: It is recommended to use the convenient methods mounted on the app instance:
    - app.start_lifespan(): Start lifespan
    - app.exit_lifespan(): Exit lifespan

    This function is used to initialize the application lifecycle without starting an HTTP server,
    including database connections, business graph structures, etc. Suitable for scripts, tests,
    or other scenarios requiring application context but not HTTP services.

    Args:
        app (FastAPI): FastAPI application instance

    Returns:
        context_manager: Lifecycle context manager instance, can be used for manual exit

    Example:
        ```python
        from app import app

        # Recommended way: directly use mounted methods
        await app.start_lifespan()
        # Perform operations requiring application context
        # ...
        await app.exit_lifespan()

        # Or use traditional way
        from base_app import manually_start_lifespan
        lifespan_manager = await manually_start_lifespan(app)
        # await lifespan_manager.__aexit__(None, None, None)
        ```
    """
    # Directly call the mounted method
    return await app.start_lifespan()


async def close_database_connection():
    """Close the database connection pool"""
    try:
        db_provider = get_bean_by_type(DatabaseConnectionProvider)
        await db_provider.close()
    except (AttributeError, RuntimeError) as e:
        logger.error("Error occurred when closing database connection: %s", str(e))

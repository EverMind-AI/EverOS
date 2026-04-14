#!/usr/bin/env python3
"""
Memsys Backend Management Script
Provides command-line tools to manage the backend application
"""

import asyncio
from IPython.terminal.embed import embed
from functools import wraps
from typing import Callable
import nest_asyncio

nest_asyncio.apply()

import typer
from typer import Typer


# Create Typer application
cli = Typer(help="Memsys Backend Management Tool")

# Global variable to store application state
_app_state = None
_initialized = False


def setup_environment_and_app(env_file: str = ".env"):
    """
    Set up environment and application

    Args:
        env_file: Environment variable file name
    """
    global _initialized
    if _initialized:
        return

    # Add src directory to Python path
    from import_parent_dir import add_parent_path

    add_parent_path(0)

    # Load environment variables
    from common_utils.load_env import setup_environment

    setup_environment(load_env_file_name=env_file, check_env_var="MONGODB_HOST")

    from application_startup import setup_all

    setup_all()
    _initialized = True


def with_app_context(func: Callable) -> Callable:
    """
    Decorator: Provide FastAPI application context for commands

    Args:
        func: The decorated asynchronous function

    Returns:
        The decorated function
    """

    @wraps(func)
    async def wrapper(*args, **kwargs):
        global _app_state

        from app import app

        # Create application context
        async with app.router.lifespan_context(app):
            # Set application state
            _app_state = app.state
            try:
                # Execute the decorated function
                result = await func(*args, **kwargs)
                return result
            finally:
                # Clean up application state
                _app_state = None

    return wrapper


def with_full_context_decorator(func: Callable) -> Callable:
    """
    Decorator: Use ContextManager.run_with_full_context to provide full context

    Args:
        func: The decorated asynchronous function

    Returns:
        The decorated function
    """

    @wraps(func)
    async def wrapper(*args, **kwargs):
        global _app_state

        from app import app
        from core.di.utils import get_bean_by_type
        from core.context.context_manager import ContextManager

        # Create application context
        async with app.router.lifespan_context(app):
            # Set application state
            _app_state = app.state
            try:
                # Get ContextManager instance
                context_manager = get_bean_by_type(ContextManager)

                # Execute function using run_with_full_context
                result = await context_manager.run_with_full_context(
                    func, *args, auto_commit=True, auto_inherit_user=True, **kwargs
                )
                return result
            finally:
                # Clean up application state
                _app_state = None

    return wrapper


def is_cli_command(func: Callable) -> Callable:
    """
    Decorator: Mark CLI command functions

    Args:
        func: The decorated function

    Returns:
        The decorated function
    """
    func._is_cli_command = True
    return func


@cli.command()
def shell(
    debug: bool = typer.Option(False, "--debug", help="Enable debug mode"),
    env_file: str = typer.Option(
        ".env", "--env-file", help="Specify the environment variable file to load"
    ),
):
    """
    Start an interactive shell with access to application context
    """
    setup_environment_and_app(env_file)

    from core.observation.logger import get_logger

    logger = get_logger(__name__)

    if debug:
        logger.info("Debug mode enabled")

    logger.info("Using environment file: %s", env_file)

    banner = """
    ========================================
    Memsys Backend Shell
    
    Available variables:
    - app: FastAPI application instance
    - app_state: Application state (if available)
    - graphs: LangGraph instances (if available)
    - logger: Logger instance
    
    Example usage:
    >>> logger.info("Hello from shell!")
    >>> app.routes  # View all routes
    >>> graphs  # View available graph instances
    ========================================
    """

    def shell_runner():
        embed(header=banner)

    func = with_app_context(with_full_context_decorator(shell_runner))
    asyncio.run(func())


@cli.command()
def list_commands(
    show_all: bool = typer.Option(False, "--all", help="Show all commands"),
    env_file: str = typer.Option(
        ".env", "--env-file", help="Specify the environment variable file to load"
    ),
):
    """
    List all available CLI commands
    """

    if show_all:
        # Show all commands including hidden ones
        commands = cli.registered_commands
    else:
        # Show only visible commands
        commands = [cmd for cmd in cli.registered_commands if not cmd.hidden]

    typer.echo("Available commands:")
    for cmd in commands:
        help_text = cmd.help if cmd.help else "No description"
        typer.echo(f"  {cmd.name:<20} {help_text}"),

    typer.echo(f"\nUsing environment file: {env_file}")


@cli.command()
def tenant_init(
    env_file: str = typer.Option(
        ".env", "--env-file", help="Specify the environment variable file to load"
    )
):
    """
    Initialize MongoDB and Milvus databases for a specific tenant

    Tenant context is specified via environment variable TENANT_INIT_STORAGE_INFO (JSON).

    Examples:
        # Set tenant context
        export TENANT_INIT_STORAGE_INFO='{"tenant_id":"s0001","isolation_mode":"shared","storage_info":{"mongodb":{"database":"s0001_memsys"},"elasticsearch":{"index_prefix":"s0001"},"milvus":{"collection_prefix":"s0001","num_partitions":256}}}'  #skip-sensitive-check

        # Run initialization
        python src/manage.py tenant-init

        # Use custom environment file
        python src/manage.py tenant-init --env-file .env.production
    """

    # First set up environment and application
    setup_environment_and_app(env_file)

    from core.observation.logger import get_logger

    logger = get_logger(__name__)

    # Import tenant initialization module
    from core.tenants.init_tenant_all import run_tenant_init

    try:
        # Execute tenant initialization (read tenant ID from environment variable)
        success = asyncio.run(run_tenant_init())

        # Set exit code based on result
        if success:
            logger.info("✅ All database initializations succeeded")
            raise typer.Exit(0)
        else:
            logger.error("❌ Partial or complete database initialization failed")
            raise typer.Exit(1)
    except ValueError as e:
        # Catch error when tenant ID is not set
        logger.error("❌ Error: %s", str(e))
        raise typer.Exit(1)


if __name__ == '__main__':
    # Run CLI
    cli()

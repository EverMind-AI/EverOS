#!/usr/bin/env python3
"""
Memsys Main Application - Main application startup script

Main business application of the Memsys memory system, including:
- Requirement extraction agent
- Outline generation and editing agent
- Full-text writing and editing agent
- Document management and resource processing services
"""
import argparse
import os
import sys
import uvicorn
import logging

# Environment variables are not loaded yet, so cannot use get_logger
logger = logging.getLogger(__name__)

# Application info
APP_NAME = "Memory System"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = "Main application of the memory system"


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description=f"Start {APP_NAME} service")
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Server listening host address (env: MEMSYS_HOST, default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Server listening port (env: MEMSYS_PORT, default: 1995)",
    )
    parser.add_argument(
        "--env-file",
        type=str,
        default=".env",
        help="Specify the environment variable file to load (default: .env)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Enable Mock mode (for testing and development)",
    )
    parser.add_argument(
        "--longjob",
        type=str,
        help="Start specified long-running job consumer (e.g.: kafka_consumer)",
    )
    parser.add_argument(
        "--skip-migrations",
        action="store_true",
        help="Skip MongoDB database migrations on startup",
    )
    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_args()

    if args.longjob:
        service_name = "longjob_" + args.longjob
    else:
        service_name = "web"

    # Add src directory to Python path
    from import_parent_dir import add_parent_path

    add_parent_path(0)

    # Use unified environment loading utility
    from common_utils.load_env import setup_environment

    # Set up environment (Python path and .env file)
    setup_environment(
        load_env_file_name=args.env_file,
        check_env_var="MONGODB_HOST",
        service_name=service_name,
    )

    # Determine host and port: CLI args > env vars > defaults
    if args.host is not None:
        host = args.host
    elif os.getenv("MEMSYS_HOST"):
        host = os.getenv("MEMSYS_HOST")
    else:
        host = "0.0.0.0"

    if args.port is not None:
        port = args.port
    elif os.getenv("MEMSYS_PORT"):
        port = int(os.getenv("MEMSYS_PORT"))
    else:
        port = 1995

    # Check if Mock mode is enabled: prioritize command line argument, then environment variable
    from core.di.utils import enable_mock_mode

    if args.mock or (
        os.getenv("MOCK_MODE") and os.getenv("MOCK_MODE").lower() == "true"
    ):
        enable_mock_mode()
        logger.info("🚀 Enabled Mock mode")
    else:
        logger.info("🚀 Disabled Mock mode")

    # Display application startup information
    logger.info("🚀 Starting %s v%s", APP_NAME, APP_VERSION)
    logger.info("📝 %s", APP_DESCRIPTION)
    logger.info("🌟 Startup parameters:")
    logger.info("  📡 Host: %s", host)
    logger.info("  🔌 Port: %s", port)
    logger.info("  📄 Env File: %s", args.env_file)
    logger.info("  🎭 Mock Mode: %s", args.mock)
    logger.info("  🔧 LongJob Mode: %s", args.longjob if args.longjob else "Disabled")
    logger.info("  🔄 Skip Migrations: %s", args.skip_migrations)

    # Execute dependency injection and async task setup
    from application_startup import setup_all

    # Perform dependency injection and async task setup during module loading
    setup_all()

    # Run MongoDB database migrations (can be skipped via --skip-migrations argument)
    from core.oxm.mongo.migration.manager import MigrationManager

    MigrationManager.run_migrations_on_startup(enabled=not args.skip_migrations)

    # Check if in LongJob mode
    if args.longjob:
        logger.info("🔧 Starting LongJob mode: %s", args.longjob)
        os.environ["LONGJOB_NAME"] = args.longjob

    from app import app

    # Attach application info to the FastAPI app
    app.title = APP_NAME
    app.version = APP_VERSION
    app.description = APP_DESCRIPTION

    # Start service using command line arguments
    try:
        uvicorn_kwargs = {"host": host, "port": port}
        uvicorn.run(app, **uvicorn_kwargs)
    except KeyboardInterrupt:
        logger.info("👋 %s stopped", APP_NAME)
    except (OSError, RuntimeError) as e:
        logger.error("❌ %s failed to start: %s", APP_NAME, e)
        sys.exit(1)


if __name__ == "__main__":
    main()

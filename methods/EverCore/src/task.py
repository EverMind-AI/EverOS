#!/usr/bin/env python3
"""
Task Worker - Async task processor startup script

Async task processing service, responsible for:
- Background task queue processing
- Long-running asynchronous tasks
- Scheduled and delayed tasks
- Task status management and monitoring

Usage:
    arq task.WorkerSettings

Environment variables:
    REDIS_HOST: Redis server address (default: localhost)
    REDIS_PORT: Redis port (default: 6379)
    REDIS_DB: Redis database number (default: 0)
    REDIS_PASSWORD: Redis password (optional)
    REDIS_SSL: Whether to use SSL (default: false)
    REDIS_USERNAME: Redis username (optional)
"""

import os
import logging

from arq.connections import RedisSettings

# Application info
APP_NAME = "Async Task Worker"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = "Asynchronous task processing service"

# Environment variables are not loaded yet, so cannot use get_logger
logger = logging.getLogger(__name__)

# Add src directory to Python path
from import_parent_dir import add_parent_path

add_parent_path(0)

# Use unified environment loading tool
# Set .env file
from common_utils.load_env import setup_environment

setup_environment(check_env_var="REDIS_HOST")

# Display application startup info
logger.info("🚀 Starting %s v%s", APP_NAME, APP_VERSION)
logger.info("⚙️ %s", APP_DESCRIPTION)

# Run main function
# Scan component & task
from application_startup import setup_all

setup_all()


# Worker startup and shutdown callback functions
async def startup(_ctx):
    """Callback function when worker starts"""
    logger.info("🔄 Initializing async task worker...")

    # Initialize application context when worker starts
    from app import app

    # Add application info to FastAPI app (must be before start_lifespan)
    app.title = APP_NAME
    app.version = APP_VERSION
    app.description = APP_DESCRIPTION

    if hasattr(app, "start_lifespan"):
        await app.start_lifespan()
        logger.info("✅ Application lifespan startup completed")
    else:
        logger.warning("⚠️ app instance has no start_lifespan method")

    logger.info("🎯 %s started, ready to process tasks", APP_NAME)


async def shutdown(_ctx):
    """Callback function when worker shuts down"""
    logger.info("🛑 Shutting down %s...", APP_NAME)

    # Clean up application context when worker shuts down
    from app import app

    if hasattr(app, "exit_lifespan"):
        await app.exit_lifespan()
        logger.info("✅ Application lifespan shutdown completed")
    else:
        logger.warning("⚠️ app instance has no exit_lifespan method")

    logger.info("👋 %s has stopped", APP_NAME)


from core.asynctasks.task_manager import get_task_manager


class WorkerSettings:
    functions = get_task_manager().get_worker_functions()
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = RedisSettings(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", "6379")),
        database=int(os.getenv("REDIS_DB", "0")),
        password=os.getenv("REDIS_PASSWORD", "123456"),
        ssl=os.getenv("REDIS_SSL", "false").lower() == "true",
        username=os.getenv("REDIS_USERNAME"),
    )
    health_check_interval = 30
    max_jobs = 10
    job_timeout = 300
    keep_result = 3600


#  arq task.WorkerSettings

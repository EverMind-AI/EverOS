# -*- coding: utf-8 -*-
"""
Async task setup module

Handles the entry function for loading async task scan paths from addons and registering tasks
"""

from core.asynctasks.task_scan_registry import TaskScanDirectoriesRegistry
from core.di.utils import get_bean_by_type
from core.observation.logger import get_logger
from core.addons.addons_registry import ADDONS_REGISTRY

logger = get_logger(__name__)


def setup_async_tasks(addons: list = None):
    """
    Set up async tasks

    Extract async task scan directories from the addon list and perform task scanning and registration

    Args:
        addons (list, optional): List of addons. If None, fetch from ADDONS_REGISTRY
    """
    logger.info("🔄 Registering async tasks...")

    try:
        # Get task manager
        from core.asynctasks.task_manager import TaskManager

        task_manager = get_bean_by_type(TaskManager)

        # If addons not provided, get from ADDONS_REGISTRY
        if addons is None:
            addons = ADDONS_REGISTRY.get_all()

        logger.info("  📦 Loading async task scan paths from %d addons...", len(addons))

        # Create task directory registry and populate from addons
        task_directories_registry = TaskScanDirectoriesRegistry()
        for addon in addons:
            if addon.has_asynctasks():
                addon_dirs = addon.asynctasks.get_scan_directories()
                for directory in addon_dirs:
                    task_directories_registry.add_scan_path(directory)
                logger.debug(
                    "  📌 addon [%s] contributes %d task directories",
                    addon.name,
                    len(addon_dirs),
                )

        task_directories = task_directories_registry.get_scan_directories()
        logger.info("📂 Number of task directories: %d", len(task_directories))
        for directory in task_directories:
            logger.debug("  + %s", directory)

        # Automatically scan and register tasks
        task_manager.scan_and_register_tasks(task_directories_registry)

        # Print registered tasks
        registered_tasks = task_manager.list_registered_task_names()
        logger.info("📋 Registered task list: %s", registered_tasks)

        logger.info("✅ Async task registration completed")
    except Exception as e:
        logger.error("❌ Async task registration failed: %s", e)
        raise


def print_registered_tasks():
    """Print registered async tasks"""
    logger.info("\n📋 Registered task list:")
    logger.info("-" * 50)

    from core.asynctasks.task_manager import TaskManager

    task_manager = get_bean_by_type(TaskManager)

    registered_tasks = task_manager.list_registered_task_names()
    logger.info("📋 Registered task list: %s", registered_tasks)

# -*- coding: utf-8 -*-
"""
Dependency injection setup module

Handles the entry function for loading dependency injection scan paths from addons
"""

from core.di.scanner import ComponentScanner
from core.di.utils import get_beans
from core.observation.logger import get_logger
from core.addons.addons_registry import ADDONS_REGISTRY

logger = get_logger(__name__)


def setup_dependency_injection(addons: list = None):
    """
    Set up the dependency injection framework

    Extract DI scan paths from the addon list and perform component scanning and registration

    Args:
        addons (list, optional): List of addons. If None, fetch from ADDONS_REGISTRY

    Returns:
        ComponentScanner: Configured component scanner
    """
    logger.info("🚀 Initializing dependency injection container...")

    # Import to trigger automatic replacement of Bean ordering strategy (executed during module loading)
    from core.addons.addonize import addon_bean_order_strategy  # noqa: F401

    # Create component scanner
    scanner = ComponentScanner()

    # If addons not provided, get from ADDONS_REGISTRY
    if addons is None:
        addons = ADDONS_REGISTRY.get_all()

    logger.info(
        "  📦 Loading dependency injection scan paths from %d addons...", len(addons)
    )

    # Collect all scan paths and register scan_context
    total_paths = 0
    for addon in addons:
        if addon.has_di():
            addon_paths = addon.di.get_scan_paths()
            logger.debug(
                "  📌 addon [%s] contributes %d scan paths",
                addon.name,
                len(addon_paths),
            )

            # Register scan_context for each scan path, marking addon_tag
            for path in addon_paths:
                # Register scan context, marking source addon
                scanner.register_scan_context(path, {"addon_tag": addon.name})
                # Add scan path
                scanner.add_scan_path(path)
                logger.debug("    + %s (addon_tag=%s)", path, addon.name)
                total_paths += 1

    logger.info(scanner.context_registry.print_tree())

    # Perform scanning and registration
    scanner.scan()
    logger.info(
        "✅ Dependency injection setup completed, scanned %d paths in total",
        total_paths,
    )

    return scanner


def print_registered_beans():
    """Print all registered Beans"""
    logger.info("\n📋 Registered Bean list:")
    logger.info("-" * 50)

    all_beans = get_beans()
    for name, bean in all_beans.items():
        logger.info("  • %s: %s", name, type(bean).__name__)

    logger.info("\n📊 Total: %d Beans", len(all_beans))

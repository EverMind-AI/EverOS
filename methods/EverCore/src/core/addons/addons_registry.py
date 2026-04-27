# -*- coding: utf-8 -*-
"""
Addons registry manager
Manages the list of all addon registries, providing a unified access interface
"""

import os
from typing import List, Optional, Set
from core.addons.addon_registry import AddonRegistry
from core.observation.logger import get_logger

logger = get_logger(__name__)


class AddonsRegistry:
    """
    Addons registry manager

    Manages the list of all addon registries, providing unified registration and access interfaces

    Usage example:
        from core.addons.addons_registry import ADDONS_REGISTRY

        # Create and register addon
        addon = AddonRegistry(name="my_addon")
        addon.register_di(di_registry).register_asynctasks(task_registry)

        ADDONS_REGISTRY.register(addon)

        # Get all addons
        all_addons = ADDONS_REGISTRY.get_all()

        # Find addon by name
        my_addon = ADDONS_REGISTRY.get_by_name("my_addon")

        # Automatically load from entry points (recommended)
        ADDONS_REGISTRY.load_entrypoints()

    Entry Points registration method:
        Configure in pyproject.toml:

        [project.entry-points."memsys.addons"]
        my_addon = "my_package.addon_module"

        Execute registration in the module:

        # my_package/addon_module.py
        from core.addons.addons_registry import ADDONS_REGISTRY

        my_addon = AddonRegistry(name="my_addon")
        # ... configure addon ...
        ADDONS_REGISTRY.register(my_addon)  # Automatically executed when module is imported
    """

    def __init__(self):
        """Initialize the addons registry manager"""
        self._addons: List[AddonRegistry] = []

    def register(self, addon: AddonRegistry) -> 'AddonsRegistry':
        """
        Register an addon

        Args:
            addon: addon registry instance

        Returns:
            self: supports method chaining
        """
        self._addons.append(addon)
        return self

    def get_all(self) -> List[AddonRegistry]:
        """
        Get all registered addons

        Returns:
            List of all addons
        """
        return self._addons.copy()

    def get_by_name(self, name: str) -> Optional[AddonRegistry]:
        """
        Get addon by name

        Args:
            name: addon name

        Returns:
            Found addon, or None if not exists
        """
        for addon in self._addons:
            if addon.name == name:
                return addon
        return None

    def clear(self) -> 'AddonsRegistry':
        """
        Clear all registered addons

        Returns:
            self: supports method chaining
        """
        self._addons.clear()
        return self

    def count(self) -> int:
        """
        Get the number of registered addons

        Returns:
            Number of addons
        """
        return len(self._addons)

    def _should_load_entrypoint(self, entrypoint_name: str) -> bool:
        """
        Determine whether to load the specified entrypoint based on environment variables

        Control which entrypoints to load via the MEMSYS_ENTRYPOINTS_FILTER environment variable
        Format: MEMSYS_ENTRYPOINTS_FILTER=ep1,ep2,ep3

        If the environment variable is not set or empty, load all entrypoints
        If the environment variable is set, only load entrypoints specified in the list

        Args:
            entrypoint_name: name of the entrypoint (ep.name)

        Returns:
            True means should load, False means should skip
        """
        filter_config = os.environ.get('MEMSYS_ENTRYPOINTS_FILTER', '').strip()

        # If environment variable is not set or empty, load all entrypoints
        if not filter_config:
            return True

        # Split by comma and filter
        allowed_entrypoints: Set[str] = {
            name.strip() for name in filter_config.split(',') if name.strip()
        }
        return entrypoint_name in allowed_entrypoints

    def load_entrypoints(self) -> 'AddonsRegistry':
        """
        Automatically load all registered addons from entry points

        Scan the 'memsys.addons' entry point group to automatically discover and load all addons registered via this mechanism.

        How it works:
        1. Scan all entry points under [project.entry-points."memsys.addons"] in pyproject.toml
        2. Filter entrypoints to be loaded based on MEMSYS_ENTRYPOINTS_FILTER environment variable (via ep.name)
        3. Load the module corresponding to each entry point (trigger module import)
        4. Module import will automatically execute module-level code, including ADDONS_REGISTRY.register(addon) calls
        5. All addons are automatically registered into the global ADDONS_REGISTRY

        Environment variable control:
        - MEMSYS_ENTRYPOINTS_FILTER: controls which entrypoints to load, format is a comma-separated list of entrypoint names
          Example: MEMSYS_ENTRYPOINTS_FILTER=ep1,ep2,ep3
          If not set or empty, load all entrypoints
          Note: One entrypoint may contain registrations of multiple addons

        Notes:
        - No need for entry points to return specific objects
        - Just ensure registration code is executed during module import
        - Avoid time-consuming operations in module-level code

        Returns:
            self: supports method chaining
        """
        try:
            # Python 3.10+ uses importlib.metadata
            from importlib.metadata import entry_points

            logger.info("🔌 Starting to load addons entry points...")

            # Get all entry points under memsys.addons group
            # Python 3.10+ uses select method, Python 3.9 uses dictionary access
            try:
                # Python 3.10+
                addon_eps = entry_points(group='memsys.addons')
            except TypeError:
                # Python 3.9 fallback
                eps = entry_points()
                if hasattr(eps, 'select'):
                    addon_eps = eps.select(group='memsys.addons')
                else:
                    # Direct dictionary access
                    addon_eps = (
                        eps.get('memsys.addons', []) if isinstance(eps, dict) else []
                    )

            for ep in addon_eps:
                try:
                    # Filter entrypoint based on environment variable
                    if not self._should_load_entrypoint(ep.name):
                        logger.info(
                            "  ⏭️  Skipping entrypoint: %s (not in MEMSYS_ENTRYPOINTS_FILTER)",
                            ep.name,
                        )
                        continue

                    # Load entry point, trigger module import and execution
                    # Module import will automatically execute registration code (e.g., ADDONS_REGISTRY.register(addon))
                    ep.load()
                    logger.info("  ✅ Loaded entrypoint: %s", ep.name)

                except Exception as e:  # pylint: disable=broad-except
                    logger.error("  ❌ Failed to load entrypoint %s: %s", ep.name, e)

            logger.info(
                "✅ Addons entry points loading completed, total %d", self.count()
            )

        except ImportError:
            logger.warning(
                "⚠️  importlib.metadata is not available, skipping entry points loading"
            )
        except Exception as e:  # pylint: disable=broad-except
            logger.error("❌ Failed to load addons entry points: %s", e)

        return self


# Global singleton instance
ADDONS_REGISTRY = AddonsRegistry()

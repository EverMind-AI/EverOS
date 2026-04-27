# -*- coding: utf-8 -*-
"""
Container for scanning and registering a single Addon
Used to host DI and async task registry for a single addon
"""

from typing import Optional
from core.di.scan_path_registry import ScannerPathsRegistry
from core.asynctasks.task_scan_registry import TaskScanDirectoriesRegistry


class AddonRegistry:
    """
    Container for scanning and registering a single Addon

    Used to host scanning configurations for a single addon:
    - di: Registry for DI component scan paths
    - asynctasks: Registry for async task scan directories

    Example:
        # Create addon registry
        addon = AddonRegistry(name="my_addon")

        # Register DI registry
        di_registry = ScannerPathsRegistry()
        di_registry.add_scan_path("/path/to/components")
        addon.register_di(di_registry)

        # Register async task registry
        task_registry = TaskScanDirectoriesRegistry()
        task_registry.add_scan_path("/path/to/tasks")
        addon.register_asynctasks(task_registry)
    """

    def __init__(self, name: str):
        """
        Initialize addon registry

        Args:
            name: addon name
        """
        self.name: str = name
        self.di: Optional[ScannerPathsRegistry] = None
        self.asynctasks: Optional[TaskScanDirectoriesRegistry] = None

    def register_di(self, registry: ScannerPathsRegistry) -> 'AddonRegistry':
        """
        Register DI component scan path registry

        Args:
            registry: DI scan path registry

        Returns:
            self: supports method chaining
        """
        self.di = registry
        return self

    def register_asynctasks(
        self, registry: TaskScanDirectoriesRegistry
    ) -> 'AddonRegistry':
        """
        Register async task scan directory registry

        Args:
            registry: async task scan directory registry

        Returns:
            self: supports method chaining
        """
        self.asynctasks = registry
        return self

    def has_di(self) -> bool:
        """Check if DI registry has been registered"""
        return self.di is not None

    def has_asynctasks(self) -> bool:
        """Check if async task registry has been registered"""
        return self.asynctasks is not None

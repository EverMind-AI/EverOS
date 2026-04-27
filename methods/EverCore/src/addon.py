import os

from core.di.scan_path_registry import ScannerPathsRegistry
from core.asynctasks.task_scan_registry import TaskScanDirectoriesRegistry
from common_utils.project_path import get_base_scan_path
from core.addons.addon_registry import AddonRegistry
from core.addons.addons_registry import ADDONS_REGISTRY

# Configure DI scan paths
paths_registry = ScannerPathsRegistry()

paths_registry.add_scan_path(
    os.path.join(get_base_scan_path(), "core/interface/controller/debug")
)
paths_registry.add_scan_path(os.path.join(get_base_scan_path(), "core/lifespan"))
paths_registry.add_scan_path(os.path.join(get_base_scan_path(), "core/lock"))
paths_registry.add_scan_path(os.path.join(get_base_scan_path(), "core/cache"))
paths_registry.add_scan_path(os.path.join(get_base_scan_path(), "core/tenants"))
paths_registry.add_scan_path(os.path.join(get_base_scan_path(), "core/events"))
paths_registry.add_scan_path(os.path.join(get_base_scan_path(), "core/context"))
paths_registry.add_scan_path(os.path.join(get_base_scan_path(), "core/request"))
paths_registry.add_scan_path(os.path.join(get_base_scan_path(), "core/component"))
paths_registry.add_scan_path(os.path.join(get_base_scan_path(), "infra_layer"))
paths_registry.add_scan_path(os.path.join(get_base_scan_path(), "agentic_layer"))
paths_registry.add_scan_path(os.path.join(get_base_scan_path(), "biz_layer"))

# Configure asynchronous task scan paths
task_directories_registry = TaskScanDirectoriesRegistry()

task_directories_registry.add_scan_path(
    os.path.join(get_base_scan_path(), "core/asynctasks/examples")
)
task_directories_registry.add_scan_path(
    os.path.join(get_base_scan_path(), "infra_layer/adapters/input/jobs")
)

# Create and register core addon
core_addon = AddonRegistry(name="core")
core_addon.register_di(paths_registry)
core_addon.register_asynctasks(task_directories_registry)

ADDONS_REGISTRY.register(core_addon)

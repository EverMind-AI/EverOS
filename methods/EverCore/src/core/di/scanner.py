# -*- coding: utf-8 -*-
"""
Component Scanner
"""

import os
import sys
import importlib
from pathlib import Path
from typing import List, Set, Optional, Dict, Any
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.observation.logger import get_logger
from core.di.scan_context import ScanContextRegistry, get_scan_context_registry


class ComponentScanner:
    """Component Scanner"""

    def __init__(self):
        self.scan_paths: List[str] = []
        self.scan_packages: List[str] = []
        # Using 'di' causes directories like audit to be filtered out, so full path matching is required
        self.exclude_paths: Set[str] = {
            '/di/',
            '/config/',
            '__pycache__',
            '.git',
            '.pytest_cache',
        }
        self.exclude_patterns: Set[str] = {'test_', '_test', 'tests'}
        self.include_patterns: Set[str] = set()
        self.recursive = True
        # self.parallel = True if os.getenv("ENV") == 'dev' else False
        self.parallel = False
        self.max_workers = 8

        # Create a dedicated logger
        self.logger = get_logger(__name__)

        # Scan context registry (using singleton)
        self.context_registry = get_scan_context_registry()

        # Key modules that need to be preloaded to avoid circular dependencies during parallel import
        self.preload_modules = [
            # SQLAlchemy core modules
            'sqlalchemy.engine',
            'sqlalchemy.engine.base',
            'sqlalchemy.engine.default',
            'sqlalchemy.pool',
            'sqlalchemy.sql',
            'sqlalchemy.sql.schema',
            'sqlalchemy.sql.sqltypes',
            'sqlalchemy.orm',
            'sqlalchemy.orm.session',
            'sqlalchemy.orm.query',
            # Pydantic core modules
            'pydantic',
            'pydantic.fields',
            'pydantic.main',
            'pydantic.validators',
            'pydantic.v1',
            'pydantic.v1.fields',
            'pydantic.v1.main',
            # Other modules that may cause circular dependencies
            'typing_extensions',
            'dataclasses',
        ]

    def add_scan_path(self, path: str) -> 'ComponentScanner':
        """Add scan path"""
        self.scan_paths.append(path)
        return self

    def add_scan_package(self, package: str) -> 'ComponentScanner':
        """Add scan package"""
        self.scan_packages.append(package)
        return self

    def exclude_path(self, path: str) -> 'ComponentScanner':
        """Exclude path"""
        self.exclude_paths.add(path)
        return self

    def exclude_pattern(self, pattern: str) -> 'ComponentScanner':
        """Exclude pattern"""
        self.exclude_patterns.add(pattern)
        return self

    def include_pattern(self, pattern: str) -> 'ComponentScanner':
        """Include pattern"""
        self.include_patterns.add(pattern)
        return self

    def set_recursive(self, recursive: bool) -> 'ComponentScanner':
        """Set whether to scan recursively"""
        self.recursive = recursive
        return self

    def set_parallel(self, parallel: bool) -> 'ComponentScanner':
        """Set whether to scan in parallel"""
        self.parallel = parallel
        return self

    def set_max_workers(self, max_workers: int) -> 'ComponentScanner':
        """Set maximum number of worker threads"""
        self.max_workers = max_workers
        return self

    def _preload_critical_modules(self):
        """
        Preload critical modules to avoid circular dependency issues during parallel import.
        Call this method before parallel scanning.
        """
        self.logger.info(
            "🔄 Preloading critical modules to avoid circular dependencies..."
        )

        loaded_count = 0
        failed_count = 0

        for module_name in self.preload_modules:
            try:
                importlib.import_module(module_name)
                loaded_count += 1
            except ImportError:
                # Some modules may not exist, which is normal
                failed_count += 1
            except Exception:
                # Other exceptions should be logged but not block execution
                failed_count += 1

        self.logger.info(
            "📦 Preloading completed: %d/%d modules successfully loaded",
            loaded_count,
            len(self.preload_modules),
        )
        if failed_count > 0:
            self.logger.debug("Skipped %d unavailable modules", failed_count)

    def add_preload_module(self, module_name: str) -> 'ComponentScanner':
        """Add module that needs to be preloaded"""
        if module_name not in self.preload_modules:
            self.preload_modules.append(module_name)
        return self

    def register_scan_context(
        self, path: str, metadata: Dict[str, Any]
    ) -> 'ComponentScanner':
        """
        Register context metadata for a scan path

        Args:
            path: Scan path
            metadata: Context metadata, can contain any custom information

        Returns:
            self, supports method chaining

        Example:
            ```python
            scanner = ComponentScanner()
            scanner.register_scan_context(
                "src/plugins",
                {"plugin_type": "core", "load_priority": 1}
            )
            scanner.add_scan_path("src/plugins").scan()
            ```
        """
        self.context_registry.register(path, metadata)
        return self

    def get_context_registry(self) -> ScanContextRegistry:
        """
        Get context registry

        Returns:
            Context registry instance
        """
        return self.context_registry

    def scan(self) -> 'ComponentScanner':
        """Execute scanning"""
        self.logger.info("🔍 Starting component scan...")

        # Collect all Python files
        python_files = self._collect_python_files()
        self.logger.info("📄 Found %d Python files", len(python_files))

        if not python_files:
            self.logger.warning("⚠️  No Python files found")
            return self

        # Scan components
        if self.parallel and len(python_files) > 1:
            self.logger.info(
                "⚡ Using parallel scan mode (max %d worker threads)", self.max_workers
            )
            # Preload critical modules before parallel scanning
            self._preload_critical_modules()
            self._parallel_scan(python_files)
        else:
            self.logger.info("📝 Using sequential scan mode")
            self._sequential_scan(python_files)

        self.logger.info("✅ Component scan completed")
        return self

    def _collect_python_files(self) -> List[Path]:
        """Collect all Python files"""
        python_files = []

        # Scan paths
        if self.scan_paths:
            self.logger.debug("Scanning paths: %s", ', '.join(self.scan_paths))
        for scan_path in self.scan_paths:
            files_from_path = self._collect_files_from_path(scan_path)
            python_files.extend(files_from_path)

        # Scan packages
        if self.scan_packages:
            self.logger.debug("Scanning packages: %s", ', '.join(self.scan_packages))
        for package in self.scan_packages:
            files_from_package = self._collect_files_from_package(package)
            python_files.extend(files_from_package)

        # Deduplicate
        unique_files = list(set(python_files))
        if len(python_files) != len(unique_files):
            self.logger.debug(
                "File count after deduplication: %d -> %d",
                len(python_files),
                len(unique_files),
            )

        return unique_files

    def _collect_files_from_path(self, path: str) -> List[Path]:
        """Collect Python files from path"""
        files = []
        path_obj = Path(path)

        if not path_obj.exists():
            self.logger.warning("Scan path does not exist: %s", path)
            return files

        if path_obj.is_file() and path_obj.suffix == '.py':
            if self._should_include_file(path_obj):
                files.append(path_obj)
        elif path_obj.is_dir():
            pattern = "**/*.py" if self.recursive else "*.py"
            for file_path in path_obj.glob(pattern):
                if self._should_include_file(file_path):
                    files.append(file_path)

        return files

    def _collect_files_from_package(self, package_name: str) -> List[Path]:
        """Collect Python files from package"""
        try:
            package = importlib.import_module(package_name)
            if hasattr(package, '__file__') and package.__file__:
                package_path = Path(package.__file__).parent
                return self._collect_files_from_path(str(package_path))
        except ImportError as e:
            self.logger.warning("Failed to import package %s: %s", package_name, e)

        return []

    def _should_include_file(self, file_path: Path) -> bool:
        """Check if file should be included"""
        # Exclude special files
        if file_path.name.startswith('__') and file_path.name.endswith('__.py'):
            return False

        # Check excluded paths
        for exclude_path in self.exclude_paths:
            if exclude_path in str(file_path):
                return False

        # Check excluded patterns
        for pattern in self.exclude_patterns:
            if pattern in file_path.name:
                return False

        # Check included patterns
        if self.include_patterns:
            return any(pattern in file_path.name for pattern in self.include_patterns)

        return True

    def _parallel_scan(self, python_files: List[Path]):
        """Parallel scan"""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._scan_file, file_path): file_path
                for file_path in python_files
            }

            for future in as_completed(futures):
                file_path = futures[future]
                try:
                    future.result()
                except Exception as e:
                    self.logger.error(
                        "Failed to scan file in parallel %s: %s", file_path, e
                    )

    def _sequential_scan(self, python_files: List[Path]):
        """Sequential scan"""
        for file_path in python_files:
            try:
                self._scan_file(file_path)
            except Exception as e:
                self.logger.error(
                    "Failed to scan file sequentially %s: %s", file_path, e
                )

    def _scan_file(self, file_path: Path):
        """
        Scan a single file.
        By importing the module, component decorators defined in the module can be triggered to complete automatic registration.
        """
        module_name = self._file_to_module_name(file_path)
        if not module_name:
            return

        try:
            importlib.import_module(module_name)
        except ImportError as e:
            self.logger.error("Failed to import module %s: %s", module_name, e)
            traceback.print_exc()
            sys.exit(1)
        except Exception as e:
            self.logger.error(
                "Unknown error occurred while scanning file %s: %s", file_path, e
            )
            traceback.print_exc()
            sys.exit(1)

    def _file_to_module_name(self, file_path: Path) -> Optional[str]:
        """Convert file path to module name"""
        try:
            # Get path relative to sys.path, sorted by path depth in descending order
            # Solve import issues between src.a.b.c and a.b.c
            sorted_sys_paths = sorted(
                [(path, len(Path(path).resolve().parts)) for path in sys.path],
                key=lambda x: x[1],
                reverse=True,  # Deeper paths come first
            )

            # Iterate through sorted paths
            for sys_path, _ in sorted_sys_paths:
                sys_path_obj = Path(sys_path).resolve()
                try:
                    relative_path = file_path.resolve().relative_to(sys_path_obj)
                    module_parts = list(relative_path.with_suffix("").parts)
                    return ".".join(module_parts)
                except ValueError:
                    continue
        except Exception:
            pass

        return None

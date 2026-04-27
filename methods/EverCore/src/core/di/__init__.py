# -*- coding: utf-8 -*-
"""
Dependency injection module

Note: For backward compatibility, this module re-exports commonly used functions and decorators.
In new code, it is recommended to import from specific submodules to improve readability:
- Decorators: from core.di.decorators import service, repository, component, mock_impl, factory
- Utility functions: from core.di.utils import get_bean_by_type, get_bean, enable_mock_mode, disable_mock_mode
- Container: from core.di.container import get_container
"""

# Decorators (from decorators.py)
from core.di.decorators import (
    component,
    service,
    repository,
    controller,
    injectable,
    mock_impl,
    factory,
    prototype,
    conditional,
    depends_on,
)

# Utility functions (from utils.py)
from core.di.utils import (
    get_bean,
    get_beans,
    get_bean_by_type,
    get_beans_by_type,
    register_bean,
    register_factory,
    register_singleton,
    register_prototype,
    register_primary,
    register_mock,
    enable_mock_mode,
    disable_mock_mode,
    is_mock_mode,
    clear_container,
    inject,
    lazy_inject,
    get_or_create,
    conditional_register,
    batch_register,
    get_bean_info,
    get_all_beans_info,
    list_all_beans,
    print_container_info,
    get_all_subclasses,
)

# Container (from container.py)
from core.di.container import get_container

# Define public API
__all__ = [
    # Decorators
    'component',
    'service',
    'repository',
    'controller',
    'injectable',
    'mock_impl',
    'factory',
    'prototype',
    'conditional',
    'depends_on',
    # Core utility functions
    'get_bean',
    'get_beans',
    'get_bean_by_type',
    'get_beans_by_type',
    'get_container',
    # Registration functions
    'register_bean',
    'register_factory',
    'register_singleton',
    'register_prototype',
    'register_primary',
    'register_mock',
    # Container checks
    # Mock mode
    'enable_mock_mode',
    'disable_mock_mode',
    'is_mock_mode',
    # Other utilities
    'clear_container',
    'inject',
    'lazy_inject',
    'get_or_create',
    'conditional_register',
    'batch_register',
    # Information queries
    'get_bean_info',
    'get_all_beans_info',
    'list_all_beans',
    'print_container_info',
    # Subclass queries
    'get_all_subclasses',
]

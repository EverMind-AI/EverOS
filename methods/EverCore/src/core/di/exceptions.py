# -*- coding: utf-8 -*-
"""
Dependency injection system exception class definitions
"""

from typing import Type, Any, List


class DIException(Exception):
    """Base exception for dependency injection system"""

    pass


class CircularDependencyError(DIException):
    """Circular dependency exception"""

    def __init__(self, dependency_chain: List[Type]):
        self.dependency_chain = dependency_chain
        chain_str = " -> ".join([cls.__name__ for cls in dependency_chain])
        super().__init__(f"Circular dependency detected: {chain_str}")


class BeanNotFoundError(DIException):
    """Bean not found exception"""

    def __init__(self, bean_type: Type = None, bean_name: str = None):
        self.bean_type = bean_type
        self.bean_name = bean_name

        if bean_name:
            super().__init__(f"Bean named '{bean_name}' not found")
        elif bean_type:
            # Handle string type bean_type
            if isinstance(bean_type, str):
                super().__init__(f"Bean of type '{bean_type}' not found")
            else:
                super().__init__(f"Bean of type '{bean_type.__name__}' not found")
        else:
            super().__init__("Specified Bean not found")


class DuplicateBeanError(DIException):
    """Duplicate Bean exception"""

    def __init__(self, bean_type: Type = None, bean_name: str = None):
        self.bean_type = bean_type
        self.bean_name = bean_name

        if bean_name:
            super().__init__(f"Bean named '{bean_name}' already exists")
        elif bean_type:
            super().__init__(f"Bean of type '{bean_type.__name__}' already exists")
        else:
            super().__init__("Bean already exists")


class FactoryError(DIException):
    """Factory exception"""

    def __init__(self, factory_type: Type, message: str = None):
        self.factory_type = factory_type
        default_msg = f"Factory '{factory_type.__name__}' failed to create instance"
        super().__init__(message or default_msg)


class DependencyResolutionError(DIException):
    """Dependency resolution exception"""

    def __init__(self, target_type: Type, missing_dependency: Type):
        self.target_type = target_type
        self.missing_dependency = missing_dependency
        super().__init__(
            f"Cannot resolve dependency '{missing_dependency.__name__}' for '{target_type.__name__}'"
        )


class MockNotEnabledError(DIException):
    """Mock mode not enabled exception"""

    def __init__(self):
        super().__init__(
            "Mock mode is not enabled, cannot register Mock implementation"
        )


class PrimaryBeanConflictError(DIException):
    """Primary Bean conflict exception"""

    def __init__(self, bean_type: Type, existing_primary: Type, new_primary: Type):
        self.bean_type = bean_type
        self.existing_primary = existing_primary
        self.new_primary = new_primary
        super().__init__(
            f"Multiple Primary implementations exist for type '{bean_type.__name__}': "
            f"'{existing_primary.__name__}' and '{new_primary.__name__}'"
        )

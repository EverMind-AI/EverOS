# -*- coding: utf-8 -*-
"""
Dependency injection decorator
"""

from typing import Type, TypeVar, Optional, Callable, Any, Dict
from functools import wraps

from core.di.container import get_container
from core.di.bean_definition import BeanScope

T = TypeVar('T')


def component(
    name: str = None,
    scope: BeanScope = BeanScope.SINGLETON,
    lazy: bool = False,
    primary: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
):
    """
    Component decorator

    Args:
        name: Bean name
        scope: Bean scope
        lazy: Whether to register lazily
        primary: Whether it is a primary Bean
        metadata: Metadata of the Bean, can be used to store additional information
    """

    def decorator(cls: Type[T]) -> Type[T]:
        cls._di_component = True
        cls._di_name = name
        cls._di_scope = scope
        cls._di_lazy = lazy
        cls._di_primary = primary
        cls._di_metadata = metadata

        # Check if marked to skip (via conditional decorator)
        if getattr(cls, '_di_skip', False):
            return cls

        if not lazy:
            # Register immediately
            container = get_container()
            container.register_bean(
                bean_type=cls,
                bean_name=name,
                scope=scope,
                is_primary=primary,
                metadata=metadata,
            )

        return cls

    return decorator


def service(
    name: str = None,
    scope: BeanScope = BeanScope.SINGLETON,
    lazy: bool = False,
    primary: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
):
    """
    Service component decorator
    """
    return component(name, scope, lazy, primary, metadata)


def repository(
    name: str = None,
    scope: BeanScope = BeanScope.SINGLETON,
    lazy: bool = False,
    primary: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
):
    """
    Repository component decorator
    """
    return component(name, scope, lazy, primary, metadata)


def controller(
    name: str = None,
    scope: BeanScope = BeanScope.SINGLETON,
    lazy: bool = False,
    primary: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
):
    """
    Controller component decorator
    """
    return component(name, scope, lazy, primary, metadata)


def injectable(
    name: str = None,
    scope: BeanScope = BeanScope.SINGLETON,
    lazy: bool = False,
    primary: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
):
    """
    Injectable component decorator
    """
    return component(name, scope, lazy, primary, metadata)


def mock_impl(
    name: str = None,
    scope: BeanScope = BeanScope.SINGLETON,
    primary: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
):
    """
    Mock implementation decorator - directly register Mock Bean, priority determined by container mechanism

    Args:
        name: Bean name
        scope: Bean scope
        primary: Whether it is a primary Bean
        metadata: Metadata of the Bean, can be used to store additional information
    """

    def decorator(cls: Type[T]) -> Type[T]:
        cls._di_mock = True
        cls._di_name = name
        cls._di_scope = scope
        cls._di_component = True  # Mark as component
        cls._di_metadata = metadata

        # Directly register Mock implementation, maintain consistency with other decorators
        container = get_container()
        container.register_bean(
            bean_type=cls,
            bean_name=name,
            scope=scope,
            is_primary=getattr(cls, '_di_primary', False),
            is_mock=True,
            metadata=metadata,
        )

        return cls

    return decorator


def factory(
    bean_type: Type[T] = None,
    name: str = None,
    lazy: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
):
    """
    Factory decorator

    Args:
        bean_type: The type of Bean to create
        name: Bean name
        lazy: Whether to register lazily
        metadata: Metadata of the Bean, can be used to store additional information
    """

    def decorator(func: Callable[[], T]) -> Callable[[], T]:
        target_type = bean_type or func.__annotations__.get('return', None)

        if not target_type:
            raise ValueError("Factory decorator must specify return type")

        func._di_factory = True
        func._di_bean_type = target_type
        func._di_name = name
        func._di_lazy = lazy
        func._di_metadata = metadata

        if not lazy:
            # Register Factory immediately
            container = get_container()
            container.register_factory(
                bean_type=target_type,
                factory_method=func,
                bean_name=name,
                metadata=metadata,
            )

        return func

    return decorator


def prototype(cls: Type[T]) -> Type[T]:
    """
    Prototype scope decorator (create a new instance every time it is retrieved)
    """
    cls._di_scope = BeanScope.PROTOTYPE

    # If already a component, update scope
    if hasattr(cls, '_di_component'):
        container = get_container()
        container.register_bean(
            bean_type=cls,
            bean_name=getattr(cls, '_di_name', None),
            scope=BeanScope.PROTOTYPE,
            is_primary=getattr(cls, '_di_primary', False),
            metadata=getattr(cls, '_di_metadata', None),
        )

    return cls


def conditional(condition: Callable[[], bool]):
    """
    Conditional decorator - control conditional registration of Bean
    Note: Should be used before decorators like @component
    """

    def decorator(cls: Type[T]) -> Type[T]:
        # Set conditional flag, let subsequent decorators (e.g., component) decide whether to register based on this
        cls._di_conditional = condition

        # If condition is not met, mark as skipped
        if not condition():
            cls._di_skip = True

        return cls

    return decorator


def depends_on(*dependencies: Type):
    """
    Dependency decorator - declare dependency relationships of Bean
    """

    def decorator(cls: Type[T]) -> Type[T]:
        cls._di_dependencies = dependencies
        return cls

    return decorator

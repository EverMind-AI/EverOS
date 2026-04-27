# -*- coding: utf-8 -*-
"""
Dependency injection utility functions
"""

from typing import Type, TypeVar, List, Dict, Any, Optional, Callable
import inspect

from core.di.container import get_container
from core.di.bean_definition import BeanScope
from core.di.exceptions import BeanNotFoundError

T = TypeVar('T')


def get_bean(name: str) -> Any:
    """
    Get Bean by name

    Args:
        name: Bean name

    Returns:
        Bean instance

    Raises:
        BeanNotFoundError: When Bean does not exist
    """
    return get_container().get_bean(name)


def get_beans() -> Dict[str, Any]:
    """
    Get all Beans

    Returns:
        Dictionary of all Beans, key is name, value is instance
    """
    return get_container().get_beans()


def get_bean_by_type(bean_type: Type[T]) -> T:
    """
    Get Bean by type (Primary implementation or unique implementation)

    Args:
        bean_type: Bean type

    Returns:
        Bean instance

    Raises:
        BeanNotFoundError: When Bean does not exist
    """
    return get_container().get_bean_by_type(bean_type)


def get_beans_by_type(bean_type: Type[T]) -> List[T]:
    """
    Get all Bean implementations by type

    Args:
        bean_type: Bean type

    Returns:
        List of Bean instances
    """
    return get_container().get_beans_by_type(bean_type)


def register_bean(
    bean_type: Type[T],
    instance: T = None,
    name: str = None,
    scope: BeanScope = BeanScope.SINGLETON,
    is_primary: bool = False,
    is_mock: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Register Bean

    Args:
        bean_type: Bean type
        instance: Bean instance (optional, will be created automatically if not provided)
        name: Bean name
        scope: Bean scope
        is_primary: Whether it is a Primary implementation
        is_mock: Whether it is a Mock implementation
        metadata: Bean metadata, can be used to store additional information
    """
    get_container().register_bean(
        bean_type=bean_type,
        instance=instance,
        bean_name=name,
        scope=scope,
        is_primary=is_primary,
        is_mock=is_mock,
        metadata=metadata,
    )


def register_factory(
    bean_type: Type[T],
    factory_method: Callable[[], T],
    name: str = None,
    is_primary: bool = False,
    is_mock: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Register Factory method

    Args:
        bean_type: Bean type
        factory_method: Factory method
        name: Bean name
        is_primary: Whether it is a Primary implementation
        is_mock: Whether it is a Mock implementation
        metadata: Bean metadata, can be used to store additional information
    """
    get_container().register_factory(
        bean_type=bean_type,
        factory_method=factory_method,
        bean_name=name,
        is_primary=is_primary,
        is_mock=is_mock,
        metadata=metadata,
    )


def register_singleton(
    bean_type: Type[T], instance: T = None, name: str = None
) -> None:
    """
    Register singleton Bean

    Args:
        bean_type: Bean type
        instance: Bean instance
        name: Bean name
    """
    register_bean(bean_type, instance, name, BeanScope.SINGLETON)


def register_prototype(bean_type: Type[T], name: str = None) -> None:
    """
    Register prototype Bean (create new instance every time it is retrieved)

    Args:
        bean_type: Bean type
        name: Bean name
    """
    register_bean(bean_type, None, name, BeanScope.PROTOTYPE)


def register_primary(bean_type: Type[T], instance: T = None, name: str = None) -> None:
    """
    Register Primary Bean

    Args:
        bean_type: Bean type
        instance: Bean instance
        name: Bean name
    """
    register_bean(bean_type, instance, name, BeanScope.SINGLETON, is_primary=True)


def register_mock(bean_type: Type[T], instance: T = None, name: str = None) -> None:
    """
    Register Mock Bean

    Args:
        bean_type: Bean type
        instance: Bean instance
        name: Bean name
    """
    register_bean(bean_type, instance, name, BeanScope.SINGLETON, is_mock=True)


def enable_mock_mode() -> None:
    """Enable mock mode"""
    get_container().enable_mock_mode()


def disable_mock_mode() -> None:
    """Disable mock mode"""
    get_container().disable_mock_mode()


def is_mock_mode() -> bool:
    """Check if in mock mode"""
    return get_container().is_mock_mode()


def clear_container() -> None:
    """Clear container"""
    get_container().clear()


def inject(target_func: Callable) -> Callable:
    """
    Function dependency injection decorator

    Automatically inject Bean by parameter type
    """

    def wrapper(*args, **kwargs):
        # Get function signature
        signature = inspect.signature(target_func)

        # Prepare injected parameters
        injected_kwargs = {}
        for param_name, param in signature.parameters.items():
            if param_name not in kwargs and param.annotation != inspect.Parameter.empty:
                try:
                    injected_kwargs[param_name] = get_bean_by_type(param.annotation)
                except BeanNotFoundError:
                    # If Bean not found and parameter has default value, use default value
                    if param.default != inspect.Parameter.empty:
                        injected_kwargs[param_name] = param.default
                    else:
                        # Required parameter but Bean not found, raise exception
                        raise

        # Merge parameters
        kwargs.update(injected_kwargs)
        return target_func(*args, **kwargs)

    return wrapper


def lazy_inject(bean_type: Type[T]) -> Callable[[], T]:
    """
    Lazy injection function

    Returns a lambda function that retrieves Bean when called

    Args:
        bean_type: Bean type

    Returns:
        Function to lazily retrieve Bean
    """
    return lambda: get_bean_by_type(bean_type)


def get_or_create(bean_type: Type[T], factory: Callable[[], T] = None) -> T:
    """
    Get Bean, create if not exists

    Args:
        bean_type: Bean type
        factory: Factory method (optional)

    Returns:
        Bean instance
    """
    try:
        return get_bean_by_type(bean_type)
    except BeanNotFoundError:
        if factory:
            instance = factory()
            register_bean(bean_type, instance)
            return instance
        else:
            # Try to create automatically
            try:
                instance = bean_type()
                register_bean(bean_type, instance)
                return instance
            except Exception as e:
                raise BeanNotFoundError(bean_type=bean_type)


def conditional_register(
    condition: Callable[[], bool],
    bean_type: Type[T],
    instance: T = None,
    name: str = None,
) -> None:
    """
    Conditionally register Bean

    Args:
        condition: Condition function
        bean_type: Bean type
        instance: Bean instance
        name: Bean name
    """
    if condition():
        register_bean(bean_type, instance, name)


def batch_register(beans: Dict[Type, Any]) -> None:
    """
    Batch register Beans

    Args:
        beans: Bean dictionary, key is type, value is instance
    """
    for bean_type, instance in beans.items():
        register_bean(bean_type, instance)


def get_bean_info(bean_type: Type = None, bean_name: str = None) -> Dict[str, Any]:
    """
    Get Bean information

    Args:
        bean_type: Bean type
        bean_name: Bean name

    Returns:
        Bean information dictionary
    """
    container = get_container()
    info = {}

    if bean_name:
        if container.contains_bean(bean_name):
            bean_def = container._named_beans[bean_name]
            info = {
                'name': bean_def.bean_name,
                'type': bean_def.bean_type.__name__,
                'scope': bean_def.scope.value,
                'is_primary': bean_def.is_primary,
                'is_mock': bean_def.is_mock,
                'has_instance': bean_def in container._singleton_instances,
            }
    elif bean_type:
        if container.contains_bean_by_type(bean_type):
            definitions = container._bean_definitions[bean_type]
            info = {
                'type': bean_type.__name__,
                'implementations': [
                    {
                        'name': def_.bean_name,
                        'scope': def_.scope.value,
                        'is_primary': def_.is_primary,
                        'is_mock': def_.is_mock,
                    }
                    for def_ in definitions
                ],
            }

    return info


def get_all_beans_info() -> List[Dict[str, Any]]:
    """
    Get information of all registered Beans (structured data)

    Returns:
        List of structured Bean information data
    """
    return get_container().list_all_beans_info()


def list_all_beans() -> List[str]:
    """
    List all registered Bean information (formatted strings)

    Returns:
        List of formatted Bean information strings
    """
    beans_info = get_all_beans_info()

    formatted_beans = []
    for bean_info in beans_info:
        flags = []
        if bean_info['is_primary']:
            flags.append("primary")
        if bean_info['is_mock']:
            flags.append("mock")
        flag_str = f" ({', '.join(flags)})" if flags else ""

        formatted_beans.append(
            f"   • {bean_info['name']} ({bean_info['type_name']}) [{bean_info['scope']}]{flag_str}"
        )

    return formatted_beans


def print_container_info():
    """Print container information"""
    formatted_beans = list_all_beans()
    from core.observation.logger import (
        info,
    )  # Convenient usage, suitable for occasional calls

    info(f"\n📦 Dependency injection container information:")
    info(f"   Total Bean count: {len(formatted_beans)}")
    info(f"   Mock mode: {'enabled' if is_mock_mode() else 'disabled'}")

    if formatted_beans:
        info("\n📋 Registered Beans:")
        for bean_line in formatted_beans:
            info(bean_line)
    else:
        info("   No registered Beans")
    info("")


# ===============================================

# subclasses


def get_all_subclasses(base_class: Type[T]) -> List[Type[T]]:
    """
    Recursively get all subclasses of specified class (including subclasses of subclasses)

    Args:
        base_class: Base class

    Returns:
        List[Type[T]]: List of all subclasses, including direct and indirect subclasses
    """
    subclasses = []
    for subclass in base_class.__subclasses__():
        if subclass != base_class:
            subclasses.append(subclass)
            # Recursively get subclasses of subclass
            subclasses.extend(get_all_subclasses(subclass))
    return subclasses

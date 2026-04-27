# -*- coding: utf-8 -*-
"""
Core implementation of dependency injection container

Lock usage strategy:
- Read-only operations (e.g., is_mock_mode, contains_bean*): no lock, because reading immutable attributes
- Operations modifying container state: protected by self._lock
- Bean retrieval operations: require lock, because they may create and cache singleton instances
- Global container creation: use _container_lock to ensure singleton
"""

import inspect
import abc
from typing import (
    Dict,
    Type,
    TypeVar,
    Optional,
    Any,
    List,
    Set,
    Callable,
    Union,
    get_origin,
    get_args,
)
from threading import RLock

from core.di.bean_definition import BeanDefinition, BeanScope
from core.di.bean_order_strategy import BeanOrderStrategy
from core.di.scan_context import ScanContextRegistry
from core.di.exceptions import (
    CircularDependencyError,
    BeanNotFoundError,
    DuplicateBeanError,
    FactoryError,
    DependencyResolutionError,
    MockNotEnabledError,
)

T = TypeVar('T')


class DIContainer:
    """Dependency injection container"""

    # Class-level Bean ordering strategy, can be replaced
    _bean_order_strategy_class = BeanOrderStrategy

    @classmethod
    def replace_bean_order_strategy(cls, strategy_class):
        """
        Replace Bean ordering strategy class

        Args:
            strategy_class: New ordering strategy class, must have interface compatible with BeanOrderStrategy

        Note:
            This is a temporary solution because the DI mechanism is not fully established.
            This method affects the ordering behavior of all DIContainer instances.
        """
        cls._bean_order_strategy_class = strategy_class

    def __init__(self):
        self._lock = RLock()
        # Store Bean definitions by type {Type: [BeanDefinition]}
        self._bean_definitions: Dict[Type, List[BeanDefinition]] = {}
        # Store Bean definitions by name {name: BeanDefinition}
        self._named_beans: Dict[str, BeanDefinition] = {}

        # Store singleton instances {BeanDefinition: instance}
        self._singleton_instances: Dict[BeanDefinition, Any] = {}

        # Mock mode
        self._mock_mode = False
        # Dependency resolution stack, used to detect circular dependencies
        self._resolving_stack: List[Type] = []

        # Performance optimization cache
        # Inheritance relationship cache {parent_type: [child_types]}
        self._inheritance_cache: Dict[Type, List[Type]] = {}
        # Candidate Bean cache {(Type, mock_mode): [BeanDefinition]}
        self._candidates_cache: Dict[tuple, List[BeanDefinition]] = {}
        # Cache invalidation flag
        self._cache_dirty = False

    def enable_mock_mode(self):
        """Enable mock mode"""
        with self._lock:
            if not self._mock_mode:
                self._mock_mode = True
                self._invalidate_cache()

    def disable_mock_mode(self):
        """Disable mock mode"""
        with self._lock:
            if self._mock_mode:
                self._mock_mode = False
                self._invalidate_cache()

    def is_mock_mode(self) -> bool:
        """Check if in mock mode"""
        return self._mock_mode

    def _create_bean_definition(
        self,
        bean_type: Type[T],
        bean_name: str = None,
        scope: BeanScope = BeanScope.SINGLETON,
        is_primary: bool = False,
        is_mock: bool = False,
        factory_method: Callable = None,
        instance: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BeanDefinition:
        """
        Create BeanDefinition, automatically merging metadata from scan context

        Args:
            bean_type: Type of the Bean
            bean_name: Name of the Bean
            scope: Scope of the Bean
            is_primary: Whether it is the primary Bean
            is_mock: Whether it is a mock implementation
            factory_method: Factory method
            instance: Pre-created instance
            metadata: Metadata of the Bean

        Returns:
            BeanDefinition instance
        """
        # Merge metadata: first get from scan_context, then merge with passed metadata
        merged_metadata = {}

        # 1. Get file path through bean_type and search for corresponding context metadata
        context_metadata = ScanContextRegistry.search_metadata_for_type(bean_type)
        if context_metadata:
            merged_metadata.update(context_metadata)

        # 2. Merge passed metadata (passed metadata has higher priority, can override scan context)
        if metadata:
            merged_metadata.update(metadata)

        # 3. Create BeanDefinition
        bean_def = BeanDefinition(
            bean_type=bean_type,
            bean_name=bean_name,
            scope=scope,
            is_primary=is_primary,
            is_mock=is_mock,
            factory_method=factory_method,
            instance=instance,
            metadata=merged_metadata if merged_metadata else None,
        )

        return bean_def

    def register_bean(
        self,
        bean_type: Type[T],
        bean_name: str = None,
        scope: BeanScope = BeanScope.SINGLETON,
        is_primary: bool = False,
        is_mock: bool = False,
        instance: T = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> 'DIContainer':
        """
        Register Bean

        Args:
            bean_type: Type of the Bean
            bean_name: Name of the Bean
            scope: Scope of the Bean
            is_primary: Whether it is the primary Bean
            is_mock: Whether it is a mock implementation
            instance: Pre-created instance
            metadata: Metadata of the Bean, can be used to store extra information
        """
        with self._lock:
            # Use unified method to create BeanDefinition, automatically merges scan context metadata
            bean_def = self._create_bean_definition(
                bean_type=bean_type,
                bean_name=bean_name,
                scope=scope,
                is_primary=is_primary,
                is_mock=is_mock,
                instance=instance,
                metadata=metadata,
            )

            # Check for duplicate registration
            if bean_def.bean_name in self._named_beans:
                existing = self._named_beans[bean_def.bean_name]
                if not (is_mock or existing.is_mock):
                    raise DuplicateBeanError(bean_name=bean_def.bean_name)

            # Register Bean definition
            if bean_type not in self._bean_definitions:
                self._bean_definitions[bean_type] = []
            self._bean_definitions[bean_type].append(bean_def)
            self._named_beans[bean_def.bean_name] = bean_def

            # Analyze dependency relationships
            self._analyze_dependencies(bean_def)

            # If instance is provided, store directly
            if instance is not None:
                self._singleton_instances[bean_def] = instance

            # Invalidate cache
            self._invalidate_cache()

            return self

    def register_factory(
        self,
        bean_type: Type[T],
        factory_method: Callable[[], T],
        bean_name: str = None,
        is_primary: bool = False,
        is_mock: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> 'DIContainer':
        """
        Register factory method

        Args:
            bean_type: Type of the Bean
            factory_method: Factory method
            bean_name: Name of the Bean
            is_primary: Whether it is the primary Bean
            is_mock: Whether it is a mock implementation
            metadata: Metadata of the Bean, can be used to store extra information
        """
        with self._lock:
            # Use unified method to create BeanDefinition, automatically merges scan context metadata
            bean_def = self._create_bean_definition(
                bean_type=bean_type,
                bean_name=bean_name,
                scope=BeanScope.FACTORY,
                is_primary=is_primary,
                is_mock=is_mock,
                factory_method=factory_method,
                metadata=metadata,
            )

            # Check for duplicate registration
            if bean_def.bean_name in self._named_beans:
                existing = self._named_beans[bean_def.bean_name]
                if not (is_mock or existing.is_mock):
                    raise DuplicateBeanError(bean_name=bean_def.bean_name)

            # Register Bean definition
            if bean_type not in self._bean_definitions:
                self._bean_definitions[bean_type] = []
            self._bean_definitions[bean_type].append(bean_def)
            self._named_beans[bean_def.bean_name] = bean_def

            # Invalidate cache
            self._invalidate_cache()

            return self

    def get_bean(self, bean_name: str) -> Any:
        """Get Bean by name"""
        with self._lock:
            if bean_name not in self._named_beans:
                raise BeanNotFoundError(bean_name=bean_name)

            bean_def = self._named_beans[bean_name]
            return self._create_instance(bean_def)

    def get_bean_by_type(self, bean_type: Type[T]) -> T:
        """Get Bean by type (return Primary or unique implementation)"""
        with self._lock:
            candidates = self._get_candidates_with_priority(bean_type)

            if not candidates:
                raise BeanNotFoundError(bean_type=bean_type)

            # If only one candidate, return it
            if len(candidates) == 1:
                return self._create_instance(candidates[0])

            # Multiple candidates, return the highest priority one
            return self._create_instance(candidates[0])

    def _get_candidates_with_priority(self, bean_type: Type) -> List[BeanDefinition]:
        """
        Get candidate Bean definitions for the type (sorted by priority)

        Priority sorting rules (from high to low):
        1. is_mock: Mock Bean > Non-Mock Bean (only effective in mock mode)
        2. Matching method: Direct match > Implementation class match
        3. primary: Primary Bean > Non-Primary Bean
        4. scope: Factory Bean > Regular Bean
        """
        # Use cache key
        cache_key = (bean_type, self._mock_mode)

        # Check cache
        if cache_key in self._candidates_cache:
            return self._candidates_cache[cache_key]

        # Ensure inheritance relationship cache is up to date
        self._build_inheritance_cache()

        # Collect all candidate Beans
        all_candidates = []
        direct_match_types = set()

        # 1. Collect directly matched Beans (including Primary and non-Primary)
        if bean_type in self._bean_definitions:
            for bean_def in self._bean_definitions[bean_type]:
                if self._is_bean_available(bean_def):
                    all_candidates.append(bean_def)
                    direct_match_types.add(bean_def.bean_type)

        # 2. Collect implementation class matched Beans (implementations of interface/abstract class)
        impl_types = self._inheritance_cache.get(bean_type, [])
        for impl_type in impl_types:
            if impl_type in self._bean_definitions:
                for bean_def in self._bean_definitions[impl_type]:
                    if self._is_bean_available(bean_def):
                        all_candidates.append(bean_def)
                        # impl_type not added to direct_match_types, because it's implementation class match

        # 3. Use current configured Bean ordering strategy for unified sorting
        priority_candidates = self._bean_order_strategy_class.sort_beans_with_context(
            bean_defs=all_candidates,
            direct_match_types=direct_match_types,
            mock_mode=self._mock_mode,
        )

        # Cache result
        self._candidates_cache[cache_key] = priority_candidates
        return priority_candidates

    def get_beans_by_type(self, bean_type: Type[T]) -> List[T]:
        """Get all Bean implementations by type"""
        with self._lock:
            candidates = self._get_candidates_with_priority(bean_type)
            return [self._create_instance(bean_def) for bean_def in candidates]

    def get_beans(self) -> Dict[str, Any]:
        """Get all registered Beans"""
        with self._lock:
            result = {}
            for name, bean_def in self._named_beans.items():
                if self._is_bean_available(bean_def):
                    try:
                        result[name] = self._create_instance(bean_def)
                    except Exception:
                        # Skip Beans that cannot be created
                        continue
            return result

    def contains_bean(self, bean_name: str) -> bool:
        """Check if container contains Bean with specified name"""
        return bean_name in self._named_beans

    def contains_bean_by_type(self, bean_type: Type) -> bool:
        """Check if container contains Bean with specified type"""
        return bean_type in self._bean_definitions

    def clear(self):
        """Clear container"""
        with self._lock:
            self._bean_definitions.clear()
            self._named_beans.clear()
            self._singleton_instances.clear()
            self._resolving_stack.clear()
            self._invalidate_cache()

    def list_all_beans_info(self) -> List[Dict[str, Any]]:
        """
        List all registered Bean information

        Returns:
            List of Bean information, each Bean contains:
            - name: Bean name
            - type_name: Bean type name
            - scope: Bean scope
            - is_primary: Whether it is a Primary Bean
            - is_mock: Whether it is a Mock Bean
        """
        beans_info = []

        # Collect all Bean information
        for name, bean_def in self._named_beans.items():
            if self._is_bean_available(bean_def):
                beans_info.append(
                    {
                        'name': name,
                        'type_name': bean_def.bean_type.__name__,
                        'scope': bean_def.scope.value,
                        'is_primary': bean_def.is_primary,
                        'is_mock': bean_def.is_mock,
                    }
                )

        return beans_info

    def _invalidate_cache(self):
        """Invalidate all caches"""
        self._inheritance_cache.clear()
        self._candidates_cache.clear()
        self._cache_dirty = True

    def _is_bean_available(self, bean_def: BeanDefinition) -> bool:
        """Check if Bean is available in current mode"""
        if self._mock_mode:
            # In mock mode, both mock and non-mock beans are available
            return True
        else:
            # In non-mock mode, only non-mock beans are available
            return not bean_def.is_mock

    def _build_inheritance_cache(self):
        """Build type inheritance relationship cache"""
        if not self._cache_dirty:
            return

        self._inheritance_cache.clear()

        # Get registered types
        registered_types = list(self._bean_definitions.keys())

        # Additionally collect ABC parent types (exclude abc.ABC base class)
        all_parent_types = set(registered_types)
        for registered_type in registered_types:
            try:
                # Get all parent classes, especially ABC abstract base classes
                for base in registered_type.__mro__[1:]:  # Skip self
                    # Exclude abc.ABC base class and object base class, they are too generic
                    if (
                        base != abc.ABC
                        and base != object
                        and hasattr(base, '__abstractmethods__')
                    ):  # ABC type
                        all_parent_types.add(base)
            except (AttributeError, TypeError):
                # Handle non-type cases
                continue

        # Build inheritance relationship index for all types (including ABC parents)
        # parent_type -> [list of its child implementations]
        for parent_type in all_parent_types:
            child_implementations = []
            for child_type in registered_types:
                if child_type != parent_type:
                    try:
                        if issubclass(child_type, parent_type):
                            child_implementations.append(child_type)
                    except TypeError:
                        # Handle non-type cases
                        continue
            if child_implementations:
                self._inheritance_cache[parent_type] = child_implementations

        self._cache_dirty = False

    def _create_instance(self, bean_def: BeanDefinition) -> Any:
        """Create Bean instance"""
        # Check for circular dependency
        if bean_def.bean_type in self._resolving_stack:
            dependency_chain = self._resolving_stack + [bean_def.bean_type]
            raise CircularDependencyError(dependency_chain)

        # Handle different scopes
        if bean_def.scope == BeanScope.SINGLETON:
            # Singleton mode: check cache, return directly if exists
            if bean_def in self._singleton_instances:
                return self._singleton_instances[bean_def]

        elif bean_def.scope == BeanScope.FACTORY:
            # Factory mode: create new instance by calling factory method each time
            if bean_def.factory_method:
                try:
                    return bean_def.factory_method()
                except Exception as e:
                    raise FactoryError(bean_def.bean_type, str(e))
            else:
                raise FactoryError(bean_def.bean_type, "Factory method not set")

        elif bean_def.scope == BeanScope.PROTOTYPE:
            # Prototype mode: create new instance each time, no caching
            try:
                self._resolving_stack.append(bean_def.bean_type)
                return self._instantiate_with_dependencies(bean_def)
            finally:
                if bean_def.bean_type in self._resolving_stack:
                    self._resolving_stack.remove(bean_def.bean_type)

        # If preset instance exists, return directly
        if bean_def.instance is not None:
            return bean_def.instance

        # Create new instance (SINGLETON scope)
        try:
            self._resolving_stack.append(bean_def.bean_type)
            instance = self._instantiate_with_dependencies(bean_def)

            # Store singleton instance
            if bean_def.scope == BeanScope.SINGLETON:
                self._singleton_instances[bean_def] = instance

            return instance
        finally:
            if bean_def.bean_type in self._resolving_stack:
                self._resolving_stack.remove(bean_def.bean_type)

    def _instantiate_with_dependencies(self, bean_def: BeanDefinition) -> Any:
        """Instantiate Bean and inject dependencies"""
        bean_type = bean_def.bean_type

        # Get constructor signature
        try:
            signature = inspect.signature(bean_type.__init__)
        except Exception:
            # If signature cannot be obtained, try parameterless constructor
            return bean_type()

        # Prepare constructor parameters
        init_params = {}
        for param_name, param in signature.parameters.items():
            if param_name == 'self':
                continue

            # Try to inject dependency by type
            if param.annotation != inspect.Parameter.empty:
                try:
                    # Check if it is a generic type (e.g., List[T])
                    origin = get_origin(param.annotation)
                    if origin is list or origin is List:
                        # Handle dependency injection for List[T] type
                        args = get_args(param.annotation)
                        if args:
                            # Get generic parameter type
                            element_type = args[0]
                            # Inject all implementations of this type
                            dependencies = self.get_beans_by_type(element_type)
                            init_params[param_name] = dependencies
                        else:
                            # If no generic parameters, try empty list
                            init_params[param_name] = []
                    else:
                        # Dependency injection for normal types
                        dependency = self.get_bean_by_type(param.annotation)
                        init_params[param_name] = dependency
                except BeanNotFoundError:
                    if param.default == inspect.Parameter.empty:
                        # Required parameter but dependency not found
                        raise DependencyResolutionError(bean_type, param.annotation)

        return bean_type(**init_params)

    def _analyze_dependencies(self, bean_def: BeanDefinition):
        """Analyze Bean's dependency relationships"""
        try:
            signature = inspect.signature(bean_def.bean_type.__init__)
            for param_name, param in signature.parameters.items():
                if param_name == 'self':
                    continue
                if param.annotation != inspect.Parameter.empty:
                    bean_def.dependencies.add(param.annotation)
        except Exception:
            # If analysis fails, skip
            pass


# Global container instance
_global_container: Optional[DIContainer] = None
_container_lock = RLock()


def get_container() -> DIContainer:
    """Get global container instance"""
    global _global_container
    if _global_container is None:
        with _container_lock:
            if _global_container is None:
                _global_container = DIContainer()
    return _global_container

# -*- coding: utf-8 -*-
"""
Bean definition module

Contains Bean definition classes and scope enumeration
"""

from enum import Enum
from typing import Type, Callable, Any, Set, Dict, Optional


class BeanScope(str, Enum):
    """Bean scope enumeration"""

    SINGLETON = "singleton"
    PROTOTYPE = "prototype"
    FACTORY = "factory"


class BeanDefinition:
    """Bean definition class"""

    def __init__(
        self,
        bean_type: Type,
        bean_name: str = None,
        scope: BeanScope = BeanScope.SINGLETON,
        is_primary: bool = False,
        is_mock: bool = False,
        factory_method: Callable = None,
        instance: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize Bean definition

        Args:
            bean_type: Type of the Bean
            bean_name: Name of the Bean, defaults to lowercase type name
            scope: Scope of the Bean, defaults to singleton
            is_primary: Whether it is the primary Bean, used to prioritize when multiple implementations exist
            is_mock: Whether it is a Mock implementation
            factory_method: Factory method used to create the Bean instance
            instance: Pre-created instance
            metadata: Metadata of the Bean, can be used to store additional information
        """
        self.bean_type = bean_type
        self.bean_name = bean_name or bean_type.__name__.lower()
        self.scope = scope
        self.is_primary = is_primary
        self.is_mock = is_mock
        self.factory_method = factory_method
        self.instance = instance
        self.metadata = metadata or {}
        # Dependency set
        self.dependencies: Set[Type] = set()

    def __repr__(self):
        metadata_str = f", metadata={self.metadata}" if self.metadata else ""
        return f"BeanDefinition(type={self.bean_type.__name__}, name={self.bean_name}, scope={self.scope.value}{metadata_str})"

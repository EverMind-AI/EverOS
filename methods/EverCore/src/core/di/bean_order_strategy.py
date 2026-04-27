# -*- coding: utf-8 -*-
"""
Bean ordering strategy module

Used to determine the priority order of Beans based on BeanDefinition attributes (such as is_primary, metadata, etc.)

Priority ranking rules (from highest to lowest):
1. is_mock: In mock mode, Mock Bean > Non-Mock Bean; in non-mock mode, Mock Beans are filtered out directly
2. Matching method: Direct match > Implementation class match
3. primary: Primary Bean > Non-Primary Bean
4. scope: Factory Bean > Regular Bean
"""

from typing import List, Tuple, Set, Type
from core.di.bean_definition import BeanDefinition, BeanScope


class BeanOrderStrategy:
    """
    Bean ordering strategy class

    Calculates and sorts Bean priorities based on various attributes of BeanDefinition
    Used to determine usage order when there are multiple candidate Beans

    Order key format: (mock_priority, match_priority, primary_priority, scope_priority)
    Smaller values indicate higher priority
    """

    @staticmethod
    def calculate_order_key(
        bean_def: BeanDefinition, is_direct_match: bool, mock_mode: bool = False
    ) -> Tuple[int, int, int, int]:
        """
        Calculate the ordering key for a Bean

        Args:
            bean_def: Bean definition object
            is_direct_match: Whether it is a direct match (True = direct match, False = implementation class match)
            mock_mode: Whether in mock mode

        Returns:
            Tuple[int, int, int, int]: Order key tuple
            Format: (mock_priority, match_priority, primary_priority, scope_priority)

        Priority rules:
            - mock_priority: In mock mode, Mock Bean = 0, Non-Mock Bean = 1; in non-mock mode, both are 0
            - match_priority: Direct match = 0, Implementation class match = 1
            - primary_priority: Primary Bean = 0, Non-Primary Bean = 1
            - scope_priority: Factory Bean = 0, Non-Factory Bean = 1
        """
        # 1. Mock priority (only differentiated in mock mode)
        if mock_mode:
            mock_priority = 0 if bean_def.is_mock else 1
        else:
            mock_priority = 0  # No distinction in non-mock mode

        # 2. Match method priority (direct match takes precedence)
        match_priority = 0 if is_direct_match else 1

        # 3. Primary priority (Primary takes precedence)
        primary_priority = 0 if bean_def.is_primary else 1

        # 4. Scope priority (Factory takes precedence)
        scope_priority = 0 if bean_def.scope == BeanScope.FACTORY else 1

        return (mock_priority, match_priority, primary_priority, scope_priority)

    @staticmethod
    def sort_beans_with_context(
        bean_defs: List[BeanDefinition],
        direct_match_types: Set[Type],
        mock_mode: bool = False,
    ) -> List[BeanDefinition]:
        """
        Sort the list of Bean definitions based on context information

        Args:
            bean_defs: List of Bean definitions
            direct_match_types: Set of types that directly match
            mock_mode: Whether in mock mode

        Returns:
            List[BeanDefinition]: Sorted list of Bean definitions

        Note:
            - In non-mock mode, Mock Beans are filtered out directly and do not participate in sorting
            - In mock mode, Mock Beans take precedence over non-Mock Beans
        """
        # Filter out all Mock Beans in non-mock mode
        if not mock_mode:
            bean_defs = [bd for bd in bean_defs if not bd.is_mock]

        # Calculate order key for each Bean, then sort by key
        sorted_beans = sorted(
            bean_defs,
            key=lambda bd: BeanOrderStrategy.calculate_order_key(
                bean_def=bd,
                is_direct_match=bd.bean_type in direct_match_types,
                mock_mode=mock_mode,
            ),
        )
        return sorted_beans

    @staticmethod
    def sort_beans(bean_defs: List[BeanDefinition]) -> List[BeanDefinition]:
        """
        Sort the list of Bean definitions simply (compatible with old interface)

        Args:
            bean_defs: List of Bean definitions

        Returns:
            List[BeanDefinition]: Sorted list of Bean definitions

        Note:
            This method only considers primary and scope, not matching method or mock mode
            It is recommended to use the sort_beans_with_context method for complete sorting functionality
        """
        # Sort by (primary_priority, scope_priority)
        sorted_beans = sorted(
            bean_defs,
            key=lambda bd: (
                0 if bd.is_primary else 1,  # Primary takes precedence
                0 if bd.scope == BeanScope.FACTORY else 1,  # Factory takes precedence
            ),
        )
        return sorted_beans

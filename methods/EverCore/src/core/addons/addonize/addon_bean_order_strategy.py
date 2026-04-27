# -*- coding: utf-8 -*-
"""
Addon Bean ordering strategy module

Extends the standard BeanOrderStrategy to add support for addon_tag priority

Priority ordering rules (from highest to lowest):
1. addon_tag: addon priority configured via environment variable (smaller number means higher priority)
2. is_mock: In mock mode, Mock Bean > Non-Mock Bean; in non-mock mode, Mock Beans are filtered out
3. Matching method: Direct match > Implementation class match
4. primary: Primary Bean > Non-Primary Bean
5. scope: Factory Bean > Regular Bean
"""

import os
from typing import List, Tuple, Set, Type, Dict
from core.di.bean_definition import BeanDefinition, BeanScope
from core.di.bean_order_strategy import BeanOrderStrategy
from core.di.container import DIContainer
from core.observation.logger import get_logger

logger = get_logger(__name__)


class AddonBeanOrderStrategy(BeanOrderStrategy):
    """
    Addon Bean ordering strategy class

    Inherits from BeanOrderStrategy, extends support for addon_tag priority
    addon_tag priority is configured via environment variable ADDON_PRIORITY
    Format: "addon1:priority1,addon2:priority2"
    Example: "core:1000,enterprise:50" means enterprise has higher priority (smaller number means higher priority)
    """

    # Default addon priority configuration
    DEFAULT_ADDON_PRIORITY = "core:1000,enterprise:50"

    # addon priority cache
    _addon_priority_map: Dict[str, int] = None

    @classmethod
    def load_addon_priority_map(cls) -> Dict[str, int]:
        """
        Load addon priority configuration from environment variables

        Returns:
            Dict[str, int]: mapping from addon name to priority (smaller number means higher priority)
        """
        if cls._addon_priority_map is not None:
            return cls._addon_priority_map

        # Read configuration from environment variable, use default if not set
        priority_config = os.getenv("ADDON_PRIORITY", cls.DEFAULT_ADDON_PRIORITY)

        priority_map = {}
        for item in priority_config.split(","):
            item = item.strip()
            if ":" in item:
                addon_name, priority_str = item.split(":", 1)
                try:
                    priority_map[addon_name.strip()] = int(priority_str.strip())
                except ValueError:
                    # Ignore invalid configuration
                    pass

        cls._addon_priority_map = priority_map
        return priority_map

    @classmethod
    def get_addon_priority(cls, bean_def: BeanDefinition) -> int:
        """
        Get the addon priority of a Bean

        Args:
            bean_def: Bean definition object

        Returns:
            int: addon priority value, smaller number means higher priority
            Returns default 99999 (lowest priority) if addon_tag is not configured or not found in config
        """
        priority_map = cls.load_addon_priority_map()

        # Get addon_tag from Bean's metadata
        addon_tag = bean_def.metadata.get("addon_tag")
        if not addon_tag:
            # No addon_tag, return lowest priority
            return 99999

        # Return configured priority, or default lowest priority if not configured
        return priority_map.get(addon_tag, 99999)

    @staticmethod
    def calculate_order_key(
        bean_def: BeanDefinition, is_direct_match: bool, mock_mode: bool = False
    ) -> Tuple[int, int, int, int, int]:
        """
        Calculate Bean's ordering key (extended version, includes addon priority)

        Args:
            bean_def: Bean definition object
            is_direct_match: Whether it's a direct match (True=direct match, False=implementation class match)
            mock_mode: Whether in Mock mode

        Returns:
            Tuple[int, int, int, int, int]: ordering key tuple
            Format: (addon_priority, mock_priority, match_priority, primary_priority, scope_priority)

        Priority rules:
            - addon_priority: retrieved from environment variable config, smaller number means higher priority
            - mock_priority: in mock mode, Mock Bean=0, Non-Mock Bean=1; in non-mock mode both are 0
            - match_priority: direct match=0, implementation class match=1
            - primary_priority: Primary Bean=0, Non-Primary Bean=1
            - scope_priority: Factory Bean=0, Non-Factory Bean=1
        """
        # 1. Addon priority (smaller number means higher priority)
        addon_priority = AddonBeanOrderStrategy.get_addon_priority(bean_def)

        # 2. Mock priority (only differentiated in Mock mode)
        if mock_mode:
            mock_priority = 0 if bean_def.is_mock else 1
        else:
            mock_priority = 0  # No distinction in non-Mock mode

        # 3. Matching method priority (direct match first)
        match_priority = 0 if is_direct_match else 1

        # 4. Primary priority (Primary first)
        primary_priority = 0 if bean_def.is_primary else 1

        # 5. Scope priority (Factory first)
        scope_priority = 0 if bean_def.scope == BeanScope.FACTORY else 1

        return (
            addon_priority,
            mock_priority,
            match_priority,
            primary_priority,
            scope_priority,
        )

    @staticmethod
    def sort_beans_with_context(
        bean_defs: List[BeanDefinition],
        direct_match_types: Set[Type],
        mock_mode: bool = False,
    ) -> List[BeanDefinition]:
        """
        Sort list of Bean definitions based on context information (extended version)

        Args:
            bean_defs: List of Bean definitions
            direct_match_types: Set of types that are direct matches
            mock_mode: Whether in Mock mode

        Returns:
            List[BeanDefinition]: Sorted list of Bean definitions

        Note:
            - In non-Mock mode, Mock Beans are filtered out and do not participate in sorting
            - In Mock mode, Mock Beans take precedence over non-Mock Beans
            - addon_tag has the highest priority, sorted according to environment variable configuration
        """
        # Filter out all Mock Beans in non-Mock mode
        if not mock_mode:
            bean_defs = [bd for bd in bean_defs if not bd.is_mock]

        # Calculate ordering key for each Bean, then sort by key
        sorted_beans = sorted(
            bean_defs,
            key=lambda bd: AddonBeanOrderStrategy.calculate_order_key(
                bean_def=bd,
                is_direct_match=bd.bean_type in direct_match_types,
                mock_mode=mock_mode,
            ),
        )
        return sorted_beans


# Automatically replace Bean ordering strategy when module loads
# Note: This is a temporary solution because the DI mechanism is not fully established yet
# Once addon mechanism is referenced, AddonBeanOrderStrategy will be automatically enabled
def _replace_strategy():
    """Automatically replace Bean ordering strategy"""
    try:
        DIContainer.replace_bean_order_strategy(AddonBeanOrderStrategy)
        logger.warning(
            "⚠️ Bean ordering strategy has been automatically replaced with AddonBeanOrderStrategy, supporting addon_tag priority"
        )
        logger.info(
            "  📌 Addon priority configuration: %s (environment variable: ADDON_PRIORITY)",
            AddonBeanOrderStrategy.load_addon_priority_map(),
        )
    except Exception as e:
        logger.error("Failed to replace Bean ordering strategy: %s", e)


# Execute automatic replacement
_replace_strategy()

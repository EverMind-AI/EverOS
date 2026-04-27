# -*- coding: utf-8 -*-
"""
cd /Users/admin/memsys_opensource
PYTHONPATH=/Users/admin/memsys_opensource/src python -m pytest src/core/addons/contrib/tests/test_addon_bean_order_strategy.py -v -s

Addon Bean sorting strategy test

Test the extension functionality of AddonBeanOrderStrategy, especially addon_tag priority
"""

import os
import pytest
from typing import Set, Type
from core.di.bean_definition import BeanDefinition, BeanScope
from core.addons.addonize.addon_bean_order_strategy import AddonBeanOrderStrategy


# ==================== Test helper classes ====================


class ServiceA:
    """Test service A"""

    pass


class ServiceB:
    """Test service B"""

    pass


class ServiceC:
    """Test service C"""

    pass


class ServiceD:
    """Test service D"""

    pass


# ==================== Test Addon priority configuration ====================


class TestAddonPriorityConfiguration:
    """Test loading of Addon priority configuration"""

    def setup_method(self):
        """Reset priority cache before each test"""
        AddonBeanOrderStrategy._addon_priority_map = None

    def teardown_method(self):
        """Restore environment variables after each test"""
        if "ADDON_PRIORITY" in os.environ:
            del os.environ["ADDON_PRIORITY"]
        AddonBeanOrderStrategy._addon_priority_map = None

    def test_load_default_priority_map(self):
        """Test loading default priority configuration"""
        # Without setting environment variable, should use default configuration
        priority_map = AddonBeanOrderStrategy.load_addon_priority_map()

        # Verify default configuration: core:1000, enterprise:50
        assert "core" in priority_map
        assert "enterprise" in priority_map
        assert priority_map["core"] == 1000
        assert priority_map["enterprise"] == 50

    def test_load_custom_priority_map(self):
        """Test loading custom priority configuration from environment variable"""
        # Set environment variable
        os.environ["ADDON_PRIORITY"] = "addon1:100,addon2:200,addon3:50"

        # Reload
        AddonBeanOrderStrategy._addon_priority_map = None
        priority_map = AddonBeanOrderStrategy.load_addon_priority_map()

        # Verify configuration
        assert priority_map["addon1"] == 100
        assert priority_map["addon2"] == 200
        assert priority_map["addon3"] == 50

    def test_priority_map_caching(self):
        """Test caching mechanism of priority configuration"""
        # First load
        priority_map1 = AddonBeanOrderStrategy.load_addon_priority_map()

        # Second load (should return from cache)
        priority_map2 = AddonBeanOrderStrategy.load_addon_priority_map()

        # Should be the same object
        assert priority_map1 is priority_map2

    def test_invalid_priority_config_ignored(self):
        """Test invalid priority configuration is ignored"""
        # Set environment variable with invalid values
        os.environ["ADDON_PRIORITY"] = "valid:100,invalid:abc,another:200"

        # Reload
        AddonBeanOrderStrategy._addon_priority_map = None
        priority_map = AddonBeanOrderStrategy.load_addon_priority_map()

        # Verify: valid configurations are loaded, invalid ones are ignored
        assert priority_map["valid"] == 100
        assert priority_map["another"] == 200
        assert "invalid" not in priority_map

    def test_priority_config_with_spaces(self):
        """Test configuration containing spaces"""
        # Set environment variable with spaces
        os.environ["ADDON_PRIORITY"] = " addon1 : 100 , addon2 : 200 "

        # Reload
        AddonBeanOrderStrategy._addon_priority_map = None
        priority_map = AddonBeanOrderStrategy.load_addon_priority_map()

        # Verify: spaces are properly handled
        assert priority_map["addon1"] == 100
        assert priority_map["addon2"] == 200


# ==================== Test getting Addon priority ====================


class TestGetAddonPriority:
    """Test getting Addon priority of Bean"""

    def setup_method(self):
        """Reset configuration before each test"""
        AddonBeanOrderStrategy._addon_priority_map = None
        os.environ["ADDON_PRIORITY"] = "core:1000,enterprise:50,custom:200"

    def teardown_method(self):
        """Clean up after each test"""
        if "ADDON_PRIORITY" in os.environ:
            del os.environ["ADDON_PRIORITY"]
        AddonBeanOrderStrategy._addon_priority_map = None

    def test_get_priority_with_addon_tag(self):
        """Test getting priority of Bean with addon_tag"""
        # Create Bean with addon_tag
        bean_core = BeanDefinition(ServiceA, metadata={"addon_tag": "core"})
        bean_enterprise = BeanDefinition(ServiceB, metadata={"addon_tag": "enterprise"})
        bean_custom = BeanDefinition(ServiceC, metadata={"addon_tag": "custom"})

        # Get priority
        priority_core = AddonBeanOrderStrategy.get_addon_priority(bean_core)
        priority_enterprise = AddonBeanOrderStrategy.get_addon_priority(bean_enterprise)
        priority_custom = AddonBeanOrderStrategy.get_addon_priority(bean_custom)

        # Verify
        assert priority_core == 1000
        assert priority_enterprise == 50
        assert priority_custom == 200

        # Verify priority order: enterprise < custom < core
        assert priority_enterprise < priority_custom < priority_core

    def test_get_priority_without_addon_tag(self):
        """Test getting priority of Bean without addon_tag"""
        # Create Bean without addon_tag
        bean_no_tag = BeanDefinition(ServiceA)

        # Get priority
        priority = AddonBeanOrderStrategy.get_addon_priority(bean_no_tag)

        # Verify: returns lowest priority
        assert priority == 99999

    def test_get_priority_with_unknown_addon_tag(self):
        """Test getting priority of unconfigured addon_tag"""
        # Create Bean with unconfigured addon_tag
        bean_unknown = BeanDefinition(ServiceA, metadata={"addon_tag": "unknown_addon"})

        # Get priority
        priority = AddonBeanOrderStrategy.get_addon_priority(bean_unknown)

        # Verify: returns lowest priority
        assert priority == 99999

    def test_get_priority_with_empty_addon_tag(self):
        """Test getting priority of empty addon_tag"""
        # Create Bean with empty addon_tag
        bean_empty = BeanDefinition(ServiceA, metadata={"addon_tag": ""})

        # Get priority
        priority = AddonBeanOrderStrategy.get_addon_priority(bean_empty)

        # Verify: returns lowest priority
        assert priority == 99999


# ==================== Test calculating sort key ====================


class TestCalculateOrderKey:
    """Test calculating extended sort key (including addon priority)"""

    def setup_method(self):
        """Reset configuration before each test"""
        AddonBeanOrderStrategy._addon_priority_map = None
        os.environ["ADDON_PRIORITY"] = "enterprise:50,core:1000"

    def teardown_method(self):
        """Clean up after each test"""
        if "ADDON_PRIORITY" in os.environ:
            del os.environ["ADDON_PRIORITY"]
        AddonBeanOrderStrategy._addon_priority_map = None

    def test_order_key_includes_addon_priority(self):
        """Test sort key includes addon priority"""
        # Create Bean with addon_tag
        bean_enterprise = BeanDefinition(ServiceA, metadata={"addon_tag": "enterprise"})
        bean_core = BeanDefinition(ServiceB, metadata={"addon_tag": "core"})

        # Calculate sort key
        key_enterprise = AddonBeanOrderStrategy.calculate_order_key(
            bean_enterprise, is_direct_match=True, mock_mode=False
        )
        key_core = AddonBeanOrderStrategy.calculate_order_key(
            bean_core, is_direct_match=True, mock_mode=False
        )

        # Verify: sort key is 5-tuple (addon, mock, match, primary, scope)
        assert len(key_enterprise) == 5
        assert len(key_core) == 5

        # Verify: first element is addon priority
        assert key_enterprise[0] == 50  # enterprise
        assert key_core[0] == 1000  # core

        # Verify: enterprise priority higher than core
        assert key_enterprise < key_core

    def test_addon_priority_overrides_other_priorities(self):
        """Test addon priority overrides all other priorities"""
        # Create two Beans:
        # Bean1: enterprise addon + non-Primary + non-Factory
        # Bean2: core addon + Primary + Factory
        bean1 = BeanDefinition(
            ServiceA,
            is_primary=False,
            scope=BeanScope.SINGLETON,
            metadata={"addon_tag": "enterprise"},
        )
        bean2 = BeanDefinition(
            ServiceB,
            is_primary=True,
            scope=BeanScope.FACTORY,
            metadata={"addon_tag": "core"},
        )

        # Calculate sort key
        key1 = AddonBeanOrderStrategy.calculate_order_key(
            bean1, is_direct_match=True, mock_mode=False
        )
        key2 = AddonBeanOrderStrategy.calculate_order_key(
            bean2, is_direct_match=True, mock_mode=False
        )

        # Verify: even though bean2 has Primary+Factory, bean1 comes first due to higher addon priority
        assert key1 < key2

    def test_order_key_with_mock_mode(self):
        """Test sort key in Mock mode"""
        # Create Mock Bean and non-Mock Bean (both enterprise addon)
        mock_bean = BeanDefinition(
            ServiceA, is_mock=True, metadata={"addon_tag": "enterprise"}
        )
        normal_bean = BeanDefinition(
            ServiceB, is_mock=False, metadata={"addon_tag": "enterprise"}
        )

        # Calculate sort key (Mock mode)
        mock_key = AddonBeanOrderStrategy.calculate_order_key(
            mock_bean, is_direct_match=True, mock_mode=True
        )
        normal_key = AddonBeanOrderStrategy.calculate_order_key(
            normal_bean, is_direct_match=True, mock_mode=True
        )

        # Verify: same addon priority, Mock has higher priority
        assert mock_key[0] == normal_key[0]  # same addon priority
        assert mock_key[1] < normal_key[1]  # different mock priority
        assert mock_key < normal_key

    def test_order_key_backward_compatible(self):
        """Test backward compatibility: sorting follows original logic when no addon_tag"""
        # Create two Beans: neither has addon_tag, one Primary and one non-Primary
        primary_bean = BeanDefinition(ServiceA, is_primary=True)
        normal_bean = BeanDefinition(ServiceB, is_primary=False)

        # Calculate sort key
        primary_key = AddonBeanOrderStrategy.calculate_order_key(
            primary_bean, is_direct_match=True, mock_mode=False
        )
        normal_key = AddonBeanOrderStrategy.calculate_order_key(
            normal_bean, is_direct_match=True, mock_mode=False
        )

        # Verify: addon priority is 99999 (lowest) for both
        assert primary_key[0] == 99999
        assert normal_key[0] == 99999

        # Verify: Primary Bean has higher priority
        assert primary_key < normal_key


# ==================== Test Bean list sorting ====================


class TestSortBeansWithContext:
    """Test sorting Bean list with context (including addon priority)"""

    def setup_method(self):
        """Reset configuration before each test"""
        AddonBeanOrderStrategy._addon_priority_map = None
        os.environ["ADDON_PRIORITY"] = "enterprise:50,core:1000,custom:500"

    def teardown_method(self):
        """Clean up after each test"""
        if "ADDON_PRIORITY" in os.environ:
            del os.environ["ADDON_PRIORITY"]
        AddonBeanOrderStrategy._addon_priority_map = None

    def test_sort_by_addon_priority(self):
        """Test sorting by addon priority"""
        # Create Beans with different addons
        bean_defs = [
            BeanDefinition(
                ServiceA, bean_name="core_bean", metadata={"addon_tag": "core"}
            ),
            BeanDefinition(
                ServiceB,
                bean_name="enterprise_bean",
                metadata={"addon_tag": "enterprise"},
            ),
            BeanDefinition(
                ServiceC, bean_name="custom_bean", metadata={"addon_tag": "custom"}
            ),
            BeanDefinition(ServiceD, bean_name="no_addon_bean"),
        ]

        # Sort
        sorted_beans = AddonBeanOrderStrategy.sort_beans_with_context(
            bean_defs=bean_defs,
            direct_match_types={ServiceA, ServiceB, ServiceC, ServiceD},
            mock_mode=False,
        )

        # Verify sort order: enterprise(50) < custom(500) < core(1000) < no_addon(99999)
        assert sorted_beans[0].bean_name == "enterprise_bean"
        assert sorted_beans[1].bean_name == "custom_bean"
        assert sorted_beans[2].bean_name == "core_bean"
        assert sorted_beans[3].bean_name == "no_addon_bean"

    def test_addon_priority_with_primary_and_scope(self):
        """Test combination of addon priority with Primary and Scope"""
        # Create Beans with various combinations
        bean_defs = [
            # Highest priority: enterprise + Primary + Factory
            BeanDefinition(
                ServiceA,
                bean_name="enterprise_primary_factory",
                is_primary=True,
                scope=BeanScope.FACTORY,
                metadata={"addon_tag": "enterprise"},
            ),
            # Second highest: enterprise + non-Primary + non-Factory
            BeanDefinition(
                ServiceB,
                bean_name="enterprise_normal",
                is_primary=False,
                scope=BeanScope.SINGLETON,
                metadata={"addon_tag": "enterprise"},
            ),
            # Medium: core + Primary + Factory
            BeanDefinition(
                ServiceC,
                bean_name="core_primary_factory",
                is_primary=True,
                scope=BeanScope.FACTORY,
                metadata={"addon_tag": "core"},
            ),
            # Lowest: no addon + Primary + Factory
            BeanDefinition(
                ServiceD,
                bean_name="no_addon_primary_factory",
                is_primary=True,
                scope=BeanScope.FACTORY,
            ),
        ]

        # Sort
        sorted_beans = AddonBeanOrderStrategy.sort_beans_with_context(
            bean_defs=bean_defs,
            direct_match_types={ServiceA, ServiceB, ServiceC, ServiceD},
            mock_mode=False,
        )

        # Verify: addon priority is highest, even if other attributes differ
        assert sorted_beans[0].bean_name == "enterprise_primary_factory"
        assert sorted_beans[1].bean_name == "enterprise_normal"
        assert sorted_beans[2].bean_name == "core_primary_factory"
        assert sorted_beans[3].bean_name == "no_addon_primary_factory"

    def test_same_addon_priority_then_by_primary(self):
        """Test sorting by Primary when addon priority is the same"""
        # Create Beans with same addon
        bean_defs = [
            BeanDefinition(
                ServiceA,
                bean_name="enterprise_normal",
                is_primary=False,
                metadata={"addon_tag": "enterprise"},
            ),
            BeanDefinition(
                ServiceB,
                bean_name="enterprise_primary",
                is_primary=True,
                metadata={"addon_tag": "enterprise"},
            ),
        ]

        # Sort
        sorted_beans = AddonBeanOrderStrategy.sort_beans_with_context(
            bean_defs=bean_defs,
            direct_match_types={ServiceA, ServiceB},
            mock_mode=False,
        )

        # Verify: when addon priority is the same, Primary takes precedence
        assert sorted_beans[0].bean_name == "enterprise_primary"
        assert sorted_beans[1].bean_name == "enterprise_normal"

    def test_filter_mock_beans_in_normal_mode(self):
        """Test filtering Mock Beans in non-Mock mode (addon version)"""
        # Create list containing Mock Bean
        bean_defs = [
            BeanDefinition(
                ServiceA,
                bean_name="enterprise_mock",
                is_mock=True,
                metadata={"addon_tag": "enterprise"},
            ),
            BeanDefinition(
                ServiceB,
                bean_name="enterprise_normal",
                is_mock=False,
                metadata={"addon_tag": "enterprise"},
            ),
            BeanDefinition(
                ServiceC,
                bean_name="core_mock",
                is_mock=True,
                metadata={"addon_tag": "core"},
            ),
        ]

        # Sort (non-Mock mode)
        sorted_beans = AddonBeanOrderStrategy.sort_beans_with_context(
            bean_defs=bean_defs,
            direct_match_types={ServiceA, ServiceB, ServiceC},
            mock_mode=False,
        )

        # Verify: only non-Mock Beans are retained
        assert len(sorted_beans) == 1
        assert sorted_beans[0].bean_name == "enterprise_normal"

    def test_mock_beans_priority_in_mock_mode(self):
        """Test Mock Beans have priority in Mock mode (addon version)"""
        # Create mixed Bean list
        bean_defs = [
            BeanDefinition(
                ServiceA,
                bean_name="core_normal",
                is_mock=False,
                metadata={"addon_tag": "core"},
            ),
            BeanDefinition(
                ServiceB,
                bean_name="enterprise_mock",
                is_mock=True,
                metadata={"addon_tag": "enterprise"},
            ),
            BeanDefinition(
                ServiceC,
                bean_name="enterprise_normal",
                is_mock=False,
                metadata={"addon_tag": "enterprise"},
            ),
        ]

        # Sort (Mock mode)
        sorted_beans = AddonBeanOrderStrategy.sort_beans_with_context(
            bean_defs=bean_defs,
            direct_match_types={ServiceA, ServiceB, ServiceC},
            mock_mode=True,
        )

        # Verify: Mock has priority within same addon, but different addons still follow addon priority
        assert sorted_beans[0].bean_name == "enterprise_mock"  # enterprise + mock
        assert sorted_beans[1].bean_name == "enterprise_normal"  # enterprise + normal
        assert sorted_beans[2].bean_name == "core_normal"  # core + normal

    def test_empty_bean_list(self):
        """Test empty Bean list"""
        sorted_beans = AddonBeanOrderStrategy.sort_beans_with_context(
            bean_defs=[], direct_match_types=set(), mock_mode=False
        )
        assert sorted_beans == []

    def test_single_bean(self):
        """Test single Bean"""
        bean_defs = [BeanDefinition(ServiceA, metadata={"addon_tag": "enterprise"})]
        sorted_beans = AddonBeanOrderStrategy.sort_beans_with_context(
            bean_defs=bean_defs, direct_match_types={ServiceA}, mock_mode=False
        )
        assert len(sorted_beans) == 1


# ==================== Complex scenario tests ====================


class TestComplexScenarios:
    """Test complex scenarios"""

    def setup_method(self):
        """Reset configuration before each test"""
        AddonBeanOrderStrategy._addon_priority_map = None
        os.environ["ADDON_PRIORITY"] = "enterprise:10,plugin1:50,plugin2:100,core:1000"

    def teardown_method(self):
        """Clean up after each test"""
        if "ADDON_PRIORITY" in os.environ:
            del os.environ["ADDON_PRIORITY"]
        AddonBeanOrderStrategy._addon_priority_map = None

    def test_multi_addon_multi_implementation(self):
        """Test multiple implementations from multiple addons"""
        # Simulate real scenario: multiple addons provide implementations for the same interface
        bean_defs = [
            # Implementation provided by enterprise (Primary + Factory)
            BeanDefinition(
                ServiceA,
                bean_name="enterprise_impl",
                is_primary=True,
                scope=BeanScope.FACTORY,
                metadata={"addon_tag": "enterprise"},
            ),
            # Implementation provided by plugin1 (Primary)
            BeanDefinition(
                ServiceB,
                bean_name="plugin1_impl",
                is_primary=True,
                metadata={"addon_tag": "plugin1"},
            ),
            # Implementation provided by plugin2
            BeanDefinition(
                ServiceC, bean_name="plugin2_impl", metadata={"addon_tag": "plugin2"}
            ),
            # Implementation provided by core (Factory)
            BeanDefinition(
                ServiceD,
                bean_name="core_impl",
                scope=BeanScope.FACTORY,
                metadata={"addon_tag": "core"},
            ),
        ]

        # Sort
        sorted_beans = AddonBeanOrderStrategy.sort_beans_with_context(
            bean_defs=bean_defs,
            direct_match_types={ServiceA, ServiceB, ServiceC, ServiceD},
            mock_mode=False,
        )

        # Verify: sorted by addon priority
        assert sorted_beans[0].bean_name == "enterprise_impl"  # 10
        assert sorted_beans[1].bean_name == "plugin1_impl"  # 50
        assert sorted_beans[2].bean_name == "plugin2_impl"  # 100
        assert sorted_beans[3].bean_name == "core_impl"  # 1000

    def test_addon_override_scenario(self):
        """Test addon override scenario: high priority addon overrides low priority addon"""
        # Create Beans: enterprise overrides core implementation
        bean_defs = [
            BeanDefinition(
                ServiceA,
                bean_name="core_default",
                is_primary=True,
                scope=BeanScope.FACTORY,
                metadata={"addon_tag": "core"},
            ),
            BeanDefinition(
                ServiceA,
                bean_name="enterprise_override",
                is_primary=False,
                scope=BeanScope.SINGLETON,
                metadata={"addon_tag": "enterprise"},
            ),
        ]

        # Sort
        sorted_beans = AddonBeanOrderStrategy.sort_beans_with_context(
            bean_defs=bean_defs, direct_match_types={ServiceA}, mock_mode=False
        )

        # Verify: enterprise comes first despite not being Primary or Factory, due to higher addon priority
        assert sorted_beans[0].bean_name == "enterprise_override"
        assert sorted_beans[1].bean_name == "core_default"

    def test_all_attributes_combination(self):
        """Test combination of all attributes"""
        # Create 16 Beans covering all possible combinations
        bean_defs = []
        counter = 0

        for addon_tag in ["enterprise", "core", None]:
            for is_mock in [False, True]:
                for is_primary in [False, True]:
                    for is_factory in [False, True]:
                        metadata = {"addon_tag": addon_tag} if addon_tag else {}
                        bean = BeanDefinition(
                            ServiceA,
                            bean_name=f"bean_{counter}",
                            is_mock=is_mock,
                            is_primary=is_primary,
                            scope=(
                                BeanScope.FACTORY if is_factory else BeanScope.SINGLETON
                            ),
                            metadata=metadata,
                        )
                        bean_defs.append(bean)
                        counter += 1

        # Sort (non-Mock mode)
        sorted_beans = AddonBeanOrderStrategy.sort_beans_with_context(
            bean_defs=bean_defs, direct_match_types={ServiceA}, mock_mode=False
        )

        # Verify: in non-Mock mode, Mock Beans are filtered out
        assert all(not bean.is_mock for bean in sorted_beans)

        # Verify: first Bean should be enterprise addon
        assert sorted_beans[0].metadata.get("addon_tag") == "enterprise"

        # Verify: last Bean should have no addon_tag
        assert sorted_beans[-1].metadata.get("addon_tag") is None


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s", "--tb=short"])

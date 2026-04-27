# -*- coding: utf-8 -*-
"""
cd /Users/admin/memsys_opensource
PYTHONPATH=/Users/admin/memsys_opensource/src python -m pytest src/core/di/tests/test_bean_order_strategy.py -v

BeanOrderStrategy Test Module

Comprehensively test various scenarios of Bean sorting strategy, including:
- Various priority calculations of the calculate_order_key method
- Comprehensive sorting and filtering logic of the sort_beans_with_context method
- Simple sorting logic of the sort_beans method
"""

import pytest
from typing import Set, Type
from core.di.bean_definition import BeanDefinition, BeanScope
from core.di.bean_order_strategy import BeanOrderStrategy


# ==================== Test Helper Classes ====================


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


# ==================== Test calculate_order_key Method ====================


class TestCalculateOrderKey:
    """Test calculate_order_key method"""

    def test_mock_priority_in_mock_mode(self):
        """Test in Mock mode, Mock Bean has higher priority than non-Mock Bean"""
        # Create Mock Bean and non-Mock Bean
        mock_bean = BeanDefinition(ServiceA, is_mock=True)
        normal_bean = BeanDefinition(ServiceA, is_mock=False)

        # Calculate sort key (Mock mode, direct match)
        mock_key = BeanOrderStrategy.calculate_order_key(
            mock_bean, is_direct_match=True, mock_mode=True
        )
        normal_key = BeanOrderStrategy.calculate_order_key(
            normal_bean, is_direct_match=True, mock_mode=True
        )

        # Verify: Mock Bean's mock_priority=0, non-Mock Bean's mock_priority=1
        assert mock_key[0] == 0  # Mock Bean's mock priority
        assert normal_key[0] == 1  # Non-Mock Bean's mock priority
        assert mock_key < normal_key  # Mock Bean has higher priority

    def test_mock_priority_in_normal_mode(self):
        """Test in non-Mock mode, both Mock and non-Mock Beans have mock_priority=0"""
        # Create Mock Bean and non-Mock Bean
        mock_bean = BeanDefinition(ServiceA, is_mock=True)
        normal_bean = BeanDefinition(ServiceA, is_mock=False)

        # Calculate sort key (non-Mock mode, direct match)
        mock_key = BeanOrderStrategy.calculate_order_key(
            mock_bean, is_direct_match=True, mock_mode=False
        )
        normal_key = BeanOrderStrategy.calculate_order_key(
            normal_bean, is_direct_match=True, mock_mode=False
        )

        # Verify: In non-Mock mode, both have mock_priority=0
        assert mock_key[0] == 0
        assert normal_key[0] == 0

    def test_direct_match_priority(self):
        """Test direct match has higher priority than implementation class match"""
        bean = BeanDefinition(ServiceA)

        # Calculate sort key
        direct_match_key = BeanOrderStrategy.calculate_order_key(
            bean, is_direct_match=True, mock_mode=False
        )
        impl_match_key = BeanOrderStrategy.calculate_order_key(
            bean, is_direct_match=False, mock_mode=False
        )

        # Verify: direct match has match_priority=0, implementation match has match_priority=1
        assert direct_match_key[1] == 0  # Direct match
        assert impl_match_key[1] == 1  # Implementation match
        assert direct_match_key < impl_match_key  # Direct match has higher priority

    def test_primary_priority(self):
        """Test Primary Bean has higher priority than non-Primary Bean"""
        primary_bean = BeanDefinition(ServiceA, is_primary=True)
        normal_bean = BeanDefinition(ServiceA, is_primary=False)

        # Calculate sort key (direct match, non-Mock mode)
        primary_key = BeanOrderStrategy.calculate_order_key(
            primary_bean, is_direct_match=True, mock_mode=False
        )
        normal_key = BeanOrderStrategy.calculate_order_key(
            normal_bean, is_direct_match=True, mock_mode=False
        )

        # Verify: Primary Bean's primary_priority=0, non-Primary Bean's primary_priority=1
        assert primary_key[2] == 0  # Primary Bean
        assert normal_key[2] == 1  # Non-Primary Bean
        assert primary_key < normal_key  # Primary Bean has higher priority

    def test_factory_scope_priority(self):
        """Test Factory Bean has higher priority than non-Factory Bean"""
        factory_bean = BeanDefinition(ServiceA, scope=BeanScope.FACTORY)
        singleton_bean = BeanDefinition(ServiceA, scope=BeanScope.SINGLETON)
        prototype_bean = BeanDefinition(ServiceA, scope=BeanScope.PROTOTYPE)

        # Calculate sort key (direct match, non-Mock mode)
        factory_key = BeanOrderStrategy.calculate_order_key(
            factory_bean, is_direct_match=True, mock_mode=False
        )
        singleton_key = BeanOrderStrategy.calculate_order_key(
            singleton_bean, is_direct_match=True, mock_mode=False
        )
        prototype_key = BeanOrderStrategy.calculate_order_key(
            prototype_bean, is_direct_match=True, mock_mode=False
        )

        # Verify: Factory Bean's scope_priority=0, others' scope_priority=1
        assert factory_key[3] == 0  # Factory Bean
        assert singleton_key[3] == 1  # Singleton Bean
        assert prototype_key[3] == 1  # Prototype Bean
        assert factory_key < singleton_key  # Factory Bean has higher priority
        assert factory_key < prototype_key  # Factory Bean has higher priority

    def test_comprehensive_priority_ordering(self):
        """Test comprehensive priority ordering: mock > match > primary > scope"""
        # In Mock mode, create Beans with various combinations
        # Highest priority: Mock + direct match + Primary + Factory
        bean1 = BeanDefinition(
            ServiceA, is_mock=True, is_primary=True, scope=BeanScope.FACTORY
        )
        key1 = BeanOrderStrategy.calculate_order_key(
            bean1, is_direct_match=True, mock_mode=True
        )

        # Second highest priority: Mock + direct match + Primary + non-Factory
        bean2 = BeanDefinition(
            ServiceA, is_mock=True, is_primary=True, scope=BeanScope.SINGLETON
        )
        key2 = BeanOrderStrategy.calculate_order_key(
            bean2, is_direct_match=True, mock_mode=True
        )

        # Third priority: Mock + direct match + non-Primary + Factory
        bean3 = BeanDefinition(
            ServiceA, is_mock=True, is_primary=False, scope=BeanScope.FACTORY
        )
        key3 = BeanOrderStrategy.calculate_order_key(
            bean3, is_direct_match=True, mock_mode=True
        )

        # Fourth priority: Mock + implementation match + Primary + Factory
        bean4 = BeanDefinition(
            ServiceA, is_mock=True, is_primary=True, scope=BeanScope.FACTORY
        )
        key4 = BeanOrderStrategy.calculate_order_key(
            bean4, is_direct_match=False, mock_mode=True
        )

        # Fifth priority: non-Mock + direct match + Primary + Factory
        bean5 = BeanDefinition(
            ServiceA, is_mock=False, is_primary=True, scope=BeanScope.FACTORY
        )
        key5 = BeanOrderStrategy.calculate_order_key(
            bean5, is_direct_match=True, mock_mode=True
        )

        # Verify sort order - key1 has highest priority
        assert (
            key1 < key2
        )  # Factory has higher priority than non-Factory (same Mock+direct+Primary)
        assert (
            key1 < key3
        )  # Primary has higher priority than non-Primary (same Mock+direct+Factory)
        assert (
            key1 < key4
        )  # Direct match has higher priority than implementation match (same Mock+Primary+Factory)
        assert (
            key1 < key5
        )  # Mock has higher priority than non-Mock (same direct+Primary+Factory)

        # Verify secondary priorities - earlier positions in the tuple have higher weight
        assert (
            key2 < key3
        )  # (0,0,0,1) < (0,0,1,0): scope difference vs primary difference
        assert (
            key3 < key4
        )  # (0,0,1,0) < (0,1,0,0): primary difference vs match difference
        assert key4 < key5  # (0,1,0,0) < (1,0,0,0): match difference vs mock difference


# ==================== Test sort_beans_with_context Method ====================


class TestSortBeansWithContext:
    """Test sort_beans_with_context method"""

    def test_filter_mock_beans_in_normal_mode(self):
        """Test in non-Mock mode, Mock Beans are filtered out"""
        # Create list containing Mock and non-Mock Beans
        bean_defs = [
            BeanDefinition(ServiceA, bean_name="mock_a", is_mock=True),
            BeanDefinition(ServiceB, bean_name="normal_b", is_mock=False),
            BeanDefinition(ServiceC, bean_name="mock_c", is_mock=True),
            BeanDefinition(ServiceD, bean_name="normal_d", is_mock=False),
        ]

        # Sort (non-Mock mode)
        sorted_beans = BeanOrderStrategy.sort_beans_with_context(
            bean_defs=bean_defs, direct_match_types=set(), mock_mode=False
        )

        # Verify: only non-Mock Beans remain
        assert len(sorted_beans) == 2
        assert all(not bd.is_mock for bd in sorted_beans)
        assert {bd.bean_name for bd in sorted_beans} == {"normal_b", "normal_d"}

    def test_keep_mock_beans_in_mock_mode(self):
        """Test in Mock mode, Mock Beans are retained and prioritized"""
        # Create list containing Mock and non-Mock Beans
        bean_defs = [
            BeanDefinition(ServiceA, bean_name="normal_a", is_mock=False),
            BeanDefinition(ServiceB, bean_name="mock_b", is_mock=True),
            BeanDefinition(ServiceC, bean_name="mock_c", is_mock=True),
            BeanDefinition(ServiceD, bean_name="normal_d", is_mock=False),
        ]

        # Sort (Mock mode, all are direct matches)
        sorted_beans = BeanOrderStrategy.sort_beans_with_context(
            bean_defs=bean_defs,
            direct_match_types={ServiceA, ServiceB, ServiceC, ServiceD},
            mock_mode=True,
        )

        # Verify: all Beans are retained
        assert len(sorted_beans) == 4

        # Verify: Mock Beans come first, non-Mock Beans come last
        assert sorted_beans[0].is_mock
        assert sorted_beans[1].is_mock
        assert not sorted_beans[2].is_mock
        assert not sorted_beans[3].is_mock

    def test_direct_match_types_sorting(self):
        """Test priority sorting of direct match types"""
        # Create Bean list
        bean_defs = [
            BeanDefinition(ServiceA, bean_name="impl_a"),
            BeanDefinition(ServiceB, bean_name="direct_b"),
            BeanDefinition(ServiceC, bean_name="impl_c"),
            BeanDefinition(ServiceD, bean_name="direct_d"),
        ]

        # Set ServiceB and ServiceD as direct match types
        direct_match_types = {ServiceB, ServiceD}

        # Sort (non-Mock mode)
        sorted_beans = BeanOrderStrategy.sort_beans_with_context(
            bean_defs=bean_defs, direct_match_types=direct_match_types, mock_mode=False
        )

        # Verify: direct match Beans come first
        assert sorted_beans[0].bean_type in direct_match_types
        assert sorted_beans[1].bean_type in direct_match_types
        assert sorted_beans[2].bean_type not in direct_match_types
        assert sorted_beans[3].bean_type not in direct_match_types

    def test_primary_beans_sorting(self):
        """Test priority sorting of Primary Beans"""
        # Create Bean list (all direct matches, non-Mock mode)
        bean_defs = [
            BeanDefinition(ServiceA, bean_name="normal_a", is_primary=False),
            BeanDefinition(ServiceB, bean_name="primary_b", is_primary=True),
            BeanDefinition(ServiceC, bean_name="normal_c", is_primary=False),
            BeanDefinition(ServiceD, bean_name="primary_d", is_primary=True),
        ]

        # Sort (all are direct matches)
        sorted_beans = BeanOrderStrategy.sort_beans_with_context(
            bean_defs=bean_defs,
            direct_match_types={ServiceA, ServiceB, ServiceC, ServiceD},
            mock_mode=False,
        )

        # Verify: Primary Beans come first
        assert sorted_beans[0].is_primary
        assert sorted_beans[1].is_primary
        assert not sorted_beans[2].is_primary
        assert not sorted_beans[3].is_primary

    def test_factory_scope_sorting(self):
        """Test priority sorting of Factory Beans"""
        # Create Bean list (all direct matches, non-Primary, non-Mock)
        bean_defs = [
            BeanDefinition(
                ServiceA, bean_name="singleton_a", scope=BeanScope.SINGLETON
            ),
            BeanDefinition(ServiceB, bean_name="factory_b", scope=BeanScope.FACTORY),
            BeanDefinition(
                ServiceC, bean_name="prototype_c", scope=BeanScope.PROTOTYPE
            ),
            BeanDefinition(ServiceD, bean_name="factory_d", scope=BeanScope.FACTORY),
        ]

        # Sort (all are direct matches)
        sorted_beans = BeanOrderStrategy.sort_beans_with_context(
            bean_defs=bean_defs,
            direct_match_types={ServiceA, ServiceB, ServiceC, ServiceD},
            mock_mode=False,
        )

        # Verify: Factory Beans come first
        assert sorted_beans[0].scope == BeanScope.FACTORY
        assert sorted_beans[1].scope == BeanScope.FACTORY
        assert sorted_beans[2].scope in {BeanScope.SINGLETON, BeanScope.PROTOTYPE}
        assert sorted_beans[3].scope in {BeanScope.SINGLETON, BeanScope.PROTOTYPE}

    def test_comprehensive_sorting(self):
        """Test comprehensive sorting scenario: mock + match + primary + scope"""
        # Create Bean list with various combinations
        bean_defs = [
            # Lowest priority: non-direct match + non-Primary + non-Factory
            BeanDefinition(
                ServiceA,
                bean_name="lowest",
                is_primary=False,
                scope=BeanScope.SINGLETON,
            ),
            # Medium priority: direct match + non-Primary + non-Factory
            BeanDefinition(
                ServiceB,
                bean_name="medium1",
                is_primary=False,
                scope=BeanScope.SINGLETON,
            ),
            # Higher priority: direct match + Primary + non-Factory
            BeanDefinition(
                ServiceC, bean_name="high1", is_primary=True, scope=BeanScope.SINGLETON
            ),
            # Highest priority: direct match + Primary + Factory
            BeanDefinition(
                ServiceD, bean_name="highest", is_primary=True, scope=BeanScope.FACTORY
            ),
            # Second highest priority: direct match + non-Primary + Factory
            BeanDefinition(
                ServiceA, bean_name="high2", is_primary=False, scope=BeanScope.FACTORY
            ),
        ]

        # Set direct match types (ServiceB, ServiceC, ServiceD, but not ServiceA)
        direct_match_types = {ServiceB, ServiceC, ServiceD}

        # Sort (non-Mock mode)
        sorted_beans = BeanOrderStrategy.sort_beans_with_context(
            bean_defs=bean_defs, direct_match_types=direct_match_types, mock_mode=False
        )

        # Verify sort order
        assert sorted_beans[0].bean_name == "highest"  # direct+Primary+Factory
        assert sorted_beans[1].bean_name in {
            "high1",
            "high2",
        }  # direct+Primary+non-Factory or direct+non-Primary+Factory
        assert (
            sorted_beans[4].bean_name == "lowest"
        )  # non-direct+non-Primary+non-Factory

    def test_empty_list(self):
        """Test empty list"""
        sorted_beans = BeanOrderStrategy.sort_beans_with_context(
            bean_defs=[], direct_match_types=set(), mock_mode=False
        )
        assert sorted_beans == []

    def test_single_bean(self):
        """Test single Bean"""
        bean_defs = [BeanDefinition(ServiceA, bean_name="single")]
        sorted_beans = BeanOrderStrategy.sort_beans_with_context(
            bean_defs=bean_defs, direct_match_types={ServiceA}, mock_mode=False
        )
        assert len(sorted_beans) == 1
        assert sorted_beans[0].bean_name == "single"

    def test_all_mock_beans_in_normal_mode(self):
        """Test when all Beans are Mock Beans in non-Mock mode"""
        bean_defs = [
            BeanDefinition(ServiceA, bean_name="mock_a", is_mock=True),
            BeanDefinition(ServiceB, bean_name="mock_b", is_mock=True),
        ]

        sorted_beans = BeanOrderStrategy.sort_beans_with_context(
            bean_defs=bean_defs, direct_match_types=set(), mock_mode=False
        )

        # Verify: all Mock Beans are filtered out, result is empty
        assert sorted_beans == []

    def test_complex_mock_mode_sorting(self):
        """Test complex sorting in Mock mode"""
        bean_defs = [
            # non-Mock + non-direct + non-Primary + non-Factory (lowest)
            BeanDefinition(
                ServiceA,
                bean_name="lowest",
                is_mock=False,
                is_primary=False,
                scope=BeanScope.SINGLETON,
            ),
            # Mock + non-direct + non-Primary + non-Factory (medium)
            BeanDefinition(
                ServiceB,
                bean_name="medium",
                is_mock=True,
                is_primary=False,
                scope=BeanScope.SINGLETON,
            ),
            # Mock + direct + non-Primary + non-Factory (higher)
            BeanDefinition(
                ServiceC,
                bean_name="high",
                is_mock=True,
                is_primary=False,
                scope=BeanScope.SINGLETON,
            ),
            # Mock + direct + Primary + Factory (highest)
            BeanDefinition(
                ServiceD,
                bean_name="highest",
                is_mock=True,
                is_primary=True,
                scope=BeanScope.FACTORY,
            ),
        ]

        # Sort (Mock mode, ServiceC and ServiceD are direct matches)
        sorted_beans = BeanOrderStrategy.sort_beans_with_context(
            bean_defs=bean_defs, direct_match_types={ServiceC, ServiceD}, mock_mode=True
        )

        # Verify sort order
        assert sorted_beans[0].bean_name == "highest"  # Mock+direct+Primary+Factory
        assert (
            sorted_beans[1].bean_name == "high"
        )  # Mock+direct+non-Primary+non-Factory
        assert (
            sorted_beans[2].bean_name == "medium"
        )  # Mock+non-direct+non-Primary+non-Factory
        assert (
            sorted_beans[3].bean_name == "lowest"
        )  # non-Mock+non-direct+non-Primary+non-Factory


# ==================== Test sort_beans Method ====================


class TestSortBeans:
    """Test sort_beans method (simple sorting)"""

    def test_primary_priority_simple(self):
        """Test Primary Bean has higher priority than non-Primary Bean (simple sorting)"""
        bean_defs = [
            BeanDefinition(ServiceA, bean_name="normal_a", is_primary=False),
            BeanDefinition(ServiceB, bean_name="primary_b", is_primary=True),
            BeanDefinition(ServiceC, bean_name="normal_c", is_primary=False),
            BeanDefinition(ServiceD, bean_name="primary_d", is_primary=True),
        ]

        sorted_beans = BeanOrderStrategy.sort_beans(bean_defs)

        # Verify: Primary Beans come first
        assert sorted_beans[0].is_primary
        assert sorted_beans[1].is_primary
        assert not sorted_beans[2].is_primary
        assert not sorted_beans[3].is_primary

    def test_factory_scope_priority_simple(self):
        """Test Factory Bean has higher priority than non-Factory Bean (simple sorting)"""
        bean_defs = [
            BeanDefinition(
                ServiceA, bean_name="singleton_a", scope=BeanScope.SINGLETON
            ),
            BeanDefinition(ServiceB, bean_name="factory_b", scope=BeanScope.FACTORY),
            BeanDefinition(
                ServiceC, bean_name="prototype_c", scope=BeanScope.PROTOTYPE
            ),
            BeanDefinition(ServiceD, bean_name="factory_d", scope=BeanScope.FACTORY),
        ]

        sorted_beans = BeanOrderStrategy.sort_beans(bean_defs)

        # Verify: Factory Beans come first
        assert sorted_beans[0].scope == BeanScope.FACTORY
        assert sorted_beans[1].scope == BeanScope.FACTORY
        assert sorted_beans[2].scope in {BeanScope.SINGLETON, BeanScope.PROTOTYPE}
        assert sorted_beans[3].scope in {BeanScope.SINGLETON, BeanScope.PROTOTYPE}

    def test_primary_and_factory_combined(self):
        """Test combined Primary and Factory priority (simple sorting)"""
        bean_defs = [
            # Lowest priority: non-Primary + non-Factory
            BeanDefinition(
                ServiceA,
                bean_name="lowest",
                is_primary=False,
                scope=BeanScope.SINGLETON,
            ),
            # Medium priority: non-Primary + Factory
            BeanDefinition(
                ServiceB, bean_name="medium", is_primary=False, scope=BeanScope.FACTORY
            ),
            # Higher priority: Primary + non-Factory
            BeanDefinition(
                ServiceC, bean_name="high", is_primary=True, scope=BeanScope.SINGLETON
            ),
            # Highest priority: Primary + Factory
            BeanDefinition(
                ServiceD, bean_name="highest", is_primary=True, scope=BeanScope.FACTORY
            ),
        ]

        sorted_beans = BeanOrderStrategy.sort_beans(bean_defs)

        # Verify sort order
        assert sorted_beans[0].bean_name == "highest"  # Primary+Factory
        assert sorted_beans[1].bean_name == "high"  # Primary+non-Factory
        assert sorted_beans[2].bean_name == "medium"  # non-Primary+Factory
        assert sorted_beans[3].bean_name == "lowest"  # non-Primary+non-Factory

    def test_mock_beans_not_filtered_in_simple_sort(self):
        """Test simple sorting does not filter Mock Beans"""
        bean_defs = [
            BeanDefinition(
                ServiceA, bean_name="mock_a", is_mock=True, is_primary=False
            ),
            BeanDefinition(
                ServiceB, bean_name="normal_b", is_mock=False, is_primary=True
            ),
        ]

        sorted_beans = BeanOrderStrategy.sort_beans(bean_defs)

        # Verify: Mock Bean is not filtered (simple sorting does not consider Mock)
        assert len(sorted_beans) == 2
        # Verify: Primary takes precedence (regardless of Mock status)
        assert sorted_beans[0].bean_name == "normal_b"
        assert sorted_beans[1].bean_name == "mock_a"

    def test_empty_list_simple(self):
        """Test empty list (simple sorting)"""
        sorted_beans = BeanOrderStrategy.sort_beans([])
        assert sorted_beans == []

    def test_single_bean_simple(self):
        """Test single Bean (simple sorting)"""
        bean_defs = [BeanDefinition(ServiceA, bean_name="single")]
        sorted_beans = BeanOrderStrategy.sort_beans(bean_defs)
        assert len(sorted_beans) == 1
        assert sorted_beans[0].bean_name == "single"

    def test_same_priority_beans(self):
        """Test Beans with same priority maintain original order"""
        bean_defs = [
            BeanDefinition(
                ServiceA, bean_name="a", is_primary=False, scope=BeanScope.SINGLETON
            ),
            BeanDefinition(
                ServiceB, bean_name="b", is_primary=False, scope=BeanScope.SINGLETON
            ),
            BeanDefinition(
                ServiceC, bean_name="c", is_primary=False, scope=BeanScope.SINGLETON
            ),
        ]

        sorted_beans = BeanOrderStrategy.sort_beans(bean_defs)

        # Verify: Beans with same priority maintain original order (stable sort)
        assert sorted_beans[0].bean_name == "a"
        assert sorted_beans[1].bean_name == "b"
        assert sorted_beans[2].bean_name == "c"


# ==================== Edge Case Tests ====================


class TestEdgeCases:
    """Test edge cases and exceptional scenarios"""

    def test_none_direct_match_types(self):
        """Test direct_match_types is empty set"""
        bean_defs = [
            BeanDefinition(ServiceA, bean_name="a"),
            BeanDefinition(ServiceB, bean_name="b"),
        ]

        sorted_beans = BeanOrderStrategy.sort_beans_with_context(
            bean_defs=bean_defs, direct_match_types=set(), mock_mode=False
        )

        # Verify: all Beans are treated as non-direct match
        assert len(sorted_beans) == 2

    def test_all_direct_match_types(self):
        """Test all Beans are direct matches"""
        bean_defs = [
            BeanDefinition(ServiceA, bean_name="a", is_primary=True),
            BeanDefinition(ServiceB, bean_name="b", is_primary=False),
        ]

        sorted_beans = BeanOrderStrategy.sort_beans_with_context(
            bean_defs=bean_defs,
            direct_match_types={ServiceA, ServiceB},
            mock_mode=False,
        )

        # Verify: Primary Bean comes first
        assert sorted_beans[0].bean_name == "a"
        assert sorted_beans[1].bean_name == "b"

    def test_multiple_beans_same_type(self):
        """Test multiple Beans of the same type"""
        bean_defs = [
            BeanDefinition(
                ServiceA, bean_name="a1", is_primary=False, scope=BeanScope.SINGLETON
            ),
            BeanDefinition(
                ServiceA, bean_name="a2", is_primary=True, scope=BeanScope.SINGLETON
            ),
            BeanDefinition(
                ServiceA, bean_name="a3", is_primary=False, scope=BeanScope.FACTORY
            ),
            BeanDefinition(
                ServiceA, bean_name="a4", is_primary=True, scope=BeanScope.FACTORY
            ),
        ]

        sorted_beans = BeanOrderStrategy.sort_beans_with_context(
            bean_defs=bean_defs, direct_match_types={ServiceA}, mock_mode=False
        )

        # Verify sort order: Primary+Factory > Primary+non-Factory > non-Primary+Factory > non-Primary+non-Factory
        assert sorted_beans[0].bean_name == "a4"  # Primary+Factory
        assert sorted_beans[1].bean_name == "a2"  # Primary+non-Factory
        assert sorted_beans[2].bean_name == "a3"  # non-Primary+Factory
        assert sorted_beans[3].bean_name == "a1"  # non-Primary+non-Factory

    def test_all_attributes_combinations(self):
        """Test all combinations of attributes (2^4=16 combinations)"""
        # Generate all possible combinations
        combinations = []
        for i in range(16):
            is_mock = bool(i & 8)
            is_direct = bool(i & 4)
            is_primary = bool(i & 2)
            is_factory = bool(i & 1)

            bean_type = [ServiceA, ServiceB, ServiceC, ServiceD][i % 4]
            bean = BeanDefinition(
                bean_type,
                bean_name=f"bean_{i}",
                is_mock=is_mock,
                is_primary=is_primary,
                scope=BeanScope.FACTORY if is_factory else BeanScope.SINGLETON,
            )
            combinations.append((bean, is_direct, bean_type))

        # Extract Bean list and direct match types
        bean_defs = [c[0] for c in combinations]
        direct_match_types = {c[2] for c in combinations if c[1]}

        # Sort (Mock mode)
        sorted_beans = BeanOrderStrategy.sort_beans_with_context(
            bean_defs=bean_defs, direct_match_types=direct_match_types, mock_mode=True
        )

        # Verify: sorted by priority, Mock Beans come first
        # Verify all Beans are retained
        assert len(sorted_beans) == 16

        # Verify the first Bean should be the highest priority
        first_bean = sorted_beans[0]
        first_key = BeanOrderStrategy.calculate_order_key(
            first_bean,
            is_direct_match=first_bean.bean_type in direct_match_types,
            mock_mode=True,
        )

        # Verify priority of all other Beans is not higher than the first Bean
        for bean in sorted_beans[1:]:
            bean_key = BeanOrderStrategy.calculate_order_key(
                bean,
                is_direct_match=bean.bean_type in direct_match_types,
                mock_mode=True,
            )
            assert first_key <= bean_key


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])

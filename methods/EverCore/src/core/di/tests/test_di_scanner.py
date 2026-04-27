# -*- coding: utf-8 -*-
"""
cd /Users/admin/memsys_opensource
PYTHONPATH=/Users/admin/memsys_opensource/src python -m pytest src/core/di/tests/test_di_scanner.py -v -s

DI Scanner Test

Test the component scanning and auto-registration functionality of Scanner
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from core.di.container import DIContainer, get_container
from core.di.scanner import ComponentScanner
from core.di.decorators import component, service, repository, mock_impl, factory
from core.di.bean_definition import BeanScope
from core.di.tests.test_fixtures import (
    UserRepository,
    MySQLUserRepository,
    NotificationService,
    EmailNotificationService,
)


class TestScannerBasicFunctionality:
    """Test basic functionality of Scanner"""

    def setup_method(self):
        """Create a new container and scanner before each test"""
        self.container = DIContainer()
        self.scanner = ComponentScanner()

    def test_scanner_initialization(self):
        """Test Scanner initialization"""
        assert self.scanner is not None
        assert self.scanner.scan_paths == []
        assert self.scanner.scan_packages == []
        assert self.scanner.recursive is True

    def test_add_scan_path(self):
        """Test adding scan path"""
        self.scanner.add_scan_path("/path/to/scan")
        assert "/path/to/scan" in self.scanner.scan_paths

        # Chained calls
        self.scanner.add_scan_path("/path1").add_scan_path("/path2")
        assert len(self.scanner.scan_paths) == 3

    def test_add_scan_package(self):
        """Test adding scan package"""
        self.scanner.add_scan_package("my.package")
        assert "my.package" in self.scanner.scan_packages

        # Chained calls
        self.scanner.add_scan_package("pkg1").add_scan_package("pkg2")
        assert len(self.scanner.scan_packages) == 3

    def test_exclude_patterns(self):
        """Test exclude patterns"""
        self.scanner.exclude_pattern("test_")
        assert "test_" in self.scanner.exclude_patterns

        # Default exclude patterns should exist
        assert "__pycache__" in self.scanner.exclude_paths
        assert "test_" in self.scanner.exclude_patterns


class TestComponentDecoratorIntegration:
    """Test integration of decorators with Container"""

    def setup_method(self):
        """Reset global container before each test"""
        # Note: decorators register to global container by default
        # Here we test the behavior of decorators
        pass

    def test_component_decorator_registers_bean(self):
        """Test @component decorator automatically registers Bean"""
        container = get_container()

        # Define a component
        @component(name="test_component_unique_1")
        class TestComponent1:
            def __init__(self):
                self.value = "test1"

        # Verify Bean is registered
        assert container.contains_bean("test_component_unique_1")

        # Get Bean
        comp = container.get_bean("test_component_unique_1")
        assert isinstance(comp, TestComponent1)
        assert comp.value == "test1"

    def test_service_decorator_registers_bean(self):
        """Test @service decorator automatically registers Bean"""
        container = get_container()

        # Define a service
        @service(name="test_service_unique_1", primary=True)
        class TestService1:
            def __init__(self):
                self.service_type = "test"

        # Verify Bean is registered
        assert container.contains_bean("test_service_unique_1")

        # Get Bean
        svc = container.get_bean("test_service_unique_1")
        assert isinstance(svc, TestService1)
        assert svc.service_type == "test"

    def test_repository_decorator_registers_bean(self):
        """Test @repository decorator automatically registers Bean"""
        container = get_container()

        # Define a repository
        @repository(name="test_repo_unique_1")
        class TestRepository1:
            def __init__(self):
                self.db = "sqlite"

        # Verify Bean is registered
        assert container.contains_bean("test_repo_unique_1")

        # Get Bean
        repo = container.get_bean("test_repo_unique_1")
        assert isinstance(repo, TestRepository1)
        assert repo.db == "sqlite"

    def test_mock_impl_decorator_registers_mock_bean(self):
        """Test @mock_impl decorator registers Mock Bean"""
        container = get_container()

        # Define a Mock implementation
        @mock_impl(name="test_mock_unique_1")
        class TestMock1:
            def __init__(self):
                self.is_mock = True

        # Verify Bean is registered
        assert container.contains_bean("test_mock_unique_1")

        # In non-Mock mode, Mock Bean should not be automatically retrieved
        # But can be retrieved by name
        mock = container.get_bean("test_mock_unique_1")
        assert isinstance(mock, TestMock1)
        assert mock.is_mock is True

    def test_factory_decorator_registers_factory(self):
        """Test @factory decorator registers Factory"""
        container = get_container()

        # Define a Factory
        class Product:
            def __init__(self, name: str):
                self.name = name

        @factory(bean_type=Product, name="test_factory_unique_1")
        def create_product() -> Product:
            return Product(name="TestProduct")

        # Verify Factory is registered
        assert container.contains_bean("test_factory_unique_1")

        # Get Bean (created by Factory)
        product = container.get_bean("test_factory_unique_1")
        assert isinstance(product, Product)
        assert product.name == "TestProduct"

    def test_component_with_scope(self):
        """Test @component decorator specifying Scope"""
        container = get_container()

        # Define component with Prototype scope
        @component(name="test_prototype_unique_1", scope=BeanScope.PROTOTYPE)
        class TestPrototype1:
            counter = 0

            def __init__(self):
                TestPrototype1.counter += 1
                self.id = TestPrototype1.counter

        # Get multiple instances
        obj1 = container.get_bean("test_prototype_unique_1")
        obj2 = container.get_bean("test_prototype_unique_1")

        # Prototype scope should create different instances
        assert obj1 is not obj2
        assert obj1.id != obj2.id


class TestInterfaceImplementationScanning:
    """Test scanning of interfaces and implementation classes"""

    def setup_method(self):
        """Create new container before each test"""
        self.container = DIContainer()

    def test_register_interface_implementations(self):
        """Test registering multiple implementations of an interface"""
        # Manually register implementation class (simulate scan result)
        self.container.register_bean(
            bean_type=MySQLUserRepository, bean_name="mysql_user_repo", is_primary=True
        )

        # Verify can be retrieved via interface type
        repo = self.container.get_bean_by_type(UserRepository)
        assert isinstance(repo, MySQLUserRepository)

    def test_multiple_implementations_of_interface(self):
        """Test multiple implementations of the same interface"""
        from core.di.tests.test_fixtures import (
            PostgreSQLUserRepository,
            MockUserRepository,
        )

        # Register multiple implementations
        self.container.register_bean(
            bean_type=MySQLUserRepository, bean_name="mysql_repo", is_primary=False
        )
        self.container.register_bean(
            bean_type=PostgreSQLUserRepository,
            bean_name="postgres_repo",
            is_primary=True,
        )
        self.container.register_bean(
            bean_type=MockUserRepository, bean_name="mock_repo", is_mock=True
        )

        # Get Primary implementation
        repo = self.container.get_bean_by_type(UserRepository)
        assert isinstance(repo, PostgreSQLUserRepository)

        # Get all implementations (non-Mock mode)
        all_repos = self.container.get_beans_by_type(UserRepository)
        assert len(all_repos) == 2  # MySQL + PostgreSQL


class TestScanContextAndMetadata:
    """Test scanning context and metadata"""

    def setup_method(self):
        """Create new container before each test"""
        self.container = DIContainer()

    def test_bean_with_metadata(self):
        """Test Bean metadata"""
        # Register Bean with metadata
        self.container.register_bean(
            bean_type=MySQLUserRepository,
            bean_name="mysql_repo",
            metadata={"version": "1.0", "author": "test", "db_type": "mysql"},
        )

        # Get Bean Definition to verify metadata
        bean_def = self.container._named_beans.get("mysql_repo")
        assert bean_def is not None
        assert bean_def.metadata["version"] == "1.0"
        assert bean_def.metadata["author"] == "test"
        assert bean_def.metadata["db_type"] == "mysql"

    def test_metadata_from_decorator(self):
        """Test metadata passed from decorator"""
        container = get_container()

        # Define component with metadata
        @component(
            name="test_metadata_comp_unique_1", metadata={"env": "test", "priority": 10}
        )
        class TestMetadataComp:
            pass

        # Get Bean Definition
        bean_def = container._named_beans.get("test_metadata_comp_unique_1")
        assert bean_def is not None
        assert bean_def.metadata["env"] == "test"
        assert bean_def.metadata["priority"] == 10


class TestConditionalRegistration:
    """Test conditional registration"""

    def test_lazy_registration(self):
        """Test lazy registration"""
        container = get_container()
        initial_bean_count = len(container._named_beans)

        # Define component with lazy registration
        @component(name="lazy_comp_unique_1", lazy=True)
        class LazyComponent:
            pass

        # Lazily registered Bean should not immediately appear in container
        # Note: In current implementation, lazy=True is just a flag, actual behavior depends on specific implementation
        # Here we only verify the component is correctly marked
        assert hasattr(LazyComponent, '_di_lazy')
        assert LazyComponent._di_lazy is True


class TestBeanDependencies:
    """Test dependencies between Beans"""

    def setup_method(self):
        """Create new container before each test"""
        self.container = DIContainer()

    def test_bean_depends_on_another(self):
        """Test Bean depends on another Bean"""
        from core.di.tests.test_fixtures import UserServiceImpl, register_standard_beans

        # Register dependent Bean
        register_standard_beans(self.container)

        # Create Service that depends on other Bean
        service = UserServiceImpl(container=self.container)

        # Verify dependency injection succeeded
        assert service.repository is not None
        assert isinstance(service.repository, UserRepository)

        # Verify Service functions correctly
        user = service.get_user(1)
        assert user is not None


class TestEdgeCases:
    """Test edge cases"""

    def setup_method(self):
        """Create new container before each test"""
        self.container = DIContainer()

    def test_empty_container(self):
        """Test empty container"""
        from core.di.exceptions import BeanNotFoundError

        # Empty container should raise exception
        with pytest.raises(BeanNotFoundError):
            self.container.get_bean_by_type(UserRepository)

    def test_duplicate_bean_name(self):
        """Test duplicate Bean name throws exception"""
        from core.di.exceptions import DuplicateBeanError
        from core.di.tests.test_fixtures import PostgreSQLUserRepository

        # Register first Bean
        self.container.register_bean(
            bean_type=MySQLUserRepository, bean_name="same_name"
        )

        # Attempt to register Bean with same name should raise exception
        with pytest.raises(DuplicateBeanError):
            self.container.register_bean(
                bean_type=PostgreSQLUserRepository, bean_name="same_name"
            )

    def test_get_all_beans_empty_result(self):
        """Test getting all Beans of non-existent type"""

        # Unregistered type
        class UnregisteredService:
            pass

        # Getting all Beans should return empty list
        beans = self.container.get_beans_by_type(UnregisteredService)
        assert beans == []


class TestRealWorldScanningScenario:
    """Test real-world scanning scenario"""

    def test_scan_test_fixtures_module(self):
        """Test scanning test_fixtures module"""
        # This test verifies we can import classes from fixtures module
        from core.di.tests.test_fixtures import (
            MySQLUserRepository,
            PostgreSQLUserRepository,
            MockUserRepository,
            EmailNotificationService,
            SMSNotificationService,
            RedisCacheService,
            MemoryCacheService,
        )

        # Verify all classes can be imported and instantiated normally
        mysql_repo = MySQLUserRepository()
        assert mysql_repo.db_type == "mysql"

        postgres_repo = PostgreSQLUserRepository()
        assert postgres_repo.db_type == "postgres"

        mock_repo = MockUserRepository()
        assert mock_repo.db_type == "mock"

        email_notif = EmailNotificationService()
        assert email_notif.sent_messages == []

        sms_notif = SMSNotificationService()
        assert sms_notif.sent_messages == []

        redis_cache = RedisCacheService()
        assert redis_cache.cache_type == "redis"

        memory_cache = MemoryCacheService()
        assert memory_cache.cache_type == "memory"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s", "--tb=short"])

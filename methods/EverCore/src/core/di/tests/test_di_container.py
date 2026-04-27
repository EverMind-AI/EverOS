# -*- coding: utf-8 -*-
"""
cd /Users/admin/memsys_opensource
PYTHONPATH=/Users/admin/memsys_opensource/src python -m pytest src/core/di/tests/test_di_container.py -v -s

DI Container integration tests

Test core Container functionalities such as Bean registration, resolution, and priority selection
"""

import pytest
from abc import ABC, abstractmethod
from typing import List
from core.di.container import DIContainer
from core.di.bean_definition import BeanScope
from core.di.exceptions import BeanNotFoundError
from core.di.tests.test_fixtures import (
    # User service related
    UserRepository,
    MySQLUserRepository,
    PostgreSQLUserRepository,
    MockUserRepository,
    UserService,
    UserServiceImpl,
    # Notification service related
    NotificationService,
    EmailNotificationService,
    SMSNotificationService,
    PushNotificationService,
    # Email service related
    EmailService,
    SMTPEmailService,
    # Database connection related
    DatabaseConnection,
    create_database_connection,
    create_readonly_connection,
    # Prototype service
    PrototypeService,
    # Cache service related
    CacheService,
    RedisCacheService,
    MemoryCacheService,
    # Utility functions
    register_standard_beans,
)


class TestBeanRegistrationAndRetrieval:
    """Test Bean registration and retrieval"""

    def setup_method(self):
        """Create a new container before each test"""
        self.container = DIContainer()

    def test_register_and_get_single_bean(self):
        """Test registering and retrieving a single Bean"""
        # Register Bean
        self.container.register_bean(
            bean_type=MySQLUserRepository, bean_name="mysql_repo"
        )

        # Retrieve by type
        repo = self.container.get_bean_by_type(MySQLUserRepository)
        assert isinstance(repo, MySQLUserRepository)
        assert repo.db_type == "mysql"

        # Retrieve by name
        repo2 = self.container.get_bean("mysql_repo")
        assert repo is repo2  # Singleton mode, should be the same instance

    def test_register_multiple_implementations(self):
        """Test registering multiple implementations of the same interface"""
        # Register multiple UserRepository implementations
        self.container.register_bean(
            bean_type=MySQLUserRepository, bean_name="mysql_repo", is_primary=True
        )
        self.container.register_bean(
            bean_type=PostgreSQLUserRepository, bean_name="postgres_repo"
        )

        # Retrieve by interface type, should return Primary implementation
        repo = self.container.get_bean_by_type(UserRepository)
        assert isinstance(repo, MySQLUserRepository)

        # Retrieve all implementations
        all_repos = self.container.get_beans_by_type(UserRepository)
        assert len(all_repos) == 2
        assert isinstance(all_repos[0], MySQLUserRepository)  # Primary first
        assert isinstance(all_repos[1], PostgreSQLUserRepository)

    def test_bean_not_found_error(self):
        """Test raising exception when retrieving non-existent Bean"""

        # Define unregistered interface
        class UnregisteredService(ABC):
            pass

        # Retrieving by type should raise exception
        with pytest.raises(BeanNotFoundError):
            self.container.get_bean_by_type(UnregisteredService)

        # Retrieving by name should raise exception
        with pytest.raises(BeanNotFoundError):
            self.container.get_bean("non_existent_bean")

    def test_contains_bean_check(self):
        """Test Bean existence check"""
        # Register Bean
        self.container.register_bean(
            bean_type=MySQLUserRepository, bean_name="mysql_repo"
        )

        # Check if Bean exists
        assert self.container.contains_bean_by_type(MySQLUserRepository)
        # Note: Interface lookup requires building inheritance relationship cache first
        # Direct interface lookup may return False because implementation class was registered
        # Correct way is get_bean_by_type automatically finds implementation class
        assert self.container.contains_bean("mysql_repo")

        # Check non-existent Bean
        assert not self.container.contains_bean_by_type(PostgreSQLUserRepository)
        assert not self.container.contains_bean("non_existent")


class TestPrimaryBeanSelection:
    """Test Primary Bean selection logic"""

    def setup_method(self):
        """Create a new container before each test"""
        self.container = DIContainer()

    def test_primary_bean_priority(self):
        """Test Primary Bean takes precedence over non-Primary Bean"""
        # Register two implementations: one Primary, one non-Primary
        self.container.register_bean(
            bean_type=PostgreSQLUserRepository,
            bean_name="postgres_repo",
            is_primary=False,
        )
        self.container.register_bean(
            bean_type=MySQLUserRepository, bean_name="mysql_repo", is_primary=True
        )

        # Retrieve UserRepository, should return Primary implementation
        repo = self.container.get_bean_by_type(UserRepository)
        assert isinstance(repo, MySQLUserRepository)
        assert repo.db_type == "mysql"

    def test_multiple_primary_beans_return_first(self):
        """Test returning the first registered when multiple Primary Beans exist"""
        # Register two Primary implementations
        self.container.register_bean(
            bean_type=MySQLUserRepository, bean_name="mysql_repo", is_primary=True
        )
        self.container.register_bean(
            bean_type=PostgreSQLUserRepository,
            bean_name="postgres_repo",
            is_primary=True,
        )

        # Should return the first Primary Bean
        repo = self.container.get_bean_by_type(UserRepository)
        assert isinstance(repo, MySQLUserRepository)

    def test_no_primary_bean_returns_first(self):
        """Test returning the first registered when no Primary Bean exists"""
        # Register two non-Primary implementations
        self.container.register_bean(
            bean_type=MySQLUserRepository, bean_name="mysql_repo", is_primary=False
        )
        self.container.register_bean(
            bean_type=PostgreSQLUserRepository,
            bean_name="postgres_repo",
            is_primary=False,
        )

        # Should return the first registered Bean
        repo = self.container.get_bean_by_type(UserRepository)
        assert isinstance(repo, MySQLUserRepository)


class TestMockMode:
    """Test Mock mode"""

    def setup_method(self):
        """Create a new container before each test"""
        self.container = DIContainer()
        # Register standard Beans (including Mock)
        register_standard_beans(self.container)

    def test_normal_mode_filters_mock_beans(self):
        """Test Mock Beans are filtered out in non-Mock mode"""
        # Ensure not in Mock mode
        assert not self.container.is_mock_mode()

        # Retrieve UserRepository, should return non-Mock implementation
        repo = self.container.get_bean_by_type(UserRepository)
        assert not isinstance(repo, MockUserRepository)
        assert isinstance(repo, MySQLUserRepository)  # Primary non-Mock implementation

        # Retrieve all implementations, should not include Mock
        all_repos = self.container.get_beans_by_type(UserRepository)
        assert len(all_repos) == 2  # MySQL + PostgreSQL
        assert all(not isinstance(r, MockUserRepository) for r in all_repos)

    def test_mock_mode_prioritizes_mock_beans(self):
        """Test Mock Beans take precedence in Mock mode"""
        # Enable Mock mode
        self.container.enable_mock_mode()
        assert self.container.is_mock_mode()

        # Clear cache to ensure Bean re-selection
        self.container._singleton_instances.clear()

        # Retrieve UserRepository, should return Mock implementation
        repo = self.container.get_bean_by_type(UserRepository)
        assert isinstance(repo, MockUserRepository)
        assert repo.db_type == "mock"

        # Verify Mock data
        user = repo.find_by_id(1)
        assert "Mock" in user["name"]
        assert user["source"] == "mock"

    def test_mock_mode_toggle(self):
        """Test toggling Mock mode"""
        # Initial: non-Mock mode
        repo1 = self.container.get_bean_by_type(UserRepository)
        assert isinstance(repo1, MySQLUserRepository)

        # Enable Mock mode
        self.container.enable_mock_mode()
        self.container._singleton_instances.clear()

        repo2 = self.container.get_bean_by_type(UserRepository)
        assert isinstance(repo2, MockUserRepository)

        # Disable Mock mode
        self.container.disable_mock_mode()
        self.container._singleton_instances.clear()

        repo3 = self.container.get_bean_by_type(UserRepository)
        assert isinstance(repo3, MySQLUserRepository)

    def test_get_all_beans_in_mock_mode(self):
        """Test retrieving all Beans in Mock mode"""
        # Enable Mock mode
        self.container.enable_mock_mode()

        # Retrieve all UserRepository implementations
        all_repos = self.container.get_beans_by_type(UserRepository)

        # Should include Mock implementation, with Mock first
        assert len(all_repos) == 3
        assert isinstance(all_repos[0], MockUserRepository)  # Mock first
        assert isinstance(all_repos[1], MySQLUserRepository)  # Primary next
        assert isinstance(all_repos[2], PostgreSQLUserRepository)


class TestBeanScopes:
    """Test different Bean scopes"""

    def setup_method(self):
        """Create a new container before each test"""
        self.container = DIContainer()
        PrototypeService.reset_counter()

    def test_singleton_scope(self):
        """Test Singleton scope (default)"""
        # Register Singleton Bean
        self.container.register_bean(
            bean_type=MySQLUserRepository,
            bean_name="mysql_repo",
            scope=BeanScope.SINGLETON,
        )

        # Multiple retrievals should return the same instance
        repo1 = self.container.get_bean_by_type(MySQLUserRepository)
        repo2 = self.container.get_bean_by_type(MySQLUserRepository)
        repo3 = self.container.get_bean("mysql_repo")

        assert repo1 is repo2
        assert repo1 is repo3

        # Modify state to verify
        repo1.call_count = 100
        assert repo2.call_count == 100
        assert repo3.call_count == 100

    def test_prototype_scope(self):
        """Test Prototype scope (create new instance each time)"""
        # Register Prototype Bean
        self.container.register_bean(
            bean_type=PrototypeService,
            bean_name="prototype_service",
            scope=BeanScope.PROTOTYPE,
        )

        # Multiple retrievals should return different instances
        service1 = self.container.get_bean_by_type(PrototypeService)
        service2 = self.container.get_bean_by_type(PrototypeService)
        service3 = self.container.get_bean("prototype_service")

        assert service1 is not service2
        assert service1 is not service3
        assert service2 is not service3

        # Verify instance IDs are different
        assert service1.instance_id == 1
        assert service2.instance_id == 2
        assert service3.instance_id == 3

        # Verify instance states are independent
        service1.add_data("data1")
        service2.add_data("data2")

        assert service1.get_data() == ["data1"]
        assert service2.get_data() == ["data2"]

    def test_factory_scope(self):
        """Test Factory scope"""
        # Register Factory Bean (register_factory defaults to FACTORY scope)
        self.container.register_factory(
            bean_type=DatabaseConnection,
            factory_method=create_database_connection,
            bean_name="db_connection",
        )

        # Retrieve Bean
        conn = self.container.get_bean_by_type(DatabaseConnection)

        # Verify Bean is created via factory method
        assert isinstance(conn, DatabaseConnection)
        assert conn.host == "localhost"
        assert conn.port == 3306
        assert conn.database == "test_db"
        assert conn.connected


class TestFactoryBeans:
    """Test Factory Beans"""

    def setup_method(self):
        """Create a new container before each test"""
        self.container = DIContainer()

    def test_factory_bean_creation(self):
        """Test creating Bean via factory method"""
        # Register Factory
        self.container.register_factory(
            bean_type=DatabaseConnection,
            factory_method=create_database_connection,
            bean_name="db_connection",
        )

        # Retrieve Bean
        conn = self.container.get_bean_by_type(DatabaseConnection)

        # Verify Bean created correctly
        assert isinstance(conn, DatabaseConnection)
        assert conn.connected

        # Call method to verify
        result = conn.execute("SELECT * FROM users")
        assert len(result) == 1
        assert "Executed" in result[0]["result"]

    def test_multiple_factories_for_same_type(self):
        """Test multiple Factories for the same type"""
        # Register multiple Factories
        self.container.register_factory(
            bean_type=DatabaseConnection,
            factory_method=create_database_connection,
            bean_name="db_connection",
            is_primary=True,
        )
        self.container.register_factory(
            bean_type=DatabaseConnection,
            factory_method=create_readonly_connection,
            bean_name="readonly_connection",
        )

        # Retrieve Bean created by Primary Factory
        conn = self.container.get_bean_by_type(DatabaseConnection)
        assert conn.host == "localhost"

        # Retrieve Bean created by another Factory by name
        readonly_conn = self.container.get_bean("readonly_connection")
        assert readonly_conn.host == "readonly.example.com"

    def test_factory_with_priority(self):
        """Test Factory Bean priority"""

        # Register Regular Bean and Factory Bean
        def factory() -> CacheService:
            cache = RedisCacheService()
            cache.set("init_key", "init_value")
            return cache

        # Regular Bean
        self.container.register_bean(
            bean_type=MemoryCacheService, bean_name="memory_cache", is_primary=False
        )

        # Factory Bean (Primary)
        self.container.register_factory(
            bean_type=CacheService,
            factory_method=factory,
            bean_name="redis_cache",
            is_primary=True,
        )

        # Should return Primary Bean created by Factory
        cache = self.container.get_bean_by_type(CacheService)
        assert isinstance(cache, RedisCacheService)
        assert cache.get("init_key") == "init_value"


class TestDependencyInjection:
    """Test dependency injection"""

    def setup_method(self):
        """Create a new container before each test"""
        self.container = DIContainer()
        register_standard_beans(self.container)

    def test_constructor_injection(self):
        """Test constructor dependency injection"""
        # Retrieve dependency
        repo = self.container.get_bean_by_type(UserRepository)

        # Manually inject dependency to create Service
        service = UserServiceImpl(repository=repo)

        # Verify dependency correctly injected
        assert isinstance(service.repository, MySQLUserRepository)

        # Verify service functionality
        user = service.get_user(1)
        assert user["id"] == 1
        assert user["source"] == "mysql"

    def test_container_based_injection(self):
        """Test resolving dependencies through container"""
        # Create Service passing container
        service = UserServiceImpl(container=self.container)

        # Verify dependency resolved through container
        assert isinstance(service.repository, MySQLUserRepository)

        # Verify service functionality
        users = service.get_all_users()
        assert len(users) == 2
        assert all(u["source"] == "mysql" for u in users)

    def test_dependency_injection_in_mock_mode(self):
        """Test dependency injection in Mock mode"""
        # Enable Mock mode
        self.container.enable_mock_mode()

        # Create Service (should inject Mock dependency)
        service = UserServiceImpl(container=self.container)

        # Verify Mock dependency is injected
        assert isinstance(service.repository, MockUserRepository)

        # Verify Mock data
        user = service.get_user(1)
        assert "Mock" in user["name"]
        assert user["source"] == "mock"


class TestBeanOrdering:
    """Test Bean ordering priority"""

    def setup_method(self):
        """Create a new container before each test"""
        self.container = DIContainer()

    def test_ordering_primary_over_non_primary(self):
        """Test Primary takes precedence over non-Primary"""
        # Register out of order
        self.container.register_bean(
            bean_type=SMSNotificationService, bean_name="sms", is_primary=False
        )
        self.container.register_bean(
            bean_type=EmailNotificationService, bean_name="email", is_primary=True
        )
        self.container.register_bean(
            bean_type=PushNotificationService, bean_name="push", is_primary=False
        )

        # Retrieve all Beans, verify Primary comes first
        services = self.container.get_beans_by_type(NotificationService)
        assert isinstance(services[0], EmailNotificationService)  # Primary

    def test_ordering_factory_over_singleton(self):
        """Test Factory takes precedence over Singleton (when both are Primary)"""

        # Define test interface
        class TestService(ABC):
            pass

        class ServiceA(TestService):
            def __init__(self):
                self.type = "singleton"

        class ServiceB(TestService):
            def __init__(self):
                self.type = "factory"

        # Register Singleton (Primary)
        self.container.register_bean(
            bean_type=ServiceA,
            bean_name="service_a",
            is_primary=True,
            scope=BeanScope.SINGLETON,
        )

        # Register Factory (Primary) - register_factory defaults to FACTORY scope
        self.container.register_factory(
            bean_type=ServiceB,
            factory_method=lambda: ServiceB(),
            bean_name="service_b",
            is_primary=True,
        )

        # Factory should take precedence
        service = self.container.get_bean_by_type(TestService)
        assert isinstance(service, ServiceB)
        assert service.type == "factory"


class TestComplexScenarios:
    """Test complex scenarios"""

    def setup_method(self):
        """Create a new container before each test"""
        self.container = DIContainer()
        register_standard_beans(self.container)

    def test_multiple_interface_implementations(self):
        """Test multiple interfaces each having multiple implementations"""
        # Retrieve Primary implementations for different interfaces
        user_repo = self.container.get_bean_by_type(UserRepository)
        notification = self.container.get_bean_by_type(NotificationService)
        cache = self.container.get_bean_by_type(CacheService)

        # Verify respective Primary implementations
        assert isinstance(user_repo, MySQLUserRepository)
        assert isinstance(notification, EmailNotificationService)
        assert isinstance(cache, RedisCacheService)

    def test_get_all_beans_for_multiple_interfaces(self):
        """Test retrieving all implementations for multiple interfaces"""
        # UserRepository: 2 implementations (non-Mock mode)
        user_repos = self.container.get_beans_by_type(UserRepository)
        assert len(user_repos) == 2

        # NotificationService: 3 implementations
        notifications = self.container.get_beans_by_type(NotificationService)
        assert len(notifications) == 3

        # CacheService: 2 implementations
        caches = self.container.get_beans_by_type(CacheService)
        assert len(caches) == 2

    def test_bean_lifecycle_and_state(self):
        """Test Bean lifecycle and state management"""
        # Retrieve Singleton Bean
        repo = self.container.get_bean_by_type(UserRepository)

        # Modify state
        repo.find_by_id(1)
        repo.find_all()
        assert repo.call_count == 2

        # Retrieve again, should be same instance with state preserved
        repo2 = self.container.get_bean_by_type(UserRepository)
        assert repo2.call_count == 2

        # Call method, state continues to accumulate
        repo2.find_by_id(2)
        assert repo.call_count == 3
        assert repo2.call_count == 3


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s", "--tb=short"])

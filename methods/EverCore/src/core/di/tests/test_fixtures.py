# -*- coding: utf-8 -*-
"""
DI Test Fixtures

Provides interfaces, implementation classes, and Mock classes for testing
These classes can be imported and used by other test files
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from core.di.container import DIContainer
from core.di.bean_definition import BeanScope


# ==================== User Service Related ====================


class UserRepository(ABC):
    """User repository interface"""

    @abstractmethod
    def find_by_id(self, user_id: int) -> Optional[dict]:
        """Find user by ID"""
        pass

    @abstractmethod
    def find_all(self) -> List[dict]:
        """Find all users"""
        pass


class MySQLUserRepository(UserRepository):
    """MySQL user repository implementation (Primary)"""

    def __init__(self):
        self.db_type = "mysql"
        self.call_count = 0

    def find_by_id(self, user_id: int) -> Optional[dict]:
        self.call_count += 1
        return {"id": user_id, "name": f"User{user_id}", "source": "mysql"}

    def find_all(self) -> List[dict]:
        self.call_count += 1
        return [
            {"id": 1, "name": "User1", "source": "mysql"},
            {"id": 2, "name": "User2", "source": "mysql"},
        ]


class PostgreSQLUserRepository(UserRepository):
    """PostgreSQL user repository implementation (Non-Primary)"""

    def __init__(self):
        self.db_type = "postgres"
        self.call_count = 0

    def find_by_id(self, user_id: int) -> Optional[dict]:
        self.call_count += 1
        return {"id": user_id, "name": f"User{user_id}", "source": "postgres"}

    def find_all(self) -> List[dict]:
        self.call_count += 1
        return [{"id": 1, "name": "User1", "source": "postgres"}]


class MockUserRepository(UserRepository):
    """Mock user repository implementation"""

    def __init__(self):
        self.db_type = "mock"
        self.call_count = 0

    def find_by_id(self, user_id: int) -> Optional[dict]:
        self.call_count += 1
        return {"id": user_id, "name": f"MockUser{user_id}", "source": "mock"}

    def find_all(self) -> List[dict]:
        self.call_count += 1
        return [{"id": 999, "name": "MockUser", "source": "mock"}]


class UserService(ABC):
    """User service interface"""

    @abstractmethod
    def get_user(self, user_id: int) -> Optional[dict]:
        """Get user"""
        pass

    @abstractmethod
    def get_all_users(self) -> List[dict]:
        """Get all users"""
        pass


class UserServiceImpl(UserService):
    """User service implementation"""

    def __init__(
        self, repository: UserRepository = None, container: DIContainer = None
    ):
        # Supports two injection methods: constructor injection or retrieval via container
        if repository:
            self.repository = repository
        elif container:
            self.repository = container.get_bean_by_type(UserRepository)
        else:
            raise ValueError("Must provide either repository or container")
        self.call_count = 0

    def get_user(self, user_id: int) -> Optional[dict]:
        self.call_count += 1
        return self.repository.find_by_id(user_id)

    def get_all_users(self) -> List[dict]:
        self.call_count += 1
        return self.repository.find_all()


# ==================== Notification Service Related ====================


class NotificationService(ABC):
    """Notification service interface"""

    @abstractmethod
    def send(self, message: str, recipient: str) -> bool:
        """Send notification"""
        pass


class EmailNotificationService(NotificationService):
    """Email notification service implementation (Primary)"""

    def __init__(self):
        self.sent_messages = []

    def send(self, message: str, recipient: str) -> bool:
        self.sent_messages.append(
            {"message": message, "recipient": recipient, "type": "email"}
        )
        return True


class SMSNotificationService(NotificationService):
    """SMS notification service implementation (Non-Primary)"""

    def __init__(self):
        self.sent_messages = []

    def send(self, message: str, recipient: str) -> bool:
        self.sent_messages.append(
            {"message": message, "recipient": recipient, "type": "sms"}
        )
        return True


class PushNotificationService(NotificationService):
    """Push notification service implementation"""

    def __init__(self):
        self.sent_messages = []

    def send(self, message: str, recipient: str) -> bool:
        self.sent_messages.append(
            {"message": message, "recipient": recipient, "type": "push"}
        )
        return True


# ==================== Email Service Related ====================


class EmailService(ABC):
    """Email service interface"""

    @abstractmethod
    def send_email(self, to: str, subject: str, body: str) -> bool:
        """Send email"""
        pass


class SMTPEmailService(EmailService):
    """SMTP email service implementation"""

    def __init__(self):
        self.host = "smtp.example.com"
        self.port = 587
        self.sent_emails = []

    def send_email(self, to: str, subject: str, body: str) -> bool:
        self.sent_emails.append({"to": to, "subject": subject, "body": body})
        return True


# ==================== Database Connection Related ====================


class DatabaseConnection:
    """Database connection class"""

    def __init__(self, host: str, port: int, database: str):
        self.host = host
        self.port = port
        self.database = database
        self.connected = True

    def execute(self, sql: str) -> List[dict]:
        return [{"result": f"Executed: {sql}"}]

    def close(self):
        self.connected = False


def create_database_connection() -> DatabaseConnection:
    """Factory method to create database connection"""
    return DatabaseConnection(host="localhost", port=3306, database="test_db")


def create_readonly_connection() -> DatabaseConnection:
    """Factory method to create read-only database connection"""
    return DatabaseConnection(
        host="readonly.example.com", port=3306, database="test_db"
    )


# ==================== Prototype Scope Test Classes ====================


class PrototypeService:
    """Prototype scope service (a new instance is created each time it is retrieved)"""

    instance_counter = 0  # Class-level counter

    def __init__(self):
        PrototypeService.instance_counter += 1
        self.instance_id = PrototypeService.instance_counter
        self.data = []

    def add_data(self, value: str):
        self.data.append(value)

    def get_data(self) -> List[str]:
        return self.data

    @classmethod
    def reset_counter(cls):
        """Reset counter (used for testing)"""
        cls.instance_counter = 0


# ==================== Cache Service Related ====================


class CacheService(ABC):
    """Cache service interface"""

    @abstractmethod
    def get(self, key: str) -> Optional[str]:
        """Get cache"""
        pass

    @abstractmethod
    def set(self, key: str, value: str, ttl: int = 3600) -> bool:
        """Set cache"""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete cache"""
        pass


class RedisCacheService(CacheService):
    """Redis cache service implementation (Primary)"""

    def __init__(self):
        self.storage = {}
        self.cache_type = "redis"

    def get(self, key: str) -> Optional[str]:
        return self.storage.get(key)

    def set(self, key: str, value: str, ttl: int = 3600) -> bool:
        self.storage[key] = value
        return True

    def delete(self, key: str) -> bool:
        if key in self.storage:
            del self.storage[key]
            return True
        return False


class MemoryCacheService(CacheService):
    """In-memory cache service implementation"""

    def __init__(self):
        self.storage = {}
        self.cache_type = "memory"

    def get(self, key: str) -> Optional[str]:
        return self.storage.get(key)

    def set(self, key: str, value: str, ttl: int = 3600) -> bool:
        self.storage[key] = value
        return True

    def delete(self, key: str) -> bool:
        if key in self.storage:
            del self.storage[key]
            return True
        return False


# ==================== Utility Helper Functions ====================


def register_standard_beans(container: DIContainer):
    """Register standard test beans into the container

    Args:
        container: DI container instance
    """
    # Register UserRepository implementations
    container.register_bean(
        bean_type=MySQLUserRepository, bean_name="mysql_user_repo", is_primary=True
    )
    container.register_bean(
        bean_type=PostgreSQLUserRepository,
        bean_name="postgres_user_repo",
        is_primary=False,
    )
    container.register_bean(
        bean_type=MockUserRepository, bean_name="mock_user_repo", is_mock=True
    )

    # Register NotificationService implementations
    container.register_bean(
        bean_type=EmailNotificationService,
        bean_name="email_notification",
        is_primary=True,
    )
    container.register_bean(
        bean_type=SMSNotificationService, bean_name="sms_notification"
    )
    container.register_bean(
        bean_type=PushNotificationService, bean_name="push_notification"
    )

    # Register CacheService implementations
    container.register_bean(
        bean_type=RedisCacheService, bean_name="redis_cache", is_primary=True
    )
    container.register_bean(bean_type=MemoryCacheService, bean_name="memory_cache")

    # Register EmailService implementation
    container.register_bean(bean_type=SMTPEmailService, bean_name="smtp_email_service")

    # Register Factory Bean
    container.register_factory(
        bean_type=DatabaseConnection,
        factory_method=create_database_connection,
        bean_name="db_connection",
    )

    # Register Prototype Bean
    container.register_bean(
        bean_type=PrototypeService,
        bean_name="prototype_service",
        scope=BeanScope.PROTOTYPE,
    )

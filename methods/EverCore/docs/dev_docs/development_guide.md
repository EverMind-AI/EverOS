# Smart Memory System Development Guide

This document provides detailed guidance for developers on interface definitions, Mock implementations, and decoupled development.

## 📋 Table of Contents

- [Interface Definition and Implementation](#interface-definition-and-implementation)
- [Mock Implementation and Decoupled Development](#mock-implementation-and-decoupled-development)
- [Dependency Injection Best Practices](#dependency-injection-best-practices)
- [Development Environment Configuration](#development-environment-configuration)

## 🔧 Interface Definition and Implementation

### 1. Define Abstract Interfaces

Use Python's Abstract Base Classes (ABC) to define clear interfaces:

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

class UserRepository(ABC):
    """User storage interface"""
    
    @abstractmethod
    def find_by_id(self, user_id: int) -> Optional[Dict]:
        """Find user by ID"""
        pass
    
    @abstractmethod
    def save(self, user: Dict) -> Dict:
        """Save user"""
        pass
    
    @abstractmethod
    def find_by_email(self, email: str) -> Optional[Dict]:
        """Find user by email"""
        pass

class NotificationService(ABC):
    """Notification service interface"""
    
    @abstractmethod
    async def send_notification(self, user_id: int, message: str) -> bool:
        """Send notification"""
        pass
```

### 2. Implement Concrete Classes

Use dependency injection decorators to mark implementation classes:

```python
from core.di.decorators import repository, service

@repository("mysql_user_repo")
class MySQLUserRepository(UserRepository):
    """MySQL user storage implementation"""
    
    def find_by_id(self, user_id: int) -> Optional[Dict]:
        # Actual database query logic
        return {"id": user_id, "name": f"User {user_id}", "source": "mysql"}
    
    def save(self, user: Dict) -> Dict:
        # Actual save logic
        return {**user, "id": 123, "created_at": "2024-01-01"}
    
    def find_by_email(self, email: str) -> Optional[Dict]:
        # Actual query logic
        return {"id": 456, "email": email, "source": "mysql"}

@repository("redis_user_repo") 
class RedisUserRepository(UserRepository):
    """Redis user storage implementation (cache layer)"""
    
    def find_by_id(self, user_id: int) -> Optional[Dict]:
        # Redis cache query logic
        return {"id": user_id, "name": f"Cached User {user_id}", "source": "redis"}
    
    def save(self, user: Dict) -> Dict:
        # Redis cache save logic
        return {**user, "cached": True}
    
    def find_by_email(self, email: str) -> Optional[Dict]:
        # Redis cache query logic
        return None  # Not in cache

@service("email_notification")
class EmailNotificationService(NotificationService):
    """Email notification service implementation"""
    
    async def send_notification(self, user_id: int, message: str) -> bool:
        # Actual email sending logic
        print(f"📧 Sending email to user {user_id}: {message}")
        return True
```

### 3. Set Primary Implementation

When there are multiple implementations, use `primary=True` to mark the default implementation:

```python
@repository("primary_user_repo", primary=True)
class PrimaryUserRepository(UserRepository):
    """Primary user storage implementation"""
    
    def __init__(self):
        # Can combine multiple implementations
        self.mysql_repo = MySQLUserRepository()
        self.redis_repo = RedisUserRepository()
    
    def find_by_id(self, user_id: int) -> Optional[Dict]:
        # Check cache first, then database
        user = self.redis_repo.find_by_id(user_id)
        if user:
            return user
        return self.mysql_repo.find_by_id(user_id)
    
    def save(self, user: Dict) -> Dict:
        # Save to database and update cache
        saved_user = self.mysql_repo.save(user)
        self.redis_repo.save(saved_user)
        return saved_user
    
    def find_by_email(self, email: str) -> Optional[Dict]:
        return self.mysql_repo.find_by_email(email)
```

## 🧪 Mock Implementation and Decoupled Development

### 1. Define Mock Implementation

Use the `@mock_impl` decorator to define Mock implementations:

```python
from core.di.decorators import mock_impl

@mock_impl("mock_user_repo")
class MockUserRepository(UserRepository):
    """Mock user storage implementation"""
    
    def __init__(self):
        # Simulated data in memory
        self.users = {
            1: {"id": 1, "name": "Mock User 1", "email": "user1@mock.com"},
            2: {"id": 2, "name": "Mock User 2", "email": "user2@mock.com"}
        }
        self.next_id = 3
    
    def find_by_id(self, user_id: int) -> Optional[Dict]:
        return self.users.get(user_id)
    
    def save(self, user: Dict) -> Dict:
        if "id" not in user:
            user["id"] = self.next_id
            self.next_id += 1
        self.users[user["id"]] = user
        return user
    
    def find_by_email(self, email: str) -> Optional[Dict]:
        for user in self.users.values():
            if user.get("email") == email:
                return user
        return None

@mock_impl("mock_notification")
class MockNotificationService(NotificationService):
    """Mock notification service implementation"""
    
    def __init__(self):
        self.sent_messages = []  # Record sent messages for test verification
    
    async def send_notification(self, user_id: int, message: str) -> bool:
        # Simulate sending, actually just record
        notification = {
            "user_id": user_id,
            "message": message,
            "timestamp": "2024-01-01T10:00:00"
        }
        self.sent_messages.append(notification)
        print(f"🧪 Mock notification sent: {notification}")
        return True
    
    def get_sent_messages(self) -> List[Dict]:
        """Get sent messages (for testing)"""
        return self.sent_messages.copy()
```

### 2. Mock Mode Toggle Control

#### 2.1 Environment Variable Control

Configure in the environment variable file (`.env`):

```bash
# Development environment configuration
MOCK_MODE=true

# Production environment configuration
# MOCK_MODE=false
```

#### 2.2 Dynamic Switching in Code

```python
import os
from core.di.utils import enable_mock_mode, disable_mock_mode, get_bean_by_type

def setup_mock_mode():
    """Set Mock mode based on environment variable"""
    if os.getenv("MOCK_MODE", "false").lower() == "true":
        enable_mock_mode()
        print("🧪 Mock mode enabled")
    else:
        disable_mock_mode()
        print("🔧 Using real implementation")

# Call during application startup
def initialize_app():
    setup_mock_mode()
    
    # Now the implementation will automatically switch based on Mock mode
    user_service = get_bean_by_type(UserService)
    return user_service
```

#### 2.3 Automatic Control at Startup

The application will automatically check the `MOCK_MODE` environment variable at startup:

```python
# Implementation in run.py
if os.getenv("MOCK_MODE") and os.getenv("MOCK_MODE").lower() == "true":
    enable_mock_mode()
    logger.info("🚀 Mock mode enabled")
else:
    logger.info("🚀 Mock mode disabled")
```

### 3. Conditional Mock Implementation

Use different Mock implementations based on different conditions:

```python
@mock_impl("mock_fast_notification")
class FastMockNotificationService(NotificationService):
    """Fast Mock notification (for testing)"""
    
    async def send_notification(self, user_id: int, message: str) -> bool:
        print(f"⚡ Fast Mock notification: User {user_id} - {message}")
        return True

@mock_impl("mock_slow_notification") 
class SlowMockNotificationService(NotificationService):
    """Slow Mock notification (for performance testing)"""
    
    async def send_notification(self, user_id: int, message: str) -> bool:
        import asyncio
        await asyncio.sleep(0.1)  # Simulate network delay
        print(f"🐌 Slow Mock notification: User {user_id} - {message}")
        return True

# Choose Mock implementation based on test type
def setup_test_environment(test_type: str):
    enable_mock_mode()
    
    if test_type == "performance":
        # Use slow Mock for performance testing
        from core.di.utils import register_bean
        slow_mock = SlowMockNotificationService()
        register_bean(NotificationService, slow_mock, "mock_notification")
    else:
        # Use fast Mock for normal testing
        fast_mock = FastMockNotificationService()
        register_bean(NotificationService, fast_mock, "mock_notification")
```

## 🏗️ Dependency Injection Best Practices

### 1. Interface Design Principles

- **Single Responsibility**: Each interface should be responsible for one clear responsibility
- **Interface Segregation**: Clients should not depend on interfaces they don't need
- **Dependency Inversion**: High-level modules should not depend on low-level modules; both should depend on abstractions

```python
# Good design: Clear responsibilities
class UserRepository(ABC):
    @abstractmethod
    def find_by_id(self, user_id: int) -> Optional[Dict]:
        pass

class UserValidator(ABC):
    @abstractmethod
    def validate(self, user: Dict) -> bool:
        pass

# Avoid this design: Mixed responsibilities
class UserService(ABC):  # Not recommended: Mixes storage and validation responsibilities
    @abstractmethod
    def find_by_id(self, user_id: int) -> Optional[Dict]:
        pass
    
    @abstractmethod
    def validate(self, user: Dict) -> bool:
        pass
```

### 2. Decorator Usage Guidelines

```python
# Import from specific decorator modules
from core.di.decorators import repository, service, component, mock_impl, factory

# Data access layer
@repository("user_repository")
class UserRepositoryImpl(UserRepository):
    pass

# Business service layer
@service("user_service")
class UserService:
    pass

# General components
@component("config_manager")
class ConfigManager:
    pass

# Mock implementation
@mock_impl("mock_external_api")
class MockExternalApiClient(ExternalApiClient):
    pass

# Factory method
@factory(DatabaseConnection, "db_connection")
def create_database_connection() -> DatabaseConnection:
    config = load_config()
    return DatabaseConnection(config.db_url)
```

### 3. Circular Dependency Handling

Use lazy injection to avoid circular dependencies:

```python
from core.di.decorators import service
from core.di.utils import lazy_inject

@service("order_service")
class OrderService:
    def __init__(self):
        # Get dependencies lazily to avoid circular dependencies
        self.user_service_lazy = lazy_inject(UserService)
        self.payment_service_lazy = lazy_inject(PaymentService)
    
    def create_order(self, order_data: Dict) -> Dict:
        user_service = self.user_service_lazy()  # Get only when called
        payment_service = self.payment_service_lazy()
        
        # Business logic
        user = user_service.get_user(order_data["user_id"])
        payment_result = payment_service.process_payment(order_data["amount"])
        
        return {"order_id": 123, "status": "created"}
```

## ⚙️ Development Environment Configuration

**Note**: Before starting development, run `make dev-setup` to set up the development environment (sync dependencies + install code check hooks).

### 1. Environment Variable Configuration

Create `.env` file:

```bash
# Development environment configuration
ENVIRONMENT=development
DEBUG=true

# Mock mode configuration
MOCK_MODE=true

# Logging configuration
LOG_LEVEL=DEBUG
LOG_FORMAT=detailed

# External service configuration (use test addresses in development)
EXTERNAL_API_URL=https://api-test.example.com
DATABASE_URL=postgresql://dbuser:dbpass@localhost:5432/app_dev
REDIS_URL=redis://localhost:6379/0
```

### 2. Development Script Template

For development scripts that need to be run, add environment initialization at the beginning of the script:

```python
#!/usr/bin/env python3
"""
Development script template - Data processing/testing scripts, etc.
"""
import os

# ============= Development Environment Initialization (Must be at top) =============
# 1. Set environment variables and Python path
from common_utils.load_env import setup_environment
setup_environment(load_env_file_name=".env", check_env_var="MONGODB_HOST")

# 2. Enable Mock mode (enabled by default in development environment)
from core.di.utils import enable_mock_mode
if os.getenv("MOCK_MODE", "true").lower() == "true":
    enable_mock_mode()
    print("🧪 Development script: Mock mode enabled")

# 3. Initialize dependency injection
from application_startup import setup_all
setup_all()
# ================================================

# Now you can normally import and use project modules
from core.di.utils import get_bean_by_type
from core.observation.logger import get_logger

logger = get_logger(__name__)

def main():
    """Script main logic"""
    logger.info("🚀 Development script execution started")
    
    # Example: Use dependency injection to get services
    # user_service = get_bean_by_type(UserService)
    # result = user_service.process_data()
    
    # Your script logic...
    
    logger.info("✅ Development script execution completed")

if __name__ == "__main__":
    main()
```

#### Actual Usage Example

```python
#!/usr/bin/env python3
"""
User data migration script
"""
import os

# ============= Development Environment Initialization =============
from common_utils.load_env import setup_environment
setup_environment(load_env_file_name=".env")

from core.di.utils import enable_mock_mode
if os.getenv("MOCK_MODE", "true").lower() == "true":
    enable_mock_mode()
    print("🧪 Data migration script: Using Mock data")

from application_startup import setup_all
setup_all()
# =======================================

from core.di.utils import get_bean_by_type
from core.observation.logger import get_logger

logger = get_logger(__name__)

def migrate_user_data():
    """Migrate user data"""
    logger.info("Starting user data migration...")
    
    # Get service (will automatically use Mock implementation if Mock mode is enabled)
    user_service = get_bean_by_type(UserService)
    
    # Process data migration
    users = user_service.get_all_users()
    for user in users:
        # Migration logic...
        logger.info(f"Migrating user: {user['name']}")
    
    logger.info("User data migration completed")

if __name__ == "__main__":
    migrate_user_data()
```

### 3. Development Startup Methods

#### Run Development Scripts
```bash
# Use bootstrap to run scripts with full application context
uv run python src/bootstrap.py your_dev_script.py
uv run python src/bootstrap.py migrate_data.py
uv run python src/bootstrap.py test_service.py
```

#### Start Development Service
```bash
# Start web service (automatically loads .env file)
uv run python src/run.py

# Or set environment variable and start
export MOCK_MODE=true
uv run python src/run.py
```

#### VS Code Debug Configuration

The project already includes launch configurations in `.vscode/launch.json`. Press `F5` or use the **Run and Debug** panel to select:

- `Python 调试程序: run` — Start API service
- `Python 调试程序: task` — Start Task Worker
- `Python 调试程序: run_this_file` — Run currently open file via bootstrap

### 4. Mock Mode Verification

After starting the application, you can confirm Mock mode status through logs:

```bash
# Mock mode status will be displayed when starting the application
python run.py

# Output example:
# 🚀 Mock mode enabled  (when MOCK_MODE=true)
# 🚀 Mock mode disabled  (when MOCK_MODE=false or not set)
```

## 📝 Practical Development Example

### Complete Development Workflow Example

```python
from core.di.decorators import service, mock_impl

# 1. Define interface
class PaymentProcessor(ABC):
    @abstractmethod
    async def process_payment(self, amount: float, payment_method: str) -> Dict:
        pass

# 2. Implement real service
@service("stripe_payment")
class StripePaymentProcessor(PaymentProcessor):
    async def process_payment(self, amount: float, payment_method: str) -> Dict:
        # Real Stripe API call
        return {"transaction_id": "stripe_123", "status": "success"}

# 3. Implement Mock service
@mock_impl("mock_payment")
class MockPaymentProcessor(PaymentProcessor):
    async def process_payment(self, amount: float, payment_method: str) -> Dict:
        # Mock implementation for development and testing
        return {"transaction_id": "mock_123", "status": "success"}

# 4. Business service uses interface
@service("order_service")
class OrderService:
    def __init__(self, payment_processor: PaymentProcessor):
        self.payment_processor = payment_processor
    
    async def place_order(self, order_data: Dict) -> Dict:
        # Process payment
        payment_result = await self.payment_processor.process_payment(
            order_data["amount"], 
            order_data["payment_method"]
        )
        
        if payment_result["status"] == "success":
            return {"order_id": 456, "status": "confirmed"}
        else:
            return {"error": "Payment failed"}

# 5. Use during development
def development_workflow():
    from core.di.utils import enable_mock_mode, get_bean_by_type
    
    # Enable Mock mode for development
    enable_mock_mode()
    
    # Get service (automatically uses Mock implementation)
    order_service = get_bean_by_type(OrderService)
    
    # Test business logic without real payment
    order_data = {
        "amount": 99.99,
        "payment_method": "credit_card"
    }
    
    result = await order_service.place_order(order_data)
    print(f"Order result: {result}")

# 6. Use in production environment
def production_workflow():
    from core.di.utils import disable_mock_mode, get_bean_by_type
    
    # Disable Mock mode to use real service
    disable_mock_mode()
    
    # Get service (automatically uses real implementation)
    order_service = get_bean_by_type(OrderService)
    
    # Real business processing
    result = await order_service.place_order(order_data)
    print(f"Real order result: {result}")
```

This approach allows developers to:

1. **Parallel Development**: Frontend and backend can develop simultaneously, with backend using Mock data
2. **Fast Testing**: No need to set up complete external service environments
3. **Decoupled Development**: Each module can be developed and tested independently
4. **Flexible Switching**: Switch between Mock and real implementations through simple configuration

This architecture greatly improves development efficiency and code quality while maintaining system testability and maintainability.

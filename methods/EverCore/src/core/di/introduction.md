# DI (依赖注入) 模块介绍

本模块提供了一个完整的Python依赖注入（Dependency Injection, DI）框架，支持接口多实现、Primary机制、Mock模式、Factory功能、循环依赖检测和自动扫描等核心特性。该框架参考了Spring Framework和Google Guice的设计理念，结合Python的特性进行了优化和扩展。

## 目录

- [目录结构](#目录结构)
- [第一部分：原理详解](#第一部分原理详解)
  - [1. 依赖注入基础原理](#1-依赖注入基础原理)
  - [2. 容器管理机制](#2-容器管理机制)
  - [3. Bean生命周期](#3-bean生命周期)
  - [4. 依赖解析算法](#4-依赖解析算法)
  - [5. Bean排序策略](#5-bean排序策略)
  - [6. 组件扫描机制](#6-组件扫描机制)
  - [7. 循环依赖处理](#7-循环依赖处理)
- [第二部分：使用指南](#第二部分使用指南)
  - [1. 快速开始](#1-快速开始)
  - [2. 装饰器详解](#2-装饰器详解)
  - [3. 容器API详解](#3-容器api详解)
  - [4. 高级特性](#4-高级特性)
  - [5. 最佳实践](#5-最佳实践)
  - [6. 实战案例](#6-实战案例)
  - [7. 性能优化](#7-性能优化)
  - [8. 故障排查](#8-故障排查)

## 目录结构

```
di/
├── __init__.py                      # 包标识文件，导出核心API
├── introduction.md                  # 本介绍文档
├── bean_definition.py               # Bean定义和作用域枚举
├── bean_order_strategy.py           # Bean排序策略实现
├── container.py                     # DI容器核心实现
├── decorators.py                    # 依赖注入装饰器集合
├── exceptions.py                    # 异常定义和层次结构
├── scan_context.py                  # 扫描上下文管理
├── scan_path_registry.py            # 扫描路径注册表
├── scanner.py                       # 组件扫描器实现
├── utils.py                         # 工具函数和辅助方法
└── tests/                           # 测试目录
    ├── introduction.md              # 测试模块介绍
    ├── test_fixtures.py             # 测试Fixtures和Mock对象
    ├── test_bean_order_strategy.py  # Bean排序策略测试（17个测试）
    ├── test_di_container.py         # Container容器测试（32个测试）
    └── test_di_scanner.py           # Scanner扫描器测试（23个测试）
```

---

# 第一部分：原理详解

## 1. 依赖注入基础原理

### 1.1 什么是依赖注入？

依赖注入（Dependency Injection, DI）是一种设计模式，用于实现**控制反转**（Inversion of Control, IoC）。其核心思想是：**将对象的依赖关系从对象内部转移到外部容器管理**。

**传统方式（硬编码依赖）：**

```python
class UserService:
    def __init__(self):
        # 硬编码依赖，紧耦合
        self.repository = MySQLUserRepository()
```

**DI方式（依赖注入）：**

```python
class UserService:
    def __init__(self, repository: UserRepository):
        # 依赖由外部注入，松耦合
        self.repository = repository
```

### 1.2 DI的核心优势

1. **降低耦合度**：组件之间通过接口交互，不依赖具体实现
2. **提高可测试性**：可以轻松注入Mock对象进行单元测试
3. **增强可维护性**：修改实现类不影响使用方
4. **支持多态**：同一接口可以有多个实现，运行时动态选择
5. **集中管理**：统一管理对象的创建、生命周期和依赖关系

### 1.3 本框架的设计理念

本DI框架基于以下设计原则：

- **约定优于配置**：使用装饰器简化配置，减少样板代码
- **类型安全**：充分利用Python类型提示，提供类型安全的依赖解析
- **灵活扩展**：支持自定义Bean排序策略、扫描策略等
- **开发友好**：提供Mock模式、详细的错误信息、循环依赖检测等开发辅助功能
- **高性能**：多级缓存、并行扫描、懒加载等优化手段

## 2. 容器管理机制

### 2.1 容器的核心职责

DI容器（`DIContainer`）是框架的核心，负责：

1. **Bean注册**：记录Bean的定义信息（类型、名称、作用域、优先级等）
2. **Bean实例化**：根据作用域策略创建Bean实例
3. **依赖解析**：分析Bean的依赖关系，递归注入依赖
4. **生命周期管理**：管理单例Bean的缓存和销毁
5. **类型匹配**：根据类型或名称查找合适的Bean实现

### 2.2 容器的数据结构

容器内部维护以下核心数据结构：

```python
class DIContainer:
    # Bean定义存储：bean_name -> BeanDefinition
    _bean_definitions: Dict[str, BeanDefinition]
    
    # 类型索引：bean_type -> [bean_name1, bean_name2, ...]
    _type_to_beans: Dict[type, List[str]]
    
    # 单例缓存：bean_name -> bean_instance
    _singletons: Dict[str, Any]
    
    # 继承关系缓存：(bean_type, target_type) -> bool
    _subclass_cache: Dict[Tuple[type, type], bool]
    
    # 候选Bean缓存：target_type -> [bean_name1, bean_name2, ...]
    _candidate_cache: Dict[type, List[str]]
```

### 2.3 单例模式实现

容器本身是单例，通过`get_container()`获取全局唯一实例：

```python
_container_instance = None
_container_lock = threading.RLock()

def get_container() -> DIContainer:
    global _container_instance
    if _container_instance is None:
        with _container_lock:
            if _container_instance is None:
                _container_instance = DIContainer()
    return _container_instance
```

## 3. Bean生命周期

### 3.1 Bean的作用域

框架支持三种Bean作用域（Scope）：

#### 3.1.1 Singleton（单例）

- **特点**：容器中只存在一个Bean实例，全局共享
- **适用场景**：无状态服务、配置对象、数据库连接池等
- **实现原理**：首次创建后缓存在`_singletons`字典中，后续请求直接返回缓存实例

```python
@service("user_service")  # 默认是Singleton
class UserService:
    pass
```

#### 3.1.2 Prototype（原型）

- **特点**：每次请求都创建新的Bean实例
- **适用场景**：有状态对象、线程不安全的对象、临时对象等
- **实现原理**：每次调用都执行`bean_type()`创建新实例，不使用缓存

```python
@prototype("request_context")
class RequestContext:
    def __init__(self):
        self.data = {}
```

#### 3.1.3 Factory（工厂）

- **特点**：通过工厂方法创建Bean，支持复杂的创建逻辑
- **适用场景**：需要配置的对象、需要资源初始化的对象、第三方库对象等
- **实现原理**：注册时保存工厂方法，创建时调用工厂方法

```python
@factory(bean_type=DatabaseConnection, name="db_conn")
def create_db_connection() -> DatabaseConnection:
    config = load_config()
    return DatabaseConnection(**config)
```

### 3.2 Bean的生命周期流程

```
1. 定义阶段
   ↓
2. 注册阶段（通过装饰器或手动注册）
   ↓
3. 扫描阶段（ComponentScanner扫描并注册Bean）
   ↓
4. 解析阶段（容器解析Bean定义，建立类型索引）
   ↓
5. 实例化阶段（根据作用域创建Bean实例）
   ↓
6. 依赖注入阶段（递归解析并注入依赖）
   ↓
7. 使用阶段（应用代码使用Bean）
   ↓
8. 销毁阶段（容器清理时销毁单例Bean）
```

## 4. 依赖解析算法

### 4.1 解析流程

当调用`container.get_bean_by_type(UserRepository)`时，容器执行以下步骤：

```
1. 查找候选Bean
   - 检查缓存：_candidate_cache
   - 遍历：_type_to_beans[UserRepository]
   - 类型匹配：issubclass(bean_type, UserRepository)
   
2. 过滤和排序
   - 应用Mock模式过滤（如果启用）
   - 应用Bean排序策略
   
3. 选择最佳Bean
   - 取排序后的第一个Bean
   - 如果有多个相同优先级，抛出异常
   
4. 实例化Bean
   - 检查单例缓存
   - 根据作用域创建实例
   - 检测循环依赖
   
5. 注入依赖
   - 分析构造函数参数
   - 递归解析依赖
   - 调用构造函数创建实例
   
6. 缓存和返回
   - 单例Bean缓存到_singletons
   - 返回Bean实例
```

### 4.2 类型匹配机制

框架支持两种类型匹配方式：

#### 4.2.1 直接匹配（Exact Match）

Bean类型与目标类型完全一致：

```python
# 注册: MySQLUserRepository
# 查询: MySQLUserRepository
# 结果: 直接匹配 ✓
```

#### 4.2.2 实现类匹配（Subclass Match）

Bean类型是目标类型的子类：

```python
# 注册: MySQLUserRepository(继承UserRepository)
# 查询: UserRepository
# 结果: 实现类匹配 ✓
```

### 4.3 循环依赖检测

容器使用**依赖栈**检测循环依赖：

```python
def _check_circular_dependency(self, bean_name: str, dependency_stack: List[str]):
    if bean_name in dependency_stack:
        # 检测到循环依赖
        cycle = dependency_stack + [bean_name]
        raise CircularDependencyError(cycle)
```

**示例：**

```python
# ServiceA -> ServiceB -> ServiceC -> ServiceA
# 依赖栈: [ServiceA, ServiceB, ServiceC]
# 当尝试解析ServiceA时，检测到循环
```

## 5. Bean排序策略

### 5.1 排序算法原理

当一个接口有多个实现时，容器需要选择最合适的实现。`BeanOrderStrategy`定义了选择规则：

```python
def sort_beans(
    self, 
    candidates: List[Tuple[str, BeanDefinition]], 
    target_type: type
) -> List[Tuple[str, BeanDefinition]]:
    """
    优先级规则（从高到低）：
    1. is_mock: Mock Bean > 非Mock Bean（仅在Mock模式下）
    2. 匹配方式: 直接匹配 > 实现类匹配
    3. is_primary: Primary Bean > 非Primary Bean
    4. scope: Factory Bean > 非Factory Bean
    """
```

### 5.2 排序规则详解

#### 规则1：Mock优先（仅Mock模式）

```python
# Mock模式下
@mock_impl("mock_repo")              # 优先级：1
class MockUserRepository: pass

@repository("real_repo")             # 优先级：2
class RealUserRepository: pass
```

#### 规则2：匹配方式

```python
# 查询: UserRepository
@repository("exact_match")           # 优先级：1（直接匹配）
class UserRepository: pass

@repository("subclass_match")        # 优先级：2（实现类匹配）
class MySQLUserRepository(UserRepository): pass
```

#### 规则3：Primary标记

```python
@repository("primary_repo", primary=True)   # 优先级：1
class PrimaryRepository: pass

@repository("normal_repo")                  # 优先级：2
class NormalRepository: pass
```

#### 规则4：作用域

```python
@factory(bean_type=Connection, name="factory_conn")  # 优先级：1
def create_connection(): pass

@component("singleton_conn")                         # 优先级：2
class Connection: pass
```

### 5.3 自定义排序策略

可以通过继承`BeanOrderStrategy`实现自定义排序：

```python
class CustomOrderStrategy(BeanOrderStrategy):
    def sort_beans(self, candidates, target_type):
        # 自定义排序逻辑
        return sorted(candidates, key=lambda x: self._custom_score(x))
    
    def _custom_score(self, candidate):
        # 计算候选Bean的分数
        pass

# 设置自定义策略
container.set_order_strategy(CustomOrderStrategy())
```

## 6. 组件扫描机制

### 6.1 扫描原理

`ComponentScanner`负责自动发现和注册带有DI装饰器的类：

```
1. 扫描路径收集
   - 添加扫描路径（文件系统路径）
   - 添加扫描包（Python包路径）
   
2. 文件遍历
   - 递归遍历目录
   - 应用排除规则（__pycache__, .pyc, test_等）
   
3. 模块导入
   - 动态导入Python模块
   - 捕获导入错误
   
4. Bean发现
   - 检查模块成员
   - 识别带有_di_metadata的类
   
5. Bean注册
   - 调用容器的register_bean方法
   - 记录注册日志
```

### 6.2 并行扫描

为了提高性能，扫描器支持多线程并行扫描：

```python
def scan(self, parallel: bool = True, max_workers: int = 4):
    if parallel:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._scan_file, file) 
                      for file in files]
            for future in as_completed(futures):
                result = future.result()
```

### 6.3 扫描策略

#### 包含模式（Include）

```python
scanner.add_scan_path("/app/services")      # 扫描services目录
scanner.add_scan_package("myapp.repos")     # 扫描repos包
```

#### 排除模式（Exclude）

```python
scanner.exclude_pattern("test_")            # 排除测试文件
scanner.exclude_pattern("_mock")            # 排除Mock文件
scanner.exclude_pattern("__pycache__")      # 排除缓存目录
```

## 7. 循环依赖处理

### 7.1 循环依赖的类型

#### 类型1：直接循环依赖

```python
class ServiceA:
    def __init__(self, service_b: ServiceB): pass

class ServiceB:
    def __init__(self, service_a: ServiceA): pass

# ServiceA -> ServiceB -> ServiceA
```

#### 类型2：间接循环依赖

```python
class ServiceA:
    def __init__(self, service_b: ServiceB): pass

class ServiceB:
    def __init__(self, service_c: ServiceC): pass

class ServiceC:
    def __init__(self, service_a: ServiceA): pass

# ServiceA -> ServiceB -> ServiceC -> ServiceA
```

### 7.2 检测机制

容器使用**依赖栈**在实例化过程中检测循环依赖：

```python
dependency_stack = []

def get_bean(self, bean_name):
    if bean_name in dependency_stack:
        raise CircularDependencyError(dependency_stack + [bean_name])
    
    dependency_stack.append(bean_name)
    try:
        # 创建Bean实例
        instance = self._create_instance(bean_name)
    finally:
        dependency_stack.pop()
    
    return instance
```

### 7.3 解决方案

#### 方案1：延迟注入（推荐）

```python
class ServiceA:
    def __init__(self):
        self._service_b = None
    
    @property
    def service_b(self):
        if self._service_b is None:
            self._service_b = get_container().get_bean("service_b")
        return self._service_b
```

#### 方案2：重构设计

```python
# 提取公共依赖到第三个类
class ServiceA:
    def __init__(self, common: CommonService): pass

class ServiceB:
    def __init__(self, common: CommonService): pass

class CommonService:
    # 不依赖ServiceA和ServiceB
    pass
```

#### 方案3：使用事件机制

```python
class ServiceA:
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.event_bus.subscribe("event_from_b", self.handle_event)

class ServiceB:
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.event_bus.publish("event_from_b", data)
```

---

# 第二部分：使用指南

## 1. 快速开始

### 1.1 安装和导入

```python
# 导入核心模块
from core.di.decorators import component, service, repository, factory
from core.di.container import get_container, DIContainer
from core.di.scanner import ComponentScanner
```

### 1.2 五分钟快速上手

```python
from abc import ABC, abstractmethod
from core.di.decorators import repository, service
from core.di.container import get_container

# 步骤1: 定义接口
class UserRepository(ABC):
    @abstractmethod
    def save(self, user: dict) -> bool:
        pass
    
    @abstractmethod
    def find_by_id(self, user_id: int) -> dict:
        pass

# 步骤2: 实现接口（使用装饰器自动注册）
@repository("mysql_user_repo", primary=True)
class MySQLUserRepository(UserRepository):
    def save(self, user: dict) -> bool:
        print(f"Saving user to MySQL: {user}")
        return True
    
    def find_by_id(self, user_id: int) -> dict:
        return {"id": user_id, "name": f"User {user_id}", "db": "mysql"}

# 步骤3: 创建服务（依赖注入）
@service("user_service")
class UserService:
    def __init__(self):
        # 从容器获取依赖
        self.repository = get_container().get_bean_by_type(UserRepository)
    
    def create_user(self, name: str) -> dict:
        user = {"id": 1, "name": name}
        self.repository.save(user)
        return user
    
    def get_user(self, user_id: int) -> dict:
        return self.repository.find_by_id(user_id)

# 步骤4: 使用服务
container = get_container()
user_service = container.get_bean("user_service")

# 调用服务方法
new_user = user_service.create_user("Alice")
found_user = user_service.get_user(1)
```

## 2. 装饰器详解

### 2.1 @component - 通用组件

最基础的Bean注册装饰器，适用于任何需要被容器管理的类。

```python
from core.di.decorators import component

@component("my_component")
class MyComponent:
    def do_something(self):
        return "Hello from core.component"

# 使用
comp = get_container().get_bean("my_component")
result = comp.do_something()
```

**参数说明：**
- `name` (str): Bean名称，必须唯一
- `primary` (bool): 是否为Primary Bean，默认False
- `metadata` (dict): 自定义元数据

**完整示例：**

```python
@component(
    name="config_manager",
    primary=True,
    metadata={"version": "1.0", "author": "team"}
)
class ConfigManager:
    def __init__(self):
        self.config = {}
    
    def get(self, key: str, default=None):
        return self.config.get(key, default)
```

### 2.2 @service - 业务服务

专门用于标注业务逻辑层的服务类，语义化更强。

```python
from core.di.decorators import service

@service("order_service")
class OrderService:
    def __init__(self):
        self.order_repo = get_container().get_bean_by_type(OrderRepository)
        self.payment_service = get_container().get_bean("payment_service")
    
    def create_order(self, items: list, user_id: int) -> dict:
        # 业务逻辑
        order = {"items": items, "user_id": user_id, "status": "pending"}
        self.order_repo.save(order)
        return order
    
    def process_payment(self, order_id: int, amount: float) -> bool:
        return self.payment_service.charge(order_id, amount)
```

**命名建议：**
- 使用`_service`后缀
- 采用小写下划线命名法
- 例如：`user_service`, `order_service`, `notification_service`

### 2.3 @repository - 数据访问

专门用于数据访问层（DAO），处理数据持久化逻辑。

```python
from core.di.decorators import repository
from abc import ABC, abstractmethod

# 定义接口
class OrderRepository(ABC):
    @abstractmethod
    def save(self, order: dict) -> bool:
        pass
    
    @abstractmethod
    def find_by_id(self, order_id: int) -> dict:
        pass
    
    @abstractmethod
    def find_by_user(self, user_id: int) -> list:
        pass

# MySQL实现
@repository("mysql_order_repo", primary=True)
class MySQLOrderRepository(OrderRepository):
    def __init__(self):
        self.db = get_container().get_bean("mysql_connection")
    
    def save(self, order: dict) -> bool:
        # 保存到MySQL
        sql = "INSERT INTO orders ..."
        return self.db.execute(sql, order)
    
    def find_by_id(self, order_id: int) -> dict:
        sql = "SELECT * FROM orders WHERE id = %s"
        return self.db.query_one(sql, [order_id])
    
    def find_by_user(self, user_id: int) -> list:
        sql = "SELECT * FROM orders WHERE user_id = %s"
        return self.db.query_all(sql, [user_id])

# MongoDB实现（多实现）
@repository("mongo_order_repo")
class MongoOrderRepository(OrderRepository):
    def __init__(self):
        self.db = get_container().get_bean("mongo_connection")
    
    def save(self, order: dict) -> bool:
        return self.db.orders.insert_one(order).acknowledged
    
    def find_by_id(self, order_id: int) -> dict:
        return self.db.orders.find_one({"id": order_id})
    
    def find_by_user(self, user_id: int) -> list:
        return list(self.db.orders.find({"user_id": user_id}))
```

### 2.4 @controller - 控制器

用于Web层的控制器类（API端点处理器）。

```python
from core.di.decorators import controller

@controller("user_controller")
class UserController:
    def __init__(self):
        self.user_service = get_container().get_bean("user_service")
    
    def get_user_api(self, user_id: int) -> dict:
        """GET /api/users/{user_id}"""
        user = self.user_service.get_user(user_id)
        return {"code": 200, "data": user}
    
    def create_user_api(self, data: dict) -> dict:
        """POST /api/users"""
        user = self.user_service.create_user(data["name"])
        return {"code": 201, "data": user}
```

### 2.5 @injectable - 可注入组件

通用的可注入组件，功能类似`@component`。

```python
from core.di.decorators import injectable

@injectable("cache_manager")
class CacheManager:
    def __init__(self):
        self.cache = {}
    
    def get(self, key: str):
        return self.cache.get(key)
    
    def set(self, key: str, value):
        self.cache[key] = value
```

### 2.6 @mock_impl - Mock实现

专门用于测试的Mock实现，只在Mock模式下生效。

```python
from core.di.decorators import mock_impl

@mock_impl("mock_user_repo")
class MockUserRepository(UserRepository):
    """Mock实现，用于测试"""
    
    def __init__(self):
        self.users = {
            1: {"id": 1, "name": "Mock User 1"},
            2: {"id": 2, "name": "Mock User 2"}
        }
    
    def find_by_id(self, user_id: int) -> dict:
        return self.users.get(user_id, {"id": user_id, "name": "Unknown"})
    
    def save(self, user: dict) -> bool:
        self.users[user["id"]] = user
        return True

# 在测试中启用Mock模式
def test_user_service():
    container = get_container()
    container.enable_mock_mode()  # 启用Mock
    
    # 自动使用MockUserRepository
    service = container.get_bean("user_service")
    user = service.get_user(1)
    
    assert user["name"] == "Mock User 1"
    
    container.disable_mock_mode()  # 恢复正常模式
```

### 2.7 @factory - 工厂方法

使用函数创建Bean，适用于需要复杂初始化的对象。

```python
from core.di.decorators import factory

# 示例1: 数据库连接
@factory(bean_type=DatabaseConnection, name="db_connection")
def create_database_connection() -> DatabaseConnection:
    """工厂方法：创建数据库连接"""
    # 读取配置
    config = {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", 3306)),
        "user": os.getenv("DB_USER", "root"),
        "password": os.getenv("DB_PASSWORD", ""),
        "database": os.getenv("DB_NAME", "mydb")
    }
    
    # 创建连接
    conn = DatabaseConnection(**config)
    
    # 初始化连接池
    conn.initialize_pool(min_size=5, max_size=20)
    
    return conn

# 示例2: 第三方库对象
@factory(bean_type=Redis, name="redis_client")
def create_redis_client() -> Redis:
    """工厂方法：创建Redis客户端"""
    return Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        db=int(os.getenv("REDIS_DB", 0)),
        decode_responses=True
    )

# 示例3: 带初始化逻辑
@factory(bean_type=MessageQueue, name="mq")
def create_message_queue() -> MessageQueue:
    """工厂方法：创建消息队列"""
    mq = MessageQueue(broker_url=os.getenv("MQ_BROKER"))
    
    # 注册消费者
    mq.register_consumer("order_events", handle_order_event)
    mq.register_consumer("user_events", handle_user_event)
    
    # 启动连接
    mq.connect()
    
    return mq
```

### 2.8 @prototype - 原型作用域

每次获取都创建新实例，适用于有状态对象。

```python
from core.di.decorators import prototype

@prototype("request_context")
class RequestContext:
    """请求上下文（有状态）"""
    
    def __init__(self):
        self.user_id = None
        self.session_id = None
        self.data = {}
    
    def set_user(self, user_id: int):
        self.user_id = user_id
    
    def set_data(self, key: str, value):
        self.data[key] = value

# 每次获取都是新实例
ctx1 = get_container().get_bean("request_context")
ctx2 = get_container().get_bean("request_context")
assert ctx1 is not ctx2  # 不是同一个实例

# 示例2: 任务对象
@prototype("data_import_task")
class DataImportTask:
    """数据导入任务（每次创建新任务）"""
    
    def __init__(self):
        self.task_id = str(uuid.uuid4())
        self.status = "pending"
        self.progress = 0
    
    def execute(self, file_path: str):
        self.status = "running"
        # 执行导入逻辑
        self.progress = 100
        self.status = "completed"
```

## 3. 容器API详解

### 3.1 获取容器

```python
from core.di.container import get_container, DIContainer

# 方式1: 获取全局容器（推荐）
container = get_container()

# 方式2: 创建新容器（不推荐，除非有特殊需求）
custom_container = DIContainer()
```

### 3.2 注册Bean

#### 3.2.1 手动注册类

```python
# 注册普通类
container.register_bean(
    bean_type=UserService,
    bean_name="user_service",
    scope=BeanScope.SINGLETON,  # 可选，默认SINGLETON
    is_primary=True,            # 可选，默认False
    is_mock=False,              # 可选，默认False
    metadata={"version": "1.0"} # 可选，默认{}
)

# 注册为Prototype
container.register_bean(
    bean_type=TaskRunner,
    bean_name="task_runner",
    scope=BeanScope.PROTOTYPE
)
```

#### 3.2.2 注册工厂方法

```python
def create_logger():
    return logging.getLogger("myapp")

container.register_factory(
    bean_type=logging.Logger,
    factory_method=create_logger,
    bean_name="app_logger"
)
```

### 3.3 获取Bean

#### 3.3.1 按名称获取

```python
# 按名称获取（最常用）
user_service = container.get_bean("user_service")

# 如果Bean不存在，抛出BeanNotFoundError
try:
    service = container.get_bean("non_existent")
except BeanNotFoundError as e:
    print(f"Bean not found: {e}")
```

#### 3.3.2 按类型获取

```python
# 按类型获取（自动选择最佳实现）
repository = container.get_bean_by_type(UserRepository)

# 如果有多个实现，返回Primary或优先级最高的
# 如果没有实现，抛出BeanNotFoundError

# 指定类型注解
from typing import Type

def get_service(service_type: Type[T]) -> T:
    return container.get_bean_by_type(service_type)
```

#### 3.3.3 获取所有实现

```python
# 获取某个类型的所有实现
all_repos = container.get_beans_by_type(UserRepository)
# 返回: [MySQLUserRepository实例, MongoUserRepository实例, ...]

# 遍历所有实现
for repo in all_repos:
    print(f"Repository: {type(repo).__name__}")
    repo.save(user_data)
```

### 3.4 检查Bean

```python
# 检查Bean是否存在（按名称）
if container.contains_bean("user_service"):
    service = container.get_bean("user_service")

# 检查Bean是否存在（按类型）
if container.contains_bean_by_type(UserRepository):
    repo = container.get_bean_by_type(UserRepository)

# 检查是否有多个实现
repos = container.get_beans_by_type(UserRepository)
if len(repos) > 1:
    print(f"Found {len(repos)} implementations")
```

### 3.5 Mock模式控制

```python
# 启用Mock模式
container.enable_mock_mode()

# 检查Mock模式状态
if container.is_mock_mode():
    print("Mock mode is enabled")

# 禁用Mock模式
container.disable_mock_mode()

# 示例：测试场景
def test_with_mock():
    container.enable_mock_mode()
    try:
        # 测试代码，自动使用Mock实现
        service = container.get_bean("user_service")
        result = service.get_user(1)
        assert result is not None
    finally:
        container.disable_mock_mode()  # 确保恢复
```

### 3.6 容器管理

```python
# 清空容器（危险操作！）
container.clear()

# 获取所有Bean定义
all_beans = container.get_all_bean_definitions()
for bean_name, bean_def in all_beans.items():
    print(f"Bean: {bean_name}, Type: {bean_def.bean_type}")

# 获取Bean定义
bean_def = container.get_bean_definition("user_service")
print(f"Scope: {bean_def.scope}")
print(f"Primary: {bean_def.is_primary}")

# 设置自定义排序策略
custom_strategy = CustomBeanOrderStrategy()
container.set_order_strategy(custom_strategy)
```

## 4. 高级特性

### 4.1 组件扫描

#### 4.1.1 基本扫描

```python
from core.di.scanner import ComponentScanner

# 创建扫描器
scanner = ComponentScanner()

# 添加扫描路径
scanner.add_scan_path("/path/to/myapp/services")
scanner.add_scan_path("/path/to/myapp/repositories")

# 添加扫描包
scanner.add_scan_package("myapp.controllers")
scanner.add_scan_package("myapp.components")

# 执行扫描（自动注册所有带装饰器的类）
scanner.scan()
```

#### 4.1.2 排除规则

```python
# 排除测试文件
scanner.exclude_pattern("test_")
scanner.exclude_pattern("_test")

# 排除Mock文件
scanner.exclude_pattern("mock_")
scanner.exclude_pattern("_mock")

# 排除特定目录
scanner.exclude_pattern("__pycache__")
scanner.exclude_pattern("migrations")

# 执行扫描
scanner.scan()
```

#### 4.1.3 并行扫描

```python
# 开启并行扫描（默认4个线程）
scanner.scan(parallel=True, max_workers=4)

# 关闭并行扫描
scanner.scan(parallel=False)
```

#### 4.1.4 全局扫描配置

```python
from core.di.scan_path_registry import ScanPathRegistry

# 获取全局注册表
registry = ScanPathRegistry.get_instance()

# 添加全局扫描路径
registry.add_scan_path("/path/to/myapp")

# 添加全局排除规则
registry.add_exclude_pattern("test_*")

# 获取所有扫描路径
paths = registry.get_all_scan_paths()
```

### 4.2 接口多实现

#### 4.2.1 定义多个实现

```python
from abc import ABC, abstractmethod

# 定义接口
class MessageSender(ABC):
    @abstractmethod
    def send(self, to: str, content: str) -> bool:
        pass

# 实现1: 邮件发送
@service("email_sender")
class EmailSender(MessageSender):
    def send(self, to: str, content: str) -> bool:
        print(f"Sending email to {to}: {content}")
        return True

# 实现2: 短信发送
@service("sms_sender")
class SmsSender(MessageSender):
    def send(self, to: str, content: str) -> bool:
        print(f"Sending SMS to {to}: {content}")
        return True

# 实现3: 推送通知（Primary）
@service("push_sender", primary=True)
class PushSender(MessageSender):
    def send(self, to: str, content: str) -> bool:
        print(f"Sending push notification to {to}: {content}")
        return True
```

#### 4.2.2 使用多实现

```python
# 获取Primary实现（PushSender）
sender = container.get_bean_by_type(MessageSender)
sender.send("user123", "Hello")

# 获取所有实现
all_senders = container.get_beans_by_type(MessageSender)
for sender in all_senders:
    sender.send("user123", "Broadcast message")

# 按名称获取特定实现
email_sender = container.get_bean("email_sender")
sms_sender = container.get_bean("sms_sender")
```

#### 4.2.3 策略模式应用

```python
@service("notification_service")
class NotificationService:
    def __init__(self):
        # 获取所有消息发送实现
        self.senders = get_container().get_beans_by_type(MessageSender)
    
    def notify_all(self, user: str, message: str):
        """使用所有渠道发送通知"""
        results = []
        for sender in self.senders:
            result = sender.send(user, message)
            results.append((type(sender).__name__, result))
        return results
    
    def notify_by_preference(self, user: str, message: str, channel: str):
        """根据用户偏好选择渠道"""
        sender_map = {
            "email": "email_sender",
            "sms": "sms_sender",
            "push": "push_sender"
        }
        sender_name = sender_map.get(channel)
        if sender_name:
            sender = get_container().get_bean(sender_name)
            return sender.send(user, message)
        return False
```

### 4.3 Primary机制详解

#### 4.3.1 Primary优先级

```python
# 场景：数据库连接有多个实现

@repository("mysql_repo")
class MySQLRepository(DataRepository):
    pass

@repository("postgres_repo", primary=True)  # 标记为Primary
class PostgresRepository(DataRepository):
    pass

@repository("mongo_repo")
class MongoRepository(DataRepository):
    pass

# 获取Bean时，自动返回Primary实现
repo = container.get_bean_by_type(DataRepository)
# 返回: PostgresRepository 实例
```

#### 4.3.2 多个Primary冲突

```python
# 错误示例：同一接口有多个Primary
@repository("repo1", primary=True)
class Repo1(DataRepository):
    pass

@repository("repo2", primary=True)  # 冲突！
class Repo2(DataRepository):
    pass

# 获取时会抛出 PrimaryBeanConflictError
try:
    repo = container.get_bean_by_type(DataRepository)
except PrimaryBeanConflictError as e:
    print(f"Multiple primary beans: {e}")
```

#### 4.3.3 不同接口的Primary

```python
# 不同接口可以各自有Primary，不会冲突

@repository("mysql_user_repo", primary=True)
class MySQLUserRepository(UserRepository):
    pass

@repository("mysql_order_repo", primary=True)
class MySQLOrderRepository(OrderRepository):
    pass

# 两者不冲突，各自是自己接口的Primary
user_repo = container.get_bean_by_type(UserRepository)    # MySQLUserRepository
order_repo = container.get_bean_by_type(OrderRepository)  # MySQLOrderRepository
```

### 4.4 Mock模式详解

#### 4.4.1 Mock模式的使用场景

1. **单元测试**：隔离外部依赖
2. **集成测试**：模拟未完成的模块
3. **开发调试**：快速验证逻辑
4. **演示Demo**：不依赖真实数据源

#### 4.4.2 完整的Mock示例

```python
# 1. 定义接口
class PaymentGateway(ABC):
    @abstractmethod
    def charge(self, amount: float, card: str) -> dict:
        pass

# 2. 真实实现
@service("stripe_payment")
class StripePayment(PaymentGateway):
    def charge(self, amount: float, card: str) -> dict:
        # 调用真实的Stripe API
        response = stripe.charge(amount, card)
        return response

# 3. Mock实现
@mock_impl("mock_payment")
class MockPayment(PaymentGateway):
    """Mock支付，用于测试"""
    
    def charge(self, amount: float, card: str) -> dict:
        # 返回模拟数据
        return {
            "success": True,
            "transaction_id": "MOCK_12345",
            "amount": amount,
            "card": card[-4:]  # 只显示后4位
        }

# 4. 使用Mock模式
def test_payment_flow():
    container = get_container()
    container.enable_mock_mode()
    
    try:
        # 自动使用MockPayment
        payment = container.get_bean_by_type(PaymentGateway)
        result = payment.charge(100.0, "4111111111111111")
        
        assert result["success"] == True
        assert result["amount"] == 100.0
    finally:
        container.disable_mock_mode()
```

#### 4.4.3 环境变量控制Mock

```python
import os
from core.di.container import get_container

# 根据环境变量启用Mock
if os.getenv("ENABLE_MOCK", "false").lower() == "true":
    get_container().enable_mock_mode()

# 或者在应用启动时
def init_app():
    container = get_container()
    if os.getenv("ENV") in ["test", "dev"]:
        container.enable_mock_mode()
        print("Mock mode enabled for testing/development")
```

### 4.5 Bean元数据（Metadata）

#### 4.5.1 定义元数据

```python
@service(
    "user_service",
    metadata={
        "version": "2.0",
        "author": "backend-team",
        "deprecated": False,
        "tags": ["core", "user-management"],
        "rate_limit": 1000
    }
)
class UserService:
    pass
```

#### 4.5.2 读取元数据

```python
# 获取Bean定义
bean_def = container.get_bean_definition("user_service")

# 访问元数据
print(f"Version: {bean_def.metadata.get('version')}")
print(f"Author: {bean_def.metadata.get('author')}")
print(f"Tags: {bean_def.metadata.get('tags')}")

# 根据元数据过滤Bean
all_defs = container.get_all_bean_definitions()
core_beans = [
    name for name, bean_def in all_defs.items()
    if "core" in bean_def.metadata.get("tags", [])
]
print(f"Core beans: {core_beans}")
```

#### 4.5.3 元数据应用场景

```python
# 场景1: 版本管理
@service("api_v1", metadata={"version": "1.0", "deprecated": True})
class ApiV1Service:
    pass

@service("api_v2", metadata={"version": "2.0", "deprecated": False})
class ApiV2Service:
    pass

# 场景2: 权限标记
@service("admin_service", metadata={"require_role": "admin"})
class AdminService:
    pass

# 场景3: 监控标记
@service("critical_service", metadata={"monitoring": "high", "alert_on_error": True})
class CriticalService:
    pass
```

## 5. 最佳实践

### 5.1 接口设计

#### 原则1: 使用抽象基类定义接口

```python
from abc import ABC, abstractmethod

# 好的做法 ✓
class UserRepository(ABC):
    @abstractmethod
    def find_by_id(self, user_id: int) -> dict:
        pass
    
    @abstractmethod
    def save(self, user: dict) -> bool:
        pass

# 不推荐的做法 ✗
class UserRepository:  # 没有ABC，缺少抽象方法
    def find_by_id(self, user_id: int) -> dict:
        pass
```

#### 原则2: 接口职责单一

```python
# 好的做法 ✓ - 职责清晰
class UserRepository(ABC):
    @abstractmethod
    def find_by_id(self, user_id: int) -> dict:
        pass

class UserValidator(ABC):
    @abstractmethod
    def validate(self, user: dict) -> bool:
        pass

# 不推荐的做法 ✗ - 职责混杂
class UserRepository(ABC):
    @abstractmethod
    def find_by_id(self, user_id: int) -> dict:
        pass
    
    @abstractmethod
    def validate_user(self, user: dict) -> bool:  # 不属于Repository职责
        pass
```

### 5.2 命名规范

#### Bean命名

```python
# 好的命名 ✓
@service("user_service")           # 小写下划线
@repository("mysql_user_repo")     # 描述性强
@controller("api_user_controller") # 语义清晰

# 不推荐的命名 ✗
@service("UserService")     # 不要用大驼峰
@service("us")              # 太简短
@service("service1")        # 没有语义
```

#### 类命名

```python
# 好的命名 ✓
class UserService:          # 大驼峰
class MySQLUserRepository:  # 包含实现细节
class ApiUserController:    # 清晰的职责

# 不推荐的命名 ✗
class user_service:         # 不要用小写下划线
class US:                   # 太简短
class Manager:              # 太模糊
```

### 5.3 依赖注入方式

#### 推荐方式: 构造函数注入

```python
# 好的做法 ✓ - 构造函数注入
@service("user_service")
class UserService:
    def __init__(self):
        self.repository = get_container().get_bean_by_type(UserRepository)
        self.validator = get_container().get_bean("user_validator")
    
    def create_user(self, data: dict):
        if self.validator.validate(data):
            return self.repository.save(data)
```

#### 延迟注入

```python
# 适用场景: 避免循环依赖
@service("service_a")
class ServiceA:
    def __init__(self):
        self._service_b = None  # 延迟初始化
    
    @property
    def service_b(self):
        if self._service_b is None:
            self._service_b = get_container().get_bean("service_b")
        return self._service_b
```

### 5.4 Mock开发模式

```python
# 1. 开发初期：先定义接口
class EmailService(ABC):
    @abstractmethod
    def send_email(self, to: str, subject: str, body: str) -> bool:
        pass

# 2. 使用Mock快速开发
@mock_impl("mock_email")
class MockEmailService(EmailService):
    def send_email(self, to: str, subject: str, body: str) -> bool:
        print(f"[MOCK] Email to {to}: {subject}")
        return True

# 3. 开发阶段启用Mock
if os.getenv("ENV") == "dev":
    get_container().enable_mock_mode()

# 4. 后期实现真实逻辑
@service("smtp_email", primary=True)
class SmtpEmailService(EmailService):
    def send_email(self, to: str, subject: str, body: str) -> bool:
        # 真实的SMTP发送逻辑
        pass

# 5. 生产环境禁用Mock，自动切换到真实实现
```

### 5.5 错误处理

```python
from core.di.exceptions import (
    BeanNotFoundError,
    CircularDependencyError,
    PrimaryBeanConflictError
)

# 优雅的错误处理
def get_service_safe(service_type: type, fallback=None):
    """安全获取服务，失败时返回fallback"""
    try:
        return get_container().get_bean_by_type(service_type)
    except BeanNotFoundError:
        if fallback:
            return fallback
        raise

# 检查后再获取
if container.contains_bean_by_type(UserRepository):
    repo = container.get_bean_by_type(UserRepository)
else:
    # 使用默认实现或抛出自定义错误
    raise ApplicationError("UserRepository not configured")
```

## 6. 实战案例

### 6.1 案例1：Web API应用

```python
# 1. 数据层
@repository("pg_user_repo", primary=True)
class PostgresUserRepository(UserRepository):
    def __init__(self):
        self.db = get_container().get_bean("db_connection")
    
    def find_by_id(self, user_id: int) -> dict:
        return self.db.query_one("SELECT * FROM users WHERE id = %s", [user_id])
    
    def find_by_email(self, email: str) -> dict:
        return self.db.query_one("SELECT * FROM users WHERE email = %s", [email])
    
    def save(self, user: dict) -> bool:
        return self.db.execute("INSERT INTO users ...", user)

# 2. 业务层
@service("user_service")
class UserService:
    def __init__(self):
        self.user_repo = get_container().get_bean_by_type(UserRepository)
        self.email_service = get_container().get_bean("email_service")
        self.cache = get_container().get_bean("cache_manager")
    
    def get_user(self, user_id: int) -> dict:
        # 先查缓存
        cache_key = f"user:{user_id}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        # 查数据库
        user = self.user_repo.find_by_id(user_id)
        
        # 写缓存
        if user:
            self.cache.set(cache_key, user, ttl=300)
        
        return user
    
    def register_user(self, email: str, password: str) -> dict:
        # 检查邮箱是否已存在
        existing = self.user_repo.find_by_email(email)
        if existing:
            raise ValueError("Email already exists")
        
        # 创建用户
        user = {"email": email, "password": hash_password(password)}
        self.user_repo.save(user)
        
        # 发送欢迎邮件
        self.email_service.send_email(
            to=email,
            subject="Welcome!",
            body="Welcome to our platform"
        )
        
        return user

# 3. 控制层
@controller("user_controller")
class UserController:
    def __init__(self):
        self.user_service = get_container().get_bean("user_service")
    
    def get_user_api(self, user_id: int) -> dict:
        """GET /api/users/{user_id}"""
        try:
            user = self.user_service.get_user(user_id)
            return {"code": 200, "data": user}
        except Exception as e:
            return {"code": 500, "error": str(e)}
    
    def register_api(self, data: dict) -> dict:
        """POST /api/users/register"""
        try:
            user = self.user_service.register_user(
                email=data["email"],
                password=data["password"]
            )
            return {"code": 201, "data": user}
        except ValueError as e:
            return {"code": 400, "error": str(e)}

# 4. 应用启动
def init_application():
    # 扫描组件
    scanner = ComponentScanner()
    scanner.add_scan_path("/app/repositories")
    scanner.add_scan_path("/app/services")
    scanner.add_scan_path("/app/controllers")
    scanner.scan()
    
    # 获取控制器
    controller = get_container().get_bean("user_controller")
    return controller
```

### 6.2 案例2：数据处理管道

```python
# 1. 定义处理器接口
class DataProcessor(ABC):
    @abstractmethod
    def process(self, data: dict) -> dict:
        pass

# 2. 实现多个处理器
@component("data_cleaner")
class DataCleaner(DataProcessor):
    def process(self, data: dict) -> dict:
        # 清洗数据
        cleaned = {k: v.strip() if isinstance(v, str) else v 
                  for k, v in data.items()}
        return cleaned

@component("data_validator")
class DataValidator(DataProcessor):
    def process(self, data: dict) -> dict:
        # 验证数据
        required_fields = ["name", "email"]
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing field: {field}")
        return data

@component("data_enricher")
class DataEnricher(DataProcessor):
    def process(self, data: dict) -> dict:
        # 丰富数据
        data["processed_at"] = get_now_with_timezone()
        data["version"] = "1.0"
        return data

# 3. 管道服务
@service("data_pipeline")
class DataPipeline:
    def __init__(self):
        # 获取所有处理器
        self.processors = get_container().get_beans_by_type(DataProcessor)
    
    def execute(self, data: dict) -> dict:
        """执行处理管道"""
        result = data
        for processor in self.processors:
            try:
                result = processor.process(result)
                print(f"Processed by {type(processor).__name__}")
            except Exception as e:
                print(f"Error in {type(processor).__name__}: {e}")
                raise
        return result

# 4. 使用
pipeline = get_container().get_bean("data_pipeline")
raw_data = {"name": "  Alice  ", "email": "alice@example.com"}
processed_data = pipeline.execute(raw_data)
```

### 6.3 案例3：多租户系统

```python
# 1. 租户上下文
@prototype("tenant_context")
class TenantContext:
    def __init__(self):
        self.tenant_id = None
        self.tenant_name = None
    
    def set_tenant(self, tenant_id: str, tenant_name: str):
        self.tenant_id = tenant_id
        self.tenant_name = tenant_name

# 2. 多租户Repository
@repository("multi_tenant_repo")
class MultiTenantRepository(DataRepository):
    def __init__(self):
        self.db = get_container().get_bean("db_connection")
    
    def _get_tenant_id(self) -> str:
        """从上下文获取租户ID"""
        context = get_container().get_bean("tenant_context")
        if not context.tenant_id:
            raise ValueError("Tenant context not set")
        return context.tenant_id
    
    def find_all(self) -> list:
        tenant_id = self._get_tenant_id()
        return self.db.query_all(
            "SELECT * FROM data WHERE tenant_id = %s",
            [tenant_id]
        )
    
    def save(self, data: dict) -> bool:
        tenant_id = self._get_tenant_id()
        data["tenant_id"] = tenant_id
        return self.db.execute("INSERT INTO data ...", data)

# 3. 使用
def handle_request(request):
    # 每个请求使用新的上下文（Prototype）
    context = get_container().get_bean("tenant_context")
    context.set_tenant(
        tenant_id=request.headers.get("X-Tenant-ID"),
        tenant_name=request.headers.get("X-Tenant-Name")
    )
    
    # Repository自动使用正确的租户ID
    repo = get_container().get_bean("multi_tenant_repo")
    data = repo.find_all()
    return data
```

## 7. 性能优化

### 7.1 缓存机制

框架内置多级缓存：

```python
# 1. 单例Bean缓存
_singletons: Dict[str, Any]
# 首次创建后缓存，后续直接返回

# 2. 类型匹配缓存
_subclass_cache: Dict[Tuple[type, type], bool]
# 缓存issubclass结果，避免重复计算

# 3. 候选Bean缓存
_candidate_cache: Dict[type, List[str]]
# 缓存类型查找结果，提升查询速度
```

### 7.2 懒加载

```python
# 延迟注册
class LazyService:
    def __init__(self):
        self._repo = None
    
    @property
    def repo(self):
        if self._repo is None:
            self._repo = get_container().get_bean_by_type(UserRepository)
        return self._repo

# 只在首次访问时才解析依赖
```

### 7.3 并行扫描

```python
# 开启并行扫描，提升启动速度
scanner = ComponentScanner()
scanner.add_scan_path("/large/codebase")
scanner.scan(parallel=True, max_workers=8)  # 使用8个线程
```

### 7.4 性能监控

```python
import time

class PerformanceMonitoringContainer(DIContainer):
    def get_bean(self, bean_name: str):
        start = time.time()
        bean = super().get_bean(bean_name)
        duration = time.time() - start
        print(f"Bean '{bean_name}' resolved in {duration:.4f}s")
        return bean
```

## 8. 故障排查

### 8.1 常见错误

#### 错误1: BeanNotFoundError

```python
# 错误信息
BeanNotFoundError: No bean found with name 'user_service'

# 原因1: Bean未注册
# 解决: 检查是否添加了装饰器，是否执行了扫描

# 原因2: Bean名称拼写错误
@service("user_service")  # 注册名
container.get_bean("userService")  # 错误！名称不匹配

# 原因3: 扫描路径不正确
scanner.add_scan_path("/wrong/path")  # 路径错误

# 调试方法
all_beans = container.get_all_bean_definitions()
print(f"Registered beans: {list(all_beans.keys())}")
```

#### 错误2: CircularDependencyError

```python
# 错误信息
CircularDependencyError: Circular dependency detected: ServiceA -> ServiceB -> ServiceA

# 原因: 循环依赖
class ServiceA:
    def __init__(self):
        self.b = get_container().get_bean("service_b")

class ServiceB:
    def __init__(self):
        self.a = get_container().get_bean("service_a")

# 解决方案1: 延迟注入
class ServiceA:
    @property
    def b(self):
        if not hasattr(self, "_b"):
            self._b = get_container().get_bean("service_b")
        return self._b

# 解决方案2: 重构设计
class ServiceA:
    def __init__(self):
        self.common = get_container().get_bean("common_service")

class ServiceB:
    def __init__(self):
        self.common = get_container().get_bean("common_service")
```

#### 错误3: PrimaryBeanConflictError

```python
# 错误信息
PrimaryBeanConflictError: Multiple primary beans found for type UserRepository

# 原因: 多个Primary Bean
@repository("repo1", primary=True)
class Repo1(UserRepository): pass

@repository("repo2", primary=True)  # 冲突
class Repo2(UserRepository): pass

# 解决: 只保留一个Primary
@repository("repo1", primary=True)
class Repo1(UserRepository): pass

@repository("repo2")  # 移除primary=True
class Repo2(UserRepository): pass
```

### 8.2 调试技巧

#### 技巧1: 打印所有Bean

```python
def debug_print_beans():
    container = get_container()
    all_beans = container.get_all_bean_definitions()
    
    print("=" * 60)
    print("Registered Beans:")
    print("=" * 60)
    
    for name, bean_def in all_beans.items():
        print(f"Name: {name}")
        print(f"  Type: {bean_def.bean_type}")
        print(f"  Scope: {bean_def.scope}")
        print(f"  Primary: {bean_def.is_primary}")
        print(f"  Mock: {bean_def.is_mock}")
        print(f"  Metadata: {bean_def.metadata}")
        print("-" * 60)
```

#### 技巧2: 跟踪依赖链

```python
def trace_dependencies(bean_name: str, visited=None):
    if visited is None:
        visited = set()
    
    if bean_name in visited:
        print(f"  [CIRCULAR] {bean_name}")
        return
    
    visited.add(bean_name)
    print(f"Bean: {bean_name}")
    
    # 获取Bean的依赖
    bean_def = container.get_bean_definition(bean_name)
    # ... 分析构造函数参数，递归跟踪
```

#### 技巧3: Mock模式检查

```python
def check_mock_status():
    container = get_container()
    print(f"Mock mode: {container.is_mock_mode()}")
    
    all_beans = container.get_all_bean_definitions()
    mock_beans = [name for name, bean_def in all_beans.items() if bean_def.is_mock]
    
    print(f"Mock beans: {mock_beans}")
```

## 9. 异常类型

```python
from core.di.exceptions import (
    CircularDependencyError,      # 循环依赖错误
    BeanNotFoundError,             # Bean未找到错误
    DuplicateBeanError,            # 重复Bean错误
    FactoryError,                  # 工厂错误
    DependencyResolutionError,     # 依赖解析错误
    MockNotEnabledError,           # Mock未启用错误
    PrimaryBeanConflictError,      # Primary Bean冲突错误
)

# 异常处理示例
try:
    service = container.get_bean("user_service")
except BeanNotFoundError as e:
    print(f"Bean not found: {e}")
except CircularDependencyError as e:
    print(f"Circular dependency: {e}")
except DependencyResolutionError as e:
    print(f"Dependency resolution error: {e}")
```

## 10. 测试

本模块包含完整的测试套件（72个测试用例），详见 [tests/introduction.md](./tests/introduction.md)

```bash
# 运行所有DI测试
PYTHONPATH=/Users/admin/memsys_opensource/src python -m pytest src/core/di/tests/ -v

# 运行特定测试文件
pytest src/core/di/tests/test_di_container.py -v
pytest src/core/di/tests/test_di_scanner.py -v
pytest src/core/di/tests/test_bean_order_strategy.py -v
```

## 11. 相关文档

- [DI框架详细文档（英文）](./README.md)
- [DI框架详细文档（中文）](./README_zh.md)
- [开发指南](../../../docs/dev_docs/development_guide.md)
- [测试模块介绍](./tests/introduction.md)

## 12. 扩展机制

本DI框架支持通过Addon机制扩展：

- [AddonBeanOrderStrategy](../addons/contrib/addon_bean_order_strategy.py) - 扩展Bean排序策略，支持addon_tag优先级
- [Addon测试](../addons/contrib/tests/introduction.md) - Addon扩展的测试

**自定义扩展示例：**

```python
from core.di.bean_order_strategy import BeanOrderStrategy

class CustomOrderStrategy(BeanOrderStrategy):
    """自定义Bean排序策略"""
    
    def sort_beans(self, candidates, target_type):
        # 自定义排序逻辑
        return sorted(candidates, key=lambda x: self._calculate_priority(x))
    
    def _calculate_priority(self, candidate):
        name, bean_def = candidate
        priority = 0
        
        # 自定义规则
        if bean_def.metadata.get("critical"):
            priority += 1000
        if bean_def.metadata.get("version") == "2.0":
            priority += 100
        
        return -priority  # 负数表示优先级高

# 应用自定义策略
container = get_container()
container.set_order_strategy(CustomOrderStrategy())
```

## 13. 总结

### 核心特性

- ✅ **依赖注入**: 自动管理对象依赖关系，降低耦合
- ✅ **接口多实现**: 支持一个接口多个实现，灵活切换
- ✅ **Primary机制**: 智能选择默认实现
- ✅ **Mock模式**: 测试友好，快速开发
- ✅ **Factory支持**: 灵活的对象创建方式
- ✅ **组件扫描**: 自动发现和注册Bean
- ✅ **循环依赖检测**: 及时发现设计问题
- ✅ **高性能**: 多级缓存、并行扫描

### 使用建议

1. **接口优先**: 总是定义接口，面向接口编程
2. **装饰器注册**: 使用装饰器简化配置
3. **合理分层**: Controller -> Service -> Repository
4. **Mock开发**: 利用Mock提升开发效率
5. **避免循环**: 注意依赖关系设计
6. **性能优化**: 启用缓存和并行扫描
7. **错误处理**: 优雅处理DI异常

### 维护者

DI机制维护团队

### 贡献指南

欢迎提交Issue和Pull Request！

---

**最后更新**: 2025-11-18
# Addons 模块介绍

## 模块简介

Addons 模块是 MemSys 的扩展机制实现，通过 Python Entry Points 技术实现插件化架构，支持商业版本与开源版本的代码切分。该模块允许在不修改核心代码的情况下，动态加载和注册外部扩展包的组件。

## 核心概念

### 1. 什么是 Addon

Addon（插件/扩展）是一个独立的功能模块，可以包含：
- **DI 组件**：依赖注入的扫描路径配置
- **异步任务**：后台任务的扫描路径配置

每个 Addon 是一个自包含的功能单元，可以独立开发、测试和部署。

### 2. Entry Points 机制

Entry Points 是 Python 包生态系统中的标准扩展机制：
- 通过 `pyproject.toml` 声明扩展点
- 在包安装时自动注册
- 运行时动态发现和加载
- 无需硬编码依赖关系

### 3. 商业版本切分

通过 Addons 机制实现开源版本（Open Core）与商业版本（Enterprise）的切分：
- **Open Core**：基础功能，以 "core" addon 形式存在
- **Enterprise**：商业功能，以独立 addon 形式存在
- **无依赖关系**：两者通过接口抽象，相互独立
- **优先级机制**：Enterprise addon 的实现优先级高于 Open Core，可自动覆盖

## 目录结构

```
src/core/addons/
├── __init__.py                  # 包标识（空文件）
├── addon_registry.py            # 单个 Addon 注册器
├── addons_registry.py           # 全局 Addons 管理器
├── introduction.md              # 本文档
└── contrib/                     # 第三方贡献的 addons（可选）
```

## 核心组件

### 1. AddonRegistry

单个 Addon 的注册器容器，用于承载一个 addon 的配置。

```python
from core.addons.addon_registry import AddonRegistry

# 创建 addon
addon = AddonRegistry(name="my_addon")

# 注册 DI 扫描路径
di_registry = ScannerPathsRegistry()
di_registry.add_scan_path("/path/to/components")
addon.register_di(di_registry)

# 注册异步任务扫描路径
task_registry = TaskScanDirectoriesRegistry()
task_registry.add_scan_path("/path/to/tasks")
addon.register_asynctasks(task_registry)
```

### 2. AddonsRegistry

全局 Addons 管理器，管理所有已注册的 addons。

```python
from core.addons.addons_registry import ADDONS_REGISTRY

# 注册 addon
ADDONS_REGISTRY.register(addon)

# 从 entry points 自动加载所有 addons
ADDONS_REGISTRY.load_entrypoints()

# 获取所有 addons
all_addons = ADDONS_REGISTRY.get_all()

# 根据名称查找 addon
my_addon = ADDONS_REGISTRY.get_by_name("my_addon")
```

## 工作原理

### 1. 注册阶段

在 `pyproject.toml` 中声明 entry point：

```toml
[project.entry-points."memsys.addons"]
core = "src.addon"
enterprise = "memsys_enterprise.addon"
```

### 2. 加载阶段

系统启动时调用 `ADDONS_REGISTRY.load_entrypoints()`：

1. 扫描所有包的 `memsys.addons` entry point group
2. 根据环境变量 `MEMSYS_ENTRYPOINTS_FILTER` 过滤需要加载的 entrypoint
3. 依次加载每个 entry point 对应的模块
4. 模块导入时自动执行注册代码（`ADDONS_REGISTRY.register(addon)`）
5. 所有 addon 被注册到全局 `ADDONS_REGISTRY` 中

### 3. 使用阶段

系统运行时从 `ADDONS_REGISTRY` 获取所有已注册的 addons：

1. **DI 组件扫描**：合并所有 addon 的 DI 扫描路径，执行组件扫描
2. **异步任务注册**：合并所有 addon 的异步任务路径，注册后台任务
3. **优先级覆盖**：后加载的 addon 组件可以覆盖先加载的同名组件

## 环境变量控制

### MEMSYS_ENTRYPOINTS_FILTER

控制加载哪些 entrypoint，格式为逗号分隔的 entrypoint 名称列表。

```bash
# 只加载 core addon
export MEMSYS_ENTRYPOINTS_FILTER=core

# 加载 core 和 enterprise addons
export MEMSYS_ENTRYPOINTS_FILTER=core,enterprise

# 不设置或为空，加载所有 addons（默认行为）
unset MEMSYS_ENTRYPOINTS_FILTER
```

**注意**：
- 一个 entrypoint 可能包含多个 addon 的注册
- 过滤是基于 entrypoint 名称（`ep.name`），不是 addon 名称

## 使用示例

### 示例 1：Open Core Addon

```python
# src/addon.py
import os
from core.di.scan_path_registry import ScannerPathsRegistry
from core.asynctasks.task_scan_registry import TaskScanDirectoriesRegistry
from common_utils.project_path import get_base_scan_path
from core.addons.addon_registry import AddonRegistry
from core.addons.addons_registry import ADDONS_REGISTRY

# 配置 DI 扫描路径
paths_registry = ScannerPathsRegistry()
paths_registry.add_scan_path(os.path.join(get_base_scan_path(), "component"))
paths_registry.add_scan_path(os.path.join(get_base_scan_path(), "infra_layer"))
paths_registry.add_scan_path(os.path.join(get_base_scan_path(), "agentic_layer"))
paths_registry.add_scan_path(os.path.join(get_base_scan_path(), "biz_layer"))

# 配置异步任务扫描路径
task_directories_registry = TaskScanDirectoriesRegistry()
task_directories_registry.add_scan_path(
    os.path.join(get_base_scan_path(), "infra_layer/adapters/input/jobs")
)

# 创建并注册 core addon
core_addon = AddonRegistry(name="core")
core_addon.register_di(paths_registry)
core_addon.register_asynctasks(task_directories_registry)

ADDONS_REGISTRY.register(core_addon)
```

### 示例 2：Enterprise Addon

```python
# memsys_enterprise/addon.py
import os
from pathlib import Path

# 从 opensource 导入 addon 机制
from core.di.scan_path_registry import ScannerPathsRegistry
from core.asynctasks.task_scan_registry import TaskScanDirectoriesRegistry
from core.addons.addon_registry import AddonRegistry
from core.addons.addons_registry import ADDONS_REGISTRY

# 获取 enterprise 的基础路径
current_file = Path(__file__)
src_dir = current_file.parent
enterprise_base_path = str(src_dir)

# 配置 DI 扫描路径
di_registry = ScannerPathsRegistry()
di_registry.add_scan_path(os.path.join(enterprise_base_path, "infra_layer"))

# 配置异步任务扫描路径
task_registry = TaskScanDirectoriesRegistry()
task_registry.add_scan_path(os.path.join(enterprise_base_path, "infra_layer/adapters/input/jobs"))

# 创建并注册 enterprise addon
enterprise_addon = AddonRegistry(name="enterprise")
enterprise_addon.register_di(di_registry)
enterprise_addon.register_asynctasks(task_registry)

# 注册到全局 ADDONS_REGISTRY（模块导入时自动执行）
ADDONS_REGISTRY.register(enterprise_addon)
```

### 示例 3：在 pyproject.toml 中注册

**Open Core (memsys_opensource/pyproject.toml)**：

```toml
[project.entry-points."memsys.addons"]
core = "src.addon"
```

**Enterprise (memsys_enterprise/pyproject.toml)**：

```toml
[project.entry-points."memsys.addons"]
enterprise = "memsys_enterprise.addon"
```

### 示例 4：系统启动时加载

```python
# src/bootstrap.py 或 src/application_startup.py
from core.addons.addons_registry import ADDONS_REGISTRY

# 加载所有通过 entry points 注册的 addons
ADDONS_REGISTRY.load_entrypoints()

# 获取所有 addons 进行后续处理
all_addons = ADDONS_REGISTRY.get_all()

for addon in all_addons:
    print(f"Loaded addon: {addon.name}")
    
    # 处理 DI 组件扫描
    if addon.has_di():
        for path in addon.di.get_scan_paths():
            # 扫描并注册 DI 组件
            ...
    
    # 处理异步任务注册
    if addon.has_asynctasks():
        for path in addon.asynctasks.get_scan_paths():
            # 扫描并注册异步任务
            ...
```

## 设计优势

### 1. 松耦合

- 通过接口抽象实现解耦
- Addon 之间无硬依赖关系
- 可以独立开发和测试

### 2. 可扩展

- 支持任意数量的 addon
- 新增 addon 无需修改核心代码
- 符合开闭原则（Open-Closed Principle）

### 3. 灵活配置

- 通过环境变量控制加载
- 支持条件性加载
- 方便测试和部署

### 4. 商业友好

- 开源版本与商业版本代码隔离
- 商业功能可以覆盖开源实现
- 便于维护不同版本

## 最佳实践

### 1. 接口抽象优先

在实现商业功能或扩展功能时，首先进行接口抽象：

```python
# core/interface/service/storage_service.py (Open Core)
from abc import ABC, abstractmethod

class StorageService(ABC):
    @abstractmethod
    async def save(self, data: bytes) -> str:
        """保存数据，返回存储ID"""
        pass
```

然后在不同的 addon 中提供不同的实现：

```python
# src/component/storage/local_storage_service.py (Open Core - 本地存储)
from core.interface.service.storage_service import StorageService
from core.di.component import Component

@Component()
class LocalStorageService(StorageService):
    async def save(self, data: bytes) -> str:
        # 本地文件系统存储实现
        ...

# memsys_enterprise/infra_layer/storage/cloud_storage_service.py (Enterprise - 云存储)
from core.interface.service.storage_service import StorageService
from core.di.component import Component

@Component()
class CloudStorageService(StorageService):
    async def save(self, data: bytes) -> str:
        # 云存储实现（S3, OSS, etc.）
        ...
```

### 2. 统一目录结构

确保 Enterprise addon 的目录结构与 Open Core 保持一致：

```
memsys_opensource/src/
├── infra_layer/
│   ├── adapters/
│   ├── services/
│   └── ...
├── agentic_layer/
└── biz_layer/

memsys_enterprise/src/memsys_enterprise/
├── infra_layer/              # 与 opensource 对应
│   ├── adapters/
│   ├── services/
│   └── ...
└── ...                       # 可以有额外的层级
```

### 3. 命名约定

- Addon 名称使用小写字母和下划线
- Entry point 名称建议与 addon 名称保持一致
- 模块文件名统一使用 `addon.py`

### 4. 文档和注释

- 每个 addon 应该有清晰的文档说明
- 注册代码添加注释说明用途
- 接口抽象应该有完整的 docstring

## 故障排查

### 1. Addon 未加载

**问题**：通过 `ADDONS_REGISTRY.get_all()` 看不到某个 addon。

**排查步骤**：
1. 检查 `pyproject.toml` 中 entry point 配置是否正确
2. 确认包是否已安装（`pip list` 或 `uv pip list`）
3. 检查 `MEMSYS_ENTRYPOINTS_FILTER` 环境变量配置
4. 查看日志，确认是否有加载错误

### 2. 组件未找到

**问题**：DI 容器中找不到某个组件。

**排查步骤**：
1. 确认 addon 的 DI 扫描路径是否正确
2. 检查组件类是否添加了 `@Component()` 装饰器
3. 确认组件所在的模块路径是否在扫描范围内
4. 查看 DI 组件扫描日志

### 3. 优先级问题

**问题**：Enterprise 实现没有覆盖 Open Core 实现。

**排查步骤**：
1. 确认两个实现使用的是同一个接口
2. 检查 `@Component()` 装饰器的优先级配置
3. 确认 addon 加载顺序（enterprise 应该在 core 之后）
4. 验证类名和接口名是否一致

## 相关文档

- [依赖注入系统](../di/introduction.md) - DI 容器的使用和原理
- [异步任务系统](../asynctasks/introduction.md) - 后台任务的注册和调度
- [开发指南](../../../docs/dev_docs/development_guide.md) - 整体开发规范和流程

## 总结

Addons 模块是 MemSys 实现插件化架构和商业版本切分的核心机制。通过 Python Entry Points 技术，实现了：

- ✅ 代码解耦和模块化
- ✅ 开源与商业版本隔离
- ✅ 动态加载和灵活配置
- ✅ 接口抽象和实现替换

理解和正确使用 Addons 机制，是进行 MemSys 扩展开发的基础。


# src/skill_evolution/ — 自进化方法评估

本目录包含不同 agent 自进化方法的评估实现。每个子目录对应一种方法，包含该方法独立的评估脚本、配置和文档。

## 评估范式

所有方法共享同一个评估框架（`src/domains/`），支持任意已注册 domain。支持两种评估模式：

### 离线模式（Offline）

1. **训练阶段** — 在 train split 上运行 agent，收集 session 轨迹
2. **提取/学习** — 从 train session 中批量提取可复用知识（skill）
3. **评测阶段** — 在 test split 上对比：baseline（无 skill） vs 有 skill 注入

### 在线模式（Online）

每完成一个 task，立即提取 skill 并更新知识库。后续 task 可以利用之前积累的 skill，实现边跑边学习。

## Split File 格式

支持两种格式：

**Cluster 格式**（适用于有天然分类的数据，如 BrowseComp-Plus 按主题分簇）：

```json
{
  "clusters": {
    "CLUSTER_A": {"train": ["id1", "id2"], "test": ["id3", "id4"]},
    "CLUSTER_B": {"train": [...], "test": [...]}
  }
}
```

**Flat 格式**（适用于无分簇的数据，自动包装为 "default" 簇）：

```json
{
  "train": ["task1", "task2"],
  "test": ["task3", "task4"]
}
```

## 已有方法

| 目录 | 方法 | 模式 | 说明 |
|------|------|------|------|
| `evermemos/` | EverMemOS | 离线/在线 | 通过 EverMemOS API 提取 skill，注入到 prompt |
| 更多方法 | Coming soon | — | — |

## 添加新方法

1. 在 `src/skill_evolution/` 下创建新目录
2. 实现知识提取脚本（离线或在线模式）
3. 实现知识注入 + 评测脚本
4. 编写 README 说明完整流程
5. 使用相同的 split file 保证对比公平

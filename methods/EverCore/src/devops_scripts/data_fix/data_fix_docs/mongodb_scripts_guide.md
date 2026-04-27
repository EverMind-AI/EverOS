# MongoDB 数据维护脚本使用指南

本文档介绍 MongoDB 相关的数据维护脚本的使用方法。

## 脚本列表

- `mongo_add_timestamp_shard.py` - 为 MemCell 集合添加基于 timestamp 的分片配置
- `mongo_fix_episodic_memory_missing_vector.py` - 修复情景记忆文档中缺失的向量字段

---

## 1. mongo_add_timestamp_shard.py

### 功能说明

为 MemCell 集合添加基于 `timestamp` 字段的时间戳分片配置，优化大规模数据的查询和存储性能。

### 使用方式

```bash
# 通过 bootstrap 运行（推荐）
python src/bootstrap.py src/devops_scripts/data_fix/mongo_add_timestamp_shard.py
```

### 参数说明

该脚本无需命令行参数，直接运行即可。

### 工作流程

1. **检查分片集群**：验证当前是否为分片集群环境
2. **启用数据库分片**：为目标数据库启用分片功能
3. **设置分片键**：将 `timestamp` 字段设置为 MemCell 集合的分片键
4. **创建预分片**：自动创建未来 12 个月的预分片点
5. **验证配置**：验证分片配置是否成功

### 分片策略

- **分片键**：`timestamp` 字段（升序）
- **预分片**：按月创建分片点，覆盖未来 12 个月
- **优势**：
  - 时间范围查询性能优化
  - 数据均匀分布到多个分片
  - 避免单个分片过大导致的性能问题

### 注意事项

⚠️ **重要提示**：
- 仅在 MongoDB 分片集群环境中生效
- 如果不是分片集群，脚本会自动跳过
- 分片配置是**不可逆**操作，设置后无法直接修改分片键
- 建议在**数据量较小时**配置分片，避免后期迁移成本

### 使用场景

适用于以下场景：
- MemCell 数据量预计超过 10GB
- 需要优化基于时间范围的查询性能
- 部署了 MongoDB 分片集群环境

### 执行结果示例

```
🔧 开始配置timestamp分片...
✅ 检测到分片集群，共 3 个分片
✅ 数据库 'memsys' 分片已启用
✅ MemCell集合timestamp分片键设置完成
📅 创建分片点: 2025-02-01 00:00:00
📅 创建分片点: 2025-03-01 00:00:00
...
✅ 创建了 12 个预分片点
✅ MemCell集合分片配置验证成功
📊 分片键: {'timestamp': 1}
🎉 timestamp分片配置完成
```

### 如何验证分片是否生效

```javascript
// 在 MongoDB Shell 中执行
use memsys
db.memcells.getShardDistribution()
```

---

## 2. mongo_fix_episodic_memory_missing_vector.py

### 功能说明

修复历史 EpisodicMemory 文档中缺失的向量字段。针对两类文档进行修复：
1. `vector` 字段不存在、为 None 或为空数组的文档
2. `vector_model` 不等于目标模型的文档（需要重新生成向量）

### 使用方式

```bash
# 通过 bootstrap 运行（推荐）
python src/bootstrap.py src/devops_scripts/data_fix/mongo_fix_episodic_memory_missing_vector.py \
  --limit 1000 \
  --batch 200 \
  --concurrency 8 \
  --start-created-at "2025-09-16T20:20:06+00:00" \
  --end-created-at "2025-09-30T23:59:59+00:00"
```

### 参数说明

| 参数 | 必需 | 默认值 | 说明 |
|------|------|--------|------|
| `--limit` | ❌ | 1000 | 最多处理的文档数量 |
| `--batch` | ❌ | 200 | 每次从数据库拉取的文档数量，越大越快但更占内存 |
| `--concurrency` | ❌ | 8 | 并发度，同时处理的文档数量 |
| `--start-created-at` | ❌ | 全部 | 只处理 `created_at` ≥ 该时间的文档（ISO 格式） |
| `--end-created-at` | ❌ | 全部 | 只处理 `created_at` ≤ 该时间的文档（ISO 格式） |

### 使用示例

```bash
# 修复最近 1000 条缺失向量的文档
python src/bootstrap.py src/devops_scripts/data_fix/mongo_fix_episodic_memory_missing_vector.py \
  --limit 1000

# 修复所有缺失向量的文档（不限制数量）
python src/bootstrap.py src/devops_scripts/data_fix/mongo_fix_episodic_memory_missing_vector.py \
  --limit 999999999

# 修复指定时间范围内的文档
python src/bootstrap.py src/devops_scripts/data_fix/mongo_fix_episodic_memory_missing_vector.py \
  --start-created-at "2025-09-01T00:00:00+00:00" \
  --end-created-at "2025-09-30T23:59:59+00:00" \
  --batch 500 \
  --concurrency 16

# 高性能模式（大批量、高并发）
python src/bootstrap.py src/devops_scripts/data_fix/mongo_fix_episodic_memory_missing_vector.py \
  --batch 1000 \
  --concurrency 32 \
  --limit 100000
```

### 工作流程

1. **查询候选文档**：
   - 查询 `episode` 不为空的文档
   - 过滤出 `vector` 字段缺失或为空的文档
   - 过滤出 `vector_model` 不等于目标模型的文档
   
2. **批量处理**：
   - 分批从 MongoDB 获取文档（batch_size 控制）
   - 按 `created_at` 倒序排列（优先处理最新数据）
   
3. **并发向量化**：
   - 调用 `vectorize_service.get_embedding()` 生成向量
   - 使用协程并发处理（concurrency 控制）
   
4. **更新文档**：
   - 精确按 `_id` 更新，避免覆盖其他字段
   - 更新 `vector` 和 `vector_model` 字段

### 目标向量模型

脚本中定义的目标向量模型：
```python
TARGET_VECTOR_MODEL = "Qwen/Qwen3-Embedding-4B"
```

如果文档的 `vector_model` 不等于此模型，也会被重新生成向量。

### 性能优化

- **批量大小**：`--batch` 控制每次从数据库拉取的文档数量
  - 建议值：200-1000
  - 过大可能导致内存不足
  - 过小会增加数据库查询次数
  
- **并发度**：`--concurrency` 控制同时处理的文档数量
  - 建议值：8-32
  - 取决于向量化服务的吞吐能力
  - 过高可能导致向量化服务过载

### 注意事项

⚠️ **重要提示**：
- 向量化操作依赖外部服务（如 OpenAI、本地模型等），确保服务可用
- 处理大量文档时注意控制并发度，避免服务过载
- 脚本会跳过 `episode` 为空的文档
- 建议先用小 `--limit` 测试，确认无误后再大规模执行

### 执行结果示例

```
🔍 开始扫描需修复文档（limit=1000, batch=200, concurrency=8）
📦 拉取到候选 200 条（已累计处理=0/1000）
⏱️ 当前处理到 created_at=2025-09-25T15:30:45+00:00
📦 拉取到候选 200 条（已累计处理=200/1000）
⏱️ 当前处理到 created_at=2025-09-24T10:20:30+00:00
...
✅ 修复完成 | total=1000, succeeded=995, failed=5
❌ 修复失败 doc=66f2a1b3c4d5e6f789012345, error=Timeout calling vectorize service
```

### 错误处理

- 单个文档处理失败不会中断整个流程
- 失败的文档会被记录在日志中
- 可以重新运行脚本处理失败的文档（基于查询条件自动重试）

---

## 常见问题

### Q1: mongo_add_timestamp_shard.py 提示 "不是分片集群环境"？

**A:** 这表示当前 MongoDB 不是分片集群，分片功能不可用。分片仅在分片集群环境中有意义：
- 如果是单节点或副本集，可以忽略此脚本
- 如果需要分片，请先搭建 MongoDB 分片集群

### Q2: 如何查看当前有多少文档缺失向量？

**A:** 可以在 MongoDB Shell 中执行：
```javascript
use memsys
db.episodic_memories.countDocuments({
  episode: { $exists: true, $ne: "" },
  $or: [
    { vector: { $exists: false } },
    { vector: null },
    { vector: [] },
    { vector_model: { $ne: "Qwen/Qwen3-Embedding-4B" } }
  ]
})
```

### Q3: 向量修复脚本运行很慢怎么办？

**A:** 可以从以下几个方面优化：
1. 增大并发度：`--concurrency 32`
2. 使用时间范围过滤：`--start-created-at` 和 `--end-created-at`
3. 检查向量化服务性能（如切换到更快的模型）
4. 分批次执行，每次处理一部分数据

### Q4: 向量修复失败了怎么办？

**A:** 
- 查看日志中的错误信息
- 常见原因：向量化服务不可用、网络超时、episode 内容过长
- 可以直接重新运行脚本，已成功的文档会被跳过（因为已经有向量了）

### Q5: 分片配置后如何回滚？

**A:** 分片配置是不可逆的，无法直接回滚。如果必须取消分片：
1. 停止均衡器：`sh.stopBalancer()`
2. 导出数据
3. 删除分片集合
4. 重新创建非分片集合
5. 导入数据

⚠️ 这是高风险操作，建议在测试环境验证。

---

## 最佳实践

### 分片配置最佳实践

1. **提前规划**：在数据量较小时（< 1GB）配置分片
2. **监控分片分布**：定期检查数据是否均匀分布
3. **合理选择分片键**：`timestamp` 适合时间序列数据
4. **预分片**：脚本自动创建预分片点，避免热点问题

### 向量修复最佳实践

1. **增量修复**：使用时间范围参数，每天修复当天的数据
2. **监控日志**：关注成功率和失败原因
3. **错误重试**：失败后可以直接重新运行（幂等操作）
4. **性能调优**：根据向量化服务能力调整并发度
5. **定期检查**：建立定时任务，自动修复新产生的缺失向量

### 定时任务示例

```bash
# crontab 配置示例：每天凌晨 2 点修复最近 2 天的数据
0 2 * * * cd /path/to/memsys && python src/bootstrap.py src/devops_scripts/data_fix/mongo_fix_episodic_memory_missing_vector.py --days 2 --batch 500 --concurrency 16
```

---

## 性能基准参考

### 向量修复性能（参考）

| 并发度 | 批量大小 | 处理速度 | 内存占用 |
|--------|---------|---------|---------|
| 8 | 200 | ~100 docs/min | ~500MB |
| 16 | 500 | ~200 docs/min | ~1GB |
| 32 | 1000 | ~300 docs/min | ~2GB |

*注：实际性能取决于向量化服务、网络环境、文档大小等因素*

### 分片效果（参考）

| 数据量 | 分片前查询时间 | 分片后查询时间 | 改善比例 |
|--------|--------------|--------------|---------|
| 10GB | 5-10s | 1-2s | 5-10x |
| 100GB | 30-60s | 3-5s | 10-20x |
| 1TB | 5-10min | 10-30s | 20-40x |

*注：实际效果取决于查询模式、分片数量、硬件配置等因素*


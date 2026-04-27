# Milvus 数据维护脚本使用指南

本文档介绍 Milvus 相关的数据维护脚本的使用方法。

## 脚本列表

- `milvus_sync_docs.py` - Milvus 数据同步主入口脚本
- `milvus_sync_episodic_memory_docs.py` - 情景记忆文档同步到 Milvus 的实现
- `milvus_rebuild_collection.py` - Milvus Collection 重建脚本

---

## 1. milvus_sync_docs.py

### 功能说明

主入口脚本，用于将 MongoDB 数据同步到 Milvus 指定 Collection。根据 Collection 名称自动路由到相应的同步实现。

### 使用方式

```bash
# 通过 bootstrap 运行（推荐）
python src/bootstrap.py src/devops_scripts/data_fix/milvus_sync_docs.py \
  --collection-name episodic_memory \
  --batch-size 500 \
  --limit 10000 \
  --days 7
```

### 参数说明

| 参数 | 缩写 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--collection-name` | `-c` | ✅ | 无 | Milvus Collection 名称，如 `episodic_memory` |
| `--batch-size` | `-b` | ❌ | 500 | 批处理大小，每批同步的文档数量 |
| `--limit` | `-l` | ❌ | 全部 | 限制处理的文档数量，默认处理全部 |
| `--days` | `-d` | ❌ | 全部 | 只处理过去 N 天创建的文档 |

### 使用示例

```bash
# 同步所有情景记忆文档
python src/bootstrap.py src/devops_scripts/data_fix/milvus_sync_docs.py \
  --collection-name episodic_memory

# 只同步最近 7 天的文档
python src/bootstrap.py src/devops_scripts/data_fix/milvus_sync_docs.py \
  --collection-name episodic_memory \
  --days 7

# 同步 10000 条文档，批量大小 1000
python src/bootstrap.py src/devops_scripts/data_fix/milvus_sync_docs.py \
  --collection-name episodic_memory \
  --batch-size 1000 \
  --limit 10000
```

### 注意事项

- 当前仅支持 `episodic_memory` Collection 类型
- 需要通过 `bootstrap.py` 运行以确保应用上下文和依赖注入正确加载
- 同步操作支持幂等，可以重复执行

---

## 2. milvus_sync_episodic_memory_docs.py

### 功能说明

情景记忆文档同步到 Milvus 的具体实现。从 MongoDB 批量获取情景记忆文档，转换后批量插入到 Milvus。

### 技术特点

- **批量处理**：批量获取、批量转换、批量插入，提高效率
- **幂等操作**：使用 insert 操作（Milvus 会自动处理重复 ID）
- **时间过滤**：支持增量同步，只处理指定时间范围的文档
- **数据验证**：插入前验证必要字段（id、vector）是否存在

### 工作流程

1. 从 MongoDB 的 `EpisodicMemoryRawRepository` 批量获取文档
2. 使用 `EpisodicMemoryMilvusConverter.from_mongo()` 转换为 Milvus 实体格式
3. 验证必要字段（id、vector）
4. 批量插入到 Milvus Collection
5. 调用 `flush()` 确保数据持久化

### 数据验证规则

脚本会跳过以下文档：
- 缺少 `id` 字段的文档
- 缺少 `vector` 字段或 `vector` 为空的文档

### 注意事项

- 建议批量大小设置为 500-1000，根据向量维度和文档大小调整
- 向量必须已经在 MongoDB 中生成，脚本不会自动生成向量
- 使用 `--days` 参数可以实现增量同步
- 同步完成后会自动调用 `flush()` 确保数据持久化

---

## 3. milvus_rebuild_collection.py

### 功能说明

重建并切换 Milvus Collection 别名。用于 Collection 结构变更、索引优化、数据迁移等场景。

### 使用方式

```bash
# 通过 bootstrap 运行（推荐）
python src/bootstrap.py src/devops_scripts/data_fix/milvus_rebuild_collection.py \
  --alias episodic_memory \
  --batch-size 3000 \
  --drop-old
```

### 参数说明

| 参数 | 缩写 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--alias` | `-a` | ✅ | 无 | Collection 别名，如 `episodic_memory` |
| `--drop-old` | `-x` | ❌ | False | 是否删除旧 Collection |
| `--no-migrate-data` | - | ❌ | False | 不迁移数据（默认会迁移） |
| `--batch-size` | `-b` | ❌ | 3000 | 每批迁移的数据量 |

### 使用示例

```bash
# 重建 Collection 并迁移数据（保留旧 Collection）
python src/bootstrap.py src/devops_scripts/data_fix/milvus_rebuild_collection.py \
  --alias episodic_memory

# 重建 Collection 但不迁移数据
python src/bootstrap.py src/devops_scripts/data_fix/milvus_rebuild_collection.py \
  --alias episodic_memory \
  --no-migrate-data

# 重建 Collection、迁移数据并指定批大小
python src/bootstrap.py src/devops_scripts/data_fix/milvus_rebuild_collection.py \
  --alias episodic_memory \
  --batch-size 5000

# 重建 Collection、迁移数据并删除旧 Collection
python src/bootstrap.py src/devops_scripts/data_fix/milvus_rebuild_collection.py \
  --alias episodic_memory \
  --drop-old
```

### 工作流程

1. 根据别名找到对应的 Collection 管理类（`MilvusCollectionBase`）
2. 创建新的 Collection（带时间戳后缀）
3. 自动创建索引并加载到内存
4. **（可选）数据迁移**：分批从旧 Collection 查询数据并插入新 Collection
5. 将别名切换到新 Collection
6. **（可选）删除旧 Collection**

### 数据迁移策略

- 使用分批查询和插入，避免内存溢出
- 基于 `id` 字段分页（字符串比较）
- 每批处理后调用 `flush()` 确保数据持久化
- 实时输出迁移进度和统计信息

### 注意事项

⚠️ **重要提示**：
- 默认会迁移数据，如果只想重建结构而不迁移数据，使用 `--no-migrate-data`
- 建议先不删除旧 Collection，确认新 Collection 工作正常后再删除
- 数据迁移时间取决于数据量，大规模数据建议在低峰期操作
- 在生产环境操作前，请先在测试环境验证

### 典型操作流程

#### 场景1：修改 Collection Schema（需要重建）

```bash
# 1. 修改代码中的 Collection 定义（如增加字段、修改索引参数）

# 2. 重建 Collection 并迁移数据
python src/bootstrap.py src/devops_scripts/data_fix/milvus_rebuild_collection.py \
  --alias episodic_memory

# 3. 验证新 Collection 数据和查询功能

# 4. 确认无误后删除旧 Collection
python src/bootstrap.py src/devops_scripts/data_fix/milvus_rebuild_collection.py \
  --alias episodic_memory \
  --no-migrate-data \
  --drop-old
```

#### 场景2：只重建索引（不需要迁移数据）

```bash
# 1. 重建 Collection（不迁移数据）
python src/bootstrap.py src/devops_scripts/data_fix/milvus_rebuild_collection.py \
  --alias episodic_memory \
  --no-migrate-data

# 2. 从 MongoDB 重新同步数据
python src/bootstrap.py src/devops_scripts/data_fix/milvus_sync_docs.py \
  --collection-name episodic_memory

# 3. 验证后删除旧 Collection
```

---

## 常见问题

### Q1: 同步速度慢怎么办？

**A:** 可以调整以下参数：
- 增大 `--batch-size`（建议 500-2000）
- 检查网络连接和 Milvus 集群性能
- 使用 `--days` 参数进行增量同步
- 确认向量维度不是特别高（高维向量传输和插入较慢）

### Q2: 同步时提示 "缺少 vector 字段" 怎么办？

**A:** 这表示 MongoDB 中的文档没有向量数据。需要先运行向量生成脚本：
```bash
python src/bootstrap.py src/devops_scripts/data_fix/mongo_fix_episodic_memory_missing_vector.py
```

### Q3: 数据迁移需要多长时间？

**A:** 取决于数据量和向量维度：
- 100万条记录（768维向量）：约 10-30 分钟
- 建议监控日志中的进度信息
- 可以通过调整 `--batch-size` 优化速度

### Q4: 重建 Collection 会影响查询吗？

**A:** 别名切换是原子操作，几乎不会影响查询。但重建过程中新数据可能写入旧 Collection，建议在低峰期操作。

### Q5: 如何验证 Collection 重建是否成功？

**A:** 可以通过以下方式验证：
```python
from pymilvus import connections, Collection

connections.connect()
collection = Collection("episodic_memory")
print(f"实际 Collection: {collection.name}")
print(f"记录数: {collection.num_entities}")
```

---

## 最佳实践

1. **增量同步**：使用 `--days 1` 每天同步增量数据
2. **批量调优**：根据向量维度和网络情况调整 `--batch-size`
3. **监控日志**：关注脚本输出的成功/失败统计和进度信息
4. **备份策略**：重建前确保 MongoDB 数据完整（Milvus 可以从 MongoDB 重建）
5. **测试验证**：生产环境操作前在测试环境充分验证
6. **分批操作**：大规模数据建议先用 `--limit` 测试少量数据
7. **错误重试**：同步失败可以直接重新运行，支持幂等操作

---

## 性能优化建议

### 同步性能优化

- **批量大小**：建议 500-1000，不宜过大（内存限制）
- **并发控制**：Milvus 插入操作本身已经是批量的，无需额外并发
- **网络优化**：确保与 Milvus 服务的网络延迟较低

### Collection 索引优化

- **HNSW 参数**：`M=16, efConstruction=256` 适合大多数场景
- **IVF 参数**：数据量大时考虑使用 IVF_FLAT 或 IVF_PQ
- **内存加载**：重要的 Collection 保持 loaded 状态以提升查询性能


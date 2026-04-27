# Elasticsearch 数据维护脚本使用指南

本文档介绍 Elasticsearch 相关的数据维护脚本的使用方法。

## 脚本列表

- `es_sync_docs.py` - ES 数据同步主入口脚本
- `es_sync_episodic_memory_docs.py` - 情景记忆文档同步到 ES 的实现
- `es_rebuild_index.py` - ES 索引重建脚本

---

## 1. es_sync_docs.py

### 功能说明

主入口脚本，用于将 MongoDB 数据同步到 Elasticsearch 指定索引。根据索引名称自动路由到相应的同步实现。

### 使用方式

```bash
# 通过 bootstrap 运行（推荐）
python src/bootstrap.py src/devops_scripts/data_fix/es_sync_docs.py \
  --index-name episodic-memory \
  --batch-size 500 \
  --limit 10000 \
  --days 7
```

### 参数说明

| 参数 | 缩写 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--index-name` | `-i` | ✅ | 无 | ES 索引别名，如 `episodic-memory` |
| `--batch-size` | `-b` | ❌ | 500 | 批处理大小，每批同步的文档数量 |
| `--limit` | `-l` | ❌ | 全部 | 限制处理的文档数量，默认处理全部 |
| `--days` | `-d` | ❌ | 全部 | 只处理过去 N 天创建的文档 |

### 使用示例

```bash
# 同步所有情景记忆文档
python src/bootstrap.py src/devops_scripts/data_fix/es_sync_docs.py \
  --index-name episodic-memory

# 只同步最近 7 天的文档
python src/bootstrap.py src/devops_scripts/data_fix/es_sync_docs.py \
  --index-name episodic-memory \
  --days 7

# 同步 10000 条文档，批量大小 1000
python src/bootstrap.py src/devops_scripts/data_fix/es_sync_docs.py \
  --index-name episodic-memory \
  --batch-size 1000 \
  --limit 10000
```

### 注意事项

- 当前仅支持 `episodic-memory` 索引类型
- 需要通过 `bootstrap.py` 运行以确保应用上下文和依赖注入正确加载
- 同步操作使用 upsert 语义，支持幂等操作

---

## 2. es_sync_episodic_memory_docs.py

### 功能说明

情景记忆文档同步到 Elasticsearch 的具体实现。从 MongoDB 批量获取情景记忆文档，转换后批量写入 ES。

### 技术特点

- **批量处理**：支持大规模数据同步，避免内存溢出
- **幂等操作**：使用 `update` + `doc_as_upsert` 模式，支持重复执行
- **时间过滤**：支持增量同步，只处理指定时间范围的文档
- **流式处理**：使用 `async_streaming_bulk` 提高性能

### 工作流程

1. 从 MongoDB 的 `EpisodicMemoryRawRepository` 批量获取文档
2. 使用 `EpisodicMemoryConverter.from_mongo()` 转换为 ES 文档格式
3. 使用 `async_streaming_bulk` 批量写入 ES
4. 自动刷新索引以确保数据可见

### 注意事项

- 建议批量大小设置为 500-1000，根据文档大小调整
- 使用 `--days` 参数可以实现增量同步
- 同步完成后会自动刷新索引

---

## 3. es_rebuild_index.py

### 功能说明

重建并切换 Elasticsearch 索引别名。用于索引结构变更、数据迁移等场景。

### 使用方式

```bash
# 通过 bootstrap 运行（推荐）
python src/bootstrap.py src/devops_scripts/data_fix/es_rebuild_index.py \
  --index-name episodic-memory \
  --close-old \
  --delete-old
```

### 参数说明

| 参数 | 缩写 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--index-name` | `-i` | ✅ | 无 | 索引别名，如 `episodic-memory` |
| `--close-old` | `-c` | ❌ | False | 是否关闭旧索引 |
| `--delete-old` | `-x` | ❌ | False | 是否删除旧索引 |

### 使用示例

```bash
# 重建索引，保留旧索引（最安全）
python src/bootstrap.py src/devops_scripts/data_fix/es_rebuild_index.py \
  --index-name episodic-memory

# 重建索引并关闭旧索引
python src/bootstrap.py src/devops_scripts/data_fix/es_rebuild_index.py \
  --index-name episodic-memory \
  --close-old

# 重建索引并删除旧索引（谨慎使用）
python src/bootstrap.py src/devops_scripts/data_fix/es_rebuild_index.py \
  --index-name episodic-memory \
  --close-old \
  --delete-old
```

### 工作流程

1. 根据索引别名查找对应的文档类（`Document`）
2. 创建新的索引（带时间戳后缀）
3. 应用新的 mapping 和 settings
4. 将别名切换到新索引
5. 可选：关闭或删除旧索引

### 注意事项

⚠️ **重要提示**：
- 重建索引**不会自动迁移数据**，需要单独运行同步脚本
- 建议先不删除旧索引，确认新索引工作正常后再删除
- 在生产环境操作前，请先在测试环境验证

### 典型操作流程

```bash
# 1. 重建索引结构（不删除旧索引）
python src/bootstrap.py src/devops_scripts/data_fix/es_rebuild_index.py \
  --index-name episodic-memory

# 2. 同步数据到新索引
python src/bootstrap.py src/devops_scripts/data_fix/es_sync_docs.py \
  --index-name episodic-memory

# 3. 验证新索引数据无误后，删除旧索引（可选）
# 手动在 Kibana 或通过 ES API 删除旧索引
```

---

## 常见问题

### Q1: 同步速度慢怎么办？

**A:** 可以调整以下参数：
- 增大 `--batch-size`（建议 500-2000）
- 检查网络连接和 ES 集群性能
- 使用 `--days` 参数进行增量同步

### Q2: 同步中断后如何继续？

**A:** 由于使用 upsert 语义，直接重新运行同步脚本即可，已同步的文档会被更新而不是重复插入。

### Q3: 如何验证同步是否成功？

**A:** 可以通过以下方式验证：
```bash
# 检查文档数量
curl -X GET "localhost:9200/episodic-memory/_count"

# 查询最近的文档
curl -X GET "localhost:9200/episodic-memory/_search?size=10&sort=created_at:desc"
```

### Q4: 重建索引会影响查询吗？

**A:** 别名切换是原子操作，几乎不会影响查询。但重建过程中新数据可能写入旧索引，建议在低峰期操作。

---

## 最佳实践

1. **增量同步**：使用 `--days 1` 每天同步增量数据
2. **批量调优**：根据文档大小调整 `--batch-size`，避免内存溢出
3. **监控日志**：关注脚本输出的成功/失败统计
4. **备份策略**：重建索引前确保有数据备份
5. **测试验证**：生产环境操作前在测试环境充分验证


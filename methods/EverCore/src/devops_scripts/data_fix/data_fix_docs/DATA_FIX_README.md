# 数据维护脚本使用指南

本目录包含 Elasticsearch、Milvus 和 MongoDB 数据维护脚本的详细使用文档。

## 📚 文档索引

### [Elasticsearch 脚本使用指南](./elasticsearch_scripts_guide.md)

包含以下脚本的使用说明：
- `es_sync_docs.py` - ES 数据同步主入口
- `es_sync_episodic_memory_docs.py` - 情景记忆文档同步到 ES
- `es_rebuild_index.py` - ES 索引重建

**适用场景**：
- 从 MongoDB 同步数据到 Elasticsearch
- 重建 ES 索引结构
- 修改索引 mapping 或 settings

---

### [Milvus 脚本使用指南](./milvus_scripts_guide.md)

包含以下脚本的使用说明：
- `milvus_sync_docs.py` - Milvus 数据同步主入口
- `milvus_sync_episodic_memory_docs.py` - 情景记忆文档同步到 Milvus
- `milvus_rebuild_collection.py` - Milvus Collection 重建

**适用场景**：
- 从 MongoDB 同步向量数据到 Milvus
- 重建 Milvus Collection 结构
- 迁移 Milvus 数据到新 Collection

---

### [MongoDB 脚本使用指南](./mongodb_scripts_guide.md)

包含以下脚本的使用说明：
- `mongo_add_timestamp_shard.py` - 添加基于 timestamp 的分片配置
- `mongo_fix_episodic_memory_missing_vector.py` - 修复情景记忆缺失向量

**适用场景**：
- 配置 MongoDB 分片以优化查询性能
- 批量修复缺失的向量字段
- 更新向量模型版本

---

## 🚀 快速开始

### 通用运行方式

所有脚本推荐通过 `bootstrap.py` 运行，以确保应用上下文和依赖注入正确加载：

```bash
python src/bootstrap.py src/devops_scripts/data_fix/<脚本名称> [参数]
```

### 常用操作示例

#### 1. 同步数据到 ES

```bash
python src/bootstrap.py src/devops_scripts/data_fix/es_sync_docs.py \
  --index-name episodic-memory \
  --days 7
```

#### 2. 同步数据到 Milvus

```bash
python src/bootstrap.py src/devops_scripts/data_fix/milvus_sync_docs.py \
  --collection-name episodic_memory \
  --days 7
```

#### 3. 修复缺失向量

```bash
python src/bootstrap.py src/devops_scripts/data_fix/mongo_fix_episodic_memory_missing_vector.py \
  --limit 1000 \
  --batch 200 \
  --concurrency 8
```

---

## 📊 数据流向图

```
┌─────────────┐
│   MongoDB   │  (主数据源)
└──────┬──────┘
       │
       ├─────────────────────────────────┬──────────────────────────────────┐
       │                                 │                                  │
       ▼                                 ▼                                  ▼
┌─────────────┐                  ┌─────────────┐                  ┌─────────────┐
│Elasticsearch│ ◄─ es_sync       │   Milvus    │ ◄─ milvus_sync  │ 向量修复脚本 │
│  (全文检索)  │                  │ (向量检索)   │                  │             │
└─────────────┘                  └─────────────┘                  └─────────────┘
       │                                 │                                  │
       │                                 │                                  │
       └─────────────────────────────────┴──────────────────────────────────┘
                              用于检索和查询
```

---

## ⚠️ 注意事项

### 运行环境

- **Python 版本**：3.10+
- **依赖管理**：使用 `uv` 或 `pip` 安装依赖
- **配置文件**：确保 `config.json` 配置正确

### 权限要求

- MongoDB：读取权限（同步脚本），读写权限（修复脚本）
- Elasticsearch：索引读写权限
- Milvus：Collection 读写权限

### 安全建议

1. **测试优先**：在测试环境充分验证后再在生产环境执行
2. **备份数据**：重要操作前备份数据
3. **监控日志**：执行过程中持续监控日志输出
4. **分批执行**：大规模数据操作建议分批执行
5. **低峰期操作**：重建索引等操作建议在低峰期进行

---

## 🛠️ 故障排查

### 脚本执行失败

1. 检查网络连接（MongoDB、ES、Milvus）
2. 检查服务是否正常运行
3. 检查配置文件是否正确
4. 查看详细日志定位问题

### 性能问题

1. 调整批量大小（`--batch-size` 或 `--batch`）
2. 调整并发度（`--concurrency`）
3. 检查服务性能瓶颈
4. 使用增量同步（`--days` 参数）

### 数据不一致

1. 重新运行同步脚本（支持幂等操作）
2. 检查源数据（MongoDB）是否正确
3. 使用 `--limit` 参数先测试少量数据
4. 检查日志中的错误信息

---

## 📞 获取帮助

如果遇到问题，可以：

1. 查看各脚本详细文档中的"常见问题"部分
2. 使用 `--help` 参数查看脚本帮助信息
3. 查看脚本源码中的注释说明
4. 联系开发团队获取支持

---

## 📝 更新日志

- **2025-10-22**: 创建文档，涵盖 ES、Milvus、MongoDB 维护脚本


# SceneExtractor 处理流程详解

## 概述

`SceneExtractor` 是一个三阶段的场景提取器，用于将新的 memcell 智能地归类到现有场景或创建新场景。

**核心思想**：
- **阶段一**：判断新 memcell 与哪些历史 memcell 相似
- **阶段二**：判断相似的 memcell 属于哪些现有场景
- **阶段三**：根据前两阶段结果，创建新场景或更新现有场景

---

## 示例场景设定

### 输入数据

**历史 MemCell（5个）**：
- `mc_001`: "我们需要设计一个多层记忆模型，包括工作记忆和长期记忆"（主题：记忆模型）
- `mc_002`: "记忆系统的架构应该支持分布式存储"（主题：记忆系统）
- `mc_003`: "记忆模型的层次结构很重要，需要考虑遗忘曲线"（主题：记忆模型）
- `mc_004`: "我们的记忆系统需要实现增量更新功能"（主题：记忆系统）
- `mc_005`: "周末打算去爬山，天气预报说很适合"（主题：日常生活）

**现有 Scene（4个）**：
- `scene_1`: "记忆模型的工作情景"
  - 包含：`mc_001`, `mc_003`
  - 摘要：讨论记忆模型的设计和层次结构
  
- `scene_2`: "记忆系统的工作情景"
  - 包含：`mc_002`, `mc_004`
  - 摘要：讨论记忆系统的架构和实现
  
- `scene_3`: "记忆部门的工作情景"
  - 包含：`mc_001`, `mc_002`, `mc_003`, `mc_004`
  - 摘要：记忆部门的整体工作讨论（包含模型和系统两个方面）
  
- `scene_4`: "日常生活"
  - 包含：`mc_005`
  - 摘要：个人生活相关的日常对话

**新 MemCell**：
- `mc_new`: "记忆系统的性能优化需要考虑缓存策略"（主题：记忆系统）

---

## 处理流程详解

### 阶段一：MemCell 相似性检测

#### 目标
判断新 memcell (`mc_new`) 与哪些历史 memcell 相似。

#### 输入
```python
SceneExtractRequest(
    history_memcell_list=[mc_001, mc_002, mc_003, mc_004, mc_005],
    new_memcell=mc_new,
    existing_scenes=[scene_1, scene_2, scene_3, scene_4]
)
```

#### LLM 提示词
```
请分析新 memcell 与历史 memcell 的相似性：

新 MemCell:
MemCell ID: mc_new
Content: 记忆系统的性能优化需要考虑缓存策略

历史 MemCell (共 5 个):
--- MemCell 1 ---
MemCell ID: mc_001
Content: 我们需要设计一个多层记忆模型，包括工作记忆和长期记忆

--- MemCell 2 ---
MemCell ID: mc_002
Content: 记忆系统的架构应该支持分布式存储

--- MemCell 3 ---
MemCell ID: mc_003
Content: 记忆模型的层次结构很重要，需要考虑遗忘曲线

--- MemCell 4 ---
MemCell ID: mc_004
Content: 我们的记忆系统需要实现增量更新功能

--- MemCell 5 ---
MemCell ID: mc_005
Content: 周末打算去爬山，天气预报说很适合

请判断新 memcell 与哪些历史 memcell 相似，并返回 JSON 格式...
```

#### LLM 响应
```json
{
    "similar_memcells": [
        {
            "memcell_id": "mc_002",
            "similarity_score": 0.90,
            "reasoning": "都在讨论记忆系统的技术实现，mc_002 讨论架构，新 memcell 讨论性能优化"
        },
        {
            "memcell_id": "mc_004",
            "similarity_score": 0.85,
            "reasoning": "都涉及记忆系统的功能改进，mc_004 讨论增量更新，新 memcell 讨论缓存优化"
        }
    ],
    "reasoning": "新 memcell 的主题是记忆系统的性能优化，与 mc_002 和 mc_004 同属记忆系统相关讨论"
}
```

#### 输出结果
```python
SimilarMemCellResult(
    has_similar=True,
    similar_memcells=[
        SimilarMemCell(
            memcell_id="mc_002",
            similarity_score=0.90,
            reasoning="都在讨论记忆系统的技术实现"
        ),
        SimilarMemCell(
            memcell_id="mc_004",
            similarity_score=0.85,
            reasoning="都涉及记忆系统的功能改进"
        )
    ],
    reasoning="新 memcell 与记忆系统相关的 mc_002 和 mc_004 相似"
)
```

#### 日志输出
```
[SceneExtractor] 开始三阶段场景提取
[Stage1] 检测相似性 - 历史: 5 个
[Stage1] 发现 2 个相似 memcell: mc_002, mc_004
🔗 发现 2 个相似 memcell
```

#### 决策
✅ 发现相似 memcell → 进入**阶段二**

---

### 阶段二：Scene 相似性检测

#### 目标
判断相似的 memcell (`mc_002`, `mc_004`) 和新 memcell 属于哪些现有场景。

#### 输入
- 新 memcell: `mc_new`
- 相关 memcell: `[mc_002, mc_004]`
- 现有场景: `[scene_1, scene_2, scene_3, scene_4]`

#### LLM 提示词
```
请分析新 memcell 和相关 memcell 是否属于现有场景：

新 MemCell:
MemCell ID: mc_new
Content: 记忆系统的性能优化需要考虑缓存策略

相关 MemCell:
--- MemCell 1 ---
MemCell ID: mc_002
Content: 记忆系统的架构应该支持分布式存储

--- MemCell 2 ---
MemCell ID: mc_004
Content: 我们的记忆系统需要实现增量更新功能

现有场景 (共 4 个):
--- Scene 1 ---
Scene ID: scene_1
Title: 记忆模型的工作情景
Summary: 讨论记忆模型的设计和层次结构
MemCell IDs: mc_001, mc_003

--- Scene 2 ---
Scene ID: scene_2
Title: 记忆系统的工作情景
Summary: 讨论记忆系统的架构和实现
MemCell IDs: mc_002, mc_004

--- Scene 3 ---
Scene ID: scene_3
Title: 记忆部门的工作情景
Summary: 记忆部门的整体工作讨论（包含模型和系统两个方面）
MemCell IDs: mc_001, mc_002, mc_003, mc_004

--- Scene 4 ---
Scene ID: scene_4
Title: 日常生活
Summary: 个人生活相关的日常对话
MemCell IDs: mc_005

请判断应该将新 memcell 加入哪些场景...
```

#### LLM 响应
```json
{
    "similar_scenes": [
        {
            "scene_id": "scene_2",
            "similarity_score": 0.92,
            "reasoning": "scene_2 包含 mc_002 和 mc_004，主题是记忆系统的架构和实现，新 memcell 讨论性能优化，完全契合"
        },
        {
            "scene_id": "scene_3",
            "similarity_score": 0.78,
            "reasoning": "scene_3 是记忆部门的综合讨论，包含 mc_002 和 mc_004，新 memcell 也属于工作讨论的一部分"
        }
    ],
    "reasoning": "新 memcell 和相关 memcell 都属于记忆系统相关讨论，应该加入 scene_2 和 scene_3"
}
```

#### 输出结果
```python
SimilarSceneResult(
    has_similar=True,
    similar_scenes=[
        SimilarScene(
            scene_id="scene_2",
            similarity_score=0.92,
            reasoning="主题完全契合记忆系统的工作情景"
        ),
        SimilarScene(
            scene_id="scene_3",
            similarity_score=0.78,
            reasoning="属于记忆部门的综合工作讨论"
        )
    ],
    reasoning="发现 2 个相关场景"
)
```

#### 日志输出
```
[Stage2] 检测场景相似性 - 场景数: 4
[Stage2] 发现 2 个相似场景: scene_2, scene_3
🔄 发现 2 个相似场景，更新所有场景: scene_2, scene_3
```

#### 决策
✅ 发现相似场景 → 进入**阶段三：更新现有场景**

---

### 阶段三：场景更新

#### 目标
将新 memcell 加入所有相似场景，并重新生成场景信息。

#### 更新流程

##### 3.1 更新 Scene_2

**LLM 提示词**：
```
请更新以下场景，加入新的 memcell：

现有场景:
Scene ID: scene_2
Title: 记忆系统的工作情景
Summary: 讨论记忆系统的架构和实现
MemCell IDs: mc_002, mc_004
Keywords: 记忆系统, 架构, 分布式, 增量更新

新 MemCell:
MemCell ID: mc_new
Content: 记忆系统的性能优化需要考虑缓存策略

请重新提取场景的标题、摘要和关键词...
```

**LLM 响应**：
```json
{
    "title": "记忆系统的工作情景",
    "summary": "讨论记忆系统的架构设计、增量更新功能和性能优化策略",
    "keywords": ["记忆系统", "架构", "分布式", "增量更新", "性能优化", "缓存策略"]
}
```

**更新后的 Scene_2**：
```python
Scene(
    scene_id="scene_2",
    title="记忆系统的工作情景",
    summary="讨论记忆系统的架构设计、增量更新功能和性能优化策略",
    memcell_ids=["mc_002", "mc_004", "mc_new"],  # 新增 mc_new
    keywords=["记忆系统", "架构", "分布式", "增量更新", "性能优化", "缓存策略"],
    # ... 其他字段
)
```

**日志输出**：
```
  更新场景: scene_2 (score: 0.92)
  [Stage3] 更新场景: scene_2
  [Stage3] ✅ 更新场景: 记忆系统的工作情景
  ✅ 成功更新场景: 记忆系统的工作情景
```

##### 3.2 更新 Scene_3

**LLM 提示词**：
```
请更新以下场景，加入新的 memcell：

现有场景:
Scene ID: scene_3
Title: 记忆部门的工作情景
Summary: 记忆部门的整体工作讨论（包含模型和系统两个方面）
MemCell IDs: mc_001, mc_002, mc_003, mc_004
Keywords: 记忆模型, 记忆系统, 工作讨论

新 MemCell:
MemCell ID: mc_new
Content: 记忆系统的性能优化需要考虑缓存策略

请重新提取场景的标题、摘要和关键词...
```

**LLM 响应**：
```json
{
    "title": "记忆部门的工作情景",
    "summary": "记忆部门的整体工作讨论，涵盖记忆模型设计、系统架构和性能优化",
    "keywords": ["记忆模型", "记忆系统", "工作讨论", "性能优化"]
}
```

**更新后的 Scene_3**：
```python
Scene(
    scene_id="scene_3",
    title="记忆部门的工作情景",
    summary="记忆部门的整体工作讨论，涵盖记忆模型设计、系统架构和性能优化",
    memcell_ids=["mc_001", "mc_002", "mc_003", "mc_004", "mc_new"],  # 新增 mc_new
    keywords=["记忆模型", "记忆系统", "工作讨论", "性能优化"],
    # ... 其他字段
)
```

**日志输出**：
```
  更新场景: scene_3 (score: 0.78)
  [Stage3] 更新场景: scene_3
  [Stage3] ✅ 更新场景: 记忆部门的工作情景
  ✅ 成功更新场景: 记忆部门的工作情景
```

---

### 最终输出结果

```python
SceneExtractResult(
    scene=Scene(scene_id="scene_2", ...),  # 主场景（第一个更新的）
    action="update_existing",
    
    similar_memcell_result=SimilarMemCellResult(
        has_similar=True,
        similar_memcells=[
            SimilarMemCell(memcell_id="mc_002", similarity_score=0.90, ...),
            SimilarMemCell(memcell_id="mc_004", similarity_score=0.85, ...)
        ],
        reasoning="新 memcell 与记忆系统相关的 mc_002 和 mc_004 相似"
    ),
    
    similar_scene_result=SimilarSceneResult(
        has_similar=True,
        similar_scenes=[
            SimilarScene(scene_id="scene_2", similarity_score=0.92, ...),
            SimilarScene(scene_id="scene_3", similarity_score=0.78, ...)
        ],
        reasoning="发现 2 个相关场景"
    ),
    
    updated_scenes=[
        Scene(scene_id="scene_2", title="记忆系统的工作情景", ...),
        Scene(scene_id="scene_3", title="记忆部门的工作情景", ...)
    ]
)
```

---

## 三种典型处理路径

### 路径 1：创建全新场景（无相似 memcell）

**场景**：新 memcell 是 "今天去超市买了菜"

**处理流程**：
1. **阶段一**：与所有历史 memcell 对比 → 无相似 memcell
2. **跳过阶段二**：直接进入阶段三
3. **阶段三**：创建新场景，仅包含新 memcell

**结果**：
```python
SceneExtractResult(
    scene=Scene(
        scene_id="scene_5",
        title="超市购物",
        memcell_ids=["mc_new"],
        ...
    ),
    action="create_new",
    similar_memcell_result=SimilarMemCellResult(has_similar=False, ...),
    similar_scene_result=None
)
```

---

### 路径 2：创建新场景（有相似 memcell，无相似场景）

**场景**：新 memcell 是 "记忆可视化界面的设计需要考虑用户体验"

**处理流程**：
1. **阶段一**：与 mc_001, mc_003 相似（都涉及记忆相关）
2. **阶段二**：检查现有场景 → 无完全匹配的场景（可视化是新主题）
3. **阶段三**：创建新场景，包含相似 memcell + 新 memcell

**结果**：
```python
SceneExtractResult(
    scene=Scene(
        scene_id="scene_6",
        title="记忆可视化项目",
        memcell_ids=["mc_001", "mc_003", "mc_new"],
        ...
    ),
    action="create_new",
    similar_memcell_result=SimilarMemCellResult(has_similar=True, ...),
    similar_scene_result=SimilarSceneResult(has_similar=False, ...)
)
```

---

### 路径 3：更新现有场景（有相似 memcell，有相似场景）

**场景**：新 memcell 是 "记忆系统的性能优化需要考虑缓存策略"（本示例）

**处理流程**：
1. **阶段一**：与 mc_002, mc_004 相似
2. **阶段二**：发现 scene_2 和 scene_3 包含相似 memcell
3. **阶段三**：更新 scene_2 和 scene_3

**结果**：如上述完整示例所示

---

## 核心特性

### 1. 多场景更新

一个新 memcell 可以同时属于多个场景。在本例中：
- `mc_new` 同时加入了 `scene_2`（专注记忆系统）
- `mc_new` 也加入了 `scene_3`（综合工作讨论）

这反映了真实场景中的**多对多关系**。

### 2. 结构化输出

每个阶段都有清晰的结构化输出：
- **阶段一**：`SimilarMemCellResult` - 包含相似度分数和推理
- **阶段二**：`SimilarSceneResult` - 包含多个相似场景
- **最终结果**：`SceneExtractResult` - 包含所有阶段信息

### 3. LLM 驱动决策

所有相似性判断都由 LLM 完成：
- 语义理解：理解 "记忆系统" 和 "性能优化" 的关系
- 主题归类：判断应该归属哪个场景
- 信息提取：重新生成场景摘要和关键词

---

## 使用示例

```python
from memory_layer.scene_extractor import SceneExtractor, SceneExtractRequest
from memory_layer.llm import LLMProvider

# 初始化提取器
extractor = SceneExtractor(llm_provider=LLMProvider)

# 构造请求
request = SceneExtractRequest(
    history_memcell_list=[mc_001, mc_002, mc_003, mc_004, mc_005],
    new_memcell=mc_new,
    existing_scenes=[scene_1, scene_2, scene_3, scene_4]
)

# 执行提取
result = await extractor.extract_scene(request)

# 处理结果
if result.action == "update_existing":
    print(f"更新了 {len(result.updated_scenes)} 个场景:")
    for scene in result.updated_scenes:
        print(f"  - {scene.title}: {len(scene.memcell_ids)} 个 memcell")
elif result.action == "create_new":
    print(f"创建新场景: {result.scene.title}")
```

---

## 设计原则

1. **增量处理**：每次只处理一个新 memcell，保证实时性
2. **语义理解**：基于 LLM 的语义理解，而非简单的关键词匹配
3. **灵活归类**：支持一个 memcell 属于多个场景
4. **信息完整**：保留所有阶段的决策信息，便于调试和优化
5. **代码简洁**：核心代码 550 行，逻辑清晰，易于维护

---

## SceneExtractorWithRetrieval - 检索增强版本

### 概述

`SceneExtractorWithRetrieval` 是 `SceneExtractor` 的改进版本，将前两个阶段（MemCell相似性检测和Scene相似性检测）从LLM调用替换为基于检索算法的实现。

### 与原版的区别

| 阶段 | 原版 (scene_extractor.py) | 检索版 (scene_extractor_w_retrieval.py) |
|------|--------------------------|----------------------------------------|
| **阶段一** | LLM判断相似MemCell | BM25/向量检索找相似MemCell |
| **阶段二** | LLM判断相似Scene | BM25/向量检索找相似Scene |
| **阶段三** | LLM创建/更新场景 | LLM创建/更新场景（相同） |
| **阶段四** | LLM分配角色权重 | LLM分配角色权重（相同） |

### 优势

1. **更快的响应速度**：检索算法（特别是BM25）比LLM调用快得多
2. **更低的成本**：减少了LLM调用次数，降低API费用
3. **可控的相似度**：通过阈值直接控制相似度判断，而不依赖LLM的不确定性
4. **批量处理友好**：检索算法天然支持批量计算，适合大规模数据处理

### 支持的检索方法

- **向量检索（Vector Retrieval）**：基于embedding的余弦相似度
- **BM25检索（Keyword Retrieval）**：基于关键词的传统检索算法
- **混合检索**：可以同时启用两种方法，取分数平均值

### 使用示例

#### 基础用法（仅使用向量检索）

```python
from memory_layer.scene_extractor.scene_extractor_w_retrieval import (
    SceneExtractorWithRetrieval,
    SceneExtractRequest
)
from memory_layer.llm import LLMProvider
from component.embedding_provider import EmbeddingProvider

# 初始化embedding provider
embedding_provider = EmbeddingProvider()

# 初始化提取器（使用向量检索）
extractor = SceneExtractorWithRetrieval(
    llm_provider=LLMProvider,
    embedding_provider=embedding_provider,
    use_vector=True,
    use_bm25=False,
    similarity_threshold=0.5,  # 相似度阈值
    top_k=5  # 返回top 5个结果
)

# 构造请求
request = SceneExtractRequest(
    history_memcell_list=[mc_001, mc_002, mc_003],
    new_memcell=mc_new,
    existing_scenes=[scene_1, scene_2]
)

# 执行提取
result, hyperedge_results = await extractor.extract_scene(request)
```

#### 使用BM25检索

```python
# 初始化提取器（使用BM25检索）
extractor = SceneExtractorWithRetrieval(
    llm_provider=LLMProvider,
    embedding_provider=None,  # 不需要embedding provider
    use_vector=False,
    use_bm25=True,
    similarity_threshold=0.3,  # BM25分数较低，可以降低阈值
    top_k=5
)
```

#### 使用混合检索

```python
# 初始化提取器（同时使用向量和BM25）
extractor = SceneExtractorWithRetrieval(
    llm_provider=LLMProvider,
    embedding_provider=embedding_provider,
    use_vector=True,
    use_bm25=True,  # 同时启用
    similarity_threshold=0.4,  # 融合后的阈值
    top_k=5
)
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|-------|------|
| `llm_provider` | LLMProvider | - | LLM提供者（用于第三、四阶段） |
| `embedding_provider` | EmbeddingProvider | None | Embedding提供者（向量检索必需） |
| `use_vector` | bool | True | 是否使用向量检索 |
| `use_bm25` | bool | False | 是否使用BM25检索 |
| `similarity_threshold` | float | 0.5 | 相似度阈值（0.0-1.0） |
| `top_k` | int | 5 | 检索返回的top k结果数量 |

### 相似度阈值建议

| 检索方法 | 推荐阈值 | 说明 |
|---------|---------|------|
| 向量检索 | 0.5-0.7 | 余弦相似度范围 [0, 1] |
| BM25检索 | 0.2-0.4 | BM25分数范围不固定，建议较低阈值 |
| 混合检索 | 0.4-0.6 | 平均后的分数 |

### 输出日志示例

```
[SceneExtractorWithRetrieval] 开始四阶段场景提取（检索版）

================================================================================
📥 [阶段一-检索] MemCell 相似性检测
================================================================================
Query: 记忆系统的性能优化需要考虑缓存策略...
Candidates: 5
================================================================================
[Stage1-Retrieval] 使用向量检索

================================================================================
📤 [阶段一-检索] 检索结果
================================================================================
  - mc_002: 0.890
  - mc_004: 0.850
================================================================================
🔗 发现 2 个相似 memcell

================================================================================
📥 [阶段二-检索] Scene 相似性检测
================================================================================
Query: 记忆系统的性能优化需要考虑缓存策略 记忆系统的架构应该支持分布式存储...
Candidates: 4
================================================================================
[Stage2-Retrieval] 使用向量检索

================================================================================
📤 [阶段二-检索] 检索结果
================================================================================
  - scene_2: 0.920
  - scene_3: 0.780
================================================================================
🔄 发现 2 个相似场景，更新所有场景: scene_2, scene_3
```

### 何时选择检索版本？

#### 适合使用检索版本的场景

- ✅ 需要快速处理大量memcells
- ✅ 对响应延迟要求高
- ✅ 希望降低LLM API成本
- ✅ 相似性判断可以基于文本匹配
- ✅ 已有高质量的embedding模型

#### 适合使用原版的场景

- ✅ 需要深度语义理解
- ✅ 数据量不大，可以接受较慢的处理速度
- ✅ 需要LLM进行复杂的推理
- ✅ 文本相似度不足以准确判断"同一情景"

### 性能对比

| 指标 | 原版 | 检索版（向量） | 检索版（BM25） |
|-----|------|--------------|--------------|
| 阶段一+二耗时 | ~5-10秒 | ~0.5-1秒 | ~0.1-0.3秒 |
| LLM调用次数 | 4次 | 2次 | 2次 |
| 准确性 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| 适用场景 | 复杂语义 | 一般场景 | 快速筛选 |

### 依赖安装

检索版本需要额外的依赖：

```bash
# BM25检索需要
pip install bm25s nltk

# 或者使用uv
uv add bm25s nltk

# 下载nltk停用词（首次使用时）
python -c "import nltk; nltk.download('stopwords')"
```


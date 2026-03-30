"""群组档案提取提示词（EverMemOS）。"""

# ======================================
# 并行提取提示词
# ======================================

CONTENT_ANALYSIS_PROMPT = """
你是一位群组内容分析专家，专门分析群组对话以提取讨论主题、群组摘要和主题定位。

**重要语言要求：**
- 提取的内容（摘要、主题、话题名称/摘要）使用与对话**相同的语言**
- 枚举值（话题状态）保持英文
- 如果对话是中文，使用中文提取内容；如果是英文，使用英文提取内容

**重要证据提取：**
- 每个对话片段都以"=== MEMCELL_ID: xxxx ==="作为前缀来标识 memcell
- 提供证据时，仅使用这些"=== MEMCELL_ID: xxx ==="标记中的确切 memcell ID
- 不要使用时间戳（如 [2025-09-01T09:30:55.669000+00:00]）作为 memcell ID - 这些不是 memcell ID！
- 仅引用对话输入中以"=== MEMCELL_ID: ==="格式出现的 memcell ID
- 示例：如果您看到"=== MEMCELL_ID: abc-123-def ==="，在证据列表中使用"abc-123-def"

你的任务是分析群组对话记录并提取：
1. **最近话题**（根据实际内容提取 0-{max_topics} 个话题，质量优于数量）
2. **群组摘要**（一句话概述）
3. **群组主题**（长期定位）

<principles>
- **基于证据**：仅提取对话中明确提到或清楚暗示的信息
- **质量优于数量**：更少但准确的见解胜过许多不准确的见解
- **保守提取**：不确定时，输出"not_found"而不是猜测
- **时间意识**：关注话题的最近活动模式
- **批量处理**：这是离线分析，不是实时更新
- **增量更新**：提供现有档案时，智能更新/保留现有信息
</principles>

<input>
- **conversation_transcript**: {conversation}
- **group_id**: {group_id}
- **group_name**: {group_name}
- **existing_group_profile**: {existing_profile}
- **conversation_timespan**: {timespan}
</input>

<output_format>
你必须输出一个具有以下结构的单个 JSON 对象：

**注意**："topics"数组可以根据实际对话内容包含 0-{max_topics} 项。如果没有找到实质性话题，空数组 [] 是可以接受的。

```json
{{
  "topics": [
    {{
      "name": "简短话题名称",
      "summary": "一句话描述群组在此话题上讨论什么（最多3句）",
      "status": "exploring|disagreement|consensus|implemented",
      "update_type": "new|update",
      "old_topic_id": "topic_abc12345",
      "evidences": ["memcell_id_1", "memcell_id_3"],
      "confidence": "strong|weak"
    }}
  ],
  "summary": "基于当前和以前话题关注当前阶段的一句话",
  "subject": "长期群组定位或not_found"
}}
```
</output_format>

<extraction_rules>
### 话题 (0-{max_topics})
- **选择**：从对话中选择前 {max_topics} 个最实质性和有意义的讨论线索
- **最低要求**：每个话题必须涉及至少 5 条消息或 3+ 个参与者讨论同一线索
- **粒度要求**：话题应代表重要的工作主题，而不是单个任务或协调活动
- **不要生成话题 ID**：系统将在提取后生成 ID
- **名称**：简短短语（2-4 个词）捕获本质
- **摘要**：一句话描述群组在此话题上讨论什么（最多 3 句）
- **增量更新逻辑**：
  - **如果 existing_group_profile 为空**：将所有话题设置为"new"（update_type="new"，old_topic_id=null）
  - **如果 existing_group_profile 有话题**：与现有话题比较并决定：
    - **"update"**：如果此话题继续/发展现有话题（提供 old_topic_id）
    - **"new"**：如果这是全新的讨论话题（old_topic_id=null）
  - **焦点**：仅提供"new"和"update"操作。系统将自动处理话题管理。
- **状态评估**：
  - **"exploring"**：初步讨论，收集信息，提出问题
  - **"disagreement"**：表达多种观点，辩论持续，无共识
  - **"consensus"**：达成一致，做出决定，准备行动
  - **"implemented"**：已执行/完成，提到结果
- **证据与置信度**：
  - **"evidences"**：支持此话题识别的 memcell ID 列表（来自提供的对话）
  - **"confidence"**："strong" 如果多个明确证据和强信号；"weak" 如果证据有限或模糊

**话题质量指南**（要包含的内容）：
- **技术讨论**：架构决策、代码审查、系统设计、API 设计
- **业务决策**：战略规划、产品路线图、功能优先级
- **问题解决**：错误调查、性能问题、故障排除
- **项目管理**：冲刺规划、里程碑审查、资源分配
- **知识分享**：技术解释、最佳实践、学习会议
- **战略规划**：长期目标、技术选择、流程改进

**话题排除指南**（要排除的内容）：
- **行政任务**：会议安排、日历邀请、会议室预订、会议取消
- **社交互动**：问候、闲聊、个人更新、天气聊天
- **系统通知**：机器人消息、自动警报、状态更新
- **后勤协调**："我会迟到 5 分钟"、"你能分享链接吗？"
- **简单确认**："好的"、"明白了"、"谢谢"、单词回复
- **程序请求**：文件共享请求、访问权限、工具设置
- **群组管理**：添加/删除成员、权限更改、实习生邀请
- **日常操作**：每日站会报告、简单状态更新、例行检查
- **活动协调**：会议安排、日程协调、场地预订
- **重复行政**：每日会议、每周站会、定期状态同步
- **人力资源/人事任务**：实习生招聘、入职程序、团队介绍
- **基本协调**：时间确认、位置共享、简单后勤

**选择优先级**：关注涉及多个参与者、跨越多条消息、包含推动群组目标前进的实质性内容的话题，代表有意义的工作讨论而不是协调开销。

### 摘要
- **来源**：基于话题数组中的话题
- **格式**：基于当前和以前话题描述当前群组焦点的一句话
- **语言**：使用与对话**相同的语言**
- **模板**：
  - 中文："目前主要关注..."
  - 英文："Currently focusing on..."

### 主题
- **优先来源**：
  1. 明确的群组描述、公告
  2. 跨对话的一致模式
  3. 群组名称分析
  4. 如果证据不足，则为"not_found"
- **稳定性**：在提取过程中应保持相对稳定
- **示例**："产品开发团队"、"营销策略组"、"技术支持"
</extraction_rules>

<update_logic>
1. **新提取**：如果未提供 existing_group_profile，则从对话中新提取
2. **增量更新**：如果存在现有档案：
   - **话题**：将新话题与现有话题比较
     - **更新**：如果话题继续/发展，提供 old_topic_id 和更新信息
     - **新增**：如果是全新的讨论话题，标记为"new"
   - **摘要**：基于现有话题和新话题重新生成
   - **主题**：除非有强有力的矛盾证据，否则保留现有的
</update_logic>

## 语言要求
- **内容语言**：提取话题、摘要和主题时使用与对话内容**相同的语言**
- **枚举值**：按照指定，将所有枚举值（状态值）保持为英文
- **示例**：如果对话是中文，topics.name 和 summary 应该是中文，但 status 应该保持为"exploring/consensus/等"

现在分析提供的对话并按照上述指南提取内容分析。重点关注基于证据的提取和保守评估。仅返回输出格式中指定的 JSON 对象。
"""

BEHAVIOR_ANALYSIS_PROMPT = """
你是一位群组行为分析专家，专门分析沟通模式以根据对话行为识别群组角色。

**重要证据提取：**
- 每个对话片段都以"=== MEMCELL_ID: xxxx ==="作为前缀来标识 memcell
- 提供证据时，仅使用这些"=== MEMCELL_ID: xxx ==="标记中的确切 memcell ID
- 不要使用时间戳（如 [2025-09-01T09:30:55.669000+00:00]）作为 memcell ID - 这些不是 memcell ID！
- 仅引用对话输入中以"=== MEMCELL_ID: ==="格式出现的 memcell ID
- 示例：如果您看到"=== MEMCELL_ID: abc-123-def ==="，在证据列表中使用"abc-123-def"

你的任务是分析群组对话记录并提取：
**角色映射**（基于行为模式的 7 个关键角色分配）

<principles>
- **基于证据**：仅在有来自对话的明确行为证据时分配角色
- **质量优于数量**：将角色留空胜过错误分配
- **保守分配**：不确定时，将角色留空而不是猜测
- **最低证据**：角色分配需要至少 2 个明确的行为示例
- **组织意识**：在可用时考虑团队/经理上下文
</principles>

<input>
- **conversation_transcript**: {conversation}
- **group_id**: {group_id}
- **group_name**: {group_name}
- **existing_group_profile**: {existing_profile}
{speaker_info}
</input>

<output_format>
你必须输出一个具有以下结构的单个 JSON 对象：

```json
{{
  "roles": {{
    "decision_maker": [
      {{
        "speaker": "speaker_id1",
        "evidences": ["memcell_id_2"],
        "confidence": "strong|weak"
      }}
    ],
    "opinion_leader": [
      {{
        "speaker": "speaker_id2",
        "evidences": ["memcell_id_4", "memcell_id_5"],
        "confidence": "strong|weak"
      }}
    ],
    "topic_initiator": [...],
    "execution_promoter": [...],
    "core_contributor": [...],
    "coordinator": [...],
    "info_summarizer": [...]
  }}
}}
```
</output_format>

<extraction_rules>
### 角色（7 个关键角色）
对于每个角色，根据对话行为识别用户，**至少需要 2 个明确示例**：

- **decision_maker（决策者）**：做最终决定，批准/拒绝提案，具有权威
  - 标志："我们采用..."、"我批准"、"决定是..."、其他人听从他们
- **opinion_leader（意见领袖）**：多人引用他们的观点，影响群组思考
  - 标志：其他人引用他们，寻求他们的意见，"如 X 提到的..."、思想领导力
- **topic_initiator（话题发起人）**：开始新讨论线索，提出新主题
  - 标志："我想讨论..."、"那么..."、"我们应该谈谈..."、引入话题
- **execution_promoter（执行推动者）**：推动行动，跟进任务，推动实施
  - 标志："什么时候完成？"、"让我们继续"、"我们需要行动"、面向任务
- **core_contributor（核心贡献者）**：提供知识、资源、专业知识、实质性投入
  - 标志：详细解释，分享资源，教导他人，领域专业知识
- **coordinator（协调者）**：促进协作，解决冲突，管理流程
  - 标志："让我们对齐..."、调解分歧、组织会议、流程焦点
- **info_summarizer（信息总结者）**：创建摘要、会议记录、总结、文档
  - 标志："总结一下..."、"这是我们决定的..."、做笔记、综合

**分配规则**：
- 一个人可以有多个角色
- **每个角色最多 3 人** - 仅选择最活跃/明确的示例
- 仅使用输入中提供的可用说话者列表中的 speaker_id
- **组织上下文**：在可用时，将团队/经理信息作为角色分配的支持证据
- 如果角色证据不足，则将其留空
- 分配需要最少 2 个明确的行为示例
- 保守 - 错过角色胜过错误分配
- **保留历史角色**：当现有档案有角色分配时，除非被新证据矛盾，否则保留它们
- **添加新角色**：基于新对话行为添加新角色分配
- **仅删除角色**：如果有明确证据表明角色变化或被新活跃说话者替换
- **证据与置信度**：对于每个角色分配，提供 memcell ID 作为证据并评估置信度级别
  - **"evidences"**：支持此角色分配的 memcell ID 列表
  - **"confidence"**："strong" 如果多个明确的行为模式；"weak" 如果证据有限
- 输出格式：[{{"speaker": "speaker_id", "evidences": ["memcell_id1"], "confidence": "strong|weak"}}] 对于每个角色

### 角色分配示例：
- 如果 Alice 经常说"我认为我们应该..."并且其他人跟随：opinion_leader
- 如果 Bob 总是问"什么时候准备好？"并推动截止日期：execution_promoter
- 如果 Carol 以"我想讨论..."开始大多数新话题：topic_initiator
</extraction_rules>

<conversation_examples>
要识别的示例行为模式：

**话题发起**："我想讨论部署计划"、"客户反馈怎么样？"
**决策制定**："我们采用选项 A"、"我批准这种方法"、"决定是..."
**意见领导**："根据我的经验..."、"如我之前提到的..."、其他人引用观点
**执行焦点**："什么时候完成？"、"我们需要推进这个"、"让我们设定截止日期"
**知识贡献**：详细的技术解释，分享资源，专家见解
**协调**："让我们对齐这个"、"我会安排会议"、"我们需要同步"
**总结**："回顾我们讨论的..."、"这是摘要..."、"主要要点是..."
</conversation_examples>

现在分析提供的对话并按照上述指南提取角色分配。重点关注基于证据的分配和保守评估。仅返回输出格式中指定的 JSON 对象。
"""


AGGREGATION_PROMPT = """
你是一位群组档案聚合专家。你的任务是分析多个每日群组档案和对话数据以创建合并的群组档案。

**重要证据提取：**
- 每个对话片段都以 [MEMCELL_ID: xxxx] 作为前缀来标识 memcell
- 提供证据时，使用这些前缀中的确切 memcell ID
- 仅引用对话输入中出现的 memcell ID

你正在从 {aggregation_level} 数据（{start_date} 到 {end_date}）聚合群组档案。

每日档案摘要：
{daily_context}

对话数据：
{conversation}

请分析并提供一个合并的群组档案，该档案综合了上述每日档案和对话数据的见解。

输出一个具有以下结构的单个 JSON 对象：
{{
  "topics": [
    {{
      "name": "话题名称",
      "summary": "话题摘要",
      "status": "exploring|disagreement|consensus|implemented",
      "update_type": "new|update",
      "old_topic_id": "topic_id",
      "evidences": ["memcell_id1", "memcell_id2"],
      "confidence": "strong|weak"
    }}
  ],
  "summary": "合并的群组摘要",
  "subject": "群组主题或not_found",
  "roles": {{
    "decision_maker": [
      {{
        "speaker": "speaker_id",
        "evidences": ["memcell_id"],
        "confidence": "strong|weak"
      }}
    ]
  }}
}}

重点关注在时间段内合并信息，识别一致模式，并提供基于证据的见解。
"""

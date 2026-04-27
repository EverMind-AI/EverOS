"""Life Profile Memory Prompts - Chinese Version.

Explicit information + Implicit traits extraction.
"""

# Incremental Update Prompt
PROFILE_UPDATE_PROMPT = '''你是用户画像更新员。根据对话记录，判断需要对用户画像做哪些操作。

【当前用户画像】（每条都有 index 编号）
{current_profile}

【对话记录】（来自同一主题的多轮对话）
{conversations}

【任务】
分析对话，输出需要执行的操作列表（可以有多条操作）。可选操作类型：
- **update**: 修改现有条目（通过 index 指定）
- **add**: 新增画像条目
- **delete**: 删除现有条目
- **none**: 无需任何操作（当对话不包含任何用户信息时使用）

【操作选择指南】
- **update**: 现有条目有信息更新、补充、修改
- **add**: 发现全新的用户信息（与现有条目无关）
- **delete**: 以下情况应该删除：
  - 用户明确否定（如"我不再吃素了"）
  - 信息已过时（如"下周要出差"但已经过了）
  - 与新信息直接矛盾

【重要规则】
1. **挖掘标签**：隐式特征必须包含【性格标签】，例如：[风险厌恶型]、[社交驱动型]、[数据考据党]。
2. 只提取用户信息，不要把 AI 助手的建议当成用户特征
3. sources 格式：使用对话 ID（方括号里的，如 ep1, ep2）
4. evidence 要包含时间信息 - 如"2024年10月用户提到..."
5. explicit_info 和 implicit_traits 的 index 是独立编号的
6. **去重**：在使用 "add" 前，仔细检查所有已有条目。如果类似的特征/信息已存在（即使措辞不同），请用 "update" 来补充而非重复添加。只有确实全新的信息才用 "add"。

【画像定义与分析框架】
- **explicit_info（显式信息）**：可以直接从对话中提取的用户事实。
  - *包含内容*：基本资料、健康状况、能力技能、明确偏好等。

- **implicit_traits（隐式特征）**：基于行为推断的心理画像、性格标签和决策风格。
  - *提取要求*：请结合对话上下文，从决策模式、社交偏好、生活哲学等维度进行自由分析和概括。
  - *命名规范*：
    1. 标签必须简练、可读、可复用（便于检索/对比），尽量控制在 2-6 个字。
    2. 避免把多个维度硬拼成一个长标签；如果信息包含多个维度，请拆成多条隐式特征分别表达。
    3. 标签应描述“稳定的行为/心理倾向”，不要写成一次性的事件或短期状态。
  - 请做合理推理，提取出用户的深层特征
【输出格式】
无操作时：
```json
{{"operations": [{{"action": "none"}}], "update_note": "对话不包含用户信息"}}
```

有操作时（可以组合多条 add/update/delete）：
```json
{{
  "operations": [
    {{"action": "add", "type": "explicit_info", "data": {{"category": "...", "description": "...", "evidence": "...", "sources": ["ep1"]}}}},
    {{"action": "add", "type": "implicit_traits", "data": {{"trait": "...", "description": "...", "basis": "...", "evidence": "...", "sources": ["ep1", "ep2"]}}}},
    {{"action": "update", "type": "explicit_info", "index": 0, "data": {{"description": "...", "sources": ["ep3"]}}}},
    {{"action": "delete", "type": "implicit_traits", "index": 1, "reason": "..."}}
  ],
  "update_note": "新增2条显式信息和1条隐式特征，更新1条，删除1条"
}}
```'''

# Compacting Prompt
PROFILE_COMPACT_PROMPT = '''当前用户画像有 {total_items} 条记录（explicit_info + implicit_traits 合计），超过了上限 {max_items} 条。

请精简画像至 **合计 {max_items} 条**（explicit_info + implicit_traits 两类加起来，不是每类 {max_items} 条）。

精简原则：
1. **合并同类项**：将同一维度的多条记录（如多次体重记录）合并为一条"当前状态+趋势"的描述。
2. **提炼标签**：隐式特征应归纳为性格标签（如[风险厌恶型]），删除重复或浅层的描述。
3. 删除不重要、已过时或短期状态。
4. 保留每条条目的字段完整（尤其是 evidence / sources）。

当前画像：
{profile_text}

**重要**：输出的 explicit_info + implicit_traits 合计必须 ≤ {max_items} 条。
```json
{{
  "explicit_info": [
    {{"category": "...", "description": "...", "evidence": "...", "sources": ["episode_id"]}}
  ],
  "implicit_traits": [
    {{"trait": "...", "description": "...", "basis": "...", "evidence": "...", "sources": ["id1", "id2"]}}
  ],
  "compact_note": "说明删除/合并了哪些内容"
}}
```'''

# Initial Extraction Prompt
PROFILE_INITIAL_EXTRACTION_PROMPT = '''你是一个"用户画像分析师"。请阅读下面的对话，构建用户画像。

【第一部分：显式信息 (explicit_info)】
用户的客观事实和当前状态，如身高体重、喜好、疾病等。

【第二部分：隐式特征 (implicit_traits)】
基于行为推断的心理画像、性格标签和决策风格。
*提取要求*：从决策、社交、生活观念等维度进行深度挖掘。
*命名规范*：Trait 字段必须简练精准，推荐“[形容词] [名词]”格式，严禁过度堆砌形容词。

【提取原则】
1. 只提取用户本人的信息，不要把助手的建议当成用户特征
2. 隐式特征必须有多个证据支撑：同一条隐式特征的 sources 至少包含 2 个来源；证据可以来自【当前对话】与/或【已有画像 current_profile 的 evidence/sources】（更新时可用），不能仅凭单条新对话臆断
3. 每条信息用一句自然语言描述，通俗易懂
4. 标注信息来源（消息编号）

【输出格式】
请直接输出 JSON，格式如下：
```json
{{
  "explicit_info": [
    {{
      "category": "分类名",
      "description": "一句话描述",
      "evidence": "一句话证据（来自对话内容）",
      "sources": ["YYYY-MM-DD HH:MM|episode_id"]
    }}
  ],
  "implicit_traits": [
    {{
      "trait": "特征名称",
      "description": "一句话描述这个特征",
      "basis": "从哪些行为/对话推断出来的",
      "evidence": "一句话证据（来自对话内容）",
      "sources": ["YYYY-MM-DD HH:MM|episode_id1", "YYYY-MM-DD HH:MM|episode_id2"]
    }}
  ]
}}
```

【对话原文】
{conversation_text}'''


# ============================================================================
# TEAM 专用提示词（多人群聊场景）
# ============================================================================

TEAM_PROFILE_UPDATE_PROMPT = '''你是**群聊场景**的用户画像更新员。你的任务是从多人对话中，仅提取和更新**一个特定用户**的画像。

**目标用户：{target_user}**
你必须只提取关于 **{target_user}** 的信息。仔细区分每条信息的发言者，不要把其他参与者的信息混入 {target_user} 的画像。

【{target_user} 的当前画像】（每条都有 index 编号）
{current_profile}

【群聊对话记录】（多个参与者 - 只提取 {target_user} 的信息）
{conversations}

【任务】
分析对话，仅输出关于 **{target_user}** 的操作列表。可选操作类型：
- **update**: 修改现有条目（通过 index 指定）
- **add**: 新增画像条目
- **delete**: 删除现有条目
- **none**: 无需任何操作（当对话不包含 {target_user} 的信息时使用）

【操作选择指南】
- **update**: 现有条目有信息更新、补充、修改
- **add**: 发现 {target_user} 的全新信息（与现有条目无关）
- **delete**: 以下情况应该删除：
  - {target_user} 明确否定（如"我不再吃素了"）
  - 信息已过时或与新信息直接矛盾

【重要规则】
1. **发言者归属**：这是一个多人群聊。只提取 **{target_user}** 说的话或明确关于 {target_user} 的信息。如果其他参与者提到某个事实，那属于其他人的画像，不是 {target_user} 的。
2. **挖掘标签**：隐式特征必须包含【性格标签】，例如：[风险厌恶型]、[社交驱动型]、[数据考据党]。
3. sources 格式：使用对话 ID（方括号里的，如 ep1, ep2）
4. evidence 要包含时间和发言者信息 - 如"2024年10月 {target_user} 提到..."
5. explicit_info 和 implicit_traits 的 index 是独立编号的
6. **去重**：在使用 "add" 前，检查所有已有条目。如果类似信息已存在，用 "update" 补充。只有确实全新的信息才用 "add"。

【画像定义】
- **explicit_info（显式信息）**：{target_user} 直接陈述的或关于其本人的事实（技能、背景、偏好、所在地等）
- **implicit_traits（隐式特征）**：从 {target_user} 在对话中的表述和行为推断出的性格特征和行为模式

【输出格式】
无操作时：
```json
{{"operations": [{{"action": "none"}}], "update_note": "对话不包含 {target_user} 的信息"}}
```

有操作时：
```json
{{
  "operations": [
    {{"action": "add", "type": "explicit_info", "data": {{"category": "...", "description": "...", "evidence": "...", "sources": ["ep1"]}}}},
    {{"action": "add", "type": "implicit_traits", "data": {{"trait": "...", "description": "...", "basis": "...", "evidence": "...", "sources": ["ep1", "ep2"]}}}},
    {{"action": "update", "type": "explicit_info", "index": 0, "data": {{"description": "...", "sources": ["ep3"]}}}},
    {{"action": "delete", "type": "implicit_traits", "index": 1, "reason": "..."}}
  ],
  "update_note": "..."
}}
```'''

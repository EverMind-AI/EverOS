"""Chinese prompt for ``rank.skill.arank`` LLM rerank.

Anchors on the enterprise hybrid-rerank ``instruction`` passed to the cross-encoder in
``search_mem_service._search_agent_skills`` — methodology + domain applicability, same-domain
preference, directly-relevant steps. The detailed 0.0-1.0 scoring bands previously carried by the
separate post-rerank verify stage are folded in here so a single LLM pass both reorders and
quality-grades candidates — the standalone verify stage is gone.

The constant is the system message of the call: the rules, ``{{top_k}}`` their only placeholder, closing
with ``<reply_format>``. The query and the candidates go in the user message,
``prompts.rerank_user.RERANK_USER_PROMPT``. Split from a single user message without rewriting a rule: the
「用户查询：」 label and the data moved to the user message, the candidates line now points at
``<candidates>``, Markdown emphasis (``**``, inline backticks) was dropped, and the skeleton's
``<0..1 浮点数>`` became a parseable number (the 0.0-1.0 range is stated in the opening sentence).
"""

from __future__ import annotations

__all__ = ["SKILL_RERANK_PROMPT_ZH"]

from everos.algo.utils import paragraphs, tagged

_OPENING = """\
判断每个 agent skill 的方法（methodology）和领域（domain）是否适用于用户查询，优先选择同领域且步骤直接相关的 skill。按 0.0（完全不适用）到 1.0（方法和领域都强适配）给每个 skill 打分。"""

_EVALUATION = """\
评估每个 skill 时同时考虑：
- skill 的步骤或方法（steps / approach）是否适用于查询的问题类型
- skill 的目标领域（由 description / 触发场景 / 关键词体现）是否与查询的主题重叠 —— 同领域的 skill 应该打更高分

打分区间：
- 0.0: 完全不相关 —— 方法不适用、领域也不重叠
- 0.1-0.3: 弱相关 —— 方法勉强能套，但领域不一样
- 0.4-0.6: 中等有帮助 —— 方法可用，领域部分重叠
- 0.7-0.8: 有帮助 —— 方法适用且领域对齐良好
- 0.9-1.0: 高度相关 —— 方法和领域都强匹配"""

_CANDIDATES = """\
候选列表在 <candidates> 中（JSON 数组；每项含 id、score，以及元数据如 name / description / content / confidence）。"""

_RANKING = """\
最多返回 {{top_k}} 项，按 score 降序排列。分数接近 0 的候选直接丢弃。"""

_REPLY_FORMAT = """\
只输出以下 JSON，不要包含任何解释文字：
{"ranked": [{"id": "<候选 id>", "score": 0.9}]}"""

SKILL_RERANK_PROMPT_ZH = paragraphs(
    _OPENING,
    tagged("evaluation", _EVALUATION),
    _CANDIDATES,
    _RANKING,
    tagged("reply_format", _REPLY_FORMAT),
)

"""Chinese prompt for ``rank.case.arank`` LLM rerank.

Derived in the same style as ``SKILL_RERANK_PROMPT_ZH`` and ``EPISODIC_RERANK_PROMPT_ZH``
(single-instruction framing). Case has no cross-encoder rerank stage in the enterprise
hybrid path (it uses ``vector_anchored`` fusion instead), so this prompt is a parallel
extrapolation rather than a port of an existing instruction.

The constant is the system message of the call: the rules, ``{{top_k}}`` their only placeholder, closing
with ``<reply_format>``. The query and the candidates go in the user message,
``prompts.rerank_user.RERANK_USER_PROMPT``. Split from a single user message without rewriting a rule: the
「用户查询：」 label and the data moved to the user message, the candidates line now points at
``<candidates>``, Markdown emphasis (``**``, inline backticks) was dropped, and the skeleton's
``<0..1 浮点数>`` became a parseable number (the 0.0-1.0 range is stated in the opening sentence).
"""

from __future__ import annotations

__all__ = ["CASE_RERANK_PROMPT_ZH"]

from everos.algo.utils import paragraphs, tagged

_OPENING = """\
判断每个 agent 执行案例（case）的任务和方法是否对回答用户查询有用，优先选择 quality_score 高、经过验证成功的方法且直接适用于查询问题的 case。按 0.0（没有可迁移的经验）到 1.0（任务模式相同且方法已被验证成功）给每个 case 打分。"""

_CANDIDATES = """\
候选列表在 <candidates> 中（JSON 数组；每项含 id、score，以及元数据如 task_intent / approach / quality_score / key_insight）。"""

_RANKING = """\
最多返回 {{top_k}} 项，按 score 降序排列。分数接近 0 的候选直接丢弃。"""

_REPLY_FORMAT = """\
只输出以下 JSON，不要包含任何解释文字：
{"ranked": [{"id": "<候选 id>", "score": 0.9}]}"""

CASE_RERANK_PROMPT_ZH = paragraphs(
    _OPENING,
    _CANDIDATES,
    _RANKING,
    tagged("reply_format", _REPLY_FORMAT),
)

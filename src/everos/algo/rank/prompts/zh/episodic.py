"""Chinese prompt for ``rank.episodic.arank`` LLM rerank.

Anchors on the enterprise default cross-encoder ``instruction`` used by
``rerank_deepinfra``/``rerank_vllm`` when no per-call instruction is supplied —
``_search_episodic_memory`` does not pass an ``instruction``, so episodic hybrid
rerank inherits this generic relevance check.

The constant is the system message of the call: the rules, ``{{top_k}}`` their only placeholder, closing
with ``<reply_format>``. The query and the candidates go in the user message,
``prompts.rerank_user.RERANK_USER_PROMPT``. Split from a single user message without rewriting a rule: the
「用户查询：」 label and the data moved to the user message, the candidates line now points at
``<candidates>``, Markdown emphasis (``**``, inline backticks) was dropped, and the skeleton's
``<0..1 浮点数>`` became a parseable number (the 0.0-1.0 range is stated in the opening sentence).
"""

from __future__ import annotations

__all__ = ["EPISODIC_RERANK_PROMPT_ZH"]

from everos.algo.utils import paragraphs, tagged

_OPENING = """\
给定用户查询和一组检索到的 episodic 记忆候选（其中可能混入了由 episode 展开出的 atomic_fact 项），判断每个候选是否包含与查询相关的信息。按 0.0（不包含相关信息）到 1.0（直接回答查询）给每个候选打分。"""

_CANDIDATES = """\
候选列表在 <candidates> 中（JSON 数组；每项含 id、score、item_type 为 episode 或 atomic_fact，以及元数据如 episode / subject / summary / parent_episode_id）。"""

_RANKING = """\
最多返回 {{top_k}} 项，按 score 降序排列。分数接近 0 的候选直接丢弃。"""

_REPLY_FORMAT = """\
只输出以下 JSON，不要包含任何解释文字：
{"ranked": [{"id": "<候选 id>", "score": 0.9}]}"""

EPISODIC_RERANK_PROMPT_ZH = paragraphs(
    _OPENING,
    _CANDIDATES,
    _RANKING,
    tagged("reply_format", _REPLY_FORMAT),
)

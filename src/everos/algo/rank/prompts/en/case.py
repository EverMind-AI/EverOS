"""English prompt for ``rank.case.arank`` LLM rerank.

Derived in the same style as ``SKILL_RERANK_PROMPT_EN`` and ``EPISODIC_RERANK_PROMPT_EN``
(single-instruction framing). Case has no cross-encoder rerank stage in the enterprise
hybrid path (it uses ``vector_anchored`` fusion instead), so this prompt is a parallel
extrapolation rather than a port of an existing instruction.

The constant is the system message of the call: the rules, ``{{top_k}}`` their only placeholder, closing
with ``<reply_format>``. The query and the candidates go in the user message,
``prompts.rerank_user.RERANK_USER_PROMPT``. Split from a single user message without rewriting a rule: the
"User query:" label and the data moved to the user message, the candidates line now points at
``<candidates>``, Markdown emphasis (``**``, inline backticks) was dropped, and the skeleton's
``<float in 0..1>`` became a parseable number (the 0.0-1.0 range is stated in the opening sentence).
"""

from __future__ import annotations

__all__ = ["CASE_RERANK_PROMPT_EN"]

from everos.algo.utils import paragraphs, tagged

_OPENING = """\
Determine whether each agent execution case's task and approach are useful for addressing the user query, preferring cases whose verified successful approach (high quality_score) directly applies to the query's problem. Score each case from 0.0 (no transferable experience) to 1.0 (same task pattern with a verified successful approach)."""

_CANDIDATES = """\
Candidates in <candidates> (JSON array; each item has id, score, plus metadata like task_intent, approach, quality_score, key_insight)."""

_RANKING = """\
Return at most {{top_k}} items, sorted by score descending. Drop candidates whose score would be near 0."""

_REPLY_FORMAT = """\
Output strictly the following JSON, no prose:
{"ranked": [{"id": "<candidate_id>", "score": 0.9}]}"""

CASE_RERANK_PROMPT_EN = paragraphs(
    _OPENING,
    _CANDIDATES,
    _RANKING,
    tagged("reply_format", _REPLY_FORMAT),
)

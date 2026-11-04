"""English prompt for ``rank.episodic.arank`` LLM rerank.

Anchors on the enterprise default cross-encoder ``instruction`` used by
``rerank_deepinfra``/``rerank_vllm`` when no per-call instruction is supplied —
``_search_episodic_memory`` does not pass an ``instruction``, so episodic hybrid
rerank inherits this generic relevance check.

The constant is the system message of the call: the rules, ``{{top_k}}`` their only placeholder, closing
with ``<reply_format>``. The query and the candidates go in the user message,
``prompts.rerank_user.RERANK_USER_PROMPT``. Split from a single user message without rewriting a rule: the
"User query:" label and the data moved to the user message, the candidates line now points at
``<candidates>``, Markdown emphasis (``**``, inline backticks) was dropped, and the skeleton's
``<float in 0..1>`` became a parseable number (the 0.0-1.0 range is stated in the opening sentence).
"""

from __future__ import annotations

__all__ = ["EPISODIC_RERANK_PROMPT_EN"]

from everos.algo.utils import paragraphs, tagged

_OPENING = """\
Given the user query and a list of retrieved episodic memories (or atomic facts extracted from them), determine whether each candidate contains information relevant to answering the query. Score each candidate from 0.0 (no relevant information) to 1.0 (directly answers the query)."""

_CANDIDATES = """\
Candidates in <candidates> (JSON array; each item has id, score, an item_type of either episode or atomic_fact, plus metadata like episode / subject / summary / parent_episode_id)."""

_RANKING = """\
Return at most {{top_k}} items, sorted by score descending. Drop candidates whose score would be near 0."""

_REPLY_FORMAT = """\
Output strictly the following JSON, no prose:
{"ranked": [{"id": "<candidate_id>", "score": 0.9}]}"""

EPISODIC_RERANK_PROMPT_EN = paragraphs(
    _OPENING,
    _CANDIDATES,
    _RANKING,
    tagged("reply_format", _REPLY_FORMAT),
)

"""The user message every LLM rerank call sends, whichever facade's rules and language the system message carries.

It holds only the data, each input in its own tag, and is always rendered by ``rank.rerank.arerank``; the rules,
``{{top_k}}`` included, are the system message (``prompts/{en,zh}/{case,episodic,skill}.py``). The tags are
language-neutral, so the English and Chinese rules share it.

Placeholders: ``{{query}}`` and ``{{candidates_json}}``, the candidates as a JSON array of ``{"id", "score",
**metadata}`` objects.
"""

from __future__ import annotations

__all__ = ["RERANK_USER_PROMPT"]

from everos.algo.utils import paragraphs, tagged

RERANK_USER_PROMPT = paragraphs(
    tagged("user_query", "{{query}}"),
    tagged("candidates", "{{candidates_json}}"),
)

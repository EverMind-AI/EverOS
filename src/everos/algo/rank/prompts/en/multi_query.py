"""Multi-query generation prompts for ``aagentic_retrieve``: a system message of rules and a user message of data.

The SYSTEM message holds the rules: the role lines, ``<strategy_selection>``, ``<query_style_requirements>``,
``<requirements>``, and ``<reply_format>`` (one sentence plus the JSON skeleton). It names no request data, so every
call sends the same text; a caller's ``custom_instructions`` follows it as a closing ``<custom_instructions>`` block.
The USER message carries only the data, each input in its own tag, and is always rendered by the operator.

Every rule line comes from the single-message prompt the split replaced, unchanged apart from the structure: the data
and the separator lines around it moved to the user message, the Markdown headings became tags (a heading that states
a rule of its own stays as the block's first line), ``**`` and inline backticks were dropped, the closing
"Now generate:" cue went with the data it introduced, and the "STRICT JSON" heading gave way to the reply-format
sentence. The skeleton was already parseable and is kept as written.

Placeholders, user template only: ``{{original_query}}``, ``{{key_info}}``, ``{{missing_info}}`` and
``{{retrieved_docs}}``, the documents as ``_format_docs`` renders them.

Reply shape: ``{"queries": list[str], "reasoning": str}``, held to ``rank._replies._MultiQueryReply``.
"""

from __future__ import annotations

__all__ = ["MULTI_QUERY_PROMPT_EN", "MULTI_QUERY_USER_PROMPT"]

from everos.algo.utils import paragraphs, tagged

_ROLE = """\
You are an expert at query reformulation for conversational memory retrieval.
Your goal is to generate 2-3 complementary queries to find the MISSING information."""

_STRATEGY_SELECTION = """\
Strategy Selection (Choose based on WHY info is missing):

[A] Pivot / Entity Association (If entity is missing)
- If the specific entity is not found, search for related entities or broader categories.
- Example: "manager's feedback" not found → try "performance review", "work evaluation".

[B] Temporal Calculation (If time is missing/unclear)
- Use Date from Retrieved Documents to anchor relative times.
- Example: doc dated 2024-03-15 mentions "last month" → search for "February 2024".
- Search for the *event* to find its timestamp: "When did the deadline change?"

[C] Concept Expansion (If vocabulary mismatch)
- Synonyms: "residence" → "living", "staying at", "moved to".
- General/Specific: "Italian cuisine" ↔ "pasta", "pizza", "restaurant".

[D] Constraint Relaxation (If too specific)
- If "quarterly sales report from Q3" fails, try "sales report", "Q3 results".
- Remove one constraint at a time."""

_QUERY_STYLE_REQUIREMENTS = """\
Query Style Requirements (Use DIFFERENT styles):

1. Keyword Combo (2-5 words): Key entities only. High recall.
   - e.g., "project deadline", "vacation plans summer"
2. Natural Question (5-10 words): Rephrased question.
   - e.g., "When was the meeting scheduled?", "What was discussed about the budget?"
3. Hypothetical Statement (HyDE, 5-10 words): A likely sentence in the memory.
   - e.g., "We decided to postpone the launch", "The client requested changes\""""

_REQUIREMENTS = """\
- Generate 2-3 queries.
- CRITICAL: Use the strategies above to target the *Missing Information*.
- Keep queries SHORT and SEARCHABLE."""

_REPLY_FORMAT = """\
Reply with one JSON object, nothing else.
{
  "queries": [
    "Query 1",
    "Query 2",
    "Query 3"
  ],
  "reasoning": "Strategy used for each query (e.g., Q1: Pivot, Q2: Temporal)"
}"""

MULTI_QUERY_PROMPT_EN = paragraphs(
    _ROLE,
    tagged("strategy_selection", _STRATEGY_SELECTION),
    tagged("query_style_requirements", _QUERY_STYLE_REQUIREMENTS),
    tagged("requirements", _REQUIREMENTS),
    tagged("reply_format", _REPLY_FORMAT),
)

MULTI_QUERY_USER_PROMPT = paragraphs(
    tagged("original_query", "{{original_query}}"),
    tagged("key_information_found", "{{key_info}}"),
    tagged("missing_information", "{{missing_info}}"),
    tagged("retrieved_documents", "{{retrieved_docs}}"),
)

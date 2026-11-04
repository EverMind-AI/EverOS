"""Sufficiency check prompts for ``aagentic_retrieve``: a system message of rules and a user message of data.

The SYSTEM message holds the rules: the role sentence, ``<instructions>``, and ``<reply_format>`` (one sentence plus
the JSON skeleton). It names no request data, so every call sends the same text; a caller's ``custom_instructions``
follows it as a closing ``<custom_instructions>`` block. The USER message carries only the data, each input in its own
tag, and is always rendered by the operator.

Every rule line comes from the single-message prompt the split replaced, unchanged apart from the structure: the data
and the separator lines around it moved to the user message, the Markdown headings became tags, ``**`` and inline
backticks were dropped, the closing "Now evaluate:" cue went with the data it introduced, and the skeleton's
``true or false`` became a parseable ``false``, its meaning moved into the reply-format sentence.

Placeholders, user template only: ``{{query}}`` and ``{{retrieved_docs}}``, the documents as ``_format_docs`` renders
them.

Reply shape: ``{"is_sufficient": bool, "reasoning": str, "key_information_found": list[str],
"missing_information": list[str]}``, held to ``rank._replies._SufficiencyReply``.
"""

from __future__ import annotations

__all__ = ["SUFFICIENCY_CHECK_PROMPT_EN", "SUFFICIENCY_CHECK_USER_PROMPT"]

from everos.algo.utils import paragraphs, tagged

_ROLE = (
    "You are an expert in information retrieval evaluation. Assess whether the retrieved documents provide "
    "sufficient information to answer the user's query."
)

_INSTRUCTIONS = """\
1. Analyze the Query's Needs
   - Entities: Who/What is being asked about?
   - Attributes: What specific details (color, time, location, quantity)?
   - Time: Does it ask for a specific time (absolute or relative like "last week")?

2. Evaluate Document Evidence
   - Check Content: Do the documents mention the entities and attributes?
   - Check Dates:
     - Use the Date field of each document.
     - For relative time queries (e.g., "last week", "yesterday"), verify if document dates fall within that timeframe.
     - If the query asks "When did X happen?", do you have the specific date or just a vague mention?

3. Judgment Logic
   - Sufficient: You can answer the query *completely* and *precisely* using ONLY the provided documents.
   - Insufficient:
     - The specific entity is not found.
     - The entity is found, but the specific attribute (e.g., "price") is missing.
     - The time reference cannot be resolved (e.g., doc says "yesterday" but has no date, or doc date doesn't match query timeframe).
     - Conflicting information without resolution."""

_REPLY_FORMAT = """\
Reply with one JSON object, nothing else. is_sufficient is true or false.
{
  "is_sufficient": false,
  "reasoning": "Brief explanation. If insufficient, state WHY (e.g., 'Found X but missing date', 'No mention of Y').",
  "key_information_found": ["Fact 1 (Source: Doc 1)", "Fact 2 (Source: Doc 2)"],
  "missing_information": ["Specific gap 1", "Specific gap 2"]
}"""

SUFFICIENCY_CHECK_PROMPT_EN = paragraphs(
    _ROLE,
    tagged("instructions", _INSTRUCTIONS),
    tagged("reply_format", _REPLY_FORMAT),
)

SUFFICIENCY_CHECK_USER_PROMPT = paragraphs(
    tagged("user_query", "{{query}}"),
    tagged("retrieved_documents", "{{retrieved_docs}}"),
)

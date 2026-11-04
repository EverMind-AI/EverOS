"""Refined-query generation prompts for ``aagentic_retrieve``: a system message of rules and a user message of data.

The SYSTEM message holds the rules: the role sentence, the task line, ``<instructions>``, ``<examples>``, and
``<reply_format>`` (one sentence plus the JSON skeleton). It names no request data, so every call sends the same text;
a caller's ``custom_instructions`` follows it as a closing ``<custom_instructions>`` block. The USER message carries
only the data, each input in its own tag, and is always rendered by the operator.

Every rule line comes from the single-message prompt the split replaced, unchanged apart from the structure: the data
moved to the user message (the "(insufficient)" note on the documents became the ``status`` attribute, and the copy
of the query and the missing information repeated at the end was dropped), the bold labels became tags, ``**`` was
dropped, and the closing "Refined Query:" cue went with that copy. The reply was free text; the contract has no
free-text tier, so the old "output only the refined query, no additional
text" line gave way to ``<reply_format>``, which asks for the refined query, and nothing else, in ``text``. The
worked examples are kept as written.

Placeholders, user template only: ``{{original_query}}``, ``{{retrieved_docs}}`` (the documents as ``_format_docs``
renders them) and ``{{missing_info}}``.

Reply shape: ``{"text": str}``, held to ``rank._replies._RefinedQueryReply``.
"""

from __future__ import annotations

__all__ = ["REFINED_QUERY_PROMPT_EN", "REFINED_QUERY_USER_PROMPT"]

from everos.algo.utils import paragraphs, tagged

_ROLE = "You are an expert at query reformulation for information retrieval."

_TASK = "Task: Generate a refined query that targets the missing information in the retrieved results."

_INSTRUCTIONS = """\
1. Keep the core intent of the original query unchanged.
2. Add specific keywords or rephrase to target the missing information.
3. Make the query more specific and focused.
4. The refined query should be a direct question that seeks to extract the missing facts.
5. Do NOT change the query's meaning or make it too broad.
6. Keep it concise (1-2 sentences maximum)."""

_EXAMPLES = """\
Example 1:
Original Query: "What happened at the conference?"
Missing Info: ["specific sessions attended", "key takeaways"]
Refined Query: "Which sessions were attended at the conference and what were the main takeaways?"

Example 2:
Original Query: "Tell me about the deadline"
Missing Info: ["deadline date", "deliverables", "responsible team"]
Refined Query: "When is the deadline, what needs to be delivered, and who is responsible?"

Example 3:
Original Query: "The new policy"
Missing Info: ["policy details", "effective date", "affected departments"]
Refined Query: "What are the details of the new policy, when does it take effect, and which departments are affected?\""""

_REPLY_FORMAT = """\
Reply with one JSON object, nothing else. text is the refined query only, with no additional text.
{"text": "..."}"""

REFINED_QUERY_PROMPT_EN = paragraphs(
    _ROLE,
    _TASK,
    tagged("instructions", _INSTRUCTIONS),
    tagged("examples", _EXAMPLES),
    tagged("reply_format", _REPLY_FORMAT),
)

REFINED_QUERY_USER_PROMPT = paragraphs(
    tagged("original_query", "{{original_query}}"),
    tagged("retrieved_documents", "{{retrieved_docs}}", status="insufficient"),
    tagged("missing_information", "{{missing_info}}"),
)

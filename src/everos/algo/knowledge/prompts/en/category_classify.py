"""English prompts for ``aclassify_category`` document classification.

A single LLM call picks one category from a caller-supplied taxonomy (``list[CategorySpec]``).

The SYSTEM message holds the role sentence, ``<task>``, ``<reply_format>`` (one sentence plus the JSON skeleton),
then ``<taxonomy>`` and, last, the caller's ``<custom_instructions>`` when given: the static rules first, then the
taxonomy, which changes per knowledge base but stays the same across the documents of one ingest run, so providers
that cache prompt prefixes hit the cache across those documents.
The USER message carries the per-document data: ``<document_title>`` and ``<document_summary>``.

Every rule sentence is the old prompt's, unchanged, save one: "the taxonomy above" now names ``<taxonomy>``, since the
taxonomy moved after the rules. Otherwise the Markdown headings became tags, inline backticks were dropped, the data
labels gave way to the data tags, and the "Output strictly this JSON shape" line became the ``<reply_format>``
sentence.

Constants:
    - ``CATEGORY_CLASSIFY_SYSTEM_PROMPT`` — the static rules. No placeholder; :func:`assemble_system_prompt` appends
      the taxonomy and the caller's instructions.
    - ``CATEGORY_CLASSIFY_USER_PROMPT`` — the data. Placeholders: ``{{title}}`` and ``{{doc_summary}}``.

Output schema: ``knowledge._replies.CategoryReply``, ``{"category_id": str}``; the caller validates the id against the
closed taxonomy and falls back to ``""`` for an out-of-set answer.
"""

from __future__ import annotations

__all__ = [
    "CATEGORY_CLASSIFY_SYSTEM_PROMPT",
    "CATEGORY_CLASSIFY_USER_PROMPT",
    "assemble_system_prompt",
]

from everos.algo.utils import append_custom_instructions, paragraphs, tagged

_ROLE = """\
You are a document classifier. Pick exactly ONE category id from the taxonomy below."""

_TASK = """\
Return a JSON object with a single key category_id whose value is the id of the best
matching category from <taxonomy>. If none of the categories fit, return the empty
string. Do NOT invent ids that are not in the taxonomy."""

_REPLY_FORMAT = """\
Reply with one JSON object, nothing else.
{"category_id": "<one of the ids listed in the taxonomy, or empty string>"}"""

CATEGORY_CLASSIFY_SYSTEM_PROMPT = paragraphs(
    _ROLE,
    tagged("task", _TASK),
    tagged("reply_format", _REPLY_FORMAT),
)

CATEGORY_CLASSIFY_USER_PROMPT = paragraphs(
    tagged("document_title", "{{title}}"),
    tagged("document_summary", "{{doc_summary}}"),
)


def assemble_system_prompt(*, taxonomy: str, custom_instructions: str | None) -> str:
    """Build the system message: the static rules, then the taxonomy, then the caller's instructions if any.

    Both values are placed verbatim and are never scanned for placeholders, so braces in them stay literal.

    Args:
        taxonomy: The ``<taxonomy>`` text, one ``- id: description`` line per category.
        custom_instructions: The caller's instructions for the ``<custom_instructions>`` block, the message's last;
            None or a blank string leaves the block out.

    Returns:
        The system message.
    """
    rules = paragraphs(CATEGORY_CLASSIFY_SYSTEM_PROMPT, tagged("taxonomy", taxonomy))
    return append_custom_instructions(rules, custom_instructions)

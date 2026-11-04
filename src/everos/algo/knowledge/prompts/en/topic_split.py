"""English prompts for splitting unsplit leaf topics in post-processing.

Some leaf topics produced by the initial extraction may still contain multiple sub-headings that warrant splitting into
a parent / children hierarchy. This pair asks the LLM to decide per topic whether to split and how to redistribute
``block_refs``.

The SYSTEM message holds only static rules: the role sentences, ``<instructions>`` and ``<reply_format>`` (one sentence,
the JSON skeleton, and the line on what a ``should_split=false`` result carries). The USER message carries the data:
``<document_title>`` and ``<topics count="N">``, each leaf with its blocks and the sub-headings found. Every rule
sentence is the old prompt's, unchanged, save the ``should_split=false`` line: every field of the strict reply model is
required, so such a result now writes ``block_refs`` and ``children`` as ``null`` rather than leaving them out, and
the skeleton's second result shows it. The skeleton was written with doubled braces that reached the model as is; it
is now valid JSON. Otherwise the data label gave way to the data tag, and the skeleton moved under one
``<reply_format>`` sentence that replaces both its old lead-in and the closing "Return only the JSON object".

Constants:
    - ``TOPIC_SPLIT_SYSTEM_PROMPT`` — the static rules. No placeholder.
    - ``TOPIC_SPLIT_USER_PROMPT`` — the data. Placeholders: ``{{title}}``, ``{{count}}`` and ``{{topics_section}}``.

Output schema: ``knowledge._replies.TopicSplitReply``.
"""

from __future__ import annotations

__all__ = ["TOPIC_SPLIT_SYSTEM_PROMPT", "TOPIC_SPLIT_USER_PROMPT"]

from everos.algo.utils import paragraphs, tagged

_ROLE = """\
You are a document topic segmentation expert. The following leaf topics were extracted from a document but each contains multiple sub-headings that may warrant splitting into a hierarchical parent-children structure.

For each topic below, decide whether it should be split. Then return the result."""

_INSTRUCTIONS = """\
- If a topic's sub-sections are substantive and independently searchable, split it into a parent with children.
- If some sub-sections are too short or tightly coupled, merge them with adjacent sections into fewer children.
- If the entire topic is cohesive and splitting would fragment the meaning, keep it as a single leaf.
- When splitting: the topic's own heading/intro blocks (before the first sub-heading) stay in the parent's block_refs. Each sub-heading and its following content becomes a child.
- Children can themselves have children if deeper sub-headings exist.
- The parent's topic name and summary are KEPT from the original extraction — do NOT regenerate them.
- Each child's summary should describe the specific information contained in that child's blocks. Include key entities, facts, and terms that someone might search for. Write "X is/does Y" not "this section discusses X".
- All summaries MUST be in the same language as the document content."""

_REPLY_FORMAT = """\
Reply with one JSON object, nothing else.
{
    "results": [
        {
            "original_topic": "The original topic name",
            "should_split": true,
            "reason": "Brief explanation",
            "block_refs": "Parent's own intro block IDs only (compressed range, e.g. '102')",
            "children": [
                {
                    "topic": "Child topic name. In document language.",
                    "summary": "Retrieval index for this child. Include key entities, facts, names, numbers, and searchable terms. Max 5 sentences. In document language.",
                    "block_refs": "Child's block IDs (compressed range, e.g. '103-107')",
                    "children": []
                }
            ]
        },
        {
            "original_topic": "Another topic",
            "should_split": false,
            "reason": "Brief explanation",
            "block_refs": null,
            "children": null
        }
    ]
}

For topics with should_split=false, only original_topic, should_split, and reason carry content; write block_refs and children as null."""

TOPIC_SPLIT_SYSTEM_PROMPT = paragraphs(
    _ROLE,
    tagged("instructions", _INSTRUCTIONS),
    tagged("reply_format", _REPLY_FORMAT),
)

TOPIC_SPLIT_USER_PROMPT = paragraphs(
    tagged("document_title", "{{title}}"),
    tagged("topics", "{{topics_section}}", count="{{count}}"),
)

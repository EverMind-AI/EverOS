"""English prompts for merging duplicate topics across batch boundaries.

When a long document is segmented in multiple batches, the same conceptual topic can show up as separate entries in
different batches. This pair asks the LLM to identify *true* duplicates (same specific subject, not merely related
themes) and produce a merged topic name + summary.

The SYSTEM message holds only static rules: the role sentences, ``<instructions>`` and ``<reply_format>`` (one
sentence plus the JSON skeleton). The USER message carries the data: ``<document_title>`` and ``<all_topics>``, the
indexed topics as a JSON array. Every rule sentence is the old prompt's, unchanged; ``**`` emphasis and the code fence
around the topics were dropped, the data labels gave way to the data tags, and the skeleton moved under one
``<reply_format>`` sentence that replaces both its old lead-in and the closing "Return only the JSON object".

Constants:
    - ``TOPIC_MERGE_SYSTEM_PROMPT`` — the static rules. No placeholder.
    - ``TOPIC_MERGE_USER_PROMPT`` — the data. Placeholders: ``{{title}}`` and ``{{all_topics_json}}``.

Output schema: ``knowledge._replies.TopicMergeReply``.
"""

from __future__ import annotations

__all__ = ["TOPIC_MERGE_SYSTEM_PROMPT", "TOPIC_MERGE_USER_PROMPT"]

from everos.algo.utils import paragraphs, tagged

_ROLE = """\
You are an expert document analyzer. A long document was analyzed in multiple sections independently, producing separate topic lists. Some topics may be true duplicates — different sections describing the exact same specific event or subject — and should be merged."""

_INSTRUCTIONS = """\
1. Scan topics by name, summary, and parent_topic context to find true duplicates only: topics that describe the exact same specific event, scene, or subject matter with different wording (e.g. "Elizabeth's Death" and "Elizabeth's Murder" both describing the same murder scene, or two copyright notices at the start and end of a document). Topics with the same parent_topic are more likely to be duplicates if their summaries are semantically similar.
2. Do NOT merge topics just because they share a broad theme, involve the same characters, or are temporally adjacent. Different events are different topics.
3. For each group of duplicates, provide:
   - "indices": the index numbers of the topics to merge
   - "topic": the best topic name (max 20 characters, use the SAME LANGUAGE as the input)
   - "summary": a coherent 1-2 sentence summary covering all merged topics (use the SAME LANGUAGE as the input)
4. If no duplicates are found, return an empty merges list."""

_REPLY_FORMAT = """\
Reply with one JSON object, nothing else.
{
    "merges": [
        {
            "indices": [0, 33],
            "topic": "Merged topic name",
            "summary": "Merged summary"
        }
    ]
}"""

TOPIC_MERGE_SYSTEM_PROMPT = paragraphs(
    _ROLE,
    tagged("instructions", _INSTRUCTIONS),
    tagged("reply_format", _REPLY_FORMAT),
)

TOPIC_MERGE_USER_PROMPT = paragraphs(
    tagged("document_title", "{{title}}"),
    tagged("all_topics", "{{all_topics_json}}"),
)

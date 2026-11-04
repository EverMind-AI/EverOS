"""English prompts for assigning uncovered blocks back to existing topics.

After topic extraction + post-processing, some paragraph IDs may remain unassigned. This pair asks the LLM to attach
each orphan block to the most appropriate existing topic by ``topic_index``.

The SYSTEM message holds only static rules: the role sentences, ``<instructions>`` and ``<reply_format>`` (one sentence
plus the JSON skeleton). The USER message carries the data: ``<document_title>``, ``<existing_topics count="N">`` (one
``index: name (blocks a-b)`` line each) and ``<uncovered_blocks count="M">``. Every rule sentence is the old prompt's,
unchanged, save one: "the list above" now names ``<existing_topics>``, since the list moved below the rules into the
user message. The skeleton was written with doubled braces that reached the model as is; it is now valid JSON.
Otherwise the data labels gave way to the data tags, and the skeleton moved under one ``<reply_format>`` sentence that
replaces both its old lead-in and the closing "Return only the JSON object".

Constants:
    - ``UNCOVERED_ASSIGN_SYSTEM_PROMPT`` — the static rules. No placeholder.
    - ``UNCOVERED_ASSIGN_USER_PROMPT`` — the data. Placeholders: ``{{title}}``, ``{{topic_count}}``,
      ``{{topics_text}}``, ``{{block_count}}`` and ``{{uncovered_text}}``.

Output schema: ``knowledge._replies.UncoveredAssignReply``.
"""

from __future__ import annotations

__all__ = ["UNCOVERED_ASSIGN_SYSTEM_PROMPT", "UNCOVERED_ASSIGN_USER_PROMPT"]

from everos.algo.utils import paragraphs, tagged

_ROLE = """\
You are a document topic segmentation expert. The following blocks were not assigned to any topic during extraction. Assign each block to the most appropriate existing topic."""

_INSTRUCTIONS = """\
- Assign each block to the topic it most naturally belongs to, based on content proximity and semantic relevance.
- Use the topic index (0-based) from <existing_topics>.
- If a block truly does not fit any topic, you may assign it to the nearest preceding topic."""

_REPLY_FORMAT = """\
Reply with one JSON object, nothing else.
{
    "assignments": [
        {
            "block_id": 10,
            "topic_index": 0,
            "reason": "Brief explanation"
        }
    ]
}"""

UNCOVERED_ASSIGN_SYSTEM_PROMPT = paragraphs(
    _ROLE,
    tagged("instructions", _INSTRUCTIONS),
    tagged("reply_format", _REPLY_FORMAT),
)

UNCOVERED_ASSIGN_USER_PROMPT = paragraphs(
    tagged("document_title", "{{title}}"),
    tagged("existing_topics", "{{topics_text}}", count="{{topic_count}}"),
    tagged("uncovered_blocks", "{{uncovered_text}}", count="{{block_count}}"),
)

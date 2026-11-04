"""English prompts for ``KnowledgeExtractor.aextract`` topic-tree segmentation: a system message of rules, a user message of data.

One call per batch produces both the topic tree (hierarchy and ``block_refs`` per topic) and the document-level
summary / subject / keywords / content_labels that populate the document root node (``KnowledgeMemory[0]``).

The SYSTEM message holds only static rules: the role sentences, then ``<topic_segmentation_requirements>``,
``<overall_analysis_requirements>``, ``<content_labels>``, ``<critical_language_rule>`` and ``<reply_format>`` (one
sentence plus the JSON skeleton). It names no request data, so every call shares it as a cacheable prefix. The USER
message carries the data: ``<document_title>`` and ``<paragraphs count="N" max_id="M">``, one ``ID: content`` line per
atom block.

Every rule sentence comes from the single-message prompt this pair replaces, unchanged, save two: the coverage rule
that named the ``{{max_id}}`` slot now names ``<paragraphs>``'s ``max_id``, and the line that let leaf topics omit
``children`` now requires ``[]``, because every field of the strict reply model is required. Otherwise only the
structure changed: the top-level headings became tags, ``**`` emphasis was dropped, the data labels gave way to the
data tags, and the skeleton moved under one ``<reply_format>`` sentence that replaces both the old lead-in and the
closing "Return only the JSON object" line.

Constants:
    - ``TOPIC_TREE_EXTRACT_SYSTEM_PROMPT`` — the static rules. No placeholder.
    - ``TOPIC_TREE_EXTRACT_USER_PROMPT`` — the data. Placeholders: ``{{title}}``, ``{{count}}``, ``{{max_id}}`` and
      ``{{numbered_paragraphs}}``.

Output schema: ``knowledge._replies.TopicTreeReply``, in the skeleton's field order.
"""

from __future__ import annotations

__all__ = ["TOPIC_TREE_EXTRACT_SYSTEM_PROMPT", "TOPIC_TREE_EXTRACT_USER_PROMPT"]

from everos.algo.utils import paragraphs, tagged

_ROLE = """\
You are an expert document analyzer. Your task is to perform TWO analyses on the following document:
1. Topic Segmentation: Segment the document into topic-based sections
2. Overall Analysis: Extract summary, subject, and keywords"""

_TOPIC_SEGMENTATION_REQUIREMENTS = """\
Hierarchy:
- Segment by semantic boundaries — each topic should cover one coherent subject. Use headings as a guide, but also split within a section if it covers multiple distinct subjects (e.g., a section covering both "deployment topology" and "disaster recovery" should become two topics even without sub-headings).
- If the document has markdown headings (##, ###, ####, etc.), use them as the primary structural guide. Map heading levels to nesting levels (## -> parent, ### -> child, #### -> grandchild). Each sub-heading becomes its own child topic.
- If a section has no sub-headings but covers multiple distinct subjects, split it into sibling topics at the semantic boundaries.
- If the document has no headings at all, use your judgment to identify natural topic boundaries and decide whether nesting is appropriate.
- If headings are malformed (skipped levels, inconsistent numbering, misleading titles), use the content to determine the correct hierarchy rather than blindly following the heading markup.
- Boundary blocks (transitional paragraphs between topics) may appear in both adjacent topics.
- Parent block_refs contain ONLY its own intro blocks (before the first child begins). Child headings belong in the child's block_refs, not the parent's.
- children key is required — use [] for leaf topics.
- Every summary serves as a retrieval index — someone reading only the summary should be able to decide whether this section answers their query. Include all key entities, facts, names, numbers, and terms that someone might search for. Be concise but do not omit searchable information. Max 5 sentences.
- A parent's summary covers the ENTIRE section including all children's content.
- A leaf's summary covers only its own paragraphs.

Coverage — every block must be assigned:
- Every paragraph ID (0 to the max_id of <paragraphs>) MUST appear in at least one topic's block_refs. Document metadata (title, version, date, author) goes in the most relevant topic. Only pure separators (e.g., "---") may be omitted.
- Overlap is allowed at boundaries — a transitional paragraph may appear in two adjacent topics (including parent and child).
- Transitional content (questions, acknowledgments, topic shifts) must be assigned, not left out.
- Use compressed range notation: "1-3,5,8-10" means [1,2,3,5,8,9,10]. Topics do not need to be contiguous.

Naming and content:
- Topic names should be specific (e.g., "Introduction to Quantum Tunneling" not just "Introduction").
- A heading introduces the content that FOLLOWS it — assign it to the same topic as the body below.
- Tables, lists, and image descriptions are regular content — group with the nearest related topic.
- Trailing attribution (bylines, credits) belongs to the preceding section's topic."""

_OVERALL_ANALYSIS_REQUIREMENTS = """\
- Summary: Should help someone quickly understand what this document is about and why it might be relevant. Include main topics, key decisions, important information, and overall purpose. Be concise but comprehensive.
- Subject: A clear, descriptive subject that captures the main theme/topic of the document. This should be better than the original title if the title is unclear or generic.
- Keywords: Extract 3-5 most important keywords that represent the core concepts, topics, or themes in the document. The extracted keywords must be in the original document."""

_CONTENT_LABELS = """\
Content Labels (apply to both document-level and each topic):
- Detect and tag content with applicable labels from: "violence", "adult", "hate_speech", "personal_info", "financial", "medical"
- "personal_info": contains non-public PII that could directly identify a private individual, such as personal phone numbers, government ID numbers (SSN, passport), personal email addresses, or home addresses. Do NOT apply for publicly disclosed information like corporate officer names, company addresses, or information in public filings/reports.
- "financial": contains financial data such as bank account numbers, credit card numbers, transaction amounts
- "medical": contains medical/health information
- Return an empty list if no labels apply. Only include labels that clearly match."""

_CRITICAL_LANGUAGE_RULE = """\
- First, identify the language of the document content and write it in the "language" field
- ALL subsequent outputs (summary, subject, keywords, topic names, topic summaries) MUST be in that detected language
- Do NOT translate the document content into another language under any circumstances"""

_REPLY_FORMAT = """\
Reply with one JSON object, nothing else.
{
    "language": "Detected language of the document (e.g., 'English', 'Chinese', 'Japanese'). Identify FIRST — all subsequent fields MUST be written in this language.",
    "summary": "A comprehensive but concise summary (2-3 sentences). MUST be in the detected language.",
    "subject": "A clear, descriptive subject/topic. MUST be in the detected language.",
    "keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"],
    "content_labels": ["label1", "label2"],
    "topics": [
        {
            "topic": "Topic name (concise, max 20 chars). MUST be in the detected language.",
            "summary": "Retrieval index for this ENTIRE section (including children). Include key entities, facts, names, numbers, and searchable terms. Max 5 sentences. MUST be in the detected language.",
            "block_refs": "Intro paragraph IDs only (e.g., '0-2')",
            "content_labels": ["label1"],
            "children": [
                {
                    "topic": "Sub-topic name. MUST be in the detected language.",
                    "summary": "Retrieval index for this sub-section. Include key entities, facts, names, numbers, and searchable terms. Max 5 sentences. MUST be in the detected language.",
                    "block_refs": "This sub-topic's paragraph IDs",
                    "content_labels": [],
                    "children": [
                        {
                            "topic": "Sub-sub-topic name (nesting can go deeper if document headings do)",
                            "summary": "...",
                            "block_refs": "...",
                            "content_labels": [],
                            "children": []
                        }
                    ]
                }
            ]
        }
    ]
}"""

TOPIC_TREE_EXTRACT_SYSTEM_PROMPT = paragraphs(
    _ROLE,
    tagged("topic_segmentation_requirements", _TOPIC_SEGMENTATION_REQUIREMENTS),
    tagged("overall_analysis_requirements", _OVERALL_ANALYSIS_REQUIREMENTS),
    tagged("content_labels", _CONTENT_LABELS),
    tagged("critical_language_rule", _CRITICAL_LANGUAGE_RULE),
    tagged("reply_format", _REPLY_FORMAT),
)

TOPIC_TREE_EXTRACT_USER_PROMPT = paragraphs(
    tagged("document_title", "{{title}}"),
    tagged(
        "paragraphs", "{{numbered_paragraphs}}", count="{{count}}", max_id="{{max_id}}"
    ),
)

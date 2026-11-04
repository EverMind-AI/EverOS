"""English prompts for merging document-level summary / subject / keywords across batches.

When a document is too long for a single LLM call, ``KnowledgeExtractor`` runs the topic-tree prompt over multiple
batches and produces partial document-level fields. This pair consolidates them into a single coherent set.

The SYSTEM message holds only static rules: the role sentence, ``<requirements>``, ``<critical_language_rule>`` and
``<reply_format>`` (one sentence plus the JSON skeleton). The USER message carries the data: ``<document_title>`` and
``<partial_analysis_results count="N">``. Every rule sentence is the old prompt's, unchanged; ``**`` emphasis and the
code fence around the partial results were dropped, the data labels gave way to the data tags, and the skeleton moved
under one ``<reply_format>`` sentence that replaces both its old lead-in and the closing "Return only the JSON object".

Constants:
    - ``CONTENT_MERGE_SYSTEM_PROMPT`` — the static rules. No placeholder.
    - ``CONTENT_MERGE_USER_PROMPT`` — the data. Placeholders: ``{{title}}``, ``{{count}}`` and
      ``{{partial_results}}``.

Output schema: ``knowledge._replies.ContentMergeReply``, in the skeleton's field order.
"""

from __future__ import annotations

__all__ = ["CONTENT_MERGE_SYSTEM_PROMPT", "CONTENT_MERGE_USER_PROMPT"]

from everos.algo.utils import paragraphs, tagged

_ROLE = """\
You are an expert document analyzer. Please merge multiple partial analysis results from different sections of a long document into one comprehensive analysis."""

_REQUIREMENTS = """\
1. Summary: Integrate all key information from partial summaries, remove redundant information while preserving important details, maintain logical flow and coherence
2. Subject: Choose or synthesize the most representative subject that captures the main theme of the entire document
3. Keywords: Merge and deduplicate keywords from all sections, keeping only the 3-5 most important ones that represent the entire document. The extracted keywords must be in the original document."""

_CRITICAL_LANGUAGE_RULE = """\
First identify the language in the "language" field. ALL outputs (summary, subject, keywords) MUST be in that language. Do NOT translate."""

_REPLY_FORMAT = """\
Reply with one JSON object, nothing else.
{
    "language": "The language used in the partial results (e.g., 'English', 'Chinese'). Identify FIRST — all subsequent fields MUST be written in this language.",
    "summary": "A comprehensive summary (3-5 sentences). MUST be in the detected language.",
    "subject": "A unified subject/topic. MUST be in the detected language.",
    "keywords": ["merged list of most important keywords (3-5 total). MUST be in the detected language."]
}"""

CONTENT_MERGE_SYSTEM_PROMPT = paragraphs(
    _ROLE,
    tagged("requirements", _REQUIREMENTS),
    tagged("critical_language_rule", _CRITICAL_LANGUAGE_RULE),
    tagged("reply_format", _REPLY_FORMAT),
)

CONTENT_MERGE_USER_PROMPT = paragraphs(
    tagged("document_title", "{{title}}"),
    tagged("partial_analysis_results", "{{partial_results}}", count="{{count}}"),
)

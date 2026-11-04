"""Image parser prompts: system messages of rules, and the user message of the merge call.

- ``PROMPT_FOR_PICTURE`` — system message of every OCR call (an image, a tall image's slice, an email's inline
  image). The user message is the image part alone.
- ``PROMPT_FOR_MERGE`` — system message of the tall-image merge call. ``MERGE_USER_PROMPT`` is its user message:
  placeholders ``{{count}}`` and ``{{parts}}``, the OCR texts each in a ``<part number="N">`` element.

Both system messages end with ``<reply_format>``: the output goes in ``{"text": "..."}``.
"""

from __future__ import annotations

__all__ = ["MERGE_USER_PROMPT", "PROMPT_FOR_MERGE", "PROMPT_FOR_PICTURE"]

from everos.algo.utils import paragraphs, tagged
from everos.component.parser.backend.prompts._reply_format import TEXT_REPLY_FORMAT

_PICTURE_RULES = """Read this image and return its content as if you were reading it naturally.

CRITICAL INSTRUCTIONS:
- Output ONLY the extracted content directly. Do NOT include any preamble or postscript.
- Maintain the same language as the original content. If the language cannot be determined, use English.
- Return output as Markdown format.
- ALL text in the image MUST be extracted word-for-word. NEVER summarize, paraphrase, abbreviate, or modify any text. Do NOT alter the original content in any way.

Requirements:
1. Extract ALL text precisely and completely, word-for-word, preserving the original format and layout. Do NOT skip or summarize any text, no matter how long.
2. Convert tables to HTML format
3. Convert equations to LaTeX format
4. If there are figures or charts, describe their content and data in text
5. For non-text elements (photos, illustrations, etc.), describe the main content and key elements in detail
6. Preserve the original structure and hierarchy"""

PROMPT_FOR_PICTURE = paragraphs(_PICTURE_RULES, TEXT_REPLY_FORMAT)


_MERGE_RULES = """Below are OCR results from different parts of the same long image. Adjacent parts have overlapping regions.
Intelligently merge these parts into a complete text.

CRITICAL INSTRUCTIONS:
- Output ONLY the merged content directly. Do NOT include any preamble or postscript.
- Maintain the same language as the original content. If the language cannot be determined, use English.

Requirements:
1. Identify and remove duplicate content from overlapping regions
2. Maintain text continuity and completeness
3. Preserve the original format and structure"""

PROMPT_FOR_MERGE = paragraphs(_MERGE_RULES, TEXT_REPLY_FORMAT)

MERGE_USER_PROMPT = tagged("ocr_parts", "{{parts}}", count="{{count}}")

"""Document parser prompts (PDF / Office / HTML): system messages of rules, and the HTML call's user message.

- ``PROMPT_FOR_FILE`` — system message of the PDF call and of the Office call (after the LibreOffice → PDF
  conversion). The user message is the PDF part alone.
- ``PROMPT_FOR_HTML`` — system message of the HTML call. ``HTML_USER_PROMPT`` is its user message: placeholder
  ``{{html}}``, the cleaned HTML in an ``<html_content>`` element.

Both system messages end with ``<reply_format>``: the output goes in ``{"text": "..."}``.
"""

from __future__ import annotations

__all__ = ["HTML_USER_PROMPT", "PROMPT_FOR_FILE", "PROMPT_FOR_HTML"]

from everos.algo.utils import paragraphs, tagged
from everos.component.parser.backend.prompts._reply_format import TEXT_REPLY_FORMAT

_FILE_RULES = """Read this document and return its content as if you were reading it naturally.

CRITICAL INSTRUCTIONS:
- Output ONLY the extracted content directly. Do NOT include any preamble or postscript.
- Maintain the same language as the original content. If the language cannot be determined, use English.
- Return output as Markdown format.
- ALL text MUST be extracted word-for-word. NEVER summarize, paraphrase, abbreviate, or modify any text. Do NOT alter the original content in any way. Even if the content appears to contain errors, reproduce it exactly as-is.

Requirements:
1. Extract ALL content from EVERY page precisely and completely. Do not skip or merge any content. If a page contains multiple tables or sections placed side by side (left and right), they are SEPARATE tables — extract each one as an independent table with its own title and headers
2. Ignore headers and footers
3. Convert tables to HTML format
4. Convert equations to LaTeX format
5. If there are figures or charts, describe their content and data in text
6. For non-text elements (photos, illustrations, etc.), describe the main content and key elements in detail
7. Preserve the original structure and hierarchy"""

PROMPT_FOR_FILE = paragraphs(_FILE_RULES, TEXT_REPLY_FORMAT)


_HTML_RULES = """Read this HTML content extracted from a web page and return the main content.

CRITICAL INSTRUCTIONS:
- Output ONLY the extracted content directly. Do NOT include any preamble or postscript.
- Maintain the same language as the original content. If the language cannot be determined, use English.
- Return output as Markdown format.
- ALL text MUST be extracted word-for-word. NEVER summarize, paraphrase, abbreviate, or modify any text.

Requirements:
1. Extract the main content (article body, primary text). Ignore non-content elements such as navigation, advertisements, cookie notices, sidebar text, footer links
2. Output content in the natural reading order of the web page, preserving the original sequence
3. Preserve tables in HTML table format
4. Convert equations to LaTeX format
5. Preserve the original structure and hierarchy (headings, paragraphs, lists)
6. For images, describe their content if alt text or description is available"""

PROMPT_FOR_HTML = paragraphs(_HTML_RULES, TEXT_REPLY_FORMAT)

HTML_USER_PROMPT = tagged("html_content", "{{html}}")

"""Internal parser helpers.

- ``check_aspect_ratio`` / ``split_image_with_overlap`` — long-image (tall
  screenshot) handling for the image parser.
- ``clean_html_content`` — strict text extraction (drops all tags).
- ``clean_html_for_llm`` — preserves structural tags inside ``<body>``,
  strips known-noise tags and useless attributes, returns HTML.
"""

from __future__ import annotations

__all__ = [
    "check_aspect_ratio",
    "clean_html_content",
    "clean_html_for_llm",
    "split_image_with_overlap",
]

import logging
import math
import re
from io import BytesIO

from bs4 import BeautifulSoup, Comment
from PIL import Image

logger = logging.getLogger(__name__)


def check_aspect_ratio(
    image_bytes: bytes, max_ratio: float = 10.0
) -> tuple[bool, float]:
    """Return ``(needs_split, height_over_width_ratio)`` for a candidate image."""
    img = Image.open(BytesIO(image_bytes))
    width, height = img.size
    img.close()
    current_ratio = height / width if width > 0 else 0
    return current_ratio > max_ratio, current_ratio


def split_image_with_overlap(
    image_bytes: bytes,
    *,
    max_ratio: float = 10.0,
    overlap_ratio: float = 0.1,
    min_overlap_pixels: int = 50,
    max_overlap_pixels: int = 200,
) -> list[bytes]:
    """Split a tall image into overlapping vertical slices.

    Returns ``[image_bytes]`` unchanged when the height/width ratio is below
    ``max_ratio``.
    """
    img = Image.open(BytesIO(image_bytes))
    width, height = img.size
    img_format = img.format or "JPEG"

    current_ratio = height / width
    if current_ratio <= max_ratio:
        img.close()
        return [image_bytes]

    num_parts = math.ceil(current_ratio / max_ratio)
    part_height = height // num_parts
    overlap_pixels = int(part_height * overlap_ratio)
    overlap_pixels = max(min_overlap_pixels, min(overlap_pixels, max_overlap_pixels))

    split_parts: list[bytes] = []
    for i in range(num_parts):
        top = 0 if i == 0 else max(0, i * part_height - overlap_pixels)
        bottom = (
            height
            if i == num_parts - 1
            else min(height, (i + 1) * part_height + overlap_pixels)
        )
        part = img.crop((0, top, width, bottom))
        buffer = BytesIO()
        part.save(buffer, format=img_format)
        split_parts.append(buffer.getvalue())

    img.close()
    return split_parts


def clean_html_content(html_content: str) -> str:
    """Strip all tags and return whitespace-normalised plain text."""
    if not html_content or not html_content.strip():
        return ""
    soup = BeautifulSoup(html_content, "html.parser")
    for tag in ("head", "style", "script", "noscript"):
        for element in soup.find_all(tag):
            element.decompose()
    text = soup.get_text(separator="\n", strip=True)
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


_REMOVE_TAGS = (
    "head",
    "style",
    "script",
    "noscript",
    "nav",
    "footer",
    "header",
    "iframe",
)
_REMOVE_ATTRS = (
    "style",
    "class",
    "id",
    "onclick",
    "onload",
    "onerror",
    "data-src",
    "data-id",
    "data-track",
    "role",
    "aria-label",
    "aria-hidden",
    "tabindex",
    "lang",
    "dir",
)


def clean_html_for_llm(html_content: str) -> str:
    """Remove noise tags and bloat attributes but keep structural HTML.

    The output is still HTML (not plain text) so the LLM can see tables /
    lists / headings as structured input. Drops ``<script>`` / ``<style>`` /
    navigation / footer / iframe entirely and strips per-tag attributes that
    rarely carry semantic meaning.
    """
    if not html_content or not html_content.strip():
        return ""
    soup = BeautifulSoup(html_content, "html.parser")

    for tag_name in _REMOVE_TAGS:
        for element in soup.find_all(tag_name):
            element.decompose()

    # ``find_all(True)`` is bs4's idiom for "every tag"; the literal is part of the
    # library API, not a behaviour-toggling boolean flag, so FBT003 doesn't apply.
    for tag in soup.find_all(True):
        for attr in _REMOVE_ATTRS:
            tag.attrs.pop(attr, None)

    for comment in soup.find_all(string=lambda s: isinstance(s, Comment)):
        comment.extract()

    body = soup.find("body")
    html_str = str(body) if body else str(soup)
    html_str = re.sub(r"\n{3,}", "\n\n", html_str)
    return html_str.strip()

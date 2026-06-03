"""Multimodal content normalisation.

Coerces raw message ``content`` into a list of ContentItem dicts and derives
the flat ``text`` the LLM-facing extractors + md writer consume. Non-text
items contribute their ``parsed_content`` (populated by the parser hook,
``memory/extract/parser``) rendered as a provenance tag
``[TYPE: name]\\n{parsed_content}``; non-text items still lacking
``parsed_content`` are counted (a warning is logged) so callers can surface
them via ``IngestResult.unparsed_non_text_count``.
"""

from __future__ import annotations

from typing import Any

from everos.core.observability.logging import get_logger

logger = get_logger(__name__)


def coerce_items(
    content: str | list[dict[str, Any]] | list[Any],
) -> list[dict[str, Any]]:
    """Coerce ``content`` into a list of ContentItem dicts.

    Accepts the simplified ``str`` form (DTO sugar) or a list of
    ContentItem-shaped dicts / Pydantic models.
    """
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return [_coerce_item(item) for item in content]


def derive_text(items: list[dict[str, Any]]) -> tuple[str, int]:
    """Render items into the derived ``text`` + count still-unparsed non-text.

    Returns ``(text, non_text_count)``: ``text`` is the newline-joined render
    of all items; ``non_text_count`` is the number of non-text items that have
    no ``parsed_content`` yet.
    """
    parts: list[str] = []
    non_text = 0
    for item in items:
        rendered = _render_item(item)
        if rendered:
            parts.append(rendered)
        elif item.get("type") != "text":
            non_text += 1
            logger.warning(
                "multimodal_content_not_parsed",
                extra={"content_type": item.get("type", "unknown")},
            )
    return "\n".join(parts), non_text


def normalise_content(
    content: str | list[dict[str, Any]] | list[Any],
) -> tuple[list[dict[str, Any]], str, int]:
    """Coerce + derive in one step (text-only path; no parsing).

    Returns ``(content_items, text, non_text_count)``. The ingest service
    splits this into :func:`coerce_items` / :func:`derive_text` so the parser
    hook can run between them; this wrapper is kept for text-only callers.
    """
    items = coerce_items(content)
    text, non_text = derive_text(items)
    return items, text, non_text


def _render_item(item: dict[str, Any]) -> str | None:
    """Render one item to text, or ``None`` if it contributes nothing.

    Text items yield their ``text``; non-text items yield
    ``[TYPE: name]\\n{parsed_content}`` once parsed; unparsed non-text yields
    ``None``.
    """
    if item.get("type") == "text":
        text = item.get("text")
        return str(text) if text else None
    parsed = item.get("parsed_content")
    if not parsed:
        return None
    kind = str(item.get("type") or "file").upper()
    name = item.get("name") or ""
    tag = f"[{kind}: {name}]" if name else f"[{kind}]"
    return f"{tag}\n{parsed}"


def _coerce_item(item: Any) -> dict[str, Any]:
    """Coerce ``item`` (dict or pydantic model) into a plain dict."""
    if hasattr(item, "model_dump"):
        return dict(item.model_dump())
    if isinstance(item, dict):
        return dict(item)
    return {"type": "unknown", "raw": repr(item)}

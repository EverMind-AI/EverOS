"""Multimodal capability detection + guard.

This module must NOT import ``everalgo.parser`` at module level — its job is
to answer "is the extra installed?" before anything tries to use it. The
probe lives inside :func:`multimodal_available`.
"""

from __future__ import annotations

from typing import Any

from everos.core.errors import MultimodalNotEnabledError

_INSTALL_HINT = (
    "Multimodal input received but the parser extra is not installed. "
    "Install it with:  pip install 'everos[multimodal]'  "
    "(or  uv add 'everos[multimodal]')."
)


def has_unparsed_multimodal(items: list[dict[str, Any]]) -> bool:
    """True if any content item needs the parser to become text.

    A stored picture never does. It has already been accepted as a picture and
    is handed to the extraction model as an image part — that is the whole
    point of the picture path, and running it through a captioner first would
    replace the picture with a lossy prose summary in every extraction. A
    picture therefore counts as *handled* once it carries ``image_id``, even
    though it has no ``parsed_content``.

    The parser route is still there for a text-only extraction model: with
    ``[extraction] vision = false`` the ingest layer does not store pictures at
    all, so they arrive here unstored and are parsed exactly as before.

    A picture ingest *refused* (``status = "rejected"``: over a limit, not a
    picture, no store) is not parser input either: it was refused on purpose,
    and parsing it anyway cost a multimodal call whose text was dropped -- or,
    with no ``[multimodal]`` configured, failed the whole request.
    """
    return any(needs_parser(item) for item in items)


def needs_parser(item: dict[str, Any]) -> bool:
    """Whether one content item is still waiting for the parser."""
    return (
        item.get("type") != "text"
        and "parsed_content" not in item
        and "image_id" not in item
        and item.get("status") != "rejected"
    )


def multimodal_available() -> bool:
    """Whether the ``everalgo.parser`` extra is importable."""
    from everos.component.parser import parser_available  # Deferred: optional dep probe

    return parser_available()


def require_multimodal() -> None:
    """Guard: raise when multimodal input arrives without the extra installed.

    Raises:
        MultimodalNotEnabledError: When ``everalgo.parser`` cannot be imported.
    """
    if not multimodal_available():
        raise MultimodalNotEnabledError(_INSTALL_HINT)

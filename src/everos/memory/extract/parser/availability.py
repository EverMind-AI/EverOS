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
    """True if any content item is non-text and not yet parsed."""
    return any(
        item.get("type") != "text" and "parsed_content" not in item for item in items
    )


def multimodal_available() -> bool:
    """Whether the ``everalgo.parser`` extra is importable."""
    try:
        import everalgo.parser  # noqa: F401
    except ImportError:
        return False
    return True


def require_multimodal() -> None:
    """Guard: raise when multimodal input arrives without the extra installed.

    Raises:
        MultimodalNotEnabledError: When ``everalgo.parser`` cannot be imported.
    """
    if not multimodal_available():
        raise MultimodalNotEnabledError(_INSTALL_HINT)

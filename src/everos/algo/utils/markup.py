"""XML-style markup for prompt text: a block wrapped in its tag, and blocks joined into one message."""

from __future__ import annotations

__all__ = ["append_custom_instructions", "paragraphs", "tagged"]


def tagged(tag: str, body: str, /, **attributes: str) -> str:
    """Wrap a block in its XML-style tag, open and close tag each alone on its line.

    Args:
        tag: Descriptive snake_case tag name.
        body: Block text, placed verbatim between the tags.
        **attributes: Rendered on the open tag as ``name="value"``, in the order given, unescaped.

    Returns:
        The tagged block, e.g. ``<messages count="3">``, a newline, ``body``, a newline, ``</messages>``.
    """
    opening = " ".join(
        [tag, *(f'{name}="{value}"' for name, value in attributes.items())]
    )
    return f"<{opening}>\n{body}\n</{tag}>"


def paragraphs(*blocks: str) -> str:
    """Join blocks into one text, one blank line between them."""
    return "\n\n".join(blocks)


def append_custom_instructions(system: str, custom_instructions: str | None, /) -> str:
    """Close a system message with the caller's ``<custom_instructions>`` block, when there is one.

    The block adds to the rules, never replaces them, and is always the message's last: call this after every other
    block of ``system`` is in place. Its text is placed verbatim, neither stripped nor rendered as a template, so a
    ``{{...}}`` in it stays literal.

    Args:
        system: The system message so far, fully rendered.
        custom_instructions: The caller's instructions; None or a blank string leaves ``system`` unchanged.

    Returns:
        ``system`` itself, or ``system`` and the tagged block, one blank line apart.
    """
    if custom_instructions is None or not custom_instructions.strip():
        return system
    return paragraphs(system, tagged("custom_instructions", custom_instructions))

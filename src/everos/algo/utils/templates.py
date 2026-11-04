"""Prompt template rendering with ``{{name}}`` placeholders."""

from __future__ import annotations

__all__ = ["render_prompt"]

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

_PLACEHOLDER_RE = re.compile(r"\{\{([A-Za-z_][A-Za-z0-9_]*)\}\}")

# A placeholder written in the pre-migration syntax. The lookarounds keep the inner ``{name}`` of a
# well-formed ``{{name}}`` out of the match, so a valid template never trips this.
_LEGACY_PLACEHOLDER_RE = re.compile(r"(?<!\{)\{([A-Za-z_][A-Za-z0-9_]*)\}(?!\})")


def render_prompt(default: str, prompt: str | None, /, **fields: Any) -> str:
    """Render ``prompt`` with ``fields``; if ``prompt`` is None, use ``default``.

    Each ``{{key}}`` placeholder is replaced with ``str(value)``; every other character, single braces and JSON
    examples included, is copied verbatim. Double braces follow Mustache and the Anthropic Console, and leave the
    single braces of a JSON example literal with nothing to escape. Every placeholder the template names must be
    substituted: a missing field or a ``None`` value raises instead of leaking the literal placeholder into the
    prompt, and so does a leftover single-brace ``{key}``, the shape of a placeholder in the old syntax.

    Substitution is single-pass, so a value lands in the result verbatim and no other field can rewrite it. Values
    are user content: re-scanning them would let a user-typed ``{{pass_rule}}`` pull another field's text into a data
    block and forge the prompt's structure.

    Args:
        default: Module-level prompt constant shipped with the operator.
        prompt: Caller override; if ``None``, falls back to ``default``.
        **fields: Each key=value renders ``{{key}}`` → ``str(value)``. A field no template
            placeholder names is unused, not an error: callers render several templates from one
            field set.

    Returns:
        Rendered prompt string.

    Raises:
        ValueError: If the template still uses the single-brace placeholder syntax, or if it names a
            placeholder that ``fields`` has no value for, or whose value is ``None``.

    Examples:
        >>> render_prompt("Hello, {{name}}!", None, name="world")
        'Hello, world!'
        >>> render_prompt("Hello, {{name}}!", "Hi {{name}}", name="world")
        'Hi world'
        >>> render_prompt('Output JSON like {"key": "value"} for {{name}}', None, name="Alice")
        'Output JSON like {"key": "value"} for Alice'
    """
    template = prompt or default
    _reject_legacy_placeholders(template)
    _reject_unrenderable_placeholders(_PLACEHOLDER_RE.findall(template), fields)
    # Callback form, not a replacement string: a value containing ``\1`` or ``\g<0>`` must stay
    # verbatim rather than be read as a backreference.
    return _PLACEHOLDER_RE.sub(lambda match: str(fields[match.group(1)]), template)


def _reject_legacy_placeholders(template: str) -> None:
    """Raise if the template carries a single-brace ``{name}``, the pre-migration placeholder shape.

    Args:
        template: Prompt template about to be rendered.

    Raises:
        ValueError: If a single-brace identifier is present. Brace shapes that were never a
            placeholder name — ``{"key": ...}`` JSON shards, a lone ``{``, ``{ spaced }``,
            ``{123}`` — are ordinary literals and pass.
    """
    names = dict.fromkeys(_LEGACY_PLACEHOLDER_RE.findall(template))
    if not names:
        return
    detail = ", ".join(f"{{{name}}} (write {{{{{name}}}}})" for name in names)
    raise ValueError(
        f"prompt template uses the single-brace placeholder syntax, which is now literal text: {detail}"
    )


def _reject_unrenderable_placeholders(
    referenced: list[str], fields: Mapping[str, Any]
) -> None:
    """Raise unless every placeholder the template names has a substitutable value.

    Args:
        referenced: Placeholder names read off the template, in order of appearance.
        fields: Caller-supplied values keyed by placeholder name.

    Raises:
        ValueError: If a named placeholder has no field, or its field is ``None`` — ``str(None)``
            would write a literal ``None`` into the prompt. An empty string is a legitimate value
            and renders normally.
    """
    faults: dict[str, str] = {}
    for name in referenced:
        if name not in fields:
            faults[name] = "no value supplied"
        elif fields[name] is None:
            faults[name] = "value is None"
    if faults:
        detail = ", ".join(
            f"{{{{{name}}}}} ({reason})" for name, reason in faults.items()
        )
        raise ValueError(f"prompt template has unrenderable placeholders: {detail}")

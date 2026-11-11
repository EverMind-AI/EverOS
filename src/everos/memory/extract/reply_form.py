"""Reading a model's JSON reply leniently in form, never in meaning.

Replies requested in JSON mode (``response_format_mode = "json_object"``, the
only mode DeepSeek accepts) are JSON, but not always the JSON the reply model
describes. On the picture route (2026-11-11, DeepSeek V4.1 Flash, MemEye's
1,696 messages) 7 of 168 replies (4.2%) failed validation, and each failure
cost a second call that carried every picture again, although none of them
said anything a valid reply would not have:

- an unknown key written empty inside a picture's entry (``"profile": []``,
  ``"ocr_note": ""``);
- an array element the model never closed: ``"data": {...}`` was followed by
  ``, {`` or ``]`` where ``}}`` was due;
- a reply split in two objects: ``{"content": ..., "images": [...]}`` and then
  ``{"profile": []}``.

What this module accepts, and what it leaves to fail:

- :func:`drop_empty_unknown_keys`: a key the reply model does not define
  reads as absent when its value is empty (``""``, ``null``, ``[]``, ``{}``).
  An unknown key that carries anything still fails -- that is content in the
  wrong place, and dropping it would lose it.
- :func:`repair_json_form`: closes an array element left open, and joins a
  reply split into consecutive objects whose keys do not overlap. The only
  character ever added is the ``}`` the next token implies, and nothing is
  removed or reordered; any other syntax error (an unterminated string, two
  braces missing, an element of an object rather than of an array, a key
  given twice across the split) is left to fail.

Neither touches meaning: required fields stay required, unknown keys with a
value stay forbidden, and every value is still validated by the reply model.
"""

from __future__ import annotations

__all__ = ["drop_empty_unknown_keys", "is_empty", "repair_json_form"]

import json
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic import BaseModel

_LITERAL = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|true|false|null")


def is_empty(value: object) -> bool:
    """Whether ``value`` says nothing: ``None``, a blank string, ``[]`` or ``{}``."""
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, list | dict):
        return not value
    return False


def drop_empty_unknown_keys(model: type[BaseModel], value: object) -> object:
    """``value`` without the keys ``model`` does not define whose value is empty.

    For a ``model_validator(mode="before")``: anything but a mapping passes
    through untouched for the model to reject. A field is known by its name
    and by its alias.
    """
    if not isinstance(value, dict):
        return value
    known: set[str] = set()
    for name, field in model.model_fields.items():
        known.add(name)
        if field.alias:
            known.add(field.alias)
    return {
        key: item for key, item in value.items() if key in known or not is_empty(item)
    }


def repair_json_form(text: str) -> str | None:
    """``text`` with its form slips mended, or ``None`` when it has none it can mend.

    Two slips, both seen in JSON-mode replies (module docstring): an array
    element left open -- a ``}`` is inserted where the next token shows the
    element ended -- and a reply split into consecutive objects, joined into
    one when no key appears in two of them. Anything else that keeps the text
    from being one JSON value returns ``None``: the caller's validation error
    stands.
    """
    closed = _close_open_elements(text)
    if closed is None:
        return None
    decoder = json.JSONDecoder()
    values: list[object] = []
    at = 0
    while True:
        at = _skip_space(closed, at)
        if at == len(closed):
            break
        try:
            value, at = decoder.raw_decode(closed, at)
        except json.JSONDecodeError:
            return None
        values.append(value)
    if len(values) == 1:
        return closed if closed != text else None
    objects = [value for value in values if isinstance(value, dict)]
    if not objects or len(objects) != len(values):
        return None
    merged: dict[str, object] = {}
    for obj in objects:
        if merged.keys() & obj.keys():
            return None
        merged.update(obj)
    return json.dumps(merged, ensure_ascii=False)


def _skip_space(text: str, at: int) -> int:
    while at < len(text) and text[at] in " \t\r\n":
        at += 1
    return at


def _string_end(text: str, at: int) -> int:
    """Index just past the string opening at ``at``; -1 when it never closes."""
    i = at + 1
    while i < len(text):
        char = text[i]
        if char == "\\":
            i += 2
            continue
        if char == '"':
            return i + 1
        i += 1
    return -1


def _close_open_elements(text: str) -> str | None:
    """``text`` with a ``}`` inserted after each array element left open.

    A stack of open containers, each with what it expects next. Two moments
    are mended, both inside an object that is an array's element: a ``{``
    right after a comma (where a key must come, so the comma was the array's
    and the element had ended before it) and a ``]`` right after a complete
    value (the array closes, so the element had ended). Returns ``None`` at
    any other syntax error, and when containers are still open at the end.
    The text may hold several top-level values; joining them is the caller's.
    """
    stack: list[list[str]] = []  # [kind, expects]; kind "{" or "["
    inserts: list[int] = []
    last_comma = -1
    i = 0
    while i < len(text):
        char = text[i]
        if char in " \t\r\n":
            i += 1
            continue
        top = stack[-1] if stack else None
        in_element = len(stack) >= 2 and stack[-2][0] == "["
        if char == '"':
            end = _string_end(text, i)
            if end < 0:
                return None
            if top is not None:
                if top[0] == "{" and top[1] in ("key_or_end", "key"):
                    top[1] = "colon"
                elif top[1] in ("value", "value_or_end"):
                    top[1] = "comma_or_end"
                else:
                    return None
            i = end
        elif char == ":":
            if top is None or top[0] != "{" or top[1] != "colon":
                return None
            top[1] = "value"
            i += 1
        elif char == ",":
            if top is None or top[1] != "comma_or_end":
                return None
            top[1] = "key" if top[0] == "{" else "value"
            last_comma = i
            i += 1
        elif char in "{[":
            if top is not None and top[0] == "{" and top[1] == "key":
                if char != "{" or not in_element:
                    return None
                inserts.append(last_comma)  # the element ended before the comma
                stack.pop()
                top = stack[-1]
            elif top is not None and top[1] not in ("value", "value_or_end"):
                return None
            if top is not None:
                top[1] = "comma_or_end"  # what the parent expects once this closes
            stack.append([char, "key_or_end" if char == "{" else "value_or_end"])
            i += 1
        elif char == "}":
            if (
                top is None
                or top[0] != "{"
                or top[1] not in ("key_or_end", "comma_or_end")
            ):
                return None
            stack.pop()
            i += 1
        elif char == "]":
            if (
                top is not None
                and top[0] == "{"
                and top[1] == "comma_or_end"
                and in_element
            ):
                inserts.append(i)  # the element ended before the array closes
                stack.pop()
                continue
            if (
                top is None
                or top[0] != "["
                or top[1] not in ("value_or_end", "comma_or_end")
            ):
                return None
            stack.pop()
            i += 1
        else:
            match = _LITERAL.match(text, i)
            if match is None or (
                top is not None and top[1] not in ("value", "value_or_end")
            ):
                return None
            if top is not None:
                top[1] = "comma_or_end"
            i = match.end()
    if stack:
        return None
    for at in sorted(inserts, reverse=True):
        text = f"{text[:at]}}}{text[at:]}"
    return text

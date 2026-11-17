"""Reading a model's JSON reply leniently in form, never in meaning.

Replies requested in JSON mode (``response_format_mode = "json_object"``, the
only mode DeepSeek accepts) are JSON, but not always the JSON the reply model
describes. On the picture route of the time (2026-11-11, DeepSeek V4.1 Flash,
MemEye's 1,696 messages) 7 of 168 replies (4.2%) failed validation, and each
failure cost a second call that carried every picture again, although none of
them said anything a valid reply would not have (every episode reply has been
read here since 2026-11-12):

- an unknown key written empty inside a picture's entry (``"profile": []``,
  ``"ocr_note": ""``) -- the episode reply now ignores every key it does not
  define, at every level (:mod:`.pipeline.episode`);
- an array element the model never closed: ``"data": {...}`` was followed by
  ``, {`` or ``]`` where ``}}`` was due;
- a reply split in two objects: ``{"content": ..., "images": [...]}`` and then
  ``{"profile": []}``.

What this module mends, and what it leaves to fail: :func:`repair_json_form`
closes an array element left open, and joins a reply split into consecutive
objects whose keys do not overlap. The only character ever added is the ``}``
the next token implies, and nothing is removed or reordered; any other syntax
error (an unterminated string, two braces missing, an element of an object
rather than of an array, a key given twice across the split) is left to fail.
It does not touch meaning: required fields stay required, and every value is
still validated by the reply model.

A second reading is for a reply nothing above can mend, and is never applied
to a reply that validates: :func:`read_leniently` reads as much of a broken
reply as it can -- a string whose closing quote is missing, a quote copied
into a string unescaped, a reply cut off by the output limit -- and marks
what it could not read whole (:class:`CutText`, :class:`OpenObject`,
:class:`OpenArray`), so a caller that salvages a reply keeps only what is
complete (the episode call's last resort,
:mod:`everos.memory.extract.pipeline.episode_salvage`).
"""

from __future__ import annotations

__all__ = ["CutText", "OpenArray", "OpenObject", "read_leniently", "repair_json_form"]

import json
import re
from typing import Literal, Self

_LITERAL = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|true|false|null")


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


class CutText(str):
    """A string a broken reply never closed, as far as it was read.

    ``how`` is ``"end"`` when the reply ended inside it (cut by the output
    limit) and ``"line"`` when it ran into a line break followed by the
    reply's next key or closing bracket: its closing quote was lost (written
    as a typographic quote, say) and everything it held is there.
    """

    how: Literal["end", "line"]

    def __new__(cls, value: str, how: Literal["end", "line"]) -> Self:
        made = super().__new__(cls, value)
        made.how = how
        return made


class OpenObject(dict[str, object]):
    """An object a broken reply never closed: the members read before it stopped."""


class OpenArray(list[object]):
    """An array a broken reply never closed: the elements read before it stopped."""


_STRING_RUN = re.compile(r'[^"\\\r\n]+')
_KEY_AHEAD = re.compile(r'"(?:[^"\\\r\n]|\\.)*"\s*:')
_VALUE_START = frozenset('"{[-0123456789tfn')
_ESCAPES = {
    '"': '"',
    "\\": "\\",
    "/": "/",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
}
_SURROGATE = re.compile(r"[\ud800-\udfff]")

_Context = Literal["key", "member", "element", "top"]

_NOTHING = object()
"""What :meth:`_LenientReader.value` reads where no whole value is."""


def read_leniently(text: str) -> object | None:
    """As much of a broken JSON reply as can be read, with what is not whole marked.

    Reads the first object in ``text`` (anything before it is skipped) and
    any object that follows it with none of its keys (a reply split in two).
    Where the reply stops being JSON it is read as a model meant it:

    - a quote inside a string ends it only when what follows can follow a
      string there (``,`` and the next key, or the closing bracket); any
      other quote is the string's own text (an OCR line quoting ``"star"``);
    - a line break inside a string ends it, as a :class:`CutText` of ``how``
      ``"line"``, when the next line opens with the next key or a closing
      bracket -- the closing quote was lost; any other line break is text;
    - an escape JSON does not know reads as the character escaped, a comma
      may be missing between members on separate lines or be one too many,
      and an array element left open is closed where the next one starts;
    - where the text ends, or turns into something no reading above makes
      JSON of, reading stops: the string being read is a :class:`CutText`
      (``how`` ``"end"``) and each object or array still open an
      :class:`OpenObject` or :class:`OpenArray` of what it held so far.

    Returns:
        The value read -- an object, possibly open -- or ``None`` when
        ``text`` holds no object at all.
    """
    start = text.find("{")
    if start < 0:
        return None
    reader = _LenientReader(text, start)
    value = reader.object(in_array=False)
    while not reader.stopped:
        reader.skip_space()
        if reader.at >= len(text) or text[reader.at] != "{":
            break
        more = reader.object(in_array=False)
        if value.keys() & more.keys():
            break
        joined = {**value, **more}
        value = OpenObject(joined) if isinstance(more, OpenObject) else joined
    return value


class _LenientReader:
    """One pass of :func:`read_leniently` over a reply, from ``at``."""

    def __init__(self, text: str, at: int) -> None:
        self.text = text
        self.at = at
        self.stopped = False

    def skip_space(self) -> None:
        while self.at < len(self.text) and self.text[self.at] in " \t\r\n":
            self.at += 1

    def _space_from(self, at: int) -> int:
        while at < len(self.text) and self.text[at] in " \t\r\n":
            at += 1
        return at

    def value(self, context: _Context) -> object:
        self.skip_space()
        if self.at >= len(self.text):
            self.stopped = True
            return _NOTHING
        char = self.text[self.at]
        if char == "{":
            return self.object(in_array=context == "element")
        if char == "[":
            return self.array()
        if char == '"':
            return self.string(context)
        match = _LITERAL.match(self.text, self.at)
        if match is None or match.end() >= len(self.text):
            # Not a value, or one the reply may have cut short.
            self.stopped = True
            return _NOTHING
        self.at = match.end()
        return json.loads(match.group(0))

    def object(self, *, in_array: bool) -> dict[str, object]:
        self.at += 1  # the opening brace
        members: dict[str, object] = {}
        while True:
            self.skip_space()
            if self.at >= len(self.text):
                self.stopped = True
                return OpenObject(members)
            char = self.text[self.at]
            if char == "}":
                self.at += 1
                return members
            if char == ",":
                self.at += 1
                continue
            if in_array and char in "{]":
                # An element left open: the next element, or the array's end.
                return members
            if char != '"':
                self.stopped = True
                return OpenObject(members)
            key = self.string("key")
            self.skip_space()
            if (
                self.stopped
                or isinstance(key, CutText)
                or self.at >= len(self.text)
                or self.text[self.at] != ":"
            ):
                self.stopped = True
                return OpenObject(members)
            self.at += 1
            value = self.value("member")
            if value is not _NOTHING:
                members[key] = value
            if self.stopped:
                return OpenObject(members)

    def array(self) -> list[object]:
        self.at += 1  # the opening bracket
        items: list[object] = []
        while True:
            self.skip_space()
            if self.at >= len(self.text):
                self.stopped = True
                return OpenArray(items)
            char = self.text[self.at]
            if char == "]":
                self.at += 1
                return items
            if char == ",":
                self.at += 1
                continue
            if char not in _VALUE_START:
                self.stopped = True
                return OpenArray(items)
            item = self.value("element")
            if item is not _NOTHING:
                items.append(item)
            if self.stopped:
                return OpenArray(items)

    def string(self, context: _Context) -> str:
        text = self.text
        chunks: list[str] = []
        at = self.at + 1  # the opening quote
        while at < len(text):
            run = _STRING_RUN.match(text, at)
            if run is not None:
                chunks.append(run.group(0))
                at = run.end()
                continue
            char = text[at]
            if char == '"':
                if self._ends_string(at + 1, context):
                    self.at = at + 1
                    return _decoded("".join(chunks))
                chunks.append(char)
                at += 1
            elif char == "\\":
                if at + 1 >= len(text):
                    break
                escaped = text[at + 1]
                if escaped == "u":
                    digits = text[at + 2 : at + 6]
                    if len(digits) < 4:
                        break
                    try:
                        chunks.append(chr(int(digits, 16)))
                        at += 6
                    except ValueError:
                        chunks.append(escaped)
                        at += 2
                else:
                    chunks.append(_ESCAPES.get(escaped, escaped))
                    at += 2
            else:  # a line break
                if self._line_ends_string(at, context):
                    self.at = at
                    return CutText(_decoded("".join(chunks)), "line")
                chunks.append(char)
                at += 1
        self.at = len(text)
        self.stopped = True
        return CutText(_decoded("".join(chunks)), "end")

    def _ends_string(self, after: int, context: _Context) -> bool:
        """Whether a quote just before ``after`` closes the string being read."""
        at = self._space_from(after)
        if at >= len(self.text) or context == "top":
            return True
        char = self.text[at]
        if context == "key":
            return char == ":"
        if context == "member":
            if char == "}":
                return True
            if char == ",":
                return self._member_follows(at + 1)
            # A comma missing between members on separate lines.
            return (
                char == '"'
                and "\n" in self.text[after:at]
                and bool(_KEY_AHEAD.match(self.text, at))
            )
        if char == "]":
            return True
        if char == ",":
            ahead = self._space_from(at + 1)
            return ahead >= len(self.text) or self.text[ahead] in _VALUE_START | {"]"}
        return False

    def _member_follows(self, at: int) -> bool:
        at = self._space_from(at)
        if at >= len(self.text):
            return True
        return self.text[at] in "}{" or bool(_KEY_AHEAD.match(self.text, at))

    def _line_ends_string(self, at: int, context: _Context) -> bool:
        """Whether a line break at ``at`` ends a string whose closing quote is lost."""
        ahead = self._space_from(at)
        if ahead >= len(self.text):
            return False
        char = self.text[ahead]
        if context == "member":
            return char in "}]," or bool(_KEY_AHEAD.match(self.text, ahead))
        if context == "element":
            return char in ",]"
        return False


def _decoded(text: str) -> str:
    """``text`` with the surrogate pairs ``\\u`` escapes wrote joined."""
    if _SURROGATE.search(text) is None:
        return text
    return text.encode("utf-16", "surrogatepass").decode("utf-16", "replace")

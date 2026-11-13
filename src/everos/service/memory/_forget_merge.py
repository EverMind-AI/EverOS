"""An undo carried over a person's edit of the version its forget wrote.

docs/api.md, ``/memory/forget/undo``. A forget writes a version in place of
another (a picture's text edit, a re-extracted episode, a group's scene
narrated again); a person may edit that version through ``/memory/update``
before the forget is undone. The undo then writes the version the forget
replaced **on top of the person's edit**, as a three-way merge per field:

* *base* -- the version the forget wrote;
* *ours* -- the person's version, the newest;
* *theirs* -- the version the forget replaced, which the undo brings back.

A field the person left as the forget wrote it takes its value from before
the forget; a field the forget did not change keeps the person's value; a
text field both changed is merged when their changes touch different parts
of it, word by word (a picture's reference put back into a text the person
corrected elsewhere), and is a conflict otherwise. The person's edit fields
(``supersedes``, ``edited_by``, ``edited_at``, ``held``) stay theirs: the
version written is still the person's, with the forget undone.
"""

from __future__ import annotations

__all__ = ["merged_over_edit"]

import re
from collections.abc import Mapping
from difflib import SequenceMatcher
from typing import Any

from pydantic import BaseModel

from everos.memory.models import EDIT_FIELDS

_OWN = frozenset({*EDIT_FIELDS, "version"})
"""Fields the merge never takes from before the forget: the person's edit
fields, and a scene's version number (the undo numbers its own)."""


def merged_over_edit[T: BaseModel](base: T, ours: T, theirs: T) -> T | None:
    """``theirs`` carried over ``ours``, the person's edit of ``base``.

    Returns ``None`` when a field holds changes of both that touch: the
    person's text would be lost.
    """
    before, edited, back = (_fields(m) for m in (base, ours, theirs))
    data = dict(edited)
    for key in dict.fromkeys([*before, *edited, *back]):
        if key in _OWN:
            continue
        was, now, then = before.get(key), edited.get(key), back.get(key)
        if now == was:
            data[key] = then
        elif then in (was, now):
            continue
        elif isinstance(was, str) and isinstance(now, str) and isinstance(then, str):
            text = merged_text(was, now, then)
            if text is None:
                return None
            data[key] = text
        else:
            return None
    return type(ours).model_validate(data)


def merged_text(base: str, ours: str, theirs: str) -> str | None:
    """``theirs``' changes of ``base`` applied to ``ours``; ``None`` when they
    touch a word of ``base`` that ``ours`` changed, or meet it (a three-way
    merge on words and the whitespace between them)."""
    words = _words(base)
    mine, yours = _changes(words, _words(ours)), _changes(words, _words(theirs))
    for a1, a2, _ in mine:
        for b1, b2, _ in yours:
            if _touch(a1, a2, b1, b2):
                return None
    out: list[str] = []
    at = 0
    for start, end, text in sorted([*mine, *yours], key=lambda c: (c[0], c[1])):
        out += words[at:start]
        out.append(text)
        at = end
    out += words[at:]
    return "".join(out)


_WORD = re.compile(r"\s+|\S+")


def _words(text: str) -> list[str]:
    return _WORD.findall(text)


def _changes(base: list[str], new: list[str]) -> list[tuple[int, int, str]]:
    """``(start, end, replacement)`` in ``base``'s words of every change to
    ``new``."""
    matcher = SequenceMatcher(None, base, new, autojunk=False)
    return [
        (i1, i2, "".join(new[j1:j2]))
        for tag, i1, i2, j1, j2 in matcher.get_opcodes()
        if tag != "equal"
    ]


def _touch(a1: int, a2: int, b1: int, b2: int) -> bool:
    """Whether two changes of one text meet: their spans overlap, or one is
    put in at the other's place or at an edge of what the other replaced."""
    if a1 < b2 and b1 < a2:
        return True
    if a1 == a2 and b1 <= a1 <= b2:
        return True
    return b1 == b2 and a1 <= b1 <= a2


def _fields(model: BaseModel) -> Mapping[str, Any]:
    return model.model_dump(mode="json")

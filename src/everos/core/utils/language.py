"""Which output language a piece of extracted text is written in.

Answers with an ``everalgo`` ``OutputLanguage`` value (the strings below), so
the language an episode was written in can be pinned on the extractions that
follow it. Decided by Unicode script, not a trained detector: the target set
is seven languages, and script ranges separate the non-Latin ones exactly.
Within the Latin script English is the default; German and Spanish need two
markers of their own, so a short German or Spanish text may read as English.
"""

from __future__ import annotations

__all__ = ["detect_output_language"]

import re
from collections import Counter
from collections.abc import Sequence

type _Ranges = Sequence[tuple[int, int]]

_KANA: _Ranges = ((0x3040, 0x30FF), (0x31F0, 0x31FF))
_HANGUL: _Ranges = ((0xAC00, 0xD7AF), (0x1100, 0x11FF), (0x3130, 0x318F))
_CYRILLIC: _Ranges = ((0x0400, 0x04FF),)
_HAN: _Ranges = ((0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0xF900, 0xFAFF))
_LATIN: _Ranges = ((0x41, 0x5A), (0x61, 0x7A), (0xC0, 0x24F))

_NON_LATIN: tuple[tuple[str, _Ranges, float], ...] = (
    # Any real share of kana means Japanese, even beside more kanji than kana.
    ("Japanese", _KANA, 0.05),
    ("Korean", _HANGUL, 0.2),
    ("Russian", _CYRILLIC, 0.2),
    ("Chinese", _HAN, 0.2),
)
"""Checked in order; a script decides once it holds this share of the letters.

The share filters out a quoted foreign name inside otherwise Latin prose,
while identifiers and paths in Latin cannot outvote a real CJK narrative.
"""

_LATIN_MARKERS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "German",
        re.compile(
            r"[äöüß]|\b(?:und|nicht|ich|ist|der|die|das|mit|für|auch|eine|dass|"
            r"wurde|haben|sind)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "Spanish",
        re.compile(
            r"[ñ¿¡áéíóú]|\b(?:el|los|las|que|por|para|con|una|pero|está|"
            r"fue|hay|del)\b",
            re.IGNORECASE,
        ),
    ),
)

_LATIN_MIN_MARKERS = 2
"""A single loanword ("café", "über") must not relabel English text."""


def detect_output_language(text: str) -> str | None:
    """The ``OutputLanguage`` value ``text`` is written in.

    Args:
        text: Extracted prose, such as an episode narrative.

    Returns:
        ``"Chinese"``, ``"English"``, ``"German"``, ``"Japanese"``,
        ``"Korean"``, ``"Russian"`` or ``"Spanish"``; ``None`` when ``text``
        has no letters to judge (empty, digits, punctuation).
    """
    counts = _script_counts(text)
    letters = sum(counts.values())
    if not letters:
        return None
    for language, _ranges, share in _NON_LATIN:
        if counts[language] / letters >= share:
            return language
    return _latin_language(text)


def _script_counts(text: str) -> Counter[str]:
    """Letters per script, keyed by the language the script stands for."""
    counts: Counter[str] = Counter()
    for char in text:
        point = ord(char)
        for language, ranges, _share in _NON_LATIN:
            if _within(point, ranges):
                counts[language] += 1
                break
        else:
            if _within(point, _LATIN):
                counts["Latin"] += 1
    return counts


def _within(point: int, ranges: _Ranges) -> bool:
    return any(low <= point <= high for low, high in ranges)


def _latin_language(text: str) -> str:
    """German or Spanish on enough markers that outnumber the other, else English."""
    hits = {name: len(pattern.findall(text)) for name, pattern in _LATIN_MARKERS}
    best = max(hits, key=hits.__getitem__)
    others = [count for name, count in hits.items() if name != best]
    if hits[best] >= _LATIN_MIN_MARKERS and all(hits[best] > c for c in others):
        return best
    return "English"

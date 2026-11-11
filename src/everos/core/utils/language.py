"""Which output language a piece of extracted text is written in.

Answers with an ``everalgo`` ``OutputLanguage`` value, so the language an
episode was written in can be pinned on the extractions that follow it. The
target set is Chinese, Japanese and English, decided by Unicode script rather
than a trained detector: any real share of kana is Japanese, a large enough
share of Han characters is Chinese, and every other text with letters --
Latin, Hangul, Cyrillic or any other script -- is English. Latin text is never
told apart by language: words a European language shares with English
("die", "Los Angeles", "café") must not relabel an English narrative.
"""

from __future__ import annotations

__all__ = ["detect_output_language"]

from collections.abc import Sequence

type _Ranges = Sequence[tuple[int, int]]

_KANA: _Ranges = ((0x3040, 0x30FF), (0x31F0, 0x31FF))
_HAN: _Ranges = ((0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0xF900, 0xFAFF))

_SCRIPT_SHARES: tuple[tuple[str, _Ranges, float], ...] = (
    # Any real share of kana means Japanese, even beside more kanji than kana.
    ("Japanese", _KANA, 0.05),
    ("Chinese", _HAN, 0.2),
)
"""Checked in order; a script decides once it holds this share of the letters.

The share filters out a quoted name inside otherwise English prose, while
identifiers and paths in Latin cannot outvote a real Chinese or Japanese
narrative.
"""


def detect_output_language(text: str) -> str | None:
    """The ``OutputLanguage`` value ``text`` is written in.

    Args:
        text: Extracted prose, such as an episode narrative.

    Returns:
        ``"Japanese"``, ``"Chinese"`` or ``"English"``; ``None`` when ``text``
        has no letters to judge (empty, digits, punctuation).
    """
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return None
    for language, ranges, share in _SCRIPT_SHARES:
        if _count_within(letters, ranges) / len(letters) >= share:
            return language
    return "English"


def _count_within(letters: Sequence[str], ranges: _Ranges) -> int:
    return sum(
        any(low <= ord(char) <= high for low, high in ranges) for char in letters
    )

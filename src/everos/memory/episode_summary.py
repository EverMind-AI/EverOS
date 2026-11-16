"""The summary an episode carries until its model-written preview lands.

``Episode.summary`` is the short preview get and search return beside an
episode. The online pipeline makes one model call per memcell (the episode
itself) and writes every copy with :func:`placeholder_summary`; the OME
strategy ``summarize_episode`` then writes the model's preview, in the
episode's language, over every copy still showing the placeholder.

Load-bearing: the placeholder is a pure function of the episode text -- no
clock, randomness or configuration -- so the strategy recomputes it from the
event and replaces only the copies that still carry it. Changing these rules
leaves episodes written under the old ones with their placeholder for good.
"""

from __future__ import annotations

__all__ = ["placeholder_summary"]

from everalgo.utils import ascii_width

_WIDTH = 120
"""Width cap in ``ascii_width`` units (an East Asian wide character counts 2)."""

_ELLIPSIS = "…"


def placeholder_summary(episode: str) -> str:
    """The summary an episode is written with, before its model preview.

    The episode itself when it fits the width cap; otherwise its opening cut
    to the cap, backed off to the previous space rather than splitting a
    Latin word, right-stripped and closed with an ellipsis (U+2026).

    Args:
        episode: The episode text.

    Returns:
        The placeholder; before the ellipsis, always a prefix of ``episode``.
    """
    if ascii_width(episode) <= _WIDTH:
        return episode
    head = _prefix_within_width(episode, _WIDTH)
    if _splits_a_latin_word(head, episode[len(head)]):
        word_start = head.rfind(" ")
        if word_start > 0:
            head = head[:word_start]
    return head.rstrip() + _ELLIPSIS


def _prefix_within_width(text: str, width: int) -> str:
    """The longest prefix of ``text`` whose ``ascii_width`` is at most ``width``."""
    used = 0
    for index, char in enumerate(text):
        used += ascii_width(char)
        if used > width:
            return text[:index]
    return text


def _splits_a_latin_word(head: str, next_char: str) -> bool:
    """Whether cutting between ``head`` and ``next_char`` splits an ASCII word."""
    last = head[-1:]
    return (
        last.isascii()
        and last.isalnum()
        and next_char.isascii()
        and next_char.isalnum()
    )

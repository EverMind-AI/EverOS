"""``placeholder_summary``: the summary an episode is written with online.

Pins the rules the ``summarize_episode`` strategy relies on to recognise a
copy it still has to replace: a pure function of the episode text, at most
120 ``ascii_width`` units before the ellipsis, the text itself when it fits,
never a Latin word cut in half.
"""

from __future__ import annotations

import pytest
from everalgo.utils import ascii_width

from everos.memory.episode_summary import placeholder_summary

_ELLIPSIS = "…"


def test_short_text_is_its_own_placeholder() -> None:
    text = "Alice booked a ryokan in Higashiyama."
    assert placeholder_summary(text) == text


def test_text_exactly_at_the_cap_is_kept_whole() -> None:
    text = "a" * 120
    assert placeholder_summary(text) == text


def test_long_text_is_cut_at_a_word_start_and_closed_with_an_ellipsis() -> None:
    words = " ".join(f"word{i:02d}" for i in range(40))  # 7 units per word + space

    placeholder = placeholder_summary(words)

    assert placeholder.endswith(_ELLIPSIS)
    head = placeholder.removesuffix(_ELLIPSIS)
    assert ascii_width(head) <= 120
    assert words.startswith(head)
    assert words[len(head)] == " "  # the cut fell between two words
    assert head == head.rstrip()


def test_a_latin_word_crossing_the_cap_is_dropped_whole() -> None:
    text = "x" * 115 + " Kyoto trip"  # "Kyoto" crosses the 120-unit cap

    assert placeholder_summary(text) == "x" * 115 + _ELLIPSIS


def test_one_unbroken_latin_run_is_cut_at_the_cap() -> None:
    """No space to back off to: the cut stays at the cap."""
    text = "y" * 300
    assert placeholder_summary(text) == "y" * 120 + _ELLIPSIS


def test_cjk_counts_two_units_per_character() -> None:
    text = "京" * 100  # 200 units of a wide character

    placeholder = placeholder_summary(text)

    assert placeholder == "京" * 60 + _ELLIPSIS
    assert ascii_width(placeholder.removesuffix(_ELLIPSIS)) == 120


def test_cjk_text_that_fits_by_width_is_kept_whole() -> None:
    text = "京都" * 30  # 60 characters, 120 units
    assert placeholder_summary(text) == text


@pytest.mark.parametrize(
    "text",
    [
        "short",
        "word " * 60,
        "京都的春天 " * 40,
        "Mixed 京都 text " * 20,
    ],
)
def test_the_placeholder_is_deterministic_and_a_prefix(text: str) -> None:
    first, second = placeholder_summary(text), placeholder_summary(text)

    assert first == second
    assert text.startswith(first.removesuffix(_ELLIPSIS))
    assert ascii_width(first.removesuffix(_ELLIPSIS)) <= 120

"""verbatim_part: the one test of whether a quote copies a group message.

docs/group-memory.md, sections 7.5 and 8.4 (Step 2 notes). What these pin:
whitespace runs and curly versus straight quotation marks are forgiven, in
both directions; nothing else is; what comes back is the message's own text.
"""

from __future__ import annotations

import pytest

from everos.memory.groups import verbatim_part

_TEXT = (
    "Demo feedback:  the “discount cap” was solid;\nit’s the ‘timeline’ that slipped."
)


@pytest.mark.parametrize(
    ("quote", "expected"),
    [
        ('the "discount cap" was solid', "the “discount cap” was solid"),
        ("it's the 'timeline'", "it’s the ‘timeline’"),
        ("solid; it’s the", "solid; it’s the"),
        ("Demo   feedback: the", "Demo feedback: the"),
    ],
)
def test_forgiven_differences_return_the_messages_own_words(
    quote: str, expected: str
) -> None:
    assert verbatim_part(quote, _TEXT) == expected


@pytest.mark.parametrize(
    "quote",
    [
        "",
        "   ",
        "the discount cap was solid",  # marks left out
        "the “discount limit” was solid",  # a word changed
        "it's the timeline that slipped",  # marks left out
        "the “discount cap” was solid -- it’s",  # punctuation added
    ],
)
def test_anything_else_is_not_verbatim(quote: str) -> None:
    assert verbatim_part(quote, _TEXT) is None


def test_a_straight_message_matches_a_curly_quote() -> None:
    text = '"Tentatively circulated" means'
    assert verbatim_part("“Tentatively circulated”", text) == '"Tentatively circulated"'

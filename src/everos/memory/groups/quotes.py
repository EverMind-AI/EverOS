"""Whether a quote is a verbatim part of a group message, and which part.

Every check that a model's quote copies a message -- the group-profile
verifier (section 7.5), the constraint binding and the read-time check of a
stored quote (section 8.4) -- goes through :func:`verbatim_part`, so they
agree on what "verbatim" means:

* whitespace is normalised on both sides (runs become one space);
* curly quotation marks read as straight ones (U+2018, U+2019, U+201A and
  U+201B as ``'``; U+201C, U+201D, U+201E and U+201F as ``"``): models
  write one where the text has the other, in both directions, and the words
  are the same;
* nothing else is forgiven -- a quote that changes, drops or reorders a word
  is not verbatim.

What comes back is the message's own characters, whitespace normalised, so a
stored quote is always a verbatim part of the message it cites, whichever
quotation marks the model wrote.
"""

from __future__ import annotations

__all__ = ["normalise_space", "verbatim_part"]

_STRAIGHT = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "‚": "'",
        "‛": "'",
        "“": '"',
        "”": '"',
        "„": '"',
        "‟": '"',
    }
)
"""Curly quotation marks to straight ones, one character for one, so
positions in the straightened text are positions in the original."""


def normalise_space(text: str) -> str:
    """``text`` with every run of whitespace one space, and no edge spaces."""
    return " ".join(text.split())


def verbatim_part(quote: str, text: str) -> str | None:
    """The part of ``text`` that ``quote`` copies; ``None`` when it copies none.

    Args:
        quote: The words the model says it copied.
        text: The message's text.

    Returns:
        The matching span of the whitespace-normalised ``text`` -- its own
        quotation marks, not the model's -- or ``None`` for an empty quote or
        one that is not part of ``text``.
    """
    wanted = normalise_space(quote)
    if not wanted:
        return None
    said = normalise_space(text)
    at = said.translate(_STRAIGHT).find(wanted.translate(_STRAIGHT))
    if at < 0:
        return None
    return said[at : at + len(wanted)]

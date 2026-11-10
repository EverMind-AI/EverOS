"""The language a cell's extractions are written in, resolved in code.

Every extraction prompt carries an ``<output_language>`` rule. Leaving the
language unnamed hands everalgo's participant rule to the model ("write in
the language the participants use"), which everalgo itself measured at about
one wrong language in thirteen, failing towards Chinese. The 2.0 backbone
fails far more often on picture cells: whole replies -- episode, picture
descriptions, profile items -- in Chinese for English conversations, and the
language read back off such a reply then pinned the cell's facts and the
owner's profile to it. Naming the language is what everalgo measured at zero
drift, so the pipeline names it, and decides it here, from the input.

The rules, in order:

- **Human messages decide.** The text of the cell's ``role="user"`` chat
  messages, without picture references (``[IMAGE: <hash>]`` -- the hash's
  hex letters would read as English), fenced code blocks and links: pasted
  code and URLs are not the language anyone speaks.
- **One vote per message.** Each message with a letter is judged on its own
  (:func:`everos.core.utils.language.detect_output_language`); the language
  with the most messages wins, then the one with more letters. A long pasted
  paragraph is one message, so it cannot outvote the language the speaker
  keeps writing in.
- **A short cell reads its session.** When the cell's human messages hold
  fewer than :data:`MIN_LETTERS` letters -- a picture with no caption, an
  "ok" -- the session's earlier cells are read too, newest first, until that
  much text has been read: a picture-only cell of an English chat is English.
- **The assistant only when no human wrote a letter.** Then its messages are
  read the same way (a session that opens with a photo and the assistant's
  reply still has a language).
- ``None`` when no message anywhere has a letter: the caller leaves the
  choice to the model, as before.

Deterministic by construction: the same session yields the same language on
every run and every replay, whatever the model would have chosen.
"""

from __future__ import annotations

__all__ = [
    "HISTORY_CELLS",
    "MIN_LETTERS",
    "conversation_language",
    "resolve_cell_language",
]

import re
from collections import Counter
from collections.abc import Iterable, Sequence

from everalgo.types import ChatMessage, MemCell
from everalgo.user_memory import OutputLanguage

from everos.core.utils.language import detect_output_language
from everos.memory.ports import MemcellStore, SessionKey

MIN_LETTERS = 40
"""Letters of human text a cell needs to be judged on its own.

About eight English words: enough for the Latin markers to separate English
from German or Spanish, and enough that a stray "ok" in a Chinese chat is not
the only evidence."""

HISTORY_CELLS = 8
"""Earlier cells of the session read, at most, for a cell below the minimum."""

_NOT_SPOKEN = re.compile(
    r"\[IMAGE:[^\]]*\]"  # a picture reference: a content hash, not text
    r"|```.*?(?:```|\Z)"  # a fenced code block, closed or not
    r"|https?://\S+",  # a link
    re.DOTALL,
)

_SPEAKER_ROLES: tuple[tuple[str, ...], ...] = (("user",), ("user", "assistant"))
"""Whose messages are read, in turn: the humans', then anyone's."""


def conversation_language(cells: Sequence[MemCell]) -> OutputLanguage | None:
    """The language the cells' speakers write in, or ``None`` when none wrote a letter.

    Args:
        cells: The cell being extracted first, then earlier cells of its
            session, newest first. Cells are read in this order until
            :data:`MIN_LETTERS` letters of speaker text have been read; the
            ones after that are not consulted.

    Returns:
        The ``OutputLanguage`` the read messages vote for.
    """
    for roles in _SPEAKER_ROLES:
        language = _vote(_spoken(cells, roles))
        if language is not None:
            return language
    return None


def _has_enough_text(cell: MemCell) -> bool:
    """Whether the cell's human messages alone are enough to decide on."""
    return _letters(_spoken([cell], _SPEAKER_ROLES[0])) >= MIN_LETTERS


async def resolve_cell_language(
    cell: MemCell,
    *,
    memcell_id: str,
    session: SessionKey,
    memcells: MemcellStore,
) -> OutputLanguage | None:
    """The language to name in every extraction prompt of ``cell``.

    Reads the session's earlier cells only when the cell's own human
    messages are below :data:`MIN_LETTERS`, so the common cell costs no
    store read.

    Args:
        cell: The cell being extracted.
        memcell_id: Its id, left out of the session read.
        session: The conversation the cell was cut from.
        memcells: Where the session's earlier cells are read from.
    """
    if _has_enough_text(cell):
        return conversation_language([cell])
    earlier = await memcells.recent_in_session(
        session, until_ms=cell.timestamp, limit=HISTORY_CELLS, exclude=memcell_id
    )
    return conversation_language([cell, *earlier])


def _spoken(cells: Iterable[MemCell], roles: Sequence[str]) -> list[str]:
    """Speaker text of ``roles``, cell by cell, until enough letters are read."""
    texts: list[str] = []
    for cell in cells:
        for item in cell.items:
            if isinstance(item, ChatMessage) and item.role in roles:
                text = _NOT_SPOKEN.sub(" ", item.content or "")
                if any(ch.isalpha() for ch in text):
                    texts.append(text)
        if _letters(texts) >= MIN_LETTERS:
            break
    return texts


def _letters(texts: Iterable[str]) -> int:
    return sum(ch.isalpha() for text in texts for ch in text)


def _vote(texts: Sequence[str]) -> OutputLanguage | None:
    """Most messages first, then most letters, then ``OutputLanguage`` order."""
    votes: Counter[str] = Counter()
    letters: Counter[str] = Counter()
    for text in texts:
        language = detect_output_language(text)
        if language is None:
            continue
        votes[language] += 1
        letters[language] += sum(ch.isalpha() for ch in text)
    if not votes:
        return None
    order = [member.value for member in OutputLanguage]
    best = max(votes, key=lambda name: (votes[name], letters[name], -order.index(name)))
    return OutputLanguage(best)

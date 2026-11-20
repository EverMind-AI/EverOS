"""Which language a Reflection merge is written in.

A merged episode spans many conversations of one owner, the way the profile
does, so it follows the owner's profile language (``Profile.primary_language``,
frozen once known). Before the owner has one, the members vote with the
languages they were stored with. A split vote names nothing: the reflector
picks the language and it is read off the merged text afterwards. The
language the merge is asked for and the one it is stored with are kept apart
so the log can say where the stored one came from.
"""

from __future__ import annotations

__all__ = ["LanguageChoice", "merged_language", "steer_language"]

from collections import Counter
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, NamedTuple

from everos.core.utils.language import detect_output_language

if TYPE_CHECKING:
    from everos.memory.models import Episode
    from everos.memory.ports import Stored

_MAJORITY_SHARE = 0.6
"""Share of the members the winning language must exceed (strictly).

A 3-2 split lands on 0.6 and is refused: answered with its narrow winner, a
genuinely mixed cluster would be merged into a language the owner only half
uses. 2 of 2 and 2 of 3 decide, 1 of 2 does not."""


class LanguageChoice(NamedTuple):
    """A language and where it came from.

    Attributes:
        value: The ``OutputLanguage`` value; ``None`` leaves it to the model.
        source: ``"profile"`` (the owner's frozen language), ``"vote"`` (the
            members' majority) or ``"text"`` (read off the merged episode).
    """

    value: str | None
    source: Literal["profile", "vote", "text"]


def steer_language(
    owner_language: str | None, members: Sequence[Stored[Episode]]
) -> LanguageChoice:
    """The language the reflector is asked to write the merge in.

    Args:
        owner_language: The owner's profile language; ``None`` without one.
        members: The cluster's episodes sent to the merge.
    """
    if owner_language:
        return LanguageChoice(owner_language, "profile")
    return LanguageChoice(_vote(members), "vote")


def merged_language(steer: LanguageChoice, merged_text: str) -> LanguageChoice:
    """The language the merged episode is stored with.

    The steer when it named one; otherwise the reflector chose, and the
    language is read off ``merged_text``.
    """
    if steer.value is not None:
        return steer
    return LanguageChoice(detect_output_language(merged_text), "text")


def _vote(members: Sequence[Stored[Episode]]) -> str | None:
    """The language a clear majority of ``members`` are written in, else ``None``.

    Each member votes with its stored ``language``; one stored without any (an
    episode written before the field existed) votes with the one read off its
    text. A member with no readable language abstains but still counts.
    """
    if not members:
        return None
    readings = [
        m.item.language or detect_output_language(m.item.episode) for m in members
    ]
    counts = Counter(reading for reading in readings if reading is not None)
    if not counts:
        return None
    language, hits = counts.most_common(1)[0]
    return language if hits / len(readings) > _MAJORITY_SHARE else None

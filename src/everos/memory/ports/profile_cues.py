"""The profile cues port: what an episode call marked for each person's profile.

When a space's profile reads cues (``[profile] input = "cues"``), the episode
call of each segment also names, per human speaker, the messages and pictures
that say something lasting about them
(:mod:`everos.memory.extract.pipeline.profile_cues`). They are derived state,
kept in the system database beside the segment's memcell and never in
markdown: one record per ``(segment, person)``, written for every human
speaker of the segment -- a speaker the call marked nothing for gets an empty
record, which is what tells "nothing lasting was said" from "never marked"
(a segment extracted before cues existed, or while no profile read them).

A forget never edits a record. A forgotten message or picture stays in the
segment's payload, and every reader skips it (the message forget log, the
forget log's pictures); a cue resting on it is skipped by the same reads, so
an undo has nothing to put back.
"""

from __future__ import annotations

__all__ = ["PersonCues", "ProfileCueStore"]

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .query import SessionKey


@dataclass(frozen=True)
class PersonCues:
    """What one segment's episode call marked for one person.

    Attributes:
        message_ids: The person's own messages that say something lasting
            about them, in conversation order.
        image_ids: The pictures they posted that show something lasting about
            them (content hashes), in conversation order.
        read: Whether the person's profile has read the segment: a segment
            read once (with an update call or without one, when it held no
            cue) is not queued on the profile again.
    """

    message_ids: tuple[str, ...] = ()
    image_ids: tuple[str, ...] = ()
    read: bool = False

    @property
    def empty(self) -> bool:
        """Whether the call marked nothing for the person."""
        return not self.message_ids and not self.image_ids


@runtime_checkable
class ProfileCueStore(Protocol):
    """The cue records of the segments, one per ``(segment, person)``."""

    async def record(
        self,
        session: SessionKey,
        memcell_id: str,
        cues: Mapping[str, PersonCues],
    ) -> None:
        """Write one segment's cues, one record per person in ``cues``.

        A record written again (a segment's extraction retried) takes the new
        messages and pictures and keeps whether it was read.
        """
        ...

    async def of_person(
        self, person_id: str, memcell_ids: Collection[str]
    ) -> dict[str, PersonCues]:
        """The person's records among ``memcell_ids``, by segment; a segment
        never marked for them is absent."""
        ...

    async def mark_read(self, person_id: str, memcell_ids: Collection[str]) -> None:
        """Record that the person's profile has read these segments (an id
        with no record is ignored)."""
        ...

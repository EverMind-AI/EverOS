"""What a forget addresses, and what a forget and its undo answer with.

One target type for every owner type and target kind, one answer for a
forget (a dry run or a real one) and one for an undo; the routes shape them
per owner (docs/api.md). Record ids are named as a read names them
(``<owner>_<entry>``), profile items by ``item_id``.
"""

from __future__ import annotations

__all__ = [
    "ForgetResult",
    "ForgetTarget",
    "ForgettableKind",
    "ForgottenPicture",
    "TargetKind",
    "UndoResult",
]

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

type TargetKind = Literal[
    "record", "session", "message", "member", "owner", "condition", "image"
]
"""What a forget addresses: records by id, one conversation, messages of one
conversation, one member's messages (a group), everything of a user, a user's
memories of some kinds in a time range, or one picture by content hash."""

type ForgettableKind = Literal["episode", "image_memory", "scene", "profile"]
"""The kinds a ``condition`` target names."""


@dataclass(frozen=True)
class ForgetTarget:
    """What one forget addresses.

    Exactly the fields of its ``kind`` are set; the request DTOs validate
    that, and which kinds an owner type takes.
    """

    kind: TargetKind
    record_ids: tuple[str, ...] = ()
    session_id: str | None = None
    message_ids: tuple[str, ...] = ()
    member_id: str | None = None
    image_id: str | None = None
    memory_types: tuple[ForgettableKind, ...] = ()
    start: int | None = None
    end: int | None = None

    def label(self, owner_id: str) -> str:
        """The forget log's ``target_id``: what an undo by target finds."""
        if self.kind == "message":
            return f"{self.session_id}:{','.join(self.message_ids)}"[:500]
        if self.kind == "member":
            return self.member_id or ""
        if self.kind == "session":
            return self.session_id or ""
        if self.kind == "image":
            return self.image_id or ""
        if self.kind == "owner":
            return owner_id
        if self.kind == "condition":
            return (
                f"{','.join(self.memory_types)}:{self.start or ''}-{self.end or ''}"
            )[:500]
        return ",".join(self.record_ids)[:500]

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "record_ids": list(self.record_ids),
            "session_id": self.session_id,
            "message_ids": list(self.message_ids),
            "member_id": self.member_id,
            "image_id": self.image_id,
            "memory_types": list(self.memory_types),
            "start": self.start,
            "end": self.end,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> ForgetTarget:
        return cls(
            kind=data["kind"],
            record_ids=tuple(data.get("record_ids") or ()),
            session_id=data.get("session_id"),
            message_ids=tuple(data.get("message_ids") or ()),
            member_id=data.get("member_id"),
            image_id=data.get("image_id"),
            memory_types=tuple(data.get("memory_types") or ()),
            start=data.get("start"),
            end=data.get("end"),
        )


@dataclass(frozen=True)
class ForgottenPicture:
    """One picture memory a forget (or its undo) touched.

    Attributes:
        entry_id: The occurrence's entry id in markdown.
        image_id: The file, by content hash.
        name: The client's file name for this arrival, ``""`` when it sent none.
    """

    entry_id: str
    image_id: str
    name: str


@dataclass(frozen=True)
class ForgetResult:
    """What one forget touched -- or, for a dry run, would touch.

    Attributes:
        forget_id: The undo token (``fg_<hex>``); ``""`` on a dry run.
        dry_run: Whether nothing was changed.
        target: What the forget addressed.
        messages: ``(session_id, message_id)`` of the messages it forgot.
        episodes: Episode versions taken out of default reads: marked
            forgotten, or replaced by a version without what was forgotten.
        new_episodes: Those replacements (a real run only).
        rewritten: How many episodes get (or got) a new version.
        image_memories: Picture memories marked forgotten.
        pictures: The same, with the name each arrival carried.
        images: Picture files taken out of circulation.
        scenes: Scene versions taken out of default reads.
        new_scenes: Scene versions written in their place (a real run only).
        rewritten_scenes: How many scenes get (or got) a new version.
        profiles: People whose whole profile document in the group went.
        profile_items: Profile items and replaced values forgotten.
        profile_item_count: How many of those are items (not replaced values).
        stale_profile_items: Profile items taken out to be written again from
            their remaining segments (a group).
        group_profile_items: Group-profile items forgotten.
        stale_group_profile_items: Group-profile items taken out to be
            written again.
    """

    forget_id: str
    dry_run: bool
    target: ForgetTarget
    messages: tuple[tuple[str, str], ...] = ()
    episodes: tuple[str, ...] = ()
    new_episodes: tuple[str, ...] = ()
    rewritten: int = 0
    image_memories: tuple[str, ...] = ()
    pictures: tuple[ForgottenPicture, ...] = ()
    images: tuple[str, ...] = ()
    scenes: tuple[str, ...] = ()
    new_scenes: tuple[str, ...] = ()
    rewritten_scenes: int = 0
    profiles: tuple[str, ...] = ()
    profile_items: tuple[str, ...] = ()
    profile_item_count: int = 0
    stale_profile_items: tuple[str, ...] = ()
    group_profile_items: tuple[str, ...] = ()
    stale_group_profile_items: tuple[str, ...] = ()


@dataclass(frozen=True)
class UndoResult:
    """What one undo put back.

    ``applied`` is ``False`` when there was nothing to undo: the forget was
    undone already, or (an undo by target) there was none.

    Attributes:
        forget_id: The forget reversed; ``""`` when there was none.
        applied: Whether anything was restored.
        target: What the forget had addressed.
        messages: Messages no longer forgotten by it.
        episodes: Episode versions current again (marks cleared, or the
            previous versions written again).
        image_memories: Picture memories current again.
        pictures: The same, with the name each arrival carried.
        images: Picture files back in circulation.
        scenes: Scene versions current again.
        profiles: People whose profile document came back.
        profile_items: Profile items put back.
        profile_item_count: How many items came back (a replaced value is
            history, not an item).
        group_profile_items: Group-profile items put back.
        retired: The versions the forget had written, now replaced.
    """

    forget_id: str
    applied: bool
    target: ForgetTarget | None = None
    messages: tuple[tuple[str, str], ...] = ()
    episodes: tuple[str, ...] = ()
    image_memories: tuple[str, ...] = ()
    pictures: tuple[ForgottenPicture, ...] = ()
    images: tuple[str, ...] = ()
    scenes: tuple[str, ...] = ()
    profiles: tuple[str, ...] = ()
    profile_items: tuple[str, ...] = ()
    profile_item_count: int = 0
    group_profile_items: tuple[str, ...] = ()
    retired: tuple[str, ...] = ()

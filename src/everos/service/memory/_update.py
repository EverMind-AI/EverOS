"""A person's change of one memory: the use case behind ``POST /memory/update``.

Nothing is edited in place. A change writes a **new version** that names the
one it replaced (``supersedes``), who wrote it (``edited_by``) and when
(``edited_at``), and the old version is marked superseded by it -- the same
link every other change of a memory writes, so a read shows only the new
version and ``GET /memory/history`` shows both:

* an episode, a picture memory, a scene version: a new entry in today's daily
  file, the old entry's ``deprecated_entries`` mark naming it; the cascade
  embeds and indexes the new entry and turns the mark into the old row's
  ``deprecated_by`` (both files are handed to it at once);
* a profile item of ``user.md`` or of a member's profile in a group: a new
  item (a fresh ``item_id``) with ``supersedes`` / ``since``, the old one moved
  to the profile's history with ``superseded_by`` -- what a model's new value
  writes (:func:`~everos.memory.extract.pipeline.profile_items.edit_item`).

What the version rests on stays: its time, conversation, segment, speakers and
pictures are the old version's, so forgetting a source still reaches it.

**Optimistic concurrency.** The caller names the version it read; when another
change wrote a newer one since, nothing is written and the answer is 409
``VERSION_CONFLICT`` naming the current version. A forgotten memory cannot be
changed (409 ``MEMORY_FORGOTTEN``): undo the forget first.

**Held.** A person's version is held unless the caller says ``hold = false``:
the model's later updates do not change it -- the profile update keeps a held
item as it is, a scene update writes no narrative over a held version -- until
a caller releases it. A release of a daily-log version writes a new version
with the same text (nothing in a daily log is edited in place); a release of a
profile item is no new value and changes the item's flag only.

One change of one owner at a time: a user's under the lock their forgets
take, a group's under the group's forget lock (after the edit rule: owners and
admins may change any record of the group, a member those resting on their own
messages alone).
"""

from __future__ import annotations

__all__ = ["MemoryChanges", "UpdateManager", "UpdateResult"]

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any

from everos.core.errors import (
    ForbiddenError,
    ForgottenMemoryError,
    InvalidInputError,
    NotFoundError,
    VersionConflictError,
)
from everos.core.observability.logging import get_logger
from everos.memory import forget_lock, profile_lock, scenes_lock
from everos.memory.access import authorize_edit
from everos.memory.extract.pipeline import profile_items
from everos.memory.extract.scene import scene_cluster_kind
from everos.memory.models import (
    FORGOTTEN_MARK,
    Episode,
    ImageMemory,
    MemberProfile,
    Profile,
    Roster,
    Scene,
)
from everos.memory.ports import NO_SOURCE_CHECK, Scope, Stored

from ._records import (
    RecordKind,
    entry_id_of,
    episode_end,
    kind_of,
    memory_id,
    picture_end,
    scene_chain,
)

if TYPE_CHECKING:
    from everos.memory.access import Identity
    from everos.memory.deps import Stores
    from everos.memory.ports import IndexEraser

logger = get_logger("everos.service.memory")


@dataclass(frozen=True)
class MemoryChanges:
    """What a person changes; ``None`` keeps the stored text.

    Which fields a record takes depends on its kind: an episode or a scene
    its ``subject`` (title), ``summary`` and ``content`` (body, narrative); a
    picture memory its ``description`` and ``ocr``; a profile item its
    ``description``.
    """

    subject: str | None = None
    summary: str | None = None
    content: str | None = None
    description: str | None = None
    ocr: str | None = None

    def given(self) -> set[str]:
        """The fields the caller set."""
        return {f.name for f in fields(self) if getattr(self, f.name) is not None}


_ACCEPTS: Mapping[RecordKind, frozenset[str]] = {
    "episode": frozenset({"subject", "summary", "content"}),
    "scene": frozenset({"subject", "summary", "content"}),
    "image_memory": frozenset({"description", "ocr"}),
    "profile_item": frozenset({"description"}),
}


@dataclass(frozen=True)
class UpdateResult:
    """What one update did.

    Attributes:
        kind: ``episode``, ``image_memory``, ``scene`` or ``profile_item``.
        id: The current version after the call (the new one, when one was
            written).
        supersedes: The version the new one replaced; ``None`` when nothing
            new was written (a release of a profile item, or no change).
        changed: Whether anything was written.
        held: Whether the current version is held against the model.
        edited_by: Who wrote the current version (``None`` for a model).
        edited_at: When (epoch ms).
        member_id: The member whose profile holds the item, in a group.
    """

    kind: RecordKind
    id: str
    supersedes: str | None
    changed: bool
    held: bool
    edited_by: str | None
    edited_at: int | None
    member_id: str | None = None


class UpdateManager:
    """Writes a person's new version of one memory.

    Args:
        stores: Markdown and system state.
        eraser: Hands the files a change wrote to the cascade at once.
        clock: Epoch ms now.
    """

    def __init__(
        self, *, stores: Stores, eraser: IndexEraser, clock: Callable[[], int]
    ) -> None:
        self._stores = stores
        self._eraser = eraser
        self._clock = clock

    async def update(
        self,
        *,
        scope: Scope,
        identity: Identity,
        record_id: str,
        changes: MemoryChanges,
        hold: bool = True,
        expected_version: str | None = None,
    ) -> UpdateResult:
        """Write a new version of ``record_id`` (or change its hold).

        Raises:
            InvalidInputError: The id names no record kind, or a field the kind
                does not take; no asker for a group.
            NotFoundError: No such record (404).
            ForbiddenError: A group asker who may not change it (403).
            VersionConflictError: ``expected_version`` (by default the id
                given) is not the current version (409).
            ForgottenMemoryError: The memory is forgotten (409).
        """
        kind, entry_id = kind_of(scope, record_id)
        if kind == "group_profile_item":
            raise InvalidInputError(
                "a group-profile item changes through the group profile's own "
                "update; forget it instead"
            )
        wrong = changes.given() - _ACCEPTS[kind]
        if wrong:
            raise InvalidInputError(
                f"a {kind} takes {sorted(_ACCEPTS[kind])}, not {sorted(wrong)}"
            )
        expected = entry_id_of(scope.owner_id, expected_version or entry_id)
        step = _Step(
            scope=scope,
            entry_id=entry_id,
            expected=expected,
            changes=changes,
            hold=hold,
            editor=identity.asker_id or scope.owner_id,
            at=self._clock(),
        )
        if scope.owner_type != "group":
            async with forget_lock(scope):
                result = await self._apply(kind, step, roster=None)
        else:
            if identity.asker_id is None:
                raise InvalidInputError("asker_id is required to change a group record")
            async with forget_lock(scope):
                record = await self._stores.groups.get(scope)
                if record is None or not record.roster.is_current_member(
                    identity.asker_id
                ):
                    raise ForbiddenError(
                        f"the asker is not a current member of group {scope.owner_id!r}"
                    )
                result = await self._apply(kind, step, roster=record.roster)
        logger.info(
            "memory.updated",
            app_id=scope.app_id,
            project_id=scope.project_id,
            owner_id=scope.owner_id,
            owner_type=scope.owner_type,
            kind=kind,
            id=result.id,
            supersedes=result.supersedes,
            changed=result.changed,
            held=result.held,
        )
        return result

    async def _apply(
        self, kind: RecordKind, step: _Step, *, roster: Roster | None
    ) -> UpdateResult:
        if kind == "episode":
            return await self._episode(step, roster)
        if kind == "image_memory":
            return await self._picture(step, roster)
        if kind == "scene":
            async with scenes_lock(step.scope):
                return await self._scene(step, roster)
        if step.scope.owner_type == "group":
            return await self._member_item(step, roster)
        return await self._profile_item(step)

    # ── daily-log records ────────────────────────────────────────────────

    async def _episode(self, step: _Step, roster: Roster | None) -> UpdateResult:
        stores, scope = self._stores, step.scope
        current = _current(step, await episode_end(stores, scope, step.entry_id))
        if roster is not None:
            authorize_edit(roster, step.editor, sources=current.item.sender_ids)
        old = current.item
        subject = _or(step.changes.subject, old.subject)
        summary = _or(step.changes.summary, old.summary)
        body = _or(step.changes.content, old.episode)
        if (subject, summary, body) == (old.subject, old.summary, old.episode) and (
            step.hold == old.held
        ):
            return _unchanged("episode", scope, current.ref.entry_id, old)
        new = old.model_copy(
            update={
                "subject": subject,
                "summary": summary,
                "episode": body,
                # The references follow the text: a picture the person took
                # out of the body is no longer one the version refers to.
                "image_ids": [i for i in old.image_ids if i in body or i in summary],
                **_edit(step, current.ref.entry_id),
            }
        )
        ref = await stores.episode.append(scope, new, only_if=NO_SOURCE_CHECK)
        if ref is None:  # NO_SOURCE_CHECK never skips; a store bug otherwise
            raise NotFoundError("the new version could not be written")
        await stores.episode.deprecate(
            scope,
            entries=[(current.ref.md_path, current.ref.entry_id)],
            replaced_by=ref.entry_id,
            only_if=NO_SOURCE_CHECK,
        )
        await self._eraser.requeue("episode", {current.ref.md_path, ref.md_path})
        if scope.owner_type == "group":
            await self._follow_in_scene_cluster(
                scope, current.ref.entry_id, ref.entry_id
            )
        return _written("episode", scope, current.ref.entry_id, ref.entry_id, step)

    async def _picture(self, step: _Step, roster: Roster | None) -> UpdateResult:
        stores, scope = self._stores, step.scope
        current = _current(step, await picture_end(stores, scope, step.entry_id))
        if roster is not None:
            authorize_edit(roster, step.editor, sources=current.item.sender_ids)
        old = current.item
        description = _or(step.changes.description, old.description)
        ocr = _or(step.changes.ocr, old.ocr)
        if (description, ocr) == (old.description, old.ocr) and step.hold == old.held:
            return _unchanged("image_memory", scope, current.ref.entry_id, old)
        new = old.model_copy(
            update={
                "description": description,
                "ocr": ocr,
                "entry_id": None,
                "md_path": None,
                "status": "current",
                "superseded_by": None,
                **_edit(step, current.ref.entry_id),
            }
        )
        ref = await stores.image_memory.append(scope, new, only_if=NO_SOURCE_CHECK)
        if ref is None:
            raise NotFoundError("the new version could not be written")
        await stores.image_memory.deprecate(
            scope,
            entries=[(current.ref.md_path, current.ref.entry_id)],
            replaced_by=ref.entry_id,
            only_if=NO_SOURCE_CHECK,
        )
        await self._eraser.requeue("image_memory", {current.ref.md_path, ref.md_path})
        return _written("image_memory", scope, current.ref.entry_id, ref.entry_id, step)

    async def _scene(self, step: _Step, roster: Roster | None) -> UpdateResult:
        stores, scope = self._stores, step.scope
        chain = scene_chain(step.entry_id, await stores.scene.read_all(scope))
        newest = chain.newest
        if newest is None:
            raise NotFoundError(f"no scene {memory_id(scope, step.entry_id)!r}")
        _check(step, newest.stored, newest.mark)
        if roster is not None:
            authorize_edit(roster, step.editor, sources=newest.stored.item.sender_ids)
        old = newest.stored.item
        subject = _or(step.changes.subject, old.subject)
        summary = _or(step.changes.summary, old.summary)
        narrative = _or(step.changes.content, old.narrative)
        if (subject, summary, narrative) == (
            old.subject,
            old.summary,
            old.narrative,
        ) and step.hold == old.held:
            return _unchanged("scene", scope, newest.entry_id, old)
        new = old.model_copy(
            update={
                "version": old.version + 1,
                "subject": subject,
                "summary": summary,
                "narrative": narrative,
                **_edit(step, newest.entry_id),
            }
        )
        ref = await stores.scene.append(
            scope, new, supersedes=newest.stored, only_if=NO_SOURCE_CHECK
        )
        if ref is None:
            raise NotFoundError("the new version could not be written")
        await self._eraser.requeue("scene", {newest.stored.ref.md_path, ref.md_path})
        return _written("scene", scope, newest.entry_id, ref.entry_id, step)

    async def _follow_in_scene_cluster(
        self, scope: Scope, old_id: str, new_id: str
    ) -> None:
        """A group scene's cluster names the new version of its episode.

        A scene reads its members through their marks, so its narrative
        follows the new version either way; the cluster names it as well, so
        a forget inside the group that takes the new version out takes it out
        of the cluster too (section 9.4).
        """
        clusters = self._stores.cluster
        kind = scene_cluster_kind(scope)
        async with scenes_lock(scope):
            cluster_id = await clusters.find_cluster_for_member(
                scope, "episode", old_id, kind=kind
            )
            cluster = await clusters.get(cluster_id) if cluster_id else None
            if cluster is None or cluster.id is None:
                return
            members = [new_id if m == old_id else m for m in cluster.members]
            await clusters.delete(cluster.id)
            await clusters.upsert(
                scope,
                kind,
                cluster.model_copy(update={"members": members}),
                member_type="episode",
            )

    # ── profile items ────────────────────────────────────────────────────

    async def _profile_item(self, step: _Step) -> UpdateResult:
        scope = step.scope
        async with profile_lock(scope):
            profile = await self._stores.profile.read(scope)
            history = profile.history if profile is not None else []
            current_id = await self._current_item(step, profile, history)
            assert profile is not None  # a current item has a profile
            same = _same_item(step, profile, current_id)
            if same is not None:
                return same
            written, new_history, held_id = self._edit_item(
                step, profile, history, current_id
            )
            written.history = new_history
            ref = await self._stores.profile.write(scope, written)
        await self._eraser.requeue("user_profile", {ref.md_path})
        return _item_result(step, current_id, held_id, written, member_id=None)

    async def _member_item(self, step: _Step, roster: Roster | None) -> UpdateResult:
        scope = step.scope
        member_id = await self._member_holding(scope, step.entry_id)
        if member_id is None:
            raise NotFoundError(f"no profile item {step.entry_id!r} in the group")
        if roster is not None:
            authorize_edit(roster, step.editor, sources=[member_id])
        async with profile_lock(scope, member_id):
            held = await self._stores.member_profile.read(scope, member_id)
            if held is None:
                raise NotFoundError(f"no profile item {step.entry_id!r} in the group")
            current_id = await self._current_item(step, held.profile, held.history)
            same = _same_item(step, held.profile, current_id, member_id=member_id)
            if same is not None:
                return same
            written, new_history, held_id = self._edit_item(
                step, held.profile, held.history, current_id
            )
            ref = await self._stores.member_profile.write(
                scope,
                MemberProfile(
                    group_id=held.group_id,
                    member_id=member_id,
                    profile=written,
                    pending_memcell_ids=held.pending_memcell_ids,
                    history=new_history,
                ),
            )
        await self._eraser.requeue("group_member_profile", {ref.md_path})
        return _item_result(step, current_id, held_id, written, member_id=member_id)

    def _edit_item(
        self,
        step: _Step,
        profile: Profile,
        history: Sequence[Mapping[str, Any]],
        current_id: str,
    ) -> tuple[Profile, list[dict[str, Any]], str]:
        written, new_history, new_id = profile_items.edit_item(
            profile,
            history,
            current_id,
            description=step.changes.description,
            hold=step.hold,
            edited_by=step.editor,
            at=step.at,
        )
        if new_id != current_id:
            new_history = profile_items.bounded_history(
                new_history[:1], new_history[1:], written
            )
        return written, new_history, new_id

    async def _current_item(
        self,
        step: _Step,
        profile: Profile | None,
        history: Sequence[Mapping[str, Any]],
    ) -> str:
        """The current item of the value ``step`` names, checked against its
        expected version.

        A replaced value leads to its current value through ``superseded_by``;
        one retired or merged (no successor that is current) has none.
        """
        found = profile_items.find_item(profile, history, step.entry_id)
        if found is None:
            if step.scope.owner_type != "group" and await self._forgotten_item(
                step.scope, step.entry_id
            ):
                raise ForgottenMemoryError(
                    f"profile item {step.entry_id!r} is forgotten; undo the "
                    "forget before changing it"
                )
            raise NotFoundError(f"no profile item {step.entry_id!r}")
        bucket, entry = found
        seen = {step.entry_id}
        while bucket == "history":
            successor = entry.get("superseded_by")
            if not isinstance(successor, str) or successor in seen:
                raise VersionConflictError(
                    f"profile item {step.entry_id!r} was retired and has no "
                    "current version",
                    current=None,
                )
            seen.add(successor)
            nxt = profile_items.find_item(profile, history, successor)
            if nxt is None:
                raise VersionConflictError(
                    f"profile item {step.entry_id!r} has no current version",
                    current=None,
                )
            bucket, entry = nxt
        current_id = str(entry.get("item_id"))
        if current_id != step.expected:
            raise VersionConflictError(
                f"version conflict: the current version of {step.entry_id!r} is "
                f"{current_id!r}",
                current=current_id,
            )
        return current_id

    async def _forgotten_item(self, scope: Scope, item_id: str) -> bool:
        """Whether a live forget of the user took the item out of ``user.md``."""
        for entry in await self._stores.forget_log.live_entries(
            app_id=scope.app_id,
            project_id=scope.project_id,
            owner_id=scope.owner_id,
            owner_type=scope.owner_type,
        ):
            for _bucket, item in entry.removed_profile_items:
                if item.get("item_id") == item_id:
                    return True
        return False

    async def _member_holding(self, scope: Scope, item_id: str) -> str | None:
        """The member whose profile in the group holds ``item_id`` (or held it)."""
        for member_id in await self._stores.member_profile.members(scope):
            held = await self._stores.member_profile.read(scope, member_id)
            if held is not None and profile_items.find_item(
                held.profile, held.history, item_id
            ):
                return member_id
        return None


# ── helpers ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _Step:
    """One update as the steps take it."""

    scope: Scope
    entry_id: str
    expected: str
    changes: MemoryChanges
    hold: bool
    editor: str
    at: int


def _or(value: str | None, stored: str) -> str:
    return stored if value is None else value


def _edit(step: _Step, replaced: str) -> dict[str, object]:
    """The edit fields of the new version (``EDIT_FIELDS``)."""
    return {
        "supersedes": replaced,
        "edited_by": step.editor,
        "edited_at": step.at,
        "held": step.hold,
    }


def _check(step: _Step, stored: Stored[Any], mark: str | None) -> None:
    """Raise unless ``stored`` is the current version ``step`` expects."""
    shown = memory_id(step.scope, step.entry_id)
    if mark == FORGOTTEN_MARK:
        raise ForgottenMemoryError(
            f"{shown!r} is forgotten; undo the forget before changing it"
        )
    if mark is not None:
        raise VersionConflictError(f"{shown!r} has no current version", current=None)
    if stored.ref.entry_id != step.expected:
        current = memory_id(step.scope, stored.ref.entry_id)
        raise VersionConflictError(
            f"version conflict: the current version of {shown!r} is {current!r}",
            current=current,
        )


def _current[T](step: _Step, end: tuple[Stored[T], str | None] | None) -> Stored[T]:
    """The current version at the end of ``step``'s chain, checked."""
    if end is None:
        raise NotFoundError(f"no memory {memory_id(step.scope, step.entry_id)!r}")
    stored, mark = end
    _check(step, stored, mark)
    return stored


def _unchanged(
    kind: RecordKind, scope: Scope, entry_id: str, item: Episode | ImageMemory | Scene
) -> UpdateResult:
    return UpdateResult(
        kind=kind,
        id=memory_id(scope, entry_id),
        supersedes=None,
        changed=False,
        held=item.held,
        edited_by=item.edited_by,
        edited_at=item.edited_at,
    )


def _written(
    kind: RecordKind, scope: Scope, old_id: str, new_id: str, step: _Step
) -> UpdateResult:
    return UpdateResult(
        kind=kind,
        id=memory_id(scope, new_id),
        supersedes=memory_id(scope, old_id),
        changed=True,
        held=step.hold,
        edited_by=step.editor,
        edited_at=step.at,
    )


def _same_item(
    step: _Step, profile: Profile, item_id: str, *, member_id: str | None = None
) -> UpdateResult | None:
    """The answer for an update that changes nothing of the item, else ``None``."""
    found = profile_items.find_item(profile, [], item_id)
    if found is None:
        return None
    item = found[1]
    same_text = step.changes.description in (None, item.get("description"))
    if not same_text or step.hold != (item.get("held") is True):
        return None
    return UpdateResult(
        kind="profile_item",
        id=item_id,
        supersedes=None,
        changed=False,
        held=step.hold,
        edited_by=item.get("edited_by"),
        edited_at=item.get("edited_at"),
        member_id=member_id,
    )


def _item_result(
    step: _Step,
    old_id: str,
    new_id: str,
    profile: Profile,
    *,
    member_id: str | None,
) -> UpdateResult:
    found = profile_items.find_item(profile, [], new_id)
    item = found[1] if found is not None else {}
    return UpdateResult(
        kind="profile_item",
        id=new_id,
        supersedes=old_id if new_id != old_id else None,
        changed=True,
        held=bool(item.get("held")),
        edited_by=item.get("edited_by"),
        edited_at=item.get("edited_at"),
        member_id=member_id,
    )

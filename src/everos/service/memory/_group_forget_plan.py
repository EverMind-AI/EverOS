"""What a forget inside a group touches: its target resolved, its propagation planned.

docs/group-memory.md, sections 5.5 and 9.4. A plan is reads only -- no model
call, no write -- so a ``dry_run`` answers with exactly the plan the real run
then carries out. Five targets resolve to what they address and to whose
messages it rests on (the access policy's ``sources``):

* ``message`` -- messages of one session of the group, by id (the client's
  id when the batch sent one): their senders;
* ``member`` -- every message the member sent in the group, and the member's
  profile in it: the member;
* ``session`` -- every message of one of the group's sessions: nobody (owners
  and admins only);
* ``record`` -- episodes, picture memories, scenes, group-profile items and
  member-profile items by id: the speakers each rests on;
* ``image`` -- the group's picture memories of one content hash: their
  posters.

Propagation (the plan's fields):

* a segment holding a newly forgotten message is read again without every
  forgotten message (and without the group's forgotten pictures): its episode
  is **rewritten** when text is left, **dropped** when none is;
* picture memories of a picture that only forgotten messages carried go, and
  so do the record and image targets' picture memories; a picture whose last
  group memory goes is **gone** for the group, and its file is released when
  no live memory of any owner refers to it;
* a scene version that holds a rewritten or dropped episode goes; the scene
  gets a new version over its remaining members when two or more remain;
* the **source rule** for member-in-group and group-profile items: an item
  all of whose sources are forgotten is forgotten; one with sources left is
  **stale** -- it goes too (it may quote what was forgotten) and its remaining
  segments are queued on the profile's next update, which writes it again
  from them.
"""

from __future__ import annotations

__all__ = [
    "ForgetPlan",
    "ForgetPlanner",
    "GroupForgetTarget",
    "MemberEdit",
    "Rewrite",
    "SceneEdit",
    "member_edit",
]

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from everalgo.types import ChatMessage

from everos.core.errors import NotFoundError
from everos.memory.extract.pipeline import profile_items
from everos.memory.extract.pipeline.member_profile import wrote_in
from everos.memory.extract.pipeline.pictures import image_ids_in, without_pictures
from everos.memory.groups import without_forgotten
from everos.memory.models import (
    Episode,
    GroupProfileItem,
    ImageMemory,
    MemCell,
    Profile,
    Scene,
)
from everos.memory.ports import (
    Scope,
    SessionKey,
    Stored,
    all_of,
    eq,
    is_null,
)

if TYPE_CHECKING:
    from everos.memory.deps import Indexes, Stores
    from everos.memory.ports import GroupRecord

type TargetKind = Literal["message", "member", "session", "record", "image"]

_RECORD_PREFIXES: Mapping[str, str] = {
    "gep_": "episode",
    "gim_": "image_memory",
    "gsc_": "scene",
    "gp_": "group_profile",
    "pi_": "member_profile",
}
"""Entry-id prefixes of the group's records, and the kind each names."""

_REFERENCE_SCAN = 1_000
"""Live picture memories (any owner) one file-release check reads."""


@dataclass(frozen=True)
class GroupForgetTarget:
    """What a forget inside a group addresses (section 5.5).

    Exactly the fields of its ``kind`` are set; the request DTO validates
    that.
    """

    kind: TargetKind
    session_id: str | None = None
    message_ids: tuple[str, ...] = ()
    member_id: str | None = None
    record_ids: tuple[str, ...] = ()
    image_id: str | None = None

    def label(self) -> str:
        """The forget log's ``target_id`` for this target."""
        if self.kind == "message":
            return f"{self.session_id}:{','.join(self.message_ids)}"[:500]
        if self.kind == "member":
            return self.member_id or ""
        if self.kind == "session":
            return self.session_id or ""
        if self.kind == "image":
            return self.image_id or ""
        return ",".join(self.record_ids)[:500]

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "session_id": self.session_id,
            "message_ids": list(self.message_ids),
            "member_id": self.member_id,
            "record_ids": list(self.record_ids),
            "image_id": self.image_id,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> GroupForgetTarget:
        return cls(
            kind=data["kind"],
            session_id=data.get("session_id"),
            message_ids=tuple(data.get("message_ids") or ()),
            member_id=data.get("member_id"),
            record_ids=tuple(data.get("record_ids") or ()),
            image_id=data.get("image_id"),
        )


@dataclass(frozen=True)
class Rewrite:
    """One episode version a forget writes again, from its segment without it.

    Attributes:
        old: The current version, which the forget takes out.
        session_id: The segment's session.
        memcell_id: The segment.
        cell: The segment as it is read now: without every forgotten message
            and picture.
    """

    old: Stored[Episode]
    session_id: str
    memcell_id: str
    cell: MemCell


@dataclass(frozen=True)
class SceneEdit:
    """One scene a forget changes.

    Attributes:
        scene_id: The scene.
        versions: Every version that holds a changed episode; all go.
        members: The member entry ids of the new version, oldest first, with
            a rewritten episode's old id standing for its new version;
            empty when no new version is written.
        newest_kept: The newest version that stays (it holds no changed
            episode), which the new version supersedes; ``None`` if none.
        whole: The scene itself is the target: its cluster goes too.
    """

    scene_id: str
    versions: tuple[Stored[Scene], ...]
    members: tuple[str, ...] = ()
    newest_kept: Stored[Scene] | None = None
    whole: bool = False


@dataclass
class MemberEdit:
    """What a forget changes in one member's profile in the group.

    Attributes:
        member_id: The member.
        forgotten: Item ids forgotten (all their sources are), and the
            replaced values a record forget names.
        stale: Item ids that go and are written again from ``queue``.
        history: Positions in the document's history (replaced items) of the
            entries that go: they may say what was forgotten as well.
        queue: Segments to queue on the member's next update.
    """

    member_id: str
    forgotten: list[str] = field(default_factory=list)
    stale: list[str] = field(default_factory=list)
    history: list[int] = field(default_factory=list)
    queue: list[str] = field(default_factory=list)

    @property
    def changes(self) -> bool:
        return bool(self.forgotten or self.stale or self.history or self.queue)


@dataclass
class ForgetPlan:
    """Everything one forget inside a group touches (section 9.4).

    Attributes:
        scope: The group.
        target: What the forget addressed.
        sources: The members the target rests on (the access policy's input).
        messages: The target's messages, per session: what ``message_forget``
            records.
        rewrites: Episode versions written again without what was forgotten.
        dropped: Episode versions that go without a successor (no text left,
            or the record target).
        image_memories: Picture memories that go.
        pictures: Pictures whose last group memory goes.
        files: Picture files taken out of circulation (no live memory of any
            owner refers to them afterwards).
        hidden_pictures: Pictures every re-extraction leaves out (the group's
            live picture forgets and this one).
        scenes: The scenes that change.
        members_removed: Members whose whole profile in the group goes.
        members: Item changes per member profile.
        group_forgotten: Group-profile items that go with all sources gone.
        group_stale: Group-profile items that go and are written again.
        group_queue: Segments queued on the group profile's next update.
        already: The sessions' forgotten messages before this forget.
        readable: The segments that lost messages, as they read afterwards.
        pictures_hidden_before: ``hidden_pictures`` without this forget's.
        record_items: Member-profile item ids the record target names.
    """

    scope: Scope
    target: GroupForgetTarget
    sources: set[str] = field(default_factory=set)
    messages: dict[SessionKey, list[str]] = field(default_factory=dict)
    rewrites: list[Rewrite] = field(default_factory=list)
    dropped: list[Stored[Episode]] = field(default_factory=list)
    image_memories: list[Stored[ImageMemory]] = field(default_factory=list)
    pictures: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    hidden_pictures: set[str] = field(default_factory=set)
    scenes: list[SceneEdit] = field(default_factory=list)
    members_removed: list[str] = field(default_factory=list)
    members: list[MemberEdit] = field(default_factory=list)
    group_forgotten: list[Stored[GroupProfileItem]] = field(default_factory=list)
    group_stale: list[Stored[GroupProfileItem]] = field(default_factory=list)
    group_queue: list[str] = field(default_factory=list)
    already: dict[SessionKey, frozenset[str]] = field(default_factory=dict)
    readable: dict[str, MemCell] = field(default_factory=dict)
    record_items: set[str] = field(default_factory=set)

    @property
    def sessions(self) -> list[str]:
        """The sessions whose messages the forget names."""
        return [key.session_id for key in self.messages]


@dataclass
class _Messages:
    """The messages of the sessions a target reads: where each is, who sent it."""

    cells: dict[str, MemCell] = field(default_factory=dict)
    session_of: dict[str, SessionKey] = field(default_factory=dict)
    sender: dict[tuple[SessionKey, str], str] = field(default_factory=dict)
    segment: dict[tuple[SessionKey, str], str] = field(default_factory=dict)
    order: dict[SessionKey, list[str]] = field(default_factory=dict)


class ForgetPlanner:
    """Resolves a target and plans its propagation, reading only.

    Args:
        stores: Markdown and system state.
        indexes: The read side, for the one question markdown cannot answer
            cheaply: whether any other owner's memory refers to a picture.
            ``None`` keeps every picture file in circulation.
        scene_members: The live episodes a scene needs (``[scenes]
            min_members``) for a forget to write it again; ``None`` writes
            no scene again.
    """

    def __init__(
        self, *, stores: Stores, indexes: Indexes | None, scene_members: int | None
    ) -> None:
        self._stores = stores
        self._indexes = indexes
        self._scene_members = scene_members

    async def plan(
        self,
        scope: Scope,
        target: GroupForgetTarget,
        record: GroupRecord,
        *,
        already: Mapping[SessionKey, frozenset[str]] | None = None,
    ) -> ForgetPlan:
        """The plan of ``target`` in the group ``scope``.

        ``already`` are the sessions' forgotten messages as they were before
        this forget recorded its own: the real run plans again once it has
        recorded them (so that nothing cut meanwhile escapes it), and must
        still see its messages as new.

        Raises:
            NotFoundError: The target names a session, message, member,
                record or picture the group does not hold.
        """
        plan = ForgetPlan(scope=scope, target=target)
        if already is not None:
            plan.already = dict(already)
        hidden = await self._stores.forget_log.live_targets(
            target="image",
            app_id=scope.app_id,
            project_id=scope.project_id,
            owner_id=scope.owner_id,
            owner_type=scope.owner_type,
        )
        plan.hidden_pictures = set(hidden)
        if target.kind in ("message", "member", "session"):
            await self._plan_messages(plan, record)
        elif target.kind == "image":
            await self._plan_image(plan)
        else:
            await self._plan_records(plan)
        await self._plan_scenes(plan)
        await self._plan_files(plan)
        return plan

    # ── messages, a member, a session ────────────────────────────────────

    async def _plan_messages(self, plan: ForgetPlan, record: GroupRecord) -> None:
        target, scope = plan.target, plan.scope
        if target.kind == "member":
            assert target.member_id is not None
            if not record.roster.intervals_of(target.member_id):
                raise NotFoundError(f"no member {target.member_id!r} in the group")
            sessions = [
                SessionKey(scope.app_id, scope.project_id, s) for s in record.sessions
            ]
        else:
            assert target.session_id is not None
            session = SessionKey(scope.app_id, scope.project_id, target.session_id)
            if await self._stores.groups.binding(session) != scope.owner_id:
                raise NotFoundError(f"no session {target.session_id!r} in the group")
            sessions = [session]
        found = await self._messages(sessions)
        if target.kind == "message":
            (session,) = sessions
            missing = [
                m for m in target.message_ids if (session, m) not in found.sender
            ]
            if missing:
                raise NotFoundError(f"no message {missing[0]!r} in the session")
            chosen = {session: list(dict.fromkeys(target.message_ids))}
            plan.sources = {found.sender[(session, m)] for m in target.message_ids}
        elif target.kind == "member":
            chosen = {
                key: [m for m in ids if found.sender.get((key, m)) == target.member_id]
                for key, ids in found.order.items()
            }
            plan.sources = {target.member_id or ""}
        else:
            chosen = dict(found.order)
        plan.messages = {key: ids for key, ids in chosen.items() if ids}
        await self._propagate_messages(plan, found)
        if target.kind == "member":
            assert target.member_id is not None
            held = await self._stores.profile.read(scope, target.member_id)
            if held is not None:
                plan.members_removed.append(target.member_id)

    async def _messages(self, sessions: Sequence[SessionKey]) -> _Messages:
        """Every message of ``sessions``: in their segments, then in their buffers."""
        found = _Messages()
        for session in sessions:
            order = found.order.setdefault(session, [])
            cells = await self._stores.memcell.cells_in_session(session)
            for memcell_id, cell in cells.items():
                found.cells[memcell_id] = cell
                found.session_of[memcell_id] = session
                for item in cell.items:
                    if isinstance(item, ChatMessage):
                        found.sender[(session, item.id)] = item.sender_id
                        found.segment[(session, item.id)] = memcell_id
                        order.append(item.id)
            for message in await self._stores.session_buffer.list(session):
                message_id = message.client_message_id or message.message_id
                if (session, message_id) not in found.sender:
                    found.sender[(session, message_id)] = message.sender_id
                    order.append(message_id)
        for session, order in found.order.items():
            found.order[session] = list(dict.fromkeys(order))
        return found

    async def _propagate_messages(self, plan: ForgetPlan, found: _Messages) -> None:
        """Rewrites, drops, picture memories and profile items of new forgets."""
        scope = plan.scope
        live: dict[SessionKey, frozenset[str]] = {}
        for session in plan.messages:
            live[session] = (
                plan.already[session]
                if session in plan.already
                else await self._stores.message_forget.forgotten(session)
            )
        plan.already = live
        fresh: dict[str, set[str]] = {}
        for session, ids in plan.messages.items():
            for message_id in ids:
                if message_id in live[session]:
                    continue  # forgotten already; its effects were planned then
                segment = found.segment.get((session, message_id))
                if segment is not None:
                    fresh.setdefault(segment, set()).add(message_id)
        gone = {
            memcell_id: set(live[found.session_of[memcell_id]])
            | set(plan.messages[found.session_of[memcell_id]])
            for memcell_id in fresh
        }
        readable = {
            memcell_id: without_pictures(
                without_forgotten(found.cells[memcell_id], gone[memcell_id]),
                plan.hidden_pictures,
            )
            for memcell_id in fresh
        }
        plan.readable = readable
        current = await self._stores.episode.read_for_segments(scope, list(fresh))
        by_segment: dict[str, list[Stored[Episode]]] = {}
        for stored in current:
            by_segment.setdefault(stored.item.parent_id, []).append(stored)
        for memcell_id in fresh:
            episodes = sorted(
                by_segment.get(memcell_id, []), key=lambda s: s.ref.entry_id
            )
            if not episodes:
                continue
            if _has_text(readable[memcell_id]):
                plan.rewrites.append(
                    Rewrite(
                        old=episodes[-1],
                        session_id=found.session_of[memcell_id].session_id,
                        memcell_id=memcell_id,
                        cell=readable[memcell_id],
                    )
                )
                plan.dropped.extend(episodes[:-1])
            else:
                plan.dropped.extend(episodes)
        await self._pictures_of_forgotten(plan, found, fresh, readable)
        await self._plan_member_profiles(plan, readable, record_items=set())
        await self._plan_group_profile(plan, found, gone)

    async def _pictures_of_forgotten(
        self,
        plan: ForgetPlan,
        found: _Messages,
        fresh: Mapping[str, set[str]],
        readable: Mapping[str, MemCell],
    ) -> None:
        """Picture memories of pictures only forgotten messages carried."""
        if not fresh:
            return
        carried: dict[str, set[str]] = {}
        for memcell_id, ids in fresh.items():
            for item in found.cells[memcell_id].items:
                if isinstance(item, ChatMessage) and item.id in ids:
                    carried.setdefault(memcell_id, set()).update(
                        image_ids_in(str(item.content or ""))
                    )
        stored = await self._stores.image_memory.read_for_segments(
            plan.scope, list(carried)
        )
        for memory in stored:
            memcell_id = memory.item.parent_id
            if memory.item.superseded_by is not None:
                continue
            if memory.item.image_id not in carried.get(memcell_id, set()):
                continue
            still = {
                image_id
                for item in readable[memcell_id].items
                for image_id in image_ids_in(str(getattr(item, "content", "") or ""))
            }
            if memory.item.image_id not in still:
                plan.image_memories.append(memory)
        await self._pictures_gone(plan)

    # ── a picture ────────────────────────────────────────────────────────

    async def _plan_image(self, plan: ForgetPlan) -> None:
        image_id = plan.target.image_id
        assert image_id is not None
        occurrences = [
            row
            for row in await self._stores.image_memory.read_by_image(
                plan.scope, image_id
            )
            if row.item.superseded_by is None
        ]
        if not occurrences:
            raise NotFoundError(f"no picture {image_id!r} in the group")
        plan.sources = {s for row in occurrences for s in row.item.sender_ids}
        plan.image_memories.extend(occurrences)
        plan.hidden_pictures.add(image_id)
        referring = await self._stores.episode.read_referring(plan.scope, image_id)
        segments = {row.item.parent_id for row in referring}
        cells = await self._stores.memcell.cells_by_id(sorted(segments))
        forgotten = await self._stores.message_forget.forgotten_in_segments(
            sorted(segments)
        )
        for row in referring:
            cell = cells.get(row.item.parent_id)
            if cell is None or row.item.parent_type != "memcell":
                continue
            readable = without_pictures(
                without_forgotten(cell, forgotten.get(row.item.parent_id, ())),
                plan.hidden_pictures,
            )
            if _has_text(readable):
                plan.rewrites.append(
                    Rewrite(
                        old=row,
                        session_id=row.item.session_id or "",
                        memcell_id=row.item.parent_id,
                        cell=readable,
                    )
                )
            else:
                plan.dropped.append(row)
        await self._pictures_gone(plan)
        await self._plan_member_profiles(plan, {}, record_items=set())

    # ── records by id ─────────────────────────────────────────────────────

    async def _plan_records(self, plan: ForgetPlan) -> None:
        scope = plan.scope
        wanted: dict[str, list[str]] = {}
        for record_id in dict.fromkeys(plan.target.record_ids):
            entry_id = _entry_id(scope.owner_id, record_id)
            kind = next(
                (k for p, k in _RECORD_PREFIXES.items() if entry_id.startswith(p)),
                None,
            )
            if kind is None:
                raise NotFoundError(f"no record {record_id!r} in the group")
            wanted.setdefault(kind, []).append(entry_id)
        episodes = await self._stores.episode.read_many(
            scope, wanted.get("episode", [])
        )
        _all_found(wanted.get("episode", []), [e.ref.entry_id for e in episodes])
        plan.dropped.extend(episodes)
        plan.sources |= {s for e in episodes for s in e.item.sender_ids}
        pictures = [
            row
            for row in await self._stores.image_memory.read_many(
                scope, wanted.get("image_memory", [])
            )
            if row.item.superseded_by is None
        ]
        _all_found(wanted.get("image_memory", []), [p.ref.entry_id for p in pictures])
        plan.image_memories.extend(pictures)
        plan.sources |= {s for p in pictures for s in p.item.sender_ids}
        await self._pictures_gone(plan)
        if wanted.get("scene"):
            versions = await self._stores.scene.versions(scope)
            named = [v for v in versions if v.ref.entry_id in set(wanted["scene"])]
            _all_found(wanted["scene"], [v.ref.entry_id for v in named])
            for scene_id in dict.fromkeys(v.item.scene_id for v in named):
                all_versions = tuple(v for v in versions if v.item.scene_id == scene_id)
                plan.scenes.append(
                    SceneEdit(scene_id=scene_id, versions=all_versions, whole=True)
                )
                plan.sources |= {s for v in all_versions for s in v.item.sender_ids}
        if wanted.get("group_profile"):
            items = await self._stores.group_profile.items(scope)
            named_items = [
                i for i in items if i.ref.entry_id in set(wanted["group_profile"])
            ]
            _all_found(wanted["group_profile"], [i.ref.entry_id for i in named_items])
            plan.group_forgotten.extend(named_items)
            plan.sources |= {
                source.speaker_id for i in named_items for source in i.item.sources
            }
        record_items = set(wanted.get("member_profile", []))
        plan.record_items = record_items
        if record_items:
            owners = await self._item_owners(scope, record_items)
            _all_found(sorted(record_items), sorted(owners))
            plan.sources |= set(owners.values())
        await self._plan_member_profiles(plan, {}, record_items=record_items)

    async def _item_owners(
        self, scope: Scope, item_ids: Collection[str]
    ) -> dict[str, str]:
        """``item_id -> member_id`` of the member-profile items named.

        A replaced value counts: a read shows it under the item that replaced
        it (section 7.4), so it is forgotten by its id like a current item.
        """
        owners: dict[str, str] = {}
        for member_id in await self._stores.profile.people(scope):
            held = await self._stores.profile.read(scope, member_id)
            if held is None:
                continue
            for extra in (*(i.model_extra or {} for i in _items(held)), *held.history):
                item_id = extra.get("item_id")
                if item_id in item_ids:
                    owners[str(item_id)] = member_id
        return owners

    # ── pictures, files ──────────────────────────────────────────────────

    async def _pictures_gone(self, plan: ForgetPlan) -> None:
        """The pictures whose last live group memory this forget takes out."""
        leaving = {row.ref.entry_id for row in plan.image_memories}
        for image_id in dict.fromkeys(row.item.image_id for row in plan.image_memories):
            if image_id in plan.pictures:
                continue
            rows = await self._stores.image_memory.read_by_image(plan.scope, image_id)
            if not any(
                row.item.superseded_by is None and row.ref.entry_id not in leaving
                for row in rows
            ):
                plan.pictures.append(image_id)

    async def _plan_files(self, plan: ForgetPlan) -> None:
        """Release a picture file when no live memory of any owner refers to it."""
        if self._indexes is None:
            return
        leaving = {row.ref.id for row in plan.image_memories}
        for image_id in plan.pictures:
            hits = await self._indexes.image_memory.find(
                all_of(eq("image_id", image_id), is_null("deprecated_by")),
                limit=_REFERENCE_SCAN,
            )
            if all(hit.stored.ref.id in leaving for hit in hits):
                plan.files.append(image_id)

    # ── scenes ───────────────────────────────────────────────────────────

    async def _plan_scenes(self, plan: ForgetPlan) -> None:
        """Every scene version holding a changed episode goes; a new one follows."""
        changed = {r.old.ref.entry_id for r in plan.rewrites} | {
            e.ref.entry_id for e in plan.dropped
        }
        whole = {edit.scene_id for edit in plan.scenes}
        if not changed:
            return
        dropped = {e.ref.entry_id for e in plan.dropped}
        versions = await self._stores.scene.versions(plan.scope)
        by_scene: dict[str, list[Stored[Scene]]] = {}
        for version in versions:
            by_scene.setdefault(version.item.scene_id, []).append(version)
        for scene_id, all_versions in by_scene.items():
            if scene_id in whole:
                continue
            hit = tuple(v for v in all_versions if set(v.item.member_ids) & changed)
            if not hit:
                continue
            kept = [v for v in all_versions if v not in hit]
            newest = max(hit, key=lambda v: v.item.version)
            present = {
                e.ref.entry_id
                for e in await self._stores.episode.read_many(
                    plan.scope, newest.item.member_ids
                )
            }
            members = tuple(
                m for m in newest.item.member_ids if m not in dropped and m in present
            )
            enough = (
                self._scene_members is not None and len(members) >= self._scene_members
            )
            plan.scenes.append(
                SceneEdit(
                    scene_id=scene_id,
                    versions=hit,
                    members=members if enough else (),
                    newest_kept=max(kept, key=lambda v: v.item.version)
                    if kept
                    else None,
                )
            )

    # ── profiles ─────────────────────────────────────────────────────────

    async def _plan_member_profiles(
        self,
        plan: ForgetPlan,
        readable: Mapping[str, MemCell],
        *,
        record_items: set[str],
    ) -> None:
        """The source rule over every member-in-group profile of the group.

        ``readable`` holds the segments that lost messages, read as they are
        after the forget; a text item resting on one of them is forgotten
        when none of its segments holds a message of the member's any more,
        stale otherwise. A picture item is forgotten when every picture it
        rests on is gone.
        """
        scope = plan.scope
        pictures = set(plan.pictures)
        for member_id in await self._stores.profile.people(scope):
            if member_id in plan.members_removed:
                continue
            if plan.target.kind == "member" and plan.target.member_id == member_id:
                continue
            held = await self._stores.profile.read(scope, member_id)
            if held is None:
                continue
            edit = member_edit(held, readable, pictures, record_items)
            if edit.changes:
                plan.members.append(edit)

    async def _plan_group_profile(
        self,
        plan: ForgetPlan,
        found: _Messages,
        gone: Mapping[str, set[str]],
    ) -> None:
        """The source rule over the group-profile items citing a new forget."""
        if not gone:
            return
        fresh = {
            (memcell_id, message_id)
            for memcell_id, ids in gone.items()
            for message_id in ids
        }
        queued: list[str] = []
        for stored in await self._stores.group_profile.items(plan.scope):
            cited = [(s.memcell_id, s.message_id) for s in stored.item.sources]
            if not any(pair in fresh for pair in cited):
                continue
            remaining = [
                pair
                for pair in cited
                if pair not in fresh
                and not (pair[0] in gone and pair[1] in gone[pair[0]])
            ]
            if not remaining:
                plan.group_forgotten.append(stored)
                continue
            plan.group_stale.append(stored)
            if stored.item.status not in ("superseded", "forgotten"):
                queued.extend(memcell_id for memcell_id, _ in remaining)
        # A segment of another session is queued as it is; one of the forget's
        # own sessions only while a message of it is left.
        plan.group_queue = [
            m
            for m in dict.fromkeys(queued)
            if m not in found.cells or _has_text(_segment_of(found, m, gone))
        ]


def member_edit(
    held: Profile,
    readable: Mapping[str, MemCell],
    pictures: set[str],
    record_items: set[str],
) -> MemberEdit:
    """What the source rule does to one member profile.

    Applied twice: by the plan, and again under the member's lock by the run,
    over the document as it is then (an update may have written it since).
    """
    member_id = held.owner_id
    edit = MemberEdit(member_id=member_id)
    queue: list[str] = []
    for item in _items(held):
        extra = item.model_extra or {}
        item_id = str(extra.get("item_id") or "")
        if item_id and item_id in record_items:
            edit.forgotten.append(item_id)
            continue
        if profile_items.item_source_images(item):
            if profile_items.rests_on_pictures(item, pictures):
                edit.forgotten.append(item_id)
            continue
        sources = [str(s) for s in extra.get("source_memcell_ids") or []]
        if not set(sources) & set(readable):
            continue
        remaining = [
            s for s in sources if s not in readable or wrote_in(readable[s], member_id)
        ]
        if remaining:
            edit.stale.append(item_id)
            queue.extend(remaining)
        else:
            edit.forgotten.append(item_id)
    for position, old in enumerate(held.history):
        sources = {str(s) for s in old.get("source_memcell_ids") or []}
        named = bool(old.get("item_id")) and old.get("item_id") in record_items
        gone = profile_items.rests_on_pictures(old, pictures)
        if sources & set(readable) or gone or named:
            edit.history.append(position)
        if named:  # a replaced value a read listed, forgotten by its id
            edit.forgotten.append(str(old["item_id"]))
    edit.queue = [m for m in dict.fromkeys(queue) if m not in held.pending_memcell_ids]
    return edit


def _items(held: Profile) -> list[Any]:
    return [*held.explicit_info, *held.implicit_traits]


def _segment_of(
    found: _Messages, memcell_id: str, gone: Mapping[str, set[str]]
) -> MemCell:
    """A segment of the forget's sessions read without its forgotten messages."""
    cell = found.cells.get(memcell_id)
    if cell is None:
        return MemCell(items=[], timestamp=0)
    return without_forgotten(cell, gone.get(memcell_id, ()))


def _has_text(cell: MemCell) -> bool:
    return any(getattr(item, "kind", None) == "text" for item in cell.items)


def _entry_id(group_id: str, record_id: str) -> str:
    """A record's entry id from its row id (``<group>_<entry>``) or itself."""
    prefix = f"{group_id}_"
    return record_id[len(prefix) :] if record_id.startswith(prefix) else record_id


def _all_found(wanted: Sequence[str], found: Sequence[str]) -> None:
    missing = [w for w in wanted if w not in set(found)]
    if missing:
        raise NotFoundError(f"no record {missing[0]!r} in the group")

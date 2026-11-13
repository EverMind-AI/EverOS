"""What a forget touches: its target resolved, its propagation planned.

docs/api.md and docs/group-memory.md, sections 5.5 and 9.4. A plan reads only
-- no model call, no write -- so a dry run answers with exactly the plan the
real run carries out (:mod:`._forget`). One planner serves every owner type:
what differs follows from who reads the memory, its owner alone (a user) or
every member (a group).

Targets, resolved:

* ``record`` -- records by the ids a read returned: an episode or a picture
  memory by any version (its chain's current version goes), a scene (its
  current version; its cluster goes too), a profile item or a replaced value,
  a group-profile item;
* ``session`` -- a user's conversation: its current episodes and picture
  memories; a group's: every message of it (owners and admins only);
* ``message`` -- messages of one conversation by id (their senders, in a group);
* ``member`` -- every message one member sent in the group, and the member's
  profile in it;
* ``owner`` -- everything of a user; ``condition`` -- a user's memories of
  some kinds whose time falls in a range;
* ``image`` -- one picture by content hash: its memories, and the text that
  names it (a text edit, no model call).

Propagation:

* a segment that lost a message is read again without every forgotten message
  and picture: its episode is **rewritten** (re-extracted) when text is left,
  **marked** forgotten when none is; the picture memories of pictures only
  forgotten messages carried are marked;
* a picture's forget writes each episode and scene that names it again
  without the reference (:mod:`._forget_apply`);
* scenes follow their episodes -- a user's scene is forgotten once none of its
  episodes is alive; a group's scene that lost or changed an episode is
  written again over the ones left (or forgotten below ``[scenes]
  min_members``), since its narrative may say what was forgotten;
* profiles follow their sources (:func:`user_edit`, :func:`member_edit`): an
  item none of whose segments is remembered goes; after a message forget, one
  resting on a segment that lost a message goes too and is written again by
  the person's next update from the segments it has left (**stale**), and in
  a group so do the group-profile items citing a forgotten message;
* a picture file is taken out of circulation when no live memory of any
  owner refers to it afterwards;
* what the target reaches that another live forget already holds -- an entry
  marked forgotten, a profile value taken out -- is **covered**: logged, not
  marked again, so it stays forgotten while either forget stands
  (:mod:`._forget_log`).
"""

from __future__ import annotations

__all__ = [
    "ForgetPlan",
    "ForgetPlanner",
    "ProfileEdit",
    "Rewrite",
    "SceneEdit",
    "member_edit",
    "reaches_value",
    "user_edit",
]

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from everalgo.types import ChatMessage

from everos.core.errors import InvalidInputError, NotFoundError
from everos.memory.extract.pipeline import profile_items
from everos.memory.extract.pipeline.member_profile import wrote_in
from everos.memory.extract.pipeline.pictures import image_ids_in, without_pictures
from everos.memory.extract.scene import current_versions, newest_versions
from everos.memory.groups import without_forgotten
from everos.memory.models import (
    FORGOTTEN_MARK,
    Episode,
    GroupProfileItem,
    ImageMemory,
    MemCell,
    Profile,
    Scene,
)
from everos.memory.ports import Scope, SessionKey, Stored, all_of, eq, is_null

from ._forget_log import Logs, Removed, segments_of
from ._forget_models import ForgetTarget
from ._records import episode_end, kind_of, picture_end, scene_chain

if TYPE_CHECKING:
    from everos.memory.deps import Indexes, Stores
    from everos.memory.ports import GroupRecord

_REFERENCE_SCAN = 1_000
"""Live picture memories (any owner) one file-release check reads."""


@dataclass(frozen=True)
class Rewrite:
    """One episode version a forget writes again.

    Attributes:
        old: The current version, which the new one supersedes.
        session_id: The segment's session.
        memcell_id: The segment.
        cell: The segment as it reads now (without every forgotten message and
            picture): the input of its re-extraction. ``None`` for a text edit.
        image_id: The picture a text edit takes out of the text.
    """

    old: Stored[Episode]
    session_id: str
    memcell_id: str
    cell: MemCell | None = None
    image_id: str | None = None


@dataclass(frozen=True)
class SceneEdit:
    """One scene a forget changes.

    Attributes:
        scene_id: The scene.
        current: Its current version, which the forget marks forgotten or a
            new version supersedes; ``None`` when no current version changes.
        versions: In a group, every version that may say what was forgotten
            (they hold a changed episode, or the scene is the target): hidden
            from the group's history while the forget stands.
        members: The member entry ids of a re-narrated version, a rewritten
            episode's old id standing for its new version; empty when none is
            narrated.
        whole: The scene is the target: its cluster goes too.
        image_id: The picture a text edit takes out of the narrative.
    """

    scene_id: str
    current: Stored[Scene] | None
    versions: tuple[Stored[Scene], ...] = ()
    members: tuple[str, ...] = ()
    whole: bool = False
    image_id: str | None = None


@dataclass
class ProfileEdit:
    """What a forget takes out of one person's profile document.

    Values are chosen by place -- an item stored before items had ids has
    none -- and named by id in the answer.

    Attributes:
        person_id: The user, or the member in the group.
        items: ``(bucket, position)`` of the items taken out.
        history: Positions of the replaced values taken out.
        forgotten: Ids of the items forgotten (none of their sources is left),
            and of the replaced values a record target names.
        stale: Ids of the items taken out to be written again from ``queue``
            (a group's).
        history_ids: Ids of the replaced values taken out.
        queue: Segments queued on the person's next update.
        by_segments: The places (``(bucket, position)``, ``history`` for a
            replaced value) of what the segment rule took (a user's): what an
            undo of another forget may bring back.
        whole: Everything goes, the summary too.
    """

    person_id: str
    items: list[tuple[str, int]] = field(default_factory=list)
    history: list[int] = field(default_factory=list)
    forgotten: list[str] = field(default_factory=list)
    stale: list[str] = field(default_factory=list)
    history_ids: list[str] = field(default_factory=list)
    queue: list[str] = field(default_factory=list)
    by_segments: list[tuple[str, int]] = field(default_factory=list)
    whole: bool = False

    @property
    def changes(self) -> bool:
        return bool(self.items or self.history or self.queue or self.whole)

    def take(self, place: tuple[str, int], item: Any, *, stale: bool = False) -> None:
        self.items.append(place)
        item_id = (item.model_extra or {}).get("item_id")
        if isinstance(item_id, str):
            (self.stale if stale else self.forgotten).append(item_id)

    def take_history(self, position: int, entry: Mapping[str, Any]) -> None:
        self.history.append(position)
        if isinstance(entry.get("item_id"), str):
            self.history_ids.append(str(entry["item_id"]))


@dataclass
class ForgetPlan:
    """Everything one forget touches.

    Attributes:
        scope: The owner.
        target: What the forget addressed.
        sources: Whose messages the target rests on (a group's access check).
        messages: The target's messages per session: what ``message_forget``
            records (message-level targets).
        already: The sessions' forgotten messages before this forget.
        readable: The segments that lost messages, as they read afterwards.
        sessions: Conversations the run lets settle before it writes.
        episodes: Episode versions marked forgotten.
        rewrites: Episode versions written again.
        image_memories: Picture memories marked forgotten.
        pictures: Pictures whose last live memory of the owner goes.
        files: Picture files taken out of circulation.
        hidden_pictures: Pictures a re-extraction leaves out (the owner's live
            picture forgets).
        scenes: The scenes that change.
        orphaned: A user's scenes left without a live episode (a dry run's
            answer; the run marks them once the episodes are marked).
        segments: A user's segments this forget forgets whole.
        item_ids: Profile items and replaced values the target names.
        whole_profile: The user's whole profile goes (``owner``).
        members_removed: Members whose whole profile document in the group goes.
        profiles: The profile edits as the plan reads them (the run reads
            each document again under its lock).
        group_forgotten: Group-profile items forgotten.
        group_stale: Group-profile items taken out to be written again.
        group_queue: Segments queued on the group profile's next update.
        retired: In a group, per kind, the versions hidden from the group's
            history while the forget stands.
        held: Per kind, the entries the owner's live forgets keep forgotten,
            each with the mark it carried before (read once, by the plan).
        covered: Per kind, ``[md_path, entry_id, previous mark(, hash)]`` of
            the entries the target reaches that another live forget already
            holds: logged, not marked again.
        covered_values: The profile values other live forgets keep out of
            ``user.md`` that the target reaches (a user's).
    """

    scope: Scope
    target: ForgetTarget
    sources: set[str] = field(default_factory=set)
    messages: dict[SessionKey, list[str]] = field(default_factory=dict)
    already: dict[SessionKey, frozenset[str]] = field(default_factory=dict)
    readable: dict[str, MemCell] = field(default_factory=dict)
    sessions: list[str] = field(default_factory=list)
    episodes: list[Stored[Episode]] = field(default_factory=list)
    rewrites: list[Rewrite] = field(default_factory=list)
    image_memories: list[Stored[ImageMemory]] = field(default_factory=list)
    pictures: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    hidden_pictures: set[str] = field(default_factory=set)
    scenes: list[SceneEdit] = field(default_factory=list)
    orphaned: list[Stored[Scene]] = field(default_factory=list)
    segments: list[str] = field(default_factory=list)
    item_ids: set[str] = field(default_factory=set)
    whole_profile: bool = False
    members_removed: list[str] = field(default_factory=list)
    profiles: list[ProfileEdit] = field(default_factory=list)
    group_forgotten: list[Stored[GroupProfileItem]] = field(default_factory=list)
    group_stale: list[Stored[GroupProfileItem]] = field(default_factory=list)
    group_queue: list[str] = field(default_factory=list)
    retired: dict[str, set[str]] = field(default_factory=dict)
    held: dict[str, dict[str, Any]] = field(default_factory=dict)
    covered: dict[str, list[list[Any]]] = field(default_factory=dict)
    covered_values: list[Removed] = field(default_factory=list)

    @property
    def is_group(self) -> bool:
        return self.scope.owner_type == "group"

    @property
    def reextracts(self) -> bool:
        """Whether the run calls the model (a segment keeps text)."""
        return any(r.cell is not None for r in self.rewrites)

    def add_episode(self, stored: Stored[Episode]) -> None:
        if all(e.ref.entry_id != stored.ref.entry_id for e in self.episodes):
            self.episodes.append(stored)

    def add_picture(self, stored: Stored[ImageMemory]) -> None:
        if all(p.ref.entry_id != stored.ref.entry_id for p in self.image_memories):
            self.image_memories.append(stored)

    def retire(self, kind: str, entry_ids: Collection[str]) -> None:
        if self.is_group and entry_ids:
            self.retired.setdefault(kind, set()).update(entry_ids)

    def cover(
        self, kind: str, md_path: str, entry_id: str, image_id: str | None = None
    ) -> None:
        """Log an entry the target reaches, forgotten already, as covered
        while a live forget holds it (one of whose mark it is)."""
        held = self.held.get(kind, {})
        if entry_id not in held:
            return
        rows = self.covered.setdefault(kind, [])
        if any(row[1] == entry_id for row in rows):
            return
        row: list[Any] = [md_path, entry_id, held[entry_id]]
        if kind == "image_memory":
            row.append(image_id)
        rows.append(row)

    def cover_picture(self, stored: Stored[ImageMemory]) -> None:
        if stored.item.superseded_by == FORGOTTEN_MARK:
            self.cover(
                "image_memory",
                stored.ref.md_path,
                stored.ref.entry_id,
                stored.item.image_id,
            )


@dataclass
class _Messages:
    """The messages of the sessions a target reads: where each is, who sent it."""

    cells: dict[str, MemCell] = field(default_factory=dict)
    session_of: dict[str, SessionKey] = field(default_factory=dict)
    sender: dict[tuple[SessionKey, str], str] = field(default_factory=dict)
    segment: dict[tuple[SessionKey, str], str] = field(default_factory=dict)
    order: dict[SessionKey, list[str]] = field(default_factory=dict)
    people: dict[SessionKey, set[str]] = field(default_factory=dict)
    """The role ``user`` senders of each session."""


class ForgetPlanner:
    """Resolves a target and plans its propagation, reading only.

    Args:
        stores: Markdown and system state.
        indexes: The read side, for the one question markdown cannot answer
            cheaply: whether another owner's memory refers to a picture.
            ``None`` answers it from the owner's own files.
        scene_members: The live episodes a group's scene needs (``[scenes]
            min_members``) for a forget to write it again; ``None`` writes no
            scene again.
    """

    def __init__(
        self, *, stores: Stores, indexes: Indexes | None, scene_members: int | None
    ) -> None:
        self._stores = stores
        self._indexes = indexes
        self._scene_members = scene_members
        self._logs = Logs(stores)

    async def plan(
        self,
        scope: Scope,
        target: ForgetTarget,
        record: GroupRecord | None = None,
        *,
        already: Mapping[SessionKey, frozenset[str]] | None = None,
        strict: bool = True,
    ) -> ForgetPlan:
        """The plan of ``target`` in ``scope``'s memory.

        ``already`` are the sessions' forgotten messages as they were before
        this forget recorded its own: the run plans again once it has recorded
        them, so that nothing cut meanwhile escapes it, and must still see its
        messages as new. ``strict`` answers a record id that names nothing
        with 404 (the delete-by-id alias skips it).

        Raises:
            NotFoundError: The target names nothing the owner holds.
            InvalidInputError: A record id of no kind of the owner's; a
                user's message target in a conversation of two people.
        """
        plan = ForgetPlan(scope=scope, target=target)
        plan.held = await self._logs.held(scope)
        if already is not None:
            plan.already = dict(already)
        plan.hidden_pictures = set(
            await self._stores.forget_log.live_targets(
                target="image",
                app_id=scope.app_id,
                project_id=scope.project_id,
                owner_id=scope.owner_id,
                owner_type=scope.owner_type,
            )
        )
        kind = target.kind
        if kind in ("message", "member") or (kind == "session" and plan.is_group):
            await self._plan_messages(plan, record)
        elif kind == "session":
            await self._plan_session(plan)
        elif kind == "image":
            await self._plan_image(plan)
        elif kind == "owner":
            await self._plan_owner(plan)
        elif kind == "condition":
            await self._plan_condition(plan)
        else:
            await self._plan_records(plan, strict=strict)
        if kind in ("record", "owner", "condition"):
            covered = await self._stores.episode.read_many(
                scope, [row[1] for row in plan.covered.get("episode", [])]
            )
            plan.segments = list(
                dict.fromkeys(
                    e.item.parent_id
                    for e in [*plan.episodes, *covered]
                    if e.item.parent_type == "memcell" and e.item.parent_id
                )
            )
        await self._pictures_gone(plan)
        if plan.is_group:
            await self._plan_group_scenes(plan)
            await self._retire(plan)
        else:
            await self._plan_orphans(plan)
        await self._plan_files(plan)
        await self._plan_profiles(plan)
        return plan

    async def _retire(self, plan: ForgetPlan) -> None:
        """A group's versions that may say what was forgotten: out of its history.

        The marked and replaced versions and every version that led to them,
        and the scene versions that change.
        """
        scope = plan.scope
        episodes = [*plan.episodes, *(r.old for r in plan.rewrites)]
        plan.retire("episode", await self._chains(scope, episodes))
        pictures: set[str] = set()
        for image_id in dict.fromkeys(p.item.image_id for p in plan.image_memories):
            rows = await self._stores.image_memory.read_by_image(scope, image_id)
            pictures |= _led_to(
                [(row.ref.entry_id, row.item.superseded_by) for row in rows],
                {p.ref.entry_id for p in plan.image_memories},
            )
        plan.retire("image_memory", pictures)
        plan.retire("scene", {v.ref.entry_id for e in plan.scenes for v in e.versions})

    # ── messages, a member, a group's session ─────────────────────────────

    async def _plan_messages(
        self, plan: ForgetPlan, record: GroupRecord | None
    ) -> None:
        target, scope = plan.target, plan.scope
        if target.kind == "member":
            assert target.member_id is not None and record is not None
            if not record.roster.intervals_of(target.member_id):
                raise NotFoundError(f"no member {target.member_id!r} in the group")
            sessions = [
                SessionKey(scope.app_id, scope.project_id, s) for s in record.sessions
            ]
        else:
            assert target.session_id is not None
            session = SessionKey(scope.app_id, scope.project_id, target.session_id)
            await self._check_session(scope, session)
            sessions = [session]
        found = await self._messages(sessions)
        if not plan.is_group and scope.owner_id not in set(found.sender.values()):
            raise NotFoundError(
                f"no session {target.session_id!r} of user {scope.owner_id!r}"
            )
        if not plan.is_group and found.people.get(sessions[0], set()) - {
            scope.owner_id
        }:
            # Two people each keep a copy of the session, read from one
            # buffer and one set of segments: forgetting a message there would
            # take it from the other person's memory too.
            raise InvalidInputError(
                f"session {target.session_id!r} is a conversation of two people: "
                "a message forget would change the other person's memory too; "
                "forget your own records of it by id, or the session"
            )
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
        plan.sessions = [key.session_id for key in plan.messages]
        await self._propagate_messages(plan, found)
        if target.kind == "member":
            assert target.member_id is not None
            held = await self._stores.profile.read(scope, target.member_id)
            if held is not None:
                plan.members_removed.append(target.member_id)

    async def _check_session(self, scope: Scope, session: SessionKey) -> None:
        """The session's memory is in ``scope``: its group's, or a user's own
        (no group's; whether the user took part, its messages say)."""
        bound = await self._stores.groups.binding(session)
        if scope.owner_type == "group" and bound == scope.owner_id:
            return
        if scope.owner_type != "group" and bound is None:
            return
        where = "in the group" if scope.owner_type == "group" else "of the user"
        raise NotFoundError(f"no session {session.session_id!r} {where}")

    async def _messages(self, sessions: Sequence[SessionKey]) -> _Messages:
        """Every message of ``sessions``: in their segments, then in their buffers."""
        found = _Messages()
        for session in sessions:
            order = found.order.setdefault(session, [])
            people = found.people.setdefault(session, set())
            cells = await self._stores.memcell.cells_in_session(session)
            for memcell_id, cell in cells.items():
                found.cells[memcell_id] = cell
                found.session_of[memcell_id] = session
                for item in cell.items:
                    if isinstance(item, ChatMessage):
                        found.sender[(session, item.id)] = item.sender_id
                        found.segment[(session, item.id)] = memcell_id
                        order.append(item.id)
                        if item.role == "user":
                            people.add(item.sender_id)
            for message in await self._stores.session_buffer.list(session):
                message_id = message.client_message_id or message.message_id
                if message.role == "user":
                    people.add(message.sender_id)
                if (session, message_id) not in found.sender:
                    found.sender[(session, message_id)] = message.sender_id
                    order.append(message_id)
        for session, order in found.order.items():
            found.order[session] = list(dict.fromkeys(order))
        return found

    async def _propagate_messages(self, plan: ForgetPlan, found: _Messages) -> None:
        """Rewrites, marks, picture memories and profile items of new forgets."""
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
        await self._cover_segments(plan, fresh)
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
                for stored in episodes[:-1]:
                    plan.add_episode(stored)
            else:
                for stored in episodes:
                    plan.add_episode(stored)
        if not plan.is_group:
            plan.segments = [
                memcell_id
                for memcell_id, cell in readable.items()
                if not wrote_in(cell, scope.owner_id)
            ]
        await self._pictures_of_forgotten(plan, found, fresh, readable)
        if plan.is_group:
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
            if memory.item.superseded_by not in (None, FORGOTTEN_MARK):
                continue
            if memory.item.image_id not in carried.get(memcell_id, set()):
                continue
            still = {
                image_id
                for item in readable[memcell_id].items
                for image_id in image_ids_in(str(getattr(item, "content", "") or ""))
            }
            if memory.item.image_id in still:
                continue
            if memory.item.superseded_by is None:
                plan.add_picture(memory)
            else:
                plan.cover_picture(memory)

    # ── a user's conversation ────────────────────────────────────────────

    async def _plan_session(self, plan: ForgetPlan) -> None:
        """A user's conversation: its current episodes and picture memories."""
        scope, session_id = plan.scope, plan.target.session_id
        assert session_id is not None
        stores = self._stores
        episodes = await current_episodes_of(stores, scope, session_id)
        plan.episodes = await stores.episode.read_many(scope, [e for _, e in episodes])
        pictures = await current_pictures_of(stores, scope, session_id)
        plan.image_memories = await stores.image_memory.read_many(
            scope, [e for _, e, _ in pictures]
        )
        plan.segments = sorted(await segments_of(stores, scope, [session_id]))
        plan.sessions = [session_id]
        for stored in await self._forgotten_episodes(plan, session_id):
            plan.cover("episode", stored.ref.md_path, stored.ref.entry_id)
        if plan.held.get("image_memory"):
            entries = await stores.image_memory.entries_for_session(scope, session_id)
            for row in await stores.image_memory.read_many(
                scope, [e for _, e, _ in entries]
            ):
                plan.cover_picture(row)

    async def _forgotten_episodes(
        self, plan: ForgetPlan, session_id: str
    ) -> list[Stored[Episode]]:
        """The conversation's episode versions a live forget holds forgotten."""
        held = plan.held.get("episode", {})
        if not held:
            return []
        scope = plan.scope
        entries = await self._stores.episode.entries_for_session(scope, session_id)
        marks: dict[str, dict[str, str]] = {}
        for md_path in {md_path for md_path, _ in entries}:
            marks[md_path] = await self._stores.episode.deprecations(scope, md_path)
        return await self._stores.episode.read_many(
            scope,
            [
                e
                for md, e in entries
                if marks[md].get(e) == FORGOTTEN_MARK and e in held
            ],
        )

    async def _cover_segments(
        self, plan: ForgetPlan, segments: Collection[str]
    ) -> None:
        """The forgotten episodes of segments that lost a message: covered."""
        if not segments or not plan.held.get("episode"):
            return
        for session in plan.messages:
            for stored in await self._forgotten_episodes(plan, session.session_id):
                if stored.item.parent_id in segments:
                    plan.cover("episode", stored.ref.md_path, stored.ref.entry_id)

    # ── a picture ────────────────────────────────────────────────────────

    async def _plan_image(self, plan: ForgetPlan) -> None:
        """The picture's live memories, and what names it (a text edit)."""
        scope, image_id = plan.scope, plan.target.image_id
        assert image_id is not None
        rows = await self._stores.image_memory.read_by_image(scope, image_id)
        for row in rows:
            plan.cover_picture(row)
        occurrences = [row for row in rows if row.item.superseded_by is None]
        if not occurrences and plan.is_group:
            raise NotFoundError(f"no picture {image_id!r} in the group")
        plan.sources = {s for row in occurrences for s in row.item.sender_ids}
        for row in occurrences:
            plan.add_picture(row)
        for stored in await self._stores.episode.read_referring(scope, image_id):
            if image_id not in stored.item.episode and image_id not in (
                stored.item.summary or ""
            ):
                continue
            plan.rewrites.append(
                Rewrite(
                    old=stored,
                    session_id=stored.item.session_id or "",
                    memcell_id=stored.item.parent_id,
                    image_id=image_id,
                )
            )
        for version, mark in await newest_versions(self._stores.scene, scope):
            scene = version.item
            if mark is None and (
                image_id in scene.narrative or image_id in (scene.summary or "")
            ):
                plan.scenes.append(
                    SceneEdit(
                        scene_id=scene.scene_id,
                        current=version,
                        versions=(version,),
                        image_id=image_id,
                    )
                )
        plan.pictures = [image_id]

    # ── records by id, everything, a condition ───────────────────────────

    async def _plan_records(self, plan: ForgetPlan, *, strict: bool) -> None:
        scope, stores = plan.scope, self._stores
        missing: list[str] = []
        group_items: list[str] = []
        scene_rows: list[tuple[Stored[Scene], str | None]] | None = None
        for record_id in dict.fromkeys(plan.target.record_ids):
            kind, entry_id = kind_of(scope, record_id)
            if kind == "episode":
                end = await episode_end(stores, scope, entry_id)
                if end is None:
                    missing.append(record_id)
                elif end[1] is None:
                    plan.add_episode(end[0])
                    plan.sources |= set(end[0].item.sender_ids)
                elif end[1] == FORGOTTEN_MARK:
                    plan.cover("episode", end[0].ref.md_path, end[0].ref.entry_id)
            elif kind == "image_memory":
                found = await picture_end(stores, scope, entry_id)
                if found is None:
                    missing.append(record_id)
                elif found[1] is None:
                    plan.add_picture(found[0])
                    plan.sources |= set(found[0].item.sender_ids)
                else:
                    plan.cover_picture(found[0])
            elif kind == "scene":
                if scene_rows is None:
                    scene_rows = await stores.scene.read_all(scope)
                chain = scene_chain(entry_id, scene_rows)
                newest = chain.newest
                if newest is None:
                    missing.append(record_id)
                    continue
                versions = tuple(v.stored for v in chain.versions)
                plan.sources |= {s for v in versions for s in v.item.sender_ids}
                if newest.mark == FORGOTTEN_MARK:
                    ref = newest.stored.ref
                    plan.cover("scene", ref.md_path, ref.entry_id)
                if all(e.scene_id != newest.stored.item.scene_id for e in plan.scenes):
                    plan.scenes.append(
                        SceneEdit(
                            scene_id=newest.stored.item.scene_id,
                            current=newest.stored if newest.mark is None else None,
                            versions=versions,
                            whole=True,
                        )
                    )
            elif kind == "group_profile_item":
                group_items.append(entry_id)
            else:
                plan.item_ids.add(entry_id)
        if group_items:
            missing += await self._plan_group_items(plan, group_items)
        if plan.item_ids:
            missing += await self._find_items(plan)
        if missing and strict:
            raise NotFoundError(
                f"no record {missing[0]!r} of {scope.owner_type} {scope.owner_id!r}"
            )

    async def _chains(
        self, scope: Scope, episodes: Sequence[Stored[Episode]]
    ) -> set[str]:
        """The given episode versions and every version that led to them."""
        if not episodes:
            return set()
        rows = await self._stores.episode.read_all(scope)
        return _led_to(
            [(stored.ref.entry_id, mark) for stored, mark in rows],
            {e.ref.entry_id for e in episodes},
        )

    async def _find_items(self, plan: ForgetPlan) -> list[str]:
        """The named profile items that name nothing (a group's: no member's)."""
        scope = plan.scope
        named = set(plan.item_ids)
        held: dict[str, str] = {}
        for person in await self._stores.profile.people(scope):
            profile = await self._stores.profile.read(scope, person)
            if profile is None:
                continue
            for item_id in named:
                if profile_items.find_item(profile, profile.history, item_id):
                    held[item_id] = person
        if plan.is_group:
            plan.sources |= set(held.values())
        missing = [i for i in sorted(named) if i not in held]
        if missing and not plan.is_group:
            # A value a live forget took out already is forgotten, not missing.
            forgotten = await self._logs.forgotten_items(scope)
            missing = [i for i in missing if i not in forgotten]
        return missing

    async def _plan_owner(self, plan: ForgetPlan) -> None:
        scope, stores = plan.scope, self._stores
        for stored, mark in await stores.episode.read_all(scope):
            if mark is None:
                plan.add_episode(stored)
            elif mark == FORGOTTEN_MARK:
                plan.cover("episode", stored.ref.md_path, stored.ref.entry_id)
        for row in await stores.image_memory.read_all(scope):
            if row.item.superseded_by is None:
                plan.add_picture(row)
            else:
                plan.cover_picture(row)
        for version, mark in await newest_versions(stores.scene, scope):
            if mark is None:
                plan.scenes.append(
                    SceneEdit(
                        scene_id=version.item.scene_id, current=version, whole=True
                    )
                )
            elif mark == FORGOTTEN_MARK:
                plan.cover("scene", version.ref.md_path, version.ref.entry_id)
        plan.whole_profile = True

    async def _plan_condition(self, plan: ForgetPlan) -> None:
        scope, stores, target = plan.scope, self._stores, plan.target
        kinds = set(target.memory_types)

        def within(at: int | None) -> bool:
            if at is None:
                return False
            return (target.start is None or at >= target.start) and (
                target.end is None or at < target.end
            )

        if "episode" in kinds:
            for stored, mark in await stores.episode.read_all(scope):
                if not within(stored.item.timestamp):
                    continue
                if mark is None:
                    plan.add_episode(stored)
                elif mark == FORGOTTEN_MARK:
                    plan.cover("episode", stored.ref.md_path, stored.ref.entry_id)
        if "image_memory" in kinds:
            for row in await stores.image_memory.read_all(scope):
                if not within(row.item.timestamp):
                    continue
                if row.item.superseded_by is None:
                    plan.add_picture(row)
                else:
                    plan.cover_picture(row)
        if "scene" in kinds:
            for version, mark in await newest_versions(stores.scene, scope):
                if not within(version.item.timestamp):
                    continue
                if mark is None:
                    plan.scenes.append(
                        SceneEdit(
                            scene_id=version.item.scene_id,
                            current=version,
                            whole=True,
                        )
                    )
                elif mark == FORGOTTEN_MARK:
                    plan.cover("scene", version.ref.md_path, version.ref.entry_id)
        if "profile" in kinds:
            profile = await stores.profile.read(scope)
            if profile is not None:
                entries: list[Mapping[str, Any]] = [
                    item.model_dump()
                    for bucket in profile_items.BUCKETS
                    for item in getattr(profile, bucket)
                ]
                entries += profile.history
                for entry in entries:
                    at = entry.get("observed_at") or entry.get("source_end")
                    item_id = entry.get("item_id")
                    if isinstance(item_id, str) and isinstance(at, int) and within(at):
                        plan.item_ids.add(item_id)

    # ── pictures, files ──────────────────────────────────────────────────

    async def _pictures_gone(self, plan: ForgetPlan) -> None:
        """The pictures whose last live memory of the owner this forget takes out."""
        leaving = {row.ref.entry_id for row in plan.image_memories}
        for image_id in dict.fromkeys(row.item.image_id for row in plan.image_memories):
            if not image_id or image_id in plan.pictures:
                continue
            rows = await self._stores.image_memory.read_by_image(plan.scope, image_id)
            if not any(
                row.item.superseded_by is None and row.ref.entry_id not in leaving
                for row in rows
            ):
                plan.pictures.append(image_id)

    async def _plan_files(self, plan: ForgetPlan) -> None:
        """Release a picture file when no live memory of any owner refers to it."""
        for image_id in plan.pictures:
            if self._indexes is not None:
                own = {
                    row.ref.id
                    for row in await self._stores.image_memory.read_by_image(
                        plan.scope, image_id
                    )
                }
                hits = await self._indexes.image_memory.find(
                    all_of(eq("image_id", image_id), is_null("deprecated_by")),
                    limit=_REFERENCE_SCAN,
                )
                if any(hit.stored.ref.id not in own for hit in hits):
                    continue
            plan.files.append(image_id)

    # ── scenes ───────────────────────────────────────────────────────────

    async def _plan_orphans(self, plan: ForgetPlan) -> None:
        """A user's scenes none of whose episodes stays alive (a dry run's view)."""
        gone = {e.ref.entry_id for e in plan.episodes}
        if not gone:
            return
        targeted = {e.scene_id for e in plan.scenes if e.whole}
        for version, mark in await newest_versions(self._stores.scene, plan.scope):
            if mark is not None or version.item.scene_id in targeted:
                continue
            alive = await current_versions(
                self._stores.episode, plan.scope, version.item.member_ids
            )
            if not [
                s
                for s in alive.values()
                if s is not None and s.ref.entry_id not in gone
            ]:
                plan.orphaned.append(version)

    async def _plan_group_scenes(self, plan: ForgetPlan) -> None:
        """A group's scene that lost or changed an episode is written again.

        Every version holding a changed episode may say what was forgotten;
        the current one is superseded by a version narrated afresh over the
        episodes left (or marked forgotten when fewer than ``[scenes]
        min_members`` are), and the older ones are hidden from the history.
        A picture's text edit changes no member: its scenes are edited apart.
        """
        changed = {r.old.ref.entry_id for r in plan.rewrites if r.cell is not None}
        changed |= {e.ref.entry_id for e in plan.episodes}
        if not changed:
            return
        edited = {e.scene_id for e in plan.scenes}
        dropped = {e.ref.entry_id for e in plan.episodes}
        by_scene: dict[str, list[tuple[Stored[Scene], str | None]]] = {}
        for version, mark in await self._stores.scene.read_all(plan.scope):
            by_scene.setdefault(version.item.scene_id, []).append((version, mark))
        # A member is named by the id a version was written with; the forget
        # reaches it through the version that id leads to now.
        now = await current_versions(
            self._stores.episode,
            plan.scope,
            {
                m
                for versions in by_scene.values()
                for v, _ in versions
                for m in v.item.member_ids
            },
        )

        def reached(member: str) -> str | None:
            found = now.get(member)
            return found.ref.entry_id if found is not None else None

        for scene_id, versions in by_scene.items():
            if scene_id in edited:
                continue
            hit = tuple(
                v
                for v, _ in versions
                if any(m in changed or reached(m) in changed for m in v.item.member_ids)
            )
            if not hit:
                continue
            newest, mark = max(versions, key=lambda pair: pair[0].item.version)
            held = {v.ref.entry_id for v in hit}
            current = newest if mark is None and newest.ref.entry_id in held else None
            members: tuple[str, ...] = ()
            if current is not None:
                members = tuple(
                    m
                    for m in current.item.member_ids
                    if reached(m) is not None and reached(m) not in dropped
                )
                enough = (
                    self._scene_members is not None
                    and len(members) >= self._scene_members
                )
                members = members if enough else ()
            plan.scenes.append(
                SceneEdit(
                    scene_id=scene_id, current=current, versions=hit, members=members
                )
            )
            plan.retire("scene", {v.ref.entry_id for v in hit})

    # ── profiles ─────────────────────────────────────────────────────────

    async def _plan_profiles(self, plan: ForgetPlan) -> None:
        """The source rule over the owner's profile documents (a dry run's view)."""
        scope = plan.scope
        if not plan.is_group:
            already = await self._logs.forgotten_segments(
                scope,
                except_session=plan.target.session_id
                if plan.target.kind == "session"
                else None,
            )
            profile = await self._stores.profile.read(scope)
            if profile is not None:
                edit = user_edit(profile, plan, already=already)
                if edit.changes:
                    plan.profiles.append(edit)
            plan.covered_values = [
                value
                for value in await self._logs.taken_values(scope)
                if reaches_value(plan, value, already=already)
            ]
            return
        for member_id in await self._stores.profile.people(scope):
            if member_id in plan.members_removed:
                continue
            if plan.target.kind == "member" and plan.target.member_id == member_id:
                continue
            held = await self._stores.profile.read(scope, member_id)
            if held is None:
                continue
            edit = member_edit(held, plan)
            if edit.changes:
                plan.profiles.append(edit)

    async def _plan_group_items(
        self, plan: ForgetPlan, entry_ids: Sequence[str]
    ) -> list[str]:
        """Group-profile items named by id; the ids naming nothing."""
        items = {
            s.ref.entry_id: s
            for s in await self._stores.group_profile.items(plan.scope)
        }
        for entry_id in entry_ids:
            if entry_id in items:
                plan.group_forgotten.append(items[entry_id])
                plan.sources |= {s.speaker_id for s in items[entry_id].item.sources}
        return [f"{plan.scope.owner_id}_{e}" for e in entry_ids if e not in items]

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


def user_edit(
    held: Profile, plan: ForgetPlan, *, already: Collection[str]
) -> ProfileEdit:
    """What a forget takes out of a user's ``user.md``.

    An item or a replaced value goes when the target names it, when every
    segment it rests on is forgotten -- by this forget or one the user made
    before (``already``) -- and one of them by this forget, or when every
    picture it was read off is gone. A message forget also takes out what
    rests on a segment that lost a message while the owner still wrote in one
    of its segments (**stale**, the rule a member's profile in a group
    follows): it may state what the message said. A stale item's remaining
    segments are queued, and the owner's next profile update writes it again
    from what they still say; a stale replaced value is history, written
    again by nothing. ``owner`` takes it all.
    """
    edit = ProfileEdit(person_id=plan.scope.owner_id, whole=plan.whole_profile)
    if plan.whole_profile:
        for place, item in _placed(held):
            edit.take(place, item)
        for position, old in enumerate(held.history):
            edit.take_history(position, old)
        return edit
    these = set(plan.segments)
    forgotten = these | set(already)
    pictures = set(plan.pictures)
    readable = plan.readable if plan.target.kind == "message" else {}
    owner = plan.scope.owner_id
    queue: list[str] = []

    def sources_of(entry: Any) -> list[str]:
        extra = entry if isinstance(entry, Mapping) else (entry.model_extra or {})
        raw = extra.get("source_memcell_ids")
        return [str(s) for s in raw] if isinstance(raw, list) else []

    def by_segments(entry: Any) -> bool:
        sources = set(sources_of(entry))
        return bool(sources) and sources <= forgotten and bool(sources & these)

    def stale(entry: Any) -> bool:
        return any(s in readable for s in sources_of(entry))

    def named(entry: Any) -> bool:
        extra = entry if isinstance(entry, Mapping) else (entry.model_extra or {})
        return extra.get("item_id") in plan.item_ids

    for place, item in _placed(held):
        if named(item):
            edit.take(place, item)
        elif these and by_segments(item):
            edit.take(place, item)
            edit.by_segments.append(place)
        elif profile_items.rests_on_pictures(item, pictures):
            edit.take(place, item)
        elif stale(item):
            edit.take(place, item, stale=True)
            queue += [
                s
                for s in sources_of(item)
                if s not in forgotten
                and (s not in readable or wrote_in(readable[s], owner))
            ]
    for position, old in enumerate(held.history):
        if named(old):
            edit.take_history(position, old)
        elif these and by_segments(old):
            edit.take_history(position, old)
            edit.by_segments.append(("history", position))
        elif profile_items.rests_on_pictures(old, pictures) or stale(old):
            edit.take_history(position, old)
    edit.queue = [m for m in dict.fromkeys(queue) if m not in held.pending_memcell_ids]
    return edit


def reaches_value(
    plan: ForgetPlan, value: Removed, *, already: Collection[str]
) -> bool:
    """Whether a forget's target reaches a value another live forget took out of
    ``user.md``: :func:`user_edit` would take it, were it still there."""
    if plan.whole_profile:
        return True
    if value.bucket == profile_items.SUMMARY:
        return False
    item = value.item
    if item.get("item_id") in plan.item_ids:
        return True
    raw = item.get("source_memcell_ids")
    sources = {str(s) for s in raw} if isinstance(raw, list) else set()
    these = set(plan.segments)
    if sources and sources <= these | set(already) and sources & these:
        return True
    if plan.target.kind == "message" and sources & set(plan.readable):
        return True
    if profile_items.rests_on_pictures(item, plan.pictures):
        return True
    target = plan.target
    if target.kind == "condition" and "profile" in target.memory_types:
        at = item.get("observed_at") or item.get("source_end")
        return (
            isinstance(at, int)
            and (target.start is None or at >= target.start)
            and (target.end is None or at < target.end)
        )
    return False


def member_edit(held: Profile, plan: ForgetPlan) -> ProfileEdit:
    """What the source rule does to one member's profile in the group.

    Applied twice: by the plan, and again under the member's lock by the run,
    over the document as it is then (an update may have written it since).
    A text item resting on a segment that lost messages is forgotten when
    none of its segments holds a message of the member's any more, and stale
    otherwise: it may quote what was forgotten, and other members read it. A
    picture item is forgotten when every picture it rests on is gone.
    """
    member_id = held.owner_id
    readable = plan.readable
    pictures = set(plan.pictures)
    edit = ProfileEdit(person_id=member_id)
    queue: list[str] = []
    for place, item in _placed(held):
        extra = item.model_extra or {}
        if extra.get("item_id") and extra.get("item_id") in plan.item_ids:
            edit.take(place, item)
            continue
        if profile_items.item_source_images(item):
            if profile_items.rests_on_pictures(item, pictures):
                edit.take(place, item)
            continue
        sources = [str(s) for s in extra.get("source_memcell_ids") or []]
        if not set(sources) & set(readable):
            continue
        remaining = [
            s for s in sources if s not in readable or wrote_in(readable[s], member_id)
        ]
        edit.take(place, item, stale=bool(remaining))
        queue.extend(remaining)
    for position, old in enumerate(held.history):
        sources = {str(s) for s in old.get("source_memcell_ids") or []}
        named = bool(old.get("item_id")) and old.get("item_id") in plan.item_ids
        gone = profile_items.rests_on_pictures(old, pictures)
        if sources & set(readable) or gone or named:
            edit.take_history(position, old)
        if named:  # a replaced value a read listed, forgotten by its id
            edit.forgotten.append(str(old["item_id"]))
    edit.queue = [m for m in dict.fromkeys(queue) if m not in held.pending_memcell_ids]
    return edit


async def current_episodes_of(
    stores: Stores, scope: Scope, session_id: str
) -> list[tuple[str, str]]:
    """``(md_path, entry_id)`` of the conversation's episode versions no mark
    retires: a superseded one keeps the mark that names its successor."""
    entries = await stores.episode.entries_for_session(scope, session_id)
    marks: dict[str, dict[str, str]] = {}
    for md_path in {md_path for md_path, _ in entries}:
        marks[md_path] = await stores.episode.deprecations(scope, md_path)
    return [(md, e) for md, e in entries if not marks[md].get(e)]


async def current_pictures_of(
    stores: Stores, scope: Scope, session_id: str
) -> list[tuple[str, str, str]]:
    """``(md_path, entry_id, image_id)`` of the conversation's current picture
    memories."""
    entries = await stores.image_memory.entries_for_session(scope, session_id)
    rows = await stores.image_memory.read_many(scope, [e for _, e, _ in entries])
    marked = {row.ref.entry_id for row in rows if row.item.superseded_by}
    return [row for row in entries if row[1] not in marked]


def _items(held: Profile) -> list[Any]:
    return [*held.explicit_info, *held.implicit_traits]


def _placed(held: Profile) -> list[tuple[tuple[str, int], Any]]:
    """Every item of ``held`` with its place, bucket by bucket."""
    return [
        ((bucket, n), item)
        for bucket in profile_items.BUCKETS
        for n, item in enumerate(getattr(held, bucket))
    ]


def _led_to(rows: Sequence[tuple[str, str | None]], ends: Collection[str]) -> set[str]:
    """``ends`` and every entry whose chain of marks leads to one of them."""
    older: dict[str, list[str]] = {}
    for entry_id, mark in rows:
        if mark:
            older.setdefault(mark, []).append(entry_id)
    found: set[str] = set()
    frontier = list(ends)
    while frontier:
        entry_id = frontier.pop()
        if entry_id in found:
            continue
        found.add(entry_id)
        frontier.extend(older.get(entry_id, []))
    return found


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

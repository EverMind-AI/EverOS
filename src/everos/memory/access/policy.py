"""The access policy: who may read which memory, and who may change a group.

One module decides every read (docs/group-memory.md, D9 and section 9):

* :meth:`AccessPolicy.read_scope` turns an identity and a requested scope into
  a :class:`ReadScope` -- the owners, the asker's membership windows per group
  and the predicate every recall puts into its ``where``. Visibility is part of
  the index query, never a filter applied to results afterwards, so a ranking
  (the multi-round decider above all) never sees a candidate the asker cannot.
* :meth:`AccessPolicy.member_groups` compiles the groups a personal read also
  covers when it asks to (``include_groups``): the owner's current groups,
  each read exactly as a group search by the owner reads it.
* :func:`visible` answers the same question for one record that does not come
  through the index (a markdown read); :meth:`AccessPolicy.picture_visible`
  for a picture's bytes, served by content hash.
* :meth:`AccessPolicy.buffered_messages` decides which in-flight messages of a
  session a read may see.
* :func:`authorize_roster_change` and :func:`authorize_manage` decide roster
  changes and meta updates; :func:`authorize_forget` decides a forget, its
  undo and a group-scoped delete (section 5.5).

Rules the implementation must keep (section 9.1):

* a personal scope is read by its owner only; a one-to-one request without an
  asker is read as asked by its owner (today's behaviour);
* a group is read by current members only (403 otherwise), and only records all
  of whose sources lie in one of the asker's windows: ``source_start >= from``
  and ``source_end < to`` (no upper bound for the open window);
* ``history_visible_from = creation`` gives a current member the whole history;
* ``audience = group`` keeps only what every current member can see;
* an owner or admin role grants roster management and forgetting, never extra
  reading; a member may forget only what rests on their own messages;
* a forgotten message is absent from every read of a session's buffer, as it
  is from every record a forget rewrote.
"""

from __future__ import annotations

__all__ = [
    "AccessPolicy",
    "Audience",
    "GroupView",
    "ReadScope",
    "ScopeRequest",
    "Window",
    "authorize_forget",
    "authorize_manage",
    "authorize_roster_change",
    "intersect_windows",
    "owner_predicate",
    "visible",
    "visible_profile",
]

from collections.abc import Awaitable, Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Literal

from everos.core.errors import ForbiddenError, InvalidInputError
from everos.core.utils.datetime import from_timestamp_ms, to_timestamp_ms
from everos.memory.models import (
    FORGOTTEN_MARK,
    MANAGER_ROLES,
    MEMBER_PREVIOUS_KEY,
    MEMBER_VERSION_KEYS,
    CanonicalMessage,
    ExplicitInfoItem,
    Group,
    ImageMemory,
    ImplicitTraitItem,
    Profile,
    Roster,
    RosterEvent,
)
from everos.memory.ports import (
    Filter,
    GroupRecord,
    GroupStore,
    MessageForgetStore,
    OwnerType,
    Scope,
    SessionKey,
    Stored,
    all_of,
    any_of,
    eq,
    gte,
    is_null,
    lt,
    ne,
)

from .identity import Identity

type Audience = Literal["asker", "group"]

_NOTHING: Filter = all_of(eq("owner_type", "group"), eq("owner_type", "user"))
"""A contradiction: the predicate of a scope that can see nothing at all.

``AnyOf`` cannot be empty (an empty group renders as "match every row"), so an
empty visibility is spelt as a condition no row satisfies."""


@dataclass(frozen=True)
class Window:
    """A half-open time range ``[start, end)`` in epoch ms; ``None`` is unbounded."""

    start: int | None
    end: int | None

    def contains_point(self, at: int) -> bool:
        return (self.start is None or at >= self.start) and (
            self.end is None or at < self.end
        )

    def contains_span(self, start: int, end: int) -> bool:
        """Whether a record whose sources span ``[start, end]`` lies inside."""
        return (self.start is None or start >= self.start) and (
            self.end is None or end < self.end
        )


@dataclass(frozen=True)
class GroupView:
    """One group of a read scope: its roster and the asker's windows in it."""

    group: Group
    roster: Roster
    windows: tuple[Window, ...]

    @property
    def group_id(self) -> str:
        return self.group.group_id

    def predicate(self) -> Filter | None:
        """This group's part of the scope predicate; ``None`` sees nothing."""
        if not self.windows:
            return None
        clauses: list[Filter] = []
        for window in self.windows:
            bounds = [
                gte("source_start", from_timestamp_ms(window.start))
                if window.start is not None
                else None,
                lt("source_end", from_timestamp_ms(window.end))
                if window.end is not None
                else None,
            ]
            if all(bound is None for bound in bounds):
                return eq("owner_id", self.group_id)  # the whole history
            clauses.append(all_of(*bounds))
        return all_of(eq("owner_id", self.group_id), any_of(*clauses))


@dataclass(frozen=True)
class ScopeRequest:
    """The scope a read names, before the policy has looked at it.

    Exactly one of ``user_id`` / ``group_ids`` / ``project_scope`` is set; the
    request DTOs validate that.
    """

    app_id: str = "default"
    project_id: str = "default"
    user_id: str | None = None
    group_ids: tuple[str, ...] = ()
    audience: Audience = "asker"
    project_scope: bool = False
    """Every group of ``project_id`` the asker currently belongs to, as one
    pool (section 8.7); ``group_ids`` is then resolved by the policy."""


@dataclass(frozen=True)
class ReadScope:
    """What one read may see, compiled.

    Attributes:
        app_id / project_id: The space.
        owner_type: ``user`` or ``group``.
        owner_ids: The one user, or every group of the scope.
        asker_id: Who asks; the owner for a personal read without one.
        audience: ``asker`` or ``group`` (section 15).
        groups: Per group, its roster and the windows in force.
        predicate: The owner, space, status and window conditions -- what
            replaces the owner part of every recall's ``where``.
        versions_predicate: The same, with every version a group record
            had except the forgotten ones: what a read of an item's history
            (``include_superseded``) uses. Equal to ``predicate`` outside
            groups.
    """

    app_id: str
    project_id: str
    owner_type: OwnerType
    owner_ids: tuple[str, ...]
    asker_id: str | None
    predicate: Filter
    audience: Audience = "asker"
    groups: tuple[GroupView, ...] = field(default_factory=tuple)
    versions_predicate: Filter | None = None

    @property
    def history_predicate(self) -> Filter:
        """``versions_predicate`` when set, else ``predicate``."""
        return self.versions_predicate or self.predicate

    @property
    def is_group(self) -> bool:
        return self.owner_type == "group"

    @property
    def owner_id(self) -> str:
        """The owner a single-owner caller names (the first group of a pool)."""
        return self.owner_ids[0]

    def group(self, group_id: str) -> GroupView | None:
        for view in self.groups:
            if view.group_id == group_id:
                return view
        return None

    def scope_of(self, owner_id: str) -> Scope:
        """The storage scope of one of this read's owners."""
        return Scope(
            app_id=self.app_id,
            project_id=self.project_id,
            owner_id=owner_id,
            owner_type=self.owner_type,
        )


def owner_predicate(
    *, owner_id: str, owner_type: str, app_id: str, project_id: str
) -> Filter:
    """The owner part of a personal read, as recalls have always used it.

    A user read also excludes the rows a later version or a forget retired
    (``deprecated_by``).
    """
    base: list[Filter] = [
        eq("owner_id", owner_id),
        eq("owner_type", owner_type),
        eq("app_id", app_id),
        eq("project_id", project_id),
    ]
    if owner_type == "user":
        base.append(is_null("deprecated_by"))
    return all_of(*base)


def intersect_windows(
    left: Sequence[Window], right: Sequence[Window]
) -> tuple[Window, ...]:
    """The time both window sets cover (each set sorted and disjoint)."""

    def lo(w: Window) -> float:
        return float("-inf") if w.start is None else w.start

    def hi(w: Window) -> float:
        return float("inf") if w.end is None else w.end

    out: list[Window] = []
    i = j = 0
    a = sorted(left, key=lo)
    b = sorted(right, key=lo)
    while i < len(a) and j < len(b):
        start = max(lo(a[i]), lo(b[j]))
        end = min(hi(a[i]), hi(b[j]))
        if start < end:
            out.append(
                Window(
                    start=None if start == float("-inf") else int(start),
                    end=None if end == float("inf") else int(end),
                )
            )
        if hi(a[i]) < hi(b[j]):
            i += 1
        else:
            j += 1
    return tuple(out)


def _member_windows(group: Group, roster: Roster, member_id: str) -> tuple[Window, ...]:
    if group.history_visible_from == "creation" and roster.is_current_member(member_id):
        return (Window(start=None, end=None),)
    return tuple(Window(start, end) for start, end in roster.windows_of(member_id))


def _view_of(record: GroupRecord, asker: str, audience: Audience) -> GroupView:
    """One group of a read scope: the asker's windows in it, cut by ``audience``.

    ``audience = group`` keeps only the time every current member can see.
    """
    windows = _member_windows(record.group, record.roster, asker)
    if audience == "group":
        for member in record.roster.current_members():
            windows = intersect_windows(
                windows,
                _member_windows(record.group, record.roster, member.member_id),
            )
    return GroupView(group=record.group, roster=record.roster, windows=windows)


def _group_read_scope(
    app_id: str,
    project_id: str,
    asker: str,
    audience: Audience,
    views: Sequence[GroupView],
) -> ReadScope:
    """The read scope of ``views``: one predicate, a disjunction over the groups.

    Each group contributes its own windows; a group whose windows are empty
    contributes nothing, and a scope left with no group sees nothing at all.
    """
    per_group = [p for p in (view.predicate() for view in views) if p is not None]
    space = (
        eq("owner_type", "group"),
        eq("app_id", app_id),
        eq("project_id", project_id),
    )
    predicate = (
        all_of(*space, is_null("deprecated_by"), any_of(*per_group))
        if per_group
        else _NOTHING
    )
    versions = (
        all_of(
            *space,
            any_of(is_null("deprecated_by"), ne("deprecated_by", FORGOTTEN_MARK)),
            any_of(*per_group),
        )
        if per_group
        else _NOTHING
    )
    return ReadScope(
        app_id=app_id,
        project_id=project_id,
        owner_type="group",
        owner_ids=tuple(view.group_id for view in views),
        asker_id=asker,
        predicate=predicate,
        audience=audience,
        groups=tuple(views),
        versions_predicate=versions,
    )


def visible(
    scope: ReadScope,
    *,
    owner_type: str,
    owner_id: str,
    source_start: int,
    source_end: int,
) -> bool:
    """Whether one record is visible in ``scope`` (for reads outside the index).

    The record is described by its owner and the time span of its sources; a
    one-to-one record's span is its own timestamp.
    """
    if owner_type != scope.owner_type or owner_id not in scope.owner_ids:
        return False
    if not scope.is_group:
        return True
    view = scope.group(owner_id)
    return view is not None and any(
        w.contains_span(source_start, source_end) for w in view.windows
    )


def visible_profile(
    scope: ReadScope, group_id: str, profile: Profile
) -> Profile | None:
    """The part of a member-in-group profile ``scope`` may see (section 9.1).

    An item is visible when the span of its sources (``source_start`` /
    ``source_end``) lies in one of the asker's windows in the group; an item
    without a span is not. The summary is written over every item, so it is
    kept only when every item is visible. ``None`` when no item is.

    The values a visible item replaced (section 7.4) are shown newest first up
    to the first one the asker may not see; an item left with none shows no
    version at all -- no ``supersedes`` naming an item the asker cannot see,
    no ``since`` -- as a decision whose newer version is out of sight names
    none.
    """

    def keep(entry: Mapping[str, object]) -> bool:
        start, end = entry.get("source_start"), entry.get("source_end")
        if not isinstance(start, int) or not isinstance(end, int):
            return False
        return visible(
            scope,
            owner_type="group",
            owner_id=group_id,
            source_start=start,
            source_end=end,
        )

    explicit = [
        _versions_shown(i, keep)
        for i in profile.explicit_info
        if keep(i.model_extra or {})
    ]
    traits = [
        _versions_shown(i, keep)
        for i in profile.implicit_traits
        if keep(i.model_extra or {})
    ]
    if not explicit and not traits:
        return None
    whole = len(explicit) == len(profile.explicit_info) and len(traits) == len(
        profile.implicit_traits
    )
    return profile.model_copy(
        update={
            "explicit_info": explicit,
            "implicit_traits": traits,
            "summary": profile.summary if whole else "",
        }
    )


def _versions_shown[ItemT: (ExplicitInfoItem, ImplicitTraitItem)](
    item: ItemT, keep: Callable[[Mapping[str, object]], bool]
) -> ItemT:
    """``item`` with the values it replaced cut to those ``keep`` lets through."""
    extra = item.model_extra or {}
    previous = extra.get(MEMBER_PREVIOUS_KEY)
    listed = previous if isinstance(previous, list) else []
    shown: list[object] = []
    for entry in listed:
        if not isinstance(entry, Mapping) or not keep(entry):
            break
        shown.append(entry)
    versioned = (MEMBER_PREVIOUS_KEY, *MEMBER_VERSION_KEYS)
    if shown and len(shown) == len(listed):
        return item
    if not shown and not any(key in extra for key in versioned):
        return item
    data = {k: v for k, v in item.model_dump().items() if k not in versioned}
    if shown:
        data |= {k: extra[k] for k in MEMBER_VERSION_KEYS if k in extra}
        data[MEMBER_PREVIOUS_KEY] = shown
    return type(item).model_validate(data)


def authorize_manage(roster: Roster, asker_id: str | None) -> None:
    """Raise unless ``asker_id`` may change the group's meta or roster.

    Owners and admins may. A group with no current owner or admin at all (one
    created implicitly by a batch without an asker) is managed by any current
    member, so that it is not locked against every change.

    Raises:
        ForbiddenError: The asker is not a current member, or holds no
            manager role in a group that has managers.
    """
    current = roster.current(asker_id) if asker_id else None
    if current is None:
        raise ForbiddenError("the asker is not a current member of the group")
    if current.role in MANAGER_ROLES or not roster.has_managers():
        return
    raise ForbiddenError("only an owner or admin may manage the group")


def authorize_roster_change(
    roster: Roster, asker_id: str | None, events: Sequence[RosterEvent]
) -> None:
    """Raise unless ``asker_id`` may append ``events`` (section 5.1).

    Owners and admins may send any event; any member may send ``leave`` for
    themselves, but not backdated -- an event earlier than the newest one in
    the roster is a replay of history, which only managers may write.

    Raises:
        ForbiddenError: The asker may not send one of the events.
    """
    current = roster.current(asker_id) if asker_id else None
    if current is None:
        raise ForbiddenError("the asker is not a current member of the group")
    if current.role in MANAGER_ROLES or not roster.has_managers():
        return
    newest = max(
        (
            t
            for i in roster.intervals
            for t in (i.joined_at, i.left_at)
            if t is not None
        ),
        default=0,
    )
    for event in events:
        if event.op == "leave" and event.member_id == asker_id and event.at >= newest:
            continue
        raise ForbiddenError("only an owner or admin may change the roster")


def authorize_edit(
    roster: Roster, asker_id: str | None, *, sources: Collection[str]
) -> None:
    """Raise unless ``asker_id`` may change a group record resting on ``sources``.

    The forget rule (:func:`authorize_forget`) applied to ``POST
    /memory/update``: owners and admins may change any record of the group; a
    member, a record resting on their own messages alone.

    Raises:
        ForbiddenError: The asker is not a current member, or the record
            rests on someone else's messages.
    """
    current = roster.current(asker_id) if asker_id else None
    if current is None:
        raise ForbiddenError("the asker is not a current member of the group")
    if current.role in MANAGER_ROLES or not roster.has_managers():
        return
    if sources and set(sources) == {asker_id}:
        return
    raise ForbiddenError(
        "only an owner or admin may change what rests on another member's messages"
    )


def authorize_forget(
    roster: Roster, asker_id: str | None, *, sources: Collection[str]
) -> None:
    """Raise unless ``asker_id`` may forget what rests on ``sources`` (section 5.5).

    ``sources`` are the members whose messages the target is made of: the
    senders of the messages, the member of a member target, the speakers a
    record rests on, the posters of a picture. Owners and admins may forget
    anything in the group -- and, as for roster changes, any current member
    of a group that has no owner or admin at all. Anyone else may forget a
    target only when it rests on their own messages alone; a target with no
    sources (a whole session) is the managers'.

    Raises:
        ForbiddenError: The asker is not a current member, or the target
            rests on someone else's messages.
    """
    current = roster.current(asker_id) if asker_id else None
    if current is None:
        raise ForbiddenError("the asker is not a current member of the group")
    if current.role in MANAGER_ROLES or not roster.has_managers():
        return
    if sources and set(sources) == {asker_id}:
        return
    raise ForbiddenError(
        "only an owner or admin may forget what rests on another member's messages"
    )


class AccessPolicy:
    """Read scopes and buffer visibility over the groups of one memory root.

    Args:
        groups: Where rosters are read; the mirror is current the moment a
            roster write returns, so a removed member is refused at once.
        forgotten: Which messages of a session are forgotten; a buffer read
            leaves them out. ``None`` (a build without group forget) leaves
            the buffer as it is.
    """

    def __init__(
        self, groups: GroupStore, *, forgotten: MessageForgetStore | None = None
    ) -> None:
        self._groups = groups
        self._forgotten = forgotten

    async def read_scope(self, identity: Identity, request: ScopeRequest) -> ReadScope:
        """The compiled scope of one read.

        Raises:
            ForbiddenError: A personal scope read by someone else, or a group
                the asker is not a current member of (a group that does not
                exist answers the same, so a probe learns nothing).
            InvalidInputError: A group scope without an asker.
        """
        if request.project_scope:
            return await self._group_scope(
                identity, await self._project_groups(identity, request)
            )
        if request.group_ids:
            return await self._group_scope(identity, request)
        assert request.user_id is not None  # the request DTOs validate one scope
        owner_id = request.user_id
        if identity.asker_id is not None and identity.asker_id != owner_id:
            raise ForbiddenError("a personal memory is read by its owner only")
        return ReadScope(
            app_id=request.app_id,
            project_id=request.project_id,
            owner_type="user",
            owner_ids=(owner_id,),
            asker_id=identity.asker_id or owner_id,
            predicate=owner_predicate(
                owner_id=owner_id,
                owner_type="user",
                app_id=request.app_id,
                project_id=request.project_id,
            ),
        )

    async def project_groups(
        self, app_id: str, project_id: str, asker: str
    ) -> list[str]:
        """The groups of a project ``asker`` currently belongs to, sorted."""
        found: list[str] = []
        for group_id in await self._groups.group_ids(app_id, project_id):
            record = await self._groups.get(
                Scope(
                    app_id=app_id,
                    project_id=project_id,
                    owner_id=group_id,
                    owner_type="group",
                )
            )
            if record is not None and record.roster.is_current_member(asker):
                found.append(group_id)
        return sorted(found)

    async def _project_groups(
        self, identity: Identity, request: ScopeRequest
    ) -> ScopeRequest:
        """The request with its groups: the asker's current groups of the project.

        Raises:
            InvalidInputError: No asker.
            ForbiddenError: The asker is in none of the project's groups.
        """
        if identity.asker_id is None:
            raise InvalidInputError("asker_id is required for a project scope")
        groups = await self.project_groups(
            request.app_id, request.project_id, identity.asker_id
        )
        if not groups:
            raise ForbiddenError(
                "the asker is not a member of any group of the project"
            )
        return replace(request, group_ids=tuple(groups), project_scope=False)

    async def member_groups(
        self, personal: ReadScope, *, only: Collection[str] | None = None
    ) -> ReadScope | None:
        """The groups a personal read also covers when it asks to (``include_groups``).

        Every group of the read's space whose roster has its owner as a
        current member -- of those in ``only``, when given -- compiled
        exactly as a group search by the owner compiles it: the same
        windows (``history_visible_from`` honoured), the same status rules
        (superseded and forgotten records left out), the owner as the asker
        and the asker's own audience. A group the owner left, or never
        joined, adds nothing; an id in ``only`` that is not such a group is
        ignored, never refused, so the answer says nothing about it.

        Args:
            personal: The personal read scope :meth:`read_scope` compiled,
                which already checked that the asker is its owner.
            only: The groups to consider; ``None`` considers every group of
                the space.

        Returns:
            The groups' read scope, or ``None`` when the owner currently
            belongs to none of them.
        """
        if personal.owner_type != "user":
            raise ValueError("only a personal read covers its owner's groups")
        owner = personal.owner_id
        wanted = None if only is None else set(only)
        views: list[GroupView] = []
        for group_id in await self._groups.group_ids(
            personal.app_id, personal.project_id
        ):
            if wanted is not None and group_id not in wanted:
                continue
            record = await self._groups.get(
                Scope(
                    app_id=personal.app_id,
                    project_id=personal.project_id,
                    owner_id=group_id,
                    owner_type="group",
                )
            )
            if record is not None and record.roster.is_current_member(owner):
                views.append(_view_of(record, owner, "asker"))
        if not views:
            return None
        return _group_read_scope(
            personal.app_id, personal.project_id, owner, "asker", views
        )

    async def _group_scope(
        self, identity: Identity, request: ScopeRequest
    ) -> ReadScope:
        asker = identity.asker_id
        if asker is None:
            raise InvalidInputError("asker_id is required for a group scope")
        views: list[GroupView] = []
        for group_id in dict.fromkeys(request.group_ids):
            record = await self._groups.get(
                Scope(
                    app_id=request.app_id,
                    project_id=request.project_id,
                    owner_id=group_id,
                    owner_type="group",
                )
            )
            if record is None or not record.roster.is_current_member(asker):
                raise ForbiddenError(
                    f"the asker is not a current member of group {group_id!r}"
                )
            views.append(_view_of(record, asker, request.audience))
        return _group_read_scope(
            request.app_id, request.project_id, asker, request.audience, views
        )

    async def picture_visible(
        self,
        identity: Identity,
        references: Sequence[Stored[ImageMemory]],
        *,
        forgotten_in_group: bool = False,
    ) -> bool:
        """Whether a picture's bytes may be served, given its live memories (row 7).

        ``references`` are the current picture memories of one content hash,
        across every space. A hash only personal memories refer to behaves as
        before groups existed (served by hash). A hash a group picture memory
        refers to needs an asker who can see one of them -- or who has a
        personal memory of the same picture, which is theirs to see. A hash
        nothing refers to any more is served by hash too, unless a group
        forgot it (``forgotten_in_group``): then it is served to nobody, the
        same as a forgotten record is read by nobody.
        """
        groups = [r for r in references if r.scope.owner_type == "group"]
        if not groups:
            return bool(references) or not forgotten_in_group
        asker = identity.asker_id
        if asker is None:
            return False
        if any(
            r.scope.owner_type == "user" and r.scope.owner_id == asker
            for r in references
        ):
            return True
        for ref in groups:
            try:
                scope = await self.read_scope(
                    identity,
                    ScopeRequest(
                        app_id=ref.scope.app_id,
                        project_id=ref.scope.project_id,
                        group_ids=(ref.scope.owner_id,),
                    ),
                )
            except ForbiddenError:
                continue
            at = ref.item.timestamp
            if visible(
                scope,
                owner_type="group",
                owner_id=ref.scope.owner_id,
                source_start=at,
                source_end=at,
            ):
                return True
        return False

    async def buffered_messages(
        self,
        scope: ReadScope,
        session: SessionKey,
        messages: Sequence[CanonicalMessage],
        *,
        owner_has_memory: Callable[[], Awaitable[bool]],
    ) -> list[CanonicalMessage]:
        """The in-flight messages of ``session`` this read may see (row 3).

        A group session's buffer is read through a scope that holds its group,
        message by message against the asker's windows. A personal
        read never sees a group session's buffer, and sees another session's
        only when its owner took part in it -- a buffered message of theirs,
        or memory of theirs from that session (``owner_has_memory``).
        """
        bound = await self._groups.binding(session)
        if scope.is_group:
            view = scope.group(bound) if bound is not None else None
            if view is None:
                return []
            forgotten = (
                await self._forgotten.forgotten(session)
                if self._forgotten is not None
                else frozenset()
            )
            return [
                m
                for m in messages
                if (m.client_message_id or m.message_id) not in forgotten
                and any(
                    w.contains_point(to_timestamp_ms(m.timestamp)) for w in view.windows
                )
            ]
        if bound is not None:
            return []
        if any(m.sender_id == scope.owner_id for m in messages):
            return list(messages)
        return list(messages) if messages and await owner_has_memory() else []

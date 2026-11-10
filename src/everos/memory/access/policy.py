"""The access policy: who may read which memory, and who may change a group.

One module decides every read (docs/group-memory.md, D9 and section 9):

* :meth:`AccessPolicy.read_scope` turns an identity and a requested scope into
  a :class:`ReadScope` -- the owners, the asker's membership windows per group
  and the predicate every recall puts into its ``where``. Visibility is part of
  the index query, never a filter applied to results afterwards, so a ranking
  (the multi-round decider above all) never sees a candidate the asker cannot.
* :func:`visible` answers the same question for one record that does not come
  through the index (a markdown read); :meth:`AccessPolicy.picture_visible`
  for a picture's bytes, served by content hash.
* :meth:`AccessPolicy.buffered_messages` decides which in-flight messages of a
  session a read may see.
* :func:`authorize_roster_change` and :func:`authorize_manage` decide roster
  changes and meta updates.

Rules the implementation must keep (section 9.1):

* a personal scope is read by its owner only; a one-to-one request without an
  asker is read as asked by its owner (today's behaviour);
* a group is read by current members only (403 otherwise), and only records all
  of whose sources lie in one of the asker's windows: ``source_start >= from``
  and ``source_end < to`` (no upper bound for the open window);
* ``history_visible_from = creation`` gives a current member the whole history;
* ``audience = group`` keeps only what every current member can see;
* an owner or admin role grants roster management, never extra reading;
* an agent's cases learned in a group (``source_group_id``) are read only in
  that group's context, by a current member of it -- the agent itself unless
  another asker is named (:attr:`ReadScope.case_filter`).
"""

from __future__ import annotations

__all__ = [
    "AccessPolicy",
    "Audience",
    "GroupView",
    "ReadScope",
    "ScopeRequest",
    "Window",
    "authorize_manage",
    "authorize_roster_change",
    "intersect_windows",
    "owner_predicate",
    "visible",
]

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

from everos.core.errors import ForbiddenError, InvalidInputError
from everos.core.utils.datetime import from_timestamp_ms, to_timestamp_ms
from everos.memory.models import (
    MANAGER_ROLES,
    CanonicalMessage,
    Group,
    ImageMemory,
    Roster,
    RosterEvent,
)
from everos.memory.ports import (
    Filter,
    GroupStore,
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

    Exactly one of ``user_id`` / ``agent_id`` / ``group_ids`` is set; the
    request DTOs validate that.
    """

    app_id: str = "default"
    project_id: str = "default"
    user_id: str | None = None
    agent_id: str | None = None
    group_ids: tuple[str, ...] = ()
    audience: Audience = "asker"
    context_group_id: str | None = None
    """An agent scope read in a group's context: the group's cases join."""


@dataclass(frozen=True)
class ReadScope:
    """What one read may see, compiled.

    Attributes:
        app_id / project_id: The space.
        owner_type: ``user``, ``agent`` or ``group``.
        owner_ids: The one user or agent, or every group of the scope.
        asker_id: Who asks; the owner for a personal read without one.
        audience: ``asker`` or ``group`` (section 15).
        groups: Per group, its roster and the windows in force.
        predicate: The owner, space, status and window conditions -- what
            replaces the owner part of every recall's ``where``.
        case_filter: For an agent scope, the extra condition on its cases:
            cases learned outside groups, plus the context group's.
    """

    app_id: str
    project_id: str
    owner_type: OwnerType
    owner_ids: tuple[str, ...]
    asker_id: str | None
    predicate: Filter
    audience: Audience = "asker"
    groups: tuple[GroupView, ...] = field(default_factory=tuple)
    case_filter: Filter | None = None

    def cases_where(self, where: Filter) -> Filter:
        """``where`` narrowed to the agent cases this read may see."""
        return where if self.case_filter is None else all_of(where, self.case_filter)

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
    """The owner part of a personal or agent read, as recalls have always used it.

    A user read also excludes rows Reflection retired (``deprecated_by``); an
    agent's cases and skills carry no such column.
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


class AccessPolicy:
    """Read scopes and buffer visibility over the groups of one memory root.

    Args:
        groups: Where rosters are read; the mirror is current the moment a
            roster write returns, so a removed member is refused at once.
    """

    def __init__(self, groups: GroupStore) -> None:
        self._groups = groups

    async def read_scope(self, identity: Identity, request: ScopeRequest) -> ReadScope:
        """The compiled scope of one read.

        Raises:
            ForbiddenError: A personal scope read by someone else, or a group
                the asker is not a current member of (a group that does not
                exist answers the same, so a probe learns nothing).
            InvalidInputError: A group scope without an asker.
        """
        if request.group_ids:
            return await self._group_scope(identity, request)
        case_filter: Filter | None = None
        if request.user_id is not None:
            if identity.asker_id is not None and identity.asker_id != request.user_id:
                raise ForbiddenError("a personal memory is read by its owner only")
            owner_id, owner_type = request.user_id, "user"
        else:
            assert request.agent_id is not None
            owner_id, owner_type = request.agent_id, "agent"
            case_filter = await self._case_filter(
                request, asker=identity.asker_id or owner_id
            )
        return ReadScope(
            app_id=request.app_id,
            project_id=request.project_id,
            owner_type=owner_type,  # type: ignore[arg-type]
            owner_ids=(owner_id,),
            asker_id=identity.asker_id or owner_id,
            predicate=owner_predicate(
                owner_id=owner_id,
                owner_type=owner_type,
                app_id=request.app_id,
                project_id=request.project_id,
            ),
            case_filter=case_filter,
        )

    async def _case_filter(self, request: ScopeRequest, *, asker: str) -> Filter:
        """Which of an agent's cases a read sees (section 9.1, row 12).

        Cases learned outside any group always; a group's cases only in that
        group's context, and only for a current member of it.

        Raises:
            ForbiddenError: The asker is not a current member of the context
                group (or it does not exist, which answers the same).
        """
        outside = is_null("source_group_id")
        group_id = request.context_group_id
        if group_id is None:
            return outside
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
        return any_of(outside, eq("source_group_id", group_id))

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
            windows = _member_windows(record.group, record.roster, asker)
            if request.audience == "group":
                for member in record.roster.current_members():
                    windows = intersect_windows(
                        windows,
                        _member_windows(record.group, record.roster, member.member_id),
                    )
            views.append(
                GroupView(group=record.group, roster=record.roster, windows=windows)
            )
        per_group = [p for p in (view.predicate() for view in views) if p is not None]
        predicate = (
            all_of(
                eq("owner_type", "group"),
                eq("app_id", request.app_id),
                eq("project_id", request.project_id),
                is_null("deprecated_by"),
                any_of(*per_group),
            )
            if per_group
            else _NOTHING
        )
        return ReadScope(
            app_id=request.app_id,
            project_id=request.project_id,
            owner_type="group",
            owner_ids=tuple(view.group_id for view in views),
            asker_id=asker,
            predicate=predicate,
            audience=request.audience,
            groups=tuple(views),
        )

    async def picture_visible(
        self, identity: Identity, references: Sequence[Stored[ImageMemory]]
    ) -> bool:
        """Whether a picture's bytes may be served, given its live memories (row 7).

        ``references`` are the current picture memories of one content hash,
        across every space. A hash only personal memories refer to behaves as
        before groups existed (served by hash). A hash a group picture memory
        refers to needs an asker who can see one of them -- or who has a
        personal memory of the same picture, which is theirs to see.
        """
        groups = [r for r in references if r.scope.owner_type == "group"]
        if not groups:
            return True
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
        message by message against the asker's windows. A personal or agent
        read never sees a group session's buffer, and sees another session's
        only when its owner took part in it -- a buffered message of theirs,
        or memory of theirs from that session (``owner_has_memory``).
        """
        bound = await self._groups.binding(session)
        if scope.is_group:
            view = scope.group(bound) if bound is not None else None
            if view is None:
                return []
            return [
                m
                for m in messages
                if any(
                    w.contains_point(to_timestamp_ms(m.timestamp)) for w in view.windows
                )
            ]
        if bound is not None:
            return []
        if any(m.sender_id == scope.owner_id for m in messages):
            return list(messages)
        return list(messages) if messages and await owner_has_memory() else []

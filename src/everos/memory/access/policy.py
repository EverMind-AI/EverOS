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
* :meth:`AccessPolicy.past_predicate` widens a scope for a read of the past
  (``include_past``): a user's own memory with every version, superseded and
  forgotten; a group's with its superseded versions only -- never a forgotten
  record nor one a live forget retired, and always inside the asker's
  windows. It never widens what the asker may see beyond what their own
  history shows them.
* Which groups belong to a project is the project links' answer
  (:class:`~everos.memory.ports.ProjectLinkStore`, section 4.4), for a
  project scope and ``include_groups`` alike; the policy never lists a
  space's groups itself, and each group's roster still decides what the
  asker sees.
* :func:`visible` answers the same question for one record that does not come
  through the index (a markdown read); :meth:`AccessPolicy.picture_visible`
  for a picture's bytes, served by content hash.
* :meth:`AccessPolicy.buffered_messages` decides which in-flight messages of a
  session a read may see.
* :func:`authorize_roster_change` and :func:`authorize_manage` decide roster
  changes, meta updates and a group-scoped delete; :func:`authorize_forget`
  decides a forget and its undo (section 5.5); :func:`may_backdate` decides
  who may write a group's history (backdated events, an implicit join older
  than the group's newest message).

Rules the implementation must keep (section 9.1):

* a personal scope is read by its owner only; a one-to-one request without an
  asker is read as asked by its owner (today's behaviour);
* a group is read by current members only (403 otherwise), and only records all
  of whose sources lie in one of the asker's windows: ``source_start >= from``
  and ``source_end < to`` (no upper bound for the open window);
* ``history_visible_from = creation`` gives a current member the whole history;
* ``audience = group`` keeps only what every current human member can see
  (an assistant member is no audience);
* an owner or admin role grants roster management and forgetting, never extra
  reading; a member may forget only what rests on their own messages;
* a group with no owner or admin is managed by its deliverer alone (a member
  may still leave): a member never lifts their own window;
* a forgotten message is absent from every read of a session's buffer, as it
  is from every record a forget rewrote;
* a group's deliverer (:func:`is_deliverer`, section 5.1) is its superadmin:
  every right of an owner, the whole history to read, and alone to deliver
  (:func:`authorize_delivery`) -- never a member, so never one of the
  ``audience = group`` intersection; its search without an ``audience``
  reads as ``group``. One rule, consulted by every ``authorize_*`` function
  and by :meth:`AccessPolicy.read_scope`.
"""

from __future__ import annotations

__all__ = [
    "AccessPolicy",
    "Audience",
    "GroupView",
    "ReadScope",
    "ScopeRequest",
    "Window",
    "admits",
    "authorize_delivery",
    "authorize_forget",
    "authorize_manage",
    "authorize_roster_change",
    "intersect_windows",
    "is_deliverer",
    "may_backdate",
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
    MembershipInterval,
    Profile,
    Roster,
    RosterEvent,
)
from everos.memory.ports import (
    Filter,
    ForgetLogStore,
    GroupRecord,
    GroupStore,
    MessageForgetStore,
    OwnerType,
    ProjectLinkStore,
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
    retired_in,
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
    audience: Audience | None = "asker"
    """``None``: the asker's default, what a search that names no audience
    asks for -- ``group`` for a group's deliverer (:func:`is_deliverer`),
    ``asker`` for anyone else."""
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

    The group's deliverer reads the whole history. ``audience = group`` keeps
    only the time every current human member can see: an assistant member
    and the deliverer are never one of them (an assistant's implicit join at
    its first reply would otherwise hide everything said before it).
    """
    roster = record.roster
    windows = (
        (Window(start=None, end=None),)
        if is_deliverer(roster, asker)
        else _member_windows(record.group, roster, asker)
    )
    if audience == "group":
        for member in roster.current_members():
            if not _in_audience(roster, member):
                continue
            windows = intersect_windows(
                windows, _member_windows(record.group, roster, member.member_id)
            )
    return GroupView(group=record.group, roster=roster, windows=windows)


def _in_audience(roster: Roster, member: MembershipInterval) -> bool:
    """Whether a current member counts in ``audience = group``: a person who
    is not the group's deliverer (an assistant is no audience)."""
    return (
        member.kind == "human"
        and member.role != "assistant"
        and not is_deliverer(roster, member.member_id)
    )


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


def is_deliverer(roster: Roster, asker_id: str | None) -> bool:
    """Whether ``asker_id`` is the group's deliverer: its superadmin.

    The one rule behind what the deliverer may do in its own group
    (docs/group-memory.md, sections 5.1 and 9.1). Every ``authorize_*``
    function lets it do what an owner may: change the meta (its own key
    included) and the roster, backdated events too, forget, undo, edit and
    delete. A read scope gives it the whole history, with no window; its
    search without an ``audience`` reads as ``audience = group``, so a reply
    it posts into the group shows only what every current member can see.
    Only it may deliver (:func:`authorize_delivery`). It is no member:
    never a current member and never one of an audience.
    """
    return asker_id is not None and asker_id == roster.deliverer_id


def admits(roster: Roster, asker_id: str | None) -> bool:
    """Whether ``asker_id`` may act in the group at all: a current member, or
    the group's deliverer. Everyone else is a stranger to it (403)."""
    if asker_id is None:
        return False
    return roster.is_current_member(asker_id) or is_deliverer(roster, asker_id)


def authorize_delivery(roster: Roster, asker_id: str | None, *, group_id: str) -> None:
    """Raise unless ``asker_id`` may add to the group's sessions or flush them.

    Anyone may while the group has no deliverer; once it has one, only the
    deliverer -- the one program that forwards the group's chat under the
    group's key (section 5.1).

    Raises:
        ForbiddenError: The group has a deliverer and the asker is not it.
    """
    if roster.deliverer_id is None or is_deliverer(roster, asker_id):
        return
    raise ForbiddenError(
        f"only the deliverer of group {group_id!r} may add messages to its "
        "sessions or flush them"
    )


def may_backdate(roster: Roster, asker_id: str | None) -> bool:
    """Whether ``asker_id`` may write the group's history: an owner, an admin
    or the group's deliverer.

    Backdated roster events need it (:func:`authorize_roster_change`), and so
    does an implicit join older than the newest message the group holds: a
    batch from anyone else opens a newcomer's interval no earlier than that
    (:meth:`~everos.memory.groups.RosterManager.admit_batch`).
    """
    if is_deliverer(roster, asker_id):
        return True
    current = roster.current(asker_id) if asker_id else None
    return current is not None and current.role in MANAGER_ROLES


def authorize_manage(roster: Roster, asker_id: str | None) -> None:
    """Raise unless ``asker_id`` may change the group's meta or roster.

    Owners and admins may, and the group's deliverer. A group with no current
    owner or admin (one created by a batch without an asker) is managed by
    its deliverer alone: a member could otherwise lift their own window --
    open the history to every member, name themselves deliverer, or promote
    themselves and backdate a join. Its deliverer, or a manager the deliverer
    adds, manages it from then on.

    Raises:
        ForbiddenError: The asker is not a current member, or holds no
            manager role.
    """
    if may_backdate(roster, asker_id):
        return
    if (roster.current(asker_id) if asker_id else None) is None:
        raise ForbiddenError("the asker is not a current member of the group")
    raise ForbiddenError(_manager_refusal(roster, "manage the group"))


def authorize_roster_change(
    roster: Roster, asker_id: str | None, events: Sequence[RosterEvent]
) -> None:
    """Raise unless ``asker_id`` may append ``events`` (section 5.1).

    Owners and admins may send any event, as may the group's deliverer (it
    mirrors the chat platform's membership); any member may send ``leave``
    for themselves, but not backdated -- an event earlier than the newest one
    in the roster is a replay of history, which only managers may write. In a
    group with no owner or admin that is all a member may send: the rest is
    its deliverer's (:func:`authorize_manage`).

    Raises:
        ForbiddenError: The asker may not send one of the events.
    """
    if may_backdate(roster, asker_id):
        return
    if (roster.current(asker_id) if asker_id else None) is None:
        raise ForbiddenError("the asker is not a current member of the group")
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
        raise ForbiddenError(_manager_refusal(roster, "change the roster"))


def _manager_refusal(roster: Roster, action: str) -> str:
    """The refusal of a member who is not a manager, naming who may ``action``."""
    if roster.has_managers():
        return f"only an owner or admin may {action}"
    return (
        f"the group has no owner or admin: only its deliverer may {action} "
        "(a member may leave)"
    )


def authorize_edit(
    roster: Roster, asker_id: str | None, *, sources: Collection[str]
) -> None:
    """Raise unless ``asker_id`` may change a group record resting on ``sources``.

    The forget rule (:func:`authorize_forget`) applied to ``POST
    /memory/update``: owners, admins and the group's deliverer may change any
    record of the group; a member, a record resting on their own messages
    alone.

    Raises:
        ForbiddenError: The asker is not a current member, or the record
            rests on someone else's messages.
    """
    if is_deliverer(roster, asker_id):
        return
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
    record rests on, the posters of a picture. Owners, admins and the group's
    deliverer may forget anything in the group -- and, as for roster changes,
    any current member of a group that has no owner or admin at all. Anyone
    else may forget a target only when it rests on their own messages alone;
    a target with no sources (a whole session) is the managers'.

    Raises:
        ForbiddenError: The asker is not a current member, or the target
            rests on someone else's messages.
    """
    if is_deliverer(roster, asker_id):
        return
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
        projects: Which groups belong to a project (section 4.4); the
            rosters decide what the asker sees of them.
        forgotten: Which messages of a session are forgotten; a buffer read
            leaves them out. ``None`` (a build without group forget) leaves
            the buffer as it is.
        forget_log: The forgets; a group's read of the past leaves out what
            its live forgets retired. ``None`` reads a group's past as its
            present.
    """

    def __init__(
        self,
        groups: GroupStore,
        *,
        projects: ProjectLinkStore,
        forgotten: MessageForgetStore | None = None,
        forget_log: ForgetLogStore | None = None,
    ) -> None:
        self._groups = groups
        self._projects = projects
        self._forgotten = forgotten
        self._forget_log = forget_log

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
        for group in await self._groups_of(app_id, project_id):
            record = await self._groups.get(group)
            if record is not None and record.roster.is_current_member(asker):
                found.append(group.owner_id)
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

    async def _groups_of(self, app_id: str, project_id: str) -> list[Scope]:
        """The project's groups, as the project links answer (section 4.4).

        A read scope covers one ``(app_id, project_id)`` space, so a linked
        group stored in another space is refused, never read as empty.

        Raises:
            ValueError: The links answer a group outside the project's space.
        """
        groups = await self._projects.groups_of(app_id, project_id)
        for group in groups:
            if (group.app_id, group.project_id, group.owner_type) != (
                app_id,
                project_id,
                "group",
            ):
                raise ValueError(
                    f"project {project_id!r} links {group.owner_type} "
                    f"{group.owner_id!r} of {group.app_id!r}/{group.project_id!r}; "
                    "a read scope covers the project's own space only"
                )
        return groups

    async def member_groups(
        self, personal: ReadScope, *, only: Collection[str] | None = None
    ) -> ReadScope | None:
        """The groups a personal read also covers when it asks to (``include_groups``).

        Every group of the read's project (the project links' answer) whose
        roster has its owner as a current member -- of those in ``only``,
        when given -- compiled exactly as a group search by the owner
        compiles it: the same windows (``history_visible_from`` honoured),
        the same status rules (superseded and forgotten records left out),
        the owner as the asker and the asker's own audience. A group the
        owner left, or never joined, adds nothing; an id in ``only`` that is
        not such a group is ignored, never refused, so the answer says
        nothing about it.

        Args:
            personal: The personal read scope :meth:`read_scope` compiled,
                which already checked that the asker is its owner.
            only: The groups to consider; ``None`` considers every group of
                the project.

        Returns:
            The groups' read scope, or ``None`` when the owner currently
            belongs to none of them.
        """
        if personal.owner_type != "user":
            raise ValueError("only a personal read covers its owner's groups")
        owner = personal.owner_id
        wanted = None if only is None else set(only)
        views: list[GroupView] = []
        for group in await self._groups_of(personal.app_id, personal.project_id):
            if wanted is not None and group.owner_id not in wanted:
                continue
            record = await self._groups.get(group)
            if record is not None and record.roster.is_current_member(owner):
                views.append(_view_of(record, owner, "asker"))
        if not views:
            return None
        return _group_read_scope(
            personal.app_id, personal.project_id, owner, "asker", views
        )

    async def past_predicate(self, scope: ReadScope) -> Filter:
        """What a read of the past may see in ``scope`` (``include_past``).

        A user's own memory, whole: every version of every record, the
        superseded and the forgotten ones too -- what the user's own history
        lists. A group's, as its history shows it to the asker: the current
        and the superseded versions inside the asker's windows, never a
        forgotten record, nor one a live forget of the group retired (a
        version it marked or replaced, or one that led to it: it may say what
        was forgotten, and other members read the group). One forget-log read
        per group. Without a forget log a group's past is its present.
        """
        if not scope.is_group:
            return scope.scope_of(scope.owner_id).to_filter()
        if self._forget_log is None:
            return scope.predicate
        clauses: list[Filter] = []
        for view in scope.groups:
            clause = view.predicate()
            if clause is None:
                continue
            retired: set[str] = set()
            for entry in await self._forget_log.live_entries(
                app_id=scope.app_id,
                project_id=scope.project_id,
                owner_id=view.group_id,
                owner_type="group",
            ):
                retired |= retired_in(entry)
            clauses.append(
                all_of(clause, *(ne("entry_id", e) for e in sorted(retired)))
            )
        if not clauses:
            return _NOTHING
        return all_of(
            eq("owner_type", "group"),
            eq("app_id", scope.app_id),
            eq("project_id", scope.project_id),
            any_of(is_null("deprecated_by"), ne("deprecated_by", FORGOTTEN_MARK)),
            any_of(*clauses),
        )

    async def forgotten_pictures(self, scope: ReadScope) -> dict[str, frozenset[str]]:
        """The pictures each group of ``scope`` forgot, by group id.

        The content hashes of the group's live picture forgets (an undo gives
        one back). A group's segment keeps the picture's reference in its
        payload, so a reader that hands segment text to the asker -- a
        constraint's quote -- takes these out first, as the group's
        re-extraction and profiles do. One forget-log read per group; nothing
        for a personal scope or without a forget log.
        """
        if not scope.is_group or self._forget_log is None:
            return {}
        found: dict[str, frozenset[str]] = {}
        for view in scope.groups:
            hashes = await self._forget_log.live_targets(
                target="image",
                app_id=scope.app_id,
                project_id=scope.project_id,
                owner_id=view.group_id,
                owner_type="group",
            )
            if hashes:
                found[view.group_id] = frozenset(hashes)
        return found

    async def _group_scope(
        self, identity: Identity, request: ScopeRequest
    ) -> ReadScope:
        """The groups' read scope, for a current member of each (or its deliverer).

        An ``audience`` left to the asker's default (``None``) is ``group``
        when the asker is the deliverer of a group of the scope, ``asker``
        otherwise.
        """
        asker = identity.asker_id
        if asker is None:
            raise InvalidInputError("asker_id is required for a group scope")
        records: list[GroupRecord] = []
        for group_id in dict.fromkeys(request.group_ids):
            record = await self._groups.get(
                Scope(
                    app_id=request.app_id,
                    project_id=request.project_id,
                    owner_id=group_id,
                    owner_type="group",
                )
            )
            if record is None or not admits(record.roster, asker):
                raise ForbiddenError(
                    f"the asker is not a current member of group {group_id!r}"
                )
            records.append(record)
        audience: Audience = request.audience or (
            "group"
            if any(is_deliverer(record.roster, asker) for record in records)
            else "asker"
        )
        views = [_view_of(record, asker, audience) for record in records]
        return _group_read_scope(
            request.app_id, request.project_id, asker, audience, views
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
        or memory of theirs from that session (``owner_has_memory``). A
        forgotten message is seen by neither.
        """
        bound = await self._groups.binding(session)
        if scope.is_group:
            view = scope.group(bound) if bound is not None else None
            if view is None:
                return []
            return [
                m
                for m in await self._remembered(session, messages)
                if any(
                    w.contains_point(to_timestamp_ms(m.timestamp)) for w in view.windows
                )
            ]
        if bound is not None:
            return []
        messages = await self._remembered(session, messages)
        if any(m.sender_id == scope.owner_id for m in messages):
            return list(messages)
        return list(messages) if messages and await owner_has_memory() else []

    async def _remembered(
        self, session: SessionKey, messages: Sequence[CanonicalMessage]
    ) -> list[CanonicalMessage]:
        """``messages`` without the ones a live forget names."""
        if not messages or self._forgotten is None:
            return list(messages)
        forgotten = await self._forgotten.forgotten(session)
        return [
            m
            for m in messages
            if (m.client_message_id or m.message_id) not in forgotten
        ]

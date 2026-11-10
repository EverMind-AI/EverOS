"""``group.md`` <-> roster rows: the one mapping the store and the cascade share.

The store writes ``group.md`` and mirrors it into SQLite in one operation; the
cascade re-derives the same rows when the file changes or the index is
rebuilt. Both go through these functions, so a rebuilt mirror is identical to
the one the write produced (docs/group-memory.md, section 14, task 1).

``group.md`` shape (section 6.1)::

    ---
    id: group_<group_id>
    type: group
    group_id: <group_id>
    track: group
    name / description / created_at / created_by / history_visible_from / extra
    sessions: [{session_id, bound_at}]
    entry_count: <n>
    ---
    <!-- entry:gr_<YYYYMMDD>_<seq> -->
    **op**: join
    **member_id**: u_chen
    ...
    <!-- /entry:gr_<YYYYMMDD>_<seq> -->

Times are ISO-8601 in UTC in markdown and epoch milliseconds everywhere else.
``aliases`` is a JSON list, because a name may contain the comma the plain
inline list form splits on.
"""

from __future__ import annotations

__all__ = [
    "binding_entry",
    "event_date",
    "event_from_entry",
    "event_inline",
    "group_from_frontmatter",
    "group_frontmatter",
    "mirror_rows",
    "record_from_rows",
    "sessions_from_frontmatter",
]

import datetime as _dt
import json
from collections.abc import Mapping, Sequence
from typing import Any, cast

from everos.core.utils.datetime import (
    ensure_utc,
    from_iso_format,
    from_timestamp_ms,
    to_timestamp_ms,
)
from everos.infra.persistence.markdown import Entry
from everos.infra.persistence.sqlite import GroupMember, GroupMeta, SessionBinding
from everos.memory.models import (
    Group,
    MembershipInterval,
    Roster,
    RosterEvent,
)
from everos.memory.ports import GroupRecord

_EVENT_FIELDS = ("kind", "display_name", "role", "title")


def _utc(ms: int) -> _dt.datetime:
    moment = ensure_utc(from_timestamp_ms(ms))
    assert moment is not None  # a concrete timestamp always converts
    return moment


def _iso(ms: int) -> str:
    return _utc(ms).isoformat()


def _ms(raw: object) -> int:
    return to_timestamp_ms(from_iso_format(str(raw)))


def event_date(event: RosterEvent) -> _dt.date:
    """The UTC date an event's entry id is minted under."""
    return _utc(event.at).date()


def event_inline(event: RosterEvent) -> dict[str, object]:
    """One event as ``group.md`` inline fields; unset fields are not written."""
    inline: dict[str, object] = {"op": event.op, "member_id": event.member_id}
    for name in _EVENT_FIELDS:
        value = getattr(event, name)
        if value is not None:
            inline[name] = value
    if event.aliases is not None:
        inline["aliases"] = json.dumps(event.aliases, ensure_ascii=False)
    inline["at"] = _iso(event.at)
    inline["source"] = event.source
    return inline


def event_from_entry(entry: Entry) -> RosterEvent:
    """The event an entry written by :func:`event_inline` holds.

    Raises:
        ValueError: A required field is missing or malformed (a hand edit).
    """
    inline = entry.as_structured().inline
    try:
        aliases = json.loads(inline["aliases"]) if "aliases" in inline else None
        return RosterEvent(
            op=inline["op"],  # type: ignore[arg-type]
            member_id=inline["member_id"],
            at=_ms(inline["at"]),
            source=inline.get("source") or "api",  # type: ignore[arg-type]
            kind=inline.get("kind") or None,  # type: ignore[arg-type]
            display_name=inline.get("display_name") or None,
            aliases=list(aliases) if aliases is not None else None,
            role=inline.get("role") or None,  # type: ignore[arg-type]
            title=inline.get("title"),
            event_id=entry.id,
        )
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError(f"unreadable roster entry {entry.id!r}: {exc}") from exc


def group_frontmatter(
    group: Group, *, sessions: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """The frontmatter ``group.md`` carries for ``group`` and its sessions."""
    meta: dict[str, Any] = {
        "id": f"group_{group.group_id}",
        "type": "group",
        "schema_version": 1,
        "group_id": group.group_id,
        "track": "group",
        "name": group.name,
        "description": group.description,
        "created_at": _iso(group.created_at),
    }
    if group.created_by is not None:
        meta["created_by"] = group.created_by
    meta["history_visible_from"] = group.history_visible_from
    if group.extra:
        meta["extra"] = dict(group.extra)
    meta["sessions"] = [dict(s) for s in sessions]
    return meta


def group_from_frontmatter(meta: Mapping[str, Any], *, group_id: str) -> Group:
    """The :class:`Group` a ``group.md`` frontmatter describes.

    ``group_id`` is the file's directory: where the file lives is what the
    group is, so a hand-edited ``group_id`` field cannot move a group.
    """
    extra = meta.get("extra")
    return Group(
        group_id=group_id,
        name=str(meta.get("name") or ""),
        description=str(meta.get("description") or ""),
        created_at=_ms(meta["created_at"]),
        created_by=meta.get("created_by"),
        extra=dict(extra) if isinstance(extra, dict) else {},
        history_visible_from=meta.get("history_visible_from") or "join",
    )


def sessions_from_frontmatter(meta: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The ``sessions`` list, tolerating an absent or malformed one."""
    raw = meta.get("sessions")
    if not isinstance(raw, list):
        return []
    return [
        dict(item)
        for item in raw
        if isinstance(item, dict) and item.get("session_id") and item.get("bound_at")
    ]


def mirror_rows(
    *,
    app_id: str,
    project_id: str,
    group: Group,
    intervals: Sequence[MembershipInterval],
    sessions: Sequence[Mapping[str, Any]],
    md_path: str,
) -> tuple[GroupMeta, list[GroupMember], list[SessionBinding]]:
    """The SQLite rows that mirror one ``group.md``."""
    meta = GroupMeta(
        app_id=app_id,
        project_id=project_id,
        group_id=group.group_id,
        name=group.name,
        description=group.description,
        group_created_at=group.created_at,
        created_by=group.created_by,
        history_visible_from=group.history_visible_from,
        extra_json=json.dumps(group.extra, ensure_ascii=False, sort_keys=True),
        md_path=md_path,
    )
    members = [
        GroupMember(
            app_id=app_id,
            project_id=project_id,
            group_id=group.group_id,
            member_id=interval.member_id,
            seq=interval.seq,
            kind=interval.kind,
            display_name=interval.display_name,
            aliases_json=json.dumps(interval.aliases, ensure_ascii=False),
            role=interval.role,
            title=interval.title,
            joined_at=interval.joined_at,
            left_at=interval.left_at,
            source=interval.source,
        )
        for interval in intervals
    ]
    bindings = [
        SessionBinding(
            app_id=app_id,
            project_id=project_id,
            session_id=str(item["session_id"]),
            group_id=group.group_id,
            bound_at=_ms(item["bound_at"]),
        )
        for item in sessions
    ]
    return meta, members, bindings


def record_from_rows(
    meta: GroupMeta,
    members: Sequence[GroupMember],
    bindings: Sequence[SessionBinding],
) -> GroupRecord:
    """The :class:`GroupRecord` the mirror rows hold."""
    extra = json.loads(meta.extra_json or "{}")
    group = Group(
        group_id=meta.group_id,
        name=meta.name,
        description=meta.description,
        created_at=meta.group_created_at,
        created_by=meta.created_by,
        extra=extra if isinstance(extra, dict) else {},
        history_visible_from=cast(Any, meta.history_visible_from),
    )
    intervals = [
        MembershipInterval(
            member_id=row.member_id,
            seq=row.seq,
            kind=cast(Any, row.kind),
            display_name=row.display_name,
            aliases=list(json.loads(row.aliases_json or "[]")),
            role=cast(Any, row.role),
            title=row.title,
            joined_at=row.joined_at,
            left_at=row.left_at,
            source=cast(Any, row.source),
        )
        for row in members
    ]
    return GroupRecord(
        group=group,
        roster=Roster(intervals=intervals),
        sessions=tuple(b.session_id for b in bindings),
    )


def binding_entry(session_id: str, bound_at: int) -> dict[str, Any]:
    """One ``sessions`` item of the frontmatter."""
    return {"session_id": session_id, "bound_at": _iso(bound_at)}

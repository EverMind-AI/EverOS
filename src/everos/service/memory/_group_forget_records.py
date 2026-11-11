"""What a forget inside a group answers with, and what its undo is kept as.

docs/group-memory.md, section 9.4. :class:`GroupForgetResult` and
:class:`GroupUndoResult` are the use case's answers; :class:`Applied` and
:func:`undo_record` are what a run did and the JSON the forget log keeps of
it (``undo_json``) -- everything an undo needs: the entries the run took out
of markdown, byte for byte, the entries it wrote, the member-profile items
it removed, the segments it queued, the picture files it released and the
scene clusters it changed. Once a run took the previous versions out of the
files, the forget log is the only place they are kept.
"""

from __future__ import annotations

__all__ = [
    "Applied",
    "GroupForgetResult",
    "GroupUndoResult",
    "forget_result",
    "merge_withdrawals",
    "row_ids",
    "undo_record",
    "with_items",
    "withdrawal_from",
    "withdrawal_json",
    "withdrawn_ref",
    "without_items",
]

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from everos.memory.models import ExplicitInfoItem, ImplicitTraitItem, MemberProfile
from everos.memory.ports import MemoryRef, Scope, Stored, Withdrawal, WithdrawnEntry

from ._group_forget_plan import ForgetPlan, GroupForgetTarget, MemberEdit


@dataclass(frozen=True)
class GroupForgetResult:
    """What one forget inside a group touched (or, for a dry run, would touch).

    Every list names a record as the read side does (``<group>_<entry>``),
    except ``messages`` (``(session_id, message_id)``), ``member_profiles``
    (member ids), ``images`` (content hashes) and the member-profile item ids.

    Attributes:
        forget_id: The log entry (``""`` for a dry run).
        dry_run: Whether nothing was changed.
        target: What the forget addressed.
        messages: The messages marked forgotten.
        episodes: Episode versions taken out (each rewritten or dropped).
        rewritten: How many segments got a new episode version.
        new_episodes: Their ids (a real run only).
        image_memories: Picture memories taken out.
        images: Picture files taken out of circulation.
        scenes: Scene versions taken out.
        rewritten_scenes: How many scenes got a new version.
        new_scenes: Their ids (a real run only).
        member_profiles: Members whose whole profile in the group went.
        member_profile_items: Member-profile items forgotten.
        stale_member_profile_items: Member-profile items taken out and queued
            for a rewrite from their remaining segments.
        group_profile_items: Group-profile items forgotten.
        stale_group_profile_items: Group-profile items taken out and queued
            for a rewrite.
    """

    forget_id: str
    dry_run: bool
    target: GroupForgetTarget
    messages: tuple[tuple[str, str], ...] = ()
    episodes: tuple[str, ...] = ()
    rewritten: int = 0
    new_episodes: tuple[str, ...] = ()
    image_memories: tuple[str, ...] = ()
    images: tuple[str, ...] = ()
    scenes: tuple[str, ...] = ()
    rewritten_scenes: int = 0
    new_scenes: tuple[str, ...] = ()
    member_profiles: tuple[str, ...] = ()
    member_profile_items: tuple[str, ...] = ()
    stale_member_profile_items: tuple[str, ...] = ()
    group_profile_items: tuple[str, ...] = ()
    stale_group_profile_items: tuple[str, ...] = ()


@dataclass(frozen=True)
class GroupUndoResult:
    """What one undo of a forget inside a group put back.

    Attributes:
        forget_id: The forget reversed.
        applied: ``False`` when it was undone already (nothing restored).
        messages: Messages that are no longer forgotten by it.
        episodes: Episode versions put back.
        image_memories: Picture memories put back.
        images: Picture files back in circulation.
        scenes: Scene versions put back.
        member_profiles: Members whose profile document came back.
        member_profile_items: Member-profile items put back.
        group_profile_items: Group-profile items put back.
        withdrawn: Records the forget had written, now taken out.
    """

    forget_id: str
    applied: bool
    messages: tuple[tuple[str, str], ...] = ()
    episodes: tuple[str, ...] = ()
    image_memories: tuple[str, ...] = ()
    images: tuple[str, ...] = ()
    scenes: tuple[str, ...] = ()
    member_profiles: tuple[str, ...] = ()
    member_profile_items: tuple[str, ...] = ()
    group_profile_items: tuple[str, ...] = ()
    withdrawn: tuple[str, ...] = ()


@dataclass
class Applied:
    """What the run did so far: the undo record, built as it goes."""

    written_episodes: list[MemoryRef] = field(default_factory=list)
    written_scenes: list[MemoryRef] = field(default_factory=list)
    withdrawn: dict[str, Withdrawal] = field(default_factory=dict)
    members_removed: list[dict[str, Any]] = field(default_factory=list)
    member_items: list[dict[str, Any]] = field(default_factory=list)
    group_queued: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    clusters: list[dict[str, Any]] = field(default_factory=list)


def withdrawn_ref(scope: Scope, entry: WithdrawnEntry) -> MemoryRef:
    return MemoryRef(
        id=f"{scope.owner_id}_{entry.entry_id}",
        entry_id=entry.entry_id,
        md_path=entry.md_path,
    )


def row_ids(scope: Scope, refs: Iterable[MemoryRef]) -> tuple[str, ...]:
    return tuple(f"{scope.owner_id}_{ref.entry_id}" for ref in refs)


def merge_withdrawals(left: Withdrawal, right: Withdrawal) -> Withdrawal:
    return Withdrawal(
        entries=(*left.entries, *right.entries),
        released=(*left.released, *right.released),
    )


def withdrawal_json(withdrawal: Withdrawal) -> dict[str, Any]:
    return {
        "entries": [
            {
                "entry_id": e.entry_id,
                "md_path": e.md_path,
                "text": e.text,
                "mark": e.mark,
            }
            for e in withdrawal.entries
        ],
        "released": [list(r) for r in withdrawal.released],
    }


def withdrawal_from(data: Mapping[str, Any]) -> Withdrawal:
    return Withdrawal(
        entries=tuple(
            WithdrawnEntry(
                entry_id=str(e["entry_id"]),
                md_path=str(e["md_path"]),
                text=str(e["text"]),
                mark=e.get("mark"),
            )
            for e in data.get("entries") or []
        ),
        released=tuple(
            (str(r[0]), str(r[1]), str(r[2])) for r in data.get("released") or []
        ),
    )


def undo_record(plan: ForgetPlan, applied: Applied) -> dict[str, Any]:
    """What an undo of the run needs, as JSON for the forget log."""
    return {
        "target": plan.target.to_json(),
        "sources": sorted(plan.sources),
        "messages": [
            [session.session_id, list(ids)] for session, ids in plan.messages.items()
        ],
        "written": {
            "episode": [ref.entry_id for ref in applied.written_episodes],
            "scene": [ref.entry_id for ref in applied.written_scenes],
        },
        "withdrawn": {
            kind: withdrawal_json(w) for kind, w in applied.withdrawn.items()
        },
        # Named apart so the picture-bytes check finds a hash by key.
        "pictures": [{"image_id": m.item.image_id} for m in plan.image_memories],
        "members_removed": applied.members_removed,
        "member_items": applied.member_items,
        "group_queued": applied.group_queued,
        "files": applied.files,
        "clusters": applied.clusters,
    }


def without_items(
    held: MemberProfile, edit: MemberEdit
) -> tuple[MemberProfile, dict[str, Any]]:
    """The profile without the edit's items; and what an undo puts back."""
    gone = set(edit.forgotten) | set(edit.stale)
    profile = held.profile

    def keep(item: Any) -> bool:
        return (item.model_extra or {}).get("item_id") not in gone

    explicit = [i for i in profile.explicit_info if not keep(i)]
    implicit = [i for i in profile.implicit_traits if not keep(i)]
    history_gone = set(edit.history)
    cleared = bool(explicit or implicit)
    kept = held.model_copy(
        update={
            "profile": profile.model_copy(
                update={
                    "explicit_info": [i for i in profile.explicit_info if keep(i)],
                    "implicit_traits": [i for i in profile.implicit_traits if keep(i)],
                    # Written over every item: it may say what was forgotten.
                    "summary": "" if cleared else profile.summary,
                }
            ),
            "history": [h for n, h in enumerate(held.history) if n not in history_gone],
            "pending_memcell_ids": [*held.pending_memcell_ids, *edit.queue],
        }
    )
    record = {
        "member_id": held.member_id,
        "explicit_info": [i.model_dump(mode="json") for i in explicit],
        "implicit_traits": [i.model_dump(mode="json") for i in implicit],
        "history": [held.history[n] for n in sorted(history_gone)],
        "summary": profile.summary if cleared else None,
        "queued": list(edit.queue),
    }
    return kept, record


def with_items(
    held: MemberProfile, change: Mapping[str, Any]
) -> tuple[MemberProfile, list[str]]:
    """The profile with an undo's items back, and their ids."""
    profile = held.profile
    present = {
        (i.model_extra or {}).get("item_id")
        for i in (*profile.explicit_info, *profile.implicit_traits)
    }
    explicit = [
        ExplicitInfoItem.model_validate(i)
        for i in change.get("explicit_info") or []
        if i.get("item_id") not in present
    ]
    implicit = [
        ImplicitTraitItem.model_validate(i)
        for i in change.get("implicit_traits") or []
        if i.get("item_id") not in present
    ]
    queued = set(change.get("queued") or [])
    summary = change.get("summary")
    restored = held.model_copy(
        update={
            "profile": profile.model_copy(
                update={
                    "explicit_info": [*profile.explicit_info, *explicit],
                    "implicit_traits": [*profile.implicit_traits, *implicit],
                    "summary": summary
                    if summary is not None and not profile.summary
                    else profile.summary,
                }
            ),
            "history": [*(change.get("history") or []), *held.history],
            "pending_memcell_ids": [
                m for m in held.pending_memcell_ids if m not in queued
            ],
        }
    )
    ids = [
        str((i.model_extra or {}).get("item_id") or "") for i in (*explicit, *implicit)
    ]
    return restored, ids


def forget_result(
    plan: ForgetPlan,
    *,
    forget_id: str,
    dry_run: bool,
    applied: Applied | None = None,
) -> GroupForgetResult:
    """The response's view of a plan (and of what the run wrote)."""
    group = plan.scope.owner_id

    def rows(stored: Iterable[Stored[Any]]) -> tuple[str, ...]:
        return tuple(dict.fromkeys(f"{group}_{s.ref.entry_id}" for s in stored))

    member_items = tuple(i for e in plan.members for i in e.forgotten)
    stale_items = tuple(i for e in plan.members for i in e.stale)
    scenes_written = sum(1 for e in plan.scenes if e.members)
    return GroupForgetResult(
        forget_id=forget_id,
        dry_run=dry_run,
        target=plan.target,
        messages=tuple(
            (session.session_id, message_id)
            for session, ids in plan.messages.items()
            for message_id in ids
        ),
        episodes=rows([*(r.old for r in plan.rewrites), *plan.dropped]),
        rewritten=len(plan.rewrites)
        if applied is None
        else len(applied.written_episodes),
        new_episodes=()
        if applied is None
        else tuple(f"{group}_{ref.entry_id}" for ref in applied.written_episodes),
        image_memories=rows(plan.image_memories),
        images=tuple(plan.files),
        scenes=rows(v for e in plan.scenes for v in e.versions),
        rewritten_scenes=scenes_written
        if applied is None
        else len(applied.written_scenes),
        new_scenes=()
        if applied is None
        else tuple(f"{group}_{ref.entry_id}" for ref in applied.written_scenes),
        member_profiles=tuple(plan.members_removed),
        member_profile_items=member_items,
        stale_member_profile_items=stale_items,
        group_profile_items=rows(plan.group_forgotten),
        stale_group_profile_items=rows(plan.group_stale),
    )

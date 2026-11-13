"""``GroupStore`` over ``group.md`` and the roster mirror in ``system.db``.

Writes hold the group's owner lock (``LOCKS`` kind ``"owner"``, the key every
markdown write of that owner holds) across the whole read-fold-write: the log
already in ``group.md`` is read, folded together with the new events (a
conflict raises before anything is written), the entries are appended, and the
mirror rows are replaced -- so the markdown and the rows change together and a
removed member loses access when :meth:`GroupRosterStore.save` returns.

Reads come from the mirror. A group whose ``group.md`` exists but whose rows do
not (a wiped index not yet rebuilt) is re-mirrored from the file on first read
rather than reported missing: answering "no such group" there would let an
implicit create overwrite the real group's meta.
"""

from __future__ import annotations

__all__ = ["GroupRosterStore"]

from collections.abc import Sequence
from pathlib import Path

from everos.core.errors import RosterConflictError
from everos.core.locks import LOCKS, owner_key
from everos.infra.persistence.markdown import GroupWriter
from everos.infra.persistence.memory_root import MemoryRoot
from everos.infra.persistence.sqlite import GroupRosterRepo, SessionBinding
from everos.memory.models import Group, Roster, RosterEvent, fold_roster
from everos.memory.ports import GroupRecord, Scope, SessionKey

from .projection import (
    binding_entry,
    event_date,
    event_from_entry,
    event_inline,
    group_from_frontmatter,
    group_frontmatter,
    mirror_rows,
    record_from_rows,
    sessions_from_frontmatter,
)


class GroupRosterStore:
    """:class:`~everos.memory.ports.GroupStore` over one memory root.

    Args:
        root: Memory root the groups live under.
        repo: The roster mirror of the root's ``system.db``.
    """

    def __init__(self, root: MemoryRoot, repo: GroupRosterRepo) -> None:
        self._root = root
        self._repo = repo
        self._writer = GroupWriter(root)

    async def get(self, scope: Scope) -> GroupRecord | None:
        rows = await self._repo.load(scope.app_id, scope.project_id, scope.owner_id)
        if rows is not None:
            return record_from_rows(*rows)
        if await self._writer.read(self._path(scope)) is None:
            return None
        return await self.remirror(scope)

    async def group_ids(self, app_id: str, project_id: str) -> list[str]:
        return await self._repo.group_ids(app_id, project_id)

    async def save(
        self, scope: Scope, *, group: Group, events: Sequence[RosterEvent]
    ) -> GroupRecord:
        path = self._path(scope)
        async with LOCKS.hold("owner", self._lock(scope)):
            existing = await self._writer.read(path)
            old_events = [event_from_entry(e) for e in existing[1]] if existing else []
            sessions = sessions_from_frontmatter(existing[0]) if existing else []
            intervals = fold_roster([*old_events, *events])
            await self._writer.append(
                path,
                frontmatter=group_frontmatter(
                    group,
                    sessions=sessions,
                    previous=existing[0] if existing else None,
                ),
                entries=[(event_date(e), event_inline(e)) for e in events],
            )
            meta, members, bindings = mirror_rows(
                app_id=scope.app_id,
                project_id=scope.project_id,
                group=group,
                intervals=intervals,
                sessions=sessions,
                md_path=self._relative(path),
            )
            await self._repo.replace(meta, members, bindings)
        return GroupRecord(
            group=group,
            roster=Roster(intervals=intervals),
            sessions=tuple(b.session_id for b in bindings),
        )

    async def binding(self, session: SessionKey) -> str | None:
        row = await self._repo.binding(
            session.app_id, session.project_id, session.session_id
        )
        return row.group_id if row is not None else None

    async def bind(self, session: SessionKey, group_id: str, *, at: int) -> str:
        bound = await self._repo.bind(
            SessionBinding(
                app_id=session.app_id,
                project_id=session.project_id,
                session_id=session.session_id,
                group_id=group_id,
                bound_at=at,
            )
        )
        if bound != group_id:
            return bound
        scope = Scope(
            app_id=session.app_id,
            project_id=session.project_id,
            owner_id=group_id,
            owner_type="group",
        )
        path = self._path(scope)
        # The durable copy: a rebuilt mirror restores the binding from here.
        async with LOCKS.hold("owner", self._lock(scope)):
            existing = await self._writer.read(path)
            if existing is not None:
                sessions = sessions_from_frontmatter(existing[0])
                if all(s["session_id"] != session.session_id for s in sessions):
                    sessions.append(binding_entry(session.session_id, at))
                    await self._writer.append(
                        path, frontmatter={"sessions": sessions}, entries=[]
                    )
        return group_id

    async def remirror(self, scope: Scope) -> GroupRecord | None:
        """Re-derive the group's rows from ``group.md``; ``None`` without one.

        Raises:
            ValueError: The log no longer folds (a hand edit broke it), or an
                entry is unreadable.
        """
        path = self._path(scope)
        async with LOCKS.hold("owner", self._lock(scope)):
            existing = await self._writer.read(path)
            if existing is None:
                await self._repo.remove(scope.app_id, scope.project_id, scope.owner_id)
                return None
            meta_fm, entries = existing
            group = group_from_frontmatter(meta_fm, group_id=scope.owner_id)
            sessions = sessions_from_frontmatter(meta_fm)
            try:
                intervals = fold_roster([event_from_entry(e) for e in entries])
            except RosterConflictError as exc:  # a hand edit broke the log
                raise ValueError(f"{self._relative(path)}: {exc}") from exc
            meta, members, bindings = mirror_rows(
                app_id=scope.app_id,
                project_id=scope.project_id,
                group=group,
                intervals=intervals,
                sessions=sessions,
                md_path=self._relative(path),
            )
            await self._repo.replace(meta, members, bindings)
        return GroupRecord(
            group=group,
            roster=Roster(intervals=intervals),
            sessions=tuple(b.session_id for b in bindings),
        )

    def _path(self, scope: Scope) -> Path:
        return self._writer.path_for(
            scope.owner_id, app_id=scope.app_id, project_id=scope.project_id
        )

    def _relative(self, path: Path) -> str:
        return path.relative_to(self._root.root).as_posix()

    @staticmethod
    def _lock(scope: Scope) -> str:
        return owner_key(scope.app_id, scope.project_id, "group", scope.owner_id)

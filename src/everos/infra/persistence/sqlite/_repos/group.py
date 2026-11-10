"""Repository for the roster mirror (``group_meta`` / ``group_member`` / bindings).

The mirror is replaced, never patched: :meth:`GroupRosterRepo.replace` writes
one group's meta and *all* of its intervals in one transaction, so a reader
never sees half of a roster change. The rows are derived from ``group.md``
(the caller folds the log), which is why the cascade can rebuild them with the
same call.

A session binding is inserted once and never changed: :meth:`bind` answers the
group the session is bound to after the call, which is the earlier binding's
group when there was one (D18: a session belongs to at most one group).
"""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import delete, select
from sqlalchemy.dialects.sqlite import insert

from everos.infra.persistence.sqlite._base.repository import RepoBase
from everos.infra.persistence.sqlite._base.session import session_scope

from .._tables import GroupMember, GroupMeta, SessionBinding


class GroupRosterRepo(RepoBase[GroupMeta]):
    """The three roster tables of one ``system.db``."""

    model = GroupMeta

    async def load(
        self, app_id: str, project_id: str, group_id: str
    ) -> tuple[GroupMeta, list[GroupMember], list[SessionBinding]] | None:
        """One group's rows, or ``None`` when the group has no meta row."""
        async with session_scope(self._factory) as s:
            meta = await s.get(GroupMeta, (app_id, project_id, group_id))
            if meta is None:
                return None
            members = (
                (
                    await s.execute(
                        select(GroupMember)
                        .where(
                            GroupMember.app_id == app_id,  # type: ignore[arg-type]
                            GroupMember.project_id == project_id,  # type: ignore[arg-type]
                            GroupMember.group_id == group_id,  # type: ignore[arg-type]
                        )
                        .order_by(GroupMember.member_id, GroupMember.seq)  # type: ignore[arg-type]
                    )
                )
                .scalars()
                .all()
            )
            bindings = (
                (
                    await s.execute(
                        select(SessionBinding)
                        .where(
                            SessionBinding.app_id == app_id,  # type: ignore[arg-type]
                            SessionBinding.project_id == project_id,  # type: ignore[arg-type]
                            SessionBinding.group_id == group_id,  # type: ignore[arg-type]
                        )
                        .order_by(SessionBinding.bound_at, SessionBinding.session_id)  # type: ignore[arg-type]
                    )
                )
                .scalars()
                .all()
            )
            return meta, list(members), list(bindings)

    async def replace(
        self,
        meta: GroupMeta,
        members: Sequence[GroupMember],
        bindings: Sequence[SessionBinding] = (),
    ) -> None:
        """Write the group's meta and every interval; add missing bindings.

        One transaction. Bindings already present are kept as they are, so a
        rebuild never rebinds a session a later batch bound elsewhere.
        """
        async with session_scope(self._factory) as s:
            await s.merge(meta)
            await s.execute(
                delete(GroupMember).where(
                    GroupMember.app_id == meta.app_id,  # type: ignore[arg-type]
                    GroupMember.project_id == meta.project_id,  # type: ignore[arg-type]
                    GroupMember.group_id == meta.group_id,  # type: ignore[arg-type]
                )
            )
            s.add_all(list(members))
            for binding in bindings:
                await s.execute(_insert_binding(binding))
            await s.commit()

    async def remove(self, app_id: str, project_id: str, group_id: str) -> None:
        """Drop the group's meta and intervals (its ``group.md`` is gone)."""
        async with session_scope(self._factory) as s:
            await s.execute(
                delete(GroupMember).where(
                    GroupMember.app_id == app_id,  # type: ignore[arg-type]
                    GroupMember.project_id == project_id,  # type: ignore[arg-type]
                    GroupMember.group_id == group_id,  # type: ignore[arg-type]
                )
            )
            await s.execute(
                delete(GroupMeta).where(
                    GroupMeta.app_id == app_id,  # type: ignore[arg-type]
                    GroupMeta.project_id == project_id,  # type: ignore[arg-type]
                    GroupMeta.group_id == group_id,  # type: ignore[arg-type]
                )
            )
            await s.commit()

    async def binding(
        self, app_id: str, project_id: str, session_id: str
    ) -> SessionBinding | None:
        """The session's binding, or ``None``."""
        async with session_scope(self._factory) as s:
            return await s.get(SessionBinding, (app_id, project_id, session_id))

    async def bind(self, binding: SessionBinding) -> str:
        """Insert the binding unless the session has one; the bound group."""
        async with session_scope(self._factory) as s:
            await s.execute(_insert_binding(binding))
            await s.commit()
            row = await s.get(
                SessionBinding,
                (binding.app_id, binding.project_id, binding.session_id),
                populate_existing=True,
            )
            assert row is not None  # just inserted, or already there
            return row.group_id


def _insert_binding(binding: SessionBinding):
    values = {
        "app_id": binding.app_id,
        "project_id": binding.project_id,
        "session_id": binding.session_id,
        "group_id": binding.group_id,
        "bound_at": binding.bound_at,
        "created_at": binding.created_at,
        "updated_at": binding.updated_at,
    }
    return insert(SessionBinding).values(**values).on_conflict_do_nothing()

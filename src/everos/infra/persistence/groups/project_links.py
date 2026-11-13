"""``ProjectLinkStore`` by the partition rule: a project's groups are its space's.

Until a project-link feature maintains the many-to-many relation between
projects and groups (docs/group-memory.md, section 4.4), a project's groups
are the groups stored in its ``(app_id, project_id)`` space: every group the
group store lists there (the roster mirror), in group-id order, each answered
by its group scope in that same space. Every answer therefore stays inside the
project's own space, the one space a read scope covers. One listing per call;
membership is not looked at here.
"""

from __future__ import annotations

__all__ = ["PartitionProjectLinkStore"]

from everos.memory.ports import GroupStore, Scope


class PartitionProjectLinkStore:
    """:class:`~everos.memory.ports.ProjectLinkStore` by the partition rule.

    Args:
        groups: The group store; the groups it lists in a space are that
            project's groups.
    """

    def __init__(self, groups: GroupStore) -> None:
        self._groups = groups

    async def groups_of(self, app_id: str, project_id: str) -> list[Scope]:
        return [
            Scope(
                app_id=app_id,
                project_id=project_id,
                owner_id=group_id,
                owner_type="group",
            )
            for group_id in await self._groups.group_ids(app_id, project_id)
        ]
